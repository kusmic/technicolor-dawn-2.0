/*
 ==========================================================================================
 FEEDBACK_TREEWALK.CC — Stellar Feedback Injection in Gadget-4
 ==========================================================================================
 
 ❖ Purpose:
 This file implements a feedback algorithm for injecting stellar feedback (energy, mass,
 and metals) from stars into the surrounding interstellar medium (ISM) in cosmological
 simulations using Gadget-4.
 
 ❖ What is "feedback"?
 In astrophysical simulations, feedback refers to the physical processes by which stars 
 influence their environment after they form — primarily through:
    • Supernova explosions (Type II and Type Ia)
    • Winds from dying stars (AGB stars)
 These processes return energy and enriched material to the surrounding gas, regulating
 galaxy formation and evolution.
 
 ❖ What this file does:
    • Loops over all active star particles (Type == 4)
    • Checks if a star is ready to release feedback based on its age and other criteria
    • Finds neighboring gas particles
    • Injects feedback energy, mass, and metal yields using different approaches:
       - Type II: Local direct deposition to neighboring gas particles
       - Type Ia: Extended distribution using a kernel-weighted approach
       - AGB: Similar to Type Ia but with different yields and timing
    • Tracks energy and mass diagnostics for logging and analysis
 
     Type II Supernovae (SNII)
 
         Localized Energy Deposition: Uses a top-hat kernel with a smaller radius (0.3 kpc) for more concentrated feedback
         Stronger Kinetic Feedback: Applies stronger radial velocity kicks to simulate the blast wave
         Metallicity-Dependent Yields: Enhanced oxygen production at low metallicity
         Quick Timescale: Activates after a short delay (~10 million years)
 
     Type Ia Supernovae (SNIa)
 
         Extended Distribution: Uses a cubic spline kernel with a larger radius (0.8 kpc) for a smoother, more extended energy distribution
         Primarily Thermal Feedback: Emphasizes thermal energy over kinetic energy
         Iron-Rich Yields: Produces much more iron than other elements
         Stochastic Implementation: Uses a realistic delay-time distribution with t^-1.1 power law and Poisson sampling
         Longer Timescales: Can occur billions of years after star formation
 
 ❖ Key components:
    - Constants defining feedback strength and timing
    - Kernel weight function for distributing feedback to nearby gas (primarily for SNIa)
    - Metal yield functions for each type of feedback
    - Diagnostic accumulators to track what's been injected
 
 ❖ Usage:
 This file is compiled and executed as part of the Gadget-4 simulation if the FEEDBACK
 flag is enabled in the build configuration.
 
 NOTE! With bitwise operations, we track which feedback types have been applied to each star:
    - 0: No feedback applied
    - 1: SNII feedback applied
    - 2: AGB feedback applied
    - 4: SNIa feedback applied
    - 3: (SNII + AGB) feedback applied
    - 5: (SNII + SNIa) feedback applied
    - 6: (AGB + SNIa) feedback applied
    - 7: (SNII + AGB + SNIa) feedback applied
 This allows us to ensure that each star only contributes feedback once for each type.
 
 ==========================================================================================
 */
 
 #include "gadgetconfig.h"
 
 #ifdef FEEDBACK
 
 #include <assert.h>
 #include <math.h>
 #include <mpi.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <random>
 
 #include "../gravtree/gravtree.h"
 #include "../main/simulation.h"
 #include "../ngbtree/ngbtree.h"
 #include "../data/simparticles.h"
 #include "../data/symtensors.h"
 #include "../domain/domain.h"
 #include "../mpi_utils/mpi_utils.h"
 #include "../tree/tree.h"
 #include "../cooling_sfr/feedback.h"
 #include "../cooling_sfr/cooling.h"
 #include "../sph/kernel.h"
 #include "../data/allvars.h"
 #include "../data/dtypes.h"
 #include "../data/mymalloc.h"
 #include "../logs/logs.h"
 #include "../logs/timer.h"
 #include "../system/system.h"
 #include "../time_integration/timestep.h"

 gravtree<simparticles> GravTree;
 simparticles SimParticles;

 /**
  * Convert cosmological scale factor to physical time in years
  * Note: This is a simple approximation - you might want to use a more accurate
  * conversion based on your cosmological parameters
  */
 double scale_factor_to_physical_time(double a) {
     return HUBBLE_TIME * pow(a, 3.0/2.0); // Simple matter-dominated universe approximation
 }
 
 /**
  * Convert physical time in years to cosmological scale factor
  */
 double physical_time_to_scale_factor(double time_physical) {
     return pow(time_physical / HUBBLE_TIME, 2.0/3.0); // Simple matter-dominated universe approximation
 }

  // In case we have ergs and needs to get it to internal units
  double erg_to_code;
  // Convert erg/g → Gadget code units
  double erg_per_mass_to_code;
 
  const double WIND_VELOCITY = 500.0;             // km/s
  
  // Per-timestep diagnostics
  double ThisStepEnergy_SNII = 0;
  double ThisStepEnergy_SNIa = 0;
  double ThisStepEnergy_AGB = 0;
  double ThisStepMassReturned = 0;
  double ThisStepMetalsInjected[4] = {0};
  
  // Cumulative totals
  double TotalEnergyInjected_SNII = 0;
  double TotalEnergyInjected_SNIa = 0;
  double TotalEnergyInjected_AGB = 0;
  double TotalMassReturned = 0;
  double TotalMetalsInjected[4] = {0};
  
  struct Yields {
      double Z, C, O, Fe;
  };

 // This random number generator will be used for stochastic SNIa events
 // Using a different name to avoid conflict with Gadget's existing random_generator
 std::mt19937 feedback_random_gen(std::random_device{}());

/*  FUNCTIONS ------------------------------------------------------------------------------------- */

/** 
   Having a constant feedback radius can lead to too few or too many neighbors
   Let's try adapting the radius (h) based on the number of neighbors found

 * Adaptive feedback radius based on local gas density
 * Dynamically determines a suitable radius h for a given star
 * so that it finds ~TARGET_NEIGHBORS gas neighbors.

   WHAT IF we could modify adaptive_feedback_radius() to return an array of valid gas indices to avoid the second scan?
 * --------------------------------------------------------------------------------------------------
 * Computes an adaptive feedback radius `h` for a star at `starPos`, returning the gas
 * particles within that radius in `neighbor_list[]`.
 *
 * Returns the final smoothing length `h`, or -1.0 if no neighbors found.
 */
 double adaptive_feedback_radius(MyDouble starPos[3], int feedback_type, simparticles *Sp,
    int *neighbors_ptr, int *neighbor_list, int max_neighbors)
{
// Target neighbor counts per feedback type
int TARGET_NEIGHBORS;
if (feedback_type == FEEDBACK_SNII)
TARGET_NEIGHBORS = 10;
else if (feedback_type == FEEDBACK_SNIa)
TARGET_NEIGHBORS = 64;
else if (feedback_type == FEEDBACK_AGB)
TARGET_NEIGHBORS = 32;
else
TARGET_NEIGHBORS = 48;

const double H_MIN = 0.05;  // kpc
const double H_MAX = 3.0;   // kpc
const int MAX_ATTEMPTS = 10;

double h = 0.7;  // Initial guess
int attempt = 0;

// Cache gas positions on first call
static double *gas_x = NULL, *gas_y = NULL, *gas_z = NULL;
static int gas_count = 0;
if (!gas_x) {
gas_x = (double *) malloc(Sp->NumPart * sizeof(double));
gas_y = (double *) malloc(Sp->NumPart * sizeof(double));
gas_z = (double *) malloc(Sp->NumPart * sizeof(double));
}

gas_count = 0;
for (int j = 0; j < Sp->NumPart; j++) {
if (Sp->P[j].getType() != 0) continue;
gas_x[gas_count] = intpos_to_kpc(Sp->P[j].IntPos[0]);
gas_y[gas_count] = intpos_to_kpc(Sp->P[j].IntPos[1]);
gas_z[gas_count] = intpos_to_kpc(Sp->P[j].IntPos[2]);
gas_count++;
}

int neighbors_found = 0;

do {
neighbors_found = 0;

for (int j = 0; j < gas_count; j++) {
double dx = NEAREST_X(gas_x[j] - starPos[0]);
double dy = NEAREST_Y(gas_y[j] - starPos[1]);
double dz = NEAREST_Z(gas_z[j] - starPos[2]);
double r2 = dx*dx + dy*dy + dz*dz;

if (sqrt(r2) < h) {
if (neighbors_found < max_neighbors)
neighbor_list[neighbors_found] = j;
neighbors_found++;
}
}

*neighbors_ptr = neighbors_found;

if (neighbors_found < TARGET_NEIGHBORS)
h *= 1.25;
else if (neighbors_found > TARGET_NEIGHBORS * 1.5)
h *= 0.8;

h = fmax(fmin(h, H_MAX), H_MIN);
attempt++;

} while ((neighbors_found < TARGET_NEIGHBORS || neighbors_found > TARGET_NEIGHBORS * 2) &&
attempt < MAX_ATTEMPTS);

if (neighbors_found == 0) {
if (ThisTask == 0) {
FEEDBACK_PRINT("[Feedback WARNING] No gas neighbors found within H_MAX=%.2f! Skipping feedback.\n", H_MAX);
}
return -1.0;
}

if (ThisTask == 0) {
FEEDBACK_PRINT("[Feedback Adaptive h] Final h=%.3f kpc after %d attempts for feedback_type=%d (%d neighbors)\n",
h, attempt, feedback_type, neighbors_found);
}

return h;
}



 /**
  * Get SNII yields - enhanced metallicity-dependent model
  */
 Yields get_SNII_yields(double m, double metallicity) {
     // Scale yields based on progenitor metallicity
     double z_factor = pow(metallicity / 0.02, 0.3);  // Metallicity scaling
 
     // Base values
     double yield_z = 0.01 * m;
     double yield_c = 0.005 * m;
     double yield_o = 0.003 * m; 
     double yield_fe = 0.002 * m;
 
     // SNII produce more oxygen and less iron at low metallicity
     return {
         yield_z * z_factor,
         yield_c * pow(z_factor, 0.8),
         yield_o * pow(z_factor, 0.5),  // Stronger O production at low Z
         yield_fe * pow(z_factor, 1.2)  // Less Fe production at low Z
     };
 }
 
 /**
  * Get AGB yields
  */
 Yields get_AGB_yields(double m, double metallicity) {
     // AGB yields are metallicity-dependent
     double z_factor = pow(metallicity / 0.02, 0.5);
 
     return {
         0.005 * m * z_factor,
         0.002 * m * pow(z_factor, 1.1),  // AGB stars are important C producers
         0.001 * m * pow(z_factor, 0.8),
         0.001 * m * z_factor
     };
 }
 
 /**
  * Get SNIa yields - primarily iron-peak elements
  */
 Yields get_SNIa_yields(double n_events) {
     // SNIa produce mainly iron and less of lighter elements
     return {
         0.002 * n_events,  // total metals
         0.0005 * n_events, // very little carbon
         0.001 * n_events,  // some oxygen
         0.005 * n_events   // lots of iron - main Fe source in universe
     };
 }
 

 

// File-scope static caches
static double *gas_x = NULL, *gas_y = NULL, *gas_z = NULL;
static int *gas_index = NULL;
static int gas_count = 0;

/**
 * Cache positions of all gas particles (Type 0) in physical units (kpc), 
 * along with their corresponding indices in Sp->P[].
 *
 * @param Sp            Pointer to simulation particles
 * @param return_count  Optional pointer to store how many gas particles were cached
 */
void cache_gas_positions(simparticles *Sp, int *return_count) {
    if (!gas_x) {
        gas_x = (double *) malloc(Sp->NumPart * sizeof(double));
        gas_y = (double *) malloc(Sp->NumPart * sizeof(double));
        gas_z = (double *) malloc(Sp->NumPart * sizeof(double));
        gas_index = (int *) malloc(Sp->NumPart * sizeof(int));
    }

    gas_count = 0;
    for (int j = 0; j < Sp->NumPart; j++) {
        if (Sp->P[j].getType() != 0)
            continue;

        gas_x[gas_count] = intpos_to_kpc(Sp->P[j].IntPos[0]);
        gas_y[gas_count] = intpos_to_kpc(Sp->P[j].IntPos[1]);
        gas_z[gas_count] = intpos_to_kpc(Sp->P[j].IntPos[2]);
        gas_index[gas_count] = j;  // Store index into Sp->P
        gas_count++;
    }

    if (return_count)
        *return_count = gas_count;
}

 
/**
 * Clamp total internal energy (utherm) to prevent extremely small timesteps
 * from too high thermal energy.
 * Let's think about what max utherms would make sense:
 * 10^6 K -- Warm-hot IGM / CGM -- utherm make should be: ~2e3
 * 10^7 k -- Typical SN/AGN-heated ISM gas -- utherm make should be: ~2e4
 * 10^8 K -- Hot bubbles, extreme feedback -- utherm make should be: ~2e5
 */
 double clamp_feedback_energy(double u_before, double delta_u, int gas_index, MyIDType gas_id) {
    double u_after = u_before + delta_u;
    double max_u = 5e3; // Maximum allowed utherm in internal units

    if (!isfinite(delta_u) || delta_u < 0.0 || delta_u > 1e10) {
        FEEDBACK_PRINT("[Feedback WARNING] Non-finite or excessive delta_u=%.3e for gas ID=%llu\n", delta_u, (unsigned long long) gas_id);
        return u_before;
    }

    if (u_after > max_u && u_after != max_u) {
        FEEDBACK_PRINT("[Feedback WARNING] Clamping u from %.3e to %.3e for gas ID=%llu\n", u_after, max_u, (unsigned long long) gas_id);
        return max_u;
    }

    return u_after;
}


 /**
  * Determine if a star particle is eligible for feedback
  */
 int feedback_isactive(int i, FeedbackWalk *fw, simparticles *Sp) {
     if (Sp->P[i].getType() != 4) // needs to be a star
         return 0;

     //printf("Star %d | current_time (a)=%.6f | StellarAge=%.6f\n", i, fw->current_time, Sp->P[i].StellarAge);

     // Convert from scale factor to physical time
     double age_physical = scale_factor_to_physical_time(fw->current_time - Sp->P[i].StellarAge);
 
     // Check if the star is too young
     //FEEDBACK_PRINT("[Feedback] Considering star %d | age=%.2e yr | type=%d | flag=%d ------------------\n",i, age_physical, fw->feedback_type, Sp->P[i].FeedbackFlag);

     // Check if this feedback type has already been applied
     if ((Sp->P[i].FeedbackFlag & fw->feedback_type) != 0) {
         return 0; // Already processed
     }

     // Different criteria for different feedback types
     if (fw->feedback_type == FEEDBACK_SNII) {
         //FEEDBACK_PRINT("[Feedback Debug] Star %d is active for SNII feedback type %d\n", i, fw->feedback_type);
         // Type II SNe happen promptly after star formation
         return (age_physical > SNII_DELAY_TIME_PHYSICAL) ? 1 : 0;
     } 
     else if (fw->feedback_type == FEEDBACK_AGB) {
         //printf("[Feedback Debug] Star %d is active for AGB feedback type %d\n", i, fw->feedback_type);
         // AGB winds are active after some delay but before stars are too old
         return (age_physical > SNII_DELAY_TIME_PHYSICAL && age_physical < AGB_END_TIME_PHYSICAL) ? 1 : 0;
     }
     else if (fw->feedback_type == FEEDBACK_SNIa) {
         // Type Ia SNe follow a delay-time distribution
         if (age_physical < SNIa_DTD_MIN_TIME) return 0;
 
         // Calculate expected number of SNIa based on DTD
         double m_star = Sp->P[i].getMass();
         double time_since_eligible = age_physical - SNIa_DTD_MIN_TIME;
 
         // Probability for SNIa events:
         // DTD ~ t^power 
         double rate = SNIa_RATE_PER_MASS * pow(time_since_eligible / 1.0e9, SNIa_DTD_POWER);
 
         // Estimate the expected number of events
         double delta_t = 1.0e8; // Approximate SN 1a timestep in years
         double mean_events = m_star * rate * delta_t;
 
         // Draw from Poisson distribution
         std::poisson_distribution<int> poisson(mean_events);
         int n_events = poisson(feedback_random_gen);
 
         // Store the number of events for later use
         Sp->P[i].SNIaEvents = n_events;
 
         FEEDBACK_PRINT("[Feedback Debug] Star %d is active for SNIa feedback type %d\n", i, fw->feedback_type);
         return (n_events > 0) ? 1 : 0;
     }
 
     return 0;
 }
 
 /**
  * Apply feedback to a neighboring gas particle
  * This is where we implement the different approaches for Type II vs Type Ia
  */
 /**
 * Apply feedback to a neighboring gas particle
 */
void feedback_to_gas_neighbor(FeedbackInput *in, FeedbackResult *out, int j, FeedbackWalk *fw, simparticles *Sp) {
    if (Sp->P[j].getType() != 0) return; // Only apply to gas particles
    //if (r > in->h) return;  // Skip if it is outside the feedback radius

    //erg_to_code = 1.0 / (All.UnitEnergy_in_cgs);
    //erg_per_mass_to_code = 1.0 / (All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s);

    double gas_mass = Sp->P[j].getMass();
    if (gas_mass <= 0 || isnan(gas_mass) || !isfinite(gas_mass)) return;

    // Radial vector from feedback source to this gas particle
    double dx[3] = {
        NEAREST_X(gas_x[j] - in->Pos[0]),
        NEAREST_Y(gas_y[j] - in->Pos[1]),
        NEAREST_Z(gas_z[j] - in->Pos[2])
    };
    
    double r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
    //double r = sqrt(r2) + 1e-10;
    double r_inv = 1.0 / (sqrt(r2) + 1e-10);
    //FEEDBACK_PRINT("[Feedback] Distance to source r=%.5f (dx=[%.5f, %.5f, %.5f])\n", r, dx[0], dx[1], dx[2]);

    // Distribute energy equally among neighbors (for SNII)
    double E_total = in->Energy;
    double E_kin = E_total * SNKickFraction;
    double E_therm = E_total * (1.0 - SNKickFraction);

    double invertNeighborCount = 1.0 / in->NeighborCount; // multiplying by 1/N is faster than dividing by N
    double E_kin_j = E_kin * invertNeighborCount;
    double E_therm_j = E_therm * invertNeighborCount;


    //double delta_u = E_therm_j * erg_to_code / gas_mass;
    double gas_mass_cgs = gas_mass * All.UnitMass_in_g;
    double inv_mass_cgs = 1.0 / gas_mass_cgs;
    double delta_u = E_therm_j * erg_per_mass_to_code * inv_mass_cgs;

    //printf("[Feedback DEBUG] GasID=%d | E_therm_j=%.3e erg | gas_mass_cgs=%.3e g | erg_per_mass_to_code=%.3e | delta_u=%.3e (u_before=%.3e)\n",
    //    Sp->P[j].ID.get(),
    //    E_therm_j, gas_mass_cgs, erg_per_mass_to_code, delta_u, Sp->get_utherm_from_entropy(j));

    if (!isfinite(delta_u) || delta_u < 0) {
        FEEDBACK_PRINT("[Feedback WARNING] Non-finite delta_u = %.3e for gas %d\n", delta_u, j);
        return;
    }
    //FEEDBACK_PRINT("[Feedback] E_therm_j=%.3e erg, delta_u=%.3e (internal units)\n", E_therm_j, delta_u);

    // Update thermal energy
    // added clamp_feedback_energy() because occasionally a gas particle might go nuts
    double utherm_before = Sp->get_utherm_from_entropy(j);
    double utherm_after = clamp_feedback_energy(utherm_before, delta_u, j, Sp->P[j].ID.get());

    double rel_increase = delta_u / (utherm_before + 1e-10);
    if (rel_increase > 10.0) {
        FEEDBACK_PRINT("[Feedback WARNING] delta_u (%.3e) is too large (%.1fx u_before=%.3e) for gas ID=%d\n", 
                delta_u, rel_increase, utherm_before, Sp->P[j].ID.get());
        return;
    }

    Sp->set_entropy_from_utherm(utherm_after, j);

    FEEDBACK_PRINT("[Feedback] u_before=%.5e, u_after=%.5e, delta_u=%.3e (internal units)\n", utherm_before, utherm_after, delta_u);

    // Apply radial kinetic kick
    double v_kick = sqrt(2.0 * E_kin_j * erg_per_mass_to_code * inv_mass_cgs);
    if (!isfinite(v_kick) || v_kick < 0 || v_kick > 1e5) {
        FEEDBACK_PRINT("[Feedback WARNING] Non-finite or huge v_kick = %.3e for gas %d\n", v_kick, j);
        return;
    }
    for (int k = 0; k < 3; k++)
        Sp->P[j].Vel[k] += v_kick * dx[k] * r_inv;

    FEEDBACK_PRINT("[Feedback] Applied radial kick v_kick=%.3e km/s\n", v_kick);

    // Mass return
    double mass_return = in->MassReturn * invertNeighborCount;
    Sp->P[j].setMass(gas_mass + mass_return);

    FEEDBACK_PRINT("[Feedback] Mass return: %.3e -> New mass: %.3e\n", mass_return, Sp->P[j].getMass());

    // Metal enrichment
    for (int k = 0; k < 4; k++) {
        double metal_add = (in->Yield[k] * invertNeighborCount) / gas_mass;
        Sp->SphP[j].Metals[k] += metal_add;
        //FEEDBACK_PRINT("[Feedback] Metal[%d] += %.3e\n", k, metal_add);
    }

    // FINAL debug check to catch extremely small or large thermal states
    double final_u = Sp->get_utherm_from_entropy(j);
    if (!isfinite(final_u) || final_u < 1e-20 || final_u > 1e10) {
        FEEDBACK_PRINT("[Feedback WARNING] Bad final entropy on gas %d: u=%.3e\n", j, final_u);
    }
}

void run_feedback(double current_time, int feedback_type, simparticles *Sp) {
    int *Active = (int *) malloc(sizeof(int) * Sp->NumPart);
    int NumActive = 0;

    for (int i = 0; i < Sp->NumPart; i++) {
        if (Sp->P[i].getType() == 4 && feedback_isactive(i, NULL, Sp)) {
            Active[NumActive++] = i;
        }
    }

    if (NumActive > 0) {
        printf("[Feedback] Running feedback for %d stars.\n", NumActive);
        feedback_tree(Active, NumActive, Sp);
    }

    free(Active);
}

#endif // FEEDBACK