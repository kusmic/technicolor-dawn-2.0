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
 
 #include "../cooling_sfr/feedback_treewalk.h"
 #include "../cooling_sfr/cooling.h"
 #include "../sph/kernel.h"
 #include "../data/allvars.h"
 #include "../data/dtypes.h"
 #include "../data/mymalloc.h"
 #include "../logs/logs.h"
 #include "../logs/timer.h"
 #include "../system/system.h"
 #include "../time_integration/timestep.h"
 
 // Define NEAREST macros for periodic wrapping (or no-op if not periodic)
 #define NEAREST(x, box) (((x) > 0.5 * (box)) ? ((x) - (box)) : (((x) < -0.5 * (box)) ? ((x) + (box)) : (x)))
 #define NEAREST_X(x) NEAREST(x, All.BoxSize)
 #define NEAREST_Y(x) NEAREST(x, All.BoxSize)
 #define NEAREST_Z(x) NEAREST(x, All.BoxSize)
 
 // Feedback type bitmask flags
 #define FEEDBACK_SNII  1
 #define FEEDBACK_AGB   2
 #define FEEDBACK_SNIa  4
 
 #define SNII_ENERGY (1.0e51 / All.UnitEnergy_in_cgs)  // in internal units
 
// Make a nice function to output printf statements, but only if 
// FeedbackDebug = 1 in the param.txt
//static int FeedbackDebug;
#define FEEDBACK_PRINT(...) \
    do { if (All.FeedbackDebug) printf(__VA_ARGS__); } while (0)

 // Physical constants and conversion factors
 const double HUBBLE_TIME = 13.8e9;              // Hubble time in years (approx)
 
 // Feedback energy/mass return constants
 const double SNII_ENERGY_PER_MASS = 1.0e50;     // erg / Msun  Should return to ~1.0e51 erg, but in tiny test volumes, 
                                                 // where the mass of gas particles is especially large, raise it higher so the effect can be seen
 const double SNKickFraction = 0.3;  // 30% kinetic, 70% thermal
 const double SNIa_ENERGY_PER_EVENT = 1.0e51;    // erg per event
 const double AGB_ENERGY_PER_MASS = 1.0e47;      // erg / Msun
 
 // Feedback timescales - physical time!
 const double SNII_DELAY_TIME_PHYSICAL = 1.0e7;     // years
 const double SNIa_DELAY_TIME_PHYSICAL = 1.0e9;     // years 
 const double AGB_END_TIME_PHYSICAL = 1.0e10;       // years
 
 // Mass return fractions
 const double MASS_RETURN_SNII = 0.10;           // fraction of m_star
 const double MASS_RETURN_AGB = 0.30;            // fraction of m_star
 
 // SNIa parameters
 const double SNIa_RATE_PER_MASS = 2.0e-3;       // events per Msun over cosmic time
 const double SNIa_DTD_MIN_TIME = 4.0e7;         // years - minimum delay time
 const double SNIa_DTD_POWER = -1.1;             // power-law slope of delay-time distribution
 
 // Feedback approach parameters
 //const double SNII_FEEDBACK_RADIUS = 0.3;        // kpc - local deposition: ~0.3 kpc for SNII, which is more localized
 //const double SNIa_FEEDBACK_RADIUS = 0.8;        // kpc - wider distribution ~0.8 kpc for SNIa to account for the more diffuse nature of these events
 //const double AGB_FEEDBACK_RADIUS = 0.5;         // kpc - intermediate distribution ~0.5 kpc for AGB winds, which are more diffuse than SNII but more concentrated than SNIa
 // Added a check to make sure the gas particle is not too close, otherwise the
 // feedback is too strong, and the timestep goes to zero.
 const double MIN_FEEDBACK_SEPARATION = 1e-2;  // kpc; adjust this as needed

 // In case we have ergs and needs to get it to internal units
 double erg_to_code;
 // Convert erg/g → Gadget code units
 double erg_per_mass_to_code;

 // Conversion from fixed-point integer positions to physical units (kpc)
 // Assuming IntPos are stored as 32-bit integers: there are 2^32 discrete positions.
 //const double NUM_INT_STEPS = 4294967296.0;        // 2^32
 //const double conversionFactor = All.BoxSize / NUM_INT_STEPS;

 inline double intpos_to_kpc(uint32_t ipos) {
    return (double) ipos * All.BoxSize / 4294967296.0;
 }

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
 
 // Dimensionless cubic spline kernel (Monaghan 1992) for 3D
inline double kernel_weight_cubic_dimless(double u) {
    if (u < 0.0) return 0.0;  // safety
    if (u < 0.5)
        return 1.0 - 6.0*u*u + 6.0*u*u*u;
    else if (u < 1.0)
        return 2.0 * pow(1.0 - u, 3);
    else
        return 0.0;
}

 // Local cubic spline kernel approximation for Type Ia SNe and AGB wind distribution
 inline double kernel_weight_cubic(double r, double h) {
    const double h_min_feedback = 0.3;  // Prevent explosive w at early times
    if (h < h_min_feedback)
        h = h_min_feedback;

    double u = r / h;
    double w_dimless = kernel_weight_cubic_dimless(u);
    double w = (8.0 / (M_PI * h * h * h)) * w_dimless;

    // Optional cap for stability
    if (!isfinite(w) || w > 10.0 || w < 0.0) {
        FEEDBACK_PRINT("[Feedback WARNING] Kernel weight w=%.3e clipped for r=%.3e h=%.3e u=%.3e\n", w, r, h, u);
        w = 0.0;
    }
    return w;
}
 
 // Step function for Type II SNe (simpler, more direct energy deposition)
 inline double kernel_weight_tophat(double r, double h) {
     if (r < h)
         return 1.0 / (4.0/3.0 * M_PI * h * h * h); // Normalized to integrate to 1
     else
         return 0.0;
 }
 
/** Having a constant feedback radius can lead to too few or too many neighbors
   Let's try adapting the radius (h) based on the number of neighbors found

 * Adaptive feedback radius based on local gas density
 * Dynamically determines a suitable radius h for a given star
 * so that it finds ~TARGET_NEIGHBORS gas neighbors.

 WHAT IF we could modify adaptive_feedback_radius() to return an array of valid gas indices to avoid the second scan?
 */
 double adaptive_feedback_radius(MyDouble starPos[3], int feedback_type, simparticles *Sp, int *neighbors_ptr) {

    int TARGET_NEIGHBORS;
    if (feedback_type == FEEDBACK_SNII)
        TARGET_NEIGHBORS = 10;   // probably increase later with bigger run!
    else if (feedback_type == FEEDBACK_SNIa)
        TARGET_NEIGHBORS = 64;
    else if (feedback_type == FEEDBACK_AGB)
        TARGET_NEIGHBORS = 32;
    else
        TARGET_NEIGHBORS = 48;

    const double H_MIN = 0.05;  // kpc
    const double H_MAX = 3.0;   // kpc
    const int MAX_ATTEMPTS = 10;

    double h = 0.7;  // initial guess
    int attempt = 0;
    int neighbors_found = 0;

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

    do {
        neighbors_found = 0;
        for (int j = 0; j < gas_count; j++) {
            double dx = NEAREST_X(gas_x[j] - starPos[0]);
            double dy = NEAREST_Y(gas_y[j] - starPos[1]);
            double dz = NEAREST_Z(gas_z[j] - starPos[2]);
            double r2 = dx*dx + dy*dy + dz*dz;
            if (sqrt(r2) < h) neighbors_found++;
        }

        FEEDBACK_PRINT("[Feedback Adaptive h] Neighbors found=%d!\n", neighbors_found);
        *neighbors_ptr = neighbors_found;

        if (neighbors_found < TARGET_NEIGHBORS)
            h *= 1.25;
        else if (neighbors_found > TARGET_NEIGHBORS * 1.5)
            h *= 0.8;

        h = fmax(fmin(h, H_MAX), H_MIN);
        attempt++;
    } while ((neighbors_found < TARGET_NEIGHBORS || neighbors_found > TARGET_NEIGHBORS * 2)
             && attempt < MAX_ATTEMPTS);

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
 
 // Structure to pass around feedback data
 struct FeedbackInput {
     MyDouble Pos[3];
     MyFloat Energy;
     MyFloat MassReturn;
     MyFloat Yield[4];
     int FeedbackType;
     int NeighborCount;
     double h;  // Smoothing length/radius for this feedback type
     int SourceIndex; // Index of the source star particle for diagnostics
 };
 
 struct FeedbackResult {};
 
 struct FeedbackWalk {
     double current_time;
     int feedback_type;
     const char* ev_label;
 };
 
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
    double max_u = 1e4; // Maximum allowed utherm in internal units

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
 static int feedback_isactive(int i, FeedbackWalk *fw, simparticles *Sp) {
     if (Sp->P[i].getType() != 4) // needs to be a star
         return 0;

     //printf("Star %d | current_time (a)=%.6f | StellarAge=%.6f\n", i, fw->current_time, Sp->P[i].StellarAge);

     // Convert from scale factor to physical time
     double age_physical = scale_factor_to_physical_time(fw->current_time - Sp->P[i].StellarAge);
 
     // Check if the star is too young
     //FEEDBACK_PRINT("[Feedback] Considering star %d | age=%.2e yr | type=%d | flag=%d ------------------\n",
        i, age_physical, fw->feedback_type, Sp->P[i].FeedbackFlag);

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
void feedback_ngb(FeedbackInput *in, FeedbackResult *out, int j, FeedbackWalk *fw, simparticles *Sp) {
    if (Sp->P[j].getType() != 0) return; // Only apply to gas particles
    //if (r > in->h) return;  // Skip if it is outside the feedback radius

    //erg_to_code = 1.0 / (All.UnitEnergy_in_cgs);
    //erg_per_mass_to_code = 1.0 / (All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s);

    double gas_mass = Sp->P[j].getMass();
    if (gas_mass <= 0 || isnan(gas_mass) || !isfinite(gas_mass)) return;

    // Radial vector from feedback source to this gas particle
    double gasPos[3];
    gasPos[0] = intpos_to_kpc(Sp->P[j].IntPos[0]);
    gasPos[1] = intpos_to_kpc(Sp->P[j].IntPos[1]);
    gasPos[2] = intpos_to_kpc(Sp->P[j].IntPos[2]);

    double dx[3] = {
        NEAREST_X(gasPos[0] - in->Pos[0]),
        NEAREST_Y(gasPos[1] - in->Pos[1]),
        NEAREST_Z(gasPos[2] - in->Pos[2])
    };
    double r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
    double r = sqrt(r2) + 1e-10;

    //FEEDBACK_PRINT("[Feedback] Distance to source r=%.5f (dx=[%.5f, %.5f, %.5f])\n", r, dx[0], dx[1], dx[2]);

    // Distribute energy equally among neighbors (for SNII)
    double E_total = in->Energy;
    double E_kin = E_total * SNKickFraction;
    double E_therm = E_total * (1.0 - SNKickFraction);

    double invertNeighborCount = 1.0 / in->NeighborCount; // multiplying by 1/N is faster than dividing by N
    double E_kin_j = E_kin * invertNeighborCount;
    double E_therm_j = E_therm * invertNeighborCount;

    //double delta_u = E_therm_j * erg_to_code / gas_mass;
    double delta_u = E_therm_j * erg_per_mass_to_code / gas_mass;


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
    double v_kick = sqrt(2.0 * E_kin_j * erg_per_mass_to_code / gas_mass);
    if (!isfinite(v_kick) || v_kick < 0 || v_kick > 1e5) {
        FEEDBACK_PRINT("[Feedback WARNING] Non-finite or huge v_kick = %.3e for gas %d\n", v_kick, j);
        return;
    }
    for (int k = 0; k < 3; k++)
        Sp->P[j].Vel[k] += v_kick * dx[k] / r;

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

/**
 * Loop over star particles and apply feedback to their neighbors
 */
 void run_feedback(double current_time, int feedback_type, simparticles *Sp) {
    FeedbackInput in;
    FeedbackResult out;
    FeedbackWalk fw;
    fw.current_time = current_time;
    fw.feedback_type = feedback_type;

    // Cache gas positions once per timestep (gas_x, gas_y, gas_z)
    int n_gas = 0;
    cache_gas_positions(Sp, &n_gas);

    // Convert erg to code units
    //erg_to_code = 1.0 / (All.UnitEnergy_in_cgs);
    erg_per_mass_to_code = 1.0 / (All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s);

    static int printed_erg_code = 0;
    if (!printed_erg_code && ThisTask == 0) {
        FEEDBACK_PRINT("[Feedback Init] erg_per_mass_to_code = %.3e (UnitEnergy = %.3e cgs)\n", erg_per_mass_to_code, All.UnitEnergy_in_cgs);
        printed_erg_code = 1;
    }

    for (int i = 0; i < Sp->NumPart; i++) {
        if (Sp->P[i].getType() != 4) continue;  // Star particles only
        if (!feedback_isactive(i, &fw, Sp)) continue;

        // Set up input for this star
        in.Pos[0] = intpos_to_kpc(Sp->P[i].IntPos[0]);
        in.Pos[1] = intpos_to_kpc(Sp->P[i].IntPos[1]);
        in.Pos[2] = intpos_to_kpc(Sp->P[i].IntPos[2]);
        in.FeedbackType = feedback_type;
        in.Energy = SNII_ENERGY_PER_MASS * Sp->P[i].getMass();
        in.MassReturn = 0.1 * Sp->P[i].getMass();

        // Track diagnostics per type
        if (feedback_type == FEEDBACK_SNII) {
            ThisStepEnergy_SNII += in.Energy;
            TotalEnergyInjected_SNII += in.Energy;
        } else if (feedback_type == FEEDBACK_SNIa) {
            ThisStepEnergy_SNIa += in.Energy;
            TotalEnergyInjected_SNIa += in.Energy;
        } else if (feedback_type == FEEDBACK_AGB) {
            ThisStepEnergy_AGB += in.Energy;
            TotalEnergyInjected_AGB += in.Energy;
        }

        ThisStepMassReturned += in.MassReturn;
        TotalMassReturned += in.MassReturn;

        // Adaptively find a good feedback radius
        // Can we cache neighbor indices instead of recomputing distances twice (once in adaptive_feedback_radius, once in loop)?
        int n_neighbors = 0;
        double h_feedback = adaptive_feedback_radius(in.Pos, feedback_type, Sp, &n_neighbors);
        if (h_feedback < 0.0) continue;  // skip if no neighbors found

        in.h = h_feedback;
        in.NeighborCount = n_neighbors;
        double h2 = in.h * in.h;  // will use for quicker distance checks

        FEEDBACK_PRINT("[Feedback] Star ID=%d will deposit feedback to %d gas neighbors within %.1f kpc\n",
               i, in.NeighborCount, in.h);

        // Apply feedback only to close neighbors
        for (int j = 0; j < Sp->NumPart; j++) {
            if (Sp->P[j].getType() != 0) continue;

            double dx = NEAREST_X(gas_x[j] - in.Pos[0]);
            double dy = NEAREST_Y(gas_y[j] - in.Pos[1]);
            double dz = NEAREST_Z(gas_z[j] - in.Pos[2]);
            double r2 = dx*dx + dy*dy + dz*dz;

            // If within feedback radius, apply feedback!
            if (r2 < h2)
                feedback_ngb(&in, &out, j, &fw, Sp);
        }
    }
}


 /**
  * Main feedback function called each timestep
  */
 void apply_stellar_feedback(double current_time, simparticles* Sp) {
 
 /* -----TEMPORARY DIAGNOSTIC: Print TIME RESOLUTION-----
     static double last_time = 0;
     double timestep = current_time - last_time;
 
     // Convert from scale factor difference to physical time if needed
     double physical_timestep = 0;
     if (current_time > 0 && last_time > 0) {
         physical_timestep = scale_factor_to_physical_time(current_time) - 
                            scale_factor_to_physical_time(last_time);
 
         if (ThisTask == 0) {
             //printf("[Feedback Diagnostic] Current scale factor: %.6e\n", current_time);
             //printf("[Feedback Diagnostic] Timestep in scale factor: %.6e\n", timestep);
             //printf("[Feedback Diagnostic] Time resolution in physical years: %.2e\n", physical_timestep);
         }
     }
 
     last_time = current_time;
 // -----TEMPORARY DIAGNOSTIC: Print TIME RESOLUTION-----
 */
 
 
     // Reset diagnostic counters
     ThisStepEnergy_SNII = 0;
     ThisStepEnergy_SNIa = 0;
     ThisStepEnergy_AGB = 0;
     ThisStepMassReturned = 0;
     std::memset(ThisStepMetalsInjected, 0, sizeof(ThisStepMetalsInjected));
 
     // Apply each feedback type
     run_feedback(current_time, FEEDBACK_SNII, Sp);
     //apply_feedback_treewalk(current_time, FEEDBACK_SNII, Sp);
     //apply_feedback_treewalk(current_time, FEEDBACK_AGB, Sp);
     //apply_feedback_treewalk(current_time, FEEDBACK_SNIa, Sp);

     for (int k = 0; k < 4; k++)
         TotalMetalsInjected[k] += ThisStepMetalsInjected[k];
 
     // Print summary (on master process only)
     if (ThisTask == 0 && (ThisStepEnergy_SNII > 0 ||
        ThisStepEnergy_SNIa > 0 ||
        ThisStepEnergy_AGB > 0)) {
         FEEDBACK_PRINT("[Feedback Timestep Summary] E_SNII=%.3e erg, E_SNIa=%.3e erg, E_AGB=%.3e erg\n",
                ThisStepEnergy_SNII, ThisStepEnergy_SNIa, ThisStepEnergy_AGB);
         FEEDBACK_PRINT("[Feedback Timestep Summary] Mass Returned=%.3e Msun\n", ThisStepMassReturned);
         FEEDBACK_PRINT("[Feedback Timestep Summary] Metals (Z=%.3e, C=%.3e, O=%.3e, Fe=%.3e) Msun\n",
                ThisStepMetalsInjected[0], ThisStepMetalsInjected[1], ThisStepMetalsInjected[2], ThisStepMetalsInjected[3]);
         FEEDBACK_PRINT("[Feedback Summary] Total SNII energy distributed: %.3e\n", TotalEnergyInjected_SNII);
     }
 }

 #endif // FEEDBACK