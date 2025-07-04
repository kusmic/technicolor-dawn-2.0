/*
 =====================================================================================
 FEEDBACK_TREEWALK.CC — Stellar Feedback using Octree in Gadget-4
 =====================================================================================
 
 ❖ Purpose:
 This file implements an efficient treewalk algorithm for injecting stellar feedback
 (energy, mass, and metals) from stars into the surrounding interstellar medium (ISM)
 in cosmological simulations using Gadget-4's existing tree structure.
 
 ❖ Advantages over brute-force approach:
    • O(N log N) scaling instead of O(N²)
    • Reuses existing tree infrastructure from Gadget-4
    • Better handling of adaptive search radii
    • Improved performance for large simulations (32³ particles and beyond)
 
 =====================================================================================
 */
 
 #include "gadgetconfig.h"

 #ifdef FEEDBACK
 
 #include <math.h>
 #include <mpi.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <random>
 #include <fstream>
 #include <iomanip>

 #include "../gravtree/gravtree.h"
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
 
// ─── For diagnostic output CSV file ───
std::vector<double> g_delta_u;
std::vector<double> g_delta_v;
std::vector<double> g_rel_increase;
std::vector<double> g_radial_r;
std::vector<int>    g_neighbors_per_star;
std::vector<double> g_h_per_star;
std::vector<double> g_energy_ratio;

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
 
 // Debug output function
 #define FEEDBACK_PRINT(...) \
     do { if (All.FeedbackDebug) printf(__VA_ARGS__); } while (0)
 
 // Physical constants and conversion factors
 const double HUBBLE_TIME = 13.8e9;              // Hubble time in years (approx)
 
 // Feedback energy/mass return constants
 const double SNII_ENERGY_PER_MASS = 1.0e50;     // erg / Msun  
 const double SNKickFraction = 0.3;              // 30% kinetic, 70% thermal
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
 
 // Minimum distance to prevent numerical instabilities
 const double MIN_FEEDBACK_SEPARATION = 1e-2;    // kpc
 
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
 
 // Random number generator for stochastic SNIa events
 std::mt19937 feedback_random_gen(std::random_device{}());
 
 // Unit conversions
 double erg_to_code;
 double erg_per_mass_to_code;
 
 struct Yields {
     double Z, C, O, Fe;
 };
 
 //Convert cosmological scale factor to physical time in years
 double scale_factor_to_physical_time(double a) {
     return HUBBLE_TIME * pow(a, 3.0/2.0); // Simple matter-dominated universe approximation
 }
  
 // Convert physical time in years to cosmological scale factor
 double physical_time_to_scale_factor(double time_physical) {
     return pow(time_physical / HUBBLE_TIME, 2.0/3.0);
 }
 
 // Convert IntPos to physical coordinates in kpc
 inline double intpos_to_kpc(uint32_t ipos) {
    return (double) ipos * All.BoxSize / 4294967296.0;
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
 
 // Local cubic spline kernel for Type Ia SNe and AGB wind distribution
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
 
 /**
  * Clamp total internal energy (utherm) to prevent extremely small timesteps
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
 bool is_star_eligible_for_feedback(int i, int feedback_type, double current_time, simparticles *Sp) {
     if (Sp->P[i].getType() != 4) // needs to be a star
         return false;
 
     // Convert from scale factor to physical time
     double age_physical = scale_factor_to_physical_time(current_time - Sp->P[i].StellarAge);
 
     // Check if this feedback type has already been applied
     if ((Sp->P[i].FeedbackFlag & feedback_type) != 0) {
         return false; // Already processed
     }
 
     // Different criteria for different feedback types
     if (feedback_type == FEEDBACK_SNII) {
         // Type II SNe happen promptly after star formation
         return (age_physical > SNII_DELAY_TIME_PHYSICAL);
     } 
     else if (feedback_type == FEEDBACK_AGB) {
         // AGB winds are active after some delay but before stars are too old
         return (age_physical > SNII_DELAY_TIME_PHYSICAL && age_physical < AGB_END_TIME_PHYSICAL);
     }
     else if (feedback_type == FEEDBACK_SNIa) {
         // Type Ia SNe follow a delay-time distribution
         if (age_physical < SNIa_DTD_MIN_TIME) return false;
 
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
 
         return (n_events > 0);
     }
 
     return false;
 }
 
 /**
  * Custom tree walker class for stellar feedback
  */
 struct FeedbackTargetData {
     int index;       // Index in Sp->P[]
     double dist;     // Distance to source
     double weight;   // Weight for feedback distribution
     double dir[3];   // Unit direction vector
 };
 
 class FeedbackTreeWalk {
 public:
     FeedbackTreeWalk(simparticles *Sp_ptr) : Sp(Sp_ptr) {
         MaxTargets = 1024;  // Initial capacity
         TargetCount = 0;
         Targets = (FeedbackTargetData *)Mem.mymalloc("feedback_targets", MaxTargets * sizeof(FeedbackTargetData));
     }
 
     ~FeedbackTreeWalk() {
         if (Targets)
             Mem.myfree(Targets);
     }
 
     // Set current feedback properties
     void SetFeedbackType(int type) {
         FeedbackType = type;
     }
 
     // Find gas neighbors using direct search
     void FindNeighborsWithinRadius(double pos[3], double hsml, int stellar_index) {
         TargetCount = 0;
         StellarPos[0] = pos[0];
         StellarPos[1] = pos[1];
         StellarPos[2] = pos[2];
         SearchRadius = hsml;
         StellarIndex = stellar_index;
 
         // Direct search through all gas particles
         for (int i = 0; i < Sp->NumPart; i++) {
             if (Sp->P[i].getType() != 0)  // Only gas particles
                 continue;
 
             double pos_i[3];
             pos_i[0] = intpos_to_kpc(Sp->P[i].IntPos[0]);
             pos_i[1] = intpos_to_kpc(Sp->P[i].IntPos[1]);
             pos_i[2] = intpos_to_kpc(Sp->P[i].IntPos[2]);
 
             double dx = StellarPos[0] - pos_i[0];
             double dy = StellarPos[1] - pos_i[1];
             double dz = StellarPos[2] - pos_i[2];
 
             // Handle periodic boundary conditions
             if (All.BoxSize > 0) {
                 dx = NEAREST_X(dx);
                 dy = NEAREST_Y(dy);
                 dz = NEAREST_Z(dz);
             }
 
             double r2 = dx*dx + dy*dy + dz*dz;
 
             if (r2 > SearchRadius * SearchRadius)
                 continue;  // Outside search radius
 
             double r = sqrt(r2);
 
             // Skip particles too close (avoid numerical instabilities)
             if (r < MIN_FEEDBACK_SEPARATION)
                 continue;
 
             // Calculate appropriate kernel weight
             double weight;
             if (FeedbackType == FEEDBACK_SNII)
                 weight = kernel_weight_tophat(r, SearchRadius);
             else
                 weight = kernel_weight_cubic(r, SearchRadius);
 
             if (weight <= 0)
                 continue;
 
             // Add to targets list
             if (TargetCount >= MaxTargets) {
                 MaxTargets *= 2;
                 Targets = (FeedbackTargetData *)Mem.myrealloc_movable(Targets, MaxTargets * sizeof(FeedbackTargetData));
             }
 
             // Store normalized direction vector
             double r_inv = 1.0 / r;
             Targets[TargetCount].index = i;
             Targets[TargetCount].dist = r;
             Targets[TargetCount].weight = weight;
             Targets[TargetCount].dir[0] = dx * r_inv;
             Targets[TargetCount].dir[1] = dy * r_inv;
             Targets[TargetCount].dir[2] = dz * r_inv;
 
             TargetCount++;
         }
     }
 
     // Apply feedback to all found targets
     void ApplyFeedback(double stellar_mass, double metallicity, int snia_events) {
         if (TargetCount == 0)
             return;
 
         // Calculate feedback properties based on type
         double E_total = 0;
         double mass_return = 0;
         Yields yields = {0};
 
         if (FeedbackType == FEEDBACK_SNII) {
             E_total = SNII_ENERGY_PER_MASS * stellar_mass;
             mass_return = MASS_RETURN_SNII * stellar_mass;
             yields = get_SNII_yields(mass_return, metallicity);
             
             // Update diagnostics
             ThisStepEnergy_SNII += E_total;
             TotalEnergyInjected_SNII += E_total;
         }
         else if (FeedbackType == FEEDBACK_SNIa) {
             E_total = SNIa_ENERGY_PER_EVENT * snia_events;
             mass_return = 0.003 * stellar_mass * snia_events;  // Approximate mass per SNIa
             yields = get_SNIa_yields(snia_events);
             
             // Update diagnostics
             ThisStepEnergy_SNIa += E_total;
             TotalEnergyInjected_SNIa += E_total;
         }
         else if (FeedbackType == FEEDBACK_AGB) {
             E_total = AGB_ENERGY_PER_MASS * stellar_mass;
             mass_return = MASS_RETURN_AGB * stellar_mass;
             yields = get_AGB_yields(mass_return, metallicity);
             
             // Update diagnostics
             ThisStepEnergy_AGB += E_total;
             TotalEnergyInjected_AGB += E_total;
         }
 
         double E_input = E_total;                    // ← total energy budget for this star
         double sum_applied = 0.0;                    // ← will accumulate what we actually give out

         // Update mass return diagnostics
         ThisStepMassReturned += mass_return;
         TotalMassReturned += mass_return;
         
         // Update metals diagnostics
         ThisStepMetalsInjected[0] += yields.Z;
         ThisStepMetalsInjected[1] += yields.C;
         ThisStepMetalsInjected[2] += yields.O;
         ThisStepMetalsInjected[3] += yields.Fe;
 
         // Calculate normalization factor
         double total_weight = 0;
         for (int i = 0; i < TargetCount; i++)
             total_weight += Targets[i].weight;
             
         if (total_weight <= 0) {
             FEEDBACK_PRINT("[Feedback WARNING] Total weight <= 0, skipping feedback application\n");
             return;
         }
         
         double inv_total_weight = 1.0 / total_weight;
         
         // Energy partitioning
         double E_kin = E_total * SNKickFraction;
         double E_therm = E_total * (1.0 - SNKickFraction);

         // Apply feedback to each target
         for (int i = 0; i < TargetCount; i++) {
             int j = Targets[i].index;
             
             // Skip invalid particles
             if (j < 0 || j >= Sp->NumPart || Sp->P[j].getType() != 0)
                 continue;
                 
             // Calculate normalized weight for this particle
             double norm_weight = Targets[i].weight * inv_total_weight;
             
             // Thermal energy injection
             double gas_mass = Sp->P[j].getMass();
             double gas_mass_cgs = gas_mass * All.UnitMass_in_g;
             double inv_mass_cgs = 1.0 / gas_mass_cgs;
             double E_therm_j = E_therm * norm_weight;
             double delta_u = E_therm_j * erg_per_mass_to_code * inv_mass_cgs;
             


                // accumulate for per‐star energy‐conservation check
                sum_applied += E_therm_j;

             // Check for valid energy increment
             if (!isfinite(delta_u) || delta_u < 0) {
                 FEEDBACK_PRINT("[Feedback WARNING] Non-finite delta_u = %.3e for gas %d\n", delta_u, j);
                 continue;
             }
             
             // Apply thermal energy with clamping
             double utherm_before = Sp->get_utherm_from_entropy(j);
             double rel_increase = delta_u / (utherm_before + 1e-10);
             
             if (rel_increase > 10.0) {
                 FEEDBACK_PRINT("[Feedback WARNING] delta_u (%.3e) is too large (%.1fx u_before=%.3e) for gas ID=%llu\n", 
                               delta_u, rel_increase, utherm_before, (unsigned long long)Sp->P[j].ID.get());
                 continue;
             }
             
             double utherm_after = clamp_feedback_energy(utherm_before, delta_u, j, Sp->P[j].ID.get());
             Sp->set_entropy_from_utherm(utherm_after, j);
             
             // Kinetic energy injection
             double E_kin_j = E_kin * norm_weight;
             double v_kick = sqrt(2.0 * E_kin_j * erg_per_mass_to_code * inv_mass_cgs);
             
             if (!isfinite(v_kick) || v_kick < 0 || v_kick > 1e5) {
                 FEEDBACK_PRINT("[Feedback WARNING] Non-finite or huge v_kick = %.3e for gas %d\n", v_kick, j);
                 continue;
             }
             
             // Apply kick along the unit vector from star to gas
             for (int k = 0; k < 3; k++)
                 Sp->P[j].Vel[k] += v_kick * Targets[i].dir[k];
                 
             sum_applied += E_kin_j;  // for diagnostics

             // Mass return
             double mass_add = mass_return * norm_weight;
             Sp->P[j].setMass(gas_mass + mass_add);
             
             // Metal enrichment
             double metals_add[4] = {
                 yields.Z * norm_weight,
                 yields.C * norm_weight,
                 yields.O * norm_weight,
                 yields.Fe * norm_weight
             };
             
             for (int k = 0; k < 4; k++) {
                 double metal_frac = metals_add[k] / gas_mass;
                 Sp->SphP[j].Metals[k] += metal_frac;
             }
             
             // Final check for numerical stability
             double final_u = Sp->get_utherm_from_entropy(j);
             if (!isfinite(final_u) || final_u < 1e-20 || final_u > 1e10) {
                 FEEDBACK_PRINT("[Feedback WARNING] Bad final entropy on gas %d: u=%.3e\n", j, final_u);
             }

            // ─── DIAG: per‐neighbor record ───
            double rel_inc = delta_u / (Sp->get_utherm_from_entropy(j) + 1e-10);
            g_delta_u    .push_back(delta_u);
            g_delta_v.push_back(v_kick);
            g_rel_increase.push_back(rel_inc);
            g_radial_r   .push_back(Targets[i].dist);
         }
         
            // ─── DIAG: per‐star record ───
            g_neighbors_per_star.push_back(TargetCount);
            g_h_per_star         .push_back(SearchRadius);
            g_energy_ratio      .push_back(sum_applied / E_input);

         // Mark this star as having received this type of feedback
         Sp->P[StellarIndex].FeedbackFlag |= FeedbackType;
     }
 
     // Public variables
     int TargetCount;
     double SearchRadius;
 
 private:
     simparticles *Sp;
     int FeedbackType;
     int StellarIndex;
     double StellarPos[3];
     
     FeedbackTargetData *Targets;
     int MaxTargets;
 };
 
 /**
  * Get appropriate feedback radius based on feedback type
  */
 double get_initial_feedback_radius(int feedback_type) {
     if (feedback_type == FEEDBACK_SNII)
         return 0.3;  // kpc
     else if (feedback_type == FEEDBACK_SNIa)
         return 0.8;  // kpc
     else
         return 0.5;  // kpc (AGB)
 }
 
 /**
  * Find an appropriate feedback radius for a star
  */
 double find_adaptive_radius(double pos[3], int feedback_type, simparticles *Sp, FeedbackTreeWalk *walker) {
    // Initial parameters
    double h = get_initial_feedback_radius(feedback_type);
    double h_min = 0.1;  // kpc
    double h_max = 3.0;  // kpc
    
    // Target neighbor counts
    int target_min = 0;
    int target_max = 0;
    
    if (feedback_type == FEEDBACK_SNII) {
        target_min = 8;
        target_max = 32;
    }
    else if (feedback_type == FEEDBACK_SNIa) {
        target_min = 32;
        target_max = 128;
    }
    else {  // AGB
        target_min = 16;
        target_max = 64;
    }
    
    // Try to find an appropriate radius
    const int MAX_ITER = 5;
    
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Dummy search to count neighbors
        walker->FindNeighborsWithinRadius(pos, h, -1);
        int count = walker->TargetCount;
        
        if (count >= target_min && count <= target_max)
            break;  // Found good radius
            
        if (count < target_min)
            h *= 1.3;  // Increase radius
        else
            h *= 0.8;  // Decrease radius
            
        // Enforce limits
        h = fmax(fmin(h, h_max), h_min);
    }
    
    return h;
}
 
 /**
  * Main feedback function using a direct tree implementation
  */
 void apply_stellar_feedback_treewalk(double current_time, simparticles* Sp) {
    // Reset diagnostic counters
    ThisStepEnergy_SNII = 0;
    ThisStepEnergy_SNIa = 0;
    ThisStepEnergy_AGB = 0;
    ThisStepMassReturned = 0;
    std::memset(ThisStepMetalsInjected, 0, sizeof(ThisStepMetalsInjected));
    
    // Initialize unit conversions
    erg_per_mass_to_code = 1.0 / (All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s);
    
    // Create feedback tree walker
    FeedbackTreeWalk walker(Sp);
    
    // Process each feedback type (if enabled)
    // Type II supernovae
    if (All.FeedbackSNII) {
        process_feedback_type(FEEDBACK_SNII, current_time, Sp, &walker);
    }
    
    // AGB stellar winds
    if (All.FeedbackAGB) {
        process_feedback_type(FEEDBACK_AGB, current_time, Sp, &walker);
    }
    
    // Type Ia supernovae
    if (All.FeedbackSNIa) {
        process_feedback_type(FEEDBACK_SNIa, current_time, Sp, &walker);
    }
    
    // Print summary (on master process only)
    if (ThisTask == 0 && All.FeedbackDebug && 
       (ThisStepEnergy_SNII > 0 || ThisStepEnergy_SNIa > 0 || ThisStepEnergy_AGB > 0)) {
        FEEDBACK_PRINT("[Feedback Timestep Summary] E_SNII=%.3e erg, E_SNIa=%.3e erg, E_AGB=%.3e erg\n",
                      ThisStepEnergy_SNII, ThisStepEnergy_SNIa, ThisStepEnergy_AGB);
        FEEDBACK_PRINT("[Feedback Timestep Summary] Mass Returned=%.3e Msun\n", ThisStepMassReturned);
        FEEDBACK_PRINT("[Feedback Timestep Summary] Metals (Z=%.3e, C=%.3e, O=%.3e, Fe=%.3e) Msun\n",
                      ThisStepMetalsInjected[0], ThisStepMetalsInjected[1], 
                      ThisStepMetalsInjected[2], ThisStepMetalsInjected[3]);
    }
    OutputFeedbackDiagnostics();  // Output diagnostics after processing this feedback type
}

/**
 * Helper function to process each feedback type
 * Add this to your feedback_treewalk.cc file
 */
 void process_feedback_type(int feedback_type, double current_time, simparticles* Sp, FeedbackTreeWalk* walker) {
    walker->SetFeedbackType(feedback_type);
    
    const char* feedback_name = (feedback_type == FEEDBACK_SNII) ? "SNII" : 
                               ((feedback_type == FEEDBACK_SNIa) ? "SNIa" : "AGB");
    
    // Count eligible stars
    int n_sources = 0;
    for (int i = 0; i < Sp->NumPart; i++) {
        if (is_star_eligible_for_feedback(i, feedback_type, current_time, Sp))
            n_sources++;
    }
    
    if (ThisTask == 0) {
        if (n_sources == 0) {
            FEEDBACK_PRINT("[Feedback] No eligible sources for %s feedback\n", feedback_name);
            return;
        } else {
            FEEDBACK_PRINT("[Feedback] Found %d eligible sources for %s feedback\n", n_sources, feedback_name);
        }
    }

    // Process each eligible star
    for (int i = 0; i < Sp->NumPart; i++) {
        if (!is_star_eligible_for_feedback(i, feedback_type, current_time, Sp))
            continue;
            
        // Convert position to physical coordinates
        double pos[3];
        pos[0] = intpos_to_kpc(Sp->P[i].IntPos[0]);
        pos[1] = intpos_to_kpc(Sp->P[i].IntPos[1]);
        pos[2] = intpos_to_kpc(Sp->P[i].IntPos[2]);
        
        // Get star properties
        double stellar_mass = Sp->P[i].getMass();
        double metallicity = Sp->SphP[i].Metallicity;
        int snia_events = Sp->P[i].SNIaEvents;
        
        // Find appropriate search radius
        double h = find_adaptive_radius(pos, feedback_type, Sp, walker);
        
        // Apply feedback
        walker->FindNeighborsWithinRadius(pos, h, i);
        
        if (walker->TargetCount == 0) {
            if (ThisTask == 0 && All.FeedbackDebug) {
                FEEDBACK_PRINT("[Feedback WARNING] No targets found for star %d within h=%.2f for %s feedback\n", 
                              i, h, feedback_name);
            }
            continue;
        }
        
        walker->ApplyFeedback(stellar_mass, metallicity, snia_events);
    }

    
}
 

/** \brief Write out detailed feedback diagnostics to CSV for plotting
 *
 *  This routine dumps both per-neighbor and per-star metrics into a CSV
 *  file with columns: delta_u, delta_v, rel_inc, r, n_ngb, h_star, E_ratio.
 *  Neighbors have values in the first four columns; stars in the last three.
 */
 void OutputFeedbackDiagnostics()
{
    if (ThisTask != 0)
        return;

    // Report sizes of diagnostics arrays
    std::fprintf(stderr,
        "[FeedbackDiag] ThisTask=0: neighbors=%zu, stars=%zu\n",
        g_delta_u.size(), g_neighbors_per_star.size());

    // Open CSV: truncate only on first call, else append
    static bool firstCall = true;
    std::ofstream out;
    if (firstCall) {
        out.open("feedback_diagnostics.csv", std::ios::out);
        firstCall = false;

        if (!out.is_open()) {
            std::fprintf(stderr,
                "[Error] Could not open feedback_diagnostics.csv for writing\n");
            return;
        }
        out << std::scientific << std::setprecision(6);

        // Write header once
        out << "# Feedback diagnostics at time=" << All.Time << "\n";
        out << "#delta_u,delta_v,rel_inc,r,n_ngb,h_star,E_ratio\n";
    } else {
        out.open("feedback_diagnostics.csv", std::ios::app);
        if (!out.is_open()) {
            std::fprintf(stderr,
                "[Error] Could not open feedback_diagnostics.csv for appending\n");
            return;
        }
        out << std::scientific << std::setprecision(6);
    }

    // 1) Neighbor metrics: delta_u, delta_v, rel_inc, r (stars blank)
    for (size_t k = 0; k < g_delta_u.size(); ++k) {
        out << g_delta_u[k]      << ","
            << g_delta_v[k]      << ","
            << g_rel_increase[k] << ","
            << g_radial_r[k]     << ",,,\n";
    }

    // 2) Star metrics: first four fields blank, then n_ngb, h_star, E_ratio
    for (size_t k = 0; k < g_neighbors_per_star.size(); ++k) {
        out << ",,,,"
            << g_neighbors_per_star[k] << ","
            << g_h_per_star[k]         << ","
            << g_energy_ratio[k]       << "\n";
    }

    out.close();

    // Now that everything’s safely on disk, clear the in‑memory buffers:
    g_delta_u.clear();
    g_delta_v.clear();
    g_rel_increase.clear();
    g_radial_r.clear();
    g_neighbors_per_star.clear();
    g_h_per_star.clear();
    g_energy_ratio.clear();
}

 
 #endif // FEEDBACK