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
 
 ❖ Key components:
    - Uses Gadget-4's existing tree traversal functions
    - Applies appropriate kernel weighting for energy/mass/metal distribution
    - Maintains the same physics as the original implementation
    - Improves performance dramatically for larger simulations
 
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
 
 #include "../gravity/forcetree.h"
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
 
 /**
  * Convert cosmological scale factor to physical time in years
  */
 double scale_factor_to_physical_time(double a) {
     return HUBBLE_TIME * pow(a, 3.0/2.0); // Simple matter-dominated universe approximation
 }
 
 /**
  * Convert physical time in years to cosmological scale factor
  */
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
  * Data structure for feedback sources (stars)
  */
 struct feedback_source_data {
     int index;              // Index of star particle
     MyDouble Pos[3];        // Position in kpc
     MyFloat Mass;           // Mass in internal units
     MyFloat StellarAge;     // Scale factor at formation
     MyFloat Metallicity;    // Metallicity
     int FeedbackType;       // Type of feedback to apply
     int SNIaEvents;         // Number of SNIa events (if applicable)
     int FeedbackFlag;       // Current feedback flags
 };
 
 /**
  * Data structure for feedback targets (gas particles)
  */
 struct feedback_target_data {
     int index;              // Index of gas particle
     MyDouble Pos[3];        // Position in kpc
     MyFloat Mass;           // Mass in internal units
     MyFloat Energy;         // Energy received from feedback
     MyFloat MassReceived;   // Mass received from feedback
     MyFloat MetalsReceived[4]; // Metals received from feedback
     MyFloat Vel[3];         // Velocity for kinetic feedback
     double dist;            // Distance to source star
     double weight;          // Kernel weight
 };
 
 // Target list for treewalk
 struct feedback_target_list {
     feedback_target_data *targets;
     int n_targets;          // Number of targets found
     int max_targets;        // Maximum capacity
     double hsml;            // Current search radius
     int target_count;       // Target count for normalization
     int feedback_type;      // Type of feedback currently being processed
 };
 
 // Parameters to determine appropriate feedback radius
 struct feedback_radius_params {
     int target_count_min;   // Minimum number of targets
     int target_count_max;   // Maximum number of targets
     double h_min;           // Minimum radius in kpc
     double h_max;           // Maximum radius in kpc
 };
 
 /**
  * Get appropriate feedback parameters based on feedback type
  */
 feedback_radius_params get_feedback_params(int feedback_type) {
     feedback_radius_params params;
     
     if (feedback_type == FEEDBACK_SNII) {
         params.target_count_min = 8;
         params.target_count_max = 32;
         params.h_min = 0.1;
         params.h_max = 1.0;
     } 
     else if (feedback_type == FEEDBACK_SNIa) {
         params.target_count_min = 32;
         params.target_count_max = 128;
         params.h_min = 0.3;
         params.h_max = 2.0;
     }
     else { // FEEDBACK_AGB
         params.target_count_min = 16;
         params.target_count_max = 64;
         params.h_min = 0.2;
         params.h_max = 1.5;
     }
     
     return params;
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
  * Create a list of eligible stellar feedback sources
  */
 feedback_source_data* build_source_list(int feedback_type, double current_time, simparticles *Sp, int *n_sources) {
     // First count eligible sources
     int count = 0;
     for (int i = 0; i < Sp->NumPart; i++) {
         if (is_star_eligible_for_feedback(i, feedback_type, current_time, Sp))
             count++;
     }
     
     *n_sources = count;
     
     if (count == 0)
         return NULL;
         
     // Allocate and fill source list
     feedback_source_data *sources = (feedback_source_data*) malloc(count * sizeof(feedback_source_data));
     
     int idx = 0;
     for (int i = 0; i < Sp->NumPart; i++) {
         if (is_star_eligible_for_feedback(i, feedback_type, current_time, Sp)) {
             sources[idx].index = i;
             sources[idx].Pos[0] = intpos_to_kpc(Sp->P[i].IntPos[0]);
             sources[idx].Pos[1] = intpos_to_kpc(Sp->P[i].IntPos[1]);
             sources[idx].Pos[2] = intpos_to_kpc(Sp->P[i].IntPos[2]);
             sources[idx].Mass = Sp->P[i].getMass();
             sources[idx].StellarAge = Sp->P[i].StellarAge;
             sources[idx].Metallicity = Sp->SphP[i].Metallicity;
             sources[idx].FeedbackType = feedback_type;
             sources[idx].SNIaEvents = Sp->P[i].SNIaEvents;
             sources[idx].FeedbackFlag = Sp->P[i].FeedbackFlag;
             idx++;
         }
     }
     
     return sources;
 }
 
 /**
  * Tree node evaluation function for feedback
  */
 bool feedback_tree_node_evaluation(TreeNodeExtended *tree, double *pos, double hsml, node_info &nodeinfo) {
     if (!(tree->getFlags() & NODE_CONTAINS_GAS))
         return false;  // Skip nodes without gas particles
 
     // Get node boundaries
     MyIntPosType minpos[3], maxpos[3];
     tree->getnodes()->get_node_ranges(tree->get_nodeid(), minpos, maxpos);
     
     // Convert to physical coordinates (kpc)
     double min_x = intpos_to_kpc(minpos[0]);
     double min_y = intpos_to_kpc(minpos[1]);
     double min_z = intpos_to_kpc(minpos[2]);
     double max_x = intpos_to_kpc(maxpos[0]);
     double max_y = intpos_to_kpc(maxpos[1]);
     double max_z = intpos_to_kpc(maxpos[2]);
     
     // Calculate distance from search position to closest point on node
     double dx = 0, dy = 0, dz = 0;
     
     // X-direction
     if (pos[0] < min_x)
         dx = NEAREST_X(min_x - pos[0]);
     else if (pos[0] > max_x)
         dx = NEAREST_X(pos[0] - max_x);
         
     // Y-direction
     if (pos[1] < min_y)
         dy = NEAREST_Y(min_y - pos[1]);
     else if (pos[1] > max_y)
         dy = NEAREST_Y(pos[1] - max_y);
         
     // Z-direction
     if (pos[2] < min_z)
         dz = NEAREST_Z(min_z - pos[2]);
     else if (pos[2] > max_z)
         dz = NEAREST_Z(pos[2] - max_z);
     
     // Distance to closest point on node
     double dist2 = dx*dx + dy*dy + dz*dz;
     
     // If node is at least partially within search radius, we need to open it
     if (dist2 <= hsml*hsml)
         return true;  // Open node
         
     return false;  // Skip node
 }
 
 /**
  * Collect target gas particles within search radius using the tree
  */
 void collect_gas_targets_tree(double pos[3], double hsml, simparticles *Sp, 
                               forcetree *Tree, feedback_target_list *target_list) {
     // Initialize target list if needed
     if (target_list->targets == NULL) {
         target_list->max_targets = 1024;  // Initial capacity
         target_list->targets = (feedback_target_data*) malloc(target_list->max_targets * sizeof(feedback_target_data));
         target_list->n_targets = 0;
     }
     
     // Clear target list
     target_list->n_targets = 0;
     target_list->hsml = hsml;
     
     // Create a node_info structure for the tree walk
     node_info nodeinfo;
     nodeinfo.Tree = Tree;
     
     // Start recursive tree walk from the root node
     Tree->treewalk_all(pos, hsml, [&](int i, double r2) {
         // Check if this is a gas particle
         if (Sp->P[i].getType() != 0)
             return;
             
         double r = sqrt(r2);
         
         // Skip particles too close to avoid numerical issues
         if (r < MIN_FEEDBACK_SEPARATION)
             return;
             
         // Get appropriate kernel weight based on feedback type
         double weight;
         if (target_list->feedback_type == FEEDBACK_SNII)
             weight = kernel_weight_tophat(r, hsml);
         else
             weight = kernel_weight_cubic(r, hsml);
             
         if (weight <= 0)
             return;
             
         // Resize target list if needed
         if (target_list->n_targets >= target_list->max_targets) {
             target_list->max_targets *= 2;
             target_list->targets = (feedback_target_data*) realloc(target_list->targets, 
                                    target_list->max_targets * sizeof(feedback_target_data));
         }
         
         // Add to target list
         int idx = target_list->n_targets++;
         target_list->targets[idx].index = i;
         target_list->targets[idx].Pos[0] = intpos_to_kpc(Sp->P[i].IntPos[0]);
         target_list->targets[idx].Pos[1] = intpos_to_kpc(Sp->P[i].IntPos[1]);
         target_list->targets[idx].Pos[2] = intpos_to_kpc(Sp->P[i].IntPos[2]);
         target_list->targets[idx].Mass = Sp->P[i].getMass();
         target_list->targets[idx].dist = r;
         target_list->targets[idx].weight = weight;
         
         // For velocity kick calculation
         MyFloat dx[3] = {
             NEAREST_X(target_list->targets[idx].Pos[0] - pos[0]),
             NEAREST_Y(target_list->targets[idx].Pos[1] - pos[1]),
             NEAREST_Z(target_list->targets[idx].Pos[2] - pos[2])
         };
         
         for (int k = 0; k < 3; k++)
             target_list->targets[idx].Vel[k] = dx[k] / r;  // Unit vector
             
     }, 
     [&](TreeNodeExtended *tree) -> bool {
         return feedback_tree_node_evaluation(tree, pos, hsml, nodeinfo);
     });
     
     // Update target count for normalization
     target_list->target_count = target_list->n_targets;
 }
 
 /**
  * Find an appropriate search radius that gives a reasonable number of neighbors
  */
 double find_adaptive_radius(double pos[3], int feedback_type, simparticles *Sp, forcetree *Tree, feedback_target_list *target_list) {
     feedback_radius_params params = get_feedback_params(feedback_type);
     
     target_list->feedback_type = feedback_type;
     
     double h = 0.5;  // Initial guess
     const int MAX_ITERATIONS = 5;
     
     for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
         // Collect targets with current radius
         collect_gas_targets_tree(pos, h, Sp, Tree, target_list);
         
         int n_targets = target_list->n_targets;
         
         FEEDBACK_PRINT("[Feedback Adaptive h] Iteration %d: h=%.3f found %d targets (want %d-%d)\n",
                        iter, h, n_targets, params.target_count_min, params.target_count_max);
         
         // Check if we have a reasonable number of targets
         if (n_targets >= params.target_count_min && n_targets <= params.target_count_max)
             break;
             
         // Adjust radius
         if (n_targets < params.target_count_min)
             h *= 1.3;  // Increase search radius
         else
             h *= 0.8;  // Decrease search radius
             
         // Clamp to allowed range
         h = fmax(fmin(h, params.h_max), params.h_min);
     }
     
     return h;
 }
 
 /**
  * Apply feedback effects to target gas particles
  */
 void apply_feedback_to_targets(feedback_source_data *source, feedback_target_list *target_list, simparticles *Sp) {
     int n_targets = target_list->n_targets;
     if (n_targets == 0)
         return;
         
     // Calculate total energy and mass to distribute
     double E_total = 0;
     double mass_return = 0;
     Yields yields = {0};
     
     int feedback_type = source->FeedbackType;
     
     if (feedback_type == FEEDBACK_SNII) {
         E_total = SNII_ENERGY_PER_MASS * source->Mass;
         mass_return = MASS_RETURN_SNII * source->Mass;
         yields = get_SNII_yields(mass_return, source->Metallicity);
         
         // Update diagnostics
         ThisStepEnergy_SNII += E_total;
         TotalEnergyInjected_SNII += E_total;
     }
     else if (feedback_type == FEEDBACK_SNIa) {
         E_total = SNIa_ENERGY_PER_EVENT * source->SNIaEvents;
         mass_return = 0.003 * source->Mass * source->SNIaEvents;  // Approximate mass per SNIa
         yields = get_SNIa_yields(source->SNIaEvents);
         
         // Update diagnostics
         ThisStepEnergy_SNIa += E_total;
         TotalEnergyInjected_SNIa += E_total;
     }
     else if (feedback_type == FEEDBACK_AGB) {
         E_total = AGB_ENERGY_PER_MASS * source->Mass;
         mass_return = MASS_RETURN_AGB * source->Mass;
         yields = get_AGB_yields(mass_return, source->Metallicity);
         
         // Update diagnostics
         ThisStepEnergy_AGB += E_total;
         TotalEnergyInjected_AGB += E_total;
     }
     
     // Update mass return diagnostics
     ThisStepMassReturned += mass_return;
     TotalMassReturned += mass_return;
     
     // Update metals diagnostics
     ThisStepMetalsInjected[0] += yields.Z;
     ThisStepMetalsInjected[1] += yields.C;
     ThisStepMetalsInjected[2] += yields.O;
     ThisStepMetalsInjected[3] += yields.Fe;
     
     // Calculate total weight for normalization
     double total_weight = 0;
     for (int i = 0; i < n_targets; i++) {
         total_weight += target_list->targets[i].weight;
     }
     
     if (total_weight <= 0) {
         FEEDBACK_PRINT("[Feedback WARNING] Total weight <= 0, skipping feedback application\n");
         return;
     }
     
     // Normalize weights
     double inv_total_weight = 1.0 / total_weight;
     
     // Apply feedback to each target
     for (int i = 0; i < n_targets; i++) {
         feedback_target_data *target = &target_list->targets[i];
         int j = target->index;
         
         // Calculate normalized weight
         double norm_weight = target->weight * inv_total_weight;
         
         // Distribute energy
         double E_kin = E_total * SNKickFraction * norm_weight;
         double E_therm = E_total * (1.0 - SNKickFraction) * norm_weight;
         
         // Convert thermal energy to specific internal energy
         double gas_mass_cgs = target->Mass * All.UnitMass_in_g;
         double inv_mass_cgs = 1.0 / gas_mass_cgs;
         double delta_u = E_therm * erg_per_mass_to_code * inv_mass_cgs;
         
         // Check for valid energy increment
         if (!isfinite(delta_u) || delta_u < 0) {
             FEEDBACK_PRINT("[Feedback WARNING] Non-finite delta_u = %.3e for gas %d\n", delta_u, j);
             continue;
         }
         
         // Apply thermal energy update with clamping
         double utherm_before = Sp->get_utherm_from_entropy(j);
         double utherm_after = clamp_feedback_energy(utherm_before, delta_u, j, Sp->P[j].ID.get());
         
         double rel_increase = delta_u / (utherm_before + 1e-10);
         if (rel_increase > 10.0) {
             FEEDBACK_PRINT("[Feedback WARNING] delta_u (%.3e) is too large (%.1fx u_before=%.3e) for gas ID=%llu\n", 
                            delta_u, rel_increase, utherm_before, (unsigned long long)Sp->P[j].ID.get());
             continue;
         }
         
         // Update gas thermal energy
         Sp->set_entropy_from_utherm(utherm_after, j);
         
         // Apply kinetic kick
         double v_kick = sqrt(2.0 * E_kin * erg_per_mass_to_code * inv_mass_cgs);
         if (!isfinite(v_kick) || v_kick < 0 || v_kick > 1e5) {
             FEEDBACK_PRINT("[Feedback WARNING] Non-finite or huge v_kick = %.3e for gas %d\n", v_kick, j);
             continue;
         }
         
         // Apply kick along unit vector from star to gas
         for (int k = 0; k < 3; k++)
             Sp->P[j].Vel[k] += v_kick * target->Vel[k];
         
         // Mass return
         double mass_add = mass_return * norm_weight;
         Sp->P[j].setMass(target->Mass + mass_add);
         
         // Metal enrichment
         double metals_add[4] = {
             yields.Z * norm_weight,
             yields.C * norm_weight,
             yields.O * norm_weight,
             yields.Fe * norm_weight
         };
         
         for (int k = 0; k < 4; k++) {
             double metal_frac = metals_add[k] / target->Mass;
             Sp->SphP[j].Metals[k] += metal_frac;
         }
         
         // Final check for numerical stability
         double final_u = Sp->get_utherm_from_entropy(j);
         if (!isfinite(final_u) || final_u < 1e-20 || final_u > 1e10) {
             FEEDBACK_PRINT("[Feedback WARNING] Bad final entropy on gas %d: u=%.3e\n", j, final_u);
         }
     }
     
     // Mark this star as having received this type of feedback
     Sp->P[source->index].FeedbackFlag |= feedback_type;
 }
 
 /**
  * Main feedback function called each timestep
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
     
     // Build tree for efficient neighbor finding
     forcetree Tree(Sp, 0);
     Tree.treeallocate(Sp->NumPart, Sp);
     Tree.treebuild(0, Sp);
     
     // Initialize target list
     feedback_target_list target_list;
     target_list.targets = NULL;
     target_list.n_targets = 0;
     target_list.max_targets = 0;
     
     // Process each feedback type
     for (int feedback_type = FEEDBACK_SNII; feedback_type <= FEEDBACK_SNIa; feedback_type *= 2) {
         // Build list of eligible source stars
         int n_sources = 0;
         feedback_source_data *sources = build_source_list(feedback_type, current_time, Sp, &n_sources);
         
         if (n_sources == 0 || sources == NULL) {
             if (ThisTask == 0 && All.FeedbackDebug) {
                 FEEDBACK_PRINT("[Feedback] No eligible sources for feedback type %d\n", feedback_type);
             }
             continue;
         }
         
         if (ThisTask == 0 && All.FeedbackDebug) {
             FEEDBACK_PRINT("[Feedback] Processing %d sources for feedback type %d\n", n_sources, feedback_type);
         }
         
         // Process each source
         for (int i = 0; i < n_sources; i++) {
             // Find adaptive radius and collect targets
             double h = find_adaptive_radius(sources[i].Pos, feedback_type, Sp, &Tree, &target_list);
             
             if (target_list.n_targets == 0) {
                 FEEDBACK_PRINT("[Feedback WARNING] No targets found for star %d within h=%.2f\n", 
                               sources[i].index, h);
                 continue;
             }
             
             // Apply feedback to targets
             apply_feedback_to_targets(&sources[i], &target_list, Sp);
         }
         
         // Clean up
         free(sources);
     }
     
     // Clean up target list
     if (target_list.targets != NULL)
         free(target_list.targets);
     
     // Free tree
     Tree.treefree();
     
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
 }
 
 #endif // FEEDBACK

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
 double adaptive_feedback_radius(MyDouble starPos[3], int feedback_type,
    simparticles *Sp, int *neighbors_ptr,
    void *unused1, int max_neighbors) {

    int *neighbor_list = static_cast<int*>(unused1);

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


extern gravtree<simparticles> GravTree;
extern domain<simparticles>  Domain;



// Maximum we’ll ever allow for a feedback neighbor list
#define MAX_FEEDBACK_NEIGHBORS 1024

void run_feedback(double current_time, int feedback_type, simparticles *Sp)
{
    FeedbackWalk fw;
    fw.current_time  = current_time;
    fw.feedback_type = feedback_type;

    // Temporary container for neighbor‐indices
    int neighbor_list[MAX_FEEDBACK_NEIGHBORS];
    int n_neighbors;

    FeedbackInput  in;
    FeedbackResult out;

    // Loop over *all* particles, pick out the stars whose feedback is active
    for (int i = 0; i < Sp->NumPart; i++)
    {
        if (Sp->P[i].getType() != 4)           continue;  // only stars
        if (!feedback_isactive(i, &fw, Sp))    continue;  // only active ones

        // Convert the star’s position into kpc (your helper)
        MyDouble starPos[3] = {
            intpos_to_kpc(Sp->P[i].IntPos[0]),
            intpos_to_kpc(Sp->P[i].IntPos[1]),
            intpos_to_kpc(Sp->P[i].IntPos[2])
        };

        // Fill in the FeedbackInput for this star
        in.Pos[0]        = starPos[0];
        in.Pos[1]        = starPos[1];
        in.Pos[2]        = starPos[2];
        in.FeedbackType  = feedback_type;
        in.Energy        = SNII_ENERGY_PER_MASS * Sp->P[i].getMass();
        in.MassReturn    = 0.1 * Sp->P[i].getMass();

        // Find a good smoothing radius & get a neighbor list
        n_neighbors = 0;
        double h = adaptive_feedback_radius(
            starPos,
            feedback_type,
            Sp,
            &n_neighbors,
            neighbor_list,
            MAX_FEEDBACK_NEIGHBORS
        );
        if (h < 0 || n_neighbors == 0)     // no neighbors found
            continue;

        // Tell the rest of the code how many and what h to use
        in.h             = h;
        in.NeighborCount = n_neighbors;

        // Finally apply feedback to each neighbor
        for (int k = 0; k < n_neighbors; k++)
        {
            int j = neighbor_list[k];  // gas‐particle index
            feedback_to_gas_neighbor(&in, &out, j, &fw, Sp);
        }
    }
}



#endif // FEEDBACK