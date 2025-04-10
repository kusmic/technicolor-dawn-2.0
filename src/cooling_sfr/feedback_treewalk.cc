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
 
 // Physical constants and conversion factors
 const double HUBBLE_TIME = 13.8e9;              // Hubble time in years (approx)
 
 // Feedback energy/mass return constants
 const double SNII_ENERGY_PER_MASS = 1.0e49;     // erg / Msun
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
 const double SNII_FEEDBACK_RADIUS = 0.3;        // kpc - local deposition: ~0.3 kpc for SNII, which is more localized
 const double SNIa_FEEDBACK_RADIUS = 0.8;        // kpc - wider distribution ~0.8 kpc for SNIa to account for the more diffuse nature of these events
 const double AGB_FEEDBACK_RADIUS = 0.5;         // kpc - intermediate distribution ~0.5 kpc for AGB winds, which are more diffuse than SNII but more concentrated than SNIa
 // Added a check to make sure the gas particle is not too close, otherwise the
 // feedback is too strong, and the timestep goes to zero.
 const double MIN_FEEDBACK_SEPARATION = 1e-2;  // kpc; adjust this as needed

 // In case we have ergs and needs to get it to internal units
 const double erg_to_code = 1.0 / (All.UnitEnergy_in_cgs / All.HubbleParam);

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
 
 // Local cubic spline kernel approximation for Type Ia SNe and AGB wind distribution
 inline double kernel_weight_cubic(double r, double h) {
     double u = r / h;
     if (u < 0.5)
         return (8.0 / (M_PI * h * h * h)) * (1 - 6 * u * u + 6 * u * u * u);
     else if (u < 1.0)
         return (8.0 / (M_PI * h * h * h)) * (2.0 * pow(1.0 - u, 3));
     else
         return 0.0;
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
 
 // Structure to pass around feedback data
 struct FeedbackInput {
     MyDouble Pos[3];
     MyFloat Energy;
     MyFloat MassReturn;
     MyFloat Yield[4];
     int FeedbackType;
     double h;  // Smoothing length/radius for this feedback type
     int SourceIndex; // Index of the source star particle for diagnostics
 };
 
 struct FeedbackResult {};
 
 struct FeedbackWalk {
     double current_time;
     int feedback_type;
     const char* ev_label;
 };
 
 /**
  * Determine if a star particle is eligible for feedback
  */
 static int feedback_isactive(int i, FeedbackWalk *fw, simparticles *Sp) {
     if (Sp->P[i].getType() != 4)
         return 0;
 
     // In feedback_isactive
     // printf("[Feedback Debug] feedback_isactive() - Checking star %d, type=%d, age=%e\n", i, Sp->P[i].getType(), fw->current_time - Sp->P[i].StellarAge);
 
     // Convert from scale factor to physical time
     double age_physical = scale_factor_to_physical_time(fw->current_time - Sp->P[i].StellarAge);
 
     // Check if this feedback type has already been applied
     if ((Sp->P[i].FeedbackFlag & fw->feedback_type) != 0) {
         return 0; // Already processed
     }
 
     // Different criteria for different feedback types
     if (fw->feedback_type == FEEDBACK_SNII) {
         //printf("[Feedback Debug] Star %d is active for SNII feedback type %d\n", i, fw->feedback_type);
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
 
         printf("[Feedback Debug] Star %d is active for SNIa feedback type %d\n", i, fw->feedback_type);
         return (n_events > 0) ? 1 : 0;
     }
 
     return 0;
 }
 
 /**
  * Copy data from a star particle to the feedback input structure
  */
 static void feedback_copy(int i, FeedbackInput *out, FeedbackWalk *fw, simparticles *Sp) {
    // Convert star particle integer positions to physical positions (in kpc)
    out->Pos[0] = intpos_to_kpc( Sp->P[i].IntPos[0] );
    out->Pos[1] = intpos_to_kpc( Sp->P[i].IntPos[1] );
    out->Pos[2] = intpos_to_kpc( Sp->P[i].IntPos[2] );
    out->FeedbackType = fw->feedback_type;
    out->SourceIndex = i;  // Store source index for diagnostics
 
     double m_star = Sp->P[i].getMass();
     double z_star = Sp->P[i].Metallicity; // Assumes this field exists
     double energy = 0, m_return = 0;
     Yields y;
 
     printf("[Feedback] Reached feedback_copy().\n");
 
     // Set smoothing length/radius based on feedback type
     if (fw->feedback_type == FEEDBACK_SNII) {
         out->h = SNII_FEEDBACK_RADIUS; // Smaller radius for SNII - more localized
         energy = SNII_ENERGY_PER_MASS * m_star;
         m_return = MASS_RETURN_SNII * m_star;
         y = get_SNII_yields(m_return, z_star);
     } 
     else if (fw->feedback_type == FEEDBACK_AGB) {
         out->h = AGB_FEEDBACK_RADIUS; // Medium radius for AGB
         energy = AGB_ENERGY_PER_MASS * m_star;
         m_return = MASS_RETURN_AGB * m_star;
         y = get_AGB_yields(m_return, z_star);
     } 
     else if (fw->feedback_type == FEEDBACK_SNIa) {
         out->h = SNIa_FEEDBACK_RADIUS; // Larger radius for SNIa - more distributed
         int n_snia = Sp->P[i].SNIaEvents;
         energy = n_snia * SNIa_ENERGY_PER_EVENT;
         m_return = 0; // SNIa don't return stellar mass
         y = get_SNIa_yields(n_snia);
     }
 
     out->Energy = energy;
     out->MassReturn = m_return;
     for (int k = 0; k < 4; k++) out->Yield[k] = (&y.Z)[k];
 
     // Mark this feedback as processed
     Sp->P[i].FeedbackFlag |= fw->feedback_type;
 
     // Update diagnostic counters
     printf("[Feedback] Update feedback counters energy=%.3f\n", energy);
     if (fw->feedback_type == FEEDBACK_SNII) ThisStepEnergy_SNII += energy;
     if (fw->feedback_type == FEEDBACK_AGB)  ThisStepEnergy_AGB  += energy;
     if (fw->feedback_type == FEEDBACK_SNIa) ThisStepEnergy_SNIa += energy;
 
     ThisStepMassReturned += m_return;
     ThisStepMetalsInjected[0] += y.Z;
     ThisStepMetalsInjected[1] += y.C;
     ThisStepMetalsInjected[2] += y.O;
     ThisStepMetalsInjected[3] += y.Fe;
 
     printf("[Feedback] Copied data for star %d. ThisStepEnergy_SNII=%.3e, ThisStepMassReturned=%.3e\n",
         i, ThisStepEnergy_SNII, ThisStepMassReturned);
 }
 
 /**
  * Apply feedback to a neighboring gas particle
  * This is where we implement the different approaches for Type II vs Type Ia
  */
  static void feedback_ngb(FeedbackInput *in, FeedbackResult *out, int j, FeedbackWalk *fw, simparticles *Sp) {
    if (Sp->P[j].getType() != 0) return; // Only apply to gas particles

    // Convert gas particle's fixed-point positions to physical positions (in kpc)
    double gasPos[3];
    gasPos[0] = intpos_to_kpc( Sp->P[j].IntPos[0] );
    gasPos[1] = intpos_to_kpc( Sp->P[j].IntPos[1] );
    gasPos[2] = intpos_to_kpc( Sp->P[j].IntPos[2] );

    double dx[3] = {
        NEAREST_X(gasPos[0] - in->Pos[0]),
        NEAREST_Y(gasPos[1] - in->Pos[1]),
        NEAREST_Z(gasPos[2] - in->Pos[2])
    };

    double r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
    double r = sqrt(r2);

    if (r < MIN_FEEDBACK_SEPARATION) {
        if (r < 1e-12) {
            dx[0] = MIN_FEEDBACK_SEPARATION;
            dx[1] = 0.0;
            dx[2] = 0.0;
        } else {
            double factor = MIN_FEEDBACK_SEPARATION / r;
            dx[0] *= factor;
            dx[1] *= factor;
            dx[2] *= factor;
        }
        r = MIN_FEEDBACK_SEPARATION;
    }

    double w = 0.0;
    if (in->FeedbackType == FEEDBACK_SNII) {
        w = kernel_weight_tophat(r, in->h);
    } else {
        w = kernel_weight_cubic(r, in->h);
    }

    if (w <= 0.0) return;

    double gas_mass = Sp->P[j].getMass();

    // Convert feedback energy to code units and calculate specific energy
    double delta_u = (in->Energy * w) * erg_to_code / gas_mass;

    double max_delta_u = 1e4;  // safe cap for code units
    if (delta_u > max_delta_u) {
        printf("[Feedback Capped] Limiting Δu=%.3e to max=%.3e for gas %d\n", delta_u, max_delta_u, j);
        delta_u = max_delta_u;
    }

    Sp->SphP[j].Utherm += delta_u;

    if (delta_u > 1000.0) {
        printf("[Feedback DEBUG] Injecting Δu = %.3e (gas id=%d, w=%.3e)\n", delta_u, j, w);
    }

    // Apply velocity kicks based on feedback type
    // Note the scaled down kick strengths for SNIa and AGB
    if (r > 0) {
        double kick_strength = 0.0;
        if (in->FeedbackType == FEEDBACK_SNII) {
            kick_strength = 0.5 * WIND_VELOCITY * w;
        } else if (in->FeedbackType == FEEDBACK_SNIa) {
            kick_strength = 0.1 * WIND_VELOCITY * w;
        } else if (in->FeedbackType == FEEDBACK_AGB) {
            kick_strength = 0.05 * WIND_VELOCITY * w;
        }
        
        Sp->P[j].Vel[0] += kick_strength * dx[0] / r;
        Sp->P[j].Vel[1] += kick_strength * dx[1] / r;
        Sp->P[j].Vel[2] += kick_strength * dx[2] / r;

        // DEBUG
        double vel_after_kick = sqrt( (Sp->P[j].Vel[0] * Sp->P[j].Vel[0]) + (Sp->P[j].Vel[1] * Sp->P[j].Vel[1]) + (Sp->P[j].Vel[2] * Sp->P[j].Vel[2]) );
        printf("[Feedback DEBUG] Applying velocity kick! Gas id=%d, feedback type=%d, kick_strength=%.3e, r=%.3e, final vel km/s=%.3e\n", j, in->FeedbackType, kick_strength, r, vel_after_kick);
        
    }

    // Add returned stellar mass to gas particle
    Sp->P[j].setMass(gas_mass + in->MassReturn * w);

    // Apply metal enrichment
    if (in->MassReturn * w > 0) {
        double total_mass = gas_mass + in->MassReturn * w;
        double old_frac = gas_mass / total_mass;
        double new_frac = (in->MassReturn * w) / total_mass;

        for (int k = 0; k < 4; k++) {
            double old_metal = Sp->SphP[j].Metals[k];
            double new_metal = in->Yield[k] / in->MassReturn;
            Sp->SphP[j].Metals[k] = old_metal * old_frac + new_metal * new_frac;
        }
    } else {
        for (int k = 0; k < 4; k++) {
            Sp->SphP[j].Metals[k] += in->Yield[k] * w / gas_mass;
        }
    }
}


 
 /*
  * Add detailed diagnostics for a feedback event
  */
  void debug_feedback_diagnostics(int i, FeedbackInput *in, FeedbackWalk *fw, simparticles *Sp) {
     // Count neighbors and track energy distribution
     int neighbor_count = 0;
     double total_energy_distributed = 0.0;
     double total_kernel_weight = 0.0;
     double max_gas_temp_before = 0.0;

     printf("[Feedback Diagnostics] Starting neighbor check for star %d\n", i);
 
     // First pass - count neighbors and calculate total kernel weight
     // Print star position for reference
     printf("[Feedback Diagnostics] Star %d position: [%.3f, %.3f, %.3f]\n", 
            i, in->Pos[0], in->Pos[1], in->Pos[2]);
     
     // Find closest gas particle and distance statistics
     double min_dist = 1e10;
     double dist_sum = 0;
     int closest_gas = -1;
     int gas_count = 0;
     int gas_within_2h = 0;
     int gas_within_5h = 0;
     int gas_within_10h = 0;
     
     // First pass - count neighbors and find closest
     for (int j = 0; j < Sp->NumPart; j++) {
         if (Sp->P[j].getType() != 0) continue;  // Skip non-gas particles
 
         gas_count++;
         
         double gasPos[3] = {
            intpos_to_kpc( Sp->P[j].IntPos[0] ),
            intpos_to_kpc( Sp->P[j].IntPos[1] ),
            intpos_to_kpc( Sp->P[j].IntPos[2] )
        };
        
        double dx[3] = {
            NEAREST_X(gasPos[0] - in->Pos[0]),
            NEAREST_Y(gasPos[1] - in->Pos[1]),
            NEAREST_Z(gasPos[2] - in->Pos[2])
        };

         double r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
         double r = sqrt(r2);
 
         // Track distance statistics
         dist_sum += r;
         
         // Count gas at different distances
         if (r < 2.0 * in->h) gas_within_2h++;
         if (r < 5.0 * in->h) gas_within_5h++;
         if (r < 10.0 * in->h) gas_within_10h++;
         
         // Find closest gas
         if (r2 < min_dist) {
             min_dist = r2;
             closest_gas = j;
         }
         
         // Check if within feedback radius
         if (r < in->h) {
             double w = 0.0;
             if (in->FeedbackType == FEEDBACK_SNII) {
                 w = kernel_weight_tophat(r, in->h);
             } else {
                 w = kernel_weight_cubic(r, in->h);
             }
 
             if (w > 0.0) {
                 neighbor_count++;
                 total_kernel_weight += w;
 
                 // Track maximum temperature
                 double temp = Sp->SphP[j].Utherm;
                 if (temp > max_gas_temp_before) max_gas_temp_before = temp;
             }
         }
     }
 
     // Report closest gas particle
     if (closest_gas >= 0) {
         double closest_dist = sqrt(min_dist);
         printf("[Feedback Diagnostics] Closest gas particle %d at distance %.3f (radius is %.3f)\n", 
                closest_gas, closest_dist, in->h);
         
         // If closest gas is too far, print its position for comparison
         if (closest_dist > in->h) {
             printf("[Feedback Diagnostics] Closest gas position: [%.3f, %.3f, %.3f]\n", 
                intpos_to_kpc( Sp->P[closest_gas].IntPos[0] ),
                intpos_to_kpc( Sp->P[closest_gas].IntPos[1] ),
                intpos_to_kpc( Sp->P[closest_gas].IntPos[2] ) );
         }
     }
     
     // Report distance statistics
     double mean_dist = (gas_count > 0) ? dist_sum / gas_count : 0;
     printf("[Feedback Diagnostics] Gas statistics: total=%d, mean_distance=%.3f\n", gas_count, mean_dist);
     printf("[Feedback Diagnostics] Gas within 2h(%.3f)=%d, 5h(%.3f)=%d, 10h(%.3f)=%d\n", 
            in->h*2, gas_within_2h, in->h*5, gas_within_5h, in->h*10, gas_within_10h);
     
     printf("[Feedback Diagnostics] Star %d (Type=%d): Found %d gas neighbors within radius %.3f\n", 
            i, in->FeedbackType, neighbor_count, in->h);
 
     if (neighbor_count == 0) {
         printf("[Feedback WARNING] Star %d has NO gas neighbors! No feedback will be applied.\n", i);
         printf("[Feedback WARNING] Consider increasing the feedback radius (currently %.3f).\n", in->h);
         
         // If there's gas in wider radius, suggest adjustment
         if (gas_within_5h > 0) {
             double suggested_radius = sqrt(min_dist) * 1.1; // 10% larger than closest gas
             printf("[Feedback SUGGESTION] Try increasing radius to at least %.3f\n", suggested_radius);
         }
         
         return;
     }
 
     // Check potential energy impact
     double gas_mass_total = 0.0;
     double avg_gas_temp = 0.0;
     int neighbors_sampled = 0;
 
     // Sample some neighbors to estimate energy impact
     for (int j = 0; j < Sp->NumPart && neighbors_sampled < 5; j++) {
         if (Sp->P[j].getType() != 0) continue;
 
         double gasPos[3] = {
            intpos_to_kpc( Sp->P[j].IntPos[0] ),
            intpos_to_kpc( Sp->P[j].IntPos[1] ),
            intpos_to_kpc( Sp->P[j].IntPos[2] )
        };

        double dx[3] = {
            NEAREST_X(gasPos[0] - in->Pos[0]),
            NEAREST_Y(gasPos[1] - in->Pos[1]),
            NEAREST_Z(gasPos[2] - in->Pos[2])
        };
         double r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
         double r = sqrt(r2);
 
         if (r < in->h) {
             double w = 0.0;
             if (in->FeedbackType == FEEDBACK_SNII) {
                 w = kernel_weight_tophat(r, in->h);
             } else {
                 w = kernel_weight_cubic(r, in->h);
             }
 
             if (w > 0.0) {
                 if (w > 1.0) w = 1.0; // prevent unusually high weights, probably wouldn't happen
                 double m = Sp->P[j].getMass();
                 double temp = Sp->SphP[j].Utherm;
                 double energy_before = temp * m;
                 double energy_to_add = in->Energy * w * erg_to_code;
                 double energy_after = energy_before + energy_to_add;
                 double temp_after = energy_after / m;
                 double energy_ratio = energy_to_add / energy_before;
 
                 printf("[Feedback Energy] Gas %d: Mass: %.3e Temp: %.3e → %.3e, Energy: %.3e → %.3e, Ratio: %.3e\n",
                        j, m, temp, temp_after, energy_before, energy_after, energy_ratio);
 
                 gas_mass_total += m;
                 avg_gas_temp += temp;
                 neighbors_sampled++;
             }
         }
     }
 
     if (neighbors_sampled > 0) {
         avg_gas_temp /= neighbors_sampled;
         printf("[Feedback Summary] Star %d: Avg Gas Temp: %.3e, Total Gas Mass: %.3e\n",
                i, avg_gas_temp, gas_mass_total);
         printf("[Feedback Summary] Available Energy: %.3e, Energy per Gas Mass: %.3e\n",
                in->Energy, in->Energy / gas_mass_total);
     }
 }
 
 /**
  * Apply feedback from a single star particle
  */
 void apply_feedback_to_star(int i, FeedbackWalk *fw, simparticles *Sp) {
     FeedbackInput in;
     FeedbackResult out;
 
     printf("[Feedback Debug] Entering apply_feedback_to_star for star %d, type=%d\n", 
         i, fw->feedback_type);
 
     // Copy star particle data to feedback input structure
     feedback_copy(i, &in, fw, Sp);
 
     // Add neighbor diagnostics before applying feedback
     debug_feedback_diagnostics(i, &in, fw, Sp);
 
     // Find neighbors and apply feedback - ONLY to gas particles
     // This is using direct particle loop, but could be replaced with tree-based search
     for (int j = 0; j < Sp->NumPart; j++) {
         if (Sp->P[j].getType() == 0) {  // Only process gas particles (Type 0)
             feedback_ngb(&in, &out, j, fw, Sp);
         }
     }
 }
 
 /**
  * Apply feedback for a specific type
  */
 void apply_feedback_treewalk(double current_time, int feedback_type, simparticles *Sp) {
     FeedbackWalk fw;
     fw.current_time = current_time;
     fw.feedback_type = feedback_type;
     fw.ev_label = "Feedback";
 
     // Loop through star particles only (could be parallelized)
     for (int i = 0; i < Sp->NumPart; i++) {
         // Only check star particles (Type 4)
         if (Sp->P[i].getType() == 4 && feedback_isactive(i, &fw, Sp)) {
             apply_feedback_to_star(i, &fw, Sp);
         }
     }
 }
 
 /**
  * Main feedback function called each timestep
  */
 void apply_stellar_feedback(double current_time, simparticles* Sp) {
 
 
 // -----TEMPORARY DIAGNOSTIC: Print TIME RESOLUTION-----
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
 
 
 
     // Reset diagnostic counters
     ThisStepEnergy_SNII = 0;
     ThisStepEnergy_SNIa = 0;
     ThisStepEnergy_AGB = 0;
     ThisStepMassReturned = 0;
     std::memset(ThisStepMetalsInjected, 0, sizeof(ThisStepMetalsInjected));
 
     // Apply each feedback type
     apply_feedback_treewalk(current_time, FEEDBACK_SNII, Sp);
     apply_feedback_treewalk(current_time, FEEDBACK_AGB, Sp);
     //apply_feedback_treewalk(current_time, FEEDBACK_SNIa, Sp);
 
     // Accumulate totals
     TotalEnergyInjected_SNII += ThisStepEnergy_SNII;
     TotalEnergyInjected_SNIa += ThisStepEnergy_SNIa;
     TotalEnergyInjected_AGB  += ThisStepEnergy_AGB;
     TotalMassReturned += ThisStepMassReturned;
     for (int k = 0; k < 4; k++)
         TotalMetalsInjected[k] += ThisStepMetalsInjected[k];
 
     // Print summary (on master process only)
     if (ThisTask == 0 & (ThisStepEnergy_SNII > 0 ||
         ThisStepEnergy_SNIa > 0 ||
         ThisStepEnergy_AGB > 0) ) {
         printf("[Feedback Timestep Summary] E_SNII=%.3e erg, E_SNIa=%.3e erg, E_AGB=%.3e erg\n",
                ThisStepEnergy_SNII, ThisStepEnergy_SNIa, ThisStepEnergy_AGB);
         printf("[Feedback Timestep Summary] Mass Returned=%.3e Msun\n", ThisStepMassReturned);
         printf("[Feedback Timestep Summary] Metals (Z=%.3e, C=%.3e, O=%.3e, Fe=%.3e) Msun\n",
                ThisStepMetalsInjected[0], ThisStepMetalsInjected[1], ThisStepMetalsInjected[2], ThisStepMetalsInjected[3]);
     }
 }

 #endif // FEEDBACK