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

// Feedback timescales - now in physical time rather than scale factor!
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
    printf("[Feedback Debug] feedback_isactive() - Checking star %d, type=%d, age=%e\n", i, Sp->P[i].getType(), fw->current_time - Sp->P[i].StellarAge);

    // Convert from scale factor to physical time
    double age_physical = scale_factor_to_physical_time(fw->current_time - Sp->P[i].StellarAge);

    // Check if this feedback type has already been applied
    if ((Sp->P[i].FeedbackFlag & fw->feedback_type) != 0) {
        return 0; // Already processed
    }

    // Different criteria for different feedback types
    if (fw->feedback_type == FEEDBACK_SNII) {

        printf("[Feedback Debug] Star %d is active for SNII feedback type %d\n", i, fw->feedback_type);
        // Type II SNe happen promptly after star formation
        return (age_physical > SNII_DELAY_TIME_PHYSICAL) ? 1 : 0;
    } 
    else if (fw->feedback_type == FEEDBACK_AGB) {
        printf("[Feedback Debug] Star %d is active for AGB feedback type %d\n", i, fw->feedback_type);
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
    out->Pos[0] = Sp->P[i].IntPos[0];
    out->Pos[1] = Sp->P[i].IntPos[1];
    out->Pos[2] = Sp->P[i].IntPos[2];
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

    // Before the neighbor loop
    printf("[Feedback Debug] feedback_ngb() -- Finding neighbors for star %d\n", j);

    double dx[3] = {
        NEAREST_X(Sp->P[j].IntPos[0] - in->Pos[0]),
        NEAREST_Y(Sp->P[j].IntPos[1] - in->Pos[1]),
        NEAREST_Z(Sp->P[j].IntPos[2] - in->Pos[2])
    };
    double r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
    double r = sqrt(r2);

    // Different kernel approaches for different feedback types
    double w = 0.0;
    if (in->FeedbackType == FEEDBACK_SNII) {
        // Type II SNe: Use a sharp top-hat kernel - more localized effect
        w = kernel_weight_tophat(r, in->h);
    } else {
        // Type Ia SNe and AGB winds: Use a smoother cubic spline kernel
        w = kernel_weight_cubic(r, in->h);
    }

    printf("[Feedback] Kernel weight for r=%.3f, h=%.3f: w=%.3e\n", r, in->h, w);

    if (w <= 0.0) return;

    // Calculate energy diagnostics before applying feedback
    double gas_mass = Sp->P[j].getMass();
    double old_thermal_energy = Sp->SphP[j].Entropy * gas_mass; // Approximate
    double feedback_energy = in->Energy * w;
    double energy_ratio = feedback_energy / old_thermal_energy;
    
    // Only print detailed diagnostics for significant contributions
    if (energy_ratio > 0.01) {
        printf("[Feedback Energy] Star%d->Gas%d: r=%.3f, OldE=%.3e, AddedE=%.3e, Ratio=%.3e\n",
               in->SourceIndex, j, r, old_thermal_energy, feedback_energy, energy_ratio);
    }

    // Apply energy - different approaches for different feedback types
    if (in->FeedbackType == FEEDBACK_SNII) {
        // SNII: Direct entropy injection plus velocity kick along radial direction
        Sp->SphP[j].Entropy += in->Energy * w;
        
        // Add a radial velocity kick - more important for SNII
        if (r > 0) {
            double kick_strength = 0.5 * WIND_VELOCITY * w;
            printf("[Feedback] Applying SN II feedback: %.4e\n", kick_strength);
            Sp->P[j].Vel[0] += kick_strength * dx[0]/r;
            Sp->P[j].Vel[1] += kick_strength * dx[1]/r;
            Sp->P[j].Vel[2] += kick_strength * dx[2]/r;
        }
    } 
    else if (in->FeedbackType == FEEDBACK_SNIa) {
        // SNIa: More thermal energy, less kinetic
        Sp->SphP[j].Entropy += in->Energy * w;
        // Small random velocity perturbation
        if (r > 0) {
            double kick_strength = 0.1 * WIND_VELOCITY * w;
            printf("[Feedback] Applying SN Ia feedback: %.4e\n", kick_strength);
            Sp->P[j].Vel[0] += kick_strength * dx[0]/r;
            Sp->P[j].Vel[1] += kick_strength * dx[1]/r;
            Sp->P[j].Vel[2] += kick_strength * dx[2]/r;
        }
    }
    else if (in->FeedbackType == FEEDBACK_AGB) {
        // AGB: Gentler feedback
        Sp->SphP[j].Entropy += in->Energy * w;
        // Very mild velocity perturbation
        if (r > 0) {
            double kick_strength = 0.05 * WIND_VELOCITY * w;
            printf("[Feedback] Applying AGB feedback: %.4e\n", kick_strength);
            Sp->P[j].Vel[0] += kick_strength * dx[0]/r;
            Sp->P[j].Vel[1] += kick_strength * dx[1]/r;
            Sp->P[j].Vel[2] += kick_strength * dx[2]/r;
        }
    }

    // Add mass from stellar winds/ejecta (except for SNIa which don't return stellar mass)
    double old_mass = Sp->P[j].getMass();
    Sp->P[j].setMass(old_mass + in->MassReturn * w);

    // Add metals - but account for mass dilution
    if (in->MassReturn * w > 0) {
        // For significant mass return, we need to properly mix old and new metals
        double old_fraction = old_mass / (old_mass + in->MassReturn * w);
        double new_fraction = (in->MassReturn * w) / (old_mass + in->MassReturn * w);
        
        for (int k = 0; k < 4; k++) {
            double old_metal = Sp->SphP[j].Metals[k];
            // New metal is the yield divided by returned mass to get concentration
            double new_metal = in->Yield[k] / in->MassReturn;
            // Mix old and new
            Sp->SphP[j].Metals[k] = old_metal * old_fraction + new_metal * new_fraction;
        }
    } else {
        // For SNIa with no mass return, we just add the metals directly
        // This slightly increases metal concentration
        for (int k = 0; k < 4; k++) {
            Sp->SphP[j].Metals[k] += in->Yield[k] * w / old_mass;
        }
    }
}

/**
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
    for (int j = 0; j < Sp->NumPart; j++) {
        if (Sp->P[j].getType() != 0) continue;  // Skip non-gas particles
        
        double dx[3] = {
            NEAREST_X(Sp->P[j].IntPos[0] - in->Pos[0]),
            NEAREST_Y(Sp->P[j].IntPos[1] - in->Pos[1]),
            NEAREST_Z(Sp->P[j].IntPos[2] - in->Pos[2])
        };
        double r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
        double r = sqrt(r2);
        
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
                double temp = Sp->SphP[j].Entropy;  // Using entropy as proxy for temperature
                if (temp > max_gas_temp_before) max_gas_temp_before = temp;
            }
        }
    }
    
    printf("[Feedback Diagnostics] Star %d (Type=%d): Found %d gas neighbors within radius %.3f\n", 
           i, in->FeedbackType, neighbor_count, in->h);
    
    if (neighbor_count == 0) {
        printf("[Feedback WARNING] Star %d has NO gas neighbors! No feedback will be applied.\n", i);
        return;
    }
    
    // Check potential energy impact
    double gas_mass_total = 0.0;
    double avg_gas_temp = 0.0;
    int neighbors_sampled = 0;
    
    // Sample some neighbors to estimate energy impact
    for (int j = 0; j < Sp->NumPart && neighbors_sampled < 5; j++) {
        if (Sp->P[j].getType() != 0) continue;
        
        double dx[3] = {
            NEAREST_X(Sp->P[j].IntPos[0] - in->Pos[0]),
            NEAREST_Y(Sp->P[j].IntPos[1] - in->Pos[1]),
            NEAREST_Z(Sp->P[j].IntPos[2] - in->Pos[2])
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
                double m = Sp->P[j].getMass();
                double temp = Sp->SphP[j].Entropy;
                double energy_before = temp * m;  // Approximate thermal energy
                double energy_to_add = in->Energy * w;
                double energy_after = energy_before + energy_to_add;
                double temp_after = energy_after / m;
                double energy_ratio = energy_to_add / energy_before;
                
                printf("[Feedback Energy] Gas %d: Temp: %.3e → %.3e, Energy: %.3e → %.3e, Ratio: %.3e\n",
                       j, temp, temp_after, energy_before, energy_after, energy_ratio);
                
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
            printf("[Feedback Diagnostic] Timestep in physical years: %.2e\n", physical_timestep);
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