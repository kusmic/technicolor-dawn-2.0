#ifndef FEEDBACK_TREEWALK_H
#define FEEDBACK_TREEWALK_H

#include "../data/simparticles.h"  // or whatever defines `simparticles`


// ------------------------
// Global feedback tracking
// ------------------------

// These variables track feedback quantities applied during the **current timestep**:

// Energy injected by Type II Supernovae (SNII) during the current timestep (in ergs)
extern double ThisStepEnergy_SNII;

// Energy injected by Type Ia Supernovae (SNIa) during the current timestep (in ergs)
extern double ThisStepEnergy_SNIa;

// Energy injected by Asymptotic Giant Branch (AGB) stars during the current timestep (in ergs)
extern double ThisStepEnergy_AGB;

// Total stellar mass returned to the gas reservoir during the current timestep (in solar masses)
extern double ThisStepMassReturned;

// Array storing the total mass of individual metal species injected during this timestep.
// Indexing convention (example): [0] = C, [1] = O, [2] = Fe, [3] = other.
extern double ThisStepMetalsInjected[4];


// These variables accumulate total feedback quantities **over the full simulation runtime**:

// Cumulative energy injected by SNII events
extern double TotalEnergyInjected_SNII;

// Cumulative energy injected by SNIa events
extern double TotalEnergyInjected_SNIa;

// Cumulative energy injected by AGB winds
extern double TotalEnergyInjected_AGB;

// Total stellar mass returned to the ISM over the full simulation
extern double TotalMassReturned;

// Cumulative mass of individual metal species injected across the full simulation
extern double TotalMetalsInjected[4];

// Feedback types
#define FEEDBACK_SNII 1
#define FEEDBACK_SNIa 2
#define FEEDBACK_AGB  3

// ------------------------
// Function Declarations
// ------------------------

// Main treewalk-driven feedback function
void run_feedback(double current_time, int feedback_type, simparticles *Sp);

// Core feedback driver:
// Applies stellar feedback at a given simulation time via a treewalk.
// The `feedback_type` parameter is a bitmask (e.g. SNII, AGB, or SNIa) to determine which phase to process.
void apply_feedback_treewalk(double current_time, int feedback_type);

// High-level interface:
// Invokes the appropriate feedback treewalk(s) for a given time and the full particle data.
void apply_stellar_feedback(double current_time, struct simparticles* Sp);

#endif
