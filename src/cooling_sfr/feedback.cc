// feedback.cc – Stellar feedback module for Gadget-4
// Includes SNII, SNIa, AGB winds, mass return, metal injection (elemental), logging

#include "gadgetconfig.h"

#ifdef FEEDBACK

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "../cooling_sfr/feedback_treewalk.h"
#include "../cooling_sfr/cooling.h"
#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/mymalloc.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../system/system.h"
#include "../time_integration/timestep.h"
#include "../ngbtree/ngbtree.h"

// Feedback type bitmask flags
#define FEEDBACK_SNII  1
#define FEEDBACK_AGB   2
#define FEEDBACK_SNIa  4

// Feedback energy/mass return constants
const double SNII_ENERGY_PER_MASS = 1.0e49;     // erg / Msun
const double SNIa_ENERGY_PER_EVENT = 1.0e51;    // erg per event
const double AGB_ENERGY_PER_MASS = 1.0e47;      // erg / Msun

const double SNIa_RATE_PER_MASS = 5e-4;         // events per Msun formed
const double SNIa_DELAY_TIME = 1.5e9;           // yr
const double SNII_DELAY_TIME = 1.0e7;           // yr
const double AGB_END_TIME = 1.0e10;             // yr

const double MASS_RETURN_SNII = 0.10;           // fraction of m_star
const double MASS_RETURN_AGB = 0.30;            // fraction of m_star

const double WIND_VELOCITY = 500.0;             // km/s

// Per-timestep diagnostics
double ThisStepEnergy_SNII = 0;
double ThisStepEnergy_SNIa = 0;
double ThisStepEnergy_AGB = 0;
double ThisStepMassReturned = 0;
double ThisStepMetalsInjected[4] = {0};  // Z, C, O, Fe

// Cumulative totals
double TotalEnergyInjected_SNII = 0;
double TotalEnergyInjected_SNIa = 0;
double TotalEnergyInjected_AGB = 0;
double TotalMassReturned = 0;
double TotalMetalsInjected[4] = {0};     // Z, C, O, Fe

// Yield struct and helpers
struct Yields {
    double Z;
    double C;
    double O;
    double Fe;
};

Yields get_SNII_yields(double m) {
    return {0.02 * m, 0.004 * m, 0.008 * m, 0.001 * m};
}

Yields get_AGB_yields(double m) {
    return {0.01 * m, 0.005 * m, 0.002 * m, 0.0005 * m};
}

Yields get_SNIa_yields(double n_events) {
    return {0.01 * n_events, 0.0, 0.0, 0.007 * n_events};
}

// Kernel weight function
// Cubic spline kernel normalized in 3D, used to weight feedback among gas neighbors

double kernel_weight(double r, double h) {
    double u = r / h;
    if (u < 0.5)
        return (8.0 / M_PI) * (1 - 6 * u * u + 6 * u * u * u);
    else if (u < 1.0)
        return (8.0 / M_PI) * 2 * pow(1 - u, 3);
    else
        return 0.0;
}

// Main feedback driver
void apply_stellar_feedback(double current_time, struct simparticles* Sp) {
    ThisStepEnergy_SNII = 0;
    ThisStepEnergy_SNIa = 0;
    ThisStepEnergy_AGB = 0;
    ThisStepMassReturned = 0;
    std::memset(ThisStepMetalsInjected, 0, sizeof(ThisStepMetalsInjected));

    apply_feedback_treewalk(current_time, FEEDBACK_SNII);
    apply_feedback_treewalk(current_time, FEEDBACK_AGB);
    apply_feedback_treewalk(current_time, FEEDBACK_SNIa);

    if (ThisTask == 0) {
        printf("[Feedback Timestep Summary] E_SNII=%.3e erg, E_SNIa=%.3e erg, E_AGB=%.3e erg\n",
               ThisStepEnergy_SNII, ThisStepEnergy_SNIa, ThisStepEnergy_AGB);
        printf("[Feedback Timestep Summary] Mass Returned=%.3e Msun\n", ThisStepMassReturned);
        printf("[Feedback Timestep Summary] Metals (Z=%.3e, C=%.3e, O=%.3e, Fe=%.3e) Msun\n",
               ThisStepMetalsInjected[0], ThisStepMetalsInjected[1], ThisStepMetalsInjected[2], ThisStepMetalsInjected[3]);
    }
}

#endif // FEEDBACK