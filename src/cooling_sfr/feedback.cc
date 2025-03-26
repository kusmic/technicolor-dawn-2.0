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

#include "../cooling_sfr/cooling.h"
#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/mymalloc.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../system/system.h"
#include "../time_integration/timestep.h"

// Feedback type bitmask flags
#define FEEDBACK_SNII  1  // Supernova Type II
#define FEEDBACK_AGB   2  // AGB wind feedback
#define FEEDBACK_SNIa  4  // Supernova Type Ia

// Physical constants for feedback energy and timing
const double SNII_ENERGY_PER_MASS = 1.0e49;     // erg/Msun, energy per unit mass for SNII
const double SNIa_ENERGY_PER_EVENT = 1.0e51;    // erg, energy per SNIa event
const double AGB_ENERGY_PER_MASS = 1.0e47;      // erg/Msun, lower energy per mass for AGB
const double SNIa_RATE_PER_MASS = 5e-4;         // SNIa events per Msun of stars
const double SNIa_DELAY_TIME = 1.5e9;           // Delay before SNIa starts (yr)
const double SNII_DELAY_TIME = 1.0e7;           // Delay before SNII starts (yr)
const double AGB_END_TIME = 1.0e10;             // AGB wind ends around 10 Gyr
const double MASS_RETURN_SNII = 0.10;           // Fractional mass return from SNII
const double MASS_RETURN_AGB = 0.30;            // Fractional mass return from AGB
const double WIND_VELOCITY = 500.0;             // km/s, simple velocity kick from feedback

// Feedback diagnostics (cumulative)
static double TotalEnergyInjected_SNII = 0;
static double TotalEnergyInjected_SNIa = 0;
static double TotalEnergyInjected_AGB = 0;
static double TotalMassReturned = 0;
static double TotalMetalsInjected[4] = {0};  // [Z, C, O, Fe]

// Per-timestep diagnostics (for logging/output)
double ThisStepEnergy_SNII = 0;
double ThisStepEnergy_SNIa = 0;
double ThisStepEnergy_AGB = 0;
double ThisStepMassReturned = 0;
double ThisStepMetalsInjected[4] = {0};

// Simple yield model (mass fraction per Msun returned)
struct Yields {
    double Z;   // total metals
    double C;
    double O;
    double Fe;
};

Yields get_SNII_yields(double m_star) {
    return {0.02 * m_star, 0.004 * m_star, 0.008 * m_star, 0.001 * m_star};
}

Yields get_AGB_yields(double m_star) {
    return {0.01 * m_star, 0.005 * m_star, 0.002 * m_star, 0.0005 * m_star};
}

Yields get_SNIa_yields(double n_snia) {
    return {0.01 * n_snia, 0.0, 0.0, 0.007 * n_snia};
}

void apply_stellar_feedback(double current_time, simparticles *Sp) {
    // Reset per-timestep counters
    ThisStepEnergy_SNII = 0;
    ThisStepEnergy_SNIa = 0;
    ThisStepEnergy_AGB = 0;
    ThisStepMassReturned = 0;
    std::memset(ThisStepMetalsInjected, 0, sizeof(ThisStepMetalsInjected));

    for (int i = 0; i < Sp->NumPart; i++) {
        if (P[i].Type != 4) continue;  // Only apply feedback to star particles

        double age = current_time - P[i].BirthTime;
        double m_star = P[i].Mass;

        // Skip particles that have completed all feedback channels
        if (P[i].FeedbackFlag == (FEEDBACK_SNII | FEEDBACK_AGB | FEEDBACK_SNIa))
            continue;

        int *ngblist = P[i].NeighborList;
        int num_ngb = P[i].NumNeighbors;
        double total_w = 0.0;
        for (int n = 0; n < num_ngb; n++) {
            int j = ngblist[n];
            if (P[j].Type != 0) continue;
            total_w += P[i].Kernel[n];
        }
        if (total_w <= 0.0) continue;

        // SNII feedback
        if (age > SNII_DELAY_TIME && !(P[i].FeedbackFlag & FEEDBACK_SNII)) {
            double e_sn = SNII_ENERGY_PER_MASS * m_star;
            double m_return = MASS_RETURN_SNII * m_star;
            Yields y = get_SNII_yields(m_return);
            for (int n = 0; n < num_ngb; n++) {
                int j = ngblist[n];
                if (P[j].Type != 0) continue;
                double w = P[i].Kernel[n] / total_w;
                SphP[j].Energy += e_sn * w;
                P[j].Mass += m_return * w;
                P[j].Vel[0] += WIND_VELOCITY * w;
                SphP[j].Metals[0] += y.Z * w;
                SphP[j].Metals[1] += y.C * w;
                SphP[j].Metals[2] += y.O * w;
                SphP[j].Metals[3] += y.Fe * w;
            }
            ThisStepEnergy_SNII += e_sn;
            ThisStepMassReturned += m_return;
            ThisStepMetalsInjected[0] += y.Z;
            ThisStepMetalsInjected[1] += y.C;
            ThisStepMetalsInjected[2] += y.O;
            ThisStepMetalsInjected[3] += y.Fe;
            P[i].FeedbackFlag |= FEEDBACK_SNII;
        }

        // AGB feedback
        if (age > SNII_DELAY_TIME && age < AGB_END_TIME && !(P[i].FeedbackFlag & FEEDBACK_AGB)) {
            double e_agb = AGB_ENERGY_PER_MASS * m_star;
            double m_return = MASS_RETURN_AGB * m_star;
            Yields y = get_AGB_yields(m_return);
            for (int n = 0; n < num_ngb; n++) {
                int j = ngblist[n];
                if (P[j].Type != 0) continue;
                double w = P[i].Kernel[n] / total_w;
                SphP[j].Energy += e_agb * w;
                P[j].Mass += m_return * w;
                SphP[j].Metals[0] += y.Z * w;
                SphP[j].Metals[1] += y.C * w;
                SphP[j].Metals[2] += y.O * w;
                SphP[j].Metals[3] += y.Fe * w;
            }
            ThisStepEnergy_AGB += e_agb;
            ThisStepMassReturned += m_return;
            ThisStepMetalsInjected[0] += y.Z;
            ThisStepMetalsInjected[1] += y.C;
            ThisStepMetalsInjected[2] += y.O;
            ThisStepMetalsInjected[3] += y.Fe;
            P[i].FeedbackFlag |= FEEDBACK_AGB;
        }

        // SNIa feedback
        if (age > SNIa_DELAY_TIME && !(P[i].FeedbackFlag & FEEDBACK_SNIa)) {
            double n_snia = m_star * SNIa_RATE_PER_MASS;
            double e_snia = n_snia * SNIa_ENERGY_PER_EVENT;
            Yields y = get_SNIa_yields(n_snia);
            for (int n = 0; n < num_ngb; n++) {
                int j = ngblist[n];
                if (P[j].Type != 0) continue;
                double w = P[i].Kernel[n] / total_w;
                SphP[j].Energy += e_snia * w;
                SphP[j].Metals[0] += y.Z * w;
                SphP[j].Metals[1] += y.C * w;
                SphP[j].Metals[2] += y.O * w;
                SphP[j].Metals[3] += y.Fe * w;
            }
            ThisStepEnergy_SNIa += e_snia;
            ThisStepMetalsInjected[0] += y.Z;
            ThisStepMetalsInjected[1] += y.C;
            ThisStepMetalsInjected[2] += y.O;
            ThisStepMetalsInjected[3] += y.Fe;
            P[i].FeedbackFlag |= FEEDBACK_SNIa;
        }

        // If all types of feedback are complete, set final flag
        if ((P[i].FeedbackFlag & (FEEDBACK_SNII | FEEDBACK_AGB | FEEDBACK_SNIa)) == (FEEDBACK_SNII | FEEDBACK_AGB | FEEDBACK_SNIa))
            P[i].FeedbackFlag = FEEDBACK_SNII | FEEDBACK_AGB | FEEDBACK_SNIa;
    }

    // Accumulate timestep values into cumulative totals
    TotalEnergyInjected_SNII += ThisStepEnergy_SNII;
    TotalEnergyInjected_SNIa += ThisStepEnergy_SNIa;
    TotalEnergyInjected_AGB  += ThisStepEnergy_AGB;
    TotalMassReturned += ThisStepMassReturned;
    for (int k = 0; k < 4; k++)
        TotalMetalsInjected[k] += ThisStepMetalsInjected[k];

    // Output per-timestep summary
    if(ThisTask == 0) {
        printf("[Feedback Timestep Summary] E_SNII=%.3e erg, E_SNIa=%.3e erg, E_AGB=%.3e erg\n",
               ThisStepEnergy_SNII, ThisStepEnergy_SNIa, ThisStepEnergy_AGB);
        printf("[Feedback Timestep Summary] Mass Returned=%.3e Msun\n", ThisStepMassReturned);
        printf("[Feedback Timestep Summary] Metals (Z=%.3e, C=%.3e, O=%.3e, Fe=%.3e) Msun\n",
               ThisStepMetalsInjected[0], ThisStepMetalsInjected[1], ThisStepMetalsInjected[2], ThisStepMetalsInjected[3]);
    }
}

#endif // FEEDBACK