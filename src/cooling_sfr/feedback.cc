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

    for (int i = 0; i < Sp->NumPart; i++) {
        if (Sp->P[i].getType() != 4) continue;

        double age = current_time - Sp->P[i].BirthTime;
        double m_star = Sp->P[i].getMass();
        int flag = Sp->P[i].FeedbackFlag;

        if ((flag & (FEEDBACK_SNII | FEEDBACK_AGB | FEEDBACK_SNIa)) == (FEEDBACK_SNII | FEEDBACK_AGB | FEEDBACK_SNIa))
            continue;

        auto feedback = [&](double energy, double mass_returned, Yields y, int feedback_type) {
            double h = 0.5;
            MyDouble pos[3] = {Sp->P[i].IntPos[0], Sp->P[i].IntPos[1], Sp->P[i].IntPos[2]};
            int num_ngb;
            int *ngblist = ngb_treefind_variable(pos, h, &num_ngb);

            double total_w = 0.0;
            std::vector<double> weights;
            std::vector<int> indices;

            for (int n = 0; n < num_ngb; n++) {
                int j = ngblist[n];
                if (Sp->P[j].getType() != 0) continue;
                double dx[3] = {Sp->P[j].IntPos[0] - pos[0], Sp->P[j].IntPos[1] - pos[1], Sp->P[j].IntPos[2] - pos[2]};
                double r = std::sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);
                double w = kernel_weight(r, h);
                if (w > 0) {
                    weights.push_back(w);
                    indices.push_back(j);
                    total_w += w;
                }
            }

            if (total_w > 0) {
                for (size_t n = 0; n < weights.size(); n++) {
                    int j = indices[n];
                    double w = weights[n] / total_w;
                    Sp->SphP[j].Energy += energy * w;
                    Sp->P[j].getMass() += mass_returned * w;
                    Sp->P[j].Vel[0] += WIND_VELOCITY * w;
                    Sp->SphP[j].Metals[0] += y.Z * w;
                    Sp->SphP[j].Metals[1] += y.C * w;
                    Sp->SphP[j].Metals[2] += y.O * w;
                    Sp->SphP[j].Metals[3] += y.Fe * w;
                }

                if (feedback_type == FEEDBACK_SNII) ThisStepEnergy_SNII += energy;
                if (feedback_type == FEEDBACK_AGB)  ThisStepEnergy_AGB  += energy;
                if (feedback_type == FEEDBACK_SNIa) ThisStepEnergy_SNIa += energy;

                ThisStepMassReturned += mass_returned;
                ThisStepMetalsInjected[0] += y.Z;
                ThisStepMetalsInjected[1] += y.C;
                ThisStepMetalsInjected[2] += y.O;
                ThisStepMetalsInjected[3] += y.Fe;

                Sp->P[i].FeedbackFlag |= feedback_type;
            }
        };

        if (age > SNII_DELAY_TIME && !(flag & FEEDBACK_SNII))
            feedback(SNII_ENERGY_PER_MASS * m_star, MASS_RETURN_SNII * m_star, get_SNII_yields(MASS_RETURN_SNII * m_star), FEEDBACK_SNII);

        if (age > SNII_DELAY_TIME && age < AGB_END_TIME && !(flag & FEEDBACK_AGB))
            feedback(AGB_ENERGY_PER_MASS * m_star, MASS_RETURN_AGB * m_star, get_AGB_yields(MASS_RETURN_AGB * m_star), FEEDBACK_AGB);

        if (age > SNIa_DELAY_TIME && !(flag & FEEDBACK_SNIa)) {
            double n_snia = m_star * SNIa_RATE_PER_MASS;
            feedback(n_snia * SNIa_ENERGY_PER_EVENT, 0, get_SNIa_yields(n_snia), FEEDBACK_SNIa);
        }
    }

    TotalEnergyInjected_SNII += ThisStepEnergy_SNII;
    TotalEnergyInjected_SNIa += ThisStepEnergy_SNIa;
    TotalEnergyInjected_AGB  += ThisStepEnergy_AGB;
    TotalMassReturned += ThisStepMassReturned;
    for (int k = 0; k < 4; k++)
        TotalMetalsInjected[k] += ThisStepMetalsInjected[k];

    if (ThisTask == 0) {
        printf("[Feedback Timestep Summary] E_SNII=%.3e erg, E_SNIa=%.3e erg, E_AGB=%.3e erg\n",
               ThisStepEnergy_SNII, ThisStepEnergy_SNIa, ThisStepEnergy_AGB);
        printf("[Feedback Timestep Summary] Mass Returned=%.3e Msun\n", ThisStepMassReturned);
        printf("[Feedback Timestep Summary] Metals (Z=%.3e, C=%.3e, O=%.3e, Fe=%.3e) Msun\n",
               ThisStepMetalsInjected[0], ThisStepMetalsInjected[1], ThisStepMetalsInjected[2], ThisStepMetalsInjected[3]);
    }
}

#endif // FEEDBACK