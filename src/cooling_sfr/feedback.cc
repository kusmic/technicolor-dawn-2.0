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
    double u = r / h;
    if (u < 0.5)
        return (8.0 / M_PI) * (1 - 6 * u * u + 6 * u * u * u);
    else if (u < 1.0)
        return (8.0 / M_PI) * 2 * pow(1 - u, 3);
    else
        return 0.0;
}

// SNII feedback
        if (age > SNII_DELAY_TIME && !(Sp->P[i].FeedbackFlag & FEEDBACK_SNII)) {
            double e_sn = SNII_ENERGY_PER_MASS * m_star;
            double m_return = MASS_RETURN_SNII * m_star;
            Yields y = get_SNII_yields(m_return);

            double h = 0.5;  // Fixed search radius (can be replaced with star's smoothing length)
            // Perform neighbor search around the star particle
            MyDouble pos[3] = {Sp->P[i].Pos[0], Sp->P[i].Pos[1], Sp->P[i].Pos[2]};
            int num_ngb;
            int *ngblist = ngb_treefind_variable(pos, h, &num_ngb);

            double total_w = 0.0;
            double weights[num_ngb];
            int valid_ngbs[num_ngb];
            int count = 0;

            // Loop through neighbors and compute kernel-weighted distances
            for (int n = 0; n < num_ngb; n++) {
                int j = ngblist[n];
                if (Sp->P[j].get_type() != 0) continue;

                double dx[3] = {
                    Sp->P[j].Pos[0] - pos[0],
                    Sp->P[j].Pos[1] - pos[1],
                    Sp->P[j].Pos[2] - pos[2]
                };
                double r = sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);
                double w = kernel_weight(r, h);
                if (w > 0) {
                    weights[count] = w;
                    valid_ngbs[count] = j;
                    total_w += w;
                    count++;
                }
            }

            // Normalize weights and apply feedback to each valid gas neighbor
            if (total_w > 0) {
                for (int n = 0; n < count; n++) {
                    int j = valid_ngbs[n];
                    double w = weights[n] / total_w;
                    Sp->SphP[j].Energy += e_sn * w;
                    Sp->P[j].Mass += m_return * w;
                    Sp->P[j].Vel[0] += WIND_VELOCITY * w;
                    Sp->SphP[j].Metals[0] += y.Z * w;
                    Sp->SphP[j].Metals[1] += y.C * w;
                    Sp->SphP[j].Metals[2] += y.O * w;
                    Sp->SphP[j].Metals[3] += y.Fe * w;
                }
                ThisStepEnergy_SNII += e_sn;
                ThisStepMassReturned += m_return;
                ThisStepMetalsInjected[0] += y.Z;
                ThisStepMetalsInjected[1] += y.C;
                ThisStepMetalsInjected[2] += y.O;
                ThisStepMetalsInjected[3] += y.Fe;
                Sp->P[i].FeedbackFlag |= FEEDBACK_SNII;
            }
        }
            ThisStepEnergy_SNII += e_sn;
            ThisStepMassReturned += m_return;
            ThisStepMetalsInjected[0] += y.Z;
            ThisStepMetalsInjected[1] += y.C;
            ThisStepMetalsInjected[2] += y.O;
            ThisStepMetalsInjected[3] += y.Fe;
            Sp->P[i].FeedbackFlag |= FEEDBACK_SNII;
        }

        // AGB feedback
        if (age > SNII_DELAY_TIME && age < AGB_END_TIME && !(Sp->P[i].FeedbackFlag & FEEDBACK_AGB)) {
            double e_agb = AGB_ENERGY_PER_MASS * m_star;
            double m_return = MASS_RETURN_AGB * m_star;
            Yields y = get_AGB_yields(m_return);

            double h = 0.5;
            MyDouble pos[3] = {Sp->P[i].Pos[0], Sp->P[i].Pos[1], Sp->P[i].Pos[2]};
            int num_ngb;
            int *ngblist = ngb_treefind_variable(pos, h, &num_ngb);

            double total_w = 0.0;
            double weights[num_ngb];
            int valid_ngbs[num_ngb];
            int count = 0;

            for (int n = 0; n < num_ngb; n++) {
                int j = ngblist[n];
                if (Sp->P[j].get_type() != 0) continue;

                double dx[3] = {
                    Sp->P[j].Pos[0] - pos[0],
                    Sp->P[j].Pos[1] - pos[1],
                    Sp->P[j].Pos[2] - pos[2]
                };
                double r = sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);
                double w = kernel_weight(r, h);
                if (w > 0) {
                    weights[count] = w;
                    valid_ngbs[count] = j;
                    total_w += w;
                    count++;
                }
            }

            if (total_w > 0) {
                for (int n = 0; n < count; n++) {
                    int j = valid_ngbs[n];
                    double w = weights[n] / total_w;
                    Sp->SphP[j].Energy += e_agb * w;
                    Sp->P[j].Mass += m_return * w;
                    Sp->SphP[j].Metals[0] += y.Z * w;
                    Sp->SphP[j].Metals[1] += y.C * w;
                    Sp->SphP[j].Metals[2] += y.O * w;
                    Sp->SphP[j].Metals[3] += y.Fe * w;
                }
                ThisStepEnergy_AGB += e_agb;
                ThisStepMassReturned += m_return;
                ThisStepMetalsInjected[0] += y.Z;
                ThisStepMetalsInjected[1] += y.C;
                ThisStepMetalsInjected[2] += y.O;
                ThisStepMetalsInjected[3] += y.Fe;
                Sp->P[i].FeedbackFlag |= FEEDBACK_AGB;
            }
        }
        }

        // SNIa feedback
        if (age > SNIa_DELAY_TIME && !(Sp->P[i].FeedbackFlag & FEEDBACK_SNIa)) {
            double n_snia = m_star * SNIa_RATE_PER_MASS;
            double e_snia = n_snia * SNIa_ENERGY_PER_EVENT;
            Yields y = get_SNIa_yields(n_snia);

            double h = 0.5;
            MyDouble pos[3] = {Sp->P[i].Pos[0], Sp->P[i].Pos[1], Sp->P[i].Pos[2]};
            int num_ngb;
            int *ngblist = ngb_treefind_variable(pos, h, &num_ngb);

            double total_w = 0.0;
            double weights[num_ngb];
            int valid_ngbs[num_ngb];
            int count = 0;

            for (int n = 0; n < num_ngb; n++) {
                int j = ngblist[n];
                if (Sp->P[j].get_type() != 0) continue;

                double dx[3] = {
                    Sp->P[j].Pos[0] - pos[0],
                    Sp->P[j].Pos[1] - pos[1],
                    Sp->P[j].Pos[2] - pos[2]
                };
                double r = sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);
                double w = kernel_weight(r, h);
                if (w > 0) {
                    weights[count] = w;
                    valid_ngbs[count] = j;
                    total_w += w;
                    count++;
                }
            }

            if (total_w > 0) {
                for (int n = 0; n < count; n++) {
                    int j = valid_ngbs[n];
                    double w = weights[n] / total_w;
                    Sp->SphP[j].Energy += e_snia * w;
                    Sp->SphP[j].Metals[0] += y.Z * w;
                    Sp->SphP[j].Metals[1] += y.C * w;
                    Sp->SphP[j].Metals[2] += y.O * w;
                    Sp->SphP[j].Metals[3] += y.Fe * w;
                }
                ThisStepEnergy_SNIa += e_snia;
                ThisStepMetalsInjected[0] += y.Z;
                ThisStepMetalsInjected[1] += y.C;
                ThisStepMetalsInjected[2] += y.O;
                ThisStepMetalsInjected[3] += y.Fe;
                Sp->P[i].FeedbackFlag |= FEEDBACK_SNIa;
            }
        }
        }

        // Mark this star as having completed all feedback phases
        if ((Sp->P[i].FeedbackFlag & (FEEDBACK_SNII | FEEDBACK_AGB | FEEDBACK_SNIa)) == (FEEDBACK_SNII | FEEDBACK_AGB | FEEDBACK_SNIa))
            Sp->P[i].FeedbackFlag = FEEDBACK_SNII | FEEDBACK_AGB | FEEDBACK_SNIa;
    }

    TotalEnergyInjected_SNII += ThisStepEnergy_SNII;
    TotalEnergyInjected_SNIa += ThisStepEnergy_SNIa;
    TotalEnergyInjected_AGB  += ThisStepEnergy_AGB;
    TotalMassReturned += ThisStepMassReturned;
    for (int k = 0; k < 4; k++)
        TotalMetalsInjected[k] += ThisStepMetalsInjected[k];

    if(ThisTask == 0) {
        printf("[Feedback Timestep Summary] E_SNII=%.3e erg, E_SNIa=%.3e erg, E_AGB=%.3e erg\n",
               ThisStepEnergy_SNII, ThisStepEnergy_SNIa, ThisStepEnergy_AGB);
        printf("[Feedback Timestep Summary] Mass Returned=%.3e Msun\n", ThisStepMassReturned);
        printf("[Feedback Timestep Summary] Metals (Z=%.3e, C=%.3e, O=%.3e, Fe=%.3e) Msun\n",
               ThisStepMetalsInjected[0], ThisStepMetalsInjected[1], ThisStepMetalsInjected[2], ThisStepMetalsInjected[3]);
    }
}
