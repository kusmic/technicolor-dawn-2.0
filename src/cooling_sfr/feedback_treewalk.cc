/*
GENERAL TREEWALK FEEDBACK ALGORITHM
Loop over star particles (Type == 4)
Check eligibility (age, already processed, etc.)
Export star properties: position, energy, yields, etc.
Search for nearby gas particles within fixed radius h
Apply feedback kernel:
Energy → added to SphP[j].Energy
Mass → added to P[j].Mass
Metals → added to SphP[j].Metals[k]
Accumulate diagnostics for later logging.
*/

#include "gadgetconfig.h"

#ifdef FEEDBACK

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
#define NEAREST_X(x) NEAREST(x, BoxSize)
#define NEAREST_Y(x) NEAREST(x, BoxSize)
#define NEAREST_Z(x) NEAREST(x, BoxSize)

// Local cubic spline kernel approximation
inline double kernel_weight_cubic(double r, double h) {
    double u = r / h;
    if (u < 0.5)
        return (8.0 / (M_PI * h * h * h)) * (1 - 6 * u * u + 6 * u * u * u);
    else if (u < 1.0)
        return (8.0 / (M_PI * h * h * h)) * (2.0 * pow(1.0 - u, 3));
    else
        return 0.0;
}

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
double ThisStepMetalsInjected[4] = {0};

// Cumulative totals
double TotalEnergyInjected_SNII = 0;
double TotalEnergyInjected_SNIa = 0;
double TotalEnergyInjected_AGB = 0;
double TotalMassReturned = 0;
double TotalMetalsInjected[4] = {0};

struct FeedbackInput {
    MyDouble Pos[3];
    MyFloat Energy;
    MyFloat MassReturn;
    MyFloat Yield[4];
};

struct FeedbackResult {};

struct FeedbackWalk {
    double h;
    double current_time;
    int feedback_type;
    const char* ev_label;
};

static int feedback_isactive(int i, FeedbackWalk *fw) {
    if (Sp->P[i].getType() != 4)
        return 0;
    double age = fw->current_time - Sp->P[i].StellarAge;
    if ((Sp->P[i].FeedbackFlag & fw->feedback_type) != 0)
        return 0;
    if (fw->feedback_type == FEEDBACK_SNII && age > SNII_DELAY_TIME) return 1;
    if (fw->feedback_type == FEEDBACK_AGB && age > SNII_DELAY_TIME && age < AGB_END_TIME) return 1;
    if (fw->feedback_type == FEEDBACK_SNIa && age > SNIa_DELAY_TIME) return 1;
    return 0;
}

static void feedback_copy(int i, FeedbackInput *out, FeedbackWalk *fw) {
    out->Pos[0] = Sp->P[i].IntPos[0];
    out->Pos[1] = Sp->P[i].IntPos[1];
    out->Pos[2] = Sp->P[i].IntPos[2];

    double m_star = Sp->P[i].getMass();
    double age = fw->current_time - Sp->P[i].StellarAge;

    double energy = 0, m_return = 0;
    Yields y;

    if (fw->feedback_type == FEEDBACK_SNII) {
        energy = SNII_ENERGY_PER_MASS * m_star;
        m_return = MASS_RETURN_SNII * m_star;
        y = get_SNII_yields(m_return);
    } else if (fw->feedback_type == FEEDBACK_AGB) {
        energy = AGB_ENERGY_PER_MASS * m_star;
        m_return = MASS_RETURN_AGB * m_star;
        y = get_AGB_yields(m_return);
    } else if (fw->feedback_type == FEEDBACK_SNIa) {
        double n_snia = m_star * SNIa_RATE_PER_MASS;
        energy = n_snia * SNIa_ENERGY_PER_EVENT;
        m_return = 0;
        y = get_SNIa_yields(n_snia);
    }

    out->Energy = energy;
    out->MassReturn = m_return;
    for (int k = 0; k < 4; k++) out->Yield[k] = (&y.Z)[k];

    Sp->P[i].FeedbackFlag |= fw->feedback_type;

    if (fw->feedback_type == FEEDBACK_SNII) ThisStepEnergy_SNII += energy;
    if (fw->feedback_type == FEEDBACK_AGB)  ThisStepEnergy_AGB  += energy;
    if (fw->feedback_type == FEEDBACK_SNIa) ThisStepEnergy_SNIa += energy;

    ThisStepMassReturned += m_return;
    ThisStepMetalsInjected[0] += y.Z;
    ThisStepMetalsInjected[1] += y.C;
    ThisStepMetalsInjected[2] += y.O;
    ThisStepMetalsInjected[3] += y.Fe;
}

static void feedback_ngb(FeedbackInput *in, FeedbackResult *out, int j, FeedbackWalk *fw) {
    if (Sp->P[j].getType() != 0) return;

    double dx[3] = {
        NEAREST_X(Sp->P[j].IntPos[0] - in->Pos[0]),
        NEAREST_Y(Sp->P[j].IntPos[1] - in->Pos[1]),
        NEAREST_Z(Sp->P[j].IntPos[2] - in->Pos[2])
    };
    double r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
    double r = sqrt(r2);

    double wk = kernel_weight_cubic(r, fw->h);
    if (wk <= 0.0) return;
    double w = wk;

    Sp->SphP[j].Energy += in->Energy * w;
    Sp->P[j].Mass += in->MassReturn * w;
    Sp->P[j].Vel[0] += WIND_VELOCITY * w;
    for (int k = 0; k < 4; k++) Sp->SphP[j].Metals[k] += in->Yield[k] * w;
}

void apply_feedback_treewalk(double current_time, int feedback_type) {
    FeedbackWalk fw;
    fw.current_time = current_time;
    fw.feedback_type = feedback_type;
    fw.h = 0.5;
    fw.ev_label = "Feedback";

    for (int i = 0; i < Sp->NumPart; i++) {
        if (!feedback_isactive(i, &fw)) continue;

        FeedbackInput in;
        FeedbackResult out;
        feedback_copy(i, &in, &fw);

        for (int j = 0; j < Sp->NumPart; j++) {
            feedback_ngb(&in, &out, j, &fw);
        }
    }
}

void apply_stellar_feedback(double current_time, struct simparticles* Sp) {
    ThisStepEnergy_SNII = 0;
    ThisStepEnergy_SNIa = 0;
    ThisStepEnergy_AGB = 0;
    ThisStepMassReturned = 0;
    std::memset(ThisStepMetalsInjected, 0, sizeof(ThisStepMetalsInjected));

    apply_feedback_treewalk(current_time, FEEDBACK_SNII);
    apply_feedback_treewalk(current_time, FEEDBACK_AGB);
    apply_feedback_treewalk(current_time, FEEDBACK_SNIa);

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


#endif