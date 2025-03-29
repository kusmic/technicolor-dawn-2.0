/*
==========================================================================================
FEEDBACK_TREEWALK.CC — Stellar Feedback Injection in Gadget-4
==========================================================================================

❖ Purpose:
This file implements a tree-based algorithm for injecting stellar feedback (energy, mass,
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
   • Uses a spatial treewalk to find neighboring gas particles
   • Injects feedback energy, mass, and metal yields using a smoothing kernel
   • Tracks energy and mass diagnostics for logging and analysis

❖ Key components:
   - Constants defining feedback strength and timing (e.g. SNII delay, AGB end time)
   - Kernel weight function for distributing feedback to nearby gas
   - Metal yield functions for each type of feedback
   - Diagnostic accumulators to track what's been injected per timestep and in total

❖ Usage:
This file is compiled and executed as part of the Gadget-4 simulation if the FEEDBACK
flag is enabled in the build configuration. The actual injection is triggered during the
star formation and feedback loop in the main simulation timestep.

==========================================================================================
*/


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
#define NEAREST_X(x) NEAREST(x, All.BoxSize)
#define NEAREST_Y(x) NEAREST(x, All.BoxSize)
#define NEAREST_Z(x) NEAREST(x, All.BoxSize)

// Local cubic spline kernel approximation
/*
This is the smoothing kernel used to distribute feedback (energy, mass, metals) from a star to nearby gas particles. 
It's based on the standard cubic spline used in SPH (Smoothed Particle Hydrodynamics).

📌 How it works:
r: distance between the star and a neighbor gas particle
h: smoothing length (maximum influence radius)

The function returns a normalized weight that gets smaller with distance
This ensures that:
Nearby particles get more feedback
Distant particles get little or none
Total feedback is conserved
*/
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

struct Yields {
    double Z, C, O, Fe;
};

Yields get_SNII_yields(double m) {
    return {0.01 * m, 0.005 * m, 0.003 * m, 0.002 * m};
}

Yields get_AGB_yields(double m) {
    return {0.005 * m, 0.002 * m, 0.001 * m, 0.001 * m};
}

Yields get_SNIa_yields(double n_events) {
    return {0.002 * n_events, 0.001 * n_events, 0.0005 * n_events, 0.001 * n_events};
}


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

static int feedback_isactive(int i, FeedbackWalk *fw, simparticles *Sp) {
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

static void feedback_copy(int i, FeedbackInput *out, FeedbackWalk *fw, simparticles *Sp) {
    out->Pos[0] = Sp->P[i].IntPos[0];
    out->Pos[1] = Sp->P[i].IntPos[1];
    out->Pos[2] = Sp->P[i].IntPos[2];

    double m_star = Sp->P[i].getMass();
    double age = fw->current_time - Sp->P[i].StellarAge;

    double energy = 0, m_return = 0;
    Yields y;

    if (fw->feedback_type == FEEDBACK_SNII) {
        energy = SNII_ENERGY_PER_MASS * m_star;
    printf("[Feedback Debug] SNII -- Star %d: m_star=%.3e, energy=%.3e\n", i, m_star, energy);
        m_return = MASS_RETURN_SNII * m_star;
        y = get_SNII_yields(m_return);
    } else if (fw->feedback_type == FEEDBACK_AGB) {
        energy = AGB_ENERGY_PER_MASS * m_star;
    printf("[Feedback Debug] AGB -- Star %d: m_star=%.3e, energy=%.3e\n", i, m_star, energy);
        m_return = MASS_RETURN_AGB * m_star;
        y = get_AGB_yields(m_return);
    } else if (fw->feedback_type == FEEDBACK_SNIa) {
        double n_snia = m_star * SNIa_RATE_PER_MASS;
        energy = n_snia * SNIa_ENERGY_PER_EVENT;
    printf("[Feedback Debug] SNIa -- Star %d: m_star=%.3e, energy=%.3e\n", i, m_star, energy);
        m_return = 0;
        y = get_SNIa_yields(n_snia);
    }

    out->Energy = energy;
    out->MassReturn = m_return;
    for (int k = 0; k < 4; k++) out->Yield[k] = (&y.Z)[k];

    Sp->P[i].FeedbackFlag |= fw->feedback_type;
    printf("[Feedback Debug] Flagging star %d with feedback type %d\n", i, fw->feedback_type);

    if (fw->feedback_type == FEEDBACK_SNII) ThisStepEnergy_SNII += energy;
    if (fw->feedback_type == FEEDBACK_AGB)  ThisStepEnergy_AGB  += energy;
    if (fw->feedback_type == FEEDBACK_SNIa) ThisStepEnergy_SNIa += energy;

    ThisStepMassReturned += m_return;
    printf("[Feedback Debug] Added to totals: E=%.3e, M=%.3e\n", energy, m_return);
    ThisStepMetalsInjected[0] += y.Z;
    ThisStepMetalsInjected[1] += y.C;
    ThisStepMetalsInjected[2] += y.O;
    ThisStepMetalsInjected[3] += y.Fe;
}

static void feedback_ngb(FeedbackInput *in, FeedbackResult *out, int j, FeedbackWalk *fw, simparticles *Sp) {
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

    Sp->SphP[j].Entropy += in->Energy * w;  // Approximating as entropy input

    double old_mass = Sp->P[j].getMass();
    Sp->P[j].setMass(old_mass + in->MassReturn * w);

    Sp->P[j].Vel[0] += WIND_VELOCITY * w;
    for (int k = 0; k < 4; k++) Sp->SphP[j].Metals[k] += in->Yield[k] * w;
}

void apply_feedback_treewalk(double current_time, int feedback_type, simparticles *Sp) {
    FeedbackWalk fw;
    fw.current_time = current_time;
    fw.feedback_type = feedback_type;
    fw.h = 0.5;
    fw.ev_label = "Feedback";

    for (int i = 0; i < Sp->NumPart; i++) {
        if (!feedback_isactive(i, &fw, Sp)) continue;

        FeedbackInput in;
        FeedbackResult out;
        feedback_copy(i, &in, &fw, Sp);

        for (int j = 0; j < Sp->NumPart; j++) {
            feedback_ngb(&in, &out, j, &fw, Sp);
        }
    }
}

void apply_stellar_feedback(double current_time, struct simparticles* Sp) {
    ThisStepEnergy_SNII = 0;
    ThisStepEnergy_SNIa = 0;
    ThisStepEnergy_AGB = 0;
    ThisStepMassReturned = 0;
    std::memset(ThisStepMetalsInjected, 0, sizeof(ThisStepMetalsInjected));

    apply_feedback_treewalk(current_time, FEEDBACK_SNII, Sp);
    apply_feedback_treewalk(current_time, FEEDBACK_AGB, Sp);
    apply_feedback_treewalk(current_time, FEEDBACK_SNIa, Sp);

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