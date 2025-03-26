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

struct FeedbackInput {
    MyDouble Pos[3];
    MyFloat Energy;
    MyFloat MassReturn;
    MyFloat Yield[4];
    int FeedbackType;
};

struct FeedbackResult {};

struct TreeWalkFeedback : public GenericTreeWalk<FeedbackInput, FeedbackResult> {
    double h = 0.5;
    double current_time;
    int feedback_type;

    TreeWalkFeedback() {
        this->usesymmetric = false;
    }

    bool is_active(int i) override {
        if (P[i].getType() != 4)
            return false;
        double age = current_time - P[i].StellarAge;
        if ((P[i].FeedbackFlag & feedback_type) != 0)
            return false;
        if (feedback_type == FEEDBACK_SNII && age > SNII_DELAY_TIME) return true;
        if (feedback_type == FEEDBACK_AGB && age > SNII_DELAY_TIME && age < AGB_END_TIME) return true;
        if (feedback_type == FEEDBACK_SNIa && age > SNIa_DELAY_TIME) return true;
        return false;
    }

    void particle2in(FeedbackInput& in, int i) override {
        double age = current_time - P[i].StellarAge;
        double m_star = P[i].getMass();
        Yields y;
        double energy = 0, m_return = 0;

        if (feedback_type == FEEDBACK_SNII) {
            energy = SNII_ENERGY_PER_MASS * m_star;
            m_return = MASS_RETURN_SNII * m_star;
            y = get_SNII_yields(m_return);
        } else if (feedback_type == FEEDBACK_AGB) {
            energy = AGB_ENERGY_PER_MASS * m_star;
            m_return = MASS_RETURN_AGB * m_star;
            y = get_AGB_yields(m_return);
        } else if (feedback_type == FEEDBACK_SNIa) {
            double n_snia = m_star * SNIa_RATE_PER_MASS;
            energy = n_snia * SNIa_ENERGY_PER_EVENT;
            m_return = 0;
            y = get_SNIa_yields(n_snia);
        }

        in.Pos[0] = P[i].IntPos[0];
        in.Pos[1] = P[i].IntPos[1];
        in.Pos[2] = P[i].IntPos[2];
        in.Energy = energy;
        in.MassReturn = m_return;
        for (int k = 0; k < 4; k++) in.Yield[k] = (&y.Z)[k];
        in.FeedbackType = feedback_type;

        P[i].FeedbackFlag |= feedback_type;

        if (feedback_type == FEEDBACK_SNII) ThisStepEnergy_SNII += energy;
        if (feedback_type == FEEDBACK_AGB)  ThisStepEnergy_AGB  += energy;
        if (feedback_type == FEEDBACK_SNIa) ThisStepEnergy_SNIa += energy;
        ThisStepMassReturned += m_return;
        ThisStepMetalsInjected[0] += y.Z;
        ThisStepMetalsInjected[1] += y.C;
        ThisStepMetalsInjected[2] += y.O;
        ThisStepMetalsInjected[3] += y.Fe;
    }

    void kernel(const FeedbackInput& in, int j, FeedbackResult& out) override {
        if (P[j].getType() != 0)
            return;
        double dx[3] = {
            NEAREST_X(P[j].IntPos[0] - in.Pos[0]),
            NEAREST_Y(P[j].IntPos[1] - in.Pos[1]),
            NEAREST_Z(P[j].IntPos[2] - in.Pos[2])
        };
        double r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
        double r = sqrt(r2);
        double wk = kernel_weight_cubic(r, h);
        if (wk <= 0.0) return;
        double w = wk;
        SphP[j].Energy += in.Energy * w;
        P[j].Mass += in.MassReturn * w;
        P[j].Vel[0] += WIND_VELOCITY * w;
        for (int k = 0; k < 4; k++) SphP[j].Metals[k] += in.Yield[k] * w;
    }
};

void apply_feedback_treewalk(double current_time, int feedback_type) {
    TreeWalkFeedback F;
    F.current_time = current_time;
    F.feedback_type = feedback_type;
    F.run();
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