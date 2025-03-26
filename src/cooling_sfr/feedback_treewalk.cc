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

// Interaction kernel for feedback
static void feedback_ngb_eval(TreeWalkQueryBase* input, TreeWalkResultBase* output, int i, int j, TreeWalk* tw)
{
    const auto* in = static_cast<feedback_data_in*>(input);
    auto* Pj = &P[j];
    auto* SphPj = &SphP[j];

    if (Pj->Type != 0) return; // Only affect gas

    double dx[3] = {
        NEAREST_X(Pj->Pos[0] - in->Pos[0]),
        NEAREST_Y(Pj->Pos[1] - in->Pos[1]),
        NEAREST_Z(Pj->Pos[2] - in->Pos[2])
    };

    double r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
    double r = sqrt(r2);

    double h = 0.5;  // hardcoded smoothing length for now
    double wk = kernel_weight_cubic(r, h);
    if (wk <= 0.0) return;

    // Weight by kernel (normalize later if needed)
    double w = wk;

    SphPj->Energy += in->Energy * w;
    Pj->Mass += in->MassReturn * w;
    Pj->Vel[0] += WIND_VELOCITY * w;
    SphPj->Metals[0] += in->Yield[0] * w;
    SphPj->Metals[1] += in->Yield[1] * w;
    SphPj->Metals[2] += in->Yield[2] * w;
    SphPj->Metals[3] += in->Yield[3] * w;
}

void apply_feedback_treewalk(double current_time, int feedback_type)
{
    TreeWalk tw{};
    tw.interact = feedback_ngb_eval;
    tw.dtype = TreeWalk::DENSITY;
    tw.flags = TREEWALK_KEEP_DM_HALOS;
    tw.use_sort = true;

    tw.prepare_func = [](int i, TreeWalkQueryBase* base, TreeWalk* tw) {
        auto* I = static_cast<feedback_data_in*>(base);
        if (P[i].Type != 4) return 0;

        double age = current_time - P[i].BirthTime;
        double m_star = P[i].Mass;

        if ((P[i].FeedbackFlag & feedback_type) != 0) return 0;

        Yields y;
        double energy = 0, m_return = 0;

        if (feedback_type == FEEDBACK_SNII && age > SNII_DELAY_TIME) {
            energy = SNII_ENERGY_PER_MASS * m_star;
            m_return = MASS_RETURN_SNII * m_star;
            y = get_SNII_yields(m_return);
        } else if (feedback_type == FEEDBACK_AGB && age > SNII_DELAY_TIME && age < AGB_END_TIME) {
            energy = AGB_ENERGY_PER_MASS * m_star;
            m_return = MASS_RETURN_AGB * m_star;
            y = get_AGB_yields(m_return);
        } else if (feedback_type == FEEDBACK_SNIa && age > SNIa_DELAY_TIME) {
            double n_snia = m_star * SNIa_RATE_PER_MASS;
            energy = n_snia * SNIa_ENERGY_PER_EVENT;
            m_return = 0;
            y = get_SNIa_yields(n_snia);
        } else {
            return 0;
        }

        I->Pos[0] = P[i].IntPos[0];
        I->Pos[1] = P[i].IntPos[1];
        I->Pos[2] = P[i].IntPos[2];
        I->Energy = energy;
        I->MassReturn = m_return;
        I->Yield[0] = y.Z;
        I->Yield[1] = y.C;
        I->Yield[2] = y.O;
        I->Yield[3] = y.Fe;
        I->FeedbackFlag = feedback_type;
        I->Mass = m_star;

        // Mark as processed
        P[i].FeedbackFlag |= feedback_type;

        return 1;
    };

    tw.size_query = sizeof(feedback_data_in);
    tw.size_result = sizeof(feedback_data_out);

    generic_treewalk<feedback_data_in, feedback_data_out>(&tw);
}

#endif