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

 /* === feedback_treewalk.cc: TreeWalk-based Feedback Module for Gadget-4 === */

#include "allvars.h"
#include "proto.h"
#include "feedback.h"

// TreeWalk Query and Result Structures

typedef struct {
    MyDouble Pos[3];       // star position
    MyIDType ID;           // star ID
    int FeedbackType;      // SNII, SNIa, AGB, etc.
    double Energy;         // total feedback energy
    double MassReturn;     // returned mass to ISM
} TreeWalkQuery_Feedback;

typedef struct {
    // optionally accumulate data per star here
} TreeWalkResult_Feedback;

// Check if the star particle should trigger feedback
static int feedback_haswork_feedback(TreeWalkQuery_Feedback *I, const int i, TreeWalk *tw)
{
    simparticles *Sp = tw->Sp;
    FeedbackWalk *fw = (FeedbackWalk *) tw->priv;

    if (Sp->P[i].getType() != 4) return 0;
    if (!feedback_isactive(i, fw, Sp)) return 0;

    I->Pos[0] = intpos_to_kpc(Sp->P[i].IntPos[0]);
    I->Pos[1] = intpos_to_kpc(Sp->P[i].IntPos[1]);
    I->Pos[2] = intpos_to_kpc(Sp->P[i].IntPos[2]);
    I->ID     = Sp->P[i].ID.get();
    I->FeedbackType = fw->feedback_type;
    I->Energy       = SNII_ENERGY_PER_MASS * Sp->P[i].getMass();
    I->MassReturn   = 0.1 * Sp->P[i].getMass();

    return 1;
}

// Set neighbor search radius per star
static void feedback_ngbiter_feedback(TreeWalkQuery_Feedback *I, TreeWalkResult_Feedback *O,
                                      TreeWalkNgbIterBase *iter, LocalTreeWalk *lv)
{
    simparticles *Sp = lv->tw->Sp;
    int dummy_neighbors;

    double h = adaptive_feedback_radius(I->Pos, I->FeedbackType, Sp, &dummy_neighbors, NULL, 0);
    iter->Hsml = h;
}

// Visit neighbor and apply feedback
static int feedback_visit_ngb_feedback(TreeWalkQuery_Feedback *I, TreeWalkResult_Feedback *O,
                                       TreeWalkNgbIterBase *iter, LocalTreeWalk *lv)
{
    simparticles *Sp = lv->tw->Sp;
    FeedbackWalk *fw = (FeedbackWalk *) lv->tw->priv;
    int j = iter->other;

    if (Sp->P[j].getType() != 0) return 0;  // only apply to gas

    FeedbackInput in;
    FeedbackResult out;

    in.Pos[0] = I->Pos[0];
    in.Pos[1] = I->Pos[1];
    in.Pos[2] = I->Pos[2];
    in.FeedbackType = I->FeedbackType;
    in.Energy       = I->Energy;
    in.MassReturn   = I->MassReturn;
    in.NeighborCount = 0;  // unused for now

    feedback_to_gas_neighbor(&in, &out, j, fw, Sp);
    return 0;
}

TreeWalk TreeWalk_Feedback = {
    .Name = "FEEDBACK",
    .QueryType = sizeof(TreeWalkQuery_Feedback),
    .ResultType = sizeof(TreeWalkResult_Feedback),
    .haswork = (TreeWalkHasWorkFunc) feedback_haswork_feedback,
    .ngbiter = (TreeWalkNgbIterFunc) feedback_ngbiter_feedback,
    .visit = (TreeWalkVisitNgbFunc) feedback_visit_ngb_feedback,
    .priv = NULL,
};

void run_feedback(double current_time, int feedback_type, simparticles *Sp)
{
    FeedbackWalk feedback_walk;
    feedback_walk.current_time = current_time;
    feedback_walk.feedback_type = feedback_type;

    ActiveParticles Act;
    Act.NumActiveParticle = 0;
    Act.ActiveParticle = (int *) malloc(sizeof(int) * Sp->NumPart);

    for (int i = 0; i < Sp->NumPart; i++) {
        if (Sp->P[i].getType() != 4) continue;
        if (feedback_isactive(i, &feedback_walk, Sp)) {
            Act.ActiveParticle[Act.NumActiveParticle++] = i;
        }
    }

    if (Act.NumActiveParticle == 0) {
        if (ThisTask == 0)
            printf("[Feedback TreeWalk] No active star particles this step!\n");
        free(Act.ActiveParticle);
        return;
    }

    if (ThisTask == 0)
        printf("[Feedback TreeWalk] Running feedback on %d active star particles...\n", Act.NumActiveParticle);

    TreeWalk_Feedback.priv = &feedback_walk;
    TreeWalk_Feedback.Sp   = Sp;

    TreeWalk_run(&TreeWalk_Feedback, &Act);

    free(Act.ActiveParticle);
}


#endif // FEEDBACK