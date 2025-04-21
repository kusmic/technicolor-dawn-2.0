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

// Check if the particle is a star eligible for feedback at the current timestep
static int feedback_haswork(TreeWalkQueryBase *I, const int i, TreeWalk *tw)
{
    simparticles *Sp = tw->Sp;
    return (Sp->P[i].getType() == 4 && feedback_isactive(i, tw->priv, Sp));
}

// Determine feedback radius adaptively for each star particle
static void feedback_ngbiter(TreeWalkQueryNgbIter *iter, TreeWalk *tw)
{
    FeedbackWalk *fw = (FeedbackWalk *) tw->priv;
    simparticles *Sp = tw->Sp;

    double starPos[3] = {
        intpos_to_kpc(iter->base.IntPos[0]),
        intpos_to_kpc(iter->base.IntPos[1]),
        intpos_to_kpc(iter->base.IntPos[2])
    };

    int dummy_neighbors;
    double h = adaptive_feedback_radius(starPos, fw->feedback_type, Sp, &dummy_neighbors, NULL, 0);

    iter->Hsml = h;
}

// Visit each gas neighbor and apply feedback explicitly
static int feedback_visit_ngb(TreeWalkQueryBase *I, TreeWalkResultBase *O,
                              TreeWalkNgbIterBase *iter, LocalTreeWalk *lv)
{
    FeedbackWalk *fw = (FeedbackWalk *) lv->tw->priv;
    simparticles *Sp = lv->tw->Sp;

    int gas_index = iter->other;

    // Skip non-gas particles
    if (Sp->P[gas_index].getType() != 0) return 0;

    FeedbackInput in;
    FeedbackResult out;

    in.Pos[0] = intpos_to_kpc(I->IntPos[0]);
    in.Pos[1] = intpos_to_kpc(I->IntPos[1]);
    in.Pos[2] = intpos_to_kpc(I->IntPos[2]);
    in.FeedbackType = fw->feedback_type;
    in.Energy = SNII_ENERGY_PER_MASS * Sp->P[I->ID].getMass();
    in.MassReturn = 0.1 * Sp->P[I->ID].getMass();

    // Directly apply feedback to this gas neighbor
    feedback_to_gas_neighbor(&in, &out, gas_index, fw, Sp);

    return 0;
}

// Define your Feedback TreeWalk
TreeWalk TreeWalk_Feedback = {
    "FEEDBACK",
    sizeof(TreeWalkQueryBase),
    sizeof(TreeWalkResultBase),
    feedback_haswork,
    NULL,               // No special primary action
    NULL,               // No special secondary action
    NULL,               // No reduce
    feedback_ngbiter,   // Neighbor search radius
    feedback_visit_ngb, // Apply feedback directly to neighbors
    NULL,
    NULL
};

// Properly run the Feedback TreeWalk
void run_feedback(double current_time, int feedback_type, simparticles *Sp)
{
    FeedbackWalk feedback_walk;
    feedback_walk.current_time = current_time;
    feedback_walk.feedback_type = feedback_type;

    // Prepare ActiveParticles explicitly
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

    // Run the TreeWalk feedback mechanism
    if (ThisTask == 0)
        printf("[Feedback TreeWalk] Running feedback on %d active star particles...\n", Act.NumActiveParticle);

    TreeWalk_Feedback.priv = &feedback_walk;  // pass feedback_walk as private data
    TreeWalk_Feedback.Sp = Sp;                // attach particle data explicitly

    TreeWalk_run(&TreeWalk_Feedback, &Act);

    free(Act.ActiveParticle);
}

#endif // FEEDBACK