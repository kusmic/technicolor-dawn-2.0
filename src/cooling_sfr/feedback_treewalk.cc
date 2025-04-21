/* === feedback_treewalk.cc: TreeWalk-based Feedback Module for Gadget-4 === */

#include "feedback.h"
#include "treewalk.h"
#include "allvars.h"

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
