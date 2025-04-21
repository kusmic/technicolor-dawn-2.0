/* feedback_tree.cc: Custom Feedback TreeWalk using Gadget-4 Gravity Tree style */

#include "gadgetconfig.h"
 
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <random>

#include "../cooling_sfr/feedback.h"
#include "../cooling_sfr/cooling.h"
#include "../sph/kernel.h"
#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/mymalloc.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../system/system.h"
#include "../time_integration/timestep.h"
#include "../domain/domain.h"


extern gravtree<simparticles> GravTree;
extern simparticles SimParticles;
#define NODE_IS_LEAF(n) ((n)->nextnode >= GravTree.MaxPart)  // fallback leaf condition

int feedback_tree_evaluate(int target, int mode, int threadid)
{
    simparticles *Sp = &SimParticles;

    if (Sp->P[target].getType() != 4) return 0;

    MyDouble starPos[3];
    starPos[0] = intpos_to_kpc(Sp->P[target].IntPos[0]);
    starPos[1] = intpos_to_kpc(Sp->P[target].IntPos[1]);
    starPos[2] = intpos_to_kpc(Sp->P[target].IntPos[2]);

    int n_neighbors;
    double h_feedback = adaptive_feedback_radius(starPos, FEEDBACK_SNII, Sp, &n_neighbors, NULL, 0);
    if (h_feedback < 0) return 0;

    FeedbackWalk fw;
    fw.current_time = All.Time;
    fw.feedback_type = FEEDBACK_SNII;

    FeedbackInput in;
    FeedbackResult out;
    in.Pos[0] = starPos[0]; in.Pos[1] = starPos[1]; in.Pos[2] = starPos[2];
    in.FeedbackType = FEEDBACK_SNII;
    in.Energy = SNII_ENERGY_PER_MASS * Sp->P[target].getMass();
    in.MassReturn = 0.1 * Sp->P[target].getMass();
    in.NeighborCount = n_neighbors;

    int no = GravTree.MaxPart;
    while (no >= 0) {
        gravnode *nop = &GravTree.Nodes[no];

        double dx = NEAREST_X(nop->center[0] - starPos[0]);
        double dy = NEAREST_Y(nop->center[1] - starPos[1]);
        double dz = NEAREST_Z(nop->center[2] - starPos[2]);
        double dist = sqrt(dx*dx + dy*dy + dz*dz);

        // Estimate node extent using radius from center to corner (if available)
        double node_extent = 0.5 * pow(GravTree.BoxSize, 1.0/3.0);  // fallback if no len available
        if (dist - node_extent > h_feedback) {
            no = nop->sibling;
            continue;
        }

        if (NODE_IS_LEAF(nop)) {
            int p = nop->nextnode;
            while (p >= 0) {
                if (Sp->P[p].getType() == 0) {
                    double gx = intpos_to_kpc(Sp->P[p].IntPos[0]);
                    double gy = intpos_to_kpc(Sp->P[p].IntPos[1]);
                    double gz = intpos_to_kpc(Sp->P[p].IntPos[2]);

                    double ddx = NEAREST_X(gx - starPos[0]);
                    double ddy = NEAREST_Y(gy - starPos[1]);
                    double ddz = NEAREST_Z(gz - starPos[2]);
                    double r2 = ddx * ddx + ddy * ddy + ddz * ddz;

                    if (sqrt(r2) <= h_feedback) {
                        feedback_to_gas_neighbor(&in, &out, p, &fw, Sp);
                    }
                }
                p = GravTree.Nextnode[p];
            }
            no = nop->sibling;
        } else {
            no = nop->nextnode;
        }
    }

    return 0;
}

void feedback_tree(int *active_list, int num_active)
{
    for (int i = 0; i < num_active; i++)
        feedback_tree_evaluate(active_list[i], 0, 0);
}
