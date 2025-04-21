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
#include "../tree/tree.h"

#define SNII_ENERGY (1.0e51 / All.UnitEnergy_in_cgs)  // in internal units

/*! Parameters for energy partitioning */
double SNKickFraction = 0.3;  // 30% kinetic, 70% thermal

extern int FeedbackDebug; // declared elsewhere

extern int treewalk_run(TreeWalk * tw, simparticles * Sp);  // Ensure linkage

/**
 * Neighbor iteration callback
 */
int feedback_treewalk_ngbiter(TreeWalkQueryBase *input, TreeWalkResultBase *output, int j, TreeWalk *tw) {
    FeedbackInput *in = (FeedbackInput *) input;
    FeedbackResult *out = (FeedbackResult *) output;
    FeedbackWalk *fw = (FeedbackWalk *) tw->data;
    simparticles *Sp = (simparticles *) tw->priv;

    feedback_ngb(in, out, j, fw, Sp);
    return 0;
}

/**
 * Copy function to prepare the query for a particle
 */
void feedback_treewalk_copy(int i, TreeWalkQueryBase *query, TreeWalk *tw) {
    FeedbackInput *in = (FeedbackInput *) query;
    simparticles *Sp = (simparticles *) tw->priv;

    in->Pos[0] = intpos_to_kpc(Sp->P[i].IntPos[0]);
    in->Pos[1] = intpos_to_kpc(Sp->P[i].IntPos[1]);
    in->Pos[2] = intpos_to_kpc(Sp->P[i].IntPos[2]);
    in->Energy = SNII_ENERGY * Sp->P[i].getMass();
    in->MassReturn = 0.1 * Sp->P[i].getMass();
    in->NeighborCount = 16; // temporary default
    in->h = 0.3; // fixed for now
    in->SourceIndex = i;
}

/**
 * Predicate for whether a particle should be included in the walk
 */
int feedback_treewalk_haswork(int i, TreeWalk *tw) {
    simparticles *Sp = (simparticles *) tw->priv;
    return (Sp->P[i].getType() == 4); // star particles
}

/**
 * Treewalk-based wrapper for applying feedback
 */
void apply_feedback_treewalk(double current_time, int feedback_type, simparticles *Sp) {
    FeedbackWalk fw;
    fw.current_time = current_time;
    fw.feedback_type = feedback_type;

    TreeWalk tw = {};
    tw.type = 4; // star particles only
    tw.buffer = NULL;
    tw.data = &fw;
    tw.priv = Sp;
    tw.Nexport = NULL;
    tw.BunchSize = 0;

    tw.query_type = sizeof(FeedbackInput);
    tw.result_type = sizeof(FeedbackResult);
    tw.copy = feedback_treewalk_copy;
    tw.ngbiter = feedback_treewalk_ngbiter;
    tw.haswork = feedback_treewalk_haswork;

    treewalk_run(&tw, Sp); // actual invocation of the walk
}

/**
 * Apply feedback to a neighboring gas particle
 */
void feedback_ngb(FeedbackInput *in, FeedbackResult *out, int j, FeedbackWalk *fw, simparticles *Sp) {
    if (Sp->P[j].getType() != 0) return; // Only apply to gas particles

    double gas_mass = Sp->P[j].getMass();
    if (gas_mass <= 0 || isnan(gas_mass) || !isfinite(gas_mass)) return;

    if (FeedbackDebug) printf("[Feedback] Processing gas particle ID=%d\n", j);

    double gasPos[3];
    gasPos[0] = intpos_to_kpc(Sp->P[j].IntPos[0]);
    gasPos[1] = intpos_to_kpc(Sp->P[j].IntPos[1]);
    gasPos[2] = intpos_to_kpc(Sp->P[j].IntPos[2]);

    double dx[3] = {
        NEAREST_X(gasPos[0] - in->Pos[0]),
        NEAREST_Y(gasPos[1] - in->Pos[1]),
        NEAREST_Z(gasPos[2] - in->Pos[2])
    };
    double r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
    double r = sqrt(r2) + 1e-10;

    if (FeedbackDebug) printf("[Feedback] Distance to source r=%.5f (dx=[%.5f, %.5f, %.5f])\n", r, dx[0], dx[1], dx[2]);

    double E_total = in->Energy;
    double E_kin = E_total * SNKickFraction;
    double E_therm = E_total * (1.0 - SNKickFraction);

    double E_kin_j = E_kin / (double)in->NeighborCount;
    double E_therm_j = E_therm / (double)in->NeighborCount;

    double delta_u = E_therm_j * erg_to_code / gas_mass;

    double utherm_before = Sp->get_utherm_from_entropy(j);
    double utherm_after = clamp_feedback_energy(utherm_before, delta_u, j, Sp->P[j].ID);
    Sp->set_entropy_from_utherm(utherm_after, j);

    double final_u = Sp->get_utherm_from_entropy(j);
    if (!isfinite(final_u) || final_u > 1e10) {
        printf("[FEEDBACK WARNING] Bad final entropy on gas %d (ID=%llu): u=%.3e\n", j, (unsigned long long) Sp->P[j].ID, final_u);
    }

    double v_kick = sqrt(2.0 * E_kin_j * erg_to_code / gas_mass);
    for (int k = 0; k < 3; k++)
        Sp->P[j].Vel[k] += v_kick * dx[k] / r;

    if (FeedbackDebug) printf("[Feedback] Applied radial kick v_kick=%.3e km/s\n", v_kick);

    double mass_return = in->MassReturn / (double)in->NeighborCount;
    Sp->P[j].setMass(gas_mass + mass_return);

    if (FeedbackDebug) printf("[Feedback] Mass return: %.3e -> New mass: %.3e\n", mass_return, Sp->P[j].getMass());

    for (int k = 0; k < 4; k++) {
        double metal_add = in->Yield[k] / (double)in->NeighborCount / gas_mass;
        Sp->SphP[j].Metals[k] += metal_add;
        if (FeedbackDebug) printf("[Feedback] Metal[%d] += %.3e\n", k, metal_add);
    }
}

 /**
  * Main feedback function called each timestep
  */
  void apply_stellar_feedback(double current_time, simparticles* Sp) {
 
 
    
    /* -----TEMPORARY DIAGNOSTIC: Print TIME RESOLUTION-----
        static double last_time = 0;
        double timestep = current_time - last_time;
    
        // Convert from scale factor difference to physical time if needed
        double physical_timestep = 0;
        if (current_time > 0 && last_time > 0) {
            physical_timestep = scale_factor_to_physical_time(current_time) - 
                               scale_factor_to_physical_time(last_time);
    
            if (ThisTask == 0) {
                //printf("[Feedback Diagnostic] Current scale factor: %.6e\n", current_time);
                //printf("[Feedback Diagnostic] Timestep in scale factor: %.6e\n", timestep);
                //printf("[Feedback Diagnostic] Time resolution in physical years: %.2e\n", physical_timestep);
            }
        }
    
        last_time = current_time;
    // -----TEMPORARY DIAGNOSTIC: Print TIME RESOLUTION-----
    */
    
    
        // Reset diagnostic counters
        ThisStepEnergy_SNII = 0;
        ThisStepEnergy_SNIa = 0;
        ThisStepEnergy_AGB = 0;
        ThisStepMassReturned = 0;
        std::memset(ThisStepMetalsInjected, 0, sizeof(ThisStepMetalsInjected));
    
        // Apply each feedback type
        apply_feedback_treewalk(current_time, FEEDBACK_SNII, Sp);
        //apply_feedback_treewalk(current_time, FEEDBACK_SNII, Sp);
        //apply_feedback_treewalk(current_time, FEEDBACK_AGB, Sp);
        //apply_feedback_treewalk(current_time, FEEDBACK_SNIa, Sp);
    
        // Accumulate totals
        TotalEnergyInjected_SNII += ThisStepEnergy_SNII;
        TotalEnergyInjected_SNIa += ThisStepEnergy_SNIa;
        TotalEnergyInjected_AGB  += ThisStepEnergy_AGB;
        TotalMassReturned += ThisStepMassReturned;
        for (int k = 0; k < 4; k++)
            TotalMetalsInjected[k] += ThisStepMetalsInjected[k];
    
        // Print summary (on master process only)
        if (ThisTask == 0 && (ThisStepEnergy_SNII > 0 ||
           ThisStepEnergy_SNIa > 0 ||
           ThisStepEnergy_AGB > 0)) {
            FEEDBACK_PRINT("[Feedback Timestep Summary] E_SNII=%.3e erg, E_SNIa=%.3e erg, E_AGB=%.3e erg\n",
                   ThisStepEnergy_SNII, ThisStepEnergy_SNIa, ThisStepEnergy_AGB);
            FEEDBACK_PRINT("[Feedback Timestep Summary] Mass Returned=%.3e Msun\n", ThisStepMassReturned);
            FEEDBACK_PRINT("[Feedback Timestep Summary] Metals (Z=%.3e, C=%.3e, O=%.3e, Fe=%.3e) Msun\n",
                   ThisStepMetalsInjected[0], ThisStepMetalsInjected[1], ThisStepMetalsInjected[2], ThisStepMetalsInjected[3]);
        }
    }

#endif