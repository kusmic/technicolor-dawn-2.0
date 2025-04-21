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
 
 #include "../data/simparticles.h"
 #include "../data/symtensors.h"
 #include "../domain/domain.h"
 #include "../mpi_utils/mpi_utils.h"
 #include "../tree/tree.h"
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

// TreeWalk Query and Result Structures

extern simparticles SimParticles;

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

typedef struct simparticles simparticles;

typedef struct {
    size_t QueryType;
    size_t ResultType;
    const char *Name;
    void *priv;
    simparticles *Sp;
    int (*haswork)(void *query, int i, void *tw);
    void (*ngbiter)(void *query, void *result, void *iter, void *lv);
    int (*visit)(void *query, void *result, void *iter, void *lv);
} TreeWalk;

typedef struct {
    TreeWalk *tw;
} LocalTreeWalk;

// Define dummy placeholder struct to satisfy compiler
typedef struct {
    double Hsml;
    int other;  // index of neighbor particle
} TreeWalkNgbIterBase;

 // Define NEAREST macros for periodic wrapping (or no-op if not periodic)
 #define NEAREST(x, box) (((x) > 0.5 * (box)) ? ((x) - (box)) : (((x) < -0.5 * (box)) ? ((x) + (box)) : (x)))
 #define NEAREST_X(x) NEAREST(x, All.BoxSize)
 #define NEAREST_Y(x) NEAREST(x, All.BoxSize)
 #define NEAREST_Z(x) NEAREST(x, All.BoxSize)
 
 // Feedback type bitmask flags
 #define FEEDBACK_SNII  1
 #define FEEDBACK_AGB   2
 #define FEEDBACK_SNIa  4
 
 #define SNII_ENERGY (1.0e51 / All.UnitEnergy_in_cgs)  // in internal units
 
// Make a nice function to output printf statements, but only if 
// FeedbackDebug = 1 in the param.txt
//static int FeedbackDebug;
#define FEEDBACK_PRINT(...) \
    do { if (All.FeedbackDebug) printf(__VA_ARGS__); } while (0)

 // Physical constants and conversion factors
 const double HUBBLE_TIME = 13.8e9;              // Hubble time in years (approx)
 
 // Feedback energy/mass return constants
 const double SNII_ENERGY_PER_MASS = 1.0e50;     // erg / Msun  Should return to ~1.0e51 erg, but in tiny test volumes, 
                                                 // where the mass of gas particles is especially large, raise it higher so the effect can be seen
 const double SNKickFraction = 0.3;  // 30% kinetic, 70% thermal
 const double SNIa_ENERGY_PER_EVENT = 1.0e51;    // erg per event
 const double AGB_ENERGY_PER_MASS = 1.0e47;      // erg / Msun
 
 // Feedback timescales - physical time!
 const double SNII_DELAY_TIME_PHYSICAL = 1.0e7;     // years
 const double SNIa_DELAY_TIME_PHYSICAL = 1.0e9;     // years 
 const double AGB_END_TIME_PHYSICAL = 1.0e10;       // years
 
 // Mass return fractions
 const double MASS_RETURN_SNII = 0.10;           // fraction of m_star
 const double MASS_RETURN_AGB = 0.30;            // fraction of m_star
 
 // SNIa parameters
 const double SNIa_RATE_PER_MASS = 2.0e-3;       // events per Msun over cosmic time
 const double SNIa_DTD_MIN_TIME = 4.0e7;         // years - minimum delay time
 const double SNIa_DTD_POWER = -1.1;             // power-law slope of delay-time distribution
 
 // Feedback approach parameters
 //const double SNII_FEEDBACK_RADIUS = 0.3;        // kpc - local deposition: ~0.3 kpc for SNII, which is more localized
 //const double SNIa_FEEDBACK_RADIUS = 0.8;        // kpc - wider distribution ~0.8 kpc for SNIa to account for the more diffuse nature of these events
 //const double AGB_FEEDBACK_RADIUS = 0.5;         // kpc - intermediate distribution ~0.5 kpc for AGB winds, which are more diffuse than SNII but more concentrated than SNIa
 // Added a check to make sure the gas particle is not too close, otherwise the
 // feedback is too strong, and the timestep goes to zero.
 const double MIN_FEEDBACK_SEPARATION = 1e-2;  // kpc; adjust this as needed

 // In case we have ergs and needs to get it to internal units
 double erg_to_code;
 // Convert erg/g → Gadget code units
 double erg_per_mass_to_code;

 // Conversion from fixed-point integer positions to physical units (kpc)
 // Assuming IntPos are stored as 32-bit integers: there are 2^32 discrete positions.
 //const double NUM_INT_STEPS = 4294967296.0;        // 2^32
 //const double conversionFactor = All.BoxSize / NUM_INT_STEPS;

 inline double intpos_to_kpc(uint32_t ipos) {
    return (double) ipos * All.BoxSize / 4294967296.0;
 }

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
 
 struct Yields {
     double Z, C, O, Fe;
 };
 
 // This random number generator will be used for stochastic SNIa events
 // Using a different name to avoid conflict with Gadget's existing random_generator
 std::mt19937 feedback_random_gen(std::random_device{}());
 
 /**
  * Convert cosmological scale factor to physical time in years
  * Note: This is a simple approximation - you might want to use a more accurate
  * conversion based on your cosmological parameters
  */
 double scale_factor_to_physical_time(double a) {
     return HUBBLE_TIME * pow(a, 3.0/2.0); // Simple matter-dominated universe approximation
 }
 
 /**
  * Convert physical time in years to cosmological scale factor
  */
 double physical_time_to_scale_factor(double time_physical) {
     return pow(time_physical / HUBBLE_TIME, 2.0/3.0); // Simple matter-dominated universe approximation
 }




/*  FUNCTIONS ------------------------------------------------------------------------------------- */

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


void run_feedback(double current_time, int feedback_type, simparticles *Sp) {
    int *Active = (int *) malloc(sizeof(int) * Sp->NumPart);
    int NumActive = 0;

    for (int i = 0; i < Sp->NumPart; i++) {
        if (Sp->P[i].getType() == 4 && feedback_isactive(i, NULL, Sp)) {
            Active[NumActive++] = i;
        }
    }

    if (NumActive > 0) {
        printf("[Feedback] Running feedback for %d stars.\n", NumActive);
        feedback_tree(Active, NumActive);
    }

    free(Active);
}


// Evaluate feedback for one star explicitly
int feedback_tree_evaluate(int target, int mode, int threadid)
{
    simparticles *Sp = &SimParticles;

    MyDouble starPos[3];
    starPos[0] = intpos_to_kpc(Sp->P[target].IntPos[0]);
    starPos[1] = intpos_to_kpc(Sp->P[target].IntPos[1]);
    starPos[2] = intpos_to_kpc(Sp->P[target].IntPos[2]);

    int n_neighbors;
    double h_feedback = adaptive_feedback_radius(starPos, FEEDBACK_SNII, Sp, &n_neighbors, NULL, 0);

    // Start Gadget-4 style tree traversal explicitly
    int no = MaxPart;  // start with the root node
    while(no >= 0)
    {
        NODE *nop = &Nodes[no];
        double dist = sqrt(pow(NEAREST_X(nop->center[0] - starPos[0]), 2) +
                           pow(NEAREST_Y(nop->center[1] - starPos[1]), 2) +
                           pow(NEAREST_Z(nop->center[2] - starPos[2]), 2));

        if(dist - nop->len > h_feedback) {
            no = nop->sibling;
            continue;
        }

        // If leaf node, check individual gas particles explicitly
        if(nop->u.d.bitflags & 1)
        {
            int p = nop->u.d.nextnode;
            while(p >= 0)
            {
                if(Sp->P[p].getType() == 0) {
                    double dx = NEAREST_X(intpos_to_kpc(Sp->P[p].IntPos[0]) - starPos[0]);
                    double dy = NEAREST_Y(intpos_to_kpc(Sp->P[p].IntPos[1]) - starPos[1]);
                    double dz = NEAREST_Z(intpos_to_kpc(Sp->P[p].IntPos[2]) - starPos[2]);

                    double r2 = dx*dx + dy*dy + dz*dz;
                    if(sqrt(r2) <= h_feedback) {
                        // explicitly apply feedback to neighbor particle p
                        FeedbackInput in;
                        FeedbackResult out;
                        FeedbackWalk fw;
                        fw.current_time = All.Time;
                        fw.feedback_type = FEEDBACK_SNII;

                        in.Pos[0] = starPos[0];
                        in.Pos[1] = starPos[1];
                        in.Pos[2] = starPos[2];
                        in.Energy = SNII_ENERGY_PER_MASS * Sp->P[target].getMass();
                        in.MassReturn = 0.1 * Sp->P[target].getMass();

                        feedback_to_gas_neighbor(&in, &out, p, &fw, Sp);
                    }
                }
                p = Nextnode[p];
            }
            no = nop->sibling;
        }
        else
        {
            no = nop->u.d.nextnode;
        }
    }

    return 0;
}

// Run feedback on active star particles explicitly
void feedback_tree(int *active_list, int num_active)
{
    for(int i = 0; i < num_active; i++)
        feedback_tree_evaluate(active_list[i], 0, 0);
}


#endif // FEEDBACK