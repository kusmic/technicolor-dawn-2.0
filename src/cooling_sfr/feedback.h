#ifndef FEEDBACK_TREEWALK_H
#define FEEDBACK_TREEWALK_H

#include "../data/simparticles.h"  // or whatever defines `simparticles`


// ------------------------
// Global feedback tracking
// ------------------------

// These variables track feedback quantities applied during the **current timestep**:

// Energy injected by Type II Supernovae (SNII) during the current timestep (in ergs)
extern double ThisStepEnergy_SNII;

// Energy injected by Type Ia Supernovae (SNIa) during the current timestep (in ergs)
extern double ThisStepEnergy_SNIa;

// Energy injected by Asymptotic Giant Branch (AGB) stars during the current timestep (in ergs)
extern double ThisStepEnergy_AGB;

// Total stellar mass returned to the gas reservoir during the current timestep (in solar masses)
extern double ThisStepMassReturned;

// Array storing the total mass of individual metal species injected during this timestep.
// Indexing convention (example): [0] = C, [1] = O, [2] = Fe, [3] = other.
extern double ThisStepMetalsInjected[4];


// These variables accumulate total feedback quantities **over the full simulation runtime**:

// Cumulative energy injected by SNII events
extern double TotalEnergyInjected_SNII;

// Cumulative energy injected by SNIa events
extern double TotalEnergyInjected_SNIa;

// Cumulative energy injected by AGB winds
extern double TotalEnergyInjected_AGB;

// Total stellar mass returned to the ISM over the full simulation
extern double TotalMassReturned;

// Cumulative mass of individual metal species injected across the full simulation
extern double TotalMetalsInjected[4];

// Feedback types
#define FEEDBACK_SNII 1
#define FEEDBACK_SNIa 2
#define FEEDBACK_AGB  3

typedef struct {
    double current_time;
    int feedback_type;
} FeedbackWalk;

typedef struct {
    MyDouble Pos[3];
    int FeedbackType;
    double Energy;
    double MassReturn;
    int NeighborCount;
} FeedbackInput;

typedef struct {
    // you can leave this empty unless you need return values
} FeedbackResult;

void feedback_to_gas_neighbor(FeedbackInput *in, FeedbackResult *out, int j, FeedbackWalk *fw, simparticles *Sp);

int feedback_isactive(int i, FeedbackWalk *fw, simparticles *Sp);
double intpos_to_kpc(MyIntPosType ipos);
double adaptive_feedback_radius(MyDouble Pos[3], int feedback_type, simparticles *Sp, int *neighbors_ptr, void *unused1, int unused2);

void feedback_tree(int *active_list, int num_active);
int feedback_tree_evaluate(int target, int mode, int threadid);

// ------------------------
// Function Declarations
// ------------------------

// Main treewalk-driven feedback function
void run_feedback(double current_time, int feedback_type, simparticles *Sp);

// Core feedback driver:
// Applies stellar feedback at a given simulation time via a treewalk.
// The `feedback_type` parameter is a bitmask (e.g. SNII, AGB, or SNIa) to determine which phase to process.
void apply_feedback_treewalk(double current_time, int feedback_type);

// High-level interface:
// Invokes the appropriate feedback treewalk(s) for a given time and the full particle data.
void apply_stellar_feedback(double current_time, struct simparticles* Sp);


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
 

 
#endif
