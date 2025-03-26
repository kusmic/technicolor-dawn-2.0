#ifndef FEEDBACK_TREEWALK_H
#define FEEDBACK_TREEWALK_H

extern double ThisStepEnergy_SNII;
extern double ThisStepEnergy_SNIa;
extern double ThisStepEnergy_AGB;
extern double ThisStepMassReturned;
extern double ThisStepMetalsInjected[4];

extern double TotalEnergyInjected_SNII;
extern double TotalEnergyInjected_SNIa;
extern double TotalEnergyInjected_AGB;
extern double TotalMassReturned;
extern double TotalMetalsInjected[4];

// These are defined in feedback_treewalk.cc and used externally:
void apply_feedback_treewalk(double current_time, int feedback_type);
void apply_stellar_feedback(double current_time, struct simparticles* Sp);

#endif

