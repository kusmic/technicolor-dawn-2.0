#ifndef FEEDBACK_TREEWALK_H
#define FEEDBACK_TREEWALK_H

// These are defined in feedback_treewalk.cc and used externally:
void apply_feedback_treewalk(double current_time, int feedback_type);
void apply_stellar_feedback(double current_time, struct simparticles* Sp);

#endif

