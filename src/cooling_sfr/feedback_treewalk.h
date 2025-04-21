/*
 ==========================================================================================
 FEEDBACK_TREEWALK.H — Stellar Feedback using Octree in Gadget-4 (Header)
 ==========================================================================================
 
 ❖ Purpose:
 Header file for the treewalk-based stellar feedback implementation.
 
 ==========================================================================================
 */
 
 #ifndef FEEDBACK_TREEWALK_H
 #define FEEDBACK_TREEWALK_H
 
 #include "../data/simparticles.h"
 
 // External variables and functions
 extern double erg_per_mass_to_code;
 
 // Main entry points for feedback
 void apply_stellar_feedback(double current_time, simparticles* Sp);
 void apply_stellar_feedback_treewalk(double current_time, simparticles* Sp);
 void apply_stellar_feedback_bruteforce(double current_time, simparticles* Sp);
 
 #endif // FEEDBACK_TREEWALK_H