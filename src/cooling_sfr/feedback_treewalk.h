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
 
 // Main entry point for treewalk-based feedback
 void apply_stellar_feedback_treewalk(double current_time, simparticles* Sp);
 
 #endif // FEEDBACK_TREEWALK_H