/*
 ==========================================================================================
 FEEDBACK_TREEWALK.H — Stellar Feedback for Gadget-4 (Header)
 ==========================================================================================
 
 ❖ Purpose:
 Header file for the stellar feedback implementation in Gadget-4.
 Includes declarations for both treewalk and brute-force implementations.
 
 ==========================================================================================
 */
 
 #ifndef FEEDBACK_TREEWALK_H
 #define FEEDBACK_TREEWALK_H
 
 #include "../data/simparticles.h"
 
 // External variables for energy conversion
 extern double erg_per_mass_to_code;
 
 // Main entry points for feedback
 void apply_stellar_feedback(double current_time, simparticles* Sp);
 void apply_stellar_feedback_treewalk(double current_time, simparticles* Sp);
 void apply_stellar_feedback_bruteforce(double current_time, simparticles* Sp);
 
 // Add this to param.h and read from param file
 namespace All {
     extern int FeedbackDebug;     // Print detailed diagnostics (0=no, 1=yes)
     extern int UseFeedbackTreewalk; // Use tree-based feedback algorithm (0=brute force, 1=tree)
 }
 
 #endif // FEEDBACK_TREEWALK_H