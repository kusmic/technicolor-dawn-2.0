/*
 ==========================================================================================
 FEEDBACK.CC — Stellar Feedback Injection in Gadget-4
 ==========================================================================================
 
 ❖ Purpose:
 This file provides the main entry point for stellar feedback in Gadget-4 simulations.
 It calls the appropriate implementation (treewalk or brute-force) based on configuration.
 
 ==========================================================================================
 */
 
 #include "gadgetconfig.h"

 #ifdef FEEDBACK
 
 #include <math.h>
 #include <mpi.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 
 #include "../cooling_sfr/feedback_treewalk.h"
 #include "../data/allvars.h"
 #include "../logs/logs.h"
 #include "../system/system.h"
 
 /**
  * Main feedback function called from the time integration loop.
  * This is the entry point for applying stellar feedback.
  * 
  * If All.UseFeedbackTreewalk is set to 1 in the parameter file,
  * the more efficient tree-based implementation will be used.
  * Otherwise, the original brute-force approach is used.
  */
 void apply_stellar_feedback(double current_time, simparticles* Sp) {
     static int first_call = 1;
     
     // Initialize unit conversions on first call
     if(first_call) {
         erg_per_mass_to_code = 1.0 / (All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s);
         if(ThisTask == 0 && All.FeedbackDebug) {
             printf("[Feedback Init] erg_per_mass_to_code = %.3e (UnitVelocity = %.3e cm/s)\n",
                   erg_per_mass_to_code, All.UnitVelocity_in_cm_per_s);
         }
         first_call = 0;
     }
     
     // Decide which implementation to use
     if(All.UseFeedbackTreewalk) {
         if(ThisTask == 0 && All.FeedbackDebug) {
             printf("[Feedback] Using tree-based implementation\n");
         }
         apply_stellar_feedback_treewalk(current_time, Sp);
     }
     else {
         if(ThisTask == 0 && All.FeedbackDebug) {
             printf("[Feedback] Using brute-force implementation\n");
         }
         // Call original brute-force implementation (your existing code)
         // For debugging, you can compare results with both implementations
         apply_stellar_feedback_bruteforce(current_time, Sp);
     }
 }
 
 #endif // FEEDBACK