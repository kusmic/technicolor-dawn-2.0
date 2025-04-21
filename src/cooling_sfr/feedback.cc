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
 */
void apply_stellar_feedback(double current_time, simparticles* Sp) {
    // Use the tree-based implementation for better performance
    apply_stellar_feedback_treewalk(current_time, Sp);
}

#endif // FEEDBACK