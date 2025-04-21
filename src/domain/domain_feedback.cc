#include "gadgetconfig.h"
#include "../domain/domain.h"
#include "../data/simparticles.h"

// We need this for the run_feedback() tree!

// Instantiate the template and define the one global Domain object
template class domain<simparticles>;
domain<simparticles> Domain;