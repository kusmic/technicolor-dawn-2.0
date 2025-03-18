/*
This module:
    Identifies old stars that should explode as supernovae.
    Distributes energy and mass to nearby gas.
    Spawns dust particles (PartType6) from SN ejecta.
*/

#include "gadgetconfig.h"

#ifdef STARFORMATION

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../cooling_sfr/cooling.h"
#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/mymalloc.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../system/system.h"
#include "../time_integration/timestep.h"
// #include "../data/simparticles.h"


// SUPERNOVA PARAMETERS
#define SN_ENERGY 1e51 // 10^51 erg per SN explosion
#define SN_FEEDBACK_RADIUS 0.1 // Interaction radius in kpc
#define SN_ENRICHMENT_FRACTION 0.3 // Fraction of mass converted to metals
#define SN_DUST_FRACTION 0.2 // Fraction of ejecta forming dust

/** \brief Check for star particles that should explode and inject feedback.
 *
 *  This function loops over all star particles, checking if they are old enough
 *  to undergo supernova explosions. If a supernova event is triggered, it injects
 *  thermal energy into nearby gas, enriches metals, and spawns dust particles.
 */
void handle_supernovae(simparticles *Sp) {
    for (int i = 0; i < Sp->NumPart; i++) {
        if (Sp->P[i].getType() == STAR_TYPE) {
            double star_age = Sp->P[i].StellarAge; // Star particle's age

            if (star_age > 3e7) { // Assume SN occurs after 30 Myr
                inject_supernova_feedback(Sp, i);
            }
        }
    }
}

/** \brief Inject supernova energy, enrich gas, and spawn dust.
 *
 *  This function distributes energy and mass from a supernova to nearby gas particles.
 *  It enriches gas with metals and spawns a dust particle from the ejecta.
 *
 *  \param i index of the exploding star particle
 */
void inject_supernova_feedback(simparticles *Sp, int i) {
    double mass_SN = SN_ENRICHMENT_FRACTION * Sp->P[i].getMass();
    double energy_SN = SN_ENERGY; // Assume 10^51 erg per SN
    int num_neighbors = 0;

    for (int j = 0; j < Sp->NumPart; j++) {
        if (Sp->P[j].getType() == 0) { // Gas particle
            double pos1[3], pos2[3];

            // Convert integer positions to floating-point positions
            for (int d = 0; d < 3; d++) {
                pos1[d] = Sp->P[i].get_position(d);
                pos2[d] = Sp->P[j].get_position(d);
            }

            double r = compute_distance(pos1, pos2, All.BoxSize);

            if (r < SN_FEEDBACK_RADIUS) {
                // Inject thermal energy
                // Sp->SphP[j].InternalEnergy += energy_SN / (r + 1e-3);

                // Inject metals
                Sp->SphP[j].Metallicity += mass_SN / num_neighbors;
                num_neighbors++;
            }
        }
    }

    // Spawn a dust particle from supernova ejecta
    spawn_dust_from_supernova(Sp, i);
}

/** \brief Create a dust particle from SN ejecta.
 *
 *  This function creates a new dust particle (PartType6) when a supernova occurs.
 *
 *  \param i index of the exploding star particle
 */
void spawn_dust_from_supernova(simparticles *Sp, int i) {
    if (Sp->NumPart + 1 >= Sp->MaxPart)
        Terminate("No space left for dust particles");

    int idust = Sp->NumPart++;
    Sp->P[idust] = Sp->P[i]; // Copy properties
    Sp->P[idust].setType(DUST_TYPE);
    Sp->P[idust].setMass(SN_DUST_FRACTION * Sp->P[i].getMass());

    // Give dust particles random velocity kick
    for (int d = 0; d < 3; d++)
        Sp->P[idust].Vel[d] += (0.1 * get_random_number()); // Random velocity perturbation

    return;
}

// Function to compute distance with periodic boundary conditions
double compute_distance(const double pos1[3], const double pos2[3], double box_size) {
    double dx, dy, dz;
    
    dx = pos1[0] - pos2[0];
    dy = pos1[1] - pos2[1];
    dz = pos1[2] - pos2[2];

    // Apply periodic boundary conditions
    if (dx > 0.5 * box_size) dx -= box_size;
    if (dx < -0.5 * box_size) dx += box_size;
    if (dy > 0.5 * box_size) dy -= box_size;
    if (dy < -0.5 * box_size) dy += box_size;
    if (dz > 0.5 * box_size) dz -= box_size;
    if (dz < -0.5 * box_size) dz += box_size;

    return sqrt(dx * dx + dy * dy + dz * dz);
}

double get_position(int d, struct particle_data *P)
{
    return ((double) P->IntPos[d]) * (All.BoxSize / (1LL << 30));
}

#endif // STARFORMATION