/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file starformation.cc
 *
 *  \brief Generic creation routines for creating star particles
 */

#include "gadgetconfig.h"

#ifdef STARFORMATION

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>

#include "../cooling_sfr/cooling.h"
#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/mymalloc.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../system/system.h"
#include "../time_integration/timestep.h"

// Get a random star mass based on an IMF-weighted probability.
// This inverse‑CDF step simply transforms a uniform random number
// into that strongly declining distribution, so small masses dominate.
static inline double sample_IMF_mass(double alpha = 2.35, double m_min = 0.1, double m_max = 100.0) {
  // 1) draw uniform variate
  double u = get_random_number();  // ∈ [0,1)

  // 2) prepare exponent and bracket terms
  const double exp1   = 1.0 - alpha;             // exponent for integration
  const double lower  = std::pow(m_min, exp1);   // m_min^(1‑alpha)
  const double upper  = std::pow(m_max, exp1);   // m_max^(1‑alpha)

  // 3) invert the CDF:  m = [ lower + u*(upper‑lower) ]^(1/exp1)
  double mass = std::pow(lower + u * (upper - lower), 1.0 / exp1);

  return mass;
}

/** \brief This routine creates star/wind particles according to their respective rates.
 *
 *  This function loops over all the active gas cells. If in a given cell the SFR is
 *  greater than zero, the probability of forming a star or a wind particle is computed
 *  and the corresponding particle is created stichastically according to the model
 *  in Springel & Hernquist (2003, MNRAS). It also saves information about the formed stellar
 *  mass and the star formation rate in the file FdSfr.
 */
 void coolsfr::sfr_create_star_particles(simparticles *Sp)
 {
   TIMER_START(CPU_COOLING_SFR); // Start timing this function for performance monitoring
 
   double dt, dtime; // Time step variables
   MyDouble mass_of_star; // Mass of a newly formed star
   double sum_sm, total_sm, rate, sum_mass_stars, total_sum_mass_stars; // Variables for accumulating star mass
   double p = 0, pall = 0, prob, p_decide; // Probabilities for star formation
   double rate_in_msunperyear; // Star formation rate in solar masses per year
   double totsfrrate; // Total star formation rate across the simulation
   double w = 0; // Random number for metallicity update

   All.set_cosmo_factors_for_current_time(); // Update cosmological factors to current time
 
   stars_spawned = stars_converted = 0; // Counters for stars spawned and converted from gas
   sum_sm = sum_mass_stars = 0; // Initialize sum variables
 
   /* Loop over all active particles */
   for(int i = 0; i < Sp->TimeBinsHydro.NActiveParticles; i++)
     {
       int target = Sp->TimeBinsHydro.ActiveParticleList[i];
       if(Sp->P[target].getType() == 0) // Only process gas particles
         {
           if(Sp->P[target].getMass() == 0 && Sp->P[target].ID.get() == 0)
             continue; // Skip cells that have been swallowed or eliminated
 
           /* Calculate the actual time-step */
           dt = (Sp->P[target].getTimeBinHydro() ? (((integertime)1) << Sp->P[target].getTimeBinHydro()) : 0) * All.Timebase_interval;
           dtime = All.cf_atime * dt / All.cf_atime_hubble_a;
 
           mass_of_star = 0;
           prob         = 0;
           p            = 0;
 
           /* Calculate the probability of star formation based on the SFR */
           if(Sp->SphP[target].Sfr > 0)
             {
               p = Sp->SphP[target].Sfr / ((All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR)) * dtime /
                   Sp->P[target].getMass();
               pall = p;
               sum_sm += Sp->P[target].getMass() * (1 - exp(-p));
 
               w = get_random_number(); // Get a random number for stochastic processes
 
               /* Update metallicity for star formation */
               Sp->SphP[target].Metallicity += w * METAL_YIELD * (1 - exp(-p));
               Sp->SphP[target].MassMetallicity = Sp->SphP[target].Metallicity * Sp->P[target].getMass();
               Sp->P[target].Metallicity        = Sp->SphP[target].Metallicity;
 
               // Initially, Gadget-4 was just setting the mass of the star to the gas mass itself, so the entire gas particle is
               // converted into a star of a constant mass. Let's try to make it smarter.
              //double mstar_phys = sample_IMF_mass();  // sample a physical mass (in M⊙)
              double target_phys = 1e6;  // e.g. 10^6 M⊙ per star particle
              double ideal_code_mass = target_phys * (SOLAR_MASS / All.UnitMass_in_g);

              mass_of_star = std::min(ideal_code_mass, Sp->P[target].getMass());  // clamp to what’s actually available in the gas cell

              // After you compute ideal_code_mass:
              mpi_printf("[STAR DEBUG] idx=%d  SFR=%.3e  gas_mass=%.6f  ideal_mass=%.12f\n",
                i,
                Sp->SphP[target].Sfr,
                Sp->P[target].getMass(),
                ideal_code_mass);

              if (mass_of_star <= 0)
                  continue;

              // 5) recompute the probability for *this* packet mass
              prob = (Sp->P[target].getMass() / mass_of_star) * (1 - exp(-pall));

              if (prob <= 0)
                  continue;
             }
 
           /* Decide whether to form a star or kick gas to wind */
           p_decide = get_random_number();
           if(p_decide < p / pall) { // A star formation event is considered
             make_star(Sp, target, prob, mass_of_star, &sum_mass_stars); // Function to form a new star
             make_dust(Sp, target, prob, mass_of_star, &sum_mass_stars); // Function to simulate dust production
           }
           if(Sp->SphP[target].Sfr > 0)
             {
               if(Sp->P[target].getType() == 0) // Check again to avoid using a converted particle
                 {
                   /* Update metallicity post-star formation */
                   Sp->SphP[target].Metallicity += (1 - w) * METAL_YIELD * (1 - exp(-p));
                   Sp->SphP[target].MassMetallicity = Sp->SphP[target].Metallicity * Sp->P[target].getMass();
                 }
             }
           Sp->P[target].Metallicity = Sp->SphP[target].Metallicity;
         }
     } /* End of main loop over active gas particles */
 
   /* Collect star formation statistics across all processors */
   MPI_Allreduce(&stars_spawned, &tot_stars_spawned, 1, MPI_INT, MPI_SUM, Communicator);
   MPI_Allreduce(&stars_converted, &tot_stars_converted, 1, MPI_INT, MPI_SUM, Communicator);
 
   if(tot_stars_spawned > 0 || tot_stars_converted > 0)
     {
      mpi_printf("A STAR IS BORN! z=%.3f  SFR: spawned %d stars, converted %d gas particles into stars\n",
        1.0/All.Time - 1.0, tot_stars_spawned, tot_stars_converted);
     }
 
   tot_altogether_spawned = tot_stars_spawned;
   altogether_spawned     = stars_spawned;
   if(tot_altogether_spawned)
     {
       /* Assign new unique IDs to newly spawned stars if needed */
       if(All.MaxID == 0) // Check if MaxID has been calculated yet
         {
           MyIDType maxid = 0;
           for(int i = 0; i < Sp->NumPart; i++)
             if(Sp->P[i].ID.get() > maxid)
               maxid = Sp->P[i].ID.get();
 
           MyIDType *tmp = (MyIDType *)Mem.mymalloc("tmp", NTask * sizeof(MyIDType));
 
           MPI_Allgather(&maxid, sizeof(MyIDType), MPI_BYTE, tmp, sizeof(MyIDType), MPI_BYTE, Communicator);
 
           for(int i = 0; i < NTask; i++)
             if(tmp[i] > maxid)
               maxid = tmp[i];
 
           All.MaxID = maxid;
 
           Mem.myfree(tmp);
         }
 
       int *list = (int *)Mem.mymalloc("list", NTask * sizeof(int));
 
       MPI_Allgather(&altogether_spawned, 1, MPI_INT, list, 1, MPI_INT, Communicator);
 
       MyIDType newid = All.MaxID + 1;
 
       for(int i = 0; i < ThisTask; i++)
         newid += list[i];
 
       Mem.myfree(list);
 
       for(int i = 0; i < altogether_spawned; i++)
         Sp->P[Sp->NumPart + i].ID.set(newid++);
 
       All.MaxID += tot_altogether_spawned;
     }
 
   /* Handle updates to the simulation particle data and tree structures */
   if(tot_stars_spawned > 0 || tot_stars_converted > 0)
     {
       Sp->TotNumPart += tot_stars_spawned;
       Sp->TotNumGas -= tot_stars_converted;
       Sp->NumPart += stars_spawned;
     }
 
   double sfrrate = 0;
   for(int bin = 0; bin < TIMEBINS; bin++)
     if(Sp->TimeBinsHydro.TimeBinCount[bin])
       sfrrate += Sp->TimeBinSfr[bin];
 
   MPI_Allreduce(&sfrrate, &totsfrrate, 1, MPI_DOUBLE, MPI_SUM, Communicator);
 
   MPI_Reduce(&sum_sm, &total_sm, 1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
   MPI_Reduce(&sum_mass_stars, &total_sum_mass_stars, 1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
   if(ThisTask == 0)
     {
       if(All.TimeStep > 0)
         rate = total_sm / (All.TimeStep / All.cf_atime_hubble_a);
       else
         rate = 0;
 
       /* Compute the cumulative mass of stars formed */
       cum_mass_stars += total_sum_mass_stars;
 
       /* Convert the total mass converted into stars to a rate in solar masses per year */
       rate_in_msunperyear = rate * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);
 
       /* Log the star formation rate and other statistics */
       fprintf(Logs.FdSfr, "%14e %14e %14e %14e %14e %14e\n", All.Time, total_sm, totsfrrate, rate_in_msunperyear, total_sum_mass_stars, cum_mass_stars);
       myflush(Logs.FdSfr);
     }
 
   TIMER_STOP(CPU_COOLING_SFR); // Stop timing the function
 }
 

/** \brief Convert a SPH particle into a star.
 *
 *  This function converts an active star-forming gas particle into a star.
 *  The particle information of the gas is copied to the
 *  location istar and the fields necessary for the creation of the star
 *  particle are initialized.
 *
 *  \param i index of the gas particle to be converted
 *  \param birthtime time of birth (in code units) of the stellar particle
 */
 void coolsfr::convert_sph_particle_into_star(simparticles *Sp, int i, double birthtime)
 {
   printf("STAR: convert gas particle into star of mass %.3f!\n", Sp->P[i].getMass());

   Sp->P[i].setType(STAR_TYPE);
 #if NSOFTCLASSES > 1
   Sp->P[i].setSofteningClass(All.SofteningClassOfPartType[Sp->P[i].getType()]);
 #endif
 #ifdef INDIVIDUAL_GRAVITY_SOFTENING
   if(((1 << Sp->P[i].getType()) & (INDIVIDUAL_GRAVITY_SOFTENING)))
     Sp->P[i].setSofteningClass(Sp->get_softening_type_from_mass(Sp->P[i].getMass()));
 #endif
 
   Sp->TimeBinSfr[Sp->P[i].getTimeBinHydro()] -= Sp->SphP[i].Sfr;
 
   Sp->P[i].StellarAge = birthtime;
 
   return;
 }

/** \brief Convert a SPH particle into a dust particle.
 *
 *  This function converts an active gas particle into a dust particle.
 *  The particle information of the gas is copied to the
 *  location idust and the fields necessary for the creation of the dust
 *  particle are initialized.
 *
 *  \param i index of the gas particle to be converted
 *  \param birthtime time of birth (in code units) of the dust particle
 */
 void coolsfr::convert_sph_particle_into_dust(simparticles *Sp, int i, double birthtime)
 {

    printf("STAR: convert gas particle into dust of mass %.3f!\n", Sp->P[i].getMass());
     Sp->P[i].setType(DUST_TYPE);  // Change type to PartType6 (Dust)
     
 #if NSOFTCLASSES > 1
     Sp->P[i].setSofteningClass(All.SofteningClassOfPartType[Sp->P[i].getType()]);
 #endif
 
 #ifdef INDIVIDUAL_GRAVITY_SOFTENING
     if(((1 << Sp->P[i].getType()) & (INDIVIDUAL_GRAVITY_SOFTENING)))
         Sp->P[i].setSofteningClass(Sp->get_softening_type_from_mass(Sp->P[i].getMass()));
 #endif
 
     // Update the mass of dust (assume a fraction of gas turns into dust)
     double dust_fraction = 0.1; // Example: Convert 10% of gas into dust
     Sp->P[i].setMass(Sp->P[i].getMass() * dust_fraction); // Reduce gas mass accordingly
     
     // Dust retains some properties of the parent gas particle
     //Sp->P[i].Density = Sp->SphP[i].Density;  
     Sp->P[i].Metallicity = Sp->SphP[i].Metallicity;
     Sp->P[i].Vel[0] = Sp->P[i].Vel[0];
     Sp->P[i].Vel[1] = Sp->P[i].Vel[1];
     Sp->P[i].Vel[2] = Sp->P[i].Vel[2];
 
     // Assign birth time for tracking dust evolution
     Sp->P[i].DustAge = birthtime;
 
     return;
 }
 

/** \brief Spawn a star particle from a SPH gas particle.
 *
 *  This function spawns a star particle from an active star-forming
 *  SPH gas particle. The particle information of the gas is copied to the
 *  location istar and the fields necessary for the creation of the star
 *  particle are initialized. The total mass of the gas particle is split
 *  between the newly spawned star and the gas particle.
 *  (This function is probably unecessary)
 *
 *  \param igas index of the gas cell from which the star is spawned
 *  \param birthtime time of birth (in code units) of the stellar particle
 *  \param istar index of the spawned stellar particle
 *  \param mass_of_star the mass of the spawned stellar particle
 */
void coolsfr::spawn_star_from_sph_particle(simparticles *Sp, int igas, double birthtime, int istar, MyDouble mass_of_star)
{
  Sp->P[istar] = Sp->P[igas];
  Sp->P[istar].setType(STAR_TYPE);
#if NSOFTCLASSES > 1
  Sp->P[istar].setSofteningClass(All.SofteningClassOfPartType[Sp->P[istar].getType()]);
#endif
#ifdef INDIVIDUAL_GRAVITY_SOFTENING
  if(((1 << Sp->P[istar].getType()) & (INDIVIDUAL_GRAVITY_SOFTENING)))
    Sp->P[istar].setSofteningClass(Sp->get_softening_type_from_mass(Sp->P[istar].getMass()));
#endif

  Sp->TimeBinsGravity.ActiveParticleList[Sp->TimeBinsGravity.NActiveParticles++] = istar;

  Sp->TimeBinsGravity.timebin_add_particle(istar, igas, Sp->P[istar].TimeBinGrav, Sp->TimeBinSynchronized[Sp->P[istar].TimeBinGrav]);

  Sp->P[istar].setMass(mass_of_star);

  Sp->P[istar].StellarAge = birthtime;

  // Assign feedback portions of the mass
  Sp->P[istar].Mass_SNII = 0.10 * Sp->P[istar].getMass();
  Sp->P[istar].Mass_AGB  = 0.35 * Sp->P[istar].getMass();
  Sp->P[istar].Mass_SNIa = 0.02 * Sp->P[istar].getMass();

  /* now change the conserved quantities in the cell in proportion */
  double fac = (Sp->P[igas].getMass() - Sp->P[istar].getMass()) / Sp->P[igas].getMass();

  Sp->P[igas].setMass(fac * Sp->P[igas].getMass());

  return;
}

/** \brief Spawn a dust particle from a SPH gas particle.
 *
 *  This function spawns a dust particle from an SPH gas particle. The gas
 *  particle's properties are copied to the new dust particle, and the mass
 *  of the gas cell is reduced accordingly.
 *
 *  \param igas index of the gas cell from which the dust is spawned
 *  \param birthtime time of birth (in code units) of the dust particle
 *  \param idust index of the spawned dust particle
 *  \param mass_of_dust the mass of the spawned dust particle
 */
 void coolsfr::spawn_dust_from_sph_particle(simparticles *Sp, int igas, double birthtime, int idust, MyDouble mass_of_dust)
 {
  printf("DUST: spawn dust from gas particle of mass %.6f!\n", mass_of_dust);

     // Copy gas properties to new dust particle
     Sp->P[idust] = Sp->P[igas];
     Sp->P[idust].setType(DUST_TYPE);  // Assign to PartType6 (Dust)
 
 #if NSOFTCLASSES > 1
     Sp->P[idust].setSofteningClass(All.SofteningClassOfPartType[Sp->P[idust].getType()]);
 #endif
 
 #ifdef INDIVIDUAL_GRAVITY_SOFTENING
     if(((1 << Sp->P[idust].getType()) & (INDIVIDUAL_GRAVITY_SOFTENING)))
         Sp->P[idust].setSofteningClass(Sp->get_softening_type_from_mass(Sp->P[idust].getMass()));
 #endif
 
     // Add the dust particle to the active particle list
     Sp->TimeBinsGravity.ActiveParticleList[Sp->TimeBinsGravity.NActiveParticles++] = idust;
 
     // Synchronize time bins for dust
     Sp->TimeBinsGravity.timebin_add_particle(idust, igas, Sp->P[idust].TimeBinGrav, Sp->TimeBinSynchronized[Sp->P[idust].TimeBinGrav]);
 
     // Assign dust properties
     Sp->P[idust].setMass(mass_of_dust);
     Sp->P[idust].DustAge = birthtime;
 
     // Adjust mass of parent gas particle
     double fac = (Sp->P[igas].getMass() - Sp->P[idust].getMass()) / Sp->P[igas].getMass();
     Sp->P[igas].setMass(fac * Sp->P[igas].getMass());

     return;
 }
 
/** \brief Make a star particle from a SPH gas particle.
 *
 *  Given a gas cell where star formation is active and the probability
 *  of forming a star, this function either converts the entire gas
 *  particle into a star or spawns a new star packet of finite mass,
 *  while enforcing a minimum packet mass and protecting against
 *  excessive zero‑mass spawns.
 *
 *  \param Sp               pointer to particle arrays
 *  \param i                index of the gas cell
 *  \param prob             probability of forming a star packet
 *  \param mass_of_star     target mass (code units) for the star packet
 *  \param sum_mass_stars   accumulator for total stellar mass formed this timestep
 */
 void coolsfr::make_star(simparticles *Sp,
  int i,
  double prob,
  MyDouble mass_of_star,
  double *sum_mass_stars)
{
  // Minimum star packet mass to avoid unresolved, near-zero spawns
  const MyDouble MIN_STAR_MASS = 1e-11;  // code units; adjust based on resolution! 1e-4 is about 10^6 solar masses
  if (mass_of_star < MIN_STAR_MASS)
    return;

  // Prevent asking for more mass than available
  if (mass_of_star > Sp->P[i].getMass())
    Terminate("mass_of_star > P[i].Mass");

  // Stochastic formation decision
  if (get_random_number() < prob)
  {
  if (mass_of_star == Sp->P[i].getMass())
  {
  // Convert the entire gas particle into a star
  stars_converted++;
  *sum_mass_stars += Sp->P[i].getMass();
  convert_sph_particle_into_star(Sp, i, All.Time);

  // Debug output (only on task 0)
  if (ThisTask == 0)
    printf("STAR: convert gas -> star, mass %.6e code-units\n", Sp->P[i].getMass());
  }
  else
  {
  // Spawn a new star packet of finite mass
  altogether_spawned = stars_spawned;
  if (Sp->NumPart + altogether_spawned >= Sp->MaxPart)
  Terminate("NumPart=%d spawn %d particles no space left (Sp.MaxPart=%d)\n",
      Sp->NumPart, altogether_spawned, Sp->MaxPart);

  int j = Sp->NumPart + altogether_spawned;  // index for new star
  spawn_star_from_sph_particle(Sp, i, All.Time, j, mass_of_star);

  *sum_mass_stars += mass_of_star;
  stars_spawned++;

    // Debug output (only on task 0)
    if (ThisTask == 0)
      printf("STAR: spawn star of mass %.6e code-units\n", mass_of_star);
    }
  }
}

 

/** \brief Make a dust particle from a SPH gas particle.
 *
 *  Given a gas cell where dust formation is active and the probability
 *  of forming dust, this function spawns a dust tracer of finite mass
 *  (rather than converting the gas cell wholesale), and updates totals.
 *
 *  \param Sp             pointer to particle arrays
 *  \param i              index of the gas cell
 *  \param prob           probability of spawning a dust packet this timestep
 *  \param mass_of_dust   desired dust mass per packet (in code units)
 *  \param sum_mass_dust  accumulator for total dust mass formed on this task
 */
 void coolsfr::make_dust(simparticles *Sp,
  int i,
  double prob,
  MyDouble mass_of_dust,
  double *sum_mass_dust)
{
  printf("mass_of_dust!= %.6f code-units\n", mass_of_dust);

// --- Dust formation prerequisites ---
const double Z_min       = 0.01;       // minimum gas metallicity
const double Density_min = 1e-24;      // minimum density (cgs)
constexpr MyDouble MIN_DUST_MASS = 1e-6; // code-unit floor to ignore tiny spawns

// Only proceed if gas has enough metals and density
if (Sp->SphP[i].Metallicity < Z_min) return;
if (Sp->SphP[i].Density     < Density_min) return;

// Clamp the packet mass to what's available in the cell
MyDouble spawn_mass = std::min(mass_of_dust, Sp->P[i].getMass());
if (spawn_mass < MIN_DUST_MASS)
return;  // too small to spawn meaningfully

// Optionally, recompute the spawn probability based on spawn_mass?
// prob = (Sp->P[i].getMass() / spawn_mass) * (1 - exp(-pall));

// Stochastic draw: skip most timesteps
if (get_random_number() >= prob)
return;

// Reserve slots: ensure we don't exceed MaxPart
    altogether_spawned = dust_spawned;
    if (Sp->NumPart + altogether_spawned >= Sp->MaxPart) {
        mpi_printf("Rank %d: skipping dust spawn (no space): NumPart=%d, spawn=%d, MaxPart=%d\n",
                   ThisTask, Sp->NumPart, altogether_spawned, Sp->MaxPart);
        return;    // skip this spawn safely
    }

// Perform the spawn: create a new dust tracer of mass spawn_mass
int j = Sp->NumPart + altogether_spawned;
spawn_dust_from_sph_particle(Sp, i, All.Time, j, spawn_mass);

// Bookkeeping
*sum_mass_dust += spawn_mass;
dust_spawned++;
}

 
 #endif /* closes SFR */