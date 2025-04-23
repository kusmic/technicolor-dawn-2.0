/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file sfr_eff.h
 *
 *  \brief Header for the effective multi-phase model of star formation and feedback
 */

#ifndef SFR_EFF_H
#define SFR_EFF_H

#include "../data/simparticles.h"

/**
 * Extension to the coolsfr class to add the effective multi-phase model
 * functionality for star formation and feedback from Springel & Hernquist (2003)
 */
class coolsfr
{
 private:
  /** \brief Gas particle state for cooling calculations */
  struct gas_state
  {
    double XH;        /* hydrogen abundance by mass */
    double yhelium;   /* helium abundance by mass */
    double nHcgs;     /* hydrogen number density in cgs units */
    double necgs;     /* electron number density in cgs units */
    double mhboltz;   /* hydrogen mass over Boltzmann constant */
    double ethmin;    /* minimum internal energy at high density */
    double ne;        /* electron abundance */
    double nH0;       /* neutral hydrogen abundance */
    double nHp;       /* ionized hydrogen abundance */
    double nHe0;      /* neutral helium abundance */
    double nHep;      /* ionized helium abundance */
    double nHepp;     /* double ionized helium abundance */
    double aHp;       /* hydrogen ionization rate */
    double aHep;      /* helium ionization rate */
    double aHepp;     /* helium 2nd ionization rate */
    double ad;        /* dielectric recombination rate */
    double geH0;      /* hydrogen excitation collisional rate */
    double geHe0;     /* helium excitation collisional rate */
    double geHep;     /* ionized helium excitation collisional rate */
    double bH0;       /* case B hydrogen recombination rate */
    double bHep;      /* case B helium recombination rate */
    double bff;       /* free-free emission rate */
    double gJH0ne;    /* ionization rate coefficient for neutral H */
    double gJHe0ne;   /* ionization rate coefficient for neutral He */
    double gJHepne;   /* ionization rate coefficient for singly ionized He */
  };

  /** \brief Data structure for cooling calculation inputs/outputs */
  struct do_cool_data
  {
    double u_old_input;    /* initial thermal energy per unit mass */
    double rho_input;      /* density */
    double dt_input;       /* time step */
    double ne_guess_input; /* initial electron number density guess */
  };

  /** \brief Interpolation table for cooling rates */
  struct rate_table
  {
    double BetaH0, BetaHep, Betaff;       /* recombination and free-free cooling */
    double AlphaHp, AlphaHep, AlphaHepp;  /* recombination rates */
    double Alphad;                        /* dielectric recombination rate */
    double GammaeH0, GammaeHe0, GammaeHep; /* collisional ionization rates */
  };

  /** \brief Interpolation table for photoionization rates */
  struct photo_table
  {
    double variable;        /* log z or linear time */
    double gH0, gHe, gHep;  /* photoionization rates */
    double eH0, eHe, eHep;  /* heating rates */
  };

  /** \brief Current photoionization and heating rates */
  struct photo_current
  {
    int J_UV;               /* flag if ionizing background exists */
    double gJH0, gJHe0, gJHep;  /* photoionization rates */
    double epsH0, epsHe0, epsHep; /* heating rates */
  };

  /** \brief Star formation statistics and counters */
  int stars_spawned;       /* Number of stars spawned in this task */
  int stars_converted;     /* Number of gas particles converted to stars in this task */
  int tot_stars_spawned;   /* Total number of stars spawned across all tasks */
  int tot_stars_converted; /* Total number of gas particles converted across all tasks */
  int altogether_spawned;  /* Total stars spawned on this task (including previous iterations) */
  int tot_altogether_spawned; /* Total stars spawned across all tasks (including previous iterations) */
  double cum_mass_stars;   /* Cumulative mass of stars formed */

  /** \brief Constants and lookup tables for cooling calculation */
  static const int MAXITER = 1000;  /* Maximum iterations for cooling solver */
  static const int NCOOLTAB = 2000; /* Number of entries in the cooling table */
  static const double Tmin = 1.0;   /* Minimum temperature in log10 for cooling tables */
  static const double Tmax = 9.0;   /* Maximum temperature in log10 for cooling tables */
  double deltaT;                    /* Temperature spacing in log10 for cooling tables */

  gas_state GasState;       /* Gas state for cooling calculations */
  do_cool_data DoCoolData;  /* Data for cooling calculations */
  photo_current pc;         /* Current photoionization rates */
  rate_table *RateT;        /* Rate tables for interpolation */
  photo_table *PhotoTUVB;   /* Photoionization rate tables */
  int NheattabUVB;          /* Number of entries in the photoionization table */

 public:
  /* Cooling functions */
  double DoCooling(double u_old, double rho, double dt, double *ne_guess, gas_state *gs, do_cool_data *DoCool);
  double GetCoolingTime(double u_old, double rho, double *ne_guess, gas_state *gs, do_cool_data *DoCool);
  double convert_u_to_temp(double u, double rho, double *ne_guess, gas_state *gs, const do_cool_data *DoCool);
  void find_abundances_and_rates(double logT, double rho, double *ne_guess, gas_state *gs, const do_cool_data *DoCool);
  double CoolingRateFromU(double u, double rho, double *ne_guess, gas_state *gs, const do_cool_data *DoCool);
  double CoolingRate(double logT, double rho, double *nelec, gas_state *gs, const do_cool_data *DoCool);
  double AbundanceRatios(double u, double rho, double *ne_guess, double *nH0_pointer, double *nHeII_pointer);
  void cool_sph_particle(simparticles *Sp, int i, gas_state *gs, do_cool_data *DoCool);
  void cooling_only(simparticles *Sp);
  void MakeRateTable(void);
  void ReadIonizeParams(char *fname);
  void IonizeParamsUVB(void);
  void SetZeroIonization(void);
  void IonizeParams(void);
  void InitCool(void);

  /* Star formation functions */
  double getZ(simparticles *Sp, int i);
  void init_clouds(void);
  void set_units_sfr(void);
  bool sf_evaluate_particle(simparticles *Sp, int i);
  double get_starformation_rate(simparticles *Sp, int i, double *cloudMassFraction);
  void update_thermodynamic_state(simparticles *Sp, int i, double dt, double cloudMassFraction);
  void cooling_and_starformation(simparticles *Sp);

#ifdef WINDS
  void winds_effective_model(simparticles *Sp, int i, double dt, double sfr, double cloudMassFraction);
#endif
};

#endif /* SFR_EFF_H */