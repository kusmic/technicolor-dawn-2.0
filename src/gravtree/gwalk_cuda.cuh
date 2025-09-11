/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  gwalk.h
 *
 *  \brief defines a class for walking the gravitational tree
 */

#ifndef GRAVTREE_WALK_H
#define GRAVTREE_WALK_H

#include "gadgetconfig.h"
#include "../mpi_utils/shared_mem_handler.h"
#include <cuda_runtime.h>
#include "gravtree.h"  // Add this to get gravtree class definition
#include "../data/simparticles.h"  // Add this for particle data types

// Forward declarations
class simparticles;
struct gravnode;
struct particle_data;
struct foreign_gravpoint_data;
struct workstack_data;
struct fetch_data;

class gwalk : public gravtree<simparticles>
{
 public:
  void gravity_tree(int timebin);
  void initialize_cuda_memory();
  void cleanup_cuda_memory();

 private:
  // Device data structure
  struct DeviceData {
    pinfo* d_pdats;
    gravnode* d_nodes;
    particle_data* d_particles;
  };

  // Device data pointer
  DeviceData* d_data;
  
  long long interactioncountPP;
  long long interactioncountPN;

  MyReal theta2;
  MyReal thetamax2;
  MyReal errTolForceAcc;

#ifdef PRESERVE_SHMEM_BINARY_INVARIANCE
  bool skip_actual_force_computation;
#endif

  __host__ __device__ void gwalk_open_node(const pinfo &pdat, int i, char ptype, 
                                          gravnode *nop, int mintopleafnode, int committed);

  __host__ __device__ void gravity_force_interact(const pinfo &pdat, int i, int no, 
                                                char ptype, char no_type, unsigned char shmrank,
                                                int mintopleafnode, int committed);

  __host__ __device__ int evaluate_particle_node_opening_criterion_and_interaction(
      const pinfo &pdat, gravnode *nop);

  __host__ __device__ void evaluate_particle_particle_interaction(
      const pinfo &pdat, const int no, const char jtype, int shmrank);
};

// CUDA kernel declarations 
__global__ void gravity_force_interact_kernel(const gwalk::pinfo *pdats, int *is, 
    int *nos, char *ptypes, char *no_types, unsigned char *shmranks, 
    int *mintopleafnodes, int *committeds, int n);

#endif
        pdat.InsideOutsideFlag = Tp->P[i].InsideOutsideFlag;
#endif

        pdat.acc = &Tp->P[i].GravAccel;
#ifdef EVALPOTENTIAL
        pdat.pot = &Tp->P[i].Potential;
#endif
        pdat.GravCost = &Tp->P[i].GravCost;
      }
    else
      {
        ptype = NODE_TYPE_TREEPOINT_PARTICLE;

        int n = i - ImportedNodeOffset;

        pdat.intpos = Points[n].IntPos;

        pdat.Type = Points[n].Type;
#if NSOFTCLASSES > 1
        pdat.SofteningClass = Points[n].SofteningClass;
#endif
        pdat.aold = Points[n].OldAcc;
#if defined(PMGRID) && defined(PLACEHIGHRESREGION)
        pdat.InsideOutsideFlag = Points[n].InsideOutsideFlag;
#endif

        int idx  = ResultIndexList[n];
        pdat.acc = &ResultsActiveImported[idx].GravAccel;
#ifdef EVALPOTENTIAL
        pdat.pot = &ResultsActiveImported[idx].Potential;
#endif
        pdat.GravCost = &ResultsActiveImported[idx].GravCost;
      }

#if NSOFTCLASSES > 1
    pdat.h_i = All.ForceSoftening[pdat.SofteningClass];
#else
    pdat.h_i = All.ForceSoftening[0];
#endif

    return ptype;
  }

  __device__ void gwalk_open_node(const pinfo &pdat, int i, char ptype, 
                                 gravnode *nop, int mintopleafnode, int committed);

  __device__ void gravity_force_interact(const pinfo &pdat, int i, int no, 
                                       char ptype, char no_type, unsigned char shmrank,
                                       int mintopleafnode, int committed);

  __device__ int evaluate_particle_node_opening_criterion_and_interaction(
      const pinfo &pdat, gravnode *nop);

  __device__ void evaluate_particle_particle_interaction(
      const pinfo &pdat, const int no, const char jtype, int shmrank);
};

// CUDA kernel declaration
__global__ void gravity_force_interact_kernel(const pinfo *pdats, int *is, 
    int *nos, char *ptypes, char *no_types, unsigned char *shmranks, 
    int *mintopleafnodes, int *committeds, int n);

#endif
