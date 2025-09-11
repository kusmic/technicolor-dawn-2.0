/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  gwalk.h
 *
 *  \brief defines a class for walking the gravitational tree
 */

#ifndef GRAVTREE_WALK_CUDA_H
#define GRAVTREE_WALK_CUDA_H

#include "gadgetconfig.h"
#include "../mpi_utils/shared_mem_handler.h"
#include <cuda_runtime.h>
#include "gravtree.h"
#include "gwalk_cuda_types.h"
#include "../data/simparticles.h"

class gwalk : public gravtree<simparticles>
{
 public:
  // Device data structure
  struct DeviceData {
    pinfo* d_pdats;
    gravnode* d_nodes;
    particle_data* d_particles;
  };
  
  void gravity_tree(int timebin);
  void initialize_cuda_memory();
  void cleanup_cuda_memory();

 private:
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

  __host__ __device__ void evaluate_particle_particle_interaction(const pinfo &pdat, const int no, const char jtype, int shmrank);
  __host__ __device__ void gravity_force_interact(const pinfo &pdat, int i, int no, char ptype, char no_type, unsigned char shmrank,
                                                int mintopleafnode, int committed);
  __host__ __device__ int evaluate_particle_node_opening_criterion_and_interaction(const pinfo &pdat, gravnode *nop);
  __host__ __device__ void gwalk_open_node(const pinfo &pdat, int i, char ptype, gravnode *nop, int mintopleafnode, int committed);
};

// CUDA kernel declarations 
__global__ void gravity_force_interact_kernel(const pinfo *pdats, int *is, int *nos, char *ptypes, char *no_types, 
                                            unsigned char *shmranks, int *mintopleafnodes, int *committeds, int n);

#endif // GRAVTREE_WALK_CUDA_H

