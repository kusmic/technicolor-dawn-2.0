/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  gwalk_cuda.cuh
 *
 *  \brief defines a class for walking the gravitational tree
 */

#ifndef GRAVTREE_WALK_CUDA_H
#define GRAVTREE_WALK_CUDA_H

#include "gadgetconfig.h"
#include "../mpi_utils/shared_mem_handler.h"
#include <cuda_runtime.h>
#include "gravtree.h"
#include "../data/simparticles.h"

#ifndef GWALK_CUDA_IMPL_H
#define GWALK_CUDA_IMPL_H


inline void gwalk::mycxxsort(workstack_data* start, workstack_data* end, 
                            int (*compare)(const workstack_data&, const workstack_data&))
{
    std::sort(start, end, compare);
}

inline int gwalk::get_pinfo(int target, pinfo& pdat)
{
    // Implementation...
    if(target < Tp->NumPart)
    {
        pdat.intpos = Tp->P[target].IntPos;
        // ... rest of implementation ...
    }
    return ptype;
}

#endif // GWALK_CUDA_IMPL_H

#ifndef GWALK_CUDA_HELPERS_H
#define GWALK_CUDA_HELPERS_H

#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

inline void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#endif

#ifndef GWALK_CUDA_TYPES_H
#define GWALK_CUDA_TYPES_H

#include "../data/dtypes.h"
#include "../data/constants.h"
#include "../logs/timer.h"
#include "../gravtree/gravtree.h"
#include "../data/simparticles.h"

// Forward declarations
class simparticles;
template <typename partset> class gravtree;

// Node types
enum NodeType {
    NODE_TYPE_LOCAL_NODE = 0,
    NODE_TYPE_TREEPOINT_PARTICLE = 1,
    NODE_TYPE_LOCAL_PARTICLE = 2,
    NODE_TYPE_FETCHED_NODE = 3,
    NODE_TYPE_FETCHED_PARTICLE = 4
};

struct pinfo {
    MyIntPosType *intpos;
    MyReal aold;
    MyReal h_i;
    int Type;
#if NSOFTCLASSES > 1
    int SofteningClass;
#endif
#if defined(PMGRID) && defined(PLACEHIGHRESREGION)
    int InsideOutsideFlag;
#endif
    vector<MyFloat> *acc;
    MyFloat *pot;
    int *GravCost;
};

// Forward declare other required types
struct gravnode;
struct particle_data;
struct foreign_gravpoint_data;
struct workstack_data;
struct fetch_data;

#endif // GWALK_CUDA_TYPES_H

// Forward declare external variables
extern int MaxPart;
extern int MaxNodes;

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

  // Required utility functions
  void mycxxsort(workstack_data* start, workstack_data* end, int (*compare)(const workstack_data&, const workstack_data&));
  int get_pinfo(int target, pinfo& pdat);

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
  __host__ __device__ void gravity_force_interact(const pinfo &pdat, int i, int no, 
                                                char ptype, char no_type, unsigned char shmrank,
                                                int mintopleafnode, int committed);
  __host__ __device__ void gwalk_open_node(const pinfo &pdat, int i, char ptype, 
                                          gravnode *nop, int mintopleafnode, int committed);
};

// CUDA kernel declarations
__global__ void gravity_force_interact_kernel(const pinfo *pdats, int *is, int *nos, char *ptypes, 
                                            char *no_types, unsigned char *shmranks, 
                                            int *mintopleafnodes, int *committeds, int n);

#endif // GRAVTREE_WALK_CUDA_H

