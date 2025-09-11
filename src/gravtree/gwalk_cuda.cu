/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  gwalk.cc
 *
 *  \brief implements the routines for walking the gravity tree and accumulating forces
 */

#include "gadgetconfig.h"

#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/intposconvert.h"
#include "../data/mymalloc.h"
#include "../domain/domain.h"
#include "../gravity/ewald.h"
#include "../gravtree/gravtree.h"
#include "../gravtree/gwalk_cuda.cuh"
#include <cuda_runtime.h>

// Remove duplicate CUDA_CHECK macro and keep only one definition
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/*! This file contains the code for the gravitational force computation by
 *  means of the tree algorithm. To this end, a tree force is computed for all
 *  active local particles, and particles are exported to other processors if
 *  needed, where they can receive additional force contributions. If the
 *  TreePM algorithm is enabled, the force computed will only be the
 *  short-range part.
 */

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


// Device data structure
struct DeviceData {
    pinfo* d_pdats;
    gravnode* d_nodes;
    particle_data* d_particles;
    // Add other necessary device pointers
};

// Global device data
__device__ DeviceData d_data;

/*! \brief This function computes the gravitational forces for all active particles.
 *
 * The tree walk is done in two phases: First the local part of the force tree is processed (gravity_primary_loop() ).
 * Whenever an external node is encountered during the walk, this node is saved on a list.
 * This node list along with data about the particles is then exchanged among tasks.
 * In the second phase (gravity_secondary_loop() ) each task now continues the tree walk for
 * the imported particles. Finally the resulting partial forces are send back to the original task
 * and are summed up there to complete the tree force calculation.
 *
 * Particles are only exported to other processors when really needed, thereby allowing a
 * good use of the communication buffer. Every particle is sent at most once to a given processor
 * together with the complete list of relevant tree nodes to be checked on the other task.
 *
 * Particles which drifted into the domain of another task are sent to this task for the force computation.
 * Afterwards the resulting force is sent back to the originating task.
 *
 * In order to improve the work load balancing during a domain decomposition, the work done by each
 * node/particle is measured. The work is measured for the interaction partners (i.e. the nodes or particles)
 * and not for the particles itself that require a force computation. This way, work done for imported
 * particles is accounted for at the task where the work actually incurred. The cost measurement is
 * only done for the "GRAVCOSTLEVELS" highest occupied time bins. The variable #MeasureCostFlag will state whether a
 * measurement is done at the present time step.
 *
 * The tree imbalance can be further reduced using chunking. The particles requiring a force computation
 * are split into chunks of size #Nchunksize. A set of every #Nchunk -th chunk is processed first.
 * Then the process is repeated, processing the next set of chunks. This way the amount of exported particles
 * is more balanced, as communication heavy regions are mixed with less communication intensive regions.
 *
 */

void gwalk::gravity_tree(int timebin)
{
  interactioncountPP = 0;
  interactioncountPN = 0;

  TIMER_STORE;
  TIMER_START(CPU_TREE);

  D->mpi_printf("GRAVTREE: Begin tree force. timebin=%d (presently allocated=%g MB)\n", timebin, Mem.getAllocatedBytesInMB());

#ifdef PMGRID
  set_mesh_factors();
#endif

  TIMER_START(CPU_TREESTACK);

  // Create list of targets (the work queue). There are initially two possible sources of points, local ones, and imported ones.

  NumOnWorkStack         = 0;
  AllocWorkStackBaseLow  = std::max<int>(1.5 * (Tp->NumPart + NumPartImported), TREE_MIN_WORKSTACK_SIZE);
  AllocWorkStackBaseHigh = AllocWorkStackBaseLow + TREE_EXPECTED_CYCLES * TREE_MIN_WORKSTACK_SIZE;
  MaxOnWorkStack         = AllocWorkStackBaseLow;

  WorkStack       = (workstack_data *)Mem.mymalloc("WorkStack", AllocWorkStackBaseHigh * sizeof(workstack_data));
  ResultIndexList = (int *)Mem.mymalloc("ResultIndexList", NumPartImported * sizeof(int));

  for(int i = 0; i < Tp->TimeBinsGravity.NActiveParticles; i++)
    {
      int target = Tp->TimeBinsGravity.ActiveParticleList[i];

      // if we have exported particles, we need to explicitly check whether this particle is among them
      if(NumPartExported > 0)
        {
          MyIntPosType xxb       = Tp->P[target].IntPos[0];
          MyIntPosType yyb       = Tp->P[target].IntPos[1];
          MyIntPosType zzb       = Tp->P[target].IntPos[2];
          MyIntPosType mask      = (((MyIntPosType)1) << (BITS_FOR_POSITIONS - 1));
          unsigned char shiftx   = (BITS_FOR_POSITIONS - 3);
          unsigned char shifty   = (BITS_FOR_POSITIONS - 2);
          unsigned char shiftz   = (BITS_FOR_POSITIONS - 1);
          unsigned char rotation = 0;

          int no = 0;
          while(D->TopNodes[no].Daughter >= 0) /* walk down top tree to find correct leaf */
            {
              unsigned char pix     = (((unsigned char)((xxb & mask) >> (shiftx--))) | ((unsigned char)((yyb & mask) >> (shifty--))) |
                                   ((unsigned char)((zzb & mask) >> (shiftz--))));
              unsigned char subnode = peano_incremental_key(pix, &rotation);
              mask >>= 1;
              no = D->TopNodes[no].Daughter + subnode;
            }

          no       = D->TopNodes[no].Leaf;
          int task = D->TaskOfLeaf[no];

          if(task == D->ThisTask)
            {
              WorkStack[NumOnWorkStack].Target         = target;
              WorkStack[NumOnWorkStack].Node           = MaxPart;
              WorkStack[NumOnWorkStack].ShmRank        = Shmem.Island_ThisTask;
              WorkStack[NumOnWorkStack].MinTopLeafNode = MaxPart + D->NTopnodes;
              NumOnWorkStack++;
            }
        }
      else
        {
          WorkStack[NumOnWorkStack].Target         = target;
          WorkStack[NumOnWorkStack].Node           = MaxPart;
          WorkStack[NumOnWorkStack].ShmRank        = Shmem.Island_ThisTask;
          WorkStack[NumOnWorkStack].MinTopLeafNode = MaxPart + D->NTopnodes;
          NumOnWorkStack++;
        }

      /* let's do a safety check here to protect against accidental use of zero softening lengths */
      int softtype = Tp->P[target].getSofteningClass();
      if(All.ForceSoftening[softtype] == 0)
        Terminate("Particle with ID=%lld of type=%d and softening type=%d was assigned zero softening\n",
                  (long long)Tp->P[target].ID.get(), Tp->P[target].getType(), softtype);
    }

  int ncount = 0;

  for(int i = 0; i < NumPartImported; i++)
    {
#ifndef HIERARCHICAL_GRAVITY
      if(Points[i].ActiveFlag)
#endif
        {
          ResultIndexList[i] = ncount++;

          WorkStack[NumOnWorkStack].Target         = i + ImportedNodeOffset;
          WorkStack[NumOnWorkStack].Node           = MaxPart;
          WorkStack[NumOnWorkStack].ShmRank        = Shmem.Island_ThisTask;
          WorkStack[NumOnWorkStack].MinTopLeafNode = MaxPart + D->NTopnodes;
          NumOnWorkStack++;
        }
    }

#ifdef PRESERVE_SHMEM_BINARY_INVARIANCE
  workstack_data *WorkStackBak = (workstack_data *)Mem.mymalloc("WorkStackBak", NumOnWorkStack * sizeof(workstack_data));
  int NumOnWorkStackBak        = NumOnWorkStack;
  memcpy(WorkStackBak, WorkStack, NumOnWorkStack * sizeof(workstack_data));
#endif

  ResultsActiveImported =
      (resultsactiveimported_data *)Mem.mymalloc_clear("ResultsActiveImported", ncount * sizeof(resultsactiveimported_data));

  /******************************************/
  /* now execute the tree walk calculations */
  /******************************************/

  theta2         = All.ErrTolTheta * All.ErrTolTheta;
  thetamax2      = All.ErrTolThetaMax * All.ErrTolThetaMax;
  errTolForceAcc = All.ErrTolForceAcc;

  sum_NumForeignNodes  = 0;
  sum_NumForeignPoints = 0;

  // set a default size of the fetch stack equal to half the work stack (this may still be somewhat too large)
  MaxOnFetchStack = std::max<int>(0.1 * (Tp->NumPart + NumPartImported), TREE_MIN_WORKSTACK_SIZE);
  StackToFetch    = (fetch_data *)Mem.mymalloc_movable(&StackToFetch, "StackToFetch", MaxOnFetchStack * sizeof(fetch_data));

  // let's grab at most half the still available memory for imported points and nodes
  int nspace = (0.5 * Mem.FreeBytes) / (sizeof(gravnode) + 8 * sizeof(foreign_gravpoint_data));

  MaxForeignNodes  = nspace;
  MaxForeignPoints = 8 * nspace;
  NumForeignNodes  = 0;
  NumForeignPoints = 0;

  /* the following two arrays hold imported tree nodes and imported points to augment the local tree */
  Foreign_Nodes  = (gravnode *)Mem.mymalloc_movable(&Foreign_Nodes, "Foreign_Nodes", MaxForeignNodes * sizeof(gravnode));
  Foreign_Points = (foreign_gravpoint_data *)Mem.mymalloc_movable(&Foreign_Points, "Foreign_Points",
                                                                  MaxForeignPoints * sizeof(foreign_gravpoint_data));

  tree_initialize_leaf_node_access_info();

  TIMER_STOP(CPU_TREESTACK);

  double t0       = Logs.second();
  int max_ncycles = 0;

  prepare_shared_memory_access();

#ifdef PRESERVE_SHMEM_BINARY_INVARIANCE
  for(int rep = 0; rep < 2; rep++)
    {
      if(rep == 0)
        {
          skip_actual_force_computation = true;
        }
      else
        {
          skip_actual_force_computation = false;
          NumOnWorkStack                = NumOnWorkStackBak;
          memcpy(WorkStack, WorkStackBak, NumOnWorkStack * sizeof(workstack_data));
        }
#endif

      while(NumOnWorkStack > 0)  // repeat until we are out of work
        {
          NewOnWorkStack  = 0;  // gives the new entries
          NumOnFetchStack = 0;
          MaxOnWorkStack  = std::min<int>(AllocWorkStackBaseLow + max_ncycles * TREE_MIN_WORKSTACK_SIZE, AllocWorkStackBaseHigh);

          TIMER_START(CPU_TREEWALK);

          int item = 0;

          while(item < NumOnWorkStack)
            {
              int committed = 8 * TREE_NUM_BEFORE_NODESPLIT;
              int min_buffer_space =
                  std::min<int>(MaxOnWorkStack - (NumOnWorkStack + NewOnWorkStack), MaxOnFetchStack - NumOnFetchStack);
              if(min_buffer_space >= committed)
                {
                  int target     = WorkStack[item].Target;
                  int no         = WorkStack[item].Node;
                  int shmrank    = WorkStack[item].ShmRank;
                  int mintopleaf = WorkStack[item].MinTopLeafNode;
                  item++;

                  pinfo pdat;
                  int ptype = get_pinfo(target, pdat);

                  if(no == MaxPart)
                    {
                      // we have a pristine particle that's processed for the first time
                      gravity_force_interact(pdat, target, no, ptype, NODE_TYPE_LOCAL_NODE, shmrank, mintopleaf, committed);
                    }
                  else
                    {
                      // we have a node that we previously could not open
                      gravnode *nop = get_nodep(no, shmrank);

                      if(nop->cannot_be_opened_locally)
                        {
                          Terminate("item=%d:  no=%d  now we should be able to open it!", item, no);
                        }
                      else
                        gwalk_open_node(pdat, target, ptype, nop, mintopleaf, committed);
                    }
                }
              else
                break;
            }

          if(item == 0 && NumOnWorkStack > 0)
            Terminate("Can't even process a single particle");

          TIMER_STOP(CPU_TREEWALK);

          TIMER_START(CPU_TREEFETCH);

          tree_fetch_foreign_nodes(FETCH_GRAVTREE);

          TIMER_STOP(CPU_TREEFETCH);

          TIMER_START(CPU_TREESTACK);

          /* now reorder the workstack such that we are first going to do residual pristine particles, and then
           * imported nodes that hang below the first leaf nodes */
          NumOnWorkStack = NumOnWorkStack - item + NewOnWorkStack;
          memmove(WorkStack, WorkStack + item, NumOnWorkStack * sizeof(workstack_data));

          /* now let's sort such that we can go deep on top-level node branches, allowing us to clear them out eventually */
          mycxxsort(WorkStack, WorkStack + NumOnWorkStack, compare_workstack);

          TIMER_STOP(CPU_TREESTACK);

          max_ncycles++;
        }

#ifdef PRESERVE_SHMEM_BINARY_INVARIANCE
    }
#endif

  TIMER_START(CPU_TREEIMBALANCE);

  MPI_Allreduce(MPI_IN_PLACE, &max_ncycles, 1, MPI_INT, MPI_MAX, D->Communicator);

  TIMER_STOP(CPU_TREEIMBALANCE);

  cleanup_shared_memory_access();

  /* free temporary buffers */

  Mem.myfree(Foreign_Points);
  Mem.myfree(Foreign_Nodes);
  Mem.myfree(StackToFetch);

  double t1 = Logs.second();

  D->mpi_printf("GRAVTREE: tree-forces are calculated, with %d cycles took %g sec\n", max_ncycles, Logs.timediff(t0, t1));

  /* now communicate the forces in ResultsActiveImported */
  gravity_exchange_forces();

  Mem.myfree(ResultsActiveImported);
#ifdef PRESERVE_SHMEM_BINARY_INVARIANCE
  Mem.myfree(WorkStackBak);
#endif
  Mem.myfree(ResultIndexList);
  Mem.myfree(WorkStack);

  TIMER_STOP(CPU_TREE);

  D->mpi_printf("GRAVTREE: tree-force is done.\n");

  /*  gather some diagnostic information */

  TIMER_START(CPU_LOGS);

  struct detailed_timings
  {
    double tree, wait, fetch, stack, all, lastpm;
    double costtotal, numnodes;
    double interactioncountPP, interactioncountPN;
    double NumForeignNodes, NumForeignPoints;
    double fillfacFgnNodes, fillfacFgnPoints;
  };
  detailed_timings timer, tisum, timax;

  timer.tree               = TIMER_DIFF(CPU_TREEWALK);
  timer.wait               = TIMER_DIFF(CPU_TREEIMBALANCE);
  timer.fetch              = TIMER_DIFF(CPU_TREEFETCH);
  timer.stack              = TIMER_DIFF(CPU_TREESTACK);
  timer.all                = timer.tree + timer.wait + timer.fetch + timer.stack + TIMER_DIFF(CPU_TREE);
  timer.lastpm             = All.CPUForLastPMExecution;
  timer.costtotal          = interactioncountPP + interactioncountPN;
  timer.numnodes           = NumNodes;
  timer.interactioncountPP = interactioncountPP;
  timer.interactioncountPN = interactioncountPN;
  timer.NumForeignNodes    = NumForeignNodes;
  timer.NumForeignPoints   = NumForeignPoints;
  timer.fillfacFgnNodes    = NumForeignNodes / ((double)MaxForeignNodes);
  timer.fillfacFgnPoints   = NumForeignPoints / ((double)MaxForeignPoints);

  MPI_Reduce((double *)&timer, (double *)&tisum, (int)(sizeof(detailed_timings) / sizeof(double)), MPI_DOUBLE, MPI_SUM, 0,
             D->Communicator);
  MPI_Reduce((double *)&timer, (double *)&timax, (int)(sizeof(detailed_timings) / sizeof(double)), MPI_DOUBLE, MPI_MAX, 0,
             D->Communicator);

  All.TotNumOfForces += Tp->TimeBinsGravity.GlobalNActiveParticles;

  if(D->ThisTask == 0)
    {
      fprintf(Logs.FdTimings, "Nf=%9lld  timebin=%d  total-Nf=%lld\n", Tp->TimeBinsGravity.GlobalNActiveParticles, timebin,
              All.TotNumOfForces);
      fprintf(Logs.FdTimings, "   work-load balance: %g   part/sec: raw=%g, effective=%g     ia/part: avg=%g   (%g|%g)\n",
              timax.tree / ((tisum.tree + 1e-20) / D->NTask), Tp->TimeBinsGravity.GlobalNActiveParticles / (tisum.tree + 1.0e-20),
              Tp->TimeBinsGravity.GlobalNActiveParticles / ((timax.tree + 1.0e-20) * D->NTask),
              tisum.costtotal / (Tp->TimeBinsGravity.GlobalNActiveParticles + 1.0e-20),
              tisum.interactioncountPP / (Tp->TimeBinsGravity.GlobalNActiveParticles + 1.0e-20),
              tisum.interactioncountPN / (Tp->TimeBinsGravity.GlobalNActiveParticles + 1.0e-20));
      fprintf(Logs.FdTimings,
              "   maximum number of nodes: %g, filled: %g  NumForeignNodes: max=%g avg=%g fill=%g NumForeignPoints: max=%g avg=%g "
              "fill=%g  cycles=%d\n",
              timax.numnodes, timax.numnodes / MaxNodes, timax.NumForeignNodes, tisum.NumForeignNodes / D->NTask,
              timax.fillfacFgnNodes, timax.NumForeignPoints, tisum.NumForeignPoints / D->NTask, timax.fillfacFgnPoints, max_ncycles);
      fprintf(Logs.FdTimings,
              "   avg times: <all>=%g  <tree>=%g  <wait>=%g  <fetch>=%g  <stack>=%g  "
              "(lastpm=%g) sec\n",
              tisum.all / D->NTask, tisum.tree / D->NTask, tisum.wait / D->NTask, tisum.fetch / D->NTask, tisum.stack / D->NTask,
              tisum.lastpm / D->NTask);
      fprintf(Logs.FdTimings, "   total interaction cost: %g  (imbalance=%g)\n", tisum.costtotal,
              timax.costtotal / (tisum.costtotal / D->NTask));
      myflush(Logs.FdTimings);
    }

  TIMER_STOP(CPU_LOGS);
}

/* make sure that we instantiate the template */
#include "../data/simparticles.h"
template class gravtree<simparticles>;

void gwalk::initialize_cuda_memory() {
    DeviceData h_data;
    CUDA_CHECK(cudaMalloc((void**)&h_data.d_pdats, MaxPart * sizeof(pinfo)));
    CUDA_CHECK(cudaMalloc((void**)&h_data.d_nodes, MaxNodes * sizeof(gravnode)));
    CUDA_CHECK(cudaMalloc((void**)&h_data.d_particles, MaxPart * sizeof(particle_data)));
    
    CUDA_CHECK(cudaMalloc((void**)&d_data, sizeof(DeviceData)));
    CUDA_CHECK(cudaMemcpy(d_data, &h_data, sizeof(DeviceData), cudaMemcpyHostToDevice));
}

void gwalk::cleanup_cuda_memory() {
    // Get device data structure
    DeviceData h_data;
    checkCudaErrors(cudaMemcpy(&h_data, d_data, sizeof(DeviceData), cudaMemcpyDeviceToHost));
    
    // Free device memory
    checkCudaErrors(cudaFree(h_data.d_pdats));
    checkCudaErrors(cudaFree(h_data.d_nodes));
    checkCudaErrors(cudaFree(h_data.d_particles));
    checkCudaErrors(cudaFree(d_data));
}

// Add __host__ __device__ qualifiers to functions that run on both host and device
__host__ __device__ void gwalk::evaluate_particle_particle_interaction(/*params*/) {
    // ...existing implementation...
}

__host__ __device__ void gwalk::gravity_force_interact(/*params*/) {
    // ...existing implementation...
}

// Add kernel launch wrapper function
__global__ void gravity_force_interact_kernel(const pinfo *pdats, int *is, int *nos, char *ptypes, char *no_types, 
                                            unsigned char *shmranks, int *mintopleafnodes, int *committeds, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n)
  {
    gravity_force_interact(pdats[idx], is[idx], nos[idx], ptypes[idx], no_types[idx], 
                          shmranks[idx], mintopleafnodes[idx], committeds[idx]);
  }
}
          next_shmrank  = nop->sibling_shmrank;
          type          = NODE_TYPE_FETCHED_NODE;
        }
      else if(p >= EndOfForeignNodes) /* an imported particle below an imported tree node */
        {
          foreign_gravpoint_data *foreignpoint = get_foreignpointsp(p - EndOfForeignNodes, shmrank);

          next         = foreignpoint->Nextnode;
          next_shmrank = foreignpoint->Nextnode_shmrank;
          type         = NODE_TYPE_FETCHED_PARTICLE;
        }
      else
        {
          /* a pseudo point */
          Terminate(
              "should not happen: p=%d MaxPart=%d MaxNodes=%d  ImportedNodeOffset=%d  EndOfTreePoints=%d  EndOfForeignNodes=%d "
              "shmrank=%d",
              p, MaxPart, MaxNodes, ImportedNodeOffset, EndOfTreePoints, EndOfForeignNodes, shmrank);
        }

      gravity_force_interact(pdat, i, p, ptype, type, shmrank, mintopleafnode, committed);

      p       = next;
      shmrank = next_shmrank;
    }
}

__device__ void gwalk::gravity_force_interact(const pinfo &pdat, int i, int no, char ptype, char no_type, unsigned char shmrank,
                                   int mintopleafnode, int committed)
{
  if(no_type <= NODE_TYPE_FETCHED_PARTICLE)  // we are interacting with a particle
    {
      evaluate_particle_particle_interaction(pdat, no, no_type, shmrank);
    }
  else  // we are interacting with a node
    {
      gravnode *nop = get_nodep(no, shmrank);

      if(nop->not_empty == 0)
        return;

      if(no < MaxPart + MaxNodes)                // we have a top-level node
        if(nop->nextnode >= MaxPart + MaxNodes)  // if the next node is not a top-level, we have a leaf node
          {
            mintopleafnode = no;

#ifdef PRESERVE_SHMEM_BINARY_INVARIANCE
            // if the leaf node is on this shared memory, we have all the data, so we for sure don't need to import anything on this branch
            if(skip_actual_force_computation)
              if(Shmem.GetNodeIDForSimulCommRank[nop->OriginTask] == Shmem.GetNodeIDForSimulCommRank[D->ThisTask])
                return;
#endif
          }

      int openflag = evaluate_particle_node_opening_criterion_and_interaction(pdat, nop);

      if(openflag == NODE_OPEN) /* cell can't be used, need to open it */
        {
          if(nop->cannot_be_opened_locally.load(std::memory_order_acquire))
            {
              // are we in the same shared memory node?
              if(Shmem.GetNodeIDForSimulCommRank[nop->OriginTask] == Shmem.GetNodeIDForSimulCommRank[D->ThisTask])
                {
                  return; // Replace Terminate() with return since we can't terminate from device
                }
              else
                {
                  tree_add_to_fetch_stack(nop, no, shmrank);  // will only add unique copies
                  tree_add_to_work_stack(i, no, shmrank, mintopleafnode);
                }
            }
          else
            {
              int min_buffer_space =
                  min(MaxOnWorkStack - (NumOnWorkStack + NewOnWorkStack), MaxOnFetchStack - NumOnFetchStack);

              if(min_buffer_space >= committed + 8 * TREE_NUM_BEFORE_NODESPLIT)
                gwalk_open_node(pdat, i, ptype, nop, mintopleafnode, committed + 8 * TREE_NUM_BEFORE_NODESPLIT);
              else
                tree_add_to_work_stack(i, no, shmrank, mintopleafnode);
            }
        }
    }
}

// Add kernel launch wrapper function
__global__ void gravity_force_interact_kernel(const pinfo *pdats, int *is, int *nos, char *ptypes, char *no_types, 
                                            unsigned char *shmranks, int *mintopleafnodes, int *committeds, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n)
  {
    gravity_force_interact(pdats[idx], is[idx], nos[idx], ptypes[idx], no_types[idx], 
                          shmranks[idx], mintopleafnodes[idx], committeds[idx]);
  }
}

/*

*/

/*! \brief This function computes the gravitational forces for all active particles.
 *
 * The tree walk is done in two phases: First the local part of the force tree is processed (gravity_primary_loop() ).
 * Whenever an external node is encountered during the walk, this node is saved on a list.
 * This node list along with data about the particles is then exchanged among tasks.
 * In the second phase (gravity_secondary_loop() ) each task now continues the tree walk for
 * the imported particles. Finally the resulting partial forces are send back to the original task
 * and are summed up there to complete the tree force calculation.
 *
 * Particles are only exported to other processors when really needed, thereby allowing a
 * good use of the communication buffer. Every particle is sent at most once to a given processor
 * together with the complete list of relevant tree nodes to be checked on the other task.
 *
 * Particles which drifted into the domain of another task are sent to this task for the force computation.
 * Afterwards the resulting force is sent back to the originating task.
 *
 * In order to improve the work load balancing during a domain decomposition, the work done by each
 * node/particle is measured. The work is measured for the interaction partners (i.e. the nodes or particles)
 * and not for the particles itself that require a force computation. This way, work done for imported
 * particles is accounted for at the task where the work actually incurred. The cost measurement is
 * only done for the "GRAVCOSTLEVELS" highest occupied time bins. The variable #MeasureCostFlag will state whether a
 * measurement is done at the present time step.
 *
 * The tree imbalance can be further reduced using chunking. The particles requiring a force computation
 * are split into chunks of size #Nchunksize. A set of every #Nchunk -th chunk is processed first.
 * Then the process is repeated, processing the next set of chunks. This way the amount of exported particles
 * is more balanced, as communication heavy regions are mixed with less communication intensive regions.
 *
 */

void gwalk::gravity_tree(int timebin)
{
  interactioncountPP = 0;
  interactioncountPN = 0;

  TIMER_STORE;
  TIMER_START(CPU_TREE);

  D->mpi_printf("GRAVTREE: Begin tree force. timebin=%d (presently allocated=%g MB)\n", timebin, Mem.getAllocatedBytesInMB());

#ifdef PMGRID
  set_mesh_factors();
#endif

  TIMER_START(CPU_TREESTACK);

  // Create list of targets (the work queue). There are initially two possible sources of points, local ones, and imported ones.

  NumOnWorkStack         = 0;
  AllocWorkStackBaseLow  = std::max<int>(1.5 * (Tp->NumPart + NumPartImported), TREE_MIN_WORKSTACK_SIZE);
  AllocWorkStackBaseHigh = AllocWorkStackBaseLow + TREE_EXPECTED_CYCLES * TREE_MIN_WORKSTACK_SIZE;
  MaxOnWorkStack         = AllocWorkStackBaseLow;

  WorkStack       = (workstack_data *)Mem.mymalloc("WorkStack", AllocWorkStackBaseHigh * sizeof(workstack_data));
  ResultIndexList = (int *)Mem.mymalloc("ResultIndexList", NumPartImported * sizeof(int));

  for(int i = 0; i < Tp->TimeBinsGravity.NActiveParticles; i++)
    {
      int target = Tp->TimeBinsGravity.ActiveParticleList[i];

      // if we have exported particles, we need to explicitly check whether this particle is among them
      if(NumPartExported > 0)
        {
          MyIntPosType xxb       = Tp->P[target].IntPos[0];
          MyIntPosType yyb       = Tp->P[target].IntPos[1];
          MyIntPosType zzb       = Tp->P[target].IntPos[2];
          MyIntPosType mask      = (((MyIntPosType)1) << (BITS_FOR_POSITIONS - 1));
          unsigned char shiftx   = (BITS_FOR_POSITIONS - 3);
          unsigned char shifty   = (BITS_FOR_POSITIONS - 2);
          unsigned char shiftz   = (BITS_FOR_POSITIONS - 1);
          unsigned char rotation = 0;

          int no = 0;
          while(D->TopNodes[no].Daughter >= 0) /* walk down top tree to find correct leaf */
            {
              unsigned char pix     = (((unsigned char)((xxb & mask) >> (shiftx--))) | ((unsigned char)((yyb & mask) >> (shifty--))) |
                                   ((unsigned char)((zzb & mask) >> (shiftz--))));
              unsigned char subnode = peano_incremental_key(pix, &rotation);
              mask >>= 1;
              no = D->TopNodes[no].Daughter + subnode;
            }

          no       = D->TopNodes[no].Leaf;
          int task = D->TaskOfLeaf[no];

          if(task == D->ThisTask)
            {
              WorkStack[NumOnWorkStack].Target         = target;
              WorkStack[NumOnWorkStack].Node           = MaxPart;
              WorkStack[NumOnWorkStack].ShmRank        = Shmem.Island_ThisTask;
              WorkStack[NumOnWorkStack].MinTopLeafNode = MaxPart + D->NTopnodes;
              NumOnWorkStack++;
            }
        }
      else
        {
          WorkStack[NumOnWorkStack].Target         = target;
          WorkStack[NumOnWorkStack].Node           = MaxPart;
          WorkStack[NumOnWorkStack].ShmRank        = Shmem.Island_ThisTask;
          WorkStack[NumOnWorkStack].MinTopLeafNode = MaxPart + D->NTopnodes;
          NumOnWorkStack++;
        }

      /* let's do a safety check here to protect against accidental use of zero softening lengths */
      int softtype = Tp->P[target].getSofteningClass();
      if(All.ForceSoftening[softtype] == 0)
        Terminate("Particle with ID=%lld of type=%d and softening type=%d was assigned zero softening\n",
                  (long long)Tp->P[target].ID.get(), Tp->P[target].getType(), softtype);
    }

  int ncount = 0;

  for(int i = 0; i < NumPartImported; i++)
    {
#ifndef HIERARCHICAL_GRAVITY
      if(Points[i].ActiveFlag)
#endif
        {
          ResultIndexList[i] = ncount++;

          WorkStack[NumOnWorkStack].Target         = i + ImportedNodeOffset;
          WorkStack[NumOnWorkStack].Node           = MaxPart;
          WorkStack[NumOnWorkStack].ShmRank        = Shmem.Island_ThisTask;
          WorkStack[NumOnWorkStack].MinTopLeafNode = MaxPart + D->NTopnodes;
          NumOnWorkStack++;
        }
    }

#ifdef PRESERVE_SHMEM_BINARY_INVARIANCE
  workstack_data *WorkStackBak = (workstack_data *)Mem.mymalloc("WorkStackBak", NumOnWorkStack * sizeof(workstack_data));
  int NumOnWorkStackBak        = NumOnWorkStack;
  memcpy(WorkStackBak, WorkStack, NumOnWorkStack * sizeof(workstack_data));
#endif

  ResultsActiveImported =
      (resultsactiveimported_data *)Mem.mymalloc_clear("ResultsActiveImported", ncount * sizeof(resultsactiveimported_data));

  /******************************************/
  /* now execute the tree walk calculations */
  /******************************************/

  theta2         = All.ErrTolTheta * All.ErrTolTheta;
  thetamax2      = All.ErrTolThetaMax * All.ErrTolThetaMax;
  errTolForceAcc = All.ErrTolForceAcc;

  sum_NumForeignNodes  = 0;
  sum_NumForeignPoints = 0;

  // set a default size of the fetch stack equal to half the work stack (this may still be somewhat too large)
  MaxOnFetchStack = std::max<int>(0.1 * (Tp->NumPart + NumPartImported), TREE_MIN_WORKSTACK_SIZE);
  StackToFetch    = (fetch_data *)Mem.mymalloc_movable(&StackToFetch, "StackToFetch", MaxOnFetchStack * sizeof(fetch_data));

  // let's grab at most half the still available memory for imported points and nodes
  int nspace = (0.5 * Mem.FreeBytes) / (sizeof(gravnode) + 8 * sizeof(foreign_gravpoint_data));

  MaxForeignNodes  = nspace;
  MaxForeignPoints = 8 * nspace;
  NumForeignNodes  = 0;
  NumForeignPoints = 0;

  /* the following two arrays hold imported tree nodes and imported points to augment the local tree */
  Foreign_Nodes  = (gravnode *)Mem.mymalloc_movable(&Foreign_Nodes, "Foreign_Nodes", MaxForeignNodes * sizeof(gravnode));
  Foreign_Points = (foreign_gravpoint_data *)Mem.mymalloc_movable(&Foreign_Points, "Foreign_Points",
                                                                  MaxForeignPoints * sizeof(foreign_gravpoint_data));

  tree_initialize_leaf_node_access_info();

  TIMER_STOP(CPU_TREESTACK);

  double t0       = Logs.second();
  int max_ncycles = 0;

  prepare_shared_memory_access();

#ifdef PRESERVE_SHMEM_BINARY_INVARIANCE
  for(int rep = 0; rep < 2; rep++)
    {
      if(rep == 0)
        {
          skip_actual_force_computation = true;
        }
      else
        {
          skip_actual_force_computation = false;
          NumOnWorkStack                = NumOnWorkStackBak;
          memcpy(WorkStack, WorkStackBak, NumOnWorkStack * sizeof(workstack_data));
        }
#endif

      while(NumOnWorkStack > 0)  // repeat until we are out of work
        {
          NewOnWorkStack  = 0;  // gives the new entries
          NumOnFetchStack = 0;
          MaxOnWorkStack  = std::min<int>(AllocWorkStackBaseLow + max_ncycles * TREE_MIN_WORKSTACK_SIZE, AllocWorkStackBaseHigh);

          TIMER_START(CPU_TREEWALK);

          int item = 0;

          while(item < NumOnWorkStack)
            {
              int committed = 8 * TREE_NUM_BEFORE_NODESPLIT;
              int min_buffer_space =
                  std::min<int>(MaxOnWorkStack - (NumOnWorkStack + NewOnWorkStack), MaxOnFetchStack - NumOnFetchStack);
              if(min_buffer_space >= committed)
                {
                  int target     = WorkStack[item].Target;
                  int no         = WorkStack[item].Node;
                  int shmrank    = WorkStack[item].ShmRank;
                  int mintopleaf = WorkStack[item].MinTopLeafNode;
                  item++;

                  pinfo pdat;
                  int ptype = get_pinfo(target, pdat);

                  if(no == MaxPart)
                    {
                      // we have a pristine particle that's processed for the first time
                      gravity_force_interact(pdat, target, no, ptype, NODE_TYPE_LOCAL_NODE, shmrank, mintopleaf, committed);
                    }
                  else
                    {
                      // we have a node that we previously could not open
                      gravnode *nop = get_nodep(no, shmrank);

                      if(nop->cannot_be_opened_locally)
                        {
                          Terminate("item=%d:  no=%d  now we should be able to open it!", item, no);
                        }
                      else
                        gwalk_open_node(pdat, target, ptype, nop, mintopleaf, committed);
                    }
                }
              else
                break;
            }

          if(item == 0 && NumOnWorkStack > 0)
            Terminate("Can't even process a single particle");

          TIMER_STOP(CPU_TREEWALK);

          TIMER_START(CPU_TREEFETCH);

          tree_fetch_foreign_nodes(FETCH_GRAVTREE);

          TIMER_STOP(CPU_TREEFETCH);

          TIMER_START(CPU_TREESTACK);

          /* now reorder the workstack such that we are first going to do residual pristine particles, and then
           * imported nodes that hang below the first leaf nodes */
          NumOnWorkStack = NumOnWorkStack - item + NewOnWorkStack;
          memmove(WorkStack, WorkStack + item, NumOnWorkStack * sizeof(workstack_data));

          /* now let's sort such that we can go deep on top-level node branches, allowing us to clear them out eventually */
          mycxxsort(WorkStack, WorkStack + NumOnWorkStack, compare_workstack);

          TIMER_STOP(CPU_TREESTACK);

          max_ncycles++;
        }

#ifdef PRESERVE_SHMEM_BINARY_INVARIANCE
    }
#endif

  TIMER_START(CPU_TREEIMBALANCE);

  MPI_Allreduce(MPI_IN_PLACE, &max_ncycles, 1, MPI_INT, MPI_MAX, D->Communicator);

  TIMER_STOP(CPU_TREEIMBALANCE);

  cleanup_shared_memory_access();

  /* free temporary buffers */

  Mem.myfree(Foreign_Points);
  Mem.myfree(Foreign_Nodes);
  Mem.myfree(StackToFetch);

  double t1 = Logs.second();

  D->mpi_printf("GRAVTREE: tree-forces are calculated, with %d cycles took %g sec\n", max_ncycles, Logs.timediff(t0, t1));

  /* now communicate the forces in ResultsActiveImported */
  gravity_exchange_forces();

  Mem.myfree(ResultsActiveImported);
#ifdef PRESERVE_SHMEM_BINARY_INVARIANCE
  Mem.myfree(WorkStackBak);
#endif
  Mem.myfree(ResultIndexList);
  Mem.myfree(WorkStack);

  TIMER_STOP(CPU_TREE);

  D->mpi_printf("GRAVTREE: tree-force is done.\n");

  /*  gather some diagnostic information */

  TIMER_START(CPU_LOGS);

  struct detailed_timings
  {
    double tree, wait, fetch, stack, all, lastpm;
    double costtotal, numnodes;
    double interactioncountPP, interactioncountPN;
    double NumForeignNodes, NumForeignPoints;
    double fillfacFgnNodes, fillfacFgnPoints;
  };
  detailed_timings timer, tisum, timax;

  timer.tree               = TIMER_DIFF(CPU_TREEWALK);
  timer.wait               = TIMER_DIFF(CPU_TREEIMBALANCE);
  timer.fetch              = TIMER_DIFF(CPU_TREEFETCH);
  timer.stack              = TIMER_DIFF(CPU_TREESTACK);
  timer.all                = timer.tree + timer.wait + timer.fetch + timer.stack + TIMER_DIFF(CPU_TREE);
  timer.lastpm             = All.CPUForLastPMExecution;
  timer.costtotal          = interactioncountPP + interactioncountPN;
  timer.numnodes           = NumNodes;
  timer.interactioncountPP = interactioncountPP;
  timer.interactioncountPN = interactioncountPN;
  timer.NumForeignNodes    = NumForeignNodes;
  timer.NumForeignPoints   = NumForeignPoints;
  timer.fillfacFgnNodes    = NumForeignNodes / ((double)MaxForeignNodes);
  timer.fillfacFgnPoints   = NumForeignPoints / ((double)MaxForeignPoints);

  MPI_Reduce((double *)&timer, (double *)&tisum, (int)(sizeof(detailed_timings) / sizeof(double)), MPI_DOUBLE, MPI_SUM, 0,
             D->Communicator);
  MPI_Reduce((double *)&timer, (double *)&timax, (int)(sizeof(detailed_timings) / sizeof(double)), MPI_DOUBLE, MPI_MAX, 0,
             D->Communicator);

  All.TotNumOfForces += Tp->TimeBinsGravity.GlobalNActiveParticles;

  if(D->ThisTask == 0)
    {
      fprintf(Logs.FdTimings, "Nf=%9lld  timebin=%d  total-Nf=%lld\n", Tp->TimeBinsGravity.GlobalNActiveParticles, timebin,
              All.TotNumOfForces);
      fprintf(Logs.FdTimings, "   work-load balance: %g   part/sec: raw=%g, effective=%g     ia/part: avg=%g   (%g|%g)\n",
              timax.tree / ((tisum.tree + 1e-20) / D->NTask), Tp->TimeBinsGravity.GlobalNActiveParticles / (tisum.tree + 1.0e-20),
              Tp->TimeBinsGravity.GlobalNActiveParticles / ((timax.tree + 1.0e-20) * D->NTask),
              tisum.costtotal / (Tp->TimeBinsGravity.GlobalNActiveParticles + 1.0e-20),
              tisum.interactioncountPP / (Tp->TimeBinsGravity.GlobalNActiveParticles + 1.0e-20),
              tisum.interactioncountPN / (Tp->TimeBinsGravity.GlobalNActiveParticles + 1.0e-20));
      fprintf(Logs.FdTimings,
              "   maximum number of nodes: %g, filled: %g  NumForeignNodes: max=%g avg=%g fill=%g NumForeignPoints: max=%g avg=%g "
              "fill=%g  cycles=%d\n",
              timax.numnodes, timax.numnodes / MaxNodes, timax.NumForeignNodes, tisum.NumForeignNodes / D->NTask,
              timax.fillfacFgnNodes, timax.NumForeignPoints, tisum.NumForeignPoints / D->NTask, timax.fillfacFgnPoints, max_ncycles);
      fprintf(Logs.FdTimings,
              "   avg times: <all>=%g  <tree>=%g  <wait>=%g  <fetch>=%g  <stack>=%g  "
              "(lastpm=%g) sec\n",
              tisum.all / D->NTask, tisum.tree / D->NTask, tisum.wait / D->NTask, tisum.fetch / D->NTask, tisum.stack / D->NTask,
              tisum.lastpm / D->NTask);
      fprintf(Logs.FdTimings, "   total interaction cost: %g  (imbalance=%g)\n", tisum.costtotal,
              timax.costtotal / (tisum.costtotal / D->NTask));
      myflush(Logs.FdTimings);
    }

  TIMER_STOP(CPU_LOGS);
}

/* make sure that we instantiate the template */
#include "../data/simparticles.h"
template class gravtree<simparticles>;

void gwalk::initialize_cuda_memory() {
    DeviceData h_data;
    CUDA_CHECK(cudaMalloc((void**)&h_data.d_pdats, MaxPart * sizeof(pinfo)));
    CUDA_CHECK(cudaMalloc((void**)&h_data.d_nodes, MaxNodes * sizeof(gravnode)));
    CUDA_CHECK(cudaMalloc((void**)&h_data.d_particles, MaxPart * sizeof(particle_data)));
    
    CUDA_CHECK(cudaMalloc((void**)&d_data, sizeof(DeviceData)));
    CUDA_CHECK(cudaMemcpy(d_data, &h_data, sizeof(DeviceData), cudaMemcpyHostToDevice));
}

void gwalk::cleanup_cuda_memory() {
    // Get device data structure
    DeviceData h_data;
    checkCudaErrors(cudaMemcpy(&h_data, d_data, sizeof(DeviceData), cudaMemcpyDeviceToHost));
    
    // Free device memory
    checkCudaErrors(cudaFree(h_data.d_pdats));
    checkCudaErrors(cudaFree(h_data.d_nodes));
    checkCudaErrors(cudaFree(h_data.d_particles));
    checkCudaErrors(cudaFree(d_data));
}

// Add __host__ __device__ qualifiers to functions that run on both host and device
__host__ __device__ void gwalk::evaluate_particle_particle_interaction(/*params*/) {
    // ...existing implementation...
}

__host__ __device__ void gwalk::gravity_force_interact(/*params*/) {
    // ...existing implementation...
}

// Add kernel launch wrapper function
__global__ void gravity_force_interact_kernel(const pinfo *pdats, int *is, int *nos, char *ptypes, char *no_types, 
                                            unsigned char *shmranks, int *mintopleafnodes, int *committeds, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n)
  {
    gravity_force_interact(pdats[idx], is[idx], nos[idx], ptypes[idx], no_types[idx], 
                          shmranks[idx], mintopleafnodes[idx], committeds[idx]);
  }
}
