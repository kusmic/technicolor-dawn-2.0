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
