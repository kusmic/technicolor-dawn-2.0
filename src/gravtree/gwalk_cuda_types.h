#ifndef GWALK_CUDA_TYPES_H
#define GWALK_CUDA_TYPES_H

#include "../data/dtypes.h"
#include "../data/constants.h"
#include "../logs/timer.h"

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

// Constants needed by CUDA code
enum NodeType {
    NODE_TYPE_LOCAL_NODE,
    NODE_TYPE_TREEPOINT_PARTICLE,
    NODE_TYPE_LOCAL_PARTICLE,
    NODE_TYPE_FETCHED_NODE,
    NODE_TYPE_FETCHED_PARTICLE
};

// Forward declarations
class gravtree;
struct gravnode;
struct particle_data;
struct foreign_gravpoint_data;
struct workstack_data;
struct fetch_data;

#endif
