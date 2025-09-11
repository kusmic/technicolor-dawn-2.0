#ifndef GWALK_CUDA_TYPES_H
#define GWALK_CUDA_TYPES_H

#include "../gravtree/gravtree.h"
#include "../data/simparticles.h"

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

#endif
