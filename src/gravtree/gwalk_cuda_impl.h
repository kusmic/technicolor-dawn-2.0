#ifndef GWALK_CUDA_IMPL_H
#define GWALK_CUDA_IMPL_H

#include "gwalk_cuda.cuh"

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
