#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

namespace erl::geometry {

    template<typename IndexType, typename MortonType, int Dim>
    __global__ void
    FindVoxelIndicesKernel(
        const MortonType *__restrict__ codes,
        int n_levels,
        const IndexType *__restrict__ children,
        IndexType *__restrict__ indices,
        const size_t n) {

        const auto i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) { return; }

        const MortonType code = codes[i];
        uint64_t shift = n_levels * Dim;
        uint64_t mask = ((1ul << Dim) - 1ul) << shift;
        IndexType &index = indices[i];

        index = 0;

        while (n_levels >= 0) {
            const auto child_index = static_cast<int>((code & mask) >> shift) + (index << Dim);
            int child_val = children[child_index];

            // Branchless update: if child_val >= 0, update index; else, keep index unchanged.
            // Use a mask to zero out further updates if invalid.

            const int valid = (child_val >= 0);
            const int invalid = 1 - valid;

            index = valid * child_val + invalid * index;
            shift = valid * (shift - Dim) + invalid * shift;
            mask = valid * (mask >> Dim) + invalid * mask;

            --n_levels;
        }
    }

}  // namespace erl::geometry
