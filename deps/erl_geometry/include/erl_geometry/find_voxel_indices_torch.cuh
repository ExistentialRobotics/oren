#pragma once

#include <torch/torch.h>

namespace erl::geometry {
    void
    FindVoxelIndicesTorchCUDA(
        const torch::Tensor &codes,
        int dims,
        int n_levels,
        const torch::Tensor &children,
        torch::Tensor &voxel_indices);
}
