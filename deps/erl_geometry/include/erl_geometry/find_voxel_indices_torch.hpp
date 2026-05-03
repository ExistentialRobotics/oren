#pragma once

#ifdef ERL_USE_LIBTORCH

    #include <torch/torch.h>

namespace erl::geometry {
    /**
     *
     * @param codes tensor of morton code with dtype torch::kUInt32 or torch::kUInt64
     * @param dims space dimension, 2 or 3
     * @param n_levels the number of levels to search from the root
     * @param children tensor of shape (M, 2^dims) that stores the tree structure.
     * @param parallel if true, use parallel for to speed up the search when using CPU.
     * @param voxel_indices tensor to store the found voxel indices, has the same shape as codes and
     * the same dtype as children.
     */
    void
    FindVoxelIndicesTorch(
        const torch::Tensor &codes,
        int dims,
        int n_levels,
        const torch::Tensor &children,
        bool parallel,
        torch::Tensor &voxel_indices);
}  // namespace erl::geometry

#endif
