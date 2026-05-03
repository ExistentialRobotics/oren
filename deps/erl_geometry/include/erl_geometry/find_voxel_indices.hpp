#pragma once

#include <cstdint>

namespace erl::geometry {

    /**
     * Find the index of a voxel in a tree given its morton code.
     * @tparam IndexType index type of the voxel, usually int32_t or int64_t.
     * @tparam MortonType type of the morton code, usually uint32_t or uint64_t.
     * @tparam Dim space dimension, 2 for quadtree, 3 for octree.
     * @param code the morton code of the voxel to find.
     * @param n_levels the number of levels to search from the root.
     * @param children the buffer that stores the child voxel indices of each voxel. The buffer
     * should be an (N, 8) matrix organized in a row-major order. Each row corresponds to a voxel,
     * and the 8 columns * correspond to the indices of its 8 children. If a child does not exist,
     * its index should be set to -1.
     * @return the index of the found voxel, or -1 if children is nullptr or the voxel is not found.
     */
    template<typename IndexType, typename MortonType, int Dim>
    IndexType
    FindVoxelIndex(const MortonType code, int n_levels, const IndexType *children) {
        if (children == nullptr) { return -1; }

        uint64_t shift = n_levels * Dim;
        uint64_t mask = ((1ul << Dim) - 1ul) << shift;
        IndexType index = 0;

        while (n_levels >= 0) {
            if (const auto child_index =
                    static_cast<IndexType>((code & mask) >> shift) + (index << Dim);
                children[child_index] >= 0) {
                index = children[child_index];
            } else {
                return index;
            }
            --n_levels;
            shift -= Dim;
            mask >>= Dim;
        }
        return index;
    }

    template<typename IndexType, typename MortonType, int Dim>
    void
    FindVoxelIndices(
        const MortonType *codes,
        std::size_t num_codes,
        int n_levels,
        const IndexType *children,
        IndexType *voxel_indices,
        bool parallel) {

#pragma omp parallel if (parallel) default(none) \
    shared(num_codes, codes, n_levels, children, voxel_indices)
        for (std::size_t i = 0; i < num_codes; ++i) {
            voxel_indices[i] =
                FindVoxelIndex<IndexType, MortonType, Dim>(codes[i], n_levels, children);
        }
    }

}  // namespace erl::geometry
