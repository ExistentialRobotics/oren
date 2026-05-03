#ifdef ERL_USE_LIBTORCH
    #include "erl_geometry/find_voxel_indices_torch.hpp"

    #include "erl_geometry/find_voxel_indices.hpp"
    #include "erl_geometry/find_voxel_indices_torch.cuh"

    #define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

namespace erl::geometry {
    void
    FindVoxelIndicesTorch(
        const torch::Tensor &codes,
        const int dims,
        const int n_levels,
        const torch::Tensor &children,
        const bool parallel,
        torch::Tensor &voxel_indices) {
        if (codes.is_cuda() || children.is_cuda()) {
            FindVoxelIndicesTorchCUDA(codes, dims, n_levels, children, voxel_indices);
            return;
        }
        if (!codes.is_cpu()) { AT_ERROR("Unsupported device of codes for FindVoxelIndicesTorch."); }
        if (!children.is_cpu()) {
            AT_ERROR("Unsupported device of children for FindVoxelIndicesTorch.");
        }

        CHECK_CONTIGUOUS(codes);
        TORCH_CHECK(dims == 2 || dims == 3, "dims must be 2 or 3.");
        TORCH_CHECK(
            children.dim() == 2 && children.size(1) == (1 << dims),
            "children must be of shape (N, 2^dims).");
        CHECK_CONTIGUOUS(children);

        voxel_indices =
            torch::empty(codes.sizes(), children.options().dtype(children.scalar_type()));

        switch (codes.scalar_type()) {
            case torch::kUInt32: {
                TORCH_CHECK(dims * n_levels <= 32, "For UInt32 codes, max level is 32 / dims.");
                if (dims == 2) {
                    AT_DISPATCH_INTEGRAL_TYPES(
                        children.scalar_type(),
                        "find_voxel_indices_2d_uint32",
                        [&] {
                            FindVoxelIndices<scalar_t, uint32_t, 2>(
                                codes.data_ptr<uint32_t>(),
                                codes.numel(),
                                n_levels,
                                children.data_ptr<scalar_t>(),
                                voxel_indices.data_ptr<scalar_t>(),
                                parallel);
                        });
                } else {
                    AT_DISPATCH_INTEGRAL_TYPES(
                        children.scalar_type(),
                        "find_voxel_indices_3d_uint32",
                        [&] {
                            FindVoxelIndices<scalar_t, uint32_t, 3>(
                                codes.data_ptr<uint32_t>(),
                                codes.numel(),
                                n_levels,
                                children.data_ptr<scalar_t>(),
                                voxel_indices.data_ptr<scalar_t>(),
                                parallel);
                        });
                }
                break;
            }
            case torch::kUInt64: {
                TORCH_CHECK(dims * n_levels <= 64, "For UInt64 codes, max level is 64 / dims.");
                if (dims == 2) {
                    AT_DISPATCH_INTEGRAL_TYPES(
                        children.scalar_type(),
                        "find_voxel_indices_2d_uint32",
                        [&] {
                            FindVoxelIndices<scalar_t, uint64_t, 2>(
                                codes.data_ptr<uint64_t>(),
                                codes.numel(),
                                n_levels,
                                children.data_ptr<scalar_t>(),
                                voxel_indices.data_ptr<scalar_t>(),
                                parallel);
                        });
                } else {
                    AT_DISPATCH_INTEGRAL_TYPES(
                        children.scalar_type(),
                        "find_voxel_indices_3d_uint32",
                        [&] {
                            FindVoxelIndices<scalar_t, uint64_t, 3>(
                                codes.data_ptr<uint64_t>(),
                                codes.numel(),
                                n_levels,
                                children.data_ptr<scalar_t>(),
                                voxel_indices.data_ptr<scalar_t>(),
                                parallel);
                        });
                }
                break;
            }
            default: {
                AT_ERROR("Unsupported data type of codes for FindVoxelIndicesTorch.");
            }
        }
    }

}  // namespace erl::geometry

#endif
