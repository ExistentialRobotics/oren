#include "erl_geometry/find_voxel_indices.cuh"
#include "erl_geometry/find_voxel_indices_torch.cuh"

#include <c10/cuda/CUDAException.h>

#define CHECK_NONEMPTY(x)   TORCH_CHECK((x).numel() > 0, #x " must be non-empty")
#define CHECK_CUDA(x)       TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_LONG(x)       TORCH_CHECK((x).scalar_type() == torch::kInt64, #x " must be a Long tensor")
#define WARP_SIZE           32l
#define MAX_BLOCK_SIZE      256l

namespace erl::geometry {

    void
    FindVoxelIndicesTorchCUDA(
        const torch::Tensor &codes,
        const int dims,
        const int n_levels,
        const torch::Tensor &children,
        torch::Tensor &voxel_indices) {

        CHECK_NONEMPTY(codes);
        CHECK_CUDA(codes);
        CHECK_CONTIGUOUS(codes);
        TORCH_CHECK(
            codes.scalar_type() == torch::kUInt32 || codes.scalar_type() == torch::kUInt64,
            "codes must be of type UInt32 or UInt64.");
        TORCH_CHECK(dims == 2 || dims == 3, "dims must be 2 or 3.");

        CHECK_NONEMPTY(children);
        CHECK_CUDA(children);
        CHECK_CONTIGUOUS(children);

        TORCH_CHECK(
            codes.device().index() == children.device().index(),
            "codes and children must be on the same device.");

        const int64_t b = codes.numel();
        const int64_t warps = (b + WARP_SIZE - 1) / WARP_SIZE;
        const int64_t threads = std::min(MAX_BLOCK_SIZE, warps * WARP_SIZE);
        const int64_t blocks = (b + threads - 1) / threads;

        TORCH_CHECK(
            children.dim() == 2 && children.size(1) == (1 << dims),
            "children must be of shape (N, 2^dims).");

        voxel_indices = torch::empty(
            codes.sizes(),
            torch::TensorOptions().dtype(children.scalar_type()).device(codes.device()));

        switch (codes.scalar_type()) {
            case torch::kUInt32: {
                TORCH_CHECK(dims * n_levels <= 32, "For UInt32 codes, max level is 32 / dims.");
                if (dims == 2) {
                    AT_DISPATCH_INTEGRAL_TYPES(
                        children.scalar_type(),
                        "find_voxel_indices_2d_uint32",
                        [&] {
                            FindVoxelIndicesKernel<scalar_t, uint32_t, 2><<<blocks, threads>>>(
                                codes.data_ptr<uint32_t>(),
                                n_levels,
                                children.data_ptr<scalar_t>(),
                                voxel_indices.data_ptr<scalar_t>(),
                                b);
                        });
                } else {
                    AT_DISPATCH_INTEGRAL_TYPES(
                        children.scalar_type(),
                        "find_voxel_indices_3d_uint32",
                        [&] {
                            FindVoxelIndicesKernel<scalar_t, uint32_t, 3><<<blocks, threads>>>(
                                codes.data_ptr<uint32_t>(),
                                n_levels,
                                children.data_ptr<scalar_t>(),
                                voxel_indices.data_ptr<scalar_t>(),
                                b);
                        });
                }
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                break;
            }
            case torch::kUInt64: {
                TORCH_CHECK(dims * n_levels <= 64, "For UInt64 codes, max level is 64 / dims.");
                if (dims == 2) {
                    AT_DISPATCH_INTEGRAL_TYPES(
                        children.scalar_type(),
                        "find_voxel_indices_2d_uint64",
                        [&] {
                            FindVoxelIndicesKernel<scalar_t, uint64_t, 2><<<blocks, threads>>>(
                                codes.data_ptr<uint64_t>(),
                                n_levels,
                                children.data_ptr<scalar_t>(),
                                voxel_indices.data_ptr<scalar_t>(),
                                b);
                        });
                } else {
                    AT_DISPATCH_INTEGRAL_TYPES(
                        children.scalar_type(),
                        "find_voxel_indices_3d_uint64",
                        [&] {
                            FindVoxelIndicesKernel<scalar_t, uint64_t, 3><<<blocks, threads>>>(
                                codes.data_ptr<uint64_t>(),
                                n_levels,
                                children.data_ptr<scalar_t>(),
                                voxel_indices.data_ptr<scalar_t>(),
                                b);
                        });
                }
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                break;
            }
            default: {
                AT_ERROR("Unsupported data type of codes for FindVoxelIndicesTorchCUDA.");
            }
        }
    }

}  // namespace erl::geometry
