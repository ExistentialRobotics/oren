#include "grad_sdf_cpp/octree_module.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

namespace grad_sdf_cpp
{
    namespace
    {

        using torch::indexing::Slice;

        std::pair<torch::Tensor, torch::Tensor> GaTrilinear(
            const torch::Tensor &points,
            const torch::Tensor &voxel_centers,
            const torch::Tensor &voxel_sizes,
            const torch::Tensor &vertex_values,
            const torch::Tensor &vertex_grad,
            double resolution,
            bool gradient_augmentation,
            bool little_endian_vertex_order);

        torch::Tensor TrilinearInterpolation(
            const torch::Tensor &points,
            const torch::Tensor &per_point_vertex_values,
            bool little_endian_vertex_order);

        torch::Tensor VertexOffsets(const torch::TensorOptions &options, bool little_endian_vertex_order)
        {
            auto offsets = torch::tensor(
                {{{-1.0F, -1.0F, -1.0F},
                  {-1.0F, -1.0F, 1.0F},
                  {-1.0F, 1.0F, -1.0F},
                  {-1.0F, 1.0F, 1.0F},
                  {1.0F, -1.0F, -1.0F},
                  {1.0F, -1.0F, 1.0F},
                  {1.0F, 1.0F, -1.0F},
                  {1.0F, 1.0F, 1.0F}}},
                options);

            if (little_endian_vertex_order)
            {
                auto perm = torch::tensor({2, 1, 0}, torch::TensorOptions().dtype(torch::kLong).device(options.device()));
                offsets = offsets.index({Slice(), Slice(), perm});
            }

            return offsets;
        }

        torch::Tensor GetVertices(
            const torch::Tensor &voxel_centers,
            const torch::Tensor &voxel_sizes,
            double resolution,
            bool little_endian_vertex_order)
        {
            auto offsets = VertexOffsets(voxel_centers.options(), little_endian_vertex_order);
            auto half_sizes = (voxel_sizes * 0.5).view({-1, 1, 1}) * offsets;
            return voxel_centers.view({-1, 1, 3}) + half_sizes * resolution;
        }

        torch::Tensor TrilinearInterpolation(
            const torch::Tensor &points,
            const torch::Tensor &per_point_vertex_values,
            bool little_endian_vertex_order)
        {
            auto offsets1 = VertexOffsets(points.options(), little_endian_vertex_order);
            auto offsets2 = offsets1 * 0.5 + 0.5;
            auto offsets3 = 1.0 - offsets2;

            auto points_unsqueezed = points.unsqueeze(1);
            auto weights = (offsets3 + points_unsqueezed * offsets1).prod(-1);

            const auto n_points = points.size(0);
            auto values_flat = per_point_vertex_values.reshape({n_points, 8, -1});
            auto interpolated = (weights.unsqueeze(-1) * values_flat).sum(1);

            if (per_point_vertex_values.dim() == 2)
            {
                return interpolated.squeeze(-1);
            }

            std::vector<int64_t> out_shape;
            out_shape.reserve(per_point_vertex_values.dim() - 1);
            out_shape.push_back(n_points);
            for (int64_t d = 2; d < per_point_vertex_values.dim(); ++d)
            {
                out_shape.push_back(per_point_vertex_values.size(d));
            }
            return interpolated.reshape(out_shape);
        }

        std::pair<torch::Tensor, torch::Tensor> GaTrilinear(
            const torch::Tensor &points,
            const torch::Tensor &voxel_centers,
            const torch::Tensor &voxel_sizes,
            const torch::Tensor &vertex_values,
            const torch::Tensor &vertex_grad,
            double resolution,
            bool gradient_augmentation,
            bool little_endian_vertex_order)
        {
            torch::Tensor per_point_vertex_values;

            auto vertices = GetVertices(voxel_centers, voxel_sizes, resolution, little_endian_vertex_order);

            if (gradient_augmentation)
            {
                auto diffs = points.unsqueeze(1) - vertices;
                auto projection = (vertex_grad * diffs).sum(-1);
                per_point_vertex_values = vertex_values + projection;
            }
            else
            {
                per_point_vertex_values = vertex_values;
            }

            auto p = (points - voxel_centers) / (voxel_sizes * resolution) + 0.5;
            auto results = TrilinearInterpolation(p, per_point_vertex_values, little_endian_vertex_order);
            return {results, p};
        }

    } // namespace

    OctreeModuleImpl::OctreeModuleImpl(const OctreeConfig &cfg)
        : cfg_(cfg)
    {
        key_offset_ = int64_t{1} << (cfg_.tree_depth - 1);

        auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
        auto long_options = torch::TensorOptions().dtype(torch::kInt64);

        sdf_priors_ = register_parameter(
            "sdf_priors",
            torch::zeros({cfg_.init_voxel_num}, float_options),
            true);

        grad_priors_ = register_parameter(
            "grad_priors",
            torch::zeros({cfg_.init_voxel_num, 3}, float_options),
            true);

        has_residual_features_ = cfg_.residual_feature_dim > 0;
        if (has_residual_features_)
        {
            residual_features_ = register_parameter(
                "residual_features",
                torch::zeros({cfg_.init_voxel_num, cfg_.residual_feature_dim}, float_options),
                true);
        }

        voxels_ = register_buffer("voxels", torch::zeros({cfg_.init_voxel_num, 4}, long_options));
        voxel_centers_ = register_buffer("voxel_centers", torch::zeros({cfg_.init_voxel_num, 3}, float_options));
        vertex_indices_ = register_buffer("vertex_indices", torch::zeros({cfg_.init_voxel_num, 8}, long_options));
        structure_ = register_buffer("structure", torch::zeros({cfg_.init_voxel_num, 8}, long_options));
    }

    torch::Tensor OctreeModuleImpl::RequireTensor(
        const std::unordered_map<std::string, torch::Tensor> &tensor_map,
        const std::string &key)
    {
        const auto it = tensor_map.find(key);
        if (it == tensor_map.end())
        {
            throw std::runtime_error("Missing tensor key in bundle: " + key);
        }
        return it->second;
    }

    torch::Tensor OctreeModuleImpl::points_to_voxels(const torch::Tensor &points) const
    {
        auto voxels = torch::floor(points / cfg_.resolution).to(torch::kInt64);
        voxels = voxels + key_offset_;
        return voxels;
    }

    torch::Tensor OctreeModuleImpl::find_voxel_indices(
        const torch::Tensor &points,
        bool are_voxels,
        int64_t level) const
    {
        auto voxels = are_voxels ? points.to(torch::kInt64) : points_to_voxels(points);

        auto voxels_cpu = voxels.to(torch::kCPU, torch::kInt64).contiguous();
        auto structure_cpu = structure_.to(torch::kCPU, torch::kInt64).contiguous();

        const auto n_points = voxels_cpu.size(0);
        auto voxel_indices_cpu = torch::full({n_points}, -1, torch::TensorOptions().dtype(torch::kInt64));

        auto voxels_a = voxels_cpu.accessor<int64_t, 2>();
        auto structure_a = structure_cpu.accessor<int64_t, 2>();
        auto out_a = voxel_indices_cpu.accessor<int64_t, 1>();

        const int64_t max_coord = int64_t{1} << cfg_.tree_depth;
        const int64_t clamped_level = std::max<int64_t>(1, std::min<int64_t>(level, cfg_.tree_depth));
        const int64_t max_steps = cfg_.tree_depth - clamped_level + 1;

        for (int64_t i = 0; i < n_points; ++i)
        {
            const int64_t x = voxels_a[i][0];
            const int64_t y = voxels_a[i][1];
            const int64_t z = voxels_a[i][2];

            if (x < 0 || y < 0 || z < 0 || x >= max_coord || y >= max_coord || z >= max_coord)
            {
                out_a[i] = -1;
                continue;
            }

            int64_t cur_node = 0;
            for (int64_t step = 0; step < max_steps; ++step)
            {
                const int64_t shift = cfg_.tree_depth - 1 - step;
                const int64_t child_id =
                    (((x >> shift) & 1) + (((y >> shift) & 1) << 1) + (((z >> shift) & 1) << 2));
                const int64_t child_idx = structure_a[cur_node][child_id];
                if (child_idx < 0)
                {
                    break;
                }
                cur_node = child_idx;
            }

            out_a[i] = cur_node;
        }

        return voxel_indices_cpu.to(points.device(), /*non_blocking=*/false, /*copy=*/true);
    }

    OctreeForwardResult OctreeModuleImpl::forward(
        const torch::Tensor &points,
        const c10::optional<torch::Tensor> &voxel_indices_opt)
    {
        if (points.numel() == 0)
        {
            OctreeForwardResult empty;
            empty.voxel_indices = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(points.device()));
            empty.sdf_prior = torch::empty({0}, points.options());
            empty.residual_features = c10::nullopt;
            return empty;
        }

        auto points_local = points.to(device(), torch::kFloat32).contiguous();

        torch::Tensor voxel_indices;
        if (voxel_indices_opt.has_value())
        {
            voxel_indices = voxel_indices_opt.value().to(device(), torch::kInt64).contiguous();
        }
        else
        {
            voxel_indices = find_voxel_indices(points_local, false, 1);
        }

        auto valid_mask = voxel_indices.ge(0);
        auto safe_indices = torch::where(valid_mask, voxel_indices, torch::zeros_like(voxel_indices));

        auto voxel_centers = voxel_centers_.index_select(0, safe_indices);
        auto vertex_indices = vertex_indices_.index_select(0, safe_indices);
        auto voxel_sizes =
            voxels_.index_select(0, safe_indices).index({Slice(), 3}).unsqueeze(-1).to(points_local.dtype());

        const auto n_points = points_local.size(0);
        auto flat_vertex_indices = vertex_indices.reshape({-1});

        auto vertex_sdf_priors = sdf_priors_.index_select(0, flat_vertex_indices).reshape({n_points, 8});
        auto vertex_grad_priors = grad_priors_.index_select(0, flat_vertex_indices).reshape({n_points, 8, 3});

        auto ga_result = GaTrilinear(
            points_local,
            voxel_centers,
            voxel_sizes,
            vertex_sdf_priors,
            vertex_grad_priors,
            cfg_.resolution,
            cfg_.gradient_augmentation,
            little_endian_vertex_order_);

        auto sdf_preds = ga_result.first;
        auto p = ga_result.second;
        sdf_preds.masked_fill_(~valid_mask, 0.0);

        OctreeForwardResult result;
        result.voxel_indices = voxel_indices;
        result.sdf_prior = sdf_preds;

        if (!has_residual_features_)
        {
            result.residual_features = c10::nullopt;
            return result;
        }

        auto per_point_vertex_residual_features_level_1 =
            residual_features_.index_select(0, flat_vertex_indices).reshape({n_points, 8, cfg_.residual_feature_dim});

        auto residual_features =
            TrilinearInterpolation(p, per_point_vertex_residual_features_level_1, little_endian_vertex_order_);
        residual_features.masked_fill_(~valid_mask.unsqueeze(-1), 0.0);

        if (cfg_.residual_num_levels > 1)
        {
            for (int level = 2; level <= cfg_.residual_num_levels; ++level)
            {
                auto residual_voxel_indices = find_voxel_indices(points_local, false, level);
                auto level_valid = residual_voxel_indices.ge(0);
                auto level_safe =
                    torch::where(level_valid, residual_voxel_indices, torch::zeros_like(residual_voxel_indices));

                auto residual_voxel_centers = voxel_centers_.index_select(0, level_safe);
                auto residual_vertex_indices = vertex_indices_.index_select(0, level_safe);
                auto residual_voxel_sizes = voxels_.index_select(0, level_safe)
                                                .index({Slice(), 3})
                                                .unsqueeze(-1)
                                                .to(points_local.dtype());

                auto p_level = (points_local - residual_voxel_centers) / (residual_voxel_sizes * cfg_.resolution) + 0.5;

                auto residual_flat_vertex_indices = residual_vertex_indices.reshape({-1});
                auto per_point_vertex_residual_features_level_n =
                    residual_features_.index_select(0, residual_flat_vertex_indices)
                        .reshape({n_points, 8, cfg_.residual_feature_dim});

                auto residual_features_level_n = TrilinearInterpolation(
                    p_level,
                    per_point_vertex_residual_features_level_n,
                    little_endian_vertex_order_);
                residual_features_level_n.masked_fill_(~level_valid.unsqueeze(-1), 0.0);

                residual_features = torch::cat({residual_features, residual_features_level_n}, -1);
            }
        }

        result.residual_features = residual_features;
        return result;
    }

    void OctreeModuleImpl::LoadFromTensorMap(
        const std::unordered_map<std::string, torch::Tensor> &tensor_map,
        const torch::Device &device)
    {
        torch::NoGradGuard no_grad;

        auto LoadAndMaybeResizeTensor = [this, device](
                                            torch::Tensor &param,
                                            const std::string &key,
                                            const std::unordered_map<std::string, torch::Tensor> &map,
                                            torch::ScalarType dtype)
        {
            auto loaded = RequireTensor(map, key).to(device, dtype).contiguous();

            // If the loaded tensor has a different first dimension, we need to handle it
            if (loaded.size(0) != param.size(0))
            {
                auto loaded_size = loaded.size(0);
                auto param_size = param.size(0);

                if (loaded_size < param_size)
                {
                    // Pad the loaded tensor to match param size
                    std::vector<int64_t> pad_shape = loaded.sizes().vec();
                    pad_shape[0] = param_size;
                    auto padded = torch::zeros(pad_shape, loaded.options());
                    padded.slice(0, 0, loaded_size).copy_(loaded);
                    param.set_data(padded);
                }
                else
                {
                    // Truncate to match param size
                    param.set_data(loaded.slice(0, 0, param_size));
                }
            }
            else
            {
                param.set_data(loaded);
            }
        };

        LoadAndMaybeResizeTensor(sdf_priors_, "octree.sdf_priors", tensor_map, torch::kFloat32);
        LoadAndMaybeResizeTensor(grad_priors_, "octree.grad_priors", tensor_map, torch::kFloat32);

        if (has_residual_features_)
        {
            LoadAndMaybeResizeTensor(residual_features_, "octree.residual_features", tensor_map, torch::kFloat32);
        }

        LoadAndMaybeResizeTensor(voxels_, "octree.voxels", tensor_map, torch::kInt64);
        LoadAndMaybeResizeTensor(voxel_centers_, "octree.voxel_centers", tensor_map, torch::kFloat32);
        LoadAndMaybeResizeTensor(vertex_indices_, "octree.vertex_indices", tensor_map, torch::kInt64);
        LoadAndMaybeResizeTensor(structure_, "octree.structure", tensor_map, torch::kInt64);
    }

} // namespace grad_sdf_cpp
