#pragma once

#include <string>
#include <unordered_map>

#include <torch/torch.h>

#include "grad_sdf_cpp/config.hpp"

namespace grad_sdf_cpp {

struct OctreeForwardResult {
    torch::Tensor voxel_indices;
    torch::Tensor sdf_prior;
    c10::optional<torch::Tensor> residual_features;
};

class OctreeModuleImpl : public torch::nn::Module {
public:
    explicit OctreeModuleImpl(const OctreeConfig& cfg);

    OctreeForwardResult forward(
        const torch::Tensor& points,
        const c10::optional<torch::Tensor>& voxel_indices = c10::nullopt);

    torch::Tensor find_voxel_indices(
        const torch::Tensor& points,
        bool are_voxels,
        int64_t level = 1) const;

    void LoadFromTensorMap(
        const std::unordered_map<std::string, torch::Tensor>& tensor_map,
        const torch::Device& device);

    [[nodiscard]] torch::Device device() const {
        return sdf_priors_.device();
    }

private:
    torch::Tensor points_to_voxels(const torch::Tensor& points) const;

    static torch::Tensor RequireTensor(
        const std::unordered_map<std::string, torch::Tensor>& tensor_map,
        const std::string& key);

    OctreeConfig cfg_;
    int64_t key_offset_ = 0;

    torch::Tensor sdf_priors_;
    torch::Tensor grad_priors_;
    torch::Tensor residual_features_;

    torch::Tensor voxels_;
    torch::Tensor voxel_centers_;
    torch::Tensor vertex_indices_;
    torch::Tensor structure_;

    bool has_residual_features_ = false;
    bool little_endian_vertex_order_ = true;
};

TORCH_MODULE(OctreeModule);

}  // namespace grad_sdf_cpp
