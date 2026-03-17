#pragma once

#include <string>

#include <torch/torch.h>

#include "grad_sdf_cpp/config.hpp"
#include "grad_sdf_cpp/octree_module.hpp"
#include "grad_sdf_cpp/residual_net_module.hpp"

namespace grad_sdf_cpp {

struct SdfNetworkForwardResult {
    torch::Tensor voxel_indices;
    torch::Tensor sdf_prior;
    c10::optional<torch::Tensor> sdf_residual;
    torch::Tensor sdf;
};

class SdfNetworkModuleImpl : public torch::nn::Module {
public:
    explicit SdfNetworkModuleImpl(const SdfNetworkConfig& cfg);

    SdfNetworkForwardResult forward(
        const torch::Tensor& points,
        const c10::optional<torch::Tensor>& voxel_indices = c10::nullopt);

    void LoadFromBundle(const std::string& bundle_dir);

    [[nodiscard]] const SdfNetworkConfig& config() const {
        return cfg_;
    }

private:
    SdfNetworkConfig cfg_;
    OctreeModule octree_{nullptr};
    ResidualNetModule residual_{nullptr};
};

TORCH_MODULE(SdfNetworkModule);

}  // namespace grad_sdf_cpp
