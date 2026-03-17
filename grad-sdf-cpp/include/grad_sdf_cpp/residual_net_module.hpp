#pragma once

#include <string>
#include <unordered_map>

#include <torch/torch.h>

#include "grad_sdf_cpp/config.hpp"

namespace grad_sdf_cpp {

class ResidualNetModuleImpl : public torch::nn::Module {
public:
    explicit ResidualNetModuleImpl(const ResidualNetConfig& cfg);

    torch::Tensor forward(const torch::Tensor& residual_features);

    void LoadFromTensorMap(
        const std::unordered_map<std::string, torch::Tensor>& tensor_map,
        const torch::Device& device);

    [[nodiscard]] const ResidualNetConfig& config() const {
        return cfg_;
    }

private:
    torch::Tensor Activate(const torch::Tensor& x) const;
    static torch::Tensor RequireTensor(
        const std::unordered_map<std::string, torch::Tensor>& tensor_map,
        const std::string& key);

    ResidualNetConfig cfg_;
    std::vector<torch::nn::Linear> linears_;
};

TORCH_MODULE(ResidualNetModule);

}  // namespace grad_sdf_cpp
