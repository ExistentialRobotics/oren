#include "grad_sdf_cpp/residual_net_module.hpp"

#include <stdexcept>

namespace grad_sdf_cpp
{

    ResidualNetModuleImpl::ResidualNetModuleImpl(const ResidualNetConfig &cfg)
        : cfg_(cfg)
    {
        int in_dim = cfg_.input_feature_dim + 1;

        for (int i = 0; i < cfg_.n_hidden_layers; ++i)
        {
            auto layer = torch::nn::Linear(in_dim, cfg_.hidden_dims);
            register_module("linear_" + std::to_string(i), layer);
            linears_.push_back(layer);
            in_dim = cfg_.hidden_dims;
        }

        auto out_layer = torch::nn::Linear(in_dim, 1);
        register_module("linear_out", out_layer);
        linears_.push_back(out_layer);
    }

    torch::Tensor ResidualNetModuleImpl::Activate(const torch::Tensor &x) const
    {
        if (cfg_.mlp_activation == "ReLU")
        {
            return torch::relu(x);
        }
        if (cfg_.mlp_activation == "LeakyReLU")
        {
            return torch::leaky_relu(x);
        }
        if (cfg_.mlp_activation == "ELU")
        {
            return torch::elu(x);
        }
        if (cfg_.mlp_activation == "GELU")
        {
            return torch::gelu(x);
        }
        throw std::runtime_error("Unsupported activation: " + cfg_.mlp_activation);
    }

    torch::Tensor ResidualNetModuleImpl::forward(const torch::Tensor &residual_features)
    {
        torch::Tensor x = residual_features;

        for (std::size_t i = 0; i + 1 < linears_.size(); ++i)
        {
            x = linears_[i]->forward(x);
            x = Activate(x);
        }

        x = linears_.back()->forward(x);
        x = x * cfg_.output_sdf_scale;
        return x;
    }

    torch::Tensor ResidualNetModuleImpl::RequireTensor(
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

    void ResidualNetModuleImpl::LoadFromTensorMap(
        const std::unordered_map<std::string, torch::Tensor> &tensor_map,
        const torch::Device &device)
    {
        torch::NoGradGuard no_grad;

        for (int i = 0; i < cfg_.n_hidden_layers; ++i)
        {
            const int py_idx = 2 * i;
            const std::string w_key = "residual.residual_net." + std::to_string(py_idx) + ".weight";
            const std::string b_key = "residual.residual_net." + std::to_string(py_idx) + ".bias";

            auto weight = RequireTensor(tensor_map, w_key).to(device, torch::kFloat32).contiguous();
            auto bias = RequireTensor(tensor_map, b_key).to(device, torch::kFloat32).contiguous();

            linears_[i]->weight.set_data(weight);
            linears_[i]->bias.set_data(bias);
        }

        const int py_out_idx = 2 * cfg_.n_hidden_layers;
        const std::string out_w_key = "residual.residual_net." + std::to_string(py_out_idx) + ".weight";
        const std::string out_b_key = "residual.residual_net." + std::to_string(py_out_idx) + ".bias";

        auto out_weight = RequireTensor(tensor_map, out_w_key).to(device, torch::kFloat32).contiguous();
        auto out_bias = RequireTensor(tensor_map, out_b_key).to(device, torch::kFloat32).contiguous();

        linears_.back()->weight.set_data(out_weight);
        linears_.back()->bias.set_data(out_bias);
    }

} // namespace grad_sdf_cpp
