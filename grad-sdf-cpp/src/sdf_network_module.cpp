#include "grad_sdf_cpp/sdf_network_module.hpp"

#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>

#include <yaml-cpp/yaml.h>

namespace grad_sdf_cpp
{
    namespace
    {

        torch::ScalarType ParseScalarType(const std::string &dtype)
        {
            if (dtype == "float32")
            {
                return torch::kFloat32;
            }
            if (dtype == "float64")
            {
                return torch::kFloat64;
            }
            if (dtype == "int64")
            {
                return torch::kInt64;
            }
            if (dtype == "int32")
            {
                return torch::kInt32;
            }
            if (dtype == "int16")
            {
                return torch::kInt16;
            }
            if (dtype == "int8")
            {
                return torch::kInt8;
            }
            if (dtype == "uint8")
            {
                return torch::kUInt8;
            }
            if (dtype == "bool")
            {
                return torch::kBool;
            }
            throw std::runtime_error("Unsupported tensor dtype in manifest: " + dtype);
        }

        size_t ScalarTypeByteSize(torch::ScalarType dtype)
        {
            switch (dtype)
            {
            case torch::kFloat32:
                return 4;
            case torch::kFloat64:
                return 8;
            case torch::kInt64:
                return 8;
            case torch::kInt32:
                return 4;
            case torch::kInt16:
                return 2;
            case torch::kInt8:
                return 1;
            case torch::kUInt8:
                return 1;
            case torch::kBool:
                return 1;
            default:
                break;
            }
            throw std::runtime_error("Unsupported scalar type for byte-size computation");
        }

        torch::Tensor LoadBinaryTensor(
            const std::filesystem::path &path,
            torch::ScalarType dtype,
            const std::vector<int64_t> &shape)
        {
            std::ifstream input(path, std::ios::binary | std::ios::ate);
            if (!input.is_open())
            {
                throw std::runtime_error("Failed to open tensor file: " + path.string());
            }

            const std::streamsize file_size_stream = input.tellg();
            if (file_size_stream < 0)
            {
                throw std::runtime_error("Failed to read tensor file size: " + path.string());
            }
            const size_t file_size = static_cast<size_t>(file_size_stream);
            input.seekg(0, std::ios::beg);

            int64_t numel = 1;
            for (const auto dim : shape)
            {
                if (dim < 0)
                {
                    throw std::runtime_error("Negative tensor dimension in manifest: " + path.string());
                }
                numel *= dim;
            }

            const size_t expected_size = static_cast<size_t>(numel) * ScalarTypeByteSize(dtype);
            if (expected_size != file_size)
            {
                throw std::runtime_error(
                    "Tensor binary size mismatch for " + path.string() +
                    ": expected " + std::to_string(expected_size) +
                    " bytes, got " + std::to_string(file_size));
            }

            std::vector<char> buffer(file_size);
            if (file_size > 0 && !input.read(buffer.data(), static_cast<std::streamsize>(file_size)))
            {
                throw std::runtime_error("Failed reading tensor bytes from: " + path.string());
            }

            auto tensor = torch::from_blob(
                buffer.data(),
                shape,
                torch::TensorOptions().dtype(dtype).device(torch::kCPU));
            return tensor.clone();
        }

        std::unordered_map<std::string, torch::Tensor> LoadTensorMapFromBundle(const std::string &bundle_dir)
        {
            const auto manifest_path = std::filesystem::path(bundle_dir) / "manifest.yaml";
            const YAML::Node manifest = YAML::LoadFile(manifest_path.string());

            if (!manifest || !manifest["tensors"])
            {
                throw std::runtime_error("Bundle manifest missing tensors section: " + manifest_path.string());
            }

            std::unordered_map<std::string, torch::Tensor> tensor_map;
            const YAML::Node tensors = manifest["tensors"];

            for (auto it = tensors.begin(); it != tensors.end(); ++it)
            {
                const std::string key = it->first.as<std::string>();
                const YAML::Node tensor_node = it->second;
                if (!tensor_node || !tensor_node["path"] || !tensor_node["dtype"] || !tensor_node["shape"])
                {
                    throw std::runtime_error("Invalid tensor entry in manifest for key: " + key);
                }

                const std::string rel_path = tensor_node["path"].as<std::string>();
                const std::string dtype_str = tensor_node["dtype"].as<std::string>();
                const auto shape = tensor_node["shape"].as<std::vector<int64_t>>();

                const auto abs_path = std::filesystem::path(bundle_dir) / rel_path;
                const auto dtype = ParseScalarType(dtype_str);

                auto tensor = LoadBinaryTensor(abs_path, dtype, shape);
                tensor_map.emplace(key, tensor);
            }

            return tensor_map;
        }

        int64_t GetOctreeNumVoxelsFromBundle(const std::string &bundle_dir)
        {
            const auto manifest_path = std::filesystem::path(bundle_dir) / "manifest.yaml";
            const YAML::Node manifest = YAML::LoadFile(manifest_path.string());

            if (manifest && manifest["metadata"] && manifest["metadata"]["octree_num_voxels"])
            {
                return manifest["metadata"]["octree_num_voxels"].as<int64_t>();
            }

            // Fallback: infer from first octree tensor
            const YAML::Node tensors = manifest["tensors"];
            if (tensors && tensors["octree.sdf_priors"])
            {
                const auto shape = tensors["octree.sdf_priors"]["shape"].as<std::vector<int64_t>>();
                if (!shape.empty())
                {
                    return shape[0];
                }
            }

            throw std::runtime_error("Could not determine octree num voxels from bundle manifest");
        }

    } // namespace

    SdfNetworkModuleImpl::SdfNetworkModuleImpl(const SdfNetworkConfig &cfg)
        : cfg_(cfg)
    {
        cfg_.residual_net_cfg.input_feature_dim =
            cfg_.octree_cfg.residual_feature_dim * cfg_.octree_cfg.residual_num_levels;

        octree_ = register_module("octree", OctreeModule(cfg_.octree_cfg));
        residual_ = register_module("residual", ResidualNetModule(cfg_.residual_net_cfg));
    }

    void SdfNetworkModuleImpl::LoadFromBundle(const std::string &bundle_dir)
    {
        auto tensor_map = LoadTensorMapFromBundle(bundle_dir);
        const auto model_device = octree_->device();
        const auto actual_num_voxels = GetOctreeNumVoxelsFromBundle(bundle_dir);

        // Note: If the loaded bundle has fewer voxels than init_voxel_num in the config,
        // the loaded tensors will be smaller. The octree's set_data will handle this.
        if (cfg_.octree_cfg.init_voxel_num > actual_num_voxels)
        {
            std::cout << "Note: Bundle has " << actual_num_voxels << " voxels, "
                      << "but config specifies " << cfg_.octree_cfg.init_voxel_num << ". "
                      << "Loaded tensors will be smaller; only first " << actual_num_voxels
                      << " voxels will be populated.\n";
        }

        octree_->LoadFromTensorMap(tensor_map, model_device);
        residual_->LoadFromTensorMap(tensor_map, model_device);
    }

    SdfNetworkForwardResult SdfNetworkModuleImpl::forward(
        const torch::Tensor &points,
        const c10::optional<torch::Tensor> &voxel_indices_opt)
    {
        if (points.size(-1) != 3)
        {
            throw std::runtime_error("Expected points shape (..., 3)");
        }

        const auto points_shape = points.sizes().vec();
        std::vector<int64_t> out_shape(points_shape.begin(), points_shape.end() - 1);

        auto points_flat = points.reshape({-1, 3});

        c10::optional<torch::Tensor> voxel_indices_flat = c10::nullopt;
        if (voxel_indices_opt.has_value())
        {
            voxel_indices_flat = voxel_indices_opt.value().reshape({-1});
        }

        auto octree_out = octree_->forward(points_flat, voxel_indices_flat);

        torch::Tensor sdf_residual;
        torch::Tensor sdf_pred;
        c10::optional<torch::Tensor> sdf_residual_opt = c10::nullopt;

        if (octree_out.residual_features.has_value())
        {
            auto residual_input = torch::cat(
                {octree_out.sdf_prior.unsqueeze(-1).detach(), octree_out.residual_features.value()},
                -1);
            sdf_residual = residual_->forward(residual_input).squeeze(-1);
            sdf_pred = octree_out.sdf_prior.detach() + sdf_residual;
            sdf_residual_opt = sdf_residual;
        }
        else
        {
            sdf_pred = octree_out.sdf_prior.detach();
        }

        SdfNetworkForwardResult out;
        out.voxel_indices = octree_out.voxel_indices.reshape(out_shape);
        out.sdf_prior = octree_out.sdf_prior.reshape(out_shape);
        out.sdf = sdf_pred.reshape(out_shape);
        if (sdf_residual_opt.has_value())
        {
            out.sdf_residual = sdf_residual_opt.value().reshape(out_shape);
        }
        else
        {
            out.sdf_residual = c10::nullopt;
        }
        return out;
    }

} // namespace grad_sdf_cpp
