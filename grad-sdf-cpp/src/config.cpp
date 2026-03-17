#include "grad_sdf_cpp/config.hpp"

#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace grad_sdf_cpp {
namespace {

template<typename T>
void AssignIfPresent(const YAML::Node& node, const char* key, T& target) {
    if (node[key]) {
        target = node[key].as<T>();
    }
}

template<typename T>
void AssignIfPresentAny(const YAML::Node& node, const std::initializer_list<const char*>& keys, T& target) {
    for (const char* key : keys) {
        if (node[key]) {
            target = node[key].as<T>();
            return;
        }
    }
}

std::vector<double> ReadVec3IfPresent(const YAML::Node& node, const char* key) {
    if (!node[key]) {
        return {};
    }
    return node[key].as<std::vector<double>>();
}

void OverrideBoundsFromDatasetArgsIfPresent(const YAML::Node& root, SdfNetworkConfig& cfg) {
    if (!root["data"] || !root["data"]["dataset_args"]) {
        return;
    }

    constexpr double kDatasetBoundPadding = 0.3;
    const YAML::Node dataset_args = root["data"]["dataset_args"];
    const auto dataset_bound_min = ReadVec3IfPresent(dataset_args, "bound_min");
    const auto dataset_bound_max = ReadVec3IfPresent(dataset_args, "bound_max");

    if (dataset_bound_min.size() == 3) {
        cfg.residual_net_cfg.bound_min = {
            dataset_bound_min[0] - kDatasetBoundPadding,
            dataset_bound_min[1] - kDatasetBoundPadding,
            dataset_bound_min[2] - kDatasetBoundPadding,
        };
    }
    if (dataset_bound_max.size() == 3) {
        cfg.residual_net_cfg.bound_max = {
            dataset_bound_max[0] + kDatasetBoundPadding,
            dataset_bound_max[1] + kDatasetBoundPadding,
            dataset_bound_max[2] + kDatasetBoundPadding,
        };
    }
}

}  // namespace

SdfNetworkConfig LoadSdfNetworkConfigFromTrainerYaml(const std::string& trainer_yaml_path) {
    const YAML::Node root = YAML::LoadFile(trainer_yaml_path);

    if (!root) {
        throw std::runtime_error("Failed to load YAML file: " + trainer_yaml_path);
    }

    const YAML::Node model_node = root["model"] ? root["model"] : root;
    if (!model_node) {
        throw std::runtime_error("YAML does not contain model config: " + trainer_yaml_path);
    }

    const YAML::Node octree_node = model_node["octree_cfg"];
    const YAML::Node residual_node = model_node["residual_net_cfg"];

    if (!octree_node || !residual_node) {
        throw std::runtime_error("Model config must contain octree_cfg and residual_net_cfg.");
    }

    SdfNetworkConfig cfg;

    AssignIfPresent(octree_node, "resolution", cfg.octree_cfg.resolution);
    AssignIfPresent(octree_node, "tree_depth", cfg.octree_cfg.tree_depth);
    AssignIfPresent(octree_node, "semi_sparse_depth", cfg.octree_cfg.semi_sparse_depth);
    AssignIfPresent(octree_node, "init_voxel_num", cfg.octree_cfg.init_voxel_num);
    AssignIfPresent(octree_node, "insertion_threshold", cfg.octree_cfg.insertion_threshold);
    AssignIfPresent(octree_node, "skip_insertion_if_exists", cfg.octree_cfg.skip_insertion_if_exists);
    AssignIfPresent(octree_node, "gradient_augmentation", cfg.octree_cfg.gradient_augmentation);
    AssignIfPresentAny(octree_node, {"residual_feature_dim"}, cfg.octree_cfg.residual_feature_dim);
    AssignIfPresentAny(
        octree_node,
        {"residual_num_levels"},
        cfg.octree_cfg.residual_num_levels);
    AssignIfPresent(
        octree_node,
        "independent_smallest_leaf_vertex",
        cfg.octree_cfg.independent_smallest_leaf_vertex);

    AssignIfPresentAny(residual_node, {"feature_dims"}, cfg.octree_cfg.residual_feature_dim);
    AssignIfPresentAny(residual_node, {"num_levels"}, cfg.octree_cfg.residual_num_levels);

    cfg.residual_net_cfg.bound_min = ReadVec3IfPresent(residual_node, "bound_min");
    cfg.residual_net_cfg.bound_max = ReadVec3IfPresent(residual_node, "bound_max");

    AssignIfPresent(residual_node, "mlp_activation", cfg.residual_net_cfg.mlp_activation);
    AssignIfPresent(residual_node, "hidden_dims", cfg.residual_net_cfg.hidden_dims);
    AssignIfPresent(residual_node, "n_hidden_layers", cfg.residual_net_cfg.n_hidden_layers);
    AssignIfPresent(residual_node, "output_sdf_scale", cfg.residual_net_cfg.output_sdf_scale);

    // In training, dataset_args bounds override residual_net_cfg bounds.
    // Mirror that behavior in C++ so sampling/extraction uses the same spatial region.
    OverrideBoundsFromDatasetArgsIfPresent(root, cfg);

    cfg.residual_net_cfg.input_feature_dim =
        cfg.octree_cfg.residual_feature_dim * cfg.octree_cfg.residual_num_levels;

    return cfg;
}

}  // namespace grad_sdf_cpp
