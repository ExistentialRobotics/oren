#pragma once

#include <string>
#include <vector>

namespace grad_sdf_cpp {

struct OctreeConfig {
    double resolution = 0.1;
    int tree_depth = 8;
    int semi_sparse_depth = 5;
    int init_voxel_num = 200000;
    int insertion_threshold = 3;
    bool skip_insertion_if_exists = true;
    bool gradient_augmentation = true;
    int residual_feature_dim = 4;
    int residual_num_levels = 3;
    bool independent_smallest_leaf_vertex = false;
};

struct ResidualNetConfig {
    std::string mlp_activation = "LeakyReLU";
    int input_feature_dim = 4;
    int hidden_dims = 64;
    int n_hidden_layers = 5;
    double output_sdf_scale = 0.1;
    std::vector<double> bound_min;
    std::vector<double> bound_max;
};

struct SdfNetworkConfig {
    OctreeConfig octree_cfg;
    ResidualNetConfig residual_net_cfg;
};

SdfNetworkConfig LoadSdfNetworkConfigFromTrainerYaml(const std::string& trainer_yaml_path);

}  // namespace grad_sdf_cpp
