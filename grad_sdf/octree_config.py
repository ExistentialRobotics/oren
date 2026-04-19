from dataclasses import dataclass

from grad_sdf.utils.config_abc import ConfigABC


@dataclass
class OctreeConfig(ConfigABC):
    resolution: float = 0.1
    tree_depth: int = 8
    semi_sparse_depth: int = 5
    init_voxel_num: int = 200000
    insertion_threshold: int = 3  # Minimum number of points to insert a voxel
    # If True, skip insertion if voxel of size 1 already exists. This can speed up insertion if the search
    # implementation is efficient enough.
    skip_insertion_if_exists: bool = True
    gradient_augmentation: bool = True
    residual_feature_dim: int = 4
    residual_num_levels: int = 3
    independent_smallest_leaf_vertex: bool = False
