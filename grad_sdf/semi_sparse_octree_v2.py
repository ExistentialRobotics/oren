from erl_geometry import SemiSparseOctreeF
from erl_geometry import find_voxel_indices
from erl_geometry import morton_encode

from grad_sdf import torch
from grad_sdf.octree_config import OctreeConfig
from grad_sdf.semi_sparse_octree_base import SemiSparseOctreeBase


class SemiSparseOctree(SemiSparseOctreeBase):
    def __init__(self, cfg: OctreeConfig):
        super(SemiSparseOctree, self).__init__(cfg)

        sso_setting = SemiSparseOctreeF.Setting()
        sso_setting.resolution = cfg.resolution
        sso_setting.tree_depth = cfg.tree_depth
        sso_setting.semi_sparse_depth = cfg.semi_sparse_depth
        sso_setting.init_voxel_num = cfg.init_voxel_num
        sso_setting.independent_smallest_leaf_vertex = cfg.independent_smallest_leaf_vertex
        sso_setting.cache_voxel_centers = True
        self.sso = SemiSparseOctreeF(sso_setting)
        self.key_offset = 1 << (self.cfg.tree_depth - 1)

    @torch.no_grad()
    def points_to_voxels(self, points: torch.Tensor):
        """
        Converts points to voxel coordinates.
        Args:
            points: (..., 3) point cloud in world coordinates
        Returns:
            voxels: (..., 3) voxel coordinates
        """
        voxels = torch.div(points, self.cfg.resolution, rounding_mode="floor").long()
        voxels += self.key_offset
        return voxels

    @torch.no_grad()
    def insert_voxels(self, voxels: torch.Tensor):
        self.ever_inserted = True
        svo_idx = self.sso.insert_keys(voxels.cpu().to(torch.uint32))  # on CPU

        # (N, 4) [x, y, z, voxel_size], (x, y, z) is the center coordinate

        self.voxels = self.sso.voxels_tensor.long().to(self.sdf_priors.device)
        self.voxel_centers = self.sso.voxel_centers_tensor.to(self.sdf_priors.device)
        self.vertex_indices = self.sso.vertices_tensor.to(self.sdf_priors.device)
        self.structure = self.sso.children_tensor.to(self.sdf_priors.device)

        return svo_idx

    @torch.no_grad()
    def find_voxel_indices(self, points: torch.Tensor, are_voxels: bool, level: int = 1) -> torch.Tensor:
        if are_voxels:
            voxels = points
        else:
            voxels = self.points_to_voxels(points)
        morton_codes = morton_encode(voxels.to(torch.uint32))
        voxel_indices = find_voxel_indices(
            codes=morton_codes,
            dims=3,
            level=self.cfg.tree_depth - level,
            children=self.structure,
        ).long()
        mask = ((voxels < 0) | (voxels >= (1 << self.cfg.tree_depth))).any(dim=-1)
        voxel_indices[mask] = -1  # Out of bounds
        return voxel_indices

    @property
    def little_endian_vertex_order(self):
        return True  # e.g. 1 -> (1, 0, 0), a vertex on the x-axis
