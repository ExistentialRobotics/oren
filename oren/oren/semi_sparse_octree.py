from erl_geometry import SemiSparseOctreeF, find_voxel_indices, morton_encode

from oren import torch
from oren.octree_config import OctreeConfig
from oren.semi_sparse_octree_base import SemiSparseOctreeBase


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

        # Copy IN-PLACE into the pre-allocated capacity-sized buffers instead of
        # rebinding to new tensors. Fixed addresses are required for CUDA-graph
        # capture of the training step (its gather reads these buffers) — a
        # replayed graph would otherwise read freed/stale memory. The sso tensors
        # are the same capacity shape, so each copy_ is a straight overwrite.
        self.voxels.copy_(self.sso.voxels_tensor.long())
        self.voxel_centers.copy_(self.sso.voxel_centers_tensor)
        self.vertex_indices.copy_(self.sso.vertices_tensor)
        self.structure.copy_(self.sso.children_tensor)

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
            n_levels=self.cfg.tree_depth - level,
            children=self.structure,
        ).long()
        mask = ((voxels < 0) | (voxels >= (1 << self.cfg.tree_depth))).any(dim=-1)
        voxel_indices[mask] = -1  # Out of bounds
        return voxel_indices

    @property
    def little_endian_vertex_order(self):
        return True  # e.g. 1 -> (1, 0, 0), a vertex on the x-axis
