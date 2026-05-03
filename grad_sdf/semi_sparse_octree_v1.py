"""
Semi-sparse voxel octree wrapper using the sparse_octree package.
sparse_octree package is a modified version of svo from H2-Mapping:
https://github.com/Robotics-STAR-Lab/H2-Mapping

There are some drawbacks of this implementation:
1. negative metric coordinates are not supported.
2. there is memory leak issue in the underlying C++ implementation.
3. the octree structure is not flexible. e.g. deleting voxels is not supported.
4. insert_points is not as efficient as the one in erl_geometry.
5. find_voxel_indices is not as efficient as the one in erl_geometry.
"""

import sparse_octree

from grad_sdf import torch
from grad_sdf.octree_config import OctreeConfig
from grad_sdf.semi_sparse_octree_base import SemiSparseOctreeBase


class SemiSparseOctree(SemiSparseOctreeBase):
    def __init__(self, cfg: OctreeConfig):
        super(SemiSparseOctree, self).__init__(cfg)
        self.svo = sparse_octree.Octree()
        self.svo.init(
            1 << self.cfg.tree_depth,
            self.cfg.init_voxel_num,
            self.cfg.resolution,
            self.cfg.semi_sparse_depth,
        )

    @torch.no_grad()
    def points_to_voxels(self, points: torch.Tensor) -> torch.Tensor:
        """
        Converts points to voxel coordinates.
        Args:
            points: (..., 3) point cloud in world coordinates
        Returns:
            voxels: (..., 3) voxel coordinates
        """
        assert (points >= 0).all(), "Octree v1 does not support negative coordinates"
        voxels = torch.div(points, self.cfg.resolution, rounding_mode="floor").long()  # Divides each element
        return voxels

    @torch.no_grad()
    def insert_voxels(self, voxels: torch.Tensor) -> torch.Tensor:
        self.ever_inserted = True
        voxels_svo, children_svo, vertices_svo, svo_mask, svo_idx = self.svo.insert(voxels.cpu().int())
        # update grid state
        self.voxels = voxels_svo.to(self.sdf_priors.device)
        self.voxel_centers = (self.voxels[:, :3] + self.voxels[:, [-1]] * 0.5) * self.cfg.resolution
        self.vertex_indices = vertices_svo.to(self.sdf_priors.device)
        self.structure = children_svo.int().to(self.sdf_priors.device)
        return svo_idx[..., 0].long()  # (N,) tensor of indices of inserted voxels

    @torch.no_grad()
    def find_voxel_indices(self, points: torch.Tensor, are_voxels: bool) -> torch.Tensor:
        assert self.voxels is not None, "Octree is empty. Please insert points first."
        assert points.device == self.voxels.device, "Points and octree must be on the same device."

        device = points.device
        n_points = points.shape[0]
        root_idx = 0

        # Initialize result to -1
        voxel_indices = torch.full((n_points,), -1, dtype=torch.long, device=device)

        # Point indices still being traversed and their current node row numbers
        active_pts = torch.arange(n_points, device=device)  # [A]
        cur_nodes = torch.full_like(active_pts, root_idx)  # Initially all at root node

        for i in range(self.cfg.tree_depth + 1):
            if active_pts.numel() == 0:
                break

            # Calculate child numbers
            c = self.voxel_centers[cur_nodes]  # [A,3]
            if are_voxels:
                c /= self.cfg.resolution

            ge_mask = (points[active_pts] >= c).long()  # [A,3]
            child_id = ge_mask[:, 0] + (ge_mask[:, 1] << 1) + (ge_mask[:, 2] << 2)
            child_idx = self.structure[cur_nodes, child_id].long()  # [A]

            # Hit condition: reach a leaf node or no expected child
            hit_mask: torch.Tensor = child_idx == -1
            if hit_mask.any():
                voxel_indices[active_pts[hit_mask]] = cur_nodes[hit_mask]
            # Continue only with those that didn't hit
            keep_mask = ~hit_mask
            if not keep_mask.any():
                break

            active_pts = active_pts[keep_mask]
            cur_nodes = child_idx[keep_mask]

        return voxel_indices

    @property
    def little_endian_vertex_order(self):
        return False  # e.g. 1 -> (0, 0, 1), a vertex on the z-axis
