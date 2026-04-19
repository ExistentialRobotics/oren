from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import torch
import torch.nn as nn

from grad_sdf.octree_config import OctreeConfig
from grad_sdf.residual_net import ResidualNet, ResidualNetConfig
from grad_sdf.semi_sparse_octree import SemiSparseOctree
from grad_sdf.utils.config_abc import ConfigABC


@dataclass
class SdfNetworkConfig(ConfigABC):
    octree_cfg: OctreeConfig = field(default_factory=OctreeConfig)
    residual_net_cfg: ResidualNetConfig = field(default_factory=ResidualNetConfig)


class SdfNetwork(nn.Module):
    def __init__(self, cfg: SdfNetworkConfig):
        super().__init__()
        self.cfg = cfg
        self.octree: SemiSparseOctree = SemiSparseOctree(cfg.octree_cfg)
        cfg.residual_net_cfg.input_feature_dim = (
            cfg.octree_cfg.residual_feature_dim * cfg.octree_cfg.residual_num_levels
        )
        self.residual: ResidualNet = ResidualNet(cfg.residual_net_cfg)

    def forward(self, points: torch.Tensor, voxel_indices: torch.Tensor = None):
        """
        Computes the SDF values for the given points.
        Args:
            points: (..., 3) points in world coordinates
            voxel_indices: (...,) optional voxel indices for the points

        Returns:
            (..., ) voxel indices for the points
            (..., ) SDF prior from the octree
            (..., ) SDF residual from the residual network
            (..., ) final SDF values (prior + residual)
        """
        shape = points.shape
        points = points.view(-1, 3)
        if voxel_indices is not None:
            voxel_indices = voxel_indices.view(-1)
        voxel_indices, sdf_prior, residual_features = self.octree(points, voxel_indices)
        if residual_features is not None:
            residual_network_input = torch.cat([sdf_prior.unsqueeze(-1).detach(), residual_features], dim=-1)
            sdf_residual = self.residual(residual_network_input).squeeze(-1)
            sdf_pred = sdf_prior.detach() + sdf_residual
        else:
            sdf_pred = sdf_prior.detach()
            sdf_residual = None

        voxel_indices = voxel_indices.view(shape[:-1])
        sdf_prior = sdf_prior.view(shape[:-1])
        sdf_pred = sdf_pred.view(shape[:-1])
        if sdf_residual is not None:
            sdf_residual = sdf_residual.view(shape[:-1])

        return voxel_indices, sdf_prior, sdf_residual, sdf_pred

    @torch.no_grad()
    def grid_vertex_filter(
        self,
        grid_points: torch.Tensor,
        min_voxel_size: int = 1,
        max_voxel_size: int = 2,
        dilation_iters: int = 1,
        batch_size: int = 204800,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Filter out grid vertices that are in voxels that are too big.
        Args:
            grid_points: (nx, ny, nz, 3) grid points in world coordinates
            min_voxel_size: minimum voxel size to keep
            max_voxel_size: maximum voxel size to keep
            dilation_iters: number of dilation iterations to fill small holes
            batch_size: number of points to process in a batch
            device: device to use, if None, use the device of grid_points

        Returns:
            (nx, ny, nz) boolean mask, True if the vertex is valid (in a voxel that is not too big)
        """
        assert grid_points.ndim == 4 and grid_points.shape[-1] == 3

        if batch_size <= 0:
            bs = grid_points.shape[0] * grid_points.shape[1] * grid_points.shape[2]
        else:
            bs = batch_size

        grid_shape = grid_points.shape
        grid_points = grid_points.view(-1, 3)

        model_device = self.octree.structure.device
        if device is None:  # device for the output mask
            device = grid_points.device

        valid_mask = []
        for start in range(0, grid_points.shape[0], bs):
            end = min(start + bs, grid_points.shape[0])
            indices = self.octree.find_voxel_indices(grid_points[start:end].to(model_device), False).view(-1)
            sizes = self.octree.get_voxel_discrete_size(indices)
            valid_mask.append(((sizes >= min_voxel_size) & (sizes <= max_voxel_size)).to(device))

        # indices = self.octree.find_voxel_indices(grid_points.view(-1, 3)).view(grid_points.shape[:-1])
        # voxel_sizes = self.octree.get_voxel_discrete_size(indices)

        if len(valid_mask) == 1:
            valid_mask = valid_mask[0]
        else:
            valid_mask = torch.cat(valid_mask, dim=0)
        valid_mask = valid_mask.view(grid_shape[:-1])  # (nx, ny, nz)

        # valid_mask = (voxel_sizes >= min_voxel_size) & (voxel_sizes <= max_voxel_size)  # (nx, ny, nz)

        # run a dilation to fill small holes: if any vertex is valid, we should keep the cube
        # such that we need to mark all 8 vertices as valid.
        # use convolution with all-ones kernel
        kernel = torch.ones((3, 3, 3), dtype=torch.float32, device=valid_mask.device).view(1, 1, 3, 3, 3)
        for _ in range(dilation_iters):
            valid_mask = (  # (nx, ny, nz)
                torch.nn.functional.conv3d(
                    input=valid_mask.view(1, 1, *valid_mask.shape).to(torch.float32),
                    weight=kernel,
                    padding=1,
                ).view(*valid_mask.shape)
                >= 1
            ).to(torch.bool)

        return valid_mask
