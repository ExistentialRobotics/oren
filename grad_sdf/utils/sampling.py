from dataclasses import dataclass
from functools import reduce

import torch

from .config_abc import ConfigABC
from .nearest_neighbor import nearest_neighbor


def generate_sample_mask(shape, num_samples: int):
    n = reduce(lambda x, y: x * y, shape)
    if num_samples >= n:
        return torch.ones(shape, dtype=torch.bool)

    indices = torch.randperm(n)[:num_samples]
    mask = torch.zeros(n, dtype=torch.bool)
    mask[indices] = True
    return mask.view(shape)


@dataclass
class SampleRaysConfig(ConfigABC):
    n_stratified: int = 20  # number of stratified samples
    n_perturbed: int = 8  # number of perturbed samples
    depth_min: float = 0.07  # minimum depth value
    depth_max: float = 10.0  # maximum depth value
    surface_margin: float = 0.10  # additional range beyond surface
    sigma_s: float = 0.05  # standard deviation for Gaussian sampling


@dataclass
class SampleResults:
    sampled_xyz: torch.Tensor
    positive_sdf_mask: torch.Tensor
    negative_sdf_mask: torch.Tensor
    valid_indices: torch.Tensor
    stratified_sdf: torch.Tensor
    perturbation_sdf: torch.Tensor
    n_stratified: int
    n_perturbed: int
    gaussian_positive_mask: torch.Tensor


@torch.no_grad()
def generate_sdf_samples(
    rays_d_all: torch.Tensor,
    rays_o_all: torch.Tensor,
    depth_samples_all: torch.Tensor,
    cfg: SampleRaysConfig,
    extra_surface_pcd: torch.Tensor = None,
    device=None,
) -> SampleResults:
    """
    Sample points along rays using surface-guided sampling strategy (GPU parallelized).
    Only processes valid rays (positive, finite depth values) and returns compact results.

    Args:
        rays_d_all: Ray directions (num_rays, 3)
        rays_o_all: Ray origins (num_rays, 3)
        depth_samples_all: Surface depth values D[u,v] (num_rays,) or (num_rays, 1)
        cfg: Configuration for sampling
        extra_surface_pcd: Additional surface points for computing SDF (num_extra_points, 3)
        device: Device for computation

    Returns:
        sampled_xyz: 3D coordinates of sampled points (num_valid_rays, N+M+1, 3)
        sampled_depth: Depth values for sampled points (num_valid_rays, N+M+1)
        negative_sdf_mask: Mask indicating positive perturbations (num_valid_rays, N+M+1)
        surface_mask: Mask indicating surface samples (num_valid_rays, N+M+1)
        perturbation_mask: Mask indicating perturbation samples (num_valid_rays, N+M+1)
        ray_sample_mask: Mask indicating free space samples (num_valid_rays, N+M+1)
        valid_indices: Indices of valid rays in original input (num_valid_rays, )
    """
    if device is None:
        device = rays_d_all.device

    n_stratified = cfg.n_stratified
    n_perturbed = cfg.n_perturbed
    depth_min = cfg.depth_min
    depth_max = cfg.depth_max
    surface_margin = cfg.surface_margin
    sigma_s = cfg.sigma_s

    # total_samples = n_stratified + n_perturbed + 1

    # Create valid mask to filter out invalid depth values (0, negative, or NaN)
    valid_mask = (
        (depth_samples_all > 0) & (depth_samples_all < depth_max) & torch.isfinite(depth_samples_all)
    )  # (num_rays,)
    valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]  # (num_valid_rays,)
    num_valid_rays = valid_indices.shape[0]

    # Extract only valid rays data
    rays_d_valid = rays_d_all[valid_indices]  # (num_valid_rays, 3)
    rays_o_valid = rays_o_all[valid_indices]  # (num_valid_rays, 3)
    depth_samples_valid = depth_samples_all[valid_indices]  # (num_valid_rays,)

    #############################################################
    # 1. Stratified sampling (vectorized) - only for valid rays #
    #############################################################
    # Compared with uniform sampling from [depth_min, d_max],
    # stratified sampling ensures coverage of free space.

    d_max = depth_samples_valid - surface_margin  # (num_valid_rays,)
    # Ensure d_max > d_min
    d_max = torch.where(d_max <= depth_min, depth_min + surface_margin, d_max)

    d_range = d_max - depth_min  # (num_valid_rays,)

    # Generate stratified samples for valid rays only
    if n_stratified == 1:
        bin_size = d_range.unsqueeze(1)  # (num_valid_rays, 1)
        bin_starts = torch.full((num_valid_rays, 1), depth_min, dtype=torch.float32, device=device)
    else:
        bin_size = d_range.unsqueeze(1) / n_stratified  # (num_valid_rays, 1)
        bin_indices = torch.arange(n_stratified, device=device, dtype=torch.float32).unsqueeze(0)
        bin_starts = depth_min + bin_indices * bin_size  # (num_valid_rays, n_stratified)

    # Uniform random samples within each bin
    uniform_samples = torch.rand(num_valid_rays, n_stratified, device="cpu").to(device)
    stratified_depths = bin_starts + uniform_samples * bin_size  # (num_valid_rays, n_stratified)

    # Record which stratified samples are beyond surface (negative sdf)
    # (num_valid_rays, n_stratified)
    # commented out because it is always false
    # stratified_positive_mask = stratified_depths > depth_samples_valid.unsqueeze(1)
    stratified_positive_mask = torch.zeros(num_valid_rays, n_stratified, dtype=torch.bool, device=device)
    stratified_negative_mask = torch.ones(num_valid_rays, n_stratified, dtype=torch.bool, device=device)

    #############################################################
    # 2. Gaussian sampling (vectorized) - perturbation by depth #
    #############################################################
    # Split perturbations: half in [-3*sigma_s, -sigma_s], half in [sigma_s, 3*sigma_s]
    n_negative = n_perturbed // 2  # samples before surface
    n_positive = n_perturbed - n_negative  # samples after surface

    # Negative perturbations: [-3*sigma_s, -sigma_s]
    negative_offsets = torch.rand(num_valid_rays, n_negative, device="cpu").to(device)  # [0, 1]
    negative_offsets = -3 * sigma_s + negative_offsets * (2 * sigma_s)  # [-3*sigma_s, -sigma_s]

    # Positive perturbations: [sigma_s, 3*sigma_s]
    positive_offsets = torch.rand(num_valid_rays, n_positive, device="cpu").to(device)  # [0, 1]
    positive_offsets = sigma_s + positive_offsets * (2 * sigma_s)  # [sigma_s, 3*sigma_s]

    # Combine offsets and compute depths
    perturbation_offsets = torch.cat([negative_offsets, positive_offsets], dim=1)  # (num_valid_rays, n_perturbed)
    gaussian_depths = depth_samples_valid.unsqueeze(1) + perturbation_offsets  # (num_valid_rays, n_perturbed)

    # Create mask: first n_negative are False (negative), last n_positive are True (positive)
    gaussian_positive_mask = torch.cat(
        [
            torch.zeros(num_valid_rays, n_negative, dtype=torch.bool, device=device),
            torch.ones(num_valid_rays, n_positive, dtype=torch.bool, device=device),
        ],
        dim=1,
    )  # (num_valid_rays, n_perturbed)

    ######################
    # 3. Surface samples #
    ######################
    surface_samples = depth_samples_valid.unsqueeze(1)  # (num_valid_rays, 1)

    #######################
    # Combine all samples #
    #######################

    # (num_valid_rays, n_stratified + n_perturbed + 1)
    all_depths = torch.cat([stratified_depths, gaussian_depths, surface_samples], dim=1)

    # Original negative_sdf_mask for backward compatibility
    negative_sdf_mask = torch.cat(
        [
            stratified_positive_mask,
            gaussian_positive_mask,
            torch.zeros(num_valid_rays, 1, dtype=torch.bool, device=device),
        ],
        dim=1,
    )  # (num_valid_rays, n_stratified + n_perturbed + 1)
    positive_sdf_mask = torch.cat(
        [
            stratified_negative_mask,
            ~gaussian_positive_mask,
            torch.zeros(num_valid_rays, 1, dtype=torch.bool, device=device),
        ],
        dim=1,
    )  # (num_valid_rays, n_stratified + n_perturbed + 1)

    # Calculate 3D coordinates (vectorized)
    # (num_valid_rays, 1, 3) + (num_valid_rays, total_samples, 1) * (num_valid_rays, 1, 3)
    sampled_xyz = rays_o_valid.unsqueeze(1) + all_depths.unsqueeze(2) * rays_d_valid.unsqueeze(1)

    sdf = nearest_neighbor(
        src=sampled_xyz[:, :-1].contiguous().view(-1, 3),
        dst=(
            sampled_xyz[:, -1].contiguous().view(-1, 3)
            if extra_surface_pcd is None
            else torch.cat([extra_surface_pcd.to(device), sampled_xyz[:, -1].contiguous().view(-1, 3)], dim=0)
        ),
    )[0].view(num_valid_rays, -1)
    stratified_sdf = sdf[:, :n_stratified].view(num_valid_rays, n_stratified)
    perturbation_sdf = sdf[:, n_stratified : n_stratified + n_perturbed].view(num_valid_rays, n_perturbed)
    perturbation_sdf = torch.where(gaussian_positive_mask, -perturbation_sdf, perturbation_sdf)

    return SampleResults(
        sampled_xyz=sampled_xyz,
        positive_sdf_mask=positive_sdf_mask,
        negative_sdf_mask=negative_sdf_mask,
        valid_indices=valid_indices,
        stratified_sdf=stratified_sdf,
        perturbation_sdf=perturbation_sdf,
        n_stratified=n_stratified,
        n_perturbed=n_perturbed,
        gaussian_positive_mask=gaussian_positive_mask,
    )
