from typing import Tuple

import torch
from pytorch3d.ops import knn_points
from scipy.spatial import cKDTree

from grad_sdf.utils.profiling import CpuTimer


def nearest_neighbor_gpu(src: torch.Tensor, dst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the nearest neighbor in dst for each point in src.
    Args:
        src: (n_points, 3) source points
        dst: (m_points, 3) destination points
    Returns:
        dists: (n_points,) distances to the nearest neighbor in dst for each point in src
        idx: (n_points,) indices of the nearest neighbor in dst for each point in src
    """
    assert src.ndim == 2 and src.shape[1] == 3
    assert dst.ndim == 2 and dst.shape[1] == 3
    dists, idx, _ = knn_points(src.unsqueeze(0), dst.unsqueeze(0), K=1)
    dists = dists[0, :, 0].sqrt()  # (n_points,)
    idx = idx[0, :, 0]  # (n_points,)
    return dists, idx


def nearest_neighbor_cpu(src: torch.Tensor, dst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CPU version of nearest neighbor search using scipy.spatial.cKDTree.
    Args:
        src: (n_points, 3) source points
        dst: (m_points, 3) destination points
    Returns:
        dists: (n_points,) distances to the nearest neighbor in dst for each point in src
        idx: (n_points,) indices of the nearest neighbor in dst for each point in src
    """
    print("Using CPU for nearest neighbor search. This may be slow for large point clouds.")
    assert src.ndim == 2 and src.shape[1] == 3
    assert dst.ndim == 2 and dst.shape[1] == 3
    device = src.device
    src = src.cpu().numpy()
    dst = dst.cpu().numpy()
    print("Building KD-tree for destination points...")
    with CpuTimer("KD-tree construction"):
        tree = cKDTree(dst)
    print("Querying nearest neighbors...")
    with CpuTimer("KD-tree query"):
        dists, idx = tree.query(src, k=1, workers=-1)  # use all available CPU cores
    dists = torch.from_numpy(dists).float().to(device)
    idx = torch.from_numpy(idx).long().to(device)
    return dists, idx


def nearest_neighbor(src: torch.Tensor, dst: torch.Tensor, use_gpu: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the nearest neighbor in dst for each point in src.
    Args:
        src: (n_points, 3) source points
        dst: (m_points, 3) destination points
        use_gpu: whether to use GPU for nearest neighbor search
    Returns:
        dists: (n_points,) distances to the nearest neighbor in dst for each point in src
        idx: (n_points,) indices of the nearest neighbor in dst for each point in src
    """
    if use_gpu and torch.cuda.is_available() and src.is_cuda and dst.is_cuda:
        return nearest_neighbor_gpu(src, dst)
    else:
        return nearest_neighbor_cpu(src, dst)
