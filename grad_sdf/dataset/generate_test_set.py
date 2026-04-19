import argparse
import os

import numpy as np
from tqdm import tqdm

from grad_sdf import o3d, torch
from grad_sdf.utils.mesh_sdf import MeshSdf
from grad_sdf.utils.nearest_neighbor import nearest_neighbor


def compute_sdf_ground_truth(
    gt_mesh: o3d.geometry.TriangleMesh, query_points: np.ndarray, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the ground truth SDF and its gradient at the query points.

    Args:
        gt_mesh: o3d.geometry.TriangleMesh, the ground truth mesh
        query_points: (N, 3) array of query points
        eps: float, small value for numerical gradient computation

    Returns:
        sdf: (N,) SDF values
        grad: (N, 3) normalized SDF gradient
    """
    # compute SDF
    tqdm.write("Computing ground truth SDF")
    f = MeshSdf(np.asarray(gt_mesh.vertices), np.asarray(gt_mesh.triangles))
    sdf = f(query_points)

    # compute gradient
    tqdm.write("Computing ground truth SDF gradient")
    grad = np.zeros_like(query_points)
    for i in range(3):
        offset = np.zeros((3,))
        offset[i] = eps
        a = f(query_points + offset)
        b = f(query_points - offset)
        grad[:, i] = (a - b) / (2 * eps)

    # normalize gradient
    grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
    grad_norm[grad_norm == 0] = 1
    grad = grad / grad_norm

    return sdf, grad


def compute_sdf_ground_truth_from_pcd(
    points_gt: np.ndarray,
    query_points: np.ndarray,
    eps: float,
    use_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute unsigned distance (as SDF) and its normalized gradient at query points
    using nearest-neighbor distance to a point cloud.

    Args:
        points_gt: (M, 3) ground truth surface point cloud
        query_points: (N, 3) array of query points
        eps: float, step size for numerical gradient computation
        use_gpu: bool, whether to use GPU for nearest neighbor search

    Returns:
        sdf: (N,) unsigned distance to nearest point on the surface
        grad: (N, 3) normalized gradient (invalid positions get zero gradient)
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    pts_gt = torch.from_numpy(points_gt.astype(np.float32)).to(device)
    q = torch.from_numpy(query_points.astype(np.float32)).to(device)

    tqdm.write("Computing ground truth SDF from point cloud (nearest-neighbor distance)")
    sdf_t, _ = nearest_neighbor(q, pts_gt, use_gpu=use_gpu)
    sdf = sdf_t.cpu().numpy().astype(np.float64)

    tqdm.write("Computing ground truth SDF gradient")
    n_pts = q.shape[0]
    grad = torch.zeros((3, n_pts), dtype=torch.float32, device=device)
    for i in tqdm(range(3), desc="Gradient dim", ncols=80):
        offset = torch.zeros((1, 3), dtype=torch.float32, device=device)
        offset[0, i] = eps
        sdf_plus = nearest_neighbor(q + offset, pts_gt, use_gpu=use_gpu)[0]
        sdf_minus = nearest_neighbor(q - offset, pts_gt, use_gpu=use_gpu)[0]
        grad[i] = (sdf_plus - sdf_minus) / (2 * eps)

    grad_norm = torch.linalg.norm(grad, dim=0, keepdim=True)
    invalid = grad_norm < 1e-6
    grad_norm = grad_norm.clone()
    grad_norm[invalid] = 1.0
    grad = (grad / grad_norm).T.cpu().numpy().astype(np.float64)  # (N, 3)

    return sdf, grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh-path",
        type=str,
        default=None,
        help="Path to the ground truth mesh file (e.g., .ply). Omit if using --pcd-path.",
    )
    parser.add_argument(
        "--pcd-path",
        type=str,
        default=None,
        help="Path to the ground truth point cloud file. If set, SDF is computed as nearest-neighbor distance instead of mesh SDF.",
    )
    parser.add_argument("--bound-min", type=float, nargs=3, default=None)
    parser.add_argument("--bound-max", type=float, nargs=3, default=None)
    parser.add_argument("--offset", type=float, nargs=3, help="Offset to move the bounds")
    parser.add_argument("--grid-resolution", type=float, default=0.0125, help="Resolution of the grid to generate")
    parser.add_argument("--eps", type=float, default=0.001, help="Small value for numerical gradient computation")
    parser.add_argument(
        "--near-surface-sdf-range",
        type=float,
        nargs=2,
        default=(-0.1, 0.2),
        help="SDF range to consider as near surface",
    )
    parser.add_argument(
        "--nn-on-gpu",
        action="store_true",
        help="Whether to use GPU for nearest neighbor search when computing SDF from point cloud",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    use_pcd = args.pcd_path is not None
    if use_pcd and args.mesh_path is not None:
        raise ValueError("Provide either --mesh-path or --pcd-path, not both.")
    if not use_pcd and args.mesh_path is None:
        raise ValueError("Provide either --mesh-path or --pcd-path.")

    mesh_path: str | None = args.mesh_path
    pcd_path: str | None = args.pcd_path
    bound_min: list[float] | None = args.bound_min
    bound_max: list[float] | None = args.bound_max
    offset: list[float] | None = args.offset
    grid_resolution: float = args.grid_resolution
    eps: float = args.eps
    near_surface_sdf_range: list[float] = args.near_surface_sdf_range
    output_dir: str = args.output_dir

    assert near_surface_sdf_range[1] > near_surface_sdf_range[0], "Invalid near surface SDF range"
    assert near_surface_sdf_range[1] > 0, "Right bound of near surface SDF range should be positive"
    padding = abs(near_surface_sdf_range[0]) * 2
    # TODO: crop the ground truth points/mesh with the bounds
    if use_pcd:
        pcd = o3d.io.read_point_cloud(pcd_path)
        points_gt = np.asarray(pcd.points).astype(np.float64)
        if bound_min is None or bound_max is None:
            bmin = np.asarray(pcd.get_min_bound())
            bmax = np.asarray(pcd.get_max_bound())
            bound_min = bmin - padding if bound_min is None else np.asarray(bound_min)
            bound_max = bmax + padding if bound_max is None else np.asarray(bound_max)
    else:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        points_gt = None
        if bound_min is None or bound_max is None:
            bmin = np.asarray(mesh.get_min_bound())
            bmax = np.asarray(mesh.get_max_bound())
            bound_min = bmin - padding if bound_min is None else np.asarray(bound_min)
            bound_max = bmax + padding if bound_max is None else np.asarray(bound_max)

    if offset is None:
        offset = np.zeros((3,))
    bound_min = np.asarray(bound_min) - np.asarray(offset)
    bound_max = np.asarray(bound_max) - np.asarray(offset)

    os.makedirs(output_dir, exist_ok=True)

    x = np.arange(bound_min[0], bound_max[0], grid_resolution)
    y = np.arange(bound_min[1], bound_max[1], grid_resolution)
    z = np.arange(bound_min[2], bound_max[2], grid_resolution)
    x_size = x.shape[0]
    y_size = y.shape[0]
    z_size = z.shape[0]
    grid_points = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)

    if use_pcd:
        gt_sdf_values, gt_sdf_grad = compute_sdf_ground_truth_from_pcd(
            points_gt, grid_points.reshape(-1, 3), eps, use_gpu=args.nn_on_gpu
        )
    else:
        gt_sdf_values, gt_sdf_grad = compute_sdf_ground_truth(mesh, grid_points.reshape(-1, 3), eps)
    gt_sdf_values = gt_sdf_values.reshape(x_size, y_size, z_size)
    gt_sdf_grad = gt_sdf_grad.reshape(x_size, y_size, z_size, 3)

    np.save(os.path.join(output_dir, "grid_points.npy"), grid_points)
    np.save(os.path.join(output_dir, "gt_sdf_values.npy"), gt_sdf_values)
    np.save(os.path.join(output_dir, "gt_sdf_grad.npy"), gt_sdf_grad)

    # remove outliers
    threshold = max(bound_max[0] - bound_min[0], bound_max[1] - bound_min[1], bound_max[2] - bound_min[2]) * 0.5
    valid_mask = np.abs(gt_sdf_values) <= threshold

    # near surface points
    near_surface_mask = (gt_sdf_values >= near_surface_sdf_range[0]) & (gt_sdf_values <= near_surface_sdf_range[1])
    near_surface_mask &= valid_mask
    print(f"Number of near surface points: {np.sum(near_surface_mask)}/{gt_sdf_values.size}")
    np.save(os.path.join(output_dir, "mask_near_surface.npy"), near_surface_mask)

    # far away points
    far_away_mask = (gt_sdf_values > near_surface_sdf_range[1]) & valid_mask
    print(f"Number of far surface points: {np.sum(far_away_mask)}/{gt_sdf_values.size}")
    np.save(os.path.join(output_dir, "mask_far_surface.npy"), far_away_mask)


if __name__ == "__main__":
    main()
