import os
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import trimesh
from ruamel import yaml
from scipy.spatial import cKDTree

from grad_sdf import MarchingCubes, np, o3d, torch


def _load_pointcloud(path: str) -> np.ndarray:
    """Load point cloud from .npy, .ply, or .pcd. Returns (N, 3) float array."""
    path_lower = path.lower()
    if path_lower.endswith(".npy"):
        pts = np.load(path).astype(np.float64)
    elif path_lower.endswith(".ply") or path_lower.endswith(".pcd"):
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported point cloud format: {path}. Use .npy, .ply, or .pcd")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Point cloud must be (N, 3), got shape {pts.shape}")
    return pts


def _load_bbox(bbox_def_file: str | None) -> Tuple[trimesh.Trimesh | None, dict | None]:
    if bbox_def_file is None:
        return None, None

    with open(bbox_def_file, "r") as f:
        bbox_def = yaml.safe_load(f)
    bbox_size = bbox_def["size"]
    bbox_center = bbox_def["center"]
    bbox_rotation = bbox_def["rotation"]  # quaternion [w, x, y, z]

    bbox_extents = np.array(bbox_size)
    bbox_pose = trimesh.transformations.quaternion_matrix(bbox_rotation)
    bbox_pose[:3, 3] = bbox_center
    bbox = trimesh.creation.box(extents=bbox_extents, transform=bbox_pose)

    bbox_def = dict(size=bbox_size, pose=bbox_pose)

    return bbox, bbox_def


def _crop_to_bbox(mesh: trimesh.Trimesh, bbox: Optional[trimesh.Trimesh]) -> trimesh.Trimesh:
    if bbox is None:
        return mesh
    mesh = mesh.slice_plane(bbox.facets_origin, -bbox.facets_normal)
    return mesh


class EvaluatorBase:

    def __init__(
        self,
        model_forward_func: Callable[
            [torch.nn.Module, torch.Tensor, bool, bool, float, Optional[str]],
            Dict[str, torch.Tensor],
        ],
        model: torch.nn.Module | None = None,
        model_path: str | None = None,
        model_create_func: Callable[[str], torch.nn.Module] | None = None,
        device: str = "cuda",
        absolute_sdf: bool = True,
        grad_err_outlier_threshold: float = 0.5,
        interactive: bool = False,
    ):
        """
        Base class for evaluators.
        Args:
            model_forward_func: function to forward the model, takes (model, input tensor, get_grad, auto_grad,
                                finite_diff_eps, device=None), returns a dict of output tensors
            model: optional, if provided, use this model
            model_path: optional, if model is not provided, load the model from this path
            model_create_func: optional, function to create the model, takes model_path as input, returns the model
            device: device to run the model on
            absolute_sdf: if True, ignore the sign of SDF values when computing metrics, only consider absolute values.
            This is useful when the ground truth SDF does not have sign information.
            grad_err_outlier_threshold: threshold to filter out outliers in gradient error computation, in radians
            interactive: whether to enable interactive visualization (e.g., for gradient angle difference)
        """
        assert model_forward_func is not None
        self.model_forward_func = model_forward_func
        self.device = device
        self.grad_err_outlier_threshold = grad_err_outlier_threshold
        self.interactive = interactive
        self.absolute_sdf = absolute_sdf

        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
        else:
            assert model_path is not None
            assert model_create_func is not None

            self.model: torch.nn.Module = model_create_func(model_path).to(self.device)
            self.model.eval()

    def _sdf_metrics(self, sdf_pred: torch.Tensor, sdf_gt: torch.Tensor):
        if self.absolute_sdf:
            sdf_pred = sdf_pred.abs()
            sdf_gt = sdf_gt.abs()
        diff = sdf_pred - sdf_gt
        return dict(mae=diff.abs().mean().item(), rmse=(diff**2).mean().sqrt().item())

    def _grad_metrics(self, grad_pred: torch.Tensor, grad_gt: torch.Tensor, mask: Optional[torch.Tensor] = None):
        grad_pred /= grad_pred.norm(dim=-1, keepdim=True) + 1e-8
        grad_gt /= grad_gt.norm(dim=-1, keepdim=True) + 1e-8
        cos_sim = (grad_pred * grad_gt).sum(dim=-1)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        angle_diff = torch.acos(cos_sim)  # in radians
        if mask is not None:
            angle_diff = angle_diff[mask]

        if self.interactive:
            # visualize as histograms
            plt.figure()
            plt.hist(angle_diff.cpu().numpy(), bins=50, color="steelblue", edgecolor="black", alpha=0.7)
            plt.xlabel("Angle Difference (radians)")
            plt.ylabel("Count")
            plt.title("Gradient Angle Difference")
            plt.show()

        angle_diff = angle_diff.abs()
        mask = angle_diff < self.grad_err_outlier_threshold
        angle_diff = angle_diff[mask]
        return angle_diff.mean().item()

    def sdf_and_grad_metrics(
        self,
        test_set_dir: str,
        sdf_fields: list[str],
        grad_method: str = "autograd",
        eps: float = 0.001,
    ):
        grid_points = np.load(os.path.join(test_set_dir, "grid_points.npy"))
        gt_sdf_values = np.load(os.path.join(test_set_dir, "gt_sdf_values.npy"))
        gt_sdf_grad = np.load(os.path.join(test_set_dir, "gt_sdf_grad.npy"))

        grid_points = torch.from_numpy(grid_points).float()
        gt_sdf_values = torch.from_numpy(gt_sdf_values).float().to(self.device)
        gt_sdf_grad = torch.from_numpy(gt_sdf_grad).float().to(self.device)

        autograd = grad_method == "autograd"
        self.model.eval()
        result = self.model_forward_func(self.model, grid_points, True, autograd, eps, None)

        near_surface_mask = np.load(os.path.join(test_set_dir, "mask_near_surface.npy"))
        far_away_mask = np.load(os.path.join(test_set_dir, "mask_far_surface.npy"))
        all_mask = near_surface_mask | far_away_mask

        sdf_metrics = dict(
            near_surface={
                k: self._sdf_metrics(result[k][near_surface_mask], gt_sdf_values[near_surface_mask]) for k in sdf_fields
            },
            far_away={k: self._sdf_metrics(result[k][far_away_mask], gt_sdf_values[far_away_mask]) for k in sdf_fields},
            all={k: self._sdf_metrics(result[k][all_mask], gt_sdf_values[all_mask]) for k in sdf_fields},
        )

        if self.interactive:
            # visualization of gradient angle difference on the grid as a 2D z-slice
            grad_pred = result["grad"]["sdf"]
            grad_gt = gt_sdf_grad

            grad_pred /= grad_pred.norm(dim=-1, keepdim=True) + 1e-8
            grad_gt /= grad_gt.norm(dim=-1, keepdim=True) + 1e-8
            cos_sim = (grad_pred * grad_gt).sum(dim=-1)
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
            angle_diff = torch.acos(cos_sim)  # in radians

            angle_diff = angle_diff.reshape(grid_points.shape[:-1])  # reshape to (nx, ny, nz)
            mask = (gt_sdf_values > 0) & (result["sdf"] > 0)  # only consider where both gt and pred are positive
            angle_diff[mask] = 0

            plt.figure()
            z = 0
            img = plt.imshow(angle_diff[:, :, z].cpu().numpy(), cmap="viridis")
            plt.colorbar(label="Gradient Angle Difference (radians)")
            plt.title(f"Gradient Angle Difference at z={z}")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")

            def keyboard_event_handler(event):
                nonlocal z
                if event.key == "up":
                    z = min(z + 1, angle_diff.shape[2] - 1)
                elif event.key == "down":
                    z = max(z - 1, 0)
                data = angle_diff[:, :, z].cpu().numpy()
                img.set_data(data)
                # update color scale to the new data range
                img.set_clim(vmin=data.min(), vmax=data.max())
                plt.title(f"Gradient Angle Difference at z={z}")
                plt.draw()

            plt.gcf().canvas.mpl_connect("key_press_event", keyboard_event_handler)
            plt.show()

        # only consider gradients where both g.t. and pred. are positive
        # because the ground truth might not have sign info
        positive_mask_gt = gt_sdf_values > 0
        positive_mask = {k: (result[k] > 0) & positive_mask_gt for k in sdf_fields}
        grad_metrics = dict(
            near_surface={
                k: self._grad_metrics(
                    result["grad"][k][near_surface_mask],
                    gt_sdf_grad[near_surface_mask],
                    positive_mask[k][near_surface_mask],
                )
                for k in sdf_fields
            },
            far_away={
                k: self._grad_metrics(
                    result["grad"][k][far_away_mask],
                    gt_sdf_grad[far_away_mask],
                    positive_mask[k][far_away_mask],
                )
                for k in sdf_fields
            },
            all={
                k: self._grad_metrics(
                    result["grad"][k][all_mask],
                    gt_sdf_grad[all_mask],
                    positive_mask[k][all_mask],
                )
                for k in sdf_fields
            },
        )

        return dict(sdf_metrics=sdf_metrics, grad_metrics=grad_metrics)

    @staticmethod
    def mesh_metrics(
        pred_mesh_path: str,
        gt_mesh_path: str,
        bbox_def_file: str | None = None,
        threshold: float = 0.05,
        num_samples: int = 200_000,
        seed: int = 0,
    ) -> dict:
        """
        Compute mesh metrics with ground truth mesh as reference.

        Args:
            pred_mesh_path: Path to predicted mesh file.
            gt_mesh_path: Path to ground-truth mesh file.
            bbox_def_file: Optional path to a file defining the bounding box for evaluation, in case the meshes are not well-aligned.
            threshold: Distance threshold for precision/recall/completion_ratio.
            num_samples: Number of points to sample on the predicted mesh for evaluation.
            seed: Random seed for sampling.

        Returns:
            dict with completion_ratio, completion, accuracy, chamfer, precision, recall, f1,
            threshold, num_samples, seed.
        """
        pred_mesh = trimesh.load_mesh(pred_mesh_path)
        gt_mesh = trimesh.load_mesh(gt_mesh_path)

        bbox, bbox_def = _load_bbox(bbox_def_file)
        pred_mesh = _crop_to_bbox(pred_mesh, bbox)
        gt_mesh = _crop_to_bbox(gt_mesh, bbox)

        pred_pts = trimesh.sample.sample_surface(pred_mesh, num_samples, seed=seed)[0]
        gt_pts = trimesh.sample.sample_surface(gt_mesh, num_samples, seed=seed)[0]

        pred_tree = cKDTree(pred_pts)
        gt_tree = cKDTree(gt_pts)

        dist_pred_to_gt, _ = gt_tree.query(pred_pts, k=1, workers=-1)
        dist_gt_to_pred, _ = pred_tree.query(gt_pts, k=1, workers=-1)

        completion_ratio = np.mean(dist_gt_to_pred < threshold).item()
        completion = np.mean(dist_gt_to_pred)

        accuracy = np.mean(dist_pred_to_gt)
        chamfer = (completion + accuracy) / 2.0

        tp = np.sum(dist_pred_to_gt < threshold).item()
        fp = num_samples - tp
        fn = np.sum(dist_gt_to_pred >= threshold).item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return dict(
            completion_ratio=completion_ratio,
            completion=completion,
            accuracy=accuracy,
            chamfer=chamfer,
            precision=precision,
            recall=recall,
            f1=f1,
            threshold=threshold,
            num_samples=num_samples,
            seed=seed,
        )

    @staticmethod
    def mesh_metrics_pointcloud_gt(
        pred_mesh_path: str,
        gt_pointcloud_path: str,
        bbox_def_file: str | None = None,
        threshold: float = 0.05,
        num_samples: int = 200_000,
        seed: int = 0,
    ):
        """
        Compute mesh metrics with point cloud as ground truth.

        Prediction is a mesh (sampled to num_pred_samples points). Ground truth is a point cloud
        loaded from file (.npy, .ply, or .pcd), used as-is.

        Args:
            pred_mesh_path: Path to predicted mesh file.
            gt_pointcloud_path: Path to ground-truth point cloud (.npy, .ply, or .pcd).
            bbox_def_file: Path to bounding box definition file (optional).
            threshold: Distance threshold for precision/recall/completion_ratio.
            num_samples: Number of points to sample on the predicted mesh.
            seed: Random seed for sampling.

        Returns:
            dict with completion_ratio, completion, accuracy, chamfer, precision, recall, f1,
            threshold, num_samples, num_gt_points, seed.
        """
        pred_mesh = trimesh.load_mesh(pred_mesh_path)
        gt_pts_all = _load_pointcloud(gt_pointcloud_path)

        bbox, bbox_def = _load_bbox(bbox_def_file)
        pred_mesh = _crop_to_bbox(pred_mesh, bbox)
        if bbox is not None:
            points = np.asarray(gt_pts_all)
            bbox_

        pred_pts = trimesh.sample.sample_surface(pred_mesh, num_samples, seed=seed)[0]
        num_gt = gt_pts_all.shape[0]
        if gt_pts_all.shape[0] > num_samples:
            rng = np.random.default_rng(seed)
            indices = rng.choice(gt_pts_all.shape[0], num_samples, replace=False)
            gt_pts = gt_pts_all[indices]

        gt_tree = cKDTree(gt_pts_all)
        pred_tree = cKDTree(pred_pts)

        dist_pred_to_gt, _ = gt_tree.query(pred_pts, k=1, workers=-1)
        dist_gt_to_pred, _ = pred_tree.query(gt_pts, k=1, workers=-1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.hist(dist_pred_to_gt, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        ax1.set_xlabel("Distance")
        ax1.set_ylabel("Count")
        ax1.set_title("dist_pred_to_gt (pred → nearest GT)")
        ax1.axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold}")
        ax1.legend()
        ax2.hist(dist_gt_to_pred, bins=50, color="coral", edgecolor="black", alpha=0.7)
        ax2.set_xlabel("Distance")
        ax2.set_ylabel("Count")
        ax2.set_title("dist_gt_to_pred (GT → nearest pred)")
        ax2.axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold}")
        ax2.legend()
        plt.tight_layout()
        plt.show()

        completion_ratio = np.mean(dist_gt_to_pred < threshold).item()
        completion = np.mean(dist_gt_to_pred)

        accuracy = np.mean(dist_pred_to_gt)
        chamfer = (completion + accuracy) / 2.0

        tp = np.sum(dist_pred_to_gt < threshold).item()
        fp = num_samples - tp
        fn = np.sum(dist_gt_to_pred >= threshold).item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return dict(
            completion_ratio=completion_ratio,
            completion=completion,
            accuracy=accuracy,
            chamfer=chamfer,
            precision=precision,
            recall=recall,
            f1=f1,
            threshold=threshold,
            num_samples=num_samples,
            num_gt_points=num_gt,
            seed=seed,
        )

    def compute_metrics(
        self,
        test_set_dir: str,
        sdf_fields: List[str],
        grad_method: str,
        eps: float,
        pred_mesh_path: str,
        gt_mesh_path: str,
        threshold: float = 0.05,
        num_samples: int = 200_000,
        seed: int = 0,
    ):
        metrics: dict = self.sdf_and_grad_metrics(test_set_dir, sdf_fields, grad_method, eps)
        metrics["mesh_metrics"] = self.mesh_metrics(
            pred_mesh_path,
            gt_mesh_path,
            threshold=threshold,
            num_samples=num_samples,
            seed=seed,
        )
        return metrics

    @torch.no_grad()
    def extract_sdf_grid(
        self,
        bound_min: List[float],
        bound_max: List[float],
        grid_resolution: float,
        grid_vertex_filter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Optional[str] = None,
    ):
        """
        Extract SDF grid from the model.
        Args:
            bound_min: Minimum bound of the 3D grid (list of 3 floats)
            bound_max: Maximum bound of the 3D grid (list of 3 floats)
            grid_resolution: Resolution of the grid (float)
            grid_vertex_filter: Optional function to filter grid vertices, takes (N, 3) tensor of vertex coordinates,
                                returns a mask of shape (N, ) indicating whether to keep the vertex or not.
            device: Optional device to run the model on, if None, use self.device
        Returns:
            dict with keys:
                grid_bound: (2, 3) tensor of boundary min and max
                grid_shape: (3, ) tensor of grid shape (nx, ny, nz)
                grid_resolution: (1, ) tensor of grid resolution
                mask: (nx, ny, nz) boolean tensor, where True indicates the vertex is valid. None if no filter is applied.
                sdf_prior: (nx, ny, nz) tensor of SDF prior values on the grid
                sdf_residual: (nx, ny, nz) tensor of SDF residual values on the grid
                sdf: (nx, ny, nz) tensor of final SDF values on the grid
        """
        x = torch.arange(bound_min[0], bound_max[0], grid_resolution)
        y = torch.arange(bound_min[1], bound_max[1], grid_resolution)
        z = torch.arange(bound_min[2], bound_max[2], grid_resolution)
        grid_points = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)

        mask = None
        if grid_vertex_filter is None:
            results = self.model_forward_func(self.model, grid_points, False, True, 0.0, device)
            # prediction has shape (nx, ny, nz)
        else:
            mask = grid_vertex_filter(grid_points)  # (nx, ny, nz) boolean mask
            results = self.model_forward_func(self.model, grid_points[mask], False, True, 0.0, device)
            # prediction has shape (n_valid, )
            if len(results) == 0:
                return dict()

        results["grid_bound"] = torch.tensor([bound_min, bound_max])
        results["grid_shape"] = torch.tensor(grid_points.shape[:-1], dtype=torch.long)
        results["grid_resolution"] = torch.tensor(grid_resolution)
        results["mask"] = mask.to(self.device if device is None else device) if mask is not None else None

        return results

    @torch.no_grad()
    def extract_mesh(
        self,
        bound_min: List[float],
        bound_max: List[float],
        grid_resolution: float,
        fields: List[str],
        iso_value: float,
        grid_vertex_filter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Extract mesh from the model using marching cubes.
        Args:
            bound_min: Minimum bound of the 3D grid (list of 3 floats)
            bound_max: Maximum bound of the 3D grid (list of 3 floats)
            grid_resolution: Resolution of the grid (float)
            fields: List of fields to extract
            iso_value: Iso value for marching cubes (float)
            grid_vertex_filter: Optional function to filter grid vertices, takes (N, 3) tensor of vertex coordinates,
                                returns a mask of shape (N,) indicating whether to keep the vertex or not.
        Returns:
            list of open3d.geometry.TriangleMesh: Extracted mesh
        """

        self.model.eval()

        sdf_grid = self.extract_sdf_grid(
            bound_min=bound_min,
            bound_max=bound_max,
            grid_resolution=grid_resolution,
            grid_vertex_filter=grid_vertex_filter,
        )

        mask = None
        if grid_vertex_filter is not None:
            mask = sdf_grid["mask"].cpu().numpy().astype(np.bool_)

        meshes: List[o3d.geometry.TriangleMesh] = []
        for field in fields:
            assert field in sdf_grid, f"Field {field} not found in model output"
            values = sdf_grid[field].cpu().numpy().astype(np.float64)
            mc = MarchingCubes()

            if mask is None:
                grid_values = values  # (nx, ny, nz)
            else:
                grid_shape = sdf_grid["grid_shape"].cpu().numpy().astype(np.int32)
                grid_values = np.ones(grid_shape, dtype=np.float64)
                grid_values[mask] = values  # (nx, ny, nz)

            vertices, triangles, triangle_normals = mc.run(
                coords_min=bound_min,
                grid_res=[grid_resolution, grid_resolution, grid_resolution],
                grid_shape=grid_values.shape,
                grid_values=grid_values.flatten(),
                mask=mask.flatten() if mask is not None else None,
                iso_value=iso_value,
                row_major=True,
                parallel=True,
            )

            # vertices: (3, n_vertices)
            # triangles: (3, n_faces)
            # triangle_normals: (3, n_faces)

            # save mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices.T)
            mesh.triangles = o3d.utility.Vector3iVector(triangles.T)
            mesh.triangle_normals = o3d.utility.Vector3dVector(triangle_normals.T)

            meshes.append(mesh)

        return meshes

    @torch.no_grad()
    def extract_slice(
        self,
        axis: int,
        pos: float,
        resolution: float,
        bound_min: List[float],
        bound_max: List[float],
        device: Optional[str] = None,
    ):
        """
        Extract a 2D slice of the prediction values along the given axis at the given position.
        Args:
            axis: int, 0 for x, 1 for y, 2 for z
            pos: float, position along the axis to extract the slice
            resolution: float, resolution of the slice
            bound_min: Minimum bound of the 3D grid (list of 3 floats)
            bound_max: Maximum bound of the 3D grid (list of 3 floats)
            device: Optional device to run the model on, if None, use self.device

        Returns:
            dict with keys:
                slice_bound: (2, ) list of boundary min and max for the slice
                sdf_prior: (n, m) torch tensor of SDF prior values on the slice
                sdf_residual: (n, m) torch tensor of SDF residual values on the slice
                sdf: (n, m) torch tensor of final SDF values on the slice
        """
        assert axis in [0, 1, 2], "axis must be 0, 1, or 2"
        self.model.eval()

        if axis == 0:
            y = torch.arange(bound_min[1], bound_max[1], resolution)  # (ny,)
            z = torch.arange(bound_min[2], bound_max[2], resolution)  # (nz,)
            yy, zz = torch.meshgrid(y, z, indexing="xy")
            grid_points = torch.stack((torch.full_like(yy, pos), yy, zz), dim=-1)
            slice_bound = [bound_min[1], bound_min[2]], [bound_max[1], bound_max[2]]
        elif axis == 1:
            x = torch.arange(bound_min[0], bound_max[0], resolution)  # (nx,)
            z = torch.arange(bound_min[2], bound_max[2], resolution)  # (nz,)
            xx, zz = torch.meshgrid(x, z, indexing="xy")
            grid_points = torch.stack((xx, torch.full_like(xx, pos), zz), dim=-1)
            slice_bound = [bound_min[0], bound_min[2]], [bound_max[0], bound_max[2]]
        else:  # axis == 2
            x = torch.arange(bound_min[0], bound_max[0], resolution)  # (nx,)
            y = torch.arange(bound_min[1], bound_max[1], resolution)  # (ny,)
            xx, yy = torch.meshgrid(x, y, indexing="xy")
            grid_points = torch.stack((xx, yy, torch.full_like(xx, pos)), dim=-1)
            slice_bound = [bound_min[0], bound_min[1]], [bound_max[0], bound_max[1]]

        results = self.model_forward_func(self.model, grid_points.to(self.device), False, True, 0.0, device)
        results["slice_bound"] = torch.tensor(slice_bound)
        return results
