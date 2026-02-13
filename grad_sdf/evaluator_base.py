import os
from typing import Callable, Dict, List, Optional

import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial import cKDTree

from grad_sdf import MarchingCubes


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
        """
        assert model_forward_func is not None
        self.model_forward_func = model_forward_func
        self.device = device

        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
        else:
            assert model_path is not None
            assert model_create_func is not None

            self.model: torch.nn.Module = model_create_func(model_path).to(self.device)
            self.model.eval()

    @staticmethod
    def _sdf_metrics(sdf_pred: torch.Tensor, sdf_gt: torch.Tensor):
        diff = sdf_pred - sdf_gt
        return dict(mae=diff.abs().mean().item(), rmse=(diff**2).mean().sqrt().item())

    @staticmethod
    def _grad_metrics(grad_pred: torch.Tensor, grad_gt: torch.Tensor):
        grad_pred /= grad_pred.norm(dim=-1, keepdim=True) + 1e-8
        grad_gt /= grad_gt.norm(dim=-1, keepdim=True) + 1e-8
        cos_sim = (grad_pred * grad_gt).sum(dim=-1)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        angle_diff = torch.acos(cos_sim)  # in radians
        return angle_diff.abs().mean().item()

    def sdf_and_grad_metrics(
        self,
        test_set_dir: str,
        sdf_fields: list[str],
        grad_method: str = "autograd",
        eps: float = 0.001,
    ):
        """计算 SDF 和梯度指标"""

        # 辅助函数：加载数据
        def load_data(prefix: str):
            points = torch.from_numpy(np.load(os.path.join(test_set_dir, f"{prefix}.npy"))).float().to(self.device)

            gt_sdf = (
                torch.from_numpy(
                    np.load(os.path.join(test_set_dir, f"gt_sdf_values_{prefix.replace('points_', '')}.npy"))
                )
                .float()
                .to(self.device)
            )

            gt_grad = (
                torch.from_numpy(np.load(os.path.join(test_set_dir, f"grad_gt_{prefix.replace('points_', '')}.npy")))
                .float()
                .to(self.device)
            )

            return points, gt_sdf, gt_grad

        # 加载三组数据
        data_sets = {}
        for name, prefix in [
            ("for_eval", "points_for_eval"),
            ("near_surface", "points_near_surface"),
            ("far_surface", "points_far_surface"),
        ]:
            try:
                data_sets[name] = load_data(prefix)
            except FileNotFoundError as e:
                print(f"Warning: Could not load data for {name}: {e}")
                continue

        autograd = grad_method == "autograd"
        self.model.eval()

        # 计算指标
        sdf_metrics = {}
        grad_metrics = {}

        for name, (points, gt_sdf, gt_grad) in data_sets.items():
            result = self.model_forward_func(self.model, points, True, autograd, eps, None)

            # 计算 SDF 指标
            sdf_metrics[name] = {k: self._sdf_metrics(result[k], gt_sdf) for k in sdf_fields if k in result}

            # 计算梯度指标（添加键存在检查）
            if "grad" in result:
                grad_metrics[name] = {
                    k: self._grad_metrics(result["grad"][k], gt_grad) for k in sdf_fields if k in result["grad"]
                }
            else:
                grad_metrics[name] = {}

        return dict(sdf_metrics=sdf_metrics, grad_metrics=grad_metrics)

    @staticmethod
    def mesh_metrics(
        pred_mesh_path: str,
        gt_mesh_path: str,
        gt_mesh_offset: List[float] | None = None,
        threshold: float = 0.05,
        num_samples: int = 200_000,
        seed: int = 0,
    ):
        pred_mesh = trimesh.load_mesh(pred_mesh_path)
        gt_mesh = trimesh.load_mesh(gt_mesh_path)
        if gt_mesh_offset is not None:
            gt_mesh.apply_translation(gt_mesh_offset)

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

    def compute_metrics(
        self,
        test_set_dir: str,
        sdf_fields: List[str],
        grad_method: str,
        eps: float,
        pred_mesh_path: str,
        gt_mesh_path: str,
        gt_mesh_offset: List[float] | None = None,
        threshold: float = 0.05,
        num_samples: int = 200_000,
        seed: int = 0,
    ):
        metrics: dict = self.sdf_and_grad_metrics(test_set_dir, sdf_fields, grad_method, eps)
        metrics["mesh_metrics"] = self.mesh_metrics(
            pred_mesh_path,
            gt_mesh_path,
            gt_mesh_offset,
            threshold,
            num_samples,
            seed,
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

        # Convert bound_min and bound_max to tensors if they aren't already
        bound_min_tensor = torch.as_tensor(bound_min) if not isinstance(bound_min, torch.Tensor) else bound_min
        bound_max_tensor = torch.as_tensor(bound_max) if not isinstance(bound_max, torch.Tensor) else bound_max
        results["grid_bound"] = torch.stack([bound_min_tensor, bound_max_tensor])
        results["grid_shape"] = torch.tensor(grid_points.shape[:-1], dtype=torch.long)
        results["grid_resolution"] = torch.tensor(grid_resolution)
        if mask is not None:
            results["mask"] = mask.to(self.device if device is None else device)  # type: ignore

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
