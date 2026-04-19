import os
from typing import Callable, Dict, Optional

import pandas as pd
from tqdm import tqdm

from grad_sdf import np, o3d, torch
from grad_sdf.evaluator_base import EvaluatorBase
from grad_sdf.model import SdfNetwork, SdfNetworkConfig
from grad_sdf.utils.dict_util import flatten_dict


class GradSdfEvaluator(EvaluatorBase):
    """
    Evaluator for GradSDF models.
    """

    def __init__(
        self,
        batch_size: int,
        clean_mesh: bool = True,
        model_cfg: SdfNetworkConfig | None = None,
        model: torch.nn.Module | None = None,
        model_path: str | None = None,
        model_input_offset: list[float] = None,
        device: str = "cuda",
        absolute_sdf: bool = False,
        grad_err_outlier_threshold: float = 0.5,
        interactive: bool = False,
    ):
        """
        Args:
            batch_size: batch size for model inference
            clean_mesh: whether to clean mesh by removing voxels that are too big
            model_cfg: configuration for the SdfNetwork, required if model is not provided
            model: optional, if provided, use this model
            model_path: optional, if model is not provided, load the model from this path
            model_input_offset: optional offset to apply to the model input
            device: device to run the model on
            absolute_sdf: whether to take absolute value of SDF for metrics computation
            grad_err_outlier_threshold: threshold to filter out outliers in gradient error when computing metrics
            interactive: whether to run some interactive visualization
        """
        self.batch_size = batch_size
        self.clean_mesh = clean_mesh
        self.model_cfg = model_cfg
        self.model_input_offset = model_input_offset

        super().__init__(
            self.forward_model,
            model,
            model_path,
            self.create_model,
            device=device,
            absolute_sdf=absolute_sdf,
            grad_err_outlier_threshold=grad_err_outlier_threshold,
            interactive=interactive,
        )

        self.model: SdfNetwork

    def create_model(self, model_path: str) -> torch.nn.Module:
        """
        Create the model and load weights from the given path.
        Args:
            model_path: path to the model weights
        Returns:
            torch.nn.Module: the created model
        """
        assert self.model_cfg is not None
        tqdm.write("Creating model...")
        model = SdfNetwork(self.model_cfg)
        model.to(self.device)
        tqdm.write(f"Loading model weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def forward_model(
        self,
        model,
        points: torch.Tensor,
        get_grad: bool,
        auto_grad: bool = True,
        finite_diff_eps: float = 0.01,
        device: str = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward the model with the given points.
        Args:
            model: the SdfNetwork model
            points: (..., 3) points in world coordinates
            get_grad: whether to compute gradients
            auto_grad: if get_grad is True, whether to use autograd to compute gradients
            finite_diff_eps: if get_grad is True and auto_grad is False, the epsilon for finite difference
            device: if specified, move the output tensors to this device

        Returns:
            dict of output tensors with keys:
                'voxel_indices': (...,) voxel indices for the points
                'sdf_prior': (...,) SDF prior from the octree
                'sdf_residual': (...,) SDF residual from the residual network
                'sdf': (...,) final SDF values (prior + residual)
        """
        if points.shape[0] == 0:
            return dict()

        if self.batch_size <= 0:
            bs = points.shape[0]
        else:
            bs = int(self.batch_size * 3 * points.shape[0] / points.numel()) + 1

        if self.model_input_offset is not None:
            points = points + torch.tensor(self.model_input_offset, device=points.device).to(points.dtype)

        voxel_indices = []
        sdf_prior = []
        sdf_residual = []
        sdf_pred = []
        sdf_prior_grad = []
        sdf_grad = []

        for i in tqdm(range(0, points.shape[0], bs), desc="Batches", ncols=120, position=1, leave=False):
            j = min(i + bs, points.shape[0])
            points_batch = points[i:j].to(self.device)
            points_batch.requires_grad_(auto_grad)
            voxel_indices_batch, sdf_prior_batch, sdf_residual_batch, _ = model(points_batch)
            sdf_pred_batch = sdf_prior_batch + sdf_residual_batch

            if get_grad:
                if auto_grad:
                    sdf_grad_batch = torch.autograd.grad(
                        outputs=[sdf_pred_batch],
                        inputs=[points_batch],
                        grad_outputs=[torch.ones_like(sdf_pred_batch)],
                        create_graph=True,
                        allow_unused=True,
                    )[0]
                    sdf_prior_grad_batch = torch.autograd.grad(
                        outputs=[sdf_prior_batch],
                        inputs=[points_batch],
                        grad_outputs=[torch.ones_like(sdf_prior_batch)],
                        create_graph=True,
                        allow_unused=True,
                    )[0]
                    sdf_prior_grad.append(sdf_prior_grad_batch.detach().cpu())
                    sdf_grad.append(sdf_grad_batch.detach().cpu())
                else:
                    sdf_grad_batch = torch.empty_like(points_batch)
                    sdf_prior_grad_batch = torch.empty_like(points_batch)
                    for k in range(3):
                        offset = torch.zeros((3,), device=points_batch.device)
                        offset[k] = finite_diff_eps
                        offset = offset.view(*[1] * (points_batch.ndim - 1), 3)
                        _, sdf_prior_plus, _, sdf_plus = model(points_batch + offset)
                        _, sdf_prior_minus, _, sdf_minus = model(points_batch - offset)
                        sdf_grad_batch[..., k] = (sdf_plus - sdf_minus) / (2 * finite_diff_eps)
                        sdf_prior_grad_batch[..., k] = (sdf_prior_plus - sdf_prior_minus) / (2 * finite_diff_eps)
                    sdf_prior_grad.append(sdf_prior_grad_batch.detach().cpu())
                    sdf_grad.append(sdf_grad_batch.detach().cpu())

            voxel_indices.append(voxel_indices_batch.detach().cpu())
            sdf_prior.append(sdf_prior_batch.detach().cpu())
            sdf_residual.append(sdf_residual_batch.detach().cpu())
            sdf_pred.append(sdf_pred_batch.detach().cpu())

        if device is None:
            device = self.device
        if len(sdf_prior) == 1:
            voxel_indices = voxel_indices[0].to(device)
            sdf_prior = sdf_prior[0].to(device)
            sdf_residual = sdf_residual[0].to(device)
            sdf_pred = sdf_pred[0].to(device)
            if get_grad:
                sdf_prior_grad = sdf_prior_grad[0].to(device)
                sdf_grad = sdf_grad[0].to(device)
        else:
            voxel_indices = torch.cat(voxel_indices, dim=0).to(device)
            sdf_prior = torch.cat(sdf_prior, dim=0).to(device)
            sdf_residual = torch.cat(sdf_residual, dim=0).to(device)
            sdf_pred = torch.cat(sdf_pred, dim=0).to(device)
            if get_grad:
                sdf_prior_grad = torch.cat(sdf_prior_grad, dim=0).to(device)
                sdf_grad = torch.cat(sdf_grad, dim=0).to(device)

        result = dict(voxel_indices=voxel_indices, sdf_prior=sdf_prior, sdf_residual=sdf_residual, sdf=sdf_pred)
        if get_grad:
            result["grad"] = dict(sdf_prior=sdf_prior_grad, sdf=sdf_grad)
        return result

    @torch.no_grad()
    def extract_mesh(
        self,
        bound_min: list[float],
        bound_max: list[float],
        grid_resolution: float,
        fields: list[str] = None,
        iso_value: float = 0.0,
        grid_vertex_filter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        if self.clean_mesh and grid_vertex_filter is None:
            grid_vertex_filter = self.model.grid_vertex_filter

        if fields is None:
            fields = ["sdf_prior", "sdf"]

        return super().extract_mesh(
            bound_min=bound_min,
            bound_max=bound_max,
            grid_resolution=grid_resolution,
            fields=fields,
            iso_value=iso_value,
            grid_vertex_filter=grid_vertex_filter,
        )


def main():
    import argparse

    from grad_sdf.trainer_config import TrainerConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--batch-size", type=int, default=40960)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--apply-dataset-offset", action="store_true")

    parser.add_argument("--extract-grid", action="store_true")
    parser.add_argument("--grid-resolution", type=float, default=0.0125)
    parser.add_argument("--bound-min", type=float, nargs=3)
    parser.add_argument("--bound-max", type=float, nargs=3)
    parser.add_argument("--extract-fields", type=str, nargs="+", default=["sdf", "sdf_prior"])

    parser.add_argument("--extract-mesh", action="store_true")
    parser.add_argument("--clean-mesh", action="store_true")
    parser.add_argument("--iso-value", type=float, default=0.0)

    parser.add_argument("--sdf-and-grad-metrics", action="store_true")
    parser.add_argument("--absolute-sdf", action="store_true", help="Whether to take absolute value of SDF for metrics computation")
    parser.add_argument("--test-set-dir", type=str, help="Directory of the test set")
    parser.add_argument("--sdf-fields", type=str, nargs="+", default=["sdf", "sdf_prior"])
    parser.add_argument("--grad-method", type=str, default="autograd", choices=["autograd", "finite_difference"])
    parser.add_argument("--finite-difference-eps", type=float, default=0.001)
    parser.add_argument("--grad-err-outlier-threshold", type=float, default=np.deg2rad(25).item())

    parser.add_argument("--mesh-metrics", action="store_true")
    parser.add_argument("--pred-mesh-paths", type=str, nargs="+")
    parser.add_argument("--gt-mesh-path", type=str, help="Path to the ground truth mesh file")
    parser.add_argument("--gt-pcd-path", type=str, help="Path to the ground truth point cloud file")
    parser.add_argument("--f1-threshold", type=float, default=0.05)
    parser.add_argument("--num-points", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interactive", action="store_true")

    args = parser.parse_args()

    trainer_cfg = TrainerConfig.from_yaml(args.config)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(args.config)), "eval")
        output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    offset = trainer_cfg.data.dataset_args.get("offset", None)

    evaluator = GradSdfEvaluator(
        batch_size=args.batch_size,
        clean_mesh=args.clean_mesh,
        model_cfg=trainer_cfg.model,
        model_path=args.model_path,
        # add offset to the model input if specified in the dataset args
        # the test data is not offset, so we need to offset the model input
        model_input_offset=offset if args.apply_dataset_offset else None,
        device=args.device,
        grad_err_outlier_threshold=args.grad_err_outlier_threshold,
        interactive=args.interactive,
    )

    if args.extract_grid:
        bound_min = args.bound_min
        bound_max = args.bound_max
        if bound_min is None:
            bound_min = trainer_cfg.model.residual_net_cfg.bound_min
        if bound_max is None:
            bound_max = trainer_cfg.model.residual_net_cfg.bound_max
        results = evaluator.extract_sdf_grid(
            bound_min=bound_min,
            bound_max=bound_max,
            grid_resolution=args.grid_resolution,
        )
        for field_name in args.extract_fields:
            assert field_name in results, f"Field {field_name} not found in model output"
            grid = results[field_name].cpu().numpy()
            grid_file = os.path.join(output_dir, f"grid_{field_name}.npy")
            with open(grid_file, "wb") as f:
                np.save(f, grid)
            tqdm.write(f"Saved SDF grid ({field_name}) to {grid_file}")

    mesh_files = []
    if args.extract_mesh:
        bound_min = args.bound_min
        bound_max = args.bound_max
        if bound_min is None:
            bound_min = trainer_cfg.model.residual_net_cfg.bound_min
        if bound_max is None:
            bound_max = trainer_cfg.model.residual_net_cfg.bound_max
        meshes = evaluator.extract_mesh(
            bound_min=bound_min,
            bound_max=bound_max,
            grid_resolution=args.grid_resolution,
            fields=args.extract_fields,
            iso_value=args.iso_value,
        )
        for mesh_name, mesh in zip(args.extract_fields, meshes):
            mesh_file = os.path.join(output_dir, f"mesh_{mesh_name}.ply")
            mesh_files.append(mesh_file)
            o3d.io.write_triangle_mesh(mesh_file, mesh)
            tqdm.write(f"Saved mesh ({mesh_name}) to {mesh_file}")

    if args.sdf_and_grad_metrics:
        assert args.test_set_dir is not None
        metrics = evaluator.sdf_and_grad_metrics(
            test_set_dir=args.test_set_dir,
            sdf_fields=args.sdf_fields,
            grad_method=args.grad_method,
            eps=args.finite_difference_eps,
        )
        metrics = flatten_dict(metrics)
        columns = list(metrics.keys())
        df = pd.DataFrame(columns=columns)
        df.loc[0] = [metrics[k] for k in columns]
        csv_file = os.path.join(output_dir, "sdf_and_grad_metrics.csv")
        df.to_csv(csv_file, index=False)
        tqdm.write(f"Saved metrics to {csv_file}")
        for k, v in metrics.items():
            tqdm.write(f"{k}: {v}")

    if args.mesh_metrics:
        pred_mesh_paths = args.pred_mesh_paths
        if pred_mesh_paths is None or len(pred_mesh_paths) == 0:
            assert (
                len(mesh_files) > 0
            ), "No extracted meshes found. Please provide --pred-mesh-paths or use --extract-mesh"
            pred_mesh_paths = mesh_files

        use_pcd_gt = args.gt_pcd_path is not None
        if use_pcd_gt:
            assert os.path.exists(args.gt_pcd_path), f"Ground truth point cloud file {args.gt_pcd_path} does not exist"
        else:
            assert args.gt_mesh_path is not None, "Provide --gt-mesh-path or --gt-pcd-path for mesh metrics"
            assert os.path.exists(args.gt_mesh_path), f"Ground truth mesh file {args.gt_mesh_path} does not exist"

        df = None
        for pred_mesh_path in pred_mesh_paths:
            assert os.path.exists(pred_mesh_path), f"Predicted mesh file {pred_mesh_path} does not exist"

            if use_pcd_gt:
                mesh_metrics = evaluator.mesh_metrics_pointcloud_gt(
                    pred_mesh_path=pred_mesh_path,
                    gt_pointcloud_path=args.gt_pcd_path,
                    gt_pointcloud_offset=offset,
                    threshold=args.f1_threshold,
                    num_samples=args.num_points,
                    seed=args.seed,
                )
            else:
                mesh_metrics = evaluator.mesh_metrics(
                    pred_mesh_path=pred_mesh_path,
                    gt_mesh_path=args.gt_mesh_path,
                    gt_mesh_offset=offset,
                    threshold=args.f1_threshold,
                    num_samples=args.num_points,
                    seed=args.seed,
                )
            mesh_metrics = flatten_dict(mesh_metrics)

            if df is None:
                columns = ["mesh_file"] + list(mesh_metrics.keys())
                df = pd.DataFrame(columns=columns)

            df.loc[len(df)] = [os.path.basename(pred_mesh_path)] + [mesh_metrics[k] for k in mesh_metrics.keys()]
            tqdm.write(f"Metrics for {pred_mesh_path}:")
            for k, v in mesh_metrics.items():
                tqdm.write(f"  {k}: {v}")
        csv_file = os.path.join(output_dir, "mesh_metrics.csv")
        df.to_csv(csv_file, index=False)
        tqdm.write(f"Saved metrics to {csv_file}")


if __name__ == "__main__":
    main()
