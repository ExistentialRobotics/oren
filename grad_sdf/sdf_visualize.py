import trimesh
from grad_sdf.evaluator_grad_sdf import GradSdfEvaluator
import argparse
from grad_sdf.trainer_config import TrainerConfig
import numpy as np
from pysdf import SDF
import matplotlib.pyplot as plt


def read_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path)
    return mesh


def compute_bound(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    bound_min = np.array(mesh.bounds[0])
    bound_max = np.array(mesh.bounds[1])
    return bound_min, bound_max


def calculate_gt_sdf(mesh: trimesh.Trimesh, grid_points: np.ndarray) -> np.ndarray:
    f = SDF(mesh.vertices, mesh.faces)
    grid_points_flat = grid_points.reshape(-1, 3)
    gt_sdf = -f(grid_points_flat)
    gt_sdf = gt_sdf.reshape(grid_points.shape[:-1])
    return gt_sdf


def get_slice_points(
    bound_min: np.ndarray,
    bound_max: np.ndarray,
    grid_resolution: float,
    slice_axis: str = "z",
    slice_index: int | None = None,
):
    axis_map = {"x": 0, "y": 1, "z": 2}
    if slice_axis not in axis_map:
        raise ValueError(f"slice_axis must be one of {list(axis_map)}, got {slice_axis}")
    axis = axis_map[slice_axis]

    coords = [
        np.arange(bound_min[0], bound_max[0], grid_resolution),
        np.arange(bound_min[1], bound_max[1], grid_resolution),
        np.arange(bound_min[2], bound_max[2], grid_resolution),
    ]
    axis_vals = coords[axis]
    if slice_index is None:
        slice_index = len(axis_vals) // 2
    if not (0 <= slice_index < len(axis_vals)):
        raise ValueError(f"slice_index out of range for axis {slice_axis}")
    pos = float(axis_vals[slice_index])

    if axis == 0:
        y = coords[1]
        z = coords[2]
        yy, zz = np.meshgrid(y, z, indexing="xy")
        grid_points = np.stack((np.full_like(yy, pos), yy, zz), axis=-1)
        extent = [float(y.min()), float(y.max()), float(z.min()), float(z.max())]
        x_label, y_label = "y", "z"
    elif axis == 1:
        x = coords[0]
        z = coords[2]
        xx, zz = np.meshgrid(x, z, indexing="xy")
        grid_points = np.stack((xx, np.full_like(xx, pos), zz), axis=-1)
        extent = [float(x.min()), float(x.max()), float(z.min()), float(z.max())]
        x_label, y_label = "x", "z"
    else:
        x = coords[0]
        y = coords[1]
        xx, yy = np.meshgrid(x, y, indexing="xy")
        grid_points = np.stack((xx, yy, np.full_like(xx, pos)), axis=-1)
        extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
        x_label, y_label = "x", "y"

    return grid_points, pos, extent, x_label, y_label


def visualize_slice(
    data: np.ndarray,
    extent: list[float],
    slice_axis: str,
    pos: float,
    x_label: str,
    y_label: str,
    title_prefix: str,
):
    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
    )
    ax.set_title(f"{title_prefix} at {slice_axis}={pos:.3f}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.colorbar(img, ax=ax, label=title_prefix)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-mesh-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--slice-axis", type=str, default="z")
    parser.add_argument("--slice-index", type=int, default=None)
    parser.add_argument("--resolution", type=float, default=0.2)
    args = parser.parse_args()

    mesh = read_mesh(args.gt_mesh_path)
    bound_min, bound_max = compute_bound(mesh)
    print(f"bound_min: {bound_min}, bound_max: {bound_max}")
    trainer_cfg = TrainerConfig.from_yaml(args.config)
    evaluator = GradSdfEvaluator(
        batch_size=40960,
        clean_mesh=True,
        model_cfg=trainer_cfg.model,
        model_path=args.model_path,
        device="cuda",
    )
    grid_points, slice_pos, extent, x_label, y_label = get_slice_points(
        bound_min,
        bound_max,
        args.resolution,
        slice_axis=args.slice_axis,
        slice_index=args.slice_index,
    )
    gt_sdf = calculate_gt_sdf(mesh, grid_points)

    axis_map = {"x": 0, "y": 1, "z": 2}
    slice_results = evaluator.extract_slice(
        axis=axis_map[args.slice_axis],
        pos=slice_pos,
        resolution=args.resolution,
        bound_min=bound_min,
        bound_max=bound_max,
    )

    prior_pred = slice_results["sdf_prior"].cpu().numpy()
    pred = slice_results["sdf"].cpu().numpy()
    prior_error = prior_pred - gt_sdf
    prior_error = np.clip(prior_error, -0.5, 0.5)
    pred_error = pred - gt_sdf
    pred_error = np.clip(pred_error, -0.5, 0.5)
    print(f"gt_sdf.shape: {gt_sdf.shape}, pred_slice.shape: {prior_pred.shape}")

    visualize_slice(gt_sdf, extent, args.slice_axis, slice_pos, x_label, y_label, "gt_sdf")
    visualize_slice(prior_pred, extent, args.slice_axis, slice_pos, x_label, y_label, "sdf_prior")
    visualize_slice(pred, extent, args.slice_axis, slice_pos, x_label, y_label, "sdf")
    visualize_slice(prior_error, extent, args.slice_axis, slice_pos, x_label, y_label, "prior_error")
    visualize_slice(pred_error, extent, args.slice_axis, slice_pos, x_label, y_label, "pred_error")


if __name__ == "__main__":
    main()
