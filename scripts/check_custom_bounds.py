#!/usr/bin/env python3
"""
Check that config bounds contain the actual point cloud extent, and
optionally open an Open3D viewer showing the point cloud vs. the config
bounding box.

Usage (from repo root, inside pipenv shell):
    python scripts/check_custom_bounds.py
    python scripts/check_custom_bounds.py garage warehouse
    python scripts/check_custom_bounds.py --visualize garage
    python scripts/check_custom_bounds.py --visualize garage --frames 20
"""
import argparse
import sys
from pathlib import Path

# Use local source, not the installed (outdated) package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oren"))

import open3d as o3d
import torch
from tqdm import tqdm

from oren.dataset.replica import DataLoader

SCENES = {
    "garage": {
        "data_path": "data/garage/replica",
        "min_depth": 0.3,
        "max_depth": 20.0,
        "bound_min": [-18.0, -20.0, -5.0],
        "bound_max": [10.0, 0.0, 2.0],
    },
    "forest": {
        "data_path": "data/forest/replica",
        "min_depth": 0.3,
        "max_depth": 20.0,
        "bound_min": [25.0, 60.0, 10.0],
        "bound_max": [130.0, 145.0, 55.0],
    },
    "industrial": {
        "data_path": "data/industrial/replica",
        "min_depth": 0.3,
        "max_depth": 20.0,
        "bound_min": [-30.0, -50.0, 0.0],
        "bound_max": [90.0, 25.0, 10.0],
    },
    "warehouse": {
        "data_path": "data/warehouse/replica",
        "min_depth": 0.3,
        "max_depth": 20.0,
        "bound_min": [-40.0, -60.0, 0.0],
        "bound_max": [40.0, 60.0, 10.0],
    },
}


def compute_bound(data_path: str, min_depth: float, max_depth: float):
    loader = DataLoader(data_path, min_depth=min_depth, max_depth=max_depth)
    mins, maxs = [], []
    for i in tqdm(range(len(loader)), ncols=100, desc="Computing bounds"):
        frame = loader[i]
        pts = frame.get_points(to_world_frame=True, device="cpu")
        if pts.shape[0] == 0:
            continue
        mins.append(pts.min(dim=0).values)
        maxs.append(pts.max(dim=0).values)
    if not mins:
        raise RuntimeError(f"No valid points found in {data_path}. Check data path and depth range.")
    return torch.stack(mins).min(dim=0).values, torch.stack(maxs).max(dim=0).values


def check_scene(name: str, cfg: dict) -> tuple[bool, list, list]:
    print(f"\n{'='*60}")
    print(f"  {name}  ({cfg['data_path']})")
    print(f"{'='*60}")

    bmin, bmax = compute_bound(cfg["data_path"], cfg["min_depth"], cfg["max_depth"])
    bmin_list = [round(v, 3) for v in bmin.tolist()]
    bmax_list = [round(v, 3) for v in bmax.tolist()]

    cfg_min = cfg["bound_min"]
    cfg_max = cfg["bound_max"]

    print(f"  computed  bound_min : {bmin_list}")
    print(f"  config    bound_min : {cfg_min}")
    print(f"  computed  bound_max : {bmax_list}")
    print(f"  config    bound_max : {cfg_max}")

    ok = True
    for axis, label in enumerate("xyz"):
        if cfg_min[axis] > bmin_list[axis]:
            print(f"  WARNING  {label}-min: config {cfg_min[axis]} > computed {bmin_list[axis]:.3f} — points clipped!")
            ok = False
        if cfg_max[axis] < bmax_list[axis]:
            print(f"  WARNING  {label}-max: config {cfg_max[axis]} < computed {bmax_list[axis]:.3f} — points clipped!")
            ok = False

    if ok:
        margin_min = [round(bmin_list[i] - cfg_min[i], 3) for i in range(3)]
        margin_max = [round(cfg_max[i] - bmax_list[i], 3) for i in range(3)]
        print(f"  OK — config box contains all points")
        print(f"  margin (min side) : {margin_min}")
        print(f"  margin (max side) : {margin_max}")

    return ok, bmin_list, bmax_list


def visualize_scene(name: str, cfg: dict, n_frames: int):
    print(f"\nLoading {n_frames} frames for visualization...")
    loader = DataLoader(cfg["data_path"], min_depth=cfg["min_depth"], max_depth=cfg["max_depth"])
    step = max(1, len(loader) // n_frames)

    pcd_pts = []
    for i in range(0, len(loader), step):
        frame = loader[i]
        pts = frame.get_points(to_world_frame=True, device="cpu")
        if pts.shape[0] > 0:
            pcd_pts.append(pts)

    if not pcd_pts:
        print("No valid points to visualize.")
        return

    pts = torch.cat(pcd_pts).numpy()
    print(f"  {len(pts):,} points from {len(pcd_pts)} frames")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    cfg_box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=cfg["bound_min"],
        max_bound=cfg["bound_max"],
    )
    cfg_box.color = [1.0, 0.0, 0.0]

    actual_box = pcd.get_axis_aligned_bounding_box()
    actual_box.color = [0.0, 0.8, 0.0]

    print(f"\nOpening viewer for '{name}'")
    print(f"  Red box   = config bounds")
    print(f"  Green box = actual point cloud extent")
    o3d.visualization.draw_geometries(
        [pcd, cfg_box, actual_box],
        window_name=f"Bounds check: {name}",
        width=1280,
        height=720,
    )


def main():
    parser = argparse.ArgumentParser(description="Check config bounds against actual point cloud extent")
    parser.add_argument(
        "scenes",
        nargs="*",
        default=list(SCENES.keys()),
        choices=list(SCENES.keys()),
        help="Which scenes to check (default: all)",
    )
    parser.add_argument(
        "--visualize",
        metavar="SCENE",
        choices=list(SCENES.keys()),
        help="Open an Open3D viewer for this scene after checking",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=15,
        help="Number of frames to load for visualization (default: 15)",
    )
    args = parser.parse_args()

    results = {}
    for name in args.scenes:
        ok, _, _ = check_scene(name, SCENES[name])
        results[name] = ok

    print(f"\n{'='*60}")
    print("Summary:")
    for name, ok in results.items():
        print(f"  {name:12s}: {'OK' if ok else 'CLIPPED'}")

    if args.visualize:
        visualize_scene(args.visualize, SCENES[args.visualize], args.frames)


if __name__ == "__main__":
    main()
