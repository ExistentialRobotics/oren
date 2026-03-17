#!/usr/bin/env python3
"""
Extract mesh from C++ SDF model using Marching Cubes.
Runs the C++ grid extraction and then uses Python Marching Cubes to create a mesh.
"""

import argparse
import json
import os
import subprocess

import numpy as np

from grad_sdf import MarchingCubes, o3d


def main():
    parser = argparse.ArgumentParser(description="Extract mesh from C++ SDF model")
    parser.add_argument(
        "--config",
        type=str,
        default="grad-sdf-cpp/models/trainer-ros.yaml",
        help="Path to trainer YAML config (default: grad-sdf-cpp/models/trainer-ros.yaml)",
    )
    parser.add_argument(
        "--bundle",
        type=str,
        default="grad-sdf-cpp/models/bundle",
        help="Path to C++ bundle directory (default: grad-sdf-cpp/models/bundle)",
    )
    parser.add_argument(
        "--cpp-binary",
        type=str,
        default="grad-sdf-cpp/build/extract_mesh",
        help="Path to C++ extract_mesh binary",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for C++ model",
    )
    parser.add_argument(
        "--bound-min",
        type=float,
        nargs=3,
        default=None,
        help="Optional minimum bound override for the 3D grid; defaults to trainer config dataset bounds",
    )
    parser.add_argument(
        "--bound-max",
        type=float,
        nargs=3,
        default=None,
        help="Optional maximum bound override for the 3D grid; defaults to trainer config dataset bounds",
    )
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=0.03,
        help="Resolution of the grid",
    )
    parser.add_argument(
        "--iso-value",
        type=float,
        default=0.0,
        help="Iso value for marching cubes",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./grad-sdf-cpp/tmp/mesh_output",
        help="Workspace-local output directory for mesh and grid files",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run C++ grid extraction
    print("=" * 80)
    print("Step 1: Extracting SDF grid from C++ model")
    print("=" * 80)

    lib_path = "/home/jason/.local/lib/python3.10/site-packages/torch/lib"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [
        args.cpp_binary,
        "--config", args.config,
        "--bundle", args.bundle,
        "--device", args.device,
        "--grid-resolution", str(args.grid_resolution),
        "--output", args.output,
    ]

    if args.bound_min is not None:
        cmd.extend(["--bound-min", str(args.bound_min[0]), str(args.bound_min[1]), str(args.bound_min[2])])
    if args.bound_max is not None:
        cmd.extend(["--bound-max", str(args.bound_max[0]), str(args.bound_max[1]), str(args.bound_max[2])])

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=env, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(f"C++ binary failed with return code {result.returncode}")

    # Load SDF grid
    print("\n" + "=" * 80)
    print("Step 2: Loading SDF grid")
    print("=" * 80)

    grid_file = os.path.join(args.output, "sdf_grid.npy")
    if not os.path.exists(grid_file):
        raise RuntimeError(f"SDF grid file not found: {grid_file}")

    sdf_grid = np.load(grid_file)
    print(f"SDF grid shape: {sdf_grid.shape}")
    print(f"SDF grid range: [{sdf_grid.min():.6f}, {sdf_grid.max():.6f}]")

    # Load metadata
    metadata_file = os.path.join(args.output, "grid_metadata.txt")
    metadata = {}
    with open(metadata_file) as f:
        for line in f:
            key, value = line.strip().split("=")
            if "bound" in key:
                metadata[key] = tuple(map(float, value.split(",")))
            else:
                try:
                    metadata[key] = float(value)
                except ValueError:
                    metadata[key] = value

    print(f"Grid metadata: {metadata}")

    # Extract mesh using Marching Cubes
    print("\n" + "=" * 80)
    print(f"Step 3: Running Marching Cubes with iso_value={args.iso_value}")
    print("=" * 80)

    bound_min = metadata["bound_min"]
    bound_max = metadata["bound_max"]
    grid_resolution = metadata["grid_resolution"]

    mc = MarchingCubes()
    vertices, triangles, triangle_normals = mc.run(
        coords_min=list(bound_min),
        grid_res=[grid_resolution, grid_resolution, grid_resolution],
        grid_shape=sdf_grid.shape,
        grid_values=sdf_grid.flatten(),
        mask=None,
        iso_value=args.iso_value,
        row_major=True,
        parallel=True,
    )

    print(f"Marching Cubes complete:")
    print(f"  Vertices shape: {vertices.shape}")
    print(f"  Triangles shape: {triangles.shape}")
    print(f"  Triangle normals shape: {triangle_normals.shape}")

    # Create and save mesh
    print("\n" + "=" * 80)
    print("Step 4: Saving mesh")
    print("=" * 80)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.T)
    mesh.triangles = o3d.utility.Vector3iVector(triangles.T)
    mesh.triangle_normals = o3d.utility.Vector3dVector(triangle_normals.T)

    # Save as PLY
    mesh_file = os.path.join(args.output, "mesh.ply")
    o3d.io.write_triangle_mesh(mesh_file, mesh)
    print(f"Saved mesh to: {mesh_file}")

    # Compute and save mesh statistics
    print("\n" + "=" * 80)
    print("Step 5: Mesh statistics")
    print("=" * 80)

    vertex_bounds_min = vertices.T.min(axis=0)
    vertex_bounds_max = vertices.T.max(axis=0)
    num_vertices = vertices.shape[1]
    num_faces = triangles.shape[1]
    bbox_volume = np.prod(vertex_bounds_max - vertex_bounds_min).item()

    stats = {
        "num_vertices": int(num_vertices),
        "num_faces": int(num_faces),
        "vertex_bounds_min": vertex_bounds_min.tolist(),
        "vertex_bounds_max": vertex_bounds_max.tolist(),
        "bbox_volume": float(bbox_volume),
        "grid_resolution": float(grid_resolution),
        "iso_value": float(args.iso_value),
        "sdf_grid_range": [float(sdf_grid.min()), float(sdf_grid.max())],
    }

    stats_file = os.path.join(args.output, "mesh_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Mesh statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\nSaved statistics to: {stats_file}")

    print("\n" + "=" * 80)
    print("✓ Mesh extraction complete!")
    print("=" * 80)
    print(f"Output files:")
    print(f"  Mesh: {mesh_file}")
    print(f"  SDF Grid: {grid_file}")
    print(f"  Statistics: {stats_file}")
    print(f"  Metadata: {metadata_file}")


if __name__ == "__main__":
    main()
