import argparse
import os

import numpy as np
import open3d as o3d
import transforms3d as t3d
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-dir",
    type=str,
    required=True,
    help="Path to the dataset directory containing meshes and traj.txt files.",
)
parser.add_argument(
    "--output-dir",
    type=str,
    required=True,
    help="Path to save the processed meshes and traj.txt files.",
)
args = parser.parse_args()

mesh_path_list = [
    "office0_mesh.ply",
    "office1_mesh.ply",
    "office2_mesh.ply",
    "office3_mesh.ply",
    "office4_mesh.ply",
    "room0_mesh.ply",
    "room1_mesh.ply",
    "room2_mesh.ply",
]

os.makedirs(args.output_dir, exist_ok=True)

for mesh_path in mesh_path_list:
    output_mesh = os.path.join(args.output_dir, mesh_path)
    output_traj = os.path.join(args.output_dir, mesh_path.replace("_mesh.ply", "/traj.txt"))
    os.makedirs(os.path.dirname(output_traj), exist_ok=True)

    mesh_path = os.path.join(args.dataset_dir, mesh_path)
    traj_path = mesh_path.replace("_mesh.ply", "/traj.txt")

    if not os.path.exists(mesh_path):
        print(f"Mesh file not found, skip: {mesh_path}")
        continue

    print(mesh_path)
    tri_mesh = trimesh.load(mesh_path)
    mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(tri_mesh.vertices).astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tri_mesh.faces).astype(np.int32))
    mesh.vertex_normals = o3d.utility.Vector3dVector(np.array(tri_mesh.vertex_normals).astype(np.float64))
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()

    obb = mesh.get_minimal_oriented_bounding_box()

    pt8 = np.asarray(obb.get_box_points())

    print("center:", obb.center)
    print("extent (width, height, depth):", obb.extent)
    print("rotation matrix:\n", obb.R)
    print("8 corner points:\n", pt8)

    score = obb.R.T @ np.array([0, 0, 1])  # which axis is most aligned with world z-axis
    axis_idx = np.argmax(np.abs(score))
    print("axis_idx:", axis_idx, "score:", score)
    if axis_idx != 2 or score[axis_idx] < 0:
        # calculate extra rotation to make the z-axis up
        new_up_axis = np.eye(3)[axis_idx]
        if score[axis_idx] < 0:
            new_up_axis = -new_up_axis
        target_up_axis = np.array([0, 0, 1])
        v = np.cross(new_up_axis, target_up_axis)
        c = np.dot(new_up_axis, target_up_axis)
        s = np.linalg.norm(v)
        if s < 1e-5:
            if c > 0:
                R2 = np.eye(3)
            else:
                R2 = t3d.axangles.axangle2mat(np.array([1, 0, 0]), np.pi)
        else:
            R2 = t3d.axangles.axangle2mat(v / s, np.arctan2(s, c))
        print("extra rotation to make z-axis up:\n", R2)
        print("R2 @ new up axis:", R2 @ new_up_axis)
        R = R2 @ obb.R.T
    else:
        R = obb.R.T

    # transform the mesh to canonical pose
    mesh_rotated = o3d.geometry.TriangleMesh(mesh)
    mesh_rotated.translate(-obb.center)  # shift the mesh to origin first
    mesh_rotated.rotate(R, center=(0, 0, 0))  # then rotate

    print("transformed mesh bounding box:")
    aabb_transformed = mesh_rotated.get_axis_aligned_bounding_box()
    print("transformed AABB min value:", aabb_transformed.min_bound)
    print("transformed AABB max value:", aabb_transformed.max_bound)
    print("transformed AABB range:", aabb_transformed.max_bound - aabb_transformed.min_bound)
    print("original OBB range:", obb.extent)
    bound_min = aabb_transformed.min_bound - 0.15
    bound_max = aabb_transformed.max_bound + 0.15
    bound = [[round(float(mn), 2), round(float(mx), 2)] for mn, mx in zip(bound_min, bound_max)]
    print("bound:", bound)

    # save the rotated mesh as the output filename
    success = o3d.io.write_triangle_mesh(output_mesh, mesh_rotated)
    if success:
        print(f"rotated mesh has been saved to: {output_mesh}")
    else:
        print(f"save failed: {output_mesh}")

    # process the corresponding traj.txt file if exists
    camera_poses = []
    if os.path.exists(traj_path):
        print(f"processing traj.txt file: {traj_path}")

        with open(traj_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            if not line.strip():
                continue
            values = list(map(float, line.strip().split()))
            if len(values) == 16:
                matrix = np.array(values).reshape(4, 4)
                camera_poses.append(matrix)

        print(f"Read {len(camera_poses)} camera poses")

        # apply the same transformation to each camera pose
        transformed_poses = []
        for pose in camera_poses:
            R_cam = pose[:3, :3]
            t_cam = pose[:3, 3]

            new_pose = np.eye(4)
            new_pose[:3, :3] = R @ R_cam
            new_pose[:3, 3] = R @ (t_cam - obb.center)

            transformed_poses.append(new_pose.flatten())
        transformed_poses = np.stack(transformed_poses, axis=0)

        np.savetxt(output_traj, transformed_poses, delimiter=" ")
        print(f"Transformed camera extrinsic have been saved to: {output_traj}")
    else:
        print(f"Camera extrinsic file not found: {traj_path}")

    # visualization for verification

    # visualize the original mesh and OBB
    obb.color = [1, 0, 0]  # red
    world_axis = o3d.geometry.TriangleMesh().create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    obb_axis = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    obb_axis.rotate(R.T, center=(0, 0, 0))
    obb_axis.translate(translation=obb.center)

    traj_lines = o3d.geometry.LineSet()
    if os.path.exists(traj_path):
        points = o3d.utility.Vector3dVector(np.stack([pose[:3, 3] for pose in camera_poses], axis=0))
        lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(camera_poses) - 1)])
        traj_lines.points = points
        traj_lines.lines = lines
    traj_lines.paint_uniform_color([0, 1, 0])

    print("Show original mesh (gray), OBB (red), world axes (RGB), OBB axes, trajectory (green)...")
    o3d.visualization.draw_geometries([mesh, obb, world_axis, obb_axis, traj_lines])

    # visualize the transformed mesh and OBB
    obb_transformed = o3d.geometry.OrientedBoundingBox(obb)
    obb_transformed.translate(-obb.center)
    obb_transformed.rotate(R, center=(0, 0, 0))
    print("transformed OBB rotation:\n", obb_transformed.R)
    print("transformed OBB center:\n", obb_transformed.center)

    obb_transformed.color = [0, 1, 0]  # transformed OBB is green
    mesh_rotated.paint_uniform_color([0, 0, 1])  # transformed mesh is blue

    # Coordinate axes of the transformed system (at the origin)
    transformed_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8, origin=[0, 0, 0])
    transformed_traj_lines = o3d.geometry.LineSet(traj_lines)
    transformed_traj_lines.translate(-obb.center)
    transformed_traj_lines.rotate(R, center=(0, 0, 0))
    transformed_traj_lines.paint_uniform_color([1, 0, 0])

    print("Show transformed mesh (blue), OBB (green), transformed axes (RGB), transformed trajectory (red)...")
    o3d.visualization.draw_geometries([mesh_rotated, obb_transformed, transformed_axis, transformed_traj_lines])
