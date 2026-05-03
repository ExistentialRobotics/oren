import argparse
import json
import os
import shutil
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm


def load_cam_params(data_path: str) -> dict:
    """
    load camera parameters from a JSON file.
    Args:
        data_path: Path to the camera parameters JSON file.
    Returns:
        dict: Dictionary containing camera parameters.
    """
    with open(data_path, "r") as f:
        params = json.load(f)
    return params["camera"]


def load_traj(traj_path: str) -> np.ndarray:
    return np.loadtxt(traj_path)


class MeshVisualizer:
    def __init__(self, mesh_path: str, cam_intrinsics: dict, img_width: int = 1200, img_height: int = 680):
        self.mesh_path = mesh_path
        self.cam_intrinsics = cam_intrinsics
        self.img_width = img_width
        self.img_height = img_height

        self.mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(mesh_path)
        if not self.mesh.has_vertex_normals():
            self.mesh.compute_vertex_normals()

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.img_width, height=self.img_height, visible=False)
        self.vis.add_geometry(self.mesh)
        self.view_control = self.vis.get_view_control()

    def set_camera(self, camera_pose: np.ndarray):
        camera_parameters = o3d.camera.PinholeCameraParameters()
        camera_parameters.extrinsic = np.linalg.inv(camera_pose).astype(np.float64)
        intrinsic_matrix = np.array(
            [
                [self.cam_intrinsics["fx"], 0, self.cam_intrinsics["cx"]],
                [0, self.cam_intrinsics["fy"], self.cam_intrinsics["cy"]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        camera_parameters.intrinsic.intrinsic_matrix = intrinsic_matrix
        camera_parameters.intrinsic.height = self.img_height
        camera_parameters.intrinsic.width = self.img_width

        assert self.view_control.convert_from_pinhole_camera_parameters(
            camera_parameters
        ), f"warning: failed to set image size {self.img_width}x{self.img_height}"

    def render(self):
        self.vis.poll_events()
        self.vis.update_renderer()
        rgb_buffer = self.vis.capture_screen_float_buffer(do_render=True)
        depth_buffer = self.vis.capture_depth_float_buffer(do_render=True)
        rgb_array = np.asarray(rgb_buffer).copy()
        depth_array = np.asarray(depth_buffer).copy()
        return depth_array, rgb_array

    def visualize(self):
        o3d.visualization.draw_geometries([self.mesh])


# wRc @ cRo = I
cRo = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
# wRc = cRo.T, which happens to be the desired upward-looking rotation
upward_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # wRc @ cRo


def add_frames(
    intrinsic_path: str,
    mesh_path: str,
    traj_path: str,
    original_dir: str,
    output_dir: str,
    insert_interval: int,
    n_rolls_per_insertion: int = 1,
    max_roll_of_insertion: float = np.pi / 3,
    ignore_existing: bool = True,
):
    """
    insert an upward-looking frame every n frames
    Args:
        intrinsic_path: path to the camera intrinsics JSON file
        mesh_path: mesh file path
        traj_path: path to the trajectory file
        original_dir: original directory of the dataset
        output_dir: output directory to save modified trajectory and images
        insert_interval: insertion interval, default is to insert an upward-looking frame every 10 frames
        n_rolls_per_insertion: number of rolls per insertion, default is 1 (add the upward-looking frame only). If greater
            than 1, will add additional frames with rolls by 360/n_rolls_per_insert degrees after the frame.
        max_roll_of_insertion: maximum roll angle for the inserted frames
        ignore_existing: whether to ignore existing frames
    """
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    cam_intrinsics = load_cam_params(intrinsic_path)
    original_poses: np.ndarray = np.loadtxt(traj_path)
    mesh_visualizer = MeshVisualizer(mesh_path, cam_intrinsics, img_width=1200, img_height=680)

    # process in order, insert an upward-looking frame every `insert_interval` frames
    new_poses = []
    for i, pose_vec in tqdm(
        enumerate(original_poses),
        total=len(original_poses),
        desc="Processing frames",
        ncols=120,
        leave=False,
    ):
        pose_vec: np.ndarray

        pose_idx = len(new_poses)

        new_poses.append(pose_vec.copy())
        original_depth_fp = os.path.join(original_dir, "results", f"depth{i:06d}.png")
        original_rgb_fp = os.path.join(original_dir, "results", f"frame{i:06d}.jpg")
        dst_depth_fp = os.path.join(results_dir, f"depth{pose_idx:06d}.png")
        dst_rgb_fp = os.path.join(results_dir, f"frame{pose_idx:06d}.jpg")
        if os.path.exists(original_depth_fp) and os.path.exists(original_rgb_fp) and not ignore_existing:
            if original_depth_fp != dst_depth_fp:
                shutil.copy(original_depth_fp, dst_depth_fp)
            if original_rgb_fp != dst_rgb_fp:
                shutil.copy(original_rgb_fp, dst_rgb_fp)
        else:
            mesh_visualizer.set_camera(pose_vec.reshape(4, 4).copy())
            depth_image, rgb_image = mesh_visualizer.render()

            depth_image = (depth_image * cam_intrinsics["scale"]).astype(np.uint16)

            rgb_image = (rgb_image * 255).astype(np.uint8)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            cv2.imwrite(dst_rgb_fp, rgb_image)
            cv2.imwrite(dst_depth_fp, depth_image)

        if (i + 1) % insert_interval == 0:
            poses_to_append = []
            if n_rolls_per_insertion == 1:
                upward_pose = pose_vec.reshape(4, 4).copy()
                upward_pose[:3, :3] = upward_rotation
                poses_to_append.append(upward_pose)
            else:
                step = max_roll_of_insertion / n_rolls_per_insertion
                angle_min = -step * (n_rolls_per_insertion - 1) / 2
                for j in range(1, n_rolls_per_insertion):
                    angle = angle_min + step * (j - 1)
                    pitch_rotation = np.array(
                        [
                            [np.cos(angle), 0, np.sin(angle)],
                            [0, 1, 0],
                            [-np.sin(angle), 0, np.cos(angle)],
                        ]
                    )
                    pose = pose_vec.reshape(4, 4).copy()
                    pose[:3, :3] = pose[:3, :3] @ cRo.T @ pitch_rotation @ cRo
                    poses_to_append.append(pose)

                n = len(poses_to_append) // 2
                poses_to_append = poses_to_append[:n][::-1] + poses_to_append[1:-1] + poses_to_append[-n:][::-1]

            for pose in poses_to_append:
                pose_idx = len(new_poses)
                new_poses.append(pose.flatten())

                mesh_visualizer.set_camera(pose)
                depth_image, rgb_image = mesh_visualizer.render()

                depth_image = (depth_image * cam_intrinsics["scale"]).astype(np.uint16)

                rgb_image = (rgb_image * 255).astype(np.uint8)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

                cv2.imwrite(os.path.join(results_dir, f"frame{pose_idx:06d}.jpg"), rgb_image)
                cv2.imwrite(os.path.join(results_dir, f"depth{pose_idx:06d}.png"), depth_image)

    new_poses = np.stack(new_poses, axis=0)
    new_traj_path = os.path.join(output_dir, "traj.txt")
    np.savetxt(new_traj_path, new_poses, delimiter=" ")
    tqdm.write(f"saved new trajectory file with {len(new_poses)} poses to {new_traj_path}")


def process_all_replica_scenes(
    base_dir: str,
    output_dir: str,
    interval: int,
    n_rolls_per_insertion: int = 1,
    max_roll_of_insertion: float = np.pi / 3,
    ignore_existing: bool = True,
    scenes: Optional[list] = None,
):
    """
    Args:
        base_dir: base directory of the Replica dataset
        output_dir: output directory to save processed scenes
        interval: insertion interval, insert an upward-looking frame every n frames
        n_rolls_per_insertion: number of rolls per insertion, default is 1 (add the upward-looking frame only). If greater
            than 1, will add additional frames with rolls by 360/n_rolls_per_insert degrees after the frame.
        max_roll_of_insertion: maximum roll angle for the inserted frames.
        ignore_existing: whether to ignore existing frames
        scenes: list of scene names to process; if None, process all default scenes
    """
    if scenes is None:
        scenes = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
    cam_params_path = os.path.join(base_dir, "cam_params.json")

    for scene in tqdm(scenes, desc="Processing scenes", ncols=120):
        add_frames(
            intrinsic_path=cam_params_path,
            mesh_path=os.path.join(base_dir, f"{scene}_mesh.ply"),
            traj_path=os.path.join(base_dir, scene, "traj.txt"),
            original_dir=os.path.join(base_dir, scene),
            output_dir=os.path.join(output_dir, scene),
            insert_interval=interval,
            n_rolls_per_insertion=n_rolls_per_insertion,
            max_roll_of_insertion=max_roll_of_insertion,
            ignore_existing=ignore_existing,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", type=str, required=True, help="Path to the original dataset directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the output dataset directory")
    parser.add_argument("--interval", type=int, default=50, help="insert an upward-looking frame every n frames")
    parser.add_argument("--n-rolls-per-insertion", type=int, default=10, help="number of rolls per insertion")
    parser.add_argument(
        "--max-roll-of-insertion",
        type=float,
        default=np.pi / 3,
        help="maximum roll angle for inserted frames",
    )
    parser.add_argument("--keep-existing", action="store_true", help="whether to ignore existing frames")
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="*",
        default=None,
        help="List of scene names to process; if not set, process all default scenes",
    )

    args = parser.parse_args()

    process_all_replica_scenes(
        base_dir=args.original_dir,
        output_dir=args.output_dir,
        interval=args.interval,
        n_rolls_per_insertion=args.n_rolls_per_insertion,
        max_roll_of_insertion=args.max_roll_of_insertion,
        ignore_existing=not args.keep_existing,
        scenes=args.scenes,
    )


if __name__ == "__main__":
    main()
