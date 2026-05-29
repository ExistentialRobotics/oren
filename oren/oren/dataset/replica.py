import json
import os.path as osp
from glob import glob
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from oren.frame import DepthFrame


class DataLoader(Dataset):
    def __init__(
        self,
        data_path: str,
        min_depth: float = 0.0,
        max_depth: float = -1.0,
        apply_bound: bool = False,
        bound_min: Optional[torch.Tensor] = None,
        bound_max: Optional[torch.Tensor] = None,
    ):
        data_path = osp.expanduser(data_path)
        data_path = osp.abspath(data_path)
        data_path = data_path.rstrip("/")

        self.data_path = data_path
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.apply_bound = apply_bound
        self.bound_min = bound_min
        self.bound_max = bound_max

        cam_params_path = osp.join(data_path, "cam_params.json")
        if osp.exists(cam_params_path):
            with open(cam_params_path) as f:
                self._cam_params = json.load(f)
        else:
            self._cam_params = None
        self._depth_scale = self._cam_params.get("depth_scale", 6553.5) if self._cam_params else 6553.5

        if self.bound_min is None or self.bound_max is None:
            scene_name = osp.basename(osp.abspath(data_path))
            mesh_path = osp.join(osp.dirname(data_path), f"{scene_name}_mesh.ply")
            if osp.exists(mesh_path):
                mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(mesh_path)
                self.bound_min = np.min(mesh.vertices[:], axis=0).flatten().tolist()
                self.bound_max = np.max(mesh.vertices[:], axis=0).flatten().tolist()
            # else: leave bounds as None; trainer will use its own config bounds

        if self.bound_min is not None:
            self.bound_min = torch.tensor(self.bound_min).float()
        if self.bound_max is not None:
            self.bound_max = torch.tensor(self.bound_max).float()

        num_jpgs = len(glob(osp.join(self.data_path, "results/*.jpg")))
        self.num_imgs = num_jpgs if num_jpgs > 0 else len(glob(osp.join(self.data_path, "results/depth*.png")))
        self.K = self.load_intrinsic()
        self.gt_pose = self.load_gt_pose()

    def load_intrinsic(self):
        K = torch.eye(3)
        if self._cam_params is not None:
            K[0, 0] = self._cam_params["fx"]
            K[1, 1] = self._cam_params["fy"]
            K[0, 2] = self._cam_params["cx"]
            K[1, 2] = self._cam_params["cy"]
        else:
            K[0, 0] = K[1, 1] = 600
            K[0, 2] = 599.5
            K[1, 2] = 339.5
        return K

    def get_init_pose(self, init_frame=None):
        if self.gt_pose is not None and init_frame is not None:
            return self.gt_pose[init_frame].reshape(4, 4)
        elif self.gt_pose is not None:
            return self.gt_pose[0].reshape(4, 4)
        else:
            return np.eye(4)

    def load_gt_pose(self):
        gt_file = osp.join(self.data_path, "traj.txt")
        gt_pose = np.loadtxt(gt_file)  # (n_imgs,16)
        gt_pose = torch.from_numpy(gt_pose).float()
        return gt_pose

    def load_depth(self, index) -> torch.Tensor:
        depth = cv2.imread(osp.join(self.data_path, "results/depth{:06d}.png".format(index)), -1)
        depth = depth / self._depth_scale
        if self.min_depth >= 0:
            depth[depth < self.min_depth] = 0
        if self.max_depth > 0:
            depth[depth > self.max_depth] = 0
        depth = torch.from_numpy(depth).float()
        return depth

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        depth = self.load_depth(index)
        pose = self.gt_pose[index]
        frame = DepthFrame(index, depth, self.K, pose)
        if self.apply_bound:
            frame.apply_bound(self.bound_min, self.bound_max)
        return frame


def compute_bound(data_path: str, max_depth: float) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(data_path, max_depth)
    bound_min = []
    bound_max = []
    for i in tqdm(range(len(loader)), ncols=120, desc="Compute bound"):
        frame = loader[i]
        frame: DepthFrame
        points = frame.get_points(to_world_frame=True, device="cpu")
        bound_min.append(points.min(dim=0).values)
        bound_max.append(points.max(dim=0).values)
    bound_min = torch.stack(bound_min, dim=0).min(dim=0).values
    bound_max = torch.stack(bound_max, dim=0).max(dim=0).values
    return bound_min, bound_max
