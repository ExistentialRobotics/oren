import os.path as osp
from glob import glob
import open3d as o3d

import cv2
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from grad_sdf.frame import LiDARFrame


class DataLoader(Dataset):
    def __init__(
        self,
        data_path: str,
        min_depth: float = 0.0,
        max_depth: float = -1.0,
        offset: torch.Tensor = None,
        bound_min: torch.Tensor = None,
        bound_max: torch.Tensor = None,
    ):
        self.data_path = data_path
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.offset = offset
        self.bound_min = bound_min
        self.bound_max = bound_max

        if self.bound_min is None or self.bound_max is None:
            mesh_path = osp.join(data_path, "gt-mesh.ply")
            assert osp.exists(mesh_path), f"Mesh file {mesh_path} does not exist."
            mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(mesh_path)
            self.bound_min = np.min(mesh.vertices[:], axis=0).flatten().tolist()
            self.bound_max = np.max(mesh.vertices[:], axis=0).flatten().tolist()

        if self.offset is None:
            self.offset: torch.Tensor = torch.zeros(3)
        else:
            self.offset: torch.Tensor = torch.tensor(self.offset).float()
        if self.bound_min is not None:
            assert self.bound_max is not None
            self.bound_min = torch.tensor(self.bound_min).float()
            self.bound_min += self.offset
        if self.bound_max is not None:
            assert self.bound_min is not None
            self.bound_max = torch.tensor(self.bound_max).float()
            self.bound_max += self.offset

        self.num_imgs = len(glob(osp.join(self.data_path, "ply/*.ply")))
        self.gt_pose = self.load_gt_pose()

    def get_init_pose(self, init_frame=None):
        if self.gt_pose is not None and init_frame is not None:
            return self.gt_pose[init_frame].reshape(4, 4)
        elif self.gt_pose is not None:
            return self.gt_pose[0].reshape(4, 4)
        else:
            return np.eye(4)

    def load_gt_pose(self):
        gt_file = osp.join(self.data_path, "poses.txt")
        gt_pose = np.loadtxt(gt_file)  # (n_imgs,16)
        gt_pose = torch.from_numpy(gt_pose).float()
        return gt_pose

    def load_pointcloud(self, index) -> torch.Tensor:
        ply_path = osp.join(self.data_path, "ply/{:04d}.ply".format(index))
        pcd = o3d.io.read_point_cloud(ply_path)
        pointcloud = torch.from_numpy(np.asarray(pcd.points)).float()

        depth = torch.norm(pointcloud, dim=-1)

        # filter points by depth
        mask = torch.ones(depth.shape[0], dtype=torch.bool)
        if self.min_depth > 0:
            mask &= depth >= self.min_depth
        if self.max_depth > 0:
            mask &= depth <= self.max_depth

        pointcloud = pointcloud[mask]
        return pointcloud

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        pointcloud = self.load_pointcloud(index)
        pose = self.gt_pose[index]
        frame = LiDARFrame(index, pointcloud, self.offset, pose)
        if self.bound_min is not None and self.bound_max is not None:
            frame.apply_bound(self.bound_min, self.bound_max)
        return frame


def compute_bound(data_path: str, max_depth: float) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(data_path, max_depth)
    bound_min = []
    bound_max = []
    for i in tqdm(range(len(loader)), ncols=120, desc="Compute bound"):
        frame = loader[i]
        frame: LiDARFrame
        points = frame.get_points(to_world_frame=True, device="cpu")
        bound_min.append(points.min(dim=0).values)
        bound_max.append(points.max(dim=0).values)
    bound_min = torch.stack(bound_min, dim=0).min(dim=0).values
    bound_max = torch.stack(bound_max, dim=0).max(dim=0).values
    return bound_min, bound_max
