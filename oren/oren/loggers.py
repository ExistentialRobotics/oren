import os
import pickle
from datetime import datetime

import cv2
import numpy as np
import open3d as o3d
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class BasicLogger:
    def __init__(self, log_dir: str, exp_name: str, config_dict: dict, for_eva: bool = False) -> None:
        self.log_dir = os.path.join(
            log_dir,
            exp_name,
            self.get_random_time_str(),
        )
        self.img_dir = os.path.join(self.log_dir, "imgs")
        self.mesh_dir = os.path.join(self.log_dir, "mesh")
        self.ckpt_dir = os.path.join(self.log_dir, "ckpt")
        self.backup_dir = os.path.join(self.log_dir, "bak")
        self.misc_dir = os.path.join(self.log_dir, "misc")
        self.for_eva = for_eva
        self.tb = None
        if not for_eva:
            os.makedirs(self.img_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.mesh_dir, exist_ok=True)
            os.makedirs(self.misc_dir, exist_ok=True)
            os.makedirs(self.backup_dir, exist_ok=True)

            self.log_config(config_dict)
            self.tb = SummaryWriter(self.log_dir)

    @staticmethod
    def get_random_time_str():
        return datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")

    def log_ckpt(self, state_dict: dict, name: str):
        path = name if os.path.isabs(name) else os.path.join(self.ckpt_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def log_config(self, config_dict: dict):
        out_path = os.path.join(self.backup_dir, "config.yaml")
        yaml.dump(config_dict, open(out_path, "w"))

    def log_profiling_stats(self, stats: dict, name: str = "profiling_stats.yaml") -> str:
        """Dump per-timer wall-clock / memory stats to `<misc_dir>/<name>` (or to an absolute path).

        Args:
            stats: profiling stats dictionary.
            name: file name; treated as absolute if it starts with `/`, otherwise placed under `misc_dir`.

        Returns:
            The absolute path the stats were written to.
        """
        out_path = name if os.path.isabs(name) else os.path.join(self.misc_dir, name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            yaml.dump(stats, f, sort_keys=False)
        return out_path

    def log_mesh(self, mesh, name="final_mesh.ply"):
        out_path = os.path.join(self.mesh_dir, name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        o3d.io.write_triangle_mesh(out_path, mesh)

    def log_point_cloud(self, pcd, name="final_points.ply"):
        out_path = os.path.join(self.mesh_dir, name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        o3d.io.write_point_cloud(out_path, pcd)

    def log_numpy_data(self, data, name, ind=None):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if ind is not None:
            np.save(os.path.join(self.misc_dir, "{}-{:05d}.npy".format(name, ind)), data)
        else:
            np.save(os.path.join(self.misc_dir, f"{name}.npy"), data)

    def log_debug_data(self, data, idx):
        with open(os.path.join(self.misc_dir, f"scene_data_{idx}.pkl"), "wb") as f:
            pickle.dump(data, f)

    def log_rgb(self, rgb, name: str):
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        rgb = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        img_path = os.path.join(self.img_dir, name)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        cv2.imwrite(img_path, rgb)

    def log_depth(self, depth, name: str):
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        depth = (depth * 1000).astype(np.uint16)  # Convert to mm and uint16
        img_path = os.path.join(self.img_dir, name)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        cv2.imwrite(img_path, depth)

    @staticmethod
    def info(msg: str):
        tqdm.write(msg)
