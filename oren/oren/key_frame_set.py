from dataclasses import dataclass
from typing import Optional

import torch

from oren.frame import Frame
from oren.utils.config_abc import ConfigABC
from oren.utils.keyframe_util import multiple_max_set_coverage


@dataclass
class KeyFrameSetConfig(ConfigABC):
    insert_method: str = "insert_method"  # naive | intersection
    insert_interval: int = 50  # number of frames between key frames
    insert_ratio: float = 0.85
    frame_selection: str = "multiple_max_set_coverage"  # multiple_max_set_coverage | random
    selection_window_size: int = 8
    frame_weight: str = "uniform"


class KeyFrameSet:
    def __init__(self, cfg: KeyFrameSetConfig, max_num_voxels: int, device: str):
        self.cfg = cfg
        self.max_num_voxels = max_num_voxels
        self.device = device

        self.frames: list[Frame] = []
        self.valid_indices: list[torch.Tensor] = []
        self.sample_counts: list[int] = []

        self.kf_seen_voxel_indices: list[torch.Tensor] = []
        self.kf_seen_voxel_num: list[int] = []
        self.kf_unoptimized_voxels: Optional[torch.Tensor] = None
        self.kf_all_voxels: Optional[torch.Tensor] = None

    def add_key_frame(self, frame: Frame, seen_voxel_indices: torch.Tensor):
        """
        Adds a key frame to the set.
        Args:
            frame: RGBDFrame to be added.
            seen_voxel_indices: indices of voxels seen by the frame.
        Returns:
            bool: True if the frame is added as a key frame, False otherwise.
        """
        if self.is_key_frame(frame, seen_voxel_indices):
            self.add_frame(frame, seen_voxel_indices)
            return True
        return False

    def is_key_frame(self, frame: Frame, seen_voxel_indices: torch.Tensor):
        """
        Decide whether to add the frame as a key frame.
        If self.frames is empty, return True.
        If self.cfg.insert_method is "naive", return True if the frame index
        is greater than the last key frame index by self.cfg.insert_interval.
        If self.cfg.insert_method is "intersection", compute the IoU of the voxels
        seen by the frame and the last key frame. Return True if IoU < self.cfg.insert_ratio.

        Args:
            frame: Frame to be added.
            seen_voxel_indices: indices of voxels seen by the frame.

        Returns:
            True if the frame should be added as a key frame, False otherwise.
        """
        if len(self.frames) == 0:
            return True

        if self.cfg.insert_method == "naive":
            if frame.get_frame_index() - self.frames[-1].get_frame_index() >= self.cfg.insert_interval:
                return True
            return False

        voxels_unique, counts = torch.unique(
            torch.cat([self.kf_seen_voxel_indices[-1], seen_voxel_indices], dim=0),
            return_counts=True,
            sorted=False,
            dim=0,
        )
        n_intersection = torch.sum(counts > 1).item()
        n_union = voxels_unique.shape[0]
        iou = n_intersection / n_union
        if iou < self.cfg.insert_ratio:
            return True
        return False

    def add_frame(self, frame: Frame, seen_voxel_indices: torch.Tensor):
        """
        Add a frame to the set.
        1. Append the frame to self.frames.
        2. Append the indices of voxels seen by the frame to self.kf_seen_voxel_indices.
        3. Append the number of voxels seen by the frame to self.kf_seen_voxel_num.
        4. Append the valid indices of the frame to self.valid_indices.
        5. Initialize the sample count of the frame.
        6. Update self.kf_unoptimized_voxels if using "multiple_max_set_coverage" selection.

        Args:
            frame: Frame to be added.
            seen_voxel_indices: indices of voxels seen by the frame.

        Returns:

        """
        self.frames.append(frame)
        self.kf_seen_voxel_indices.append(seen_voxel_indices)
        self.kf_seen_voxel_num.append(seen_voxel_indices.shape[0])

        valid_idx = torch.nonzero(frame.get_valid_mask().view(-1))
        self.valid_indices.append(valid_idx)
        self.sample_counts.append(sum(self.sample_counts) // (len(self.sample_counts) + 2))

        if self.cfg.frame_selection == "multiple_max_set_coverage" and self.kf_unoptimized_voxels is not None:
            self.kf_unoptimized_voxels.index_fill_(0, seen_voxel_indices.long().view(-1).to(self.device), True)

    def select_key_frames(self) -> list[int]:
        """
        Pick self.cfg.selection_window_size key frames from self.frames.
        The selection strategy is set by self.cfg.frame_selection.
        If the number of frames is less than or equal to selection_window_size,
        we return all frames.

        Returns:
            list of indices of selected key frames.
        """
        if len(self.frames) <= self.cfg.selection_window_size:
            return list(range(len(self.frames)))

        if self.cfg.frame_selection == "random":
            selected_frame_indices = torch.randperm(len(self.frames))[: self.cfg.selection_window_size].tolist()
            return selected_frame_indices

        if self.cfg.frame_selection == "multiple_max_set_coverage":
            selected_frame_indices, self.kf_unoptimized_voxels, self.kf_all_voxels = multiple_max_set_coverage(
                self.kf_seen_voxel_num,
                self.kf_seen_voxel_indices,
                self.kf_unoptimized_voxels,
                self.kf_all_voxels,
                self.cfg.selection_window_size,
                num_voxels=self.max_num_voxels,
                device=self.device,
            )
            return selected_frame_indices

        raise ValueError(f"Unknown frame selection method: {self.cfg.frame_selection}")

    def sample_points(self, ratio: float, key_frame_indices: list, current_frame: Optional[Frame]) -> torch.Tensor:
        frames: list[Frame] = [self.frames[i] for i in key_frame_indices]
        if current_frame is not None and current_frame != self.frames[-1]:
            frames.append(current_frame)
        points = [frame.sample_points(ratio=ratio, to_world_frame=True, device=self.device) for frame in frames]
        points = torch.cat(points, dim=0)
        return points

    def sample_rays(
        self,
        num_samples: int,
        key_frame_indices: list,
        current_frame: Frame | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample rays from the key frames. The sampling strategy is set by self.cfg.frame_weight.
        When the strategy is "uniform", we sample uniformly from each frame.
        Otherwise, we do:
            1. Distribute num_samples to each frame based on the sample counts.
                Higher sample count -> fewer samples.
            2. Sample the rays from each frame.
            3. Update the sample counts for next sampling.
        Args:
            num_samples: number of rays to sample.
            key_frame_indices: indices of key frames to sample from.
            current_frame: the current frame, if not None, we also sample from it.

        Returns:
            (num_samples, 3) ray origins in world coordinates.
            (num_samples, 3) ray directions in world coordinates.
            (num_samples,) depth values in meter.
        """

        # distribute num_samples to each frame based on the sample counts
        # higher sample count -> fewer samples

        frames: list[Frame] = [self.frames[i] for i in key_frame_indices]
        sample_counts = [self.sample_counts[i] for i in key_frame_indices]
        if current_frame is not None and current_frame != self.frames[-1]:
            frames.append(current_frame)
            sample_counts.append(sum(sample_counts) // (len(sample_counts) + 2))

        total_count = sum(sample_counts)
        n_frames = len(frames)

        if n_frames == 0 or num_samples == 0:
            return None, None, None, None

        if self.cfg.frame_weight == "uniform":
            samples_per_frame = [num_samples // n_frames] * n_frames
            for i in range(num_samples % n_frames):
                samples_per_frame[i] += 1
        else:
            if total_count == 0:
                samples_per_frame = [num_samples // n_frames] * n_frames
                for i in range(num_samples % n_frames):
                    samples_per_frame[i] += 1
            elif n_frames == 1:
                samples_per_frame = [num_samples]
            else:
                m = total_count * (n_frames - 1)
                samples_per_frame = [max(1, int(num_samples * (total_count - count) / m)) for count in sample_counts]
                # adjust to make sum exactly num_samples
                diff = num_samples - sum(samples_per_frame)
                for i in range(abs(diff)):
                    idx = i % len(self.frames)
                    if diff > 0:
                        samples_per_frame[idx] += 1
                    elif samples_per_frame[idx] > 1:
                        samples_per_frame[idx] -= 1

        rays_o_all = []
        rays_d_all = []
        depth_samples_all = []
        for frame_idx, frame in enumerate(frames):
            n_frame_samples = samples_per_frame[frame_idx]

            if frame_idx < len(key_frame_indices):
                i = key_frame_indices[frame_idx]
                self.sample_counts[i] += n_frame_samples
                valid_idx = self.valid_indices[i]
            else:
                valid_idx = torch.nonzero(frame.get_valid_mask().view(-1))
            # randint must live on valid_idx's device (frames are now on the GPU)
            # or the gather below hits a cross-device indexing error.
            sample_idx = valid_idx[
                torch.randint(0, valid_idx.shape[0], (n_frame_samples,), device=valid_idx.device)
            ]
            sample_idx = sample_idx.view(-1)

            pose = frame.get_ref_pose()
            rotation = pose[:3, :3]
            sampled_rays_d = frame.get_rays_direction().view(-1, 3)[sample_idx]  # (n_frame_samples, 3)
            sampled_rays_d = sampled_rays_d @ rotation.T  # (n_frame_samples, 3)
            sampled_rays_o = pose[:3, 3].view(1, 3).expand_as(sampled_rays_d)  # (n_frame_samples, 3)
            rays_o_all.append(sampled_rays_o)
            rays_d_all.append(sampled_rays_d)

            sampled_depth = frame.get_depth().view(-1)[sample_idx]  # (n_frame_samples,)
            depth_samples_all.append(sampled_depth)

        rays_o_all = torch.cat(rays_o_all, dim=0)  # (num_samples, 3)
        rays_d_all = torch.cat(rays_d_all, dim=0)  # (num_samples, 3)
        depth_samples_all = torch.cat(depth_samples_all, dim=0)  # (num_samples,)
        return rays_o_all, rays_d_all, depth_samples_all
