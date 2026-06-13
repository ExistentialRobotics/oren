from dataclasses import dataclass
from typing import Optional

import torch

from torch.nn.utils.rnn import pad_sequence

from oren.frame import DepthFrame, Frame
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

        # Persistent all-False bool buffer reused for the is_key_frame IoU (see
        # there). Allocated lazily on first use; kept clean by resetting only
        # touched entries so it never needs a full re-zero.
        self._iou_mask: Optional[torch.Tensor] = None

        # Cached GPU pad_sequence of kf_seen_voxel_indices for select_key_frames.
        # Rebuilt only when a key frame is added (tracked by count) — between adds
        # select_key_frames runs every frame but this input is unchanged.
        self._padded_voxels: Optional[torch.Tensor] = None
        self._padded_voxels_n: int = -1

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

        # IoU of the two voxel-index sets via a persistent bool occupancy mask
        # (O(|A|+|B|) scatter/gather) instead of a sort-based torch.unique on the
        # concatenation — the flamegraph/py-spy flagged that unique as ~14% of
        # wall. Identical result: A and B are each sets of distinct octree node
        # indices, so |A∩B| = #(B already marked) and |A∪B| = |A|+|B|-|A∩B|.
        a = self.kf_seen_voxel_indices[-1].reshape(-1)
        b = seen_voxel_indices.reshape(-1)
        if a.numel() == 0 or b.numel() == 0:
            return True
        if self._iou_mask is None or self._iou_mask.device != a.device:
            self._iou_mask = torch.zeros(self.max_num_voxels, dtype=torch.bool, device=a.device)
        mask = self._iou_mask
        mask[a] = True
        n_intersection = int(mask[b].sum().item())
        mask[a] = False  # reset only touched entries; buffer stays all-False
        n_union = a.numel() + b.numel() - n_intersection
        iou = n_intersection / n_union if n_union > 0 else 1.0
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
            # Rebuild the padded voxel tensor only when a key frame was added.
            n = len(self.kf_seen_voxel_indices)
            if self._padded_voxels is None or self._padded_voxels_n != n:
                self._padded_voxels = pad_sequence(
                    self.kf_seen_voxel_indices, batch_first=True, padding_value=-1
                ).long().to(self.device)
                self._padded_voxels_n = n
            selected_frame_indices, self.kf_unoptimized_voxels, self.kf_all_voxels = multiple_max_set_coverage(
                self.kf_seen_voxel_num,
                self.kf_seen_voxel_indices,
                self.kf_unoptimized_voxels,
                self.kf_all_voxels,
                self.cfg.selection_window_size,
                num_voxels=self.max_num_voxels,
                device=self.device,
                padded_tensor=self._padded_voxels,
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

        # Per-key-frame sample-count bookkeeping (drives the non-uniform weighting
        # on the next call); applies to the key frames only, not the appended
        # current_frame. Pure Python — kept out of the GPU work below.
        for frame_idx in range(len(key_frame_indices)):
            self.sample_counts[key_frame_indices[frame_idx]] += samples_per_frame[frame_idx]

        device = frames[0].get_ref_pose().device

        # Fast path: camera DepthFrames share one pixel->ray template (constant
        # intrinsics), so all frames can be sampled with a handful of batched ops
        # instead of ~6 GPU kernels per frame in a Python loop. With num_rays_total
        # tiny (e.g. 1024) the arithmetic is microseconds — the per-launch GIL
        # contention in that loop was the real cost, so collapsing the launch count
        # (and making it independent of n_frames) is the win. Non-DepthFrame inputs
        # (e.g. LiDAR, whose ray directions differ per frame) take the loop below.
        template_numel = frames[0].get_rays_direction().numel()
        if all(isinstance(f, DepthFrame) and f.get_rays_direction().numel() == template_numel
               for f in frames):
            # Concatenate every frame's valid-pixel indices into one flat buffer,
            # remembering each frame's [offset, offset+len) span within it.
            valid_list = []
            for frame_idx in range(n_frames):
                if frame_idx < len(key_frame_indices):
                    valid_list.append(self.valid_indices[key_frame_indices[frame_idx]].view(-1))
                else:
                    valid_list.append(torch.nonzero(frames[frame_idx].get_valid_mask().view(-1)).view(-1))
            valid_lens = torch.tensor([v.numel() for v in valid_list], device=device)
            valid_cat = torch.cat(valid_list)
            valid_offset = torch.cumsum(valid_lens, 0) - valid_lens

            # Frame id for each of the num_samples draws (per-frame budgets concatenated).
            spf = torch.tensor(samples_per_frame, device=device)
            frame_id = torch.repeat_interleave(torch.arange(n_frames, device=device), spf)

            # Uniform random pixel within each draw's frame's valid set.
            lens_per_sample = valid_lens[frame_id]
            rand_pos = (torch.rand(num_samples, device=device) * lens_per_sample).long()
            rand_pos = torch.minimum(rand_pos, lens_per_sample - 1)  # guard fp rounding up to len
            sample_idx = valid_cat[valid_offset[frame_id] + rand_pos]  # (num_samples,) pixel index

            # Batched pose + shared-template gathers.
            poses = torch.stack([f.get_ref_pose() for f in frames])[frame_id]  # (num_samples,4,4)
            rays_o_all = poses[:, :3, 3]  # (num_samples, 3)
            rays_d_cam = frames[0].get_rays_direction().view(-1, 3)[sample_idx]  # (num_samples, 3)
            # world dir = R @ d_cam (same as the per-frame d_cam @ R.T).
            rays_d_all = torch.bmm(poses[:, :3, :3], rays_d_cam.unsqueeze(-1)).squeeze(-1)

            depth_stack = torch.stack([f.get_depth().view(-1) for f in frames])  # (n_frames, HW)
            depth_samples_all = depth_stack[frame_id, sample_idx]  # (num_samples,)

            return rays_o_all, rays_d_all, depth_samples_all

        # Fallback: per-frame loop (general; handles non-DepthFrame ray templates).
        rays_o_all, rays_d_all, depth_samples_all = [], [], []
        for frame_idx, frame in enumerate(frames):
            n_frame_samples = samples_per_frame[frame_idx]
            if frame_idx < len(key_frame_indices):
                valid_idx = self.valid_indices[key_frame_indices[frame_idx]]
            else:
                valid_idx = torch.nonzero(frame.get_valid_mask().view(-1))
            # randint must live on valid_idx's device (frames are on the GPU).
            sample_idx = valid_idx[
                torch.randint(0, valid_idx.shape[0], (n_frame_samples,), device=valid_idx.device)
            ].view(-1)
            pose = frame.get_ref_pose()
            sampled_rays_d = frame.get_rays_direction().view(-1, 3)[sample_idx] @ pose[:3, :3].T
            rays_d_all.append(sampled_rays_d)
            rays_o_all.append(pose[:3, 3].view(1, 3).expand_as(sampled_rays_d))
            depth_samples_all.append(frame.get_depth().view(-1)[sample_idx])
        return torch.cat(rays_o_all, dim=0), torch.cat(rays_d_all, dim=0), torch.cat(depth_samples_all, dim=0)
