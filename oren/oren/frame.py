from typing import Optional

import torch

from oren.utils.depth_utils import _camera_ray_directions, camera_ray_directions


class Frame:
    def get_frame_index(self) -> int:
        raise NotImplementedError

    def get_ref_pose(self) -> torch.Tensor:
        raise NotImplementedError

    def get_points(self, to_world_frame: bool, device: str) -> torch.Tensor:
        raise NotImplementedError

    def get_rays_direction(self) -> torch.Tensor:
        raise NotImplementedError

    def get_depth(self, mask=None) -> torch.Tensor:
        """Float depth in meters. With `mask` (flat index or boolean over the flattened
        depth), return only that subset — converting only it for compressed storage."""
        raise NotImplementedError

    def get_valid_mask(self) -> torch.Tensor:
        raise NotImplementedError

    def apply_bound(self, bound_min: torch.Tensor, bound_max: torch.Tensor):
        raise NotImplementedError

    def sample_points(
        self,
        num_points: int = -1,
        ratio: float = 0.25,
        to_world_frame: bool = True,
        device: str = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class DepthFrame(Frame):

    def __init__(
        self,
        fid: int,
        depth: torch.Tensor,
        intrinsic: torch.Tensor,
        ref_pose: torch.Tensor,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None,
        depth_storage_dtype: str = "fp32",
        depth_scale: float = 6553.5,
        device: str = None,
    ) -> None:
        """
        Args:
            fid: int, frame idx
            depth: (H, W) in meter
            intrinsic: (3, 3) intrinsic matrix
            ref_pose: (4, 4) reference pose in world coordinates
            min_depth: float, min depth in meter
            max_depth: float, max depth in meter
            depth_storage_dtype: storage dtype for depth: "fp32" (no compression), "fp16"
                (half, ~1-4 mm error), or "uint16" (int * 1/depth_scale, lossless to the
                source PNG quantization; halves storage vs fp32). rays_d and valid_mask are
                always derived from the original float depth before compression.
            depth_scale: meters-per-unit divisor for "uint16" storage (depth_units =
                round(depth_m * depth_scale), reconstructed as units / depth_scale).
            device: str, device to put the tensors on
        """

        super().__init__()
        self.stamp = fid
        self.h, self.w = depth.shape
        if not isinstance(depth, torch.Tensor):
            depth = torch.FloatTensor(depth)  # / 2
        self.K = intrinsic

        if ref_pose.ndim != 2:
            ref_pose = ref_pose.reshape(4, 4)
        if not isinstance(ref_pose, torch.Tensor):  # from gt data
            self.ref_pose = torch.tensor(ref_pose, requires_grad=False, dtype=torch.float32)
        else:  # from tracked data
            self.ref_pose = ref_pose.clone().requires_grad_(False)

        # Derive rays_d and valid_mask from the original FLOAT depth, then store depth in the
        # configured storage dtype. The (H, W, 3) camera-frame points are reconstructed on demand
        # via the `points` property (rays_d * depth), so no 3-channel tensor is retained per key
        # frame. `rays_d` is the shared cache entry (same H/W/K/device/dtype) and must not be
        # mutated in place. depth is exposed as float meters via get_depth(mask=None).
        self.depth_storage_dtype = depth_storage_dtype
        self.depth_scale = float(depth_scale)
        self.rays_d: torch.Tensor = camera_ray_directions(depth, self.K)  # (H, W, 3) unit-depth rays
        if min_depth is not None and max_depth is not None:
            self.valid_mask: torch.Tensor = (depth > min_depth) & (depth < max_depth)  # (H, W) depth > 0
        else:
            self.valid_mask: torch.Tensor = depth > 0  # (H, W) depth > 0
        self._depth_raw: torch.Tensor = self._encode_depth(depth)  # (H, W) in storage dtype

        if device is not None:
            self._depth_raw = self._depth_raw.to(device)
            self.ref_pose = self.ref_pose.to(device)
            self.rays_d = self.rays_d.to(device)
            self.valid_mask = self.valid_mask.to(device)

    def _encode_depth(self, depth_m: torch.Tensor) -> torch.Tensor:
        """float meters -> storage dtype."""
        if self.depth_storage_dtype == "uint16":
            return torch.clamp((depth_m.float() * self.depth_scale).round(), 0, 65535).to(torch.uint16)
        if self.depth_storage_dtype == "fp16":
            return depth_m.half()
        return depth_m.float()

    def _decode_depth(self, raw: torch.Tensor) -> torch.Tensor:
        """storage dtype -> float meters."""
        if self.depth_storage_dtype == "uint16":
            return raw.float() / self.depth_scale
        if self.depth_storage_dtype == "fp16":
            return raw.float()
        return raw  # fp32: already float meters (no copy)

    def get_depth(self, mask=None) -> torch.Tensor:
        """Float depth in meters, decoded from storage. With `mask` (flat index or boolean
        over the flattened depth), decode only that subset to save computation."""
        raw = self._depth_raw if mask is None else self._depth_raw.reshape(-1)[mask]
        return self._decode_depth(raw)

    @property
    def points(self) -> torch.Tensor:
        """(H, W, 3) camera-frame points, reconstructed on demand as `rays_d * depth`."""
        return self.rays_d * self.get_depth()[..., None]

    def get_frame_index(self):
        return self.stamp

    def get_ref_pose(self):
        return self.ref_pose

    def get_ref_translation(self):
        return self.ref_pose[:3, 3]

    def get_ref_rotation(self):
        return self.ref_pose[:3, :3]

    @torch.no_grad()
    def get_rays(self, w=None, h=None, K=None):
        w = self.w if w is None else w
        h = self.h if h is None else h
        if K is None:
            # Scale the stored intrinsics to the requested (w, h) resolution.
            fx = float(self.K[0, 0]) * w / self.w
            fy = float(self.K[1, 1]) * h / self.h
            cx = float(self.K[0, 2]) * w / self.w
            cy = float(self.K[1, 2]) * h / self.h
        else:
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        # Pixel-center rays via the shared cache keyed on (h, w, intrinsics, device, dtype).
        # The returned tensor is shared across calls and must not be mutated in place.
        return _camera_ray_directions(int(h), int(w), fx, fy, cx, cy, self.rays_d.device, self.rays_d.dtype)

    def get_points(self, to_world_frame: bool, device: str):
        points = self.points.to(device)  # (H, W, 3)
        valid_mask = self.valid_mask.to(device)  # (H, W)

        points = points[valid_mask].reshape(-1, 3)  # [N,3]

        if to_world_frame:
            pose = self.get_ref_pose().to(device)
            points = points @ pose[:3, :3].T + pose[:3, 3]  # to world coordinates
        return points

    def get_rays_direction(self):
        return self.rays_d

    def get_valid_mask(self):
        return self.valid_mask

    @torch.no_grad()
    def project_to_bound(self, bound_min: torch.Tensor, bound_max: torch.Tensor):
        """Applies a bounding box constraint to points in the frame.

        Any points in camera coordinates that, when transformed to world coordinates,
        fall outside the axis-aligned bounding box defined by [bound_min, bound_max]
        are projected back onto the box boundary along the ray from the camera origin.

        Args:
            bound_min (torch.Tensor): Lower corner (3,) of the bounding box, in world coordinates.
            bound_max (torch.Tensor): Upper corner (3,) of the bounding box, in world coordinates.
        """
        device = self._depth_raw.device
        bound_min = bound_min.to(device)
        bound_max = bound_max.to(device)

        mask = self.valid_mask
        if not mask.any():
            return

        R_c2w = self.ref_pose[:3, :3]
        t_c2w = self.ref_pose[:3, 3]

        points_cam = self.points[mask]
        points_world = torch.addmm(t_c2w, points_cam, R_c2w.T)

        # Out-of-bounds mask
        oob_mask = (points_world < bound_min).any(-1) | (points_world > bound_max).any(-1)
        if not oob_mask.any():
            return

        pts_to_proj = points_world[oob_mask]
        origin = t_c2w.view(1, 3)

        # Ray-AABB intersection using "slab method"
        ray_dir = pts_to_proj - origin
        safe_dir = torch.where(ray_dir.abs() < 1e-8, torch.sign(ray_dir) * 1e-8, ray_dir)
        inv_dir = 1.0 / safe_dir
        t_near = (bound_min - origin) * inv_dir
        t_far = (bound_max - origin) * inv_dir
        t_exit = torch.min(torch.max(t_near, t_far), dim=-1)[0]

        # Project to boundary, keeping just inside
        projected_world = origin + (t_exit.unsqueeze(-1) * 0.9999) * ray_dir

        # Back to camera coordinates
        projected_cam = (projected_world - t_c2w) @ R_c2w

        # Write projected points back (decode to float, modify, re-encode to storage dtype).
        full_mask = mask.clone()
        full_mask[mask] = oob_mask
        depth_f = self.get_depth()  # float (H, W); for fp32 this is the stored tensor itself
        depth_f[full_mask] = projected_cam[:, 2]
        self._depth_raw = self._encode_depth(depth_f)

    def apply_bound(self, bound_min: torch.Tensor, bound_max: torch.Tensor):
        points = self.points @ self.ref_pose[:3, :3].T + self.ref_pose[:3, 3]
        mask = points >= bound_min.view(1, 1, 3)
        mask = mask & (points <= bound_max.view(1, 1, 3))
        mask = mask.all(dim=-1)
        self.valid_mask = self.valid_mask & mask

    def sample_points(
        self,
        num_points: int = -1,
        ratio: float = 0.25,
        to_world_frame: bool = True,
        device: str = None,
    ) -> torch.Tensor:
        if num_points <= 0:
            num_points = int(self.h * self.w * ratio)
        indices = torch.argwhere(self.valid_mask)
        if len(indices) <= num_points:
            sampled_indices = indices
        else:
            perm = torch.randperm(len(indices))[:num_points]
            sampled_indices = indices[perm]
        points = self.points[sampled_indices[:, 0], sampled_indices[:, 1]]
        if device is not None:
            points = points.to(device)
        if to_world_frame:
            pose = self.get_ref_pose().to(points.device)
            points = points @ pose[:3, :3].T + pose[:3, 3]  # to world coordinates
        return points


class LiDARFrame:
    def __init__(
        self,
        fid: int,
        pointcloud: torch.Tensor,
        ref_pose: torch.Tensor,
    ) -> None:
        self.stamp = fid
        self.points = pointcloud

        if ref_pose.ndim != 2:
            ref_pose = ref_pose.reshape(4, 4)
        if not isinstance(ref_pose, torch.Tensor):  # from gt data
            self.ref_pose = torch.tensor(ref_pose, requires_grad=False, dtype=torch.float32)
        else:  # from tracked data
            self.ref_pose = ref_pose.clone().requires_grad_(False)
        self.rays_d: torch.Tensor = self.get_rays()  # (N, 3) in world coordinates

        self.valid_mask: torch.Tensor = torch.ones(pointcloud.shape[0], dtype=torch.bool)

    def get_frame_index(self):
        return self.stamp

    def get_ref_pose(self):
        return self.ref_pose

    def get_ref_translation(self):
        return self.ref_pose[:3, 3]

    def get_ref_rotation(self):
        return self.ref_pose[:3, :3]

    def get_points(self, to_world_frame: bool, device: str):
        points = self.points[self.valid_mask].reshape(-1, 3).to(device)
        if to_world_frame:
            pose = self.get_ref_pose().to(device)
            points = points @ pose[:3, :3].T + pose[:3, 3]  # to world coordinates
        return points

    def get_depth(self, mask=None):
        pts = self.points if mask is None else self.points.reshape(-1, 3)[mask]
        return torch.norm(pts, dim=-1)  # (N,)

    @torch.no_grad()
    def get_rays(self):
        rays_d = torch.nn.functional.normalize(self.points, p=2, dim=-1)
        return rays_d

    def get_rays_direction(self):
        return self.rays_d

    def get_valid_mask(self):
        return self.valid_mask

    def apply_bound(self, bound_min: torch.Tensor, bound_max: torch.Tensor):
        points = self.points @ self.ref_pose[:3, :3].T + self.ref_pose[:3, 3]
        mask = points >= bound_min.view(1, 3)
        mask = mask & (points <= bound_max.view(1, 3))
        mask = mask.all(dim=-1)
        self.valid_mask = mask & self.valid_mask

    def sample_points(
        self,
        num_points: int = -1,
        ratio: float = 0.25,
        to_world_frame: bool = True,
        device: str = None,
    ) -> torch.Tensor:
        if num_points <= 0:
            num_points = int(self.points.shape[0] * ratio)
        indices = torch.argwhere(self.valid_mask).flatten()
        if len(indices) <= num_points:
            sampled_indices = indices
        else:
            perm = torch.randperm(len(indices))[:num_points]
            sampled_indices = indices[perm]
        points = self.points[sampled_indices]
        if device is not None:
            points = points.to(device)
        if to_world_frame:
            pose = self.get_ref_pose().to(points.device)
            points = points @ pose[:3, :3].T + pose[:3, 3]  # to world coordinates
        return points
