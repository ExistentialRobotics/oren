import torch
from typing import Optional


class Frame:
    def get_frame_index(self) -> int:
        raise NotImplementedError

    def get_ref_pose(self) -> torch.Tensor:
        raise NotImplementedError

    def get_points(self, to_world_frame: bool, device: str) -> torch.Tensor:
        raise NotImplementedError

    def get_rays_direction(self) -> torch.Tensor:
        raise NotImplementedError

    def get_depth(self) -> torch.Tensor:
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
        offset: torch.Tensor,
        ref_pose: torch.Tensor,
        max_depth: Optional[float] = None,
        min_depth: Optional[float] = None,
        device: str = None,
    ) -> None:
        """
        Args:
            fid: int, frame idx
            depth: (H, W) in meter
            intrinsic: (3, 3) intrinsic matrix
            offset: (3, ) offset to be added to the translation of ref_pose
            ref_pose: (4, 4) reference pose in world coordinates
            max_depth: float, max depth in meter
            min_depth: float, min depth in meter
        """
        super().__init__()
        self.stamp = fid
        self.h, self.w = depth.shape
        if not isinstance(depth, torch.Tensor):
            depth = torch.FloatTensor(depth)  # / 2
        self.depth = depth
        self.K = intrinsic

        if ref_pose.ndim != 2:
            ref_pose = ref_pose.reshape(4, 4)
        if not isinstance(ref_pose, torch.Tensor):  # from gt data
            self.ref_pose = torch.tensor(ref_pose, requires_grad=False, dtype=torch.float32)
        else:  # from tracked data
            self.ref_pose = ref_pose.clone().requires_grad_(False)
        self.ref_pose[:3, 3] += offset  # Offset ensures voxel coordinates > 0

        self.rays_d: torch.Tensor = self.get_rays(K=self.K)  # (H, W, 3) in camera coordinates
        self.points: torch.Tensor = self.rays_d * self.depth[..., None]  # (H, W, 3) in camera coordinates
        if min_depth is not None and max_depth is not None:
            self.valid_mask: torch.Tensor = (self.depth > min_depth) & (self.depth < max_depth)  # (H, W) depth > 0
        else:
            self.valid_mask: torch.Tensor = self.depth > 0  # (H, W) depth > 0

        if device is not None:
            self.depth = self.depth.to(device)
            self.ref_pose = self.ref_pose.to(device)
            self.rays_d = self.rays_d.to(device)
            self.points = self.points.to(device)
            self.valid_mask = self.valid_mask.to(device)

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
            K = torch.eye(3)
            K[0, 0] = self.K[0, 0] * w / self.w
            K[1, 1] = self.K[1, 1] * h / self.h
            K[0, 2] = self.K[0, 2] * w / self.w
            K[1, 2] = self.K[1, 2] * h / self.h
        ix, iy = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        rays_d = torch.stack(
            [(ix - K[0, 2]) / K[0, 0], (iy - K[1, 2]) / K[1, 1], torch.ones_like(ix)], -1
        ).float()  # camera coordinate
        return rays_d

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

    def get_depth(self):
        return self.depth

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
        device = self.points.device
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

        # Write projected points back
        full_mask = mask.clone()
        full_mask[mask] = oob_mask
        self.points[full_mask] = projected_cam
        self.depth[full_mask] = projected_cam[:, 2]

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
        offset: torch.Tensor,
        ref_pose: torch.Tensor,
    ) -> None:
        self.stamp = fid
        self.points = pointcloud
        self.offset = offset

        if ref_pose.ndim != 2:
            ref_pose = ref_pose.reshape(4, 4)
        if not isinstance(ref_pose, torch.Tensor):  # from gt data
            self.ref_pose = torch.tensor(ref_pose, requires_grad=False, dtype=torch.float32)
        else:  # from tracked data
            self.ref_pose = ref_pose.clone().requires_grad_(False)
        self.ref_pose[:3, 3] += offset  # Offset ensures voxel coordinates > 0
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

    def get_depth(self):
        return torch.norm(self.points, dim=-1)  # (N,)

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
