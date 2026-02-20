import torch


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
        max_depth: float,
        min_depth: float,
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
        self.valid_mask: torch.Tensor = (self.depth > min_depth) & (self.depth < max_depth)  # (H, W) depth > 0

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

        # 在同一设备上进行索引操作
        points = points[valid_mask].reshape(-1, 3)  # [N,3]

        if to_world_frame:
            pose = self.get_ref_pose().to(device)
            points = points @ pose[:3, :3].T + pose[:3, 3]  # to world coordinates
            # print(f"points_min: {points.reshape(-1, 3).min(dim=0)}, points_max: {points.reshape(-1, 3).max(dim=0)}")
        return points

    def get_rays_direction(self):
        return self.rays_d

    def get_depth(self):
        return self.depth

    def get_valid_mask(self):
        return self.valid_mask

    def _project_points_to_boundary(
        self, bound_min: torch.Tensor, bound_max: torch.Tensor, points_out_of_bound: torch.Tensor
    ):
        origin = self.get_ref_translation().view(1, 3)  # (1, 3) camera position (assumed to be inside the boundary)

        if (origin < bound_min).any() or (origin > bound_max).any():
            raise ValueError("Camera origin is outside the bounding box, which is not allowed for projection.")

        ray_dir = points_out_of_bound - origin  # direction from camera to out-of-bound points

        inv_dir = 1.0 / (ray_dir + 1e-8)
        t_min_planes = (bound_min.view(1, 3) - origin) * inv_dir
        t_max_planes = (bound_max.view(1, 3) - origin) * inv_dir

        # Compute exit point of ray and bounding box (first intersection since origin is inside boundary)
        t_max_each_dim = torch.max(t_min_planes, t_max_planes)
        t_exit = torch.min(t_max_each_dim, dim=-1)[0]

        # Slightly shrink the projected point toward interior (epsilon) to avoid floating-point border issues
        epsilon = 0.9999
        projected_points_world = origin + (t_exit * epsilon).view(-1, 1) * ray_dir

        # Transform back to camera coordinate system
        R_w2c = self.ref_pose[:3, :3].T
        t_w2c = -R_w2c @ self.ref_pose[:3, 3]
        projected_points_cam = projected_points_world @ R_w2c.T + t_w2c

        return projected_points_cam

    def apply_bound(self, bound_min: torch.Tensor, bound_max: torch.Tensor):
        points = self.points[self.valid_mask] @ self.ref_pose[:3, :3].T + self.ref_pose[:3, 3]

        in_bound_mask = (points >= bound_min.view(1, 3)) & (points <= bound_max.view(1, 3))
        out_of_bound_mask = ~in_bound_mask.all(dim=-1)

        points_out_of_bound = points[out_of_bound_mask]
        if points_out_of_bound.shape[0] > 0:
            new_points = self._project_points_to_boundary(bound_min, bound_max, points_out_of_bound)
            # Flatten points and depth for processing
            points_flat = self.points[self.valid_mask]  # (N, 3)
            depth_flat = self.depth[self.valid_mask]  # (N,)

            # Modify the flattened data for out-of-bound points
            points_flat[out_of_bound_mask] = new_points
            depth_flat[out_of_bound_mask] = new_points[:, 2]

            # Write back to original tensor
            self.points[self.valid_mask] = points_flat
            self.depth[self.valid_mask] = depth_flat

    # def apply_bound(self, bound_min: torch.Tensor, bound_max: torch.Tensor):
    #     points = self.points @ self.ref_pose[:3, :3].T + self.ref_pose[:3, 3]
    #     mask = points >= bound_min.view(1, 1, 3)
    #     mask = mask & (points <= bound_max.view(1, 1, 3))
    #     mask = mask.all(dim=-1)
    #     if mask.any():
    #         # We want the min/max across the H and W dimensions, not the channel dim
    #         # Use .view(-1, 3) to flatten spatial dims for easy min/max calculation
    #         actual_min = points.view(-1, 3).min(dim=0)[0]
    #         actual_max = points.view(-1, 3).max(dim=0)[0]
    #         print(f'Actual Point Bounds - Min: {actual_min}, Max: {actual_max}')
    #     self.valid_mask = self.valid_mask & mask

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
        # 返回局部坐标系中的归一化射线方向，与 DepthFrame 保持一致
        # 后续在 key_frame_set.sample_rays 中会通过旋转矩阵转换到世界坐标系
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
