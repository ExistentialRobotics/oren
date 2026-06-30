import functools

import torch


@functools.lru_cache(maxsize=16)
def _camera_ray_grid(
    height: int,
    width: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the cached per-pixel unit-depth ray grid for a given image size, intrinsics, device, and dtype.

    Returns `(dir_x, dir_y)`, each `(H, W)`, with `dir_x = (u + 0.5 - cx) / fx` and `dir_y = (v + 0.5 - cy) / fy`
    (pixel centers at `(u + 0.5, v + 0.5)`). Multiplying these by a depth map yields camera-frame x / y. The grid
    depends only on `(H, W, K)`, so caching it removes the per-frame `arange` / `meshgrid` rebuild that dominated
    `depth_to_camera_points`. The cache is keyed on scalar intrinsics (Python floats) so a `torch.Tensor` K does not
    break hashing. The returned tensors are shared across calls and must not be mutated in place.
    """
    u = torch.arange(width, device=device, dtype=dtype) + 0.5
    v = torch.arange(height, device=device, dtype=dtype) + 0.5
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    return (uu - cx) / fx, (vv - cy) / fy


@functools.lru_cache(maxsize=16)
def _camera_ray_directions(
    height: int,
    width: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x, y = _camera_ray_grid(height, width, fx, fy, cx, cy, device, dtype)
    z = torch.ones_like(x)
    rays = torch.stack([x, y, z], dim=-1)
    return rays


def camera_ray_directions(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Return the per-pixel unit-depth ray directions for a given intrinsics matrix K.

    Args:
        depth: (H, W) meters.
        K: (3, 3) intrinsics at the depth scale.
    """
    height, width = depth.shape
    return _camera_ray_directions(
        height,
        width,
        float(K[0, 0]),
        float(K[1, 1]),
        float(K[0, 2]),
        float(K[1, 2]),
        depth.device,
        depth.dtype,
    )


def depth_to_camera_points(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Backproject a depth map to camera-frame points.

    Pixel centers are at `(u + 0.5, v + 0.5)`.

    The per-pixel ray grid is cached by `_camera_ray_grid` keyed on `(H, W, K, device, dtype)`,
    so repeated calls with the same image size and intrinsics only pay the elementwise
    multiply by `depth`.

    Args:
        depth: (H, W) meters.
        K: (3, 3) intrinsics at the depth scale.

    Returns:
        camera_points: `(H, W, 3)`.
    """
    height, width = depth.shape
    dir_x, dir_y = _camera_ray_grid(
        int(height),
        int(width),
        float(K[0, 0]),
        float(K[1, 1]),
        float(K[0, 2]),
        float(K[1, 2]),
        depth.device,
        depth.dtype,
    )
    x_cam = dir_x * depth
    y_cam = dir_y * depth
    return torch.stack([x_cam, y_cam, depth], dim=-1)


def depth_to_world_points(
    depth: torch.Tensor,
    pose: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """Backproject a depth map to world-space points.

    Thin wrapper around :func:`depth_to_camera_points` that applies `pose` after the camera-frame
    backprojection. Kept for callers that already operate on world-frame points and don't
    construct a frame object.

    Args:
        depth: (H, W) meters.
        pose: (4, 4) cam->world (T_wc).
        K: (3, 3) intrinsics at the depth scale.

    Returns:
        world_points: `(H, W, 3)`.
    """
    pts_cam = depth_to_camera_points(depth, K)
    return pts_cam @ pose[:3, :3].T + pose[:3, 3]
