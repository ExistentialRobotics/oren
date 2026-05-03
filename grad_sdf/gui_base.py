import multiprocessing as mp
import os
import queue
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import yaml
from matplotlib import cm
from tqdm import tqdm

from grad_sdf import MarchingCubes, np, o3d, o3d_gui, o3d_rendering
from grad_sdf.utils.config_abc import ConfigABC
from grad_sdf.utils.dict_util import flatten_dict
from grad_sdf.utils.profiling import CpuTimer


@dataclass
class GuiBaseConfig(ConfigABC):
    scene_width: int = 2560
    scene_height: int = 1440
    panel_split_ratio: float = 0.7

    view_option: str = "follow"  # follow, keyboard, from_file
    view_file: Optional[str] = None

    # scan, traj, kf_cams, current_cam, octree, sdf, sdf_prior, sdf_residual, mesh, mesh_prior,
    # gt_mesh, sample_surf, sample_perturb, sample_free, sample_extra
    objects: list = field(default_factory=lambda: ["scan", "traj", "current_camera", "sdf"])

    scan_point_size: int = 2
    scan_point_downsample: int = 2
    scan_point_color: list = field(default_factory=lambda: [0.9, 0.9, 0.9])

    traj_line_width: int = 2
    traj_color: list = field(default_factory=lambda: [1.0, 0.0, 0.0])

    sample_update_freq: int = 1
    sample_point_size: int = 2
    sample_point_downsample: int = 2
    sample_color_surf: list = field(default_factory=lambda: [1.0, 0.5, 0.5])
    sample_color_perturbed: list = field(default_factory=lambda: [0.5, 0.5, 1.0])
    sample_color_free: list = field(default_factory=lambda: [1.0, 0.5, 0.5])
    sample_color_extra: list = field(default_factory=lambda: [1.0, 0.7, 0.7])

    camera_line_width: int = 3
    camera_size: float = 1.0
    camera_color_current: list = field(default_factory=lambda: [0.0, 1.0, 0.0])
    camera_color_key_frame: list = field(default_factory=lambda: [0.0, 0.0, 1.0])
    camera_color_selected_key_frame: list = field(default_factory=lambda: [1.0, 0.5, 0.0])

    octree_update_freq: int = 20
    octree_min_size: int = 1
    octree_line_width: int = 3
    octree_line_color: list = field(default_factory=lambda: [0.7, 0.7, 0.7])

    mesh_update_freq: int = 10
    mesh_resolution: int = 100
    mesh_clean: bool = True  # extract mesh only when a surface is likely present
    mesh_height_color_map: str = "jet"
    mesh_color_option: str = "normal"  # normal, height

    sdf_slice_update_freq: int = 10
    sdf_slice_resolution: int = 100
    sdf_point_size: int = 20
    sdf_slice_axis: int = 2
    sdf_slice_position: float = 0.0
    sdf_color_map: str = "jet"  # jet, bwr, viridis

    experiment_name: str = "grad_sdf"
    scene_bound_min: Optional[list] = None
    scene_bound_max: Optional[list] = None
    gt_mesh_path: Optional[str] = None

    mesh_remove_ceiling: bool = True
    mesh_ceiling_thickness: float = 0.2

    save_video_path: Optional[str] = None
    video_fps: int = 30
    video_codec: str = "mp4v"  # Alternative: "XVID", "MJPG"
    video_auto_record: bool = False
    video_auto_end: bool = False  # if True, end the video when mapping ends


@dataclass
class GuiControlPacket:
    """
    This class is used to send control signals from GUI to the mapping process.
    """

    flag_mapping_run: bool = True
    flag_gui_closed: bool = False

    sample_update_frequency: int = -1  # if > 0, update samples every N frames

    octree_update_frequency: int = -1  # if > 0, update octree every N frames
    octree_min_size: int = 1  # if > 0, set the min size of the octree

    sdf_slice_frequency: int = -1
    sdf_slice_axis: int = 2
    sdf_slice_position: float = 0.0
    sdf_slice_resolution: float = -1

    sdf_grid_frequency: int = -1
    sdf_grid_resolution: float = -1
    sdf_grid_ignore_large_voxels: bool = True

    save_model_to_path: Optional[str] = None  # if not None, save the model to this path


@dataclass
class GuiDataPacket:
    """
    This class is used to receive data from the mapping process to GUI.
    """

    flag_exit: bool = False  # whether the mapping process is going to exit
    mapping_end: bool = False  # whether the mapping has ended

    flag_new_frame: bool = False  # whether a new frame is received
    num_iterations: int = -1  # number of iterations used for the current frame
    frame_idx: int = -1  # current frame index
    frame_pose: Optional[np.ndarray] = None  # used to build trajectory
    scan_points: Optional[np.ndarray] = None  # (N, 3) point cloud of the scan
    key_frame_indices: Optional[list] = None  # indices of key frames
    selected_key_frame_indices: Optional[list] = None  # indices of selected key frames

    flag_samples_updated: bool = False  # whether the samples have been updated
    sampled_xyz: Optional[np.ndarray] = None  # (M, 3) array of sampled points
    extra_surface_pcd: Optional[np.ndarray] = None  # (K, 3) array of extra surface points
    n_free: int = 0
    n_perturbed: int = 0
    n_surf: int = 0

    flag_octree_updated: bool = False  # whether the octree has been updated
    octree_voxel_centers: Optional[np.ndarray] = None  # (M, 3) array of voxel centers (x, y, z)
    octree_voxel_sizes: Optional[np.ndarray] = None  # (M, 1) array of voxel sizes
    octree_vertices: Optional[np.ndarray] = None  # (N, 3) array of voxel corner indices
    octree_little_endian_vertex_order: bool = False
    octree_resolution: Optional[float] = None  # voxel size in meters

    flag_sdf_slice_updated: bool = False  # whether the mesh has been updated
    sdf_slice_bounds: Optional[list] = None  # (bound_min, bound_max)
    sdf_slice_axis: Optional[int] = None  # 0, 1, or 2
    sdf_slice_position: Optional[float] = None  # position along the axis
    sdf_slice_resolution: Optional[float] = None  # cell size in meters
    sdf_slice: Optional[dict[str, np.ndarray]] = None  # str -> (H, W) array of SDF values

    flag_sdf_grid_updated: bool = False  # whether the SDF grid has been updated
    sdf_grid_bounds: Optional[list] = None  # (bound_min, bound_max)
    sdf_grid_resolution: Optional[float] = None  # voxel size in meters
    sdf_grid_mask: Optional[np.ndarray] = None  # (X, Y, Z) boolean mask of valid grid vertices
    sdf_grid_shape: Optional[list] = None  # (3, ) shape of the grid
    sdf_grid: Optional[dict[str, np.ndarray]] = None  # str -> (X, Y, Z) array of SDF values in a grid

    model_saved_path: Optional[str] = None  # path where the model is saved

    time_stats: Optional[dict] = None  # time statistics
    loss_stats: Optional[dict] = None  # loss statistics
    gpu_mem_usage: Optional[float] = None  # GPU memory usage in GB


@dataclass
class MarchingCubesTask:
    name: str
    coords_min: list
    grid_res: list
    grid_shape: np.ndarray
    grid_values: np.ndarray
    grid_mask: Optional[np.ndarray]


@dataclass
class MarchingCubesResult:
    name: str
    vertices: np.ndarray
    triangles: np.ndarray
    triangle_normals: np.ndarray


def marching_cubes_process(queue_in: mp.Queue, queue_out: mp.Queue):
    tqdm.write("[Marching Cubes] Marching cubes process started.")
    mc = MarchingCubes()
    while True:
        try:
            task: MarchingCubesTask = queue_in.get(timeout=0.5)  # Reduced timeout for faster response
        except queue.Empty:
            continue
        if task is None:
            tqdm.write("[Marching Cubes] Received termination signal.")
            break
        mask = task.grid_mask
        if mask is None:
            grid_values = task.grid_values.flatten()
        else:
            grid_values = np.zeros(mask.shape, dtype=np.float64)
            mask = mask.astype(bool)
            grid_values[mask] = task.grid_values
            mask = mask.flatten()
            grid_values = grid_values.flatten()

        try:
            vertices, triangles, triangle_normals = mc.run(
                coords_min=task.coords_min,
                grid_res=task.grid_res,
                grid_shape=task.grid_shape,
                grid_values=grid_values,
                mask=mask,
                iso_value=0.0,
                row_major=True,
                parallel=True,
            )
        except KeyboardInterrupt:
            break
        except Exception as e:
            tqdm.write(f"[Marching Cubes] Exception in marching cubes process: {e}")
            continue
        result = MarchingCubesResult(
            name=task.name,
            vertices=np.array(vertices.T).astype(np.float64),
            triangles=np.array(triangles.T).astype(np.int32),
            triangle_normals=np.array(triangle_normals.T).astype(np.float64),
        )
        queue_out.put_nowait(result)
    tqdm.write("[Marching Cubes] Exiting marching cubes process.")


def octree_lineset_from_voxels(
    vertex_offsets,
    octree_voxel_lines,
    voxel_vertices,
    voxel_centers,
    voxel_sizes,
    resolution,
    scene_bound_min=None,
    scene_bound_max=None,
):
    n_vertices = np.max(voxel_vertices) + 1
    vertices = voxel_centers.reshape(-1, 1, 3) + voxel_sizes.reshape(-1, 1, 1) * vertex_offsets
    vertices_unique = np.zeros((n_vertices, 3), dtype=np.float64)
    vertices_unique[voxel_vertices.flatten()] = vertices.reshape(-1, 3)
    lines = voxel_vertices[:, octree_voxel_lines].reshape(-1, 2)  # (N, 12, 2) -> (N*12, 2)

    # remove duplicate lines
    lines = np.sort(lines, axis=1)  # (N*12, 2)
    lines = np.unique(lines, axis=0)  # (M, 2), M <= N*12

    # remove lines outside the bounding box
    if scene_bound_min is not None and scene_bound_max is not None:
        mask = np.all(  # (M, 2, 3)
            (vertices_unique[lines] >= scene_bound_min) & (vertices_unique[lines] <= scene_bound_max),
            axis=(1, 2),
        )
        lines = lines[mask].copy()

    # keep only vertices that are used by the remaining lines
    unique_vertex_indices = np.unique(lines)
    index_mapping = -np.ones(n_vertices, dtype=np.int32)
    index_mapping[unique_vertex_indices] = np.arange(len(unique_vertex_indices), dtype=np.int32)
    lines = index_mapping[lines]  # (M, 2)
    vertices_unique = vertices_unique[unique_vertex_indices]

    # remove short lines overlapped by longer lines
    line_vertices = np.round(vertices_unique[lines] / resolution).astype(np.int64)  # (M, 2, 3), in voxel coordinates
    line_axes = np.argmax(np.abs(line_vertices[:, 1] - line_vertices[:, 0]), axis=1)  # (M,)

    kept_lines = []

    for axis in range(3):
        axis_mask = line_axes == axis
        lines_axis = lines[axis_mask]
        if len(lines_axis) == 0:
            continue
        line_vertices_axis = line_vertices[axis_mask]  # (M', 2, 3)
        flip = line_vertices_axis[:, 0, axis] > line_vertices_axis[:, 1, axis]
        line_vertices_axis[flip] = line_vertices_axis[flip][:, ::-1, :]
        line_coords = np.concatenate(  # (M', 4)
            [
                line_vertices_axis[:, 1],  # end point
                line_vertices_axis[:, [0], axis] - line_vertices_axis[:, [1], axis],  # negative length
            ],
            axis=1,
        )

        order = np.lexsort(line_coords.T[::-1])  # sort by (end_x, end_y, end_z, -length)
        line_coords = line_coords[order]
        lines_axis = lines_axis[order]

        _, unique_indices = np.unique(line_coords[:, :3], axis=0, return_index=True)
        lines_axis = lines_axis[unique_indices]
        kept_lines.append(lines_axis)

    if len(kept_lines) > 0:
        lines = np.concatenate(kept_lines, axis=0)
    else:
        return None, None

    return vertices_unique, lines


@dataclass
class OctreeLinesetTask:
    voxel_centers: np.ndarray
    voxel_sizes: np.ndarray
    voxel_vertices: np.ndarray
    octree_resolution: float
    little_endian_vertex_order: bool
    scene_bound_min: Optional[list] = None
    scene_bound_max: Optional[list] = None


@dataclass
class OctreeLinesetResult:
    vertices: np.ndarray
    lines: np.ndarray


def octree_lineset_process(queue_in: mp.Queue, queue_out: mp.Queue):
    cut = np.array([-0.5, 0.5], dtype=np.float32)
    xx, yy, zz = np.meshgrid(cut, cut, cut, indexing="ij")  # big-endian
    octree_vertex_offsets = np.stack([xx, yy, zz], axis=-1).reshape(1, 8, 3)  # (1,8,3)
    octree_vertex_offsets = [
        octree_vertex_offsets,  # big-endian
        octree_vertex_offsets[:, :, ::-1].copy(),  # little-endian
    ]

    octree_voxel_lines = np.array(
        [
            [0, 1],
            [1, 3],
            [3, 2],
            [2, 0],
            [0, 4],
            [4, 5],
            [5, 7],
            [7, 6],
            [6, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=np.int32,
    )

    tqdm.write("[Octree Lineset] Octree lineset process started.")

    while True:
        try:
            task: OctreeLinesetTask = queue_in.get(timeout=0.5)
        except queue.Empty:
            continue
        if task is None:
            break
        vertex_offsets = octree_vertex_offsets[1 if task.little_endian_vertex_order else 0]
        vertices, lines = octree_lineset_from_voxels(
            vertex_offsets=vertex_offsets,
            octree_voxel_lines=octree_voxel_lines,
            voxel_vertices=task.voxel_vertices,
            voxel_centers=task.voxel_centers,
            voxel_sizes=task.voxel_sizes,
            resolution=task.octree_resolution,
            scene_bound_min=task.scene_bound_min,
            scene_bound_max=task.scene_bound_max,
        )
        if vertices is not None and lines is not None:
            result = OctreeLinesetResult(
                vertices=vertices.astype(np.float64),
                lines=lines.astype(np.int32),
            )
            queue_out.put_nowait(result)

    tqdm.write("[Octree Lineset] Exiting octree lineset process.")


@dataclass
class VideoFrame:
    flag_start: bool = False
    flag_end: bool = False
    path: Optional[str] = None
    fps: int = 30
    frame_size: Optional[tuple] = None  # (width, height)
    frame: Optional[np.ndarray] = None


def video_writer_process(queue_in: mp.Queue, queue_out: mp.Queue):
    tqdm.write("[Video Writer] Video writer process started.")

    video_writer = None
    video_path = None

    def init_video_writer(path, fps, frame_size):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can choose other codecs
        writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        if not writer.isOpened():
            return None
        tqdm.write(f"[Video Writer] Video writer initialized for {path} at {fps} FPS and size {frame_size}.")
        return writer

    def write_frame(writer, frame):
        if writer is not None and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Note: RGB to BGR for video
            writer.write(frame)

    def close_video_writer(writer):
        if writer is not None:
            writer.release()
            tqdm.write(f"[Video Writer] Video writer closed.")

    while True:
        try:
            task: VideoFrame = queue_in.get(timeout=0.5)
        except queue.Empty:
            continue
        if task is None:
            close_video_writer(video_writer)
            break
        if task.flag_start:
            if video_writer is not None:
                close_video_writer(video_writer)
            video_writer = init_video_writer(task.path, task.fps, task.frame_size)
            if video_writer is None:
                tqdm.write(f"[Video Writer] Failed to open video writer for {task.path}")
                queue_out.put_nowait(None)
            else:
                tqdm.write(f"[Video Writer] Started recording video to {task.path}")
                video_path = task.path
                queue_out.put_nowait(task)  # Notify that recording has started
        elif task.flag_end:
            if video_writer is not None:
                close_video_writer(video_writer)
                tqdm.write(f"[Video Writer] Stopped recording video to {video_path}")
                video_writer = None
                video_path = None
                queue_out.put_nowait(task)  # Notify that recording has stopped
        else:
            write_frame(video_writer, task.frame)

    tqdm.write("[Video Writer] Exiting video writer process.")


class GuiBase:

    def __init__(self, cfg: GuiBaseConfig, queue_to_gui: mp.Queue = None, queue_from_gui: mp.Queue = None):
        o3d_gui.Application.instance.initialize()

        self.cfg = cfg
        if self.cfg.objects is None:
            self.cfg.objects = []

        self.queue_in = queue_to_gui
        self.queue_out = queue_from_gui

        self.last_control_packet_timestamp = time.time()
        self.data_packet = GuiDataPacket()  # buffer to hold the latest of each field received
        self.sleep_interval = 0.01  # second
        self.frame_poses = []
        self.frame_indices = []
        self.traj_length = 0
        self.checked_objects = set()

        self.camera_init = False
        self.view_file_loaded = False
        self.sdf_slice_to_save = None

        self.timer_gui_update = CpuTimer(message="[GUI] Update", warmup=10, verbose=False)

        self._cleanup_lock = threading.Lock()
        self._cleaned = False

        self._init_widgets()

        self.mc_queue_in = mp.Queue()
        self.mc_queue_out = mp.Queue()
        self.mc_process = mp.Process(
            target=marching_cubes_process,
            args=(self.mc_queue_in, self.mc_queue_out),
        )
        self.mc_process.start()

        self.octree_queue_in = mp.Queue()
        self.octree_queue_out = mp.Queue()
        self.octree_process = mp.Process(
            target=octree_lineset_process, args=(self.octree_queue_in, self.octree_queue_out)
        )
        self.octree_process.start()

        self.video_queue_in = mp.Queue()
        self.video_queue_out = mp.Queue()
        self.video_writer_process = mp.Process(
            target=video_writer_process, args=(self.video_queue_in, self.video_queue_out)
        )
        self.video_writer_process.start()
        if self.cfg.save_video_path is not None and self.cfg.video_auto_record:
            self._init_video_writer(self.cfg.save_video_path)

        self.comm_thread = threading.Thread(target=self.communicate_thread)
        self.comm_thread.start()

    def _update_checked_objects(self):
        self.checked_objects = set()
        if self.checkbox_show_axis.checked:
            self.checked_objects.add("axis")
        if self.checkbox_show_scan.checked:
            self.checked_objects.add("scan")
        if self.checkbox_show_traj.checked:
            self.checked_objects.add("traj")
        if self.checkbox_show_kf_cams.checked:
            self.checked_objects.add("kf_cams")
        if self.checkbox_show_curr_cam.checked:
            self.checked_objects.add("curr_cam")
        if self.checkbox_show_octree.checked:
            self.checked_objects.add("octree")
        if self.checkbox_show_sdf.checked:
            self.checked_objects.add("sdf")
        if self.checkbox_show_sdf_prior.checked:
            self.checked_objects.add("sdf_prior")
        if self.checkbox_show_sdf_res.checked:
            self.checked_objects.add("sdf_res")
        if self.checkbox_show_mesh.checked:
            self.checked_objects.add("mesh")
        if self.checkbox_show_mesh_by_prior.checked:
            self.checked_objects.add("mesh_prior")
        if self.checkbox_show_gt_mesh.checked:
            self.checked_objects.add("gt_mesh")
        if self.checkbox_show_sample_surf.checked:
            self.checked_objects.add("sample_surf")
        if self.checkbox_show_sample_perturbed.checked:
            self.checked_objects.add("sample_perturbed")
        if self.checkbox_show_sample_free.checked:
            self.checked_objects.add("sample_free")
        if self.checkbox_show_sample_extra.checked:
            self.checked_objects.add("sample_extra")

    @staticmethod
    def _create_collapsable_vert(*widgets, title: str, sp: int, is_open: bool = True):
        cv = o3d_gui.CollapsableVert(title, sp)
        for w in widgets:
            cv.add_child(w)
        cv.set_is_open(is_open)
        return cv

    def _init_widgets(self):
        self.window_width = int(self.cfg.scene_width / self.cfg.panel_split_ratio)
        self.window_height = self.cfg.scene_height

        self.window = o3d_gui.Application.instance.create_window("grad_sdf", self.window_width, self.window_height)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        # left: 3d widget, right: panel

        # I. initialize 3d widget
        # 1. create scene widget
        self.widget3d = o3d_gui.SceneWidget()
        self.widget3d.scene = o3d_rendering.Open3DScene(self.window.renderer)
        cg_setting = o3d_rendering.ColorGrading(
            o3d_rendering.ColorGrading.Quality.ULTRA,
            o3d_rendering.ColorGrading.ToneMapping.LINEAR,
        )
        self.widget3d.scene.view.set_color_grading(cg_setting)
        self.window.add_child(self.widget3d)
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(90.0, bounds, bounds.get_center())  # type: ignore

        self.widget3d_width = int(self.window.size.width * self.cfg.panel_split_ratio)
        self.widget3d_height = self.window.size.height

        # 2. set renders
        # scan
        self.scan_name = "scan"
        self.scan_render = o3d_rendering.MaterialRecord()
        self.scan_render.shader = "defaultLit"  # defaultUnlit, defaultLit, normals, depth
        self.scan_render.point_size = self.cfg.scan_point_size * self.window.scaling

        # trajectory
        self.traj_name = "traj"
        self.traj_render = o3d_rendering.MaterialRecord()
        self.traj_render.shader = "unlitLine"
        self.traj_render.line_width = self.cfg.traj_line_width * self.window.scaling

        # sampled points
        self.sample_free_name = "sample_free"
        self.sample_perturbed_name = "sample_perturbed"
        self.sample_surf_name = "sample_surf"
        self.sample_extra_name = "sample_extra"
        self.sample_render = o3d_rendering.MaterialRecord()
        self.sample_render.shader = "defaultLit"
        self.sample_render.point_size = self.cfg.sample_point_size * self.window.scaling

        # key frame cameras
        self.kf_cams_name = "key_frame_cams"
        self.kf_cams_render = o3d_rendering.MaterialRecord()
        self.kf_cams_render.shader = "unlitLine"
        self.kf_cams_render.line_width = self.cfg.camera_line_width * self.window.scaling

        # current camera
        self.curr_cam_name = "current_cam"
        self.curr_cam_render = o3d_rendering.MaterialRecord()
        self.curr_cam_render.shader = "unlitLine"
        self.curr_cam_render.line_width = self.cfg.camera_line_width * self.window.scaling

        # current camera axis
        self.curr_cam_axis_name = "current_cam_axis"
        self.curr_cam_axis_render = o3d_rendering.MaterialRecord()
        self.curr_cam_axis_render.shader = "defaultLit"

        # octree
        self.octree_name = "octree"
        self.octree_render = o3d_rendering.MaterialRecord()
        self.octree_render.shader = "unlitLine"
        self.octree_render.line_width = self.cfg.octree_line_width * self.window.scaling

        # sdf slice
        self.sdf_name = "sdf_slice"
        self.sdf_prior_name = "sdf_slice_prior"
        self.sdf_residual_name = "sdf_slice_residual"
        self.sdf_render = o3d_rendering.MaterialRecord()
        self.sdf_render.shader = "defaultLit"
        self.sdf_render.point_size = self.cfg.sdf_point_size * self.window.scaling
        self.sdf_render.base_color = [1.0, 1.0, 1.0, 1.0]  # type: ignore

        # mesh
        self.mesh_name = "mesh"
        self.mesh_prior_name = "mesh_prior"
        self.mesh_render = o3d_rendering.MaterialRecord()
        self.mesh_render.shader = "normals"

        # gt mesh
        self.gt_mesh_name = "gt_mesh"
        self.gt_mesh_render = o3d_rendering.MaterialRecord()
        self.gt_mesh_render.shader = "defaultLit"
        self.gt_mesh_render.base_color = [0.9, 0.9, 0.9, 1.0]  # type: ignore

        # 3. create geometry instances
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])  # type: ignore
        self.scan = o3d.geometry.PointCloud()
        self.traj = o3d.geometry.LineSet()
        self.sample_surf = o3d.geometry.PointCloud()
        self.sample_perturbed = o3d.geometry.PointCloud()
        self.sample_free = o3d.geometry.PointCloud()
        self.sample_extra = o3d.geometry.PointCloud()
        self.kf_cams = o3d.geometry.LineSet()
        self.curr_cam = o3d.geometry.LineSet()
        self.curr_cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5 * self.cfg.camera_size, origin=[0, 0, 0]
        )
        self.octree = o3d.geometry.LineSet()
        self.sdf_slice = o3d.geometry.PointCloud()
        self.sdf_slice_prior = o3d.geometry.PointCloud()
        self.sdf_slice_res = o3d.geometry.PointCloud()
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh_prior = o3d.geometry.TriangleMesh()
        self.gt_mesh = None
        if self.cfg.gt_mesh_path is not None and os.path.exists(self.cfg.gt_mesh_path):
            self.gt_mesh = o3d.io.read_triangle_mesh(self.cfg.gt_mesh_path)
            if self.cfg.scene_bound_min is None:
                self.cfg.scene_bound_min = self.gt_mesh.get_min_bound().tolist()
            if self.cfg.scene_bound_max is None:
                self.cfg.scene_bound_max = self.gt_mesh.get_max_bound().tolist()
            if self.cfg.mesh_remove_ceiling:
                self.remove_ceiling_from_mesh(self.gt_mesh)
        self.org_cam = o3d.geometry.LineSet()  # camera of identity pose
        sx = f = self.cfg.camera_size * self.window.scaling
        sy = 0.5 * sx
        self.org_cam.points = o3d.utility.Vector3dVector(
            np.array(
                [
                    [-sx, -sy, f],
                    [sx, -sy, f],
                    [sx, sy, f],
                    [-sx, sy, f],
                    [0, 0, 0],
                ],
                dtype=np.float64,
            )
        )
        self.org_cam.lines = o3d.utility.Vector2iVector(
            np.array(
                [
                    [4, 0],
                    [4, 1],
                    [4, 2],
                    [4, 3],
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0],
                ],
                dtype=np.int32,
            )
        )
        self.org_cam_axis = deepcopy(self.curr_cam_axis)

        # II. initialize panel
        # 1. create panel
        em = self.window.theme.font_size
        sp = 0.5 * em
        self.panel = o3d_gui.Vert(sp, o3d_gui.Margins(left=sp, right=em, top=sp, bottom=sp))

        # 2. add widgets to panel
        # a. switches to pause / resume mapping and visualization
        switch_line = o3d_gui.Horiz(spacing=sp, margins=o3d_gui.Margins())
        self.switch_run = o3d_gui.ToggleSwitch("Pause / Resume Mapping")
        self.switch_run.is_on = True
        self.switch_run.set_on_clicked(self._on_switch_run)
        switch_line.add_child(self.switch_run)

        self.switch_vis = o3d_gui.ToggleSwitch("Pause / Resume Visualization")
        self.switch_vis.is_on = True
        self.switch_vis.set_on_clicked(self._on_switch_vis)
        switch_line.add_child(self.switch_vis)

        self.panel.add_child(switch_line)

        # b. options to control view point
        self.panel.add_child(o3d_gui.Label("View Options:"))
        view_options_line = o3d_gui.Horiz(spacing=sp)

        self.checkbox_view_follow = o3d_gui.Checkbox("Follow")
        self.checkbox_view_follow.checked = True
        self.checkbox_view_follow.set_on_checked(self._on_checkbox_view_follow)
        view_options_line.add_child(self.checkbox_view_follow)

        self.checkbox_view_keyboard = o3d_gui.Checkbox("Keyboard Control")
        self.checkbox_view_keyboard.checked = False
        self.checkbox_view_keyboard.set_on_checked(self._on_checkbox_view_keyboard)
        view_options_line.add_child(self.checkbox_view_keyboard)

        self.checkbox_view_from_file = o3d_gui.Checkbox("From File")
        self.checkbox_view_from_file.checked = False
        self.checkbox_view_from_file.set_on_checked(self._on_checkbox_view_from_file)
        view_options_line.add_child(self.checkbox_view_from_file)

        if self.cfg.view_option.lower() == "follow":
            pass
        elif self.cfg.view_option.lower() == "keyboard":
            self.checkbox_view_follow.checked = False
            self.checkbox_view_keyboard.checked = True
            self.checkbox_view_from_file.checked = False
            self._on_checkbox_view_keyboard(True)
        elif self.cfg.view_option.lower() == "from_file":
            self.checkbox_view_follow.checked = False
            self.checkbox_view_keyboard.checked = False
            self.checkbox_view_from_file.checked = True
            self._on_checkbox_view_keyboard(False)
        else:
            raise ValueError(f"Unknown view option: {self.cfg.view_option}")

        self.panel.add_child(view_options_line)

        # c. options to show / hide different objects
        self.panel.add_child(o3d_gui.Label("3D Objects:"))
        object_options_line1 = o3d_gui.Horiz(spacing=sp)

        init_objects = [obj.lower() for obj in self.cfg.objects]

        self.checkbox_show_axis = o3d_gui.Checkbox("Axis")
        self.checkbox_show_axis.checked = "axis" in init_objects
        self.checkbox_show_axis.set_on_checked(self._on_checkbox_show_axis)
        object_options_line1.add_child(self.checkbox_show_axis)
        self.widget3d.scene.show_axes(self.checkbox_show_axis.checked)

        self.checkbox_show_scan = o3d_gui.Checkbox("Scan")
        self.checkbox_show_scan.checked = "scan" in init_objects
        self.checkbox_show_scan.set_on_checked(self._on_checkbox_show_scan)
        object_options_line1.add_child(self.checkbox_show_scan)

        self.checkbox_show_traj = o3d_gui.Checkbox("Trajectory")
        self.checkbox_show_traj.checked = ("trajectory" in init_objects) or ("traj" in init_objects)
        self.checkbox_show_traj.set_on_checked(self._on_checkbox_show_traj)
        object_options_line1.add_child(self.checkbox_show_traj)

        self.checkbox_show_kf_cams = o3d_gui.Checkbox("KF Cams")
        self.checkbox_show_kf_cams.checked = (
            ("key_frame_cams" in init_objects)
            or ("key_frame_cam" in init_objects)
            or ("key_frame_cameras" in init_objects)
            or ("key_frame_camera" in init_objects)
            or ("kf_cams" in init_objects)
        )
        self.checkbox_show_kf_cams.set_on_checked(self._on_checkbox_show_kf_cams)
        object_options_line1.add_child(self.checkbox_show_kf_cams)

        self.checkbox_show_curr_cam = o3d_gui.Checkbox("Current Cam")
        self.checkbox_show_curr_cam.checked = (
            ("current_camera" in init_objects) or ("curr_cam" in init_objects) or ("current_cam" in init_objects)
        )
        self.checkbox_show_curr_cam.set_on_checked(self._on_checkbox_show_curr_cam)
        object_options_line1.add_child(self.checkbox_show_curr_cam)

        object_options_line2 = o3d_gui.Horiz(spacing=sp)

        self.checkbox_show_octree = o3d_gui.Checkbox("Octree")
        self.checkbox_show_octree.checked = "octree" in init_objects
        self.checkbox_show_octree.set_on_checked(self._on_checkbox_show_octree)
        object_options_line2.add_child(self.checkbox_show_octree)

        self.checkbox_show_sdf = o3d_gui.Checkbox("SDF Pred.")
        self.checkbox_show_sdf.checked = ("sdf_slice" in init_objects) or ("sdf" in init_objects)
        self.checkbox_show_sdf.set_on_checked(self._on_checkbox_show_sdf)
        object_options_line2.add_child(self.checkbox_show_sdf)

        self.checkbox_show_sdf_prior = o3d_gui.Checkbox("SDF Prior")
        self.checkbox_show_sdf_prior.checked = ("sdf_slice_prior" in init_objects) or ("sdf_prior" in init_objects)
        self.checkbox_show_sdf_prior.set_on_checked(self._on_checkbox_show_sdf_prior)
        object_options_line2.add_child(self.checkbox_show_sdf_prior)

        self.checkbox_show_sdf_res = o3d_gui.Checkbox("SDF Residual")
        self.checkbox_show_sdf_res.checked = ("sdf_slice_residual" in init_objects) or ("sdf_residual" in init_objects)
        self.checkbox_show_sdf_res.set_on_checked(self._on_checkbox_show_sdf_residual)
        object_options_line2.add_child(self.checkbox_show_sdf_res)

        object_options_line3 = o3d_gui.Horiz(spacing=sp)

        self.checkbox_show_mesh = o3d_gui.Checkbox("Mesh")
        self.checkbox_show_mesh.checked = "mesh" in init_objects
        self.checkbox_show_mesh.set_on_checked(self._on_checkbox_show_mesh)
        object_options_line3.add_child(self.checkbox_show_mesh)

        self.checkbox_show_mesh_by_prior = o3d_gui.Checkbox("Mesh (Prior)")
        self.checkbox_show_mesh_by_prior.checked = "mesh_prior" in init_objects
        self.checkbox_show_mesh_by_prior.set_on_checked(self._on_checkbox_show_mesh_by_prior)
        object_options_line3.add_child(self.checkbox_show_mesh_by_prior)

        self.checkbox_show_gt_mesh = o3d_gui.Checkbox("GT Mesh")
        self.checkbox_show_gt_mesh.checked = "gt_mesh" in init_objects
        self.checkbox_show_gt_mesh.set_on_checked(self._on_checkbox_show_gt_mesh)
        object_options_line3.add_child(self.checkbox_show_gt_mesh)

        self.checkbox_show_sample_surf = o3d_gui.Checkbox("Samples (Surface)")
        self.checkbox_show_sample_surf.checked = "surf" in init_objects or "sample_surf" in init_objects
        self.checkbox_show_sample_surf.set_on_checked(self._on_checkbox_show_sample_surf)
        object_options_line3.add_child(self.checkbox_show_sample_surf)

        object_options_line4 = o3d_gui.Horiz(spacing=sp)

        self.checkbox_show_sample_perturbed = o3d_gui.Checkbox("Samples (Perturbed)")
        self.checkbox_show_sample_perturbed.checked = "perturbed" in init_objects or "sample_perturbed" in init_objects
        self.checkbox_show_sample_perturbed.set_on_checked(self._on_checkbox_show_sample_perturbed)
        object_options_line4.add_child(self.checkbox_show_sample_perturbed)

        self.checkbox_show_sample_free = o3d_gui.Checkbox("Samples (Free)")
        self.checkbox_show_sample_free.checked = "free" in init_objects or "sample_free" in init_objects
        self.checkbox_show_sample_free.set_on_checked(self._on_checkbox_show_sample_free)
        object_options_line4.add_child(self.checkbox_show_sample_free)

        self.checkbox_show_sample_extra = o3d_gui.Checkbox("Samples (Extra)")
        self.checkbox_show_sample_extra.checked = "extra" in init_objects or "sample_extra" in init_objects
        self.checkbox_show_sample_extra.set_on_checked(self._on_checkbox_show_sample_extra)
        object_options_line4.add_child(self.checkbox_show_sample_extra)

        self.panel.add_child(object_options_line1)
        self.panel.add_child(object_options_line2)
        self.panel.add_child(object_options_line3)
        self.panel.add_child(object_options_line4)

        # d. control styles, frequency, etc.
        self.panel.add_child(o3d_gui.Label("Rendering Options:"))

        # add options to control scan rendering
        scan_point_size_line = o3d_gui.Horiz(spacing=sp)
        scan_point_size_line.add_child(o3d_gui.Label("Scan Point Size:"))
        self.slider_scan_point_size = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_scan_point_size.set_limits(1, 20)
        self.slider_scan_point_size.int_value = self.cfg.scan_point_size
        self.slider_scan_point_size.set_on_value_changed(self._on_slider_scan_point_size)
        scan_point_size_line.add_child(self.slider_scan_point_size)

        scan_point_downsample_line = o3d_gui.Horiz(spacing=sp)
        scan_point_downsample_line.add_child(o3d_gui.Label("Scan Downsample Factor:"))
        self.slider_scan_point_downsample = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_scan_point_downsample.set_limits(1, 20)
        self.slider_scan_point_downsample.int_value = self.cfg.scan_point_downsample
        self.slider_scan_point_downsample.set_on_value_changed(self._on_slider_scan_point_downsample)
        scan_point_downsample_line.add_child(self.slider_scan_point_downsample)

        scan_point_color_line = o3d_gui.Horiz(spacing=sp)
        scan_point_color_line.add_child(o3d_gui.Label("Scan Color Option:"))
        self.color_edit_scan_point_color = o3d_gui.ColorEdit()
        self.color_edit_scan_point_color.color_value.set_color(*self.cfg.scan_point_color)
        self.color_edit_scan_point_color.set_on_value_changed(self._on_color_edit_scan_point_color)
        scan_point_color_line.add_child(self.color_edit_scan_point_color)

        self.panel.add_child(
            self._create_collapsable_vert(
                scan_point_size_line,
                scan_point_downsample_line,
                scan_point_color_line,
                title="Scan Rendering Options",
                sp=sp,
                is_open=False,
            )
        )

        # options to control trajectory rendering
        traj_line_width_line = o3d_gui.Horiz(spacing=sp)
        traj_line_width_line.add_child(o3d_gui.Label("Trajectory Line Width:"))
        self.slider_traj_line_width = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_traj_line_width.set_limits(1, 10)
        self.slider_traj_line_width.int_value = self.cfg.traj_line_width
        self.slider_traj_line_width.set_on_value_changed(self._on_slider_traj_line_width)
        traj_line_width_line.add_child(self.slider_traj_line_width)

        traj_line_color_line = o3d_gui.Horiz(spacing=sp)
        traj_line_color_line.add_child(o3d_gui.Label("Trajectory Line Color:"))
        self.color_edit_traj_line_color = o3d_gui.ColorEdit()
        self.color_edit_traj_line_color.color_value.set_color(*self.cfg.traj_color)
        self.color_edit_traj_line_color.set_on_value_changed(self._on_color_edit_traj_line_color)
        traj_line_color_line.add_child(self.color_edit_traj_line_color)

        self.panel.add_child(
            self._create_collapsable_vert(
                traj_line_width_line,
                traj_line_color_line,
                title="Trajectory Rendering Options",
                sp=sp,
                is_open=False,
            )
        )

        # options to control sample point rendering
        sample_update_freq_line = o3d_gui.Horiz(spacing=sp)
        sample_update_freq_line.add_child(o3d_gui.Label("Sample Point Update Frequency:"))
        self.slider_sample_update_freq = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_sample_update_freq.set_limits(1, 20)
        self.slider_sample_update_freq.int_value = self.cfg.sample_update_freq
        sample_update_freq_line.add_child(self.slider_sample_update_freq)

        sample_point_size_line = o3d_gui.Horiz(spacing=sp)
        sample_point_size_line.add_child(o3d_gui.Label("Sample Point Size:"))
        self.slider_sample_point_size = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_sample_point_size.set_limits(1, 20)
        self.slider_sample_point_size.int_value = self.cfg.sample_point_size
        self.slider_sample_point_size.set_on_value_changed(self._on_slider_sample_point_size)
        sample_point_size_line.add_child(self.slider_sample_point_size)

        sample_point_downsample_line = o3d_gui.Horiz(spacing=sp)
        sample_point_downsample_line.add_child(o3d_gui.Label("Sample Point Downsample Factor:"))
        self.slider_sample_point_downsample = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_sample_point_downsample.set_limits(1, 20)
        self.slider_sample_point_downsample.int_value = self.cfg.sample_point_downsample
        self.slider_sample_point_downsample.set_on_value_changed(self._on_slider_sample_point_downsample)
        sample_point_downsample_line.add_child(self.slider_sample_point_downsample)

        sample_point_color_surf_line = o3d_gui.Horiz(spacing=sp, margins=o3d_gui.Margins(left=em))
        sample_point_color_surf_line.add_child(o3d_gui.Label("Surface:"))
        self.color_edit_sample_point_color_surf = o3d_gui.ColorEdit()
        self.color_edit_sample_point_color_surf.color_value.set_color(*self.cfg.sample_color_surf)
        self.color_edit_sample_point_color_surf.set_on_value_changed(self._on_color_edit_sample_point_color_surf)
        sample_point_color_surf_line.add_child(self.color_edit_sample_point_color_surf)

        sample_point_color_perturbed_line = o3d_gui.Horiz(spacing=sp, margins=o3d_gui.Margins(left=em))
        sample_point_color_perturbed_line.add_child(o3d_gui.Label("Perturbed Surface:"))
        self.color_edit_sample_point_color_perturbed = o3d_gui.ColorEdit()
        self.color_edit_sample_point_color_perturbed.color_value.set_color(*self.cfg.sample_color_perturbed)
        self.color_edit_sample_point_color_perturbed.set_on_value_changed(
            self._on_color_edit_sample_point_color_perturbed
        )
        sample_point_color_perturbed_line.add_child(self.color_edit_sample_point_color_perturbed)

        sample_point_color_extra_line = o3d_gui.Horiz(spacing=sp, margins=o3d_gui.Margins(left=em))
        sample_point_color_extra_line.add_child(o3d_gui.Label("Extra Samples:"))
        self.color_edit_sample_point_color_extra = o3d_gui.ColorEdit()
        self.color_edit_sample_point_color_extra.color_value.set_color(*self.cfg.sample_color_extra)
        self.color_edit_sample_point_color_extra.set_on_value_changed(self._on_color_edit_sample_point_color_extra)
        sample_point_color_extra_line.add_child(self.color_edit_sample_point_color_extra)

        sample_point_color_free_line = o3d_gui.Horiz(spacing=sp, margins=o3d_gui.Margins(left=em))
        sample_point_color_free_line.add_child(o3d_gui.Label("Free Space:"))
        self.color_edit_sample_point_color_free = o3d_gui.ColorEdit()
        self.color_edit_sample_point_color_free.color_value.set_color(*self.cfg.sample_color_free)
        self.color_edit_sample_point_color_free.set_on_value_changed(self._on_color_edit_sample_point_color_free)
        sample_point_color_free_line.add_child(self.color_edit_sample_point_color_free)

        self.panel.add_child(
            self._create_collapsable_vert(
                sample_update_freq_line,
                sample_point_size_line,
                sample_point_downsample_line,
                o3d_gui.Label("Sample Point Color:"),
                sample_point_color_surf_line,
                sample_point_color_perturbed_line,
                sample_point_color_free_line,
                title="Sample Point Rendering Options",
                sp=sp,
                is_open=False,
            )
        )

        # options to control camera rendering
        camera_line_width_line = o3d_gui.Horiz(spacing=sp)
        camera_line_width_line.add_child(o3d_gui.Label("Camera Line Width:"))
        self.slider_camera_line_width = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_camera_line_width.set_limits(1, 20)
        self.slider_camera_line_width.int_value = self.cfg.camera_line_width
        self.slider_camera_line_width.set_on_value_changed(self._on_slider_camera_line_width)
        camera_line_width_line.add_child(self.slider_camera_line_width)

        camera_size_line = o3d_gui.Horiz(spacing=sp)
        camera_size_line.add_child(o3d_gui.Label("Camera Size:"))
        self.slider_camera_size = o3d_gui.Slider(o3d_gui.Slider.DOUBLE)
        self.slider_camera_size.set_limits(0.05, 2)
        self.slider_camera_size.double_value = self.cfg.camera_size
        self.slider_camera_size.set_on_value_changed(self._on_slider_camera_size)
        camera_size_line.add_child(self.slider_camera_size)

        camera_color_current_line = o3d_gui.Horiz(spacing=sp)
        camera_color_current_line.add_child(o3d_gui.Label("Current Camera Color:"))
        self.color_edit_camera_color_current = o3d_gui.ColorEdit()
        self.color_edit_camera_color_current.color_value.set_color(*self.cfg.camera_color_current)
        self.color_edit_camera_color_current.set_on_value_changed(self._on_color_edit_camera_color_current)
        camera_color_current_line.add_child(self.color_edit_camera_color_current)

        camera_color_kf_line = o3d_gui.Horiz(spacing=sp)
        camera_color_kf_line.add_child(o3d_gui.Label("Key Frame Camera Color:"))
        self.color_edit_camera_color_kf = o3d_gui.ColorEdit()
        self.color_edit_camera_color_kf.color_value.set_color(*self.cfg.camera_color_key_frame)
        self.color_edit_camera_color_kf.set_on_value_changed(self._on_color_edit_camera_color_kf)
        camera_color_kf_line.add_child(self.color_edit_camera_color_kf)

        camera_color_selected_kf_line = o3d_gui.Horiz(spacing=sp)
        camera_color_selected_kf_line.add_child(o3d_gui.Label("Selected Key Frame Camera Color:"))
        self.color_edit_camera_color_selected_kf = o3d_gui.ColorEdit()
        self.color_edit_camera_color_selected_kf.color_value.set_color(*self.cfg.camera_color_selected_key_frame)
        self.color_edit_camera_color_selected_kf.set_on_value_changed(self._on_color_edit_camera_color_selected_kf)
        camera_color_selected_kf_line.add_child(self.color_edit_camera_color_selected_kf)

        self.panel.add_child(
            self._create_collapsable_vert(
                camera_line_width_line,
                camera_size_line,
                camera_color_current_line,
                camera_color_kf_line,
                camera_color_selected_kf_line,
                title="Camera Rendering Options",
                sp=sp,
                is_open=False,
            )
        )

        # options to control octree rendering
        octree_update_freq_line = o3d_gui.Horiz(spacing=sp)
        octree_update_freq_line.add_child(o3d_gui.Label("Octree Update Frequency:"))
        self.slider_octree_update_freq = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_octree_update_freq.set_limits(1, 100)
        self.slider_octree_update_freq.int_value = self.cfg.octree_update_freq
        octree_update_freq_line.add_child(self.slider_octree_update_freq)

        octree_line_width_line = o3d_gui.Horiz(spacing=sp)
        octree_line_width_line.add_child(o3d_gui.Label("Octree Line Width:"))
        self.slider_octree_line_width = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_octree_line_width.set_limits(1, 10)
        self.slider_octree_line_width.int_value = self.cfg.octree_line_width
        self.slider_octree_line_width.set_on_value_changed(self._on_slider_octree_line_width)
        octree_line_width_line.add_child(self.slider_octree_line_width)

        octree_min_size_line = o3d_gui.Horiz(spacing=sp)
        octree_min_size_line.add_child(o3d_gui.Label("Octree Min Size:"))
        self.slider_octree_min_size = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_octree_min_size.set_limits(1, 32)
        self.slider_octree_min_size.int_value = self.cfg.octree_min_size
        octree_min_size_line.add_child(self.slider_octree_min_size)

        octree_color_line = o3d_gui.Horiz(spacing=sp)
        octree_color_line.add_child(o3d_gui.Label("Octree Line Color:"))
        self.color_edit_octree_color = o3d_gui.ColorEdit()
        self.color_edit_octree_color.color_value.set_color(*self.cfg.octree_line_color)
        self.color_edit_octree_color.set_on_value_changed(self._on_color_edit_octree_color)
        octree_color_line.add_child(self.color_edit_octree_color)

        self.panel.add_child(
            self._create_collapsable_vert(
                octree_update_freq_line,
                octree_line_width_line,
                octree_min_size_line,
                octree_color_line,
                title="Octree Rendering Options",
                sp=sp,
                is_open=False,
            )
        )

        # options to control mesh rendering
        mesh_update_freq_line = o3d_gui.Horiz(spacing=sp)
        mesh_update_freq_line.add_child(o3d_gui.Label("Mesh Update Frequency:"))
        self.slider_mesh_update_freq = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_mesh_update_freq.set_limits(1, 100)
        self.slider_mesh_update_freq.int_value = self.cfg.mesh_update_freq
        mesh_update_freq_line.add_child(self.slider_mesh_update_freq)

        mesh_resolution_line = o3d_gui.Horiz(spacing=sp)
        mesh_resolution_line.add_child(o3d_gui.Label("Mesh Resolution (#voxels per meter):"))
        self.slider_mesh_resolution = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_mesh_resolution.set_limits(10, 110)
        self.slider_mesh_resolution.int_value = self.cfg.mesh_resolution
        mesh_resolution_line.add_child(self.slider_mesh_resolution)

        mesh_color_options_line = o3d_gui.Horiz(spacing=sp)
        mesh_color_options_line.add_child(o3d_gui.Label("Mesh Color Options:"))

        self.checkbox_mesh_color_by_normal = o3d_gui.Checkbox("Normal")
        self.checkbox_mesh_color_by_normal.checked = True
        self.checkbox_mesh_color_by_normal.set_on_checked(self._on_checkbox_mesh_color_by_normal)
        mesh_color_options_line.add_child(self.checkbox_mesh_color_by_normal)

        self.checkbox_mesh_color_by_height = o3d_gui.Checkbox("Height")
        self.checkbox_mesh_color_by_height.checked = False
        self.checkbox_mesh_color_by_height.set_on_checked(self._on_checkbox_mesh_color_by_height)
        mesh_color_options_line.add_child(self.checkbox_mesh_color_by_height)

        if self.cfg.mesh_color_option.lower() == "normal":
            pass
        elif self.cfg.mesh_color_option.lower() == "height":
            self._on_checkbox_mesh_color_by_height(True)
        else:
            raise ValueError(f"Unknown mesh color option: {self.cfg.mesh_color_option}")

        self.combobox_mesh_height_color_map = o3d_gui.Combobox()
        self.combobox_mesh_height_color_map.add_item("viridis")
        self.combobox_mesh_height_color_map.add_item("bwr")
        self.combobox_mesh_height_color_map.add_item("jet")
        height_color_map_dict = {"viridis": 0, "bwr": 1, "jet": 2}
        if self.cfg.mesh_height_color_map.lower() not in height_color_map_dict:
            raise ValueError(f"Unknown mesh height color map: {self.cfg.mesh_height_color_map}")
        self.combobox_mesh_height_color_map.selected_index = height_color_map_dict[
            self.cfg.mesh_height_color_map.lower()
        ]
        self.combobox_mesh_height_color_map.set_on_selection_changed(self._on_combobox_mesh_height_color_map)
        mesh_color_options_line.add_child(self.combobox_mesh_height_color_map)

        self.panel.add_child(
            self._create_collapsable_vert(
                mesh_update_freq_line,
                mesh_resolution_line,
                mesh_color_options_line,
                title="Mesh Rendering Options",
                sp=sp,
            )
        )

        # options to control sdf slice rendering
        sdf_update_freq_line = o3d_gui.Horiz(spacing=sp)
        sdf_update_freq_line.add_child(o3d_gui.Label("SDF Slice Update Frequency:"))
        self.slider_sdf_update_freq = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_sdf_update_freq.set_limits(1, 100)
        self.slider_sdf_update_freq.int_value = self.cfg.sdf_slice_update_freq
        sdf_update_freq_line.add_child(self.slider_sdf_update_freq)

        sdf_resolution_line = o3d_gui.Horiz(spacing=sp)
        sdf_resolution_line.add_child(o3d_gui.Label("SDF Slice Resolution (#points per meter):"))
        self.slider_sdf_resolution = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_sdf_resolution.set_limits(10, 200)
        self.slider_sdf_resolution.int_value = self.cfg.sdf_slice_resolution
        sdf_resolution_line.add_child(self.slider_sdf_resolution)

        sdf_point_size_line = o3d_gui.Horiz(spacing=sp)
        sdf_point_size_line.add_child(o3d_gui.Label("SDF Point Size:"))
        self.slider_sdf_point_size = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_sdf_point_size.set_limits(1, 50)
        self.slider_sdf_point_size.int_value = self.cfg.sdf_point_size
        self.slider_sdf_point_size.set_on_value_changed(self._on_slider_sdf_point_size)
        sdf_point_size_line.add_child(self.slider_sdf_point_size)

        sdf_axis_position_line = o3d_gui.Horiz(spacing=sp)
        sdf_axis_position_line.add_child(o3d_gui.Label("SDF Slice Axis:"))
        self.combobox_sdf_axis = o3d_gui.Combobox()
        self.combobox_sdf_axis.add_item("X")
        self.combobox_sdf_axis.add_item("Y")
        self.combobox_sdf_axis.add_item("Z")
        self.combobox_sdf_axis.set_on_selection_changed(self._on_combobox_sdf_axis)
        assert 0 <= self.cfg.sdf_slice_axis <= 2, f"Invalid sdf_slice_axis: {self.cfg.sdf_slice_axis}"
        self.combobox_sdf_axis.selected_index = self.cfg.sdf_slice_axis
        sdf_axis_position_line.add_child(self.combobox_sdf_axis)
        sdf_axis_position_line.add_child(o3d_gui.Label("Position:"))
        self.slider_sdf_position = o3d_gui.Slider(o3d_gui.Slider.DOUBLE)
        if self.cfg.scene_bound_min is None or self.cfg.scene_bound_max is None:
            self.slider_sdf_position.set_limits(-10.0, 10.0)
        else:
            self.slider_sdf_position.set_limits(
                self.cfg.scene_bound_min[self.cfg.sdf_slice_axis],
                self.cfg.scene_bound_max[self.cfg.sdf_slice_axis],
            )
        self.slider_sdf_position.double_value = self.cfg.sdf_slice_position
        self.slider_sdf_position.set_on_value_changed(self._on_slider_sdf_position)
        sdf_axis_position_line.add_child(self.slider_sdf_position)

        sdf_color_map_line = o3d_gui.Horiz(spacing=sp)
        sdf_color_map_line.add_child(o3d_gui.Label("SDF Color Map:"))
        self.combobox_sdf_color_map = o3d_gui.Combobox()
        self.combobox_sdf_color_map.add_item("viridis")
        self.combobox_sdf_color_map.add_item("bwr")
        self.combobox_sdf_color_map.add_item("jet")
        sdf_color_map_dict = {"viridis": 0, "bwr": 1, "jet": 2}
        if self.cfg.sdf_color_map.lower() not in sdf_color_map_dict:
            raise ValueError(f"Unknown sdf color map: {self.cfg.sdf_color_map}")
        self.combobox_sdf_color_map.selected_index = sdf_color_map_dict[self.cfg.sdf_color_map.lower()]
        self.combobox_sdf_color_map.set_on_selection_changed(self._on_combobox_sdf_color_map)
        sdf_color_map_line.add_child(self.combobox_sdf_color_map)

        self.panel.add_child(
            self._create_collapsable_vert(
                sdf_update_freq_line,
                sdf_resolution_line,
                sdf_point_size_line,
                sdf_axis_position_line,
                sdf_color_map_line,
                title="SDF Slice Rendering Options",
                sp=sp,
            )
        )

        # k. buttons to save mesh and sdf
        save_buttons_line = o3d_gui.Horiz(spacing=sp)
        self.button_save_mesh = o3d_gui.Button("Save Mesh")
        self.button_save_mesh.set_on_clicked(self._on_button_save_mesh)
        save_buttons_line.add_child(self.button_save_mesh)

        self.button_save_sdf = o3d_gui.Button("Save SDF")
        self.button_save_sdf.set_on_clicked(self._on_button_save_sdf)
        save_buttons_line.add_child(self.button_save_sdf)

        self.button_save_model = o3d_gui.Button("Save Model")
        self.button_save_model.set_on_clicked(self._on_button_save_model)
        save_buttons_line.add_child(self.button_save_model)

        self.button_save_screenshot = o3d_gui.Button("Save Screenshot")
        self.button_save_screenshot.set_on_clicked(self._on_button_save_screenshot)
        save_buttons_line.add_child(self.button_save_screenshot)

        self.panel.add_child(save_buttons_line)

        # Add video recording buttons on a new line
        video_buttons_line = o3d_gui.Horiz(spacing=sp)
        self.button_start_video = o3d_gui.Button("Start Video Recording")
        self.button_start_video.enabled = True
        self.button_start_video.set_on_clicked(self._on_button_start_video_recording)
        video_buttons_line.add_child(self.button_start_video)

        self.button_stop_video = o3d_gui.Button("Stop Video Recording")
        self.button_stop_video.set_on_clicked(self._on_button_stop_video_recording)
        self.button_stop_video.enabled = False  # Disable if not recording
        video_buttons_line.add_child(self.button_stop_video)

        self.button_save_gui_config = o3d_gui.Button("Save GUI Config")
        self.button_save_gui_config.set_on_clicked(self._on_button_save_gui_config)
        video_buttons_line.add_child(self.button_save_gui_config)

        self.panel.add_child(video_buttons_line)

        # l. info tab
        tabs = o3d_gui.TabControl()
        tab_info = o3d_gui.Vert(spacing=sp)
        self.label_info_experiment_name = o3d_gui.Label(f"Experiment: {self.cfg.experiment_name}")
        tab_info.add_child(self.label_info_experiment_name)
        self.label_info_frame_idx = o3d_gui.Label("Frame:")
        tab_info.add_child(self.label_info_frame_idx)
        self.label_info_traj_length = o3d_gui.Label("Travel Distance:")
        tab_info.add_child(self.label_info_traj_length)
        self.label_info_gpu_mem = o3d_gui.Label("GPU Memory Usage:")
        tab_info.add_child(self.label_info_gpu_mem)
        self.label_info_num_iterations = o3d_gui.Label("Num Iterations:")
        tab_info.add_child(self.label_info_num_iterations)
        self.label_info_num_key_frames = o3d_gui.Label("Num Key Frames:")
        tab_info.add_child(self.label_info_num_key_frames)
        self.label_info_training_fps = o3d_gui.Label("Training FPS:")
        tab_info.add_child(self.label_info_training_fps)
        self.label_info_gui_fps = o3d_gui.Label("GUI FPS:")
        tab_info.add_child(self.label_info_gui_fps)
        self.label_info_timing_and_loss = o3d_gui.Label("Timing:\nLoss:")
        tab_info.add_child(self.label_info_timing_and_loss)
        tabs.add_tab("Info", tab_info)
        tab_info = o3d_gui.Vert(spacing=sp)
        self.label_info_scene_camera = o3d_gui.Label("Scene Camera:\n")
        tab_info.add_child(self.label_info_scene_camera)
        tabs.add_tab("Camera", tab_info)
        self.panel.add_child(tabs)

        # 3. add panel to window
        self.window.add_child(self.panel)

    def _on_layout(self, layout_context):
        content_rect = self.window.content_rect
        self.widget3d_width = int(self.window.size.width * self.cfg.panel_split_ratio)
        self.widget3d_height = content_rect.height
        self.widget3d.frame = o3d_gui.Rect(content_rect.x, content_rect.y, self.widget3d_width, self.widget3d_height)
        self.panel.frame = o3d_gui.Rect(
            self.widget3d.frame.get_right(),
            content_rect.y,
            self.window_width - self.widget3d_width,
            content_rect.height,
        )

    def _send_flag_gui_closed(self):
        if self.queue_out is None:
            return
        while not self.queue_out.empty():
            try:
                self.queue_out.get_nowait()
            except queue.Empty:
                break
        try:
            self.queue_out.put_nowait(GuiControlPacket(flag_gui_closed=True))
        except Exception as e:
            tqdm.write(f"[GUI] Error sending close signal: {e}")

    def _on_close(self):
        self.data_packet.flag_exit = True
        tqdm.write("[GUI] Window is being closed.")
        self._send_flag_gui_closed()
        return True

    def _on_switch_run(self, is_on: bool) -> None:
        if is_on:
            tqdm.write("[GUI] Resume mapping.")
        else:
            tqdm.write("[GUI] Pause mapping.")

    def _on_switch_vis(self, is_on: bool) -> None:
        if is_on:
            tqdm.write("[GUI] Resume visualization.")
        else:
            tqdm.write("[GUI] Pause visualization.")

    def _on_checkbox_view_follow(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] View follow mode on.")
            self.checkbox_view_keyboard.checked = False
            self._on_checkbox_view_keyboard(False)
            self.checkbox_view_from_file.checked = False
        else:
            tqdm.write("[GUI] View follow mode off.")

    def _on_checkbox_view_keyboard(self, is_checked: bool) -> None:
        if is_checked:
            self.checkbox_view_follow.checked = False
            self.checkbox_view_from_file.checked = False
            tqdm.write("[GUI] View keyboard control on.")
            # control like a game using WASD,Q,Z,E,R, up, right, left, down
            self.widget3d.set_view_controls(o3d_gui.SceneWidget.Controls.FLY)
        else:
            tqdm.write("[GUI] View keyboard control off.")
            self.widget3d.set_view_controls(o3d_gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE)

    def _on_checkbox_view_from_file(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] View from file on.")
            self.checkbox_view_follow.checked = False
            self.checkbox_view_keyboard.checked = False
            self._on_checkbox_view_keyboard(False)

            dialog = o3d_gui.FileDialog(o3d_gui.FileDialog.OPEN, "Select Camera View File", self.window.theme)
            dialog.add_filter("", "*.yaml")
            dialog.set_on_cancel(lambda: self.window.close_dialog())

            def on_done(filename: str):
                self.set_view_from_file(filename)
                self.cfg.view_file = filename
                self.view_file_loaded = True
                self.window.close_dialog()

            dialog.set_on_done(on_done)
            self.window.show_dialog(dialog)
        else:
            tqdm.write("[GUI] View from file off.")
            self.view_file_loaded = False

    def _on_checkbox_show_axis(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show axis.")
        else:
            tqdm.write("[GUI] Hide axis.")
        self.widget3d.scene.show_axes(is_checked)

    def _on_checkbox_show_scan(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show scan.")
        else:
            tqdm.write("[GUI] Hide scan.")
        if self.data_packet.flag_new_frame:
            return  # will be handled soon
        self.visualize_scan()

    def _on_checkbox_show_traj(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show trajectory.")
        else:
            tqdm.write("[GUI] Hide trajectory.")
        if self.data_packet.flag_new_frame:
            return  # will be handled soon
        self.visualize_trajectory()

    def _on_checkbox_show_kf_cams(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show key frame cameras.")
        else:
            tqdm.write("[GUI] Hide key frame cameras.")
        if self.data_packet.flag_new_frame:
            return  # will be handled soon
        self.visualize_kf_cams()

    def _on_checkbox_show_curr_cam(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show current camera.")
        else:
            tqdm.write("[GUI] Hide current camera.")
        if self.data_packet.flag_new_frame:
            return  # will be handled soon
        self.visualize_curr_cam()

    def _on_checkbox_show_octree(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show octree.")
        else:
            tqdm.write("[GUI] Hide octree.")
        if self.data_packet.flag_octree_updated:
            return  # will be handled soon
        self.visualize_octree()

    def _on_checkbox_show_sdf(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show SDF slice.")
        else:
            tqdm.write("[GUI] Hide SDF slice.")
        if self.data_packet.flag_sdf_slice_updated:
            return  # will be handled soon
        self.visualize_sdf_slice(self.sdf_name, self.sdf_slice, is_checked)

    def _on_checkbox_show_sdf_prior(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show SDF slice by prior.")
        else:
            tqdm.write("[GUI] Hide SDF slice by prior.")
        if self.data_packet.flag_sdf_slice_updated:
            return  # will be handled soon
        self.visualize_sdf_slice(self.sdf_prior_name, self.sdf_slice_prior, is_checked)

    def _on_checkbox_show_sdf_residual(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show SDF slice residual.")
        else:
            tqdm.write("[GUI] Hide SDF slice residual.")
        if self.data_packet.flag_sdf_slice_updated:
            return  # will be handled soon
        self.visualize_sdf_slice(self.sdf_residual_name, self.sdf_slice_res, is_checked)

    def _on_checkbox_show_mesh(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show mesh.")
        else:
            tqdm.write("[GUI] Hide mesh.")
        if self.data_packet.flag_sdf_grid_updated:
            return  # will be handled soon
        self.visualize_mesh()

    def _on_checkbox_show_mesh_by_prior(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show mesh by prior.")
        else:
            tqdm.write("[GUI] Hide mesh by prior.")
        if self.data_packet.flag_sdf_grid_updated:
            return  # will be handled soon
        self.visualize_mesh()

    def _on_checkbox_show_gt_mesh(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show ground truth mesh.")
        else:
            tqdm.write("[GUI] Hide ground truth mesh.")
        self.visualize_gt_mesh()

    def _on_slider_scan_point_size(self, point_size: int) -> None:
        self.cfg.point_size = point_size
        self.scan_render.point_size = point_size * self.window.scaling
        self.visualize_scan()

    def _on_slider_scan_point_downsample(self, downsample_factor: int) -> None:
        self.cfg.scan_point_downsample = int(downsample_factor)
        if self.data_packet.flag_new_frame:
            return  # will be handled soon
        self.visualize_scan(self.data_packet.scan_points)

    def _on_color_edit_scan_point_color(self, color: o3d_gui.Color) -> None:
        red = color.red.real
        green = color.green.real
        blue = color.blue.real
        self.cfg.scan_point_color = [red, green, blue]
        if self.data_packet.flag_new_frame:
            return  # will be handled soon
        self.scan.paint_uniform_color([red, green, blue])
        self.visualize_scan()

    def _on_slider_traj_line_width(self, line_width: int) -> None:
        self.cfg.traj_line_width = line_width
        self.traj_render.line_width = line_width * self.window.scaling
        if self.data_packet.flag_new_frame:
            return  # will be handled soon
        self.visualize_trajectory()

    def _on_color_edit_traj_line_color(self, color: o3d_gui.Color) -> None:
        red = color.red.real
        green = color.green.real
        blue = color.blue.real
        self.cfg.traj_color = [red, green, blue]
        if self.data_packet.flag_new_frame:
            return  # will be handled soon
        self.traj.paint_uniform_color([red, green, blue])
        self.visualize_trajectory()

    def _on_slider_sample_point_size(self, point_size: int) -> None:
        self.cfg.sample_point_size = point_size
        self.sample_render.point_size = point_size * self.window.scaling
        if self.data_packet.flag_samples_updated:
            return  # will be handled soon
        if self.checkbox_show_sample_surf.checked:
            self.visualize_sample_points(self.sample_surf_name, self.sample_surf, True)
        if self.checkbox_show_sample_perturbed.checked:
            self.visualize_sample_points(self.sample_perturbed_name, self.sample_perturbed, True)
        if self.checkbox_show_sample_free.checked:
            self.visualize_sample_points(self.sample_free_name, self.sample_free, True)
        if self.checkbox_show_sample_extra.checked:
            self.visualize_sample_points(self.sample_extra_name, self.sample_extra, True)

    def _on_slider_sample_point_downsample(self, downsample_factor: int) -> None:
        self.cfg.sample_point_downsample = int(downsample_factor)
        if self.data_packet.sampled_xyz is not None:
            self.data_packet.flag_samples_updated = True  # to trigger sample point update

    def _on_checkbox_show_sample_surf(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show surface sample points.")
            if self.data_packet.sampled_xyz is not None:
                self.data_packet.flag_samples_updated = True
        else:
            tqdm.write("[GUI] Hide surface sample points.")
            self.visualize_sample_points(self.sample_surf_name, self.sample_surf, False)

    def _on_checkbox_show_sample_perturbed(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show perturbed surface sample points.")
            if self.data_packet.sampled_xyz is not None:
                self.data_packet.flag_samples_updated = True
        else:
            tqdm.write("[GUI] Hide perturbed surface sample points.")
            self.visualize_sample_points(self.sample_perturbed_name, self.sample_perturbed, False)

    def _on_checkbox_show_sample_free(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show free space sample points.")
            if self.data_packet.sampled_xyz is not None:
                self.data_packet.flag_samples_updated = True
        else:
            tqdm.write("[GUI] Hide free space sample points.")
            self.visualize_sample_points(self.sample_free_name, self.sample_free, False)

    def _on_checkbox_show_sample_extra(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show extra sample points.")
            if self.data_packet.sampled_xyz is not None:
                self.data_packet.flag_samples_updated = True
        else:
            tqdm.write("[GUI] Hide extra sample points.")
            self.visualize_sample_points(self.sample_extra_name, self.sample_extra, False)

    def _on_color_edit_sample_point_color_surf(self, color: o3d_gui.Color) -> None:
        red = color.red.real
        green = color.green.real
        blue = color.blue.real
        self.cfg.sample_color_surf = [red, green, blue]
        if self.data_packet.flag_samples_updated:
            return  # will be handled soon
        self.sample_surf.paint_uniform_color(self.cfg.sample_color_surf)
        self.visualize_sample_points(self.sample_surf_name, self.sample_surf, self.checkbox_show_sample_surf.checked)

    def _on_color_edit_sample_point_color_perturbed(self, color: o3d_gui.Color) -> None:
        red = color.red.real
        green = color.green.real
        blue = color.blue.real
        self.cfg.sample_color_perturbed = [red, green, blue]
        if self.data_packet.flag_samples_updated:
            return  # will be handled soon
        self.sample_perturbed.paint_uniform_color(self.cfg.sample_color_perturbed)
        show = self.checkbox_show_sample_perturbed.checked
        self.visualize_sample_points(self.sample_perturbed_name, self.sample_perturbed, show)

    def _on_color_edit_sample_point_color_extra(self, color: o3d_gui.Color) -> None:
        red = color.red.real
        green = color.green.real
        blue = color.blue.real
        self.cfg.sample_color_extra = [red, green, blue]
        if self.data_packet.flag_samples_updated:
            return  # will be handled soon
        self.sample_extra.paint_uniform_color(self.cfg.sample_color_extra)
        self.visualize_sample_points(self.sample_extra_name, self.sample_extra, self.checkbox_show_sample_extra.checked)

    def _on_color_edit_sample_point_color_free(self, color: o3d_gui.Color) -> None:
        red = color.red.real
        green = color.green.real
        blue = color.blue.real
        self.cfg.sample_color_free = [red, green, blue]
        if self.data_packet.flag_samples_updated:
            return  # will be handled soon
        self.sample_free.paint_uniform_color(self.cfg.sample_color_free)
        self.visualize_sample_points(self.sample_free_name, self.sample_free, self.checkbox_show_sample_free.checked)

    def _on_slider_camera_line_width(self, line_width: int) -> None:
        self.cfg.camera_line_width = line_width
        self.curr_cam_render.line_width = line_width * self.window.scaling
        self.kf_cams_render.line_width = line_width * self.window.scaling
        if self.data_packet.flag_new_frame:
            return  # will be handled soon
        self.visualize_curr_cam()
        self.visualize_kf_cams()

    def _on_slider_camera_size(self, size: float) -> None:
        r = size / self.cfg.camera_size
        self.org_cam.scale(r, center=np.array([0.0, 0.0, 0.0]))
        self.cfg.camera_size = size
        if self.data_packet.flag_new_frame:
            return  # will be handled soon
        self.visualize_curr_cam(self.data_packet.frame_pose)
        self.kf_cams.clear()
        self.visualize_kf_cams(
            self.data_packet.key_frame_indices,
            self.data_packet.selected_key_frame_indices,
        )

    def _on_color_edit_camera_color_current(self, color: o3d_gui.Color) -> None:
        red = color.red.real
        green = color.green.real
        blue = color.blue.real
        self.cfg.camera_color_current = [red, green, blue]
        if self.data_packet.flag_new_frame:
            return  # will be handled soon
        self.curr_cam.paint_uniform_color([red, green, blue])
        self.visualize_curr_cam()

    def _on_color_edit_camera_color_kf(self, color: o3d_gui.Color) -> None:
        red = color.red.real
        green = color.green.real
        blue = color.blue.real
        self.cfg.camera_color_key_frame = [red, green, blue]
        if self.data_packet.flag_new_frame:
            return  # will be handled soon
        self.visualize_kf_cams(
            self.data_packet.key_frame_indices,
            self.data_packet.selected_key_frame_indices,
        )

    def _on_color_edit_camera_color_selected_kf(self, color: o3d_gui.Color) -> None:
        red = color.red.real
        green = color.green.real
        blue = color.blue.real
        self.cfg.camera_color_selected_key_frame = [red, green, blue]
        if self.data_packet.flag_new_frame:
            return  # will be handled soon
        self.visualize_kf_cams(
            self.data_packet.key_frame_indices,
            self.data_packet.selected_key_frame_indices,
        )

    def _on_slider_octree_line_width(self, line_width: int) -> None:
        self.cfg.octree_line_width = line_width
        self.octree_render.line_width = line_width * self.window.scaling
        if self.data_packet.flag_octree_updated:
            return  # will be handled soon
        self.visualize_octree(force_update=True)

    def _on_color_edit_octree_color(self, color: o3d_gui.Color) -> None:
        red = color.red.real
        green = color.green.real
        blue = color.blue.real
        self.cfg.octree_line_color = [red, green, blue]
        if self.data_packet.flag_octree_updated:
            return  # will be handled soon
        self.octree.paint_uniform_color([red, green, blue])
        self.visualize_octree(force_update=True)

    def _on_checkbox_mesh_color_by_normal(self, is_checked: bool) -> None:
        if is_checked:
            self.cfg.mesh_color_option = "normal"
            self.mesh_render.shader = "normals"
            self.checkbox_mesh_color_by_height.checked = False
            if self.data_packet.flag_sdf_grid_updated:
                return  # will be handled soon
            if not self.checkbox_show_mesh.checked and not self.checkbox_show_mesh_by_prior.checked:
                return
            self.visualize_mesh(force_update=True)

    def _on_checkbox_mesh_color_by_height(self, is_checked: bool) -> None:
        if is_checked:
            self.cfg.mesh_color_option = "height"
            self.mesh_render.shader = "defaultLit"
            self.checkbox_mesh_color_by_normal.checked = False
            if self.data_packet.flag_sdf_grid_updated:
                return  # will be handled soon
            if not self.checkbox_show_mesh.checked and not self.checkbox_show_mesh_by_prior.checked:
                return
            if self.checkbox_show_mesh.checked:
                self.colorize_mesh(self.mesh)
            if self.checkbox_show_mesh_by_prior.checked:
                self.colorize_mesh(self.mesh_prior)
            self.visualize_mesh(force_update=True)

    def _on_combobox_mesh_height_color_map(self, color_map_name: str, index: int) -> None:
        tqdm.write(f"[GUI] Set mesh height color map to {color_map_name}.")
        self.cfg.mesh_height_color_map = color_map_name
        if self.data_packet.flag_sdf_grid_updated:
            return  # will be handled soon
        if not self.checkbox_show_mesh.checked and not self.checkbox_show_mesh_by_prior.checked:
            return
        if self.checkbox_mesh_color_by_height.checked:
            if self.checkbox_show_mesh.checked:
                self.colorize_mesh(self.mesh)
            if self.checkbox_show_mesh_by_prior.checked:
                self.colorize_mesh(self.mesh_prior)
            self.visualize_mesh(force_update=True)

    def _on_slider_sdf_point_size(self, point_size: float) -> None:
        tqdm.write(f"[GUI] Set SDF point size to {point_size}.")
        self.cfg.sdf_point_size = point_size
        self.sdf_render.point_size = point_size * self.window.scaling
        if self.data_packet.flag_sdf_slice_updated:
            return  # will be handled soon
        self.visualize_sdf_slice(self.sdf_name, self.sdf_slice, self.checkbox_show_sdf.checked)
        self.visualize_sdf_slice(self.sdf_prior_name, self.sdf_slice_prior, self.checkbox_show_sdf_prior.checked)
        self.visualize_sdf_slice(self.sdf_residual_name, self.sdf_slice_res, self.checkbox_show_sdf_res.checked)

    def _on_combobox_sdf_axis(self, axis_name: str, axis: int) -> None:
        tqdm.write(f"[GUI] Set SDF slice axis to {axis_name}.")
        self.cfg.sdf_slice_axis = axis
        position = self.slider_sdf_position.double_value
        if self.cfg.scene_bound_min is not None and self.cfg.scene_bound_max is not None:
            self.slider_sdf_position.set_limits(self.cfg.scene_bound_min[axis], self.cfg.scene_bound_max[axis])
            if position < self.cfg.scene_bound_min[axis]:
                position = self.cfg.scene_bound_min[axis]
            if position > self.cfg.scene_bound_max[axis]:
                position = self.cfg.scene_bound_max[axis]
        self.slider_sdf_position.double_value = position
        self._on_slider_sdf_position(position)

    def _on_slider_sdf_position(self, position: float) -> None:
        self.cfg.sdf_slice_position = position
        tqdm.write(f"[GUI] Set SDF slice position to {position:.3f} m.")

    def _on_combobox_sdf_color_map(self, color_map_name: str, index: int) -> None:
        tqdm.write(f"[GUI] Set SDF color map to {color_map_name}.")
        self.cfg.sdf_color_map = color_map_name
        if self.sdf_slice_to_save is None:
            return
        if self.data_packet.flag_sdf_slice_updated:
            return  # will be handled soon

        if self.checkbox_show_sdf.checked:
            self.visualize_sdf_slice(
                self.sdf_name,
                self.sdf_slice,
                self.checkbox_show_sdf.checked,
                self.sdf_slice_to_save["bounds"],
                self.sdf_slice_to_save["axis"],
                self.sdf_slice_to_save["position"],
                self.sdf_slice_to_save["resolution"],
                self.sdf_slice_to_save["sdf_slice"]["sdf"],
            )

        if self.checkbox_show_sdf_prior.checked:
            self.visualize_sdf_slice(
                self.sdf_prior_name,
                self.sdf_slice_prior,
                self.checkbox_show_sdf_prior.checked,
                self.sdf_slice_to_save["bounds"],
                self.sdf_slice_to_save["axis"],
                self.sdf_slice_to_save["position"],
                self.sdf_slice_to_save["resolution"],
                self.sdf_slice_to_save["sdf_slice"]["sdf_prior"],
            )

        if self.checkbox_show_sdf_res.checked:
            self.visualize_sdf_slice(
                self.sdf_residual_name,
                self.sdf_slice_res,
                self.checkbox_show_sdf_res.checked,
                self.sdf_slice_to_save["bounds"],
                self.sdf_slice_to_save["axis"],
                self.sdf_slice_to_save["position"],
                self.sdf_slice_to_save["resolution"],
                self.sdf_slice_to_save["sdf_slice"]["sdf_residual"],
            )

    def _on_button_save_mesh(self) -> None:
        if self.mesh.is_empty():
            tqdm.write("[GUI] Warning: mesh is empty, not saving.")
            return
        dialog = o3d_gui.FileDialog(o3d_gui.FileDialog.SAVE, "Save Mesh As", self.window.theme)
        dialog.set_path(os.path.join(os.curdir, "mesh.ply"))

        def on_done(path: str) -> None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            o3d.io.write_triangle_mesh(path, self.mesh)
            tqdm.write(f"[GUI] Mesh saved to {path}.")
            self.window.close_dialog()

        dialog.set_on_done(on_done)
        dialog.set_on_cancel(lambda: self.window.close_dialog())
        self.window.show_dialog(dialog)

    def _on_button_save_sdf(self) -> None:
        if self.sdf_slice_to_save is None:
            tqdm.write("[GUI] Warning: no SDF slice to save.")
            return
        dialog = o3d_gui.FileDialog(o3d_gui.FileDialog.SAVE, "Save SDF Slice At", self.window.theme)
        dialog.set_path(os.curdir)

        def on_done(folder_path: str) -> None:
            if self.sdf_slice_to_save is None:
                return

            os.makedirs(folder_path, exist_ok=True)

            slice_configs = [
                {
                    "axis_name": "x",
                    "xlabel": "y (m)",
                    "ylabel": "z (m)",
                },
                {
                    "axis_name": "y",
                    "xlabel": "x (m)",
                    "ylabel": "z (m)",
                },
                {
                    "axis_name": "z",
                    "xlabel": "x (m)",
                    "ylabel": "y (m)",
                },
            ]
            fontsize = 12

            bound_min, bound_max = self.sdf_slice_to_save["slice_bound"]  # type: ignore
            sdf_slice = self.sdf_slice_to_save["sdf_slice"]
            axis: int = self.sdf_slice_to_save["axis"]  # type: ignore
            axis_name: str = slice_configs[axis]["axis_name"]
            pos: float = self.sdf_slice_to_save["pos"]  # type: ignore
            for slice_name in ["sdf_prior", "sdf_residual", "sdf"]:
                slice_values = sdf_slice[slice_name]  # type: ignore
                plt.figure()
                im = plt.imshow(
                    slice_values,
                    extent=(bound_min[0], bound_max[0], bound_min[1], bound_max[1]),
                    origin="lower",
                    cmap="jet",
                )
                plt.colorbar(im, shrink=0.8)
                plt.xlabel(slice_configs[axis]["xlabel"], fontsize=fontsize)
                plt.ylabel(slice_configs[axis]["ylabel"], fontsize=fontsize)
                plt.title(f"At {axis_name} = {pos:.2f} m", fontsize=fontsize)
                plt.tight_layout()
                img_path = os.path.join(folder_path, f"slice_{axis_name}_{slice_name}.png")
                plt.savefig(img_path, dpi=300)
                plt.close()

            tqdm.write(f"[GUI] SDF slice saved to {folder_path}.")
            self.window.close_dialog()

        dialog.set_on_done(on_done)
        dialog.set_on_cancel(lambda: self.window.close_dialog())
        self.window.show_dialog(dialog)

    def _on_button_save_model(self) -> None:
        dialog = o3d_gui.FileDialog(o3d_gui.FileDialog.SAVE, "Save Model As", self.window.theme)

        def on_done(path: str):
            self.queue_out.put_nowait(GuiControlPacket(save_model_to_path=path))
            self.window.close_dialog()

        dialog.set_on_done(on_done)
        dialog.set_on_cancel(lambda: self.window.close_dialog())
        dialog.add_filter(".pth", "PyTorch Model (.pth)")
        self.window.show_dialog(dialog)

    def _on_button_save_screenshot(self) -> None:
        dialog = o3d_gui.FileDialog(o3d_gui.FileDialog.SAVE, "Save Screenshot As", self.window.theme)
        dialog.set_path(os.path.join(os.curdir, "screenshot.png"))

        def on_done(path: str) -> None:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            height = self.window.size.height
            width = self.widget3d_width
            app = o3d.visualization.gui.Application.instance
            img = np.asarray(app.render_to_image(self.widget3d.scene, width, height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path, img)
            tqdm.write(f"[GUI] Screenshot saved to {path}.")
            self.window.close_dialog()

        dialog.set_on_done(on_done)
        dialog.set_on_cancel(lambda: self.window.close_dialog())
        self.window.show_dialog(dialog)

    def _on_button_save_gui_config(self) -> None:
        dialog = o3d_gui.FileDialog(o3d_gui.FileDialog.SAVE, "Save GUI Config As", self.window.theme)
        dialog.set_path(os.path.join(os.curdir, "gui_config.yaml"))

        def on_done(path: str) -> None:
            path = path.strip()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._update_checked_objects()
            self.cfg.objects = list(self.checked_objects)
            self.cfg.as_yaml(path)
            tqdm.write(f"[GUI] GUI config saved to {path}.")
            self.window.close_dialog()

        dialog.set_on_done(on_done)
        dialog.set_on_cancel(lambda: self.window.close_dialog())
        self.window.show_dialog(dialog)

    def _init_video_writer(self, path: str) -> bool:
        """Initialize video writer with proper settings"""
        if self.video_queue_in is None or self.video_queue_out is None:
            return
        try:
            path = path.strip()
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)

            self.video_queue_in.put_nowait(
                VideoFrame(
                    flag_start=True,
                    path=path,
                    fps=self.cfg.video_fps,
                    frame_size=(self.widget3d_width, self.widget3d_height),
                )
            )
            ack = self.video_queue_out.get()
            if ack is None:
                tqdm.write("[GUI] Error: Video writer thread failed to start.")
                self.button_start_video.enabled = True
                self.button_stop_video.enabled = False
            else:
                tqdm.write(f"[GUI] Video writer initialized: {path}")
                self.button_start_video.enabled = False
                self.button_stop_video.enabled = True
        except Exception as e:
            tqdm.write(f"[GUI] Error initializing video writer: {e}")

    def _capture_frame_to_video(self) -> None:
        """Capture current frame and write to video"""
        if self.video_queue_in is None or self.video_queue_out is None:
            return
        if self.button_start_video.enabled:
            # Video recording is not active
            return
        try:
            height = self.window.size.height
            width = self.widget3d_width
            app = o3d.visualization.gui.Application.instance
            img = np.asarray(app.render_to_image(self.widget3d.scene, width, height))

            self.video_queue_in.put_nowait(VideoFrame(frame=img))
        except Exception as e:
            tqdm.write(f"[GUI] Error capturing frame to video: {e}")

    def _close_video_writer(self) -> None:
        """Release video writer resources"""
        if self.video_queue_in is None or self.video_queue_out is None:
            return
        try:
            self.video_queue_in.put_nowait(VideoFrame(flag_end=True))
            # don't wait for ack here, as it may block the GUI
        except Exception as e:
            tqdm.write(f"[GUI] Error closing video writer: {e}")

    def _on_button_start_video_recording(self) -> None:
        dialog = o3d_gui.FileDialog(o3d_gui.FileDialog.SAVE, "Save Video As", self.window.theme)
        dialog.set_path(os.path.join(os.curdir, "video.mp4"))

        def on_done(path: str) -> None:
            self.window.close_dialog()
            self._init_video_writer(path)

        dialog.set_on_done(on_done)
        dialog.set_on_cancel(lambda: self.window.close_dialog())
        self.window.show_dialog(dialog)

    def _on_button_stop_video_recording(self) -> None:
        self._close_video_writer()
        self.button_start_video.enabled = True
        self.button_stop_video.enabled = False

    def visualize_scan(self, points: np.ndarray = None):
        if points is not None:
            points = points[:: self.cfg.scan_point_downsample].copy()  # copy to make it contiguous
            self.scan.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            self.scan.paint_uniform_color(np.array(self.cfg.scan_point_color, dtype=np.float64))

        if not self.switch_vis.is_on or len(self.scan.points) == 0:
            return

        if self.checkbox_show_scan.checked:
            if self.widget3d.scene.has_geometry(self.scan_name):
                self.widget3d.scene.remove_geometry(self.scan_name)
            self.widget3d.scene.add_geometry(self.scan_name, self.scan, self.scan_render)

        self.widget3d.scene.show_geometry(self.scan_name, self.checkbox_show_scan.checked)

    def visualize_trajectory(self):
        if len(self.traj.points) < len(self.frame_poses):
            points = np.stack([pose[:3, 3] for pose in self.frame_poses], axis=0)
            self.traj.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            if len(points) > 1:
                lines = np.array([[i, i + 1] for i in range(len(points) - 1)], dtype=np.int32)
                self.traj.lines = o3d.utility.Vector2iVector(lines)
                self.traj.paint_uniform_color(self.cfg.traj_color)
                self.traj_length = np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))

        if not self.switch_vis.is_on or len(self.traj.lines) == 0 or self.traj_length < 0.01:
            return

        if self.checkbox_show_traj.checked:
            if self.widget3d.scene.has_geometry(self.traj_name):
                self.widget3d.scene.remove_geometry(self.traj_name)
            self.widget3d.scene.add_geometry(self.traj_name, self.traj, self.traj_render)

        self.widget3d.scene.show_geometry(self.traj_name, self.checkbox_show_traj.checked)

    def visualize_sample_points(
        self,
        name: str,
        pcd: o3d.geometry.PointCloud,
        show: bool,
        points: np.ndarray = None,
        color: list = None,
    ):
        if points is not None:
            assert color is not None
            points = points.reshape(-1, 3)[:: self.cfg.sample_point_downsample].copy()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pcd.paint_uniform_color(np.array(color, dtype=np.float64))

        if not self.switch_vis.is_on or len(pcd.points) == 0:
            return

        if show:
            if self.widget3d.scene.has_geometry(name):
                self.widget3d.scene.remove_geometry(name)
            self.widget3d.scene.add_geometry(name, pcd, self.sample_render)

        self.widget3d.scene.show_geometry(name, show)

    def visualize_kf_cams(self, key_frame_indices: list = None, selected_key_frame_indices: list = None):
        if key_frame_indices is not None and selected_key_frame_indices is not None:
            n = len(key_frame_indices) * len(self.org_cam.points)
            if len(self.kf_cams.points) < n:  # need to add more cameras
                existing_kf_count = len(self.kf_cams.points) // len(self.org_cam.points)
                for idx in key_frame_indices[existing_kf_count:]:
                    pose = self.frame_poses[idx]
                    points = np.asarray(self.org_cam.points)
                    points = points @ pose[:3, :3].T + pose[:3, [3]].T
                    start_idx = len(self.kf_cams.points)
                    self.kf_cams.points.extend(o3d.utility.Vector3dVector(points.astype(np.float64)))
                    self.kf_cams.lines.extend(np.asarray(self.org_cam.lines) + start_idx)
            self.kf_cams.paint_uniform_color(np.array(self.cfg.camera_color_key_frame, dtype=np.float64))
            for idx in selected_key_frame_indices:
                m = len(self.org_cam.lines)
                for i in range(m * idx, m * (idx + 1)):
                    self.kf_cams.colors[i] = np.array(self.cfg.camera_color_selected_key_frame, dtype=np.float64)

        if not self.switch_vis.is_on or len(self.kf_cams.lines) == 0:
            return

        if self.checkbox_show_kf_cams.checked:
            if self.widget3d.scene.has_geometry(self.kf_cams_name):
                self.widget3d.scene.remove_geometry(self.kf_cams_name)
            self.widget3d.scene.add_geometry(self.kf_cams_name, self.kf_cams, self.kf_cams_render)

        self.widget3d.scene.show_geometry(self.kf_cams_name, self.checkbox_show_kf_cams.checked)

    def visualize_curr_cam(self, pose: np.ndarray = None):
        if pose is not None:  # update visualization only when new pose is given
            points = np.asarray(self.org_cam.points)
            points = points @ pose[:3, :3].T + pose[:3, [3]].T
            self.curr_cam.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            if len(self.curr_cam.lines) == 0:
                self.curr_cam.lines = self.org_cam.lines
                self.curr_cam.paint_uniform_color(np.array(self.cfg.camera_color_current, dtype=np.float64))
            points = np.asarray(self.org_cam_axis.vertices)
            points = points @ pose[:3, :3].T + pose[:3, [3]].T
            self.curr_cam_axis.vertices = o3d.utility.Vector3dVector(points.astype(np.float64))

        if not self.switch_vis.is_on or len(self.curr_cam.lines) == 0:
            return

        if self.checkbox_show_curr_cam.checked:
            if self.widget3d.scene.has_geometry(self.curr_cam_name):
                self.widget3d.scene.remove_geometry(self.curr_cam_name)
                self.widget3d.scene.remove_geometry(self.curr_cam_axis_name)
            self.widget3d.scene.add_geometry(self.curr_cam_name, self.curr_cam, self.curr_cam_render)
            self.widget3d.scene.add_geometry(self.curr_cam_axis_name, self.curr_cam_axis, self.curr_cam_axis_render)

        self.widget3d.scene.show_geometry(self.curr_cam_name, self.checkbox_show_curr_cam.checked)
        self.widget3d.scene.show_geometry(self.curr_cam_axis_name, self.checkbox_show_curr_cam.checked)

    def visualize_octree(self, force_update: bool = False):
        processed_octrees = 0
        max_octrees_per_call = 2  # Limit processing to prevent blocking

        while processed_octrees < max_octrees_per_call and not self.octree_queue_out.empty():
            try:
                octree_data: OctreeLinesetResult = self.octree_queue_out.get_nowait()

                self.octree.points = o3d.utility.Vector3dVector(octree_data.vertices)
                self.octree.lines = o3d.utility.Vector2iVector(octree_data.lines)
                self.octree.paint_uniform_color(np.array(self.cfg.octree_line_color, dtype=np.float64))

                processed_octrees += 1
            except queue.Empty:
                break
            except Exception as e:
                tqdm.write(f"[GUI] Warning: failed to get octree from queue: {e}")
                break

        if not self.switch_vis.is_on or self.octree.is_empty():
            return

        if self.checkbox_show_octree.checked:
            if self.widget3d.scene.has_geometry(self.octree_name) and (processed_octrees > 0 or force_update):
                # already added but updated
                self.widget3d.scene.remove_geometry(self.octree_name)
            if not self.widget3d.scene.has_geometry(self.octree_name):
                self.widget3d.scene.add_geometry(self.octree_name, self.octree, self.octree_render)

        self.widget3d.scene.show_geometry(self.octree_name, self.checkbox_show_octree.checked)

    def visualize_sdf_slice(
        self,
        sdf_name: str,
        sdf_slice: o3d.geometry.PointCloud,
        show: bool,
        bounds: list = None,
        axis: int = None,
        pos: float = None,
        resolution: float = None,
        sdf_values: np.ndarray = None,
    ):
        if (
            bounds is not None
            and axis is not None
            and pos is not None
            and resolution is not None
            and sdf_values is not None
        ):
            if axis == 0:
                # bounds = (bounds_min, bounds_max)
                y, z = np.meshgrid(
                    np.arange(bounds[0][0], bounds[1][0], resolution),
                    np.arange(bounds[0][1], bounds[1][1], resolution),
                    indexing="xy",
                )
                points = np.stack((np.full_like(y, pos), y, z), axis=-1).astype(np.float64)
            elif axis == 1:
                x, z = np.meshgrid(
                    np.arange(bounds[0][0], bounds[1][0], resolution),
                    np.arange(bounds[0][2], bounds[1][2], resolution),
                    indexing="xy",
                )
                points = np.stack((x, np.full_like(x, pos), z), axis=-1).astype(np.float64)
            else:  # axis == 2
                x, y = np.meshgrid(
                    np.arange(bounds[0][0], bounds[1][0], resolution),
                    np.arange(bounds[0][1], bounds[1][1], resolution),
                    indexing="xy",
                )
                points = np.stack((x, y, np.full_like(x, pos)), axis=-1).astype(np.float64)

            color_map = cm.get_cmap(self.cfg.sdf_color_map)
            sdf_min = np.min(sdf_values).item()
            sdf_max = np.max(sdf_values).item()
            sdf_values = np.clip((sdf_values - sdf_min) / (sdf_max - sdf_min + 1e-6), 0, 1)
            colors = color_map(sdf_values.flatten())[:, :3]
            sdf_slice.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
            sdf_slice.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        if not self.switch_vis.is_on or len(sdf_slice.points) == 0:
            return

        if show:
            if self.widget3d.scene.has_geometry(sdf_name):
                self.widget3d.scene.remove_geometry(sdf_name)
            self.widget3d.scene.add_geometry(sdf_name, sdf_slice, self.sdf_render)

        self.widget3d.scene.show_geometry(sdf_name, show)

    def colorize_mesh(self, mesh: o3d.geometry.TriangleMesh):
        if mesh.is_empty():
            return
        if self.checkbox_mesh_color_by_height.checked:
            z_values = np.asarray(mesh.vertices)[:, 2]
            z_min, z_max = np.min(z_values).item(), np.max(z_values).item()
            z_normalized = np.clip((z_values - z_min) / (z_max - z_min + 1e-6), 0.0, 1.0)
            color_map = cm.get_cmap(self.cfg.mesh_height_color_map)
            colors = color_map(z_normalized)[:, :3].astype(np.float64)
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        else:
            mesh.vertex_colors.clear()

    def remove_ceiling_from_mesh(self, mesh: o3d.geometry.TriangleMesh):
        z_max = mesh.get_max_bound()[2]
        ceiling_thickness = self.cfg.mesh_ceiling_thickness
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        face_z_max = np.max(vertices[faces][:, :, 2], axis=1)
        mask = face_z_max >= z_max - ceiling_thickness
        mesh.remove_triangles_by_index(np.where(mask)[0].tolist())

    def visualize_mesh(self, force_update: bool = False):
        processed_meshes = 0
        processed_prior_meshes = 0
        max_meshes_per_call = 5  # Limit processing to prevent blocking

        while processed_meshes < max_meshes_per_call and not self.mc_queue_out.empty():
            try:
                mesh_data: MarchingCubesResult = self.mc_queue_out.get_nowait()
                processed_meshes += 1

                if mesh_data.name == "sdf":
                    self.mesh.vertices = o3d.utility.Vector3dVector(mesh_data.vertices)
                    self.mesh.triangles = o3d.utility.Vector3iVector(mesh_data.triangles)
                    self.mesh.triangle_normals = o3d.utility.Vector3dVector(mesh_data.triangle_normals)
                    self.mesh.remove_degenerate_triangles()
                    if self.cfg.mesh_remove_ceiling:
                        self.remove_ceiling_from_mesh(self.mesh)
                    self.colorize_mesh(self.mesh)
                elif mesh_data.name == "sdf_prior":
                    self.mesh_prior.vertices = o3d.utility.Vector3dVector(mesh_data.vertices)
                    self.mesh_prior.triangles = o3d.utility.Vector3iVector(mesh_data.triangles)
                    self.mesh_prior.triangle_normals = o3d.utility.Vector3dVector(mesh_data.triangle_normals)
                    self.mesh_prior.remove_degenerate_triangles()
                    if self.cfg.mesh_remove_ceiling:
                        self.remove_ceiling_from_mesh(self.mesh_prior)
                    self.colorize_mesh(self.mesh_prior)
                    processed_prior_meshes += 1
            except queue.Empty:
                break
            except Exception as e:
                tqdm.write(f"[GUI] Error processing marching cubes result: {e}")
                break

        if not self.switch_vis.is_on:
            return

        processed_meshes -= processed_prior_meshes

        if not self.mesh.is_empty():
            if self.checkbox_show_mesh.checked:
                if self.widget3d.scene.has_geometry(self.mesh_name) and (processed_meshes > 0 or force_update):
                    self.widget3d.scene.remove_geometry(self.mesh_name)
                if not self.widget3d.scene.has_geometry(self.mesh_name):
                    self.widget3d.scene.add_geometry(self.mesh_name, self.mesh, self.mesh_render)
            self.widget3d.scene.show_geometry(self.mesh_name, self.checkbox_show_mesh.checked)

        if not self.mesh_prior.is_empty():
            if self.checkbox_show_mesh_by_prior.checked:
                if self.widget3d.scene.has_geometry(self.mesh_prior_name) and (
                    processed_prior_meshes > 0 or force_update
                ):
                    self.widget3d.scene.remove_geometry(self.mesh_prior_name)
                if not self.widget3d.scene.has_geometry(self.mesh_prior_name):
                    self.widget3d.scene.add_geometry(self.mesh_prior_name, self.mesh_prior, self.mesh_render)
            self.widget3d.scene.show_geometry(self.mesh_prior_name, self.checkbox_show_mesh_by_prior.checked)

    def visualize_gt_mesh(self):
        if self.gt_mesh is None:
            return

        if not self.switch_vis.is_on or self.gt_mesh.is_empty():
            return

        if self.checkbox_show_gt_mesh.checked:
            if self.widget3d.scene.has_geometry(self.gt_mesh_name):
                self.widget3d.scene.remove_geometry(self.gt_mesh_name)
            self.widget3d.scene.add_geometry(self.gt_mesh_name, self.gt_mesh, self.gt_mesh_render)
        self.widget3d.scene.show_geometry(self.gt_mesh_name, self.checkbox_show_gt_mesh.checked)

    def set_camera(self):
        if self.checkbox_view_follow.checked:
            self.center_bev()
        elif self.checkbox_view_keyboard.checked:
            pass  # do nothing, user controls the view
        elif self.checkbox_view_from_file.checked and self.cfg.view_file is not None and not self.view_file_loaded:
            assert os.path.exists(self.cfg.view_file), f"View file {self.cfg.view_file} does not exist."
            self.set_view_from_file(self.cfg.view_file)
            self.view_file_loaded = True

    def center_bev(self):
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(90, bounds, bounds.get_center())  # type: ignore

    def set_view_from_file(self, path: str):
        if not os.path.isfile(path):
            return
        # in the file, we have the projection matrix (4, 4),
        with open(path, "r") as f:
            view = yaml.safe_load(f)
        intrinsics = np.array(view["intrinsics"])
        extrinsics = np.array(view["extrinsics"])
        bounds = deepcopy(self.widget3d.scene.bounding_box)
        bounds = bounds.scale(1.2, bounds.get_center())
        self.widget3d.setup_camera(intrinsics, extrinsics, self.widget3d_width, self.widget3d_height, bounds)

    @classmethod
    def run(cls, *args, **kwargs):
        app = o3d_gui.Application.instance
        app.initialize()
        window = cls(*args, **kwargs)
        app.run()
        window.cleanup()
        app.quit()

    def send_control_packet(self):
        if self.queue_out is None:
            return
        try:
            packet = GuiControlPacket()
            packet.flag_mapping_run = self.switch_run.is_on
            if self.switch_vis.is_on and (
                self.checkbox_show_sample_surf.checked
                or self.checkbox_show_sample_perturbed.checked
                or self.checkbox_show_sample_free.checked
                or self.checkbox_show_sample_extra.checked
            ):
                packet.sample_update_frequency = self.slider_sample_update_freq.int_value

            if self.switch_vis.is_on and self.checkbox_show_octree.checked:
                packet.octree_update_frequency = self.slider_octree_update_freq.int_value
                packet.octree_min_size = self.slider_octree_min_size.int_value
            if self.switch_vis.is_on and (
                self.checkbox_show_sdf.checked
                or self.checkbox_show_sdf_prior.checked
                or self.checkbox_show_sdf_res.checked
            ):
                packet.sdf_slice_frequency = self.slider_sdf_update_freq.int_value
                packet.sdf_slice_axis = self.combobox_sdf_axis.selected_index
                packet.sdf_slice_position = self.slider_sdf_position.double_value
                packet.sdf_slice_resolution = 1.0 / self.slider_sdf_resolution.int_value
            if self.switch_vis.is_on and (self.checkbox_show_mesh.checked or self.checkbox_show_mesh_by_prior.checked):
                packet.sdf_grid_frequency = self.slider_mesh_update_freq.int_value
                packet.sdf_grid_resolution = 1.0 / self.slider_mesh_resolution.int_value
                packet.sdf_grid_ignore_large_voxels = self.cfg.mesh_clean
            self.queue_out.put_nowait(packet)
        except Exception as e:
            tqdm.write(f"[GUI] Error sending control packet: {e}")

    def receive_data_packet(self, get_latest: bool = True):
        if self.queue_in is None:
            return None
        packet: Optional[GuiDataPacket] = None
        processed_packets = 0
        max_packets_per_call = 10  # Limit processing to prevent blocking

        while processed_packets < max_packets_per_call and not self.queue_in.empty():
            try:
                packet = self.queue_in.get_nowait()
                processed_packets += 1

                if packet.flag_exit:
                    tqdm.write("[GUI] Received exit signal. Closing GUI...")
                    self.data_packet.flag_exit = True
                    self._send_flag_gui_closed()
                    break

                if packet.mapping_end:
                    self.data_packet.mapping_end = True

                if packet.flag_new_frame:
                    self.data_packet.num_iterations = packet.num_iterations
                    self.data_packet.flag_new_frame = True
                    self.data_packet.frame_idx = packet.frame_idx
                    self.data_packet.frame_pose = packet.frame_pose
                    self.data_packet.scan_points = packet.scan_points
                    self.data_packet.key_frame_indices = packet.key_frame_indices
                    self.data_packet.selected_key_frame_indices = packet.selected_key_frame_indices

                    self.frame_indices.append(packet.frame_idx)
                    self.frame_poses.append(packet.frame_pose)

                if packet.flag_samples_updated:
                    self.data_packet.flag_samples_updated = True
                    self.data_packet.sampled_xyz = packet.sampled_xyz
                    self.data_packet.extra_surface_pcd = packet.extra_surface_pcd
                    self.data_packet.n_free = packet.n_free
                    self.data_packet.n_perturbed = packet.n_perturbed
                    self.data_packet.n_surf = packet.n_surf

                if packet.flag_octree_updated:
                    self.data_packet.flag_octree_updated = True
                    self.data_packet.octree_voxel_centers = packet.octree_voxel_centers
                    self.data_packet.octree_voxel_sizes = packet.octree_voxel_sizes
                    self.data_packet.octree_vertices = packet.octree_vertices
                    self.data_packet.octree_little_endian_vertex_order = packet.octree_little_endian_vertex_order
                    self.data_packet.octree_resolution = packet.octree_resolution

                if packet.flag_sdf_slice_updated:
                    self.data_packet.flag_sdf_slice_updated = True
                    self.data_packet.sdf_slice_bounds = packet.sdf_slice_bounds
                    self.data_packet.sdf_slice_axis = packet.sdf_slice_axis
                    self.data_packet.sdf_slice_position = packet.sdf_slice_position
                    self.data_packet.sdf_slice_resolution = packet.sdf_slice_resolution
                    self.data_packet.sdf_slice = packet.sdf_slice

                if packet.flag_sdf_grid_updated:
                    self.data_packet.flag_sdf_grid_updated = True
                    self.data_packet.sdf_grid_bounds = packet.sdf_grid_bounds
                    self.data_packet.sdf_grid_resolution = packet.sdf_grid_resolution
                    self.data_packet.sdf_grid_mask = packet.sdf_grid_mask
                    self.data_packet.sdf_grid_shape = packet.sdf_grid_shape
                    self.data_packet.sdf_grid = packet.sdf_grid

                if packet.model_saved_path is not None:
                    self.data_packet.model_saved_path = packet.model_saved_path
                if packet.time_stats is not None:
                    self.data_packet.time_stats = packet.time_stats
                if packet.loss_stats is not None:
                    self.data_packet.loss_stats = packet.loss_stats
                if packet.gpu_mem_usage is not None:
                    self.data_packet.gpu_mem_usage = packet.gpu_mem_usage

                if not get_latest:
                    break
            except queue.Empty:
                break
            except Exception as e:
                tqdm.write(f"[GUI] Error processing queue packet: {e}")
                break
        return packet

    def update_wrapper(self, data_packet: Optional[GuiDataPacket] = None):
        with self.timer_gui_update:
            self.update(data_packet)

    def update(self, data_packet: Optional[GuiDataPacket] = None):
        if self.data_packet.flag_exit:
            tqdm.write("[GUI] Received exit signal. Closing GUI...")
            self._send_flag_gui_closed()
            return

        if data_packet is not None:
            # we received some new data
            # multiple data packets may be merged into one and stored in self.data_packet
            if self.data_packet.mapping_end:
                self.sleep_interval = 0.1  # slow down

                # auto-end video recording if a video is being recorded (the process will check.)
                if self.cfg.video_auto_end:
                    self._close_video_writer()
                    self.button_start_video.enabled = True
                    self.button_stop_video.enabled = False

            if self.data_packet.flag_new_frame:
                self.data_packet.flag_new_frame = False

                assert self.data_packet.num_iterations >= 0
                assert self.data_packet.frame_idx >= 0
                assert self.data_packet.frame_pose is not None
                assert self.data_packet.scan_points is not None
                assert self.data_packet.key_frame_indices is not None
                assert self.data_packet.selected_key_frame_indices is not None

                self.visualize_scan(self.data_packet.scan_points)  # update scan
                self.visualize_trajectory()  # update trajectory
                self.visualize_curr_cam(self.data_packet.frame_pose)  # update current camera
                self.visualize_kf_cams(
                    self.data_packet.key_frame_indices,
                    self.data_packet.selected_key_frame_indices,
                )

                self.label_info_num_iterations.text = f"Num Iterations: {self.data_packet.num_iterations}"
                self.label_info_num_key_frames.text = f"Num Key Frames: {len(self.data_packet.key_frame_indices)}"
                self.label_info_frame_idx.text = f"Frame: {self.data_packet.frame_idx}"
                self.label_info_traj_length.text = f"Travel Distance: {self.traj_length:.3f} m"

            if self.data_packet.flag_samples_updated:
                self.data_packet.flag_samples_updated = False

                n_free = self.data_packet.n_free
                n_perturbed = self.data_packet.n_perturbed
                n_surf = self.data_packet.n_surf

                assert self.data_packet.sampled_xyz is not None
                assert self.data_packet.sampled_xyz.shape[1] == n_free + n_perturbed + n_surf

                if n_surf > 0:
                    self.visualize_sample_points(
                        self.sample_surf_name,
                        self.sample_surf,
                        self.checkbox_show_sample_surf.checked,
                        self.data_packet.sampled_xyz[:, -n_surf:],
                        self.cfg.sample_color_surf,
                    )
                if n_perturbed > 0:
                    self.visualize_sample_points(
                        self.sample_perturbed_name,
                        self.sample_perturbed,
                        self.checkbox_show_sample_perturbed.checked,
                        self.data_packet.sampled_xyz[:, n_free : n_free + n_perturbed],
                        self.cfg.sample_color_perturbed,
                    )
                if n_free > 0:
                    self.visualize_sample_points(
                        self.sample_free_name,
                        self.sample_free,
                        self.checkbox_show_sample_free.checked,
                        self.data_packet.sampled_xyz[:, :n_free],
                        self.cfg.sample_color_free,
                    )
                if self.data_packet.extra_surface_pcd is not None:
                    self.visualize_sample_points(
                        self.sample_extra_name,
                        self.sample_extra,
                        self.checkbox_show_sample_extra.checked,
                        self.data_packet.extra_surface_pcd,
                        self.cfg.sample_color_extra,
                    )

            if self.data_packet.flag_octree_updated:
                self.data_packet.flag_octree_updated = False

                assert self.data_packet.octree_voxel_centers is not None
                assert self.data_packet.octree_voxel_sizes is not None
                assert self.data_packet.octree_vertices is not None
                assert self.data_packet.octree_resolution is not None

                self.octree_queue_in.put_nowait(
                    OctreeLinesetTask(
                        voxel_centers=self.data_packet.octree_voxel_centers,
                        voxel_sizes=self.data_packet.octree_voxel_sizes,
                        voxel_vertices=self.data_packet.octree_vertices,
                        octree_resolution=self.data_packet.octree_resolution,
                        little_endian_vertex_order=self.data_packet.octree_little_endian_vertex_order,
                        scene_bound_min=self.cfg.scene_bound_min,
                        scene_bound_max=self.cfg.scene_bound_max,
                    )
                )

            self.visualize_octree()

            # update sdf slice
            if self.data_packet.flag_sdf_slice_updated:
                self.data_packet.flag_sdf_slice_updated = False

                assert self.data_packet.sdf_slice_bounds is not None
                assert self.data_packet.sdf_slice_axis is not None
                assert self.data_packet.sdf_slice_position is not None
                assert self.data_packet.sdf_slice_resolution is not None
                assert self.data_packet.sdf_slice is not None

                self.sdf_slice_to_save = dict(
                    bounds=self.data_packet.sdf_slice_bounds,
                    axis=self.data_packet.sdf_slice_axis,
                    position=self.data_packet.sdf_slice_position,
                    resolution=self.data_packet.sdf_slice_resolution,
                    sdf_slice=self.data_packet.sdf_slice,
                )

                self.visualize_sdf_slice(
                    self.sdf_name,
                    self.sdf_slice,
                    self.checkbox_show_sdf.checked,
                    self.data_packet.sdf_slice_bounds,
                    self.data_packet.sdf_slice_axis,
                    self.data_packet.sdf_slice_position,
                    self.data_packet.sdf_slice_resolution,
                    self.data_packet.sdf_slice["sdf"],
                )
                self.visualize_sdf_slice(
                    self.sdf_prior_name,
                    self.sdf_slice_prior,
                    self.checkbox_show_sdf_prior.checked,
                    self.data_packet.sdf_slice_bounds,
                    self.data_packet.sdf_slice_axis,
                    self.data_packet.sdf_slice_position,
                    self.data_packet.sdf_slice_resolution,
                    self.data_packet.sdf_slice["sdf_prior"],
                )
                self.visualize_sdf_slice(
                    self.sdf_residual_name,
                    self.sdf_slice_res,
                    self.checkbox_show_sdf_res.checked,
                    self.data_packet.sdf_slice_bounds,
                    self.data_packet.sdf_slice_axis,
                    self.data_packet.sdf_slice_position,
                    self.data_packet.sdf_slice_resolution,
                    self.data_packet.sdf_slice["sdf_residual"],
                )

            # update mesh
            if self.data_packet.flag_sdf_grid_updated:
                self.data_packet.flag_sdf_grid_updated = False

                assert self.data_packet.sdf_grid_bounds is not None
                assert self.data_packet.sdf_grid_resolution is not None
                assert self.data_packet.sdf_grid_shape is not None
                assert self.data_packet.sdf_grid is not None

                self.mc_queue_in.put_nowait(
                    MarchingCubesTask(
                        name="sdf",
                        coords_min=self.data_packet.sdf_grid_bounds[0],
                        grid_res=[self.data_packet.sdf_grid_resolution] * 3,
                        grid_shape=self.data_packet.sdf_grid_shape,
                        grid_values=self.data_packet.sdf_grid["sdf"],
                        grid_mask=self.data_packet.sdf_grid_mask,
                    )
                )
                self.mc_queue_in.put_nowait(
                    MarchingCubesTask(
                        name="sdf_prior",
                        coords_min=self.data_packet.sdf_grid_bounds[0],
                        grid_res=[self.data_packet.sdf_grid_resolution] * 3,
                        grid_shape=self.data_packet.sdf_grid_shape,
                        grid_values=self.data_packet.sdf_grid["sdf_prior"],
                        grid_mask=self.data_packet.sdf_grid_mask,
                    )
                )

            self.visualize_mesh()  # update mesh visualization

            if self.data_packet.model_saved_path is not None:
                tqdm.write(f"[GUI] Model saved to {self.data_packet.model_saved_path}.")
                self.data_packet.model_saved_path = None  # reset

            if self.data_packet.time_stats is not None:
                fps = 1.0 / (self.data_packet.time_stats.get("train_frame", 1e-6))
                self.label_info_training_fps.text = f"Training FPS: {fps:.3f}"
                self.label_info_timing_and_loss.text = f"Timing:\n" + "\n".join(
                    [f"  {k}: {v:.6f} s" for k, v in self.data_packet.time_stats.items()]
                )
                self.data_packet.time_stats = None  # reset

            if self.timer_gui_update.average_t > 0:
                fps = 1.0 / self.timer_gui_update.average_t
                self.label_info_gui_fps.text = f"GUI FPS: {fps:.3f}"

            if self.data_packet.loss_stats is not None:
                loss_stats = flatten_dict(self.data_packet.loss_stats)
                self.label_info_timing_and_loss.text += f"\n\nLoss:\n" + "\n".join(
                    [f"  {k}: {v:.6f}" for k, v in loss_stats.items()]
                )
                self.data_packet.loss_stats = None  # reset

            if self.data_packet.gpu_mem_usage is not None:
                self.label_info_gpu_mem.text = f"GPU Memory Usage: {self.data_packet.gpu_mem_usage:.3f} GB"
                self.data_packet.gpu_mem_usage = None  # reset

            self.widget3d.scene.camera.get_model_matrix()
            camera = self.widget3d.scene.camera
            model_matrix = camera.get_model_matrix()
            extrinsic = np.linalg.inv(
                model_matrix
                @ np.array(  # toGL
                    [
                        [1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1],
                    ]
                )
            )
            projection_matrix = camera.get_projection_matrix()
            intrinsics = np.array(
                [
                    [
                        projection_matrix[0, 0] * self.widget3d_width / 2,
                        0,
                        (1 - projection_matrix[2, 0]) * self.widget3d_width / 2,
                    ],
                    [
                        0,
                        projection_matrix[1, 1] * self.widget3d_height / 2,
                        (1 + projection_matrix[2, 1]) * self.widget3d_height / 2,
                    ],
                    [0, 0, 1],
                ]
            )
            far = camera.get_far()
            near = camera.get_near()
            fov = camera.get_field_of_view()
            scene_bounds = self.widget3d.scene.bounding_box
            self.label_info_scene_camera.text = f"Scene & Camera Info:\n"
            self.label_info_scene_camera.text += f"Extrinsic:\n"
            for row in extrinsic:
                self.label_info_scene_camera.text += "  " + " ".join(f"{val:.3f}" for val in row) + "\n"
            self.label_info_scene_camera.text += f"Intrinsics:\n"
            for row in intrinsics:
                self.label_info_scene_camera.text += "  " + " ".join(f"{val:.3f}" for val in row) + "\n"
            self.label_info_scene_camera.text += f"FOV: {fov:.3f} deg\nNear: {near:.3f} m\nFar: {far:.3f} m\n"
            self.label_info_scene_camera.text += f"Scene Bounds:\n"
            self.label_info_scene_camera.text += f"  Min: {scene_bounds.min_bound}\n"
            self.label_info_scene_camera.text += f"  Max: {scene_bounds.max_bound}\n"
            self.label_info_scene_camera.text += f"  Extent: {scene_bounds.get_extent()}\n"

        if not self.camera_init:
            self.camera_init = True
            self.visualize_gt_mesh()  # show gt mesh at the beginning if available and enabled

        self.set_camera()

        # Capture frame for video recording if active
        if not self.button_start_video.enabled and self.switch_vis.is_on:
            self._capture_frame_to_video()

        self.window.post_redraw()

    def communicate(self):
        # debug_counter = 0  # Counter to reduce debug output frequency
        while True:
            time.sleep(self.sleep_interval)

            # Check exit flag first
            if self.data_packet.flag_exit:
                tqdm.write("[GUI] Exiting communicate loop due to flag_exit.")
                self._send_flag_gui_closed()
                break  # exit thread

            # # Only print debug message every 100 iterations (once per second at 10ms intervals)
            # debug_counter += 1
            # if debug_counter % 100 == 0:
            #     tqdm.write("[GUI] Communicate thread awake.")
            #     debug_counter = 0

            current_time = time.time()
            if current_time - self.last_control_packet_timestamp > 0.01:  # 100 Hz
                self.send_control_packet()
                self.last_control_packet_timestamp = current_time

            try:
                data_packet = self.receive_data_packet(get_latest=True)
                if not self.data_packet.flag_exit:  # Don't post updates if exiting
                    o3d_gui.Application.instance.post_to_main_thread(
                        self.window, lambda: self.update_wrapper(data_packet)
                    )
            except Exception as e:
                tqdm.write(f"[GUI] Error posting to main thread: {e}")

    def communicate_thread(self):
        try:
            self.communicate()
        except Exception as e:
            tqdm.write(f"[GUI] Communicate thread encountered an error: {e}")
        tqdm.write("[GUI] Communicate thread exited.")

    def _clean_up_marching_cube_process(self):
        if self.mc_process is None:
            return

        # Clear input queue to unblock the process
        while not self.mc_queue_in.empty():
            try:
                self.mc_queue_in.get_nowait()
            except queue.Empty:
                break
        tqdm.write("[GUI] Cleaned up mc_queue_in.")

        # Send termination signal
        try:
            self.mc_queue_in.put_nowait(None)
        except Exception as e:
            tqdm.write(f"[GUI] Error sending termination signal: {e}")

        # Clear output queue
        while not self.mc_queue_out.empty():
            try:
                self.mc_queue_out.get_nowait()
            except queue.Empty:
                break
        tqdm.write("[GUI] Cleaned up mc_queue_out.")

        # Wait for process to terminate with timeout
        try:
            self.mc_process.join(timeout=2.0)
            if self.mc_process.is_alive():
                tqdm.write("[GUI] Marching cubes process didn't exit gracefully, terminating...")
                self.mc_process.terminate()
                self.mc_process.join(timeout=1.0)
                if self.mc_process.is_alive():
                    tqdm.write("[GUI] Force killing marching cubes process...")
                    self.mc_process.kill()
        except Exception as e:
            tqdm.write(f"[GUI] Error terminating marching cubes process: {e}")

        # Cleanup resources
        try:
            self.mc_queue_in.close()
            self.mc_queue_out.close()
            self.mc_queue_in.join_thread()
            self.mc_queue_out.join_thread()
        except Exception as e:
            tqdm.write(f"[GUI] Error closing marching cubes queues: {e}")

        self.mc_queue_in = None
        self.mc_queue_out = None
        self.mc_process = None
        tqdm.write("[GUI] Terminated marching cubes process.")

    def _clean_up_octree_process(self):
        if self.octree_process is None:
            return

        # Clear input queue to unblock the process
        while not self.octree_queue_in.empty():
            try:
                self.octree_queue_in.get_nowait()
            except queue.Empty:
                break
        tqdm.write("[GUI] Cleaned up octree_queue_in.")

        # Send termination signal
        try:
            self.octree_queue_in.put_nowait(None)
        except Exception as e:
            tqdm.write(f"[GUI] Error sending termination signal: {e}")

        # Clear output queue
        while not self.octree_queue_out.empty():
            try:
                self.octree_queue_out.get_nowait()
            except queue.Empty:
                break
        tqdm.write("[GUI] Cleaned up octree_queue_out.")

        # Wait for process to terminate with timeout
        try:
            self.octree_process.join(timeout=2.0)
            if self.octree_process.is_alive():
                tqdm.write("[GUI] Octree process didn't exit gracefully, terminating...")
                self.octree_process.terminate()
                self.octree_process.join(timeout=1.0)
                if self.octree_process.is_alive():
                    tqdm.write("[GUI] Force killing octree process...")
                    self.octree_process.kill()
        except Exception as e:
            tqdm.write(f"[GUI] Error terminating octree process: {e}")

        # Cleanup resources
        try:
            self.octree_queue_in.close()
            self.octree_queue_out.close()
            self.octree_queue_in.join_thread()
            self.octree_queue_out.join_thread()
        except Exception as e:
            tqdm.write(f"[GUI] Error closing octree queues: {e}")

        self.octree_queue_in = None
        self.octree_queue_out = None
        self.octree_process = None
        tqdm.write("[GUI] Terminated octree process.")

    def _clean_up_video_writer_process(self):
        if self.video_writer_process is None:
            return

        self._close_video_writer()  # Ensure video writer is closed properly

        # Wait for process to terminate with timeout
        try:
            while self.video_writer_process.is_alive():  # Wait until it finishes processing
                # Send termination signal
                try:
                    self.video_queue_in.put_nowait(None)
                except Exception as e:
                    tqdm.write(f"[GUI] Error sending termination signal to video writer: {e}")
                    if self.video_writer_process.is_alive():
                        self.video_writer_process.kill()  # Force kill if we can't send the signal
                self.video_writer_process.join(timeout=2.0)
        except Exception as e:
            tqdm.write(f"[GUI] Error joining video writer process: {e}")
            if self.video_writer_process.is_alive():
                tqdm.write("[GUI] Video writer process didn't exit gracefully, terminating...")
                self.video_writer_process.terminate()
                self.video_writer_process.join(timeout=1.0)
                if self.video_writer_process.is_alive():
                    tqdm.write("[GUI] Force killing video writer process...")
                    self.video_writer_process.kill()

        # Cleanup queues
        while not self.video_queue_in.empty():
            try:
                self.video_queue_in.get_nowait()
            except queue.Empty:
                break
        tqdm.write("[GUI] Cleaned up video_queue_in.")
        while not self.video_queue_out.empty():
            try:
                self.video_queue_out.get_nowait()
            except queue.Empty:
                break
        tqdm.write("[GUI] Cleaned up video_queue_out.")

        # Cleanup resources
        try:
            self.video_queue_in.close()
            self.video_queue_out.close()
            self.video_queue_in.join_thread()
            self.video_queue_out.join_thread()
        except Exception as e:
            tqdm.write(f"[GUI] Error closing video writer queue: {e}")

        self.video_queue_in = None
        self.video_queue_out = None
        self.video_writer_process = None
        tqdm.write("[GUI] Terminated video writer process.")

    def cleanup(self):
        with self._cleanup_lock:
            if self._cleaned:
                return
            self._cleaned = True  # Prevent re-entrance

        tqdm.write("[GUI] Cleaning up resources...")

        # Send final close signal
        self._send_flag_gui_closed()

        # Stop video recording and cleanup video writer

        # Clean up processes
        self._clean_up_marching_cube_process()
        self._clean_up_octree_process()
        self._clean_up_video_writer_process()

        # Stop communicate thread
        self.data_packet.flag_exit = True  # Signal thread to exit
        self.comm_thread.join(timeout=2.0)
        if self.comm_thread.is_alive():
            tqdm.write("[GUI] Communicate thread didn't exit gracefully, terminating...")
            # Note: Python threads cannot be forcefully killed; we rely on the flag_exit to stop it.
            # If it doesn't stop, it will exit when the main program exits

        tqdm.write("[GUI] Cleanup complete. Exiting now.")
