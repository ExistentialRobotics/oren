import json
import os
import random
import time
from typing import Callable, Optional

import torch as _torch  # used only for anomaly detection setup

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from oren import torch
from oren.criterion import Criterion
from oren.evaluator_oren import OrenEvaluator
from oren.frame import Frame
from oren.key_frame_set import KeyFrameSet
from oren.loggers import BasicLogger
from oren.model import SdfNetwork
from oren.trainer_config import TrainerConfig
from oren.utils.import_util import get_dataset
from oren.utils.profiling import GpuTimer
from oren.utils.sampling import SampleResults, generate_sdf_samples


class Trainer:
    def __init__(self, cfg: TrainerConfig, data_stream=None):
        self.cfg = cfg

        self.setup_seed(self.cfg.seed)

        # TF32 tensor cores for float32 matmuls when set to "high" (faster; small
        # precision loss). Default "highest" keeps full fp32.
        torch.set_float32_matmul_precision(
            getattr(self.cfg, "float32_matmul_precision", "highest"))

        if data_stream is None:
            self.data_stream = get_dataset(cfg.data.dataset_name, cfg.data.dataset_args)
        else:
            self.data_stream = data_stream

        # Streaming sources advertise themselves with a class attribute.
        # Python's built-in len() rejects negative returns from __len__, so a length-based sentinel can't be used.
        self.streaming = getattr(self.data_stream, "streaming", False)

        # set the bound automatically from the dataset if available.
        # the bound is used for evaluation and mesh extraction.
        # the training does not rely on the bound.
        if self.cfg.bound_min is None and self.data_stream.bound_min is not None and self.data_stream.bound_max is not None:
            self.cfg.bound_min = (self.data_stream.bound_min - 0.1).cpu().tolist()
            self.cfg.bound_max = (self.data_stream.bound_max + 0.1).cpu().tolist()

        self.bound_min = torch.tensor(self.cfg.bound_min, dtype=torch.float32, device=self.cfg.device)
        self.bound_max = torch.tensor(self.cfg.bound_max, dtype=torch.float32, device=self.cfg.device)

        if not self.streaming:
            if self.cfg.data.end_frame < 0:
                self.cfg.data.end_frame = len(self.data_stream)
            self.cfg.data.start_frame = min(self.cfg.data.start_frame, len(self.data_stream) - 1)
            self.cfg.data.end_frame = min(self.cfg.data.end_frame, len(self.data_stream))
        self.current_frame_idx = self.cfg.data.start_frame

        self.key_frame_set = KeyFrameSet(
            cfg=self.cfg.key_frame_set,
            max_num_voxels=self.cfg.model.octree_cfg.init_voxel_num,
            device=self.cfg.device,
        )
        self.model = SdfNetwork(self.cfg.model)
        self.model.to(self.cfg.device)
        if getattr(self.cfg, "compile_model", False):
            try:
                self.model = torch.compile(self.model, mode=self.cfg.compile_mode)
                print(f"[Trainer] torch.compile enabled (mode={self.cfg.compile_mode})",
                      flush=True)
            except Exception as exc:  # noqa: BLE001 — fall back to eager
                print(f"[Trainer] torch.compile unavailable, using eager: {exc}",
                      flush=True)

        self.logger = BasicLogger(cfg.log_dir, cfg.exp_name, cfg.as_dict())

        # Per-frame timing log (mapping-speed figure). Opened lazily-free here so
        # a row is appended per processed frame; closed in _finalize_timing_logs().
        self._frame_timing_fh = None
        self._map_start_t: Optional[float] = None
        if self.cfg.log_frame_timing:
            path = os.path.join(self.logger.log_dir, "frame_timing.csv")
            self._frame_timing_fh = open(path, "w")
            self._frame_timing_fh.write(
                "frame_id,wall_t_s,frame_dt_ms,n_iter,is_key_frame,n_voxels\n")
            self.logger.info(f"Frame timing -> {path}")

        self.epoch = 0
        self.global_step = 0
        self.num_iterations = 0
        # fused=True collapses the whole Adam update into a single CUDA kernel.
        # The model is tiny, so the per-step launch overhead of the default
        # (multi-kernel) path dominates on this launch-bound workload.
        # capturable=True is required so optimizer.step() can run inside a CUDA
        # graph (cuda_graph_training); it's a no-op cost otherwise.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, fused=True,
            capturable=getattr(self.cfg, "cuda_graph_training", False),
        )
        # Lazily-built CUDA graph of the training step + its static I/O buffers.
        self._cuda_graph = None
        self._graph_buffers = None
        self._graph_loss = None
        self.criterion = Criterion(
            cfg=self.cfg.criterion,
            n_stratified=self.cfg.sample_rays.n_stratified,
            n_perturbed=self.cfg.sample_rays.n_perturbed,
        )

        self.selected_key_frame_indices = []
        self.samples: Optional[SampleResults] = None
        self.extra_surface_pcd: Optional[torch.Tensor] = None
        self.loss_dict = dict()

        timer_on = self.cfg.profiling
        verbose = self.cfg.profiling_verbose
        # profiling_sync=False -> per-stage CPU dispatch time, no cuda.synchronize
        # (no GPU drains). Use it to attribute wall time on this dispatch-bound
        # workload at ~production speed; profiling_sync=True gives accurate GPU
        # per-stage times but forces ~10 syncs/frame.
        sync = getattr(self.cfg, "profiling_sync", True)

        def _mk(name):
            return GpuTimer(name, enable=timer_on, verbose=verbose, use_cuda_sync=sync)

        self.timer_get_points = _mk("get points")
        self.timer_octree_insert = _mk("octree insert")
        self.timer_key_frame_set_update = _mk("key frame set update")
        self.timer_train_frame = _mk("train with frame")
        self.timer_select_key_frames = _mk("select key frames")
        self.timer_sample_rays = _mk("sample rays")
        self.timer_generate_sdf_samples = _mk("generate sdf samples")
        self.timer_compute_offset_points = _mk("compute offset points")
        self.timer_find_voxel_indices_offset_points = _mk("find voxel indices for offset points")
        self.timer_find_voxel_indices_sampled_xyz = _mk("find voxel indices for sampled_xyz")
        self.timer_graph_input_copy = _mk("graph input copy")
        self.timer_training_iteration = _mk("training iteration")

        # Ordered (name, timer) pairs for the per-frame stage CSV below. Each
        # timer's `.t` holds its last call's elapsed time; all 12 fire every frame
        # (finite-difference + cuda-graph path), so the row is never stale.
        self._stage_timers = [
            ("get_points", self.timer_get_points),
            ("octree_insert", self.timer_octree_insert),
            ("key_frame_set_update", self.timer_key_frame_set_update),
            ("train_frame", self.timer_train_frame),
            ("select_key_frames", self.timer_select_key_frames),
            ("sample_rays", self.timer_sample_rays),
            ("generate_sdf_samples", self.timer_generate_sdf_samples),
            ("compute_offset_points", self.timer_compute_offset_points),
            ("find_voxel_indices_offset_points", self.timer_find_voxel_indices_offset_points),
            ("find_voxel_indices_sampled_xyz", self.timer_find_voxel_indices_sampled_xyz),
            ("graph_input_copy", self.timer_graph_input_copy),
            ("training_iteration", self.timer_training_iteration),
        ]
        # Per-frame stage timing -> medians (not tail-inflated like average_t) and
        # tail attribution. Needs profiling=True so the GpuTimers are enabled.
        self._stage_timing_fh = None
        if self.cfg.log_frame_timing and self.cfg.profiling:
            spath = os.path.join(self.logger.log_dir, "stage_timing_per_frame.csv")
            self._stage_timing_fh = open(spath, "w")
            self._stage_timing_fh.write(
                "frame_id,frame_dt_ms,is_key_frame,"
                + ",".join(n + "_ms" for n, _ in self._stage_timers) + "\n")
            self.logger.info(f"Per-frame stage timing -> {spath}")

        self.training_iteration_end_callback: Callable[[Trainer], None] = None  # type: ignore
        self.training_frame_start_callback: Callable[[Trainer, Frame], bool] = None  # type: ignore
        self.training_end_callback: Callable[[Trainer], None] = None  # type: ignore

        if self.cfg.detect_nan:
            _torch.autograd.set_detect_anomaly(True)
            self.logger.info("NaN detection enabled (anomaly detection + per-tensor checks). Training will be slow.")

        self.evaluator = OrenEvaluator(
            batch_size=self.cfg.batch_size,
            clean_mesh=self.cfg.clean_mesh,
            model_cfg=self.cfg.model,
            model=self.model,
            device=self.cfg.device,
        )

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train(self):
        try:
            if self.streaming:
                self._train_streaming()
            else:
                self._train_bounded()
        finally:
            for _ in range(self.cfg.final_iterations):
                self.train_with_frame(None)

            self._finalize_timing_logs()
            self.logger.info("Training completed.")
            if self.training_end_callback is not None:
                self.training_end_callback(self)

            if self.cfg.final_evaluate:
                self.evaluate()
            if self.cfg.final_save_model:
                self.save_model("final.pth")

    def _train_streaming(self) -> None:
        pbar = tqdm(desc="Mapping (streaming)", ncols=120, leave=False)
        try:
            # init frame_id for streaming source, which is only used for logging and checkpoint naming.
            # self.current_frame_idx is the fetching counter for streaming source, which is increased
            # whenever we fetch a frame (even if it's None or has bad pose).
            frame_id = self.current_frame_idx
            while True:
                frame = self.fetch_one_frame()
                if frame is None:
                    # `None` from a streaming source means either shutdown or transient idle;
                    # the loader exposes which via `is_shutdown` (default True for sources without it,
                    # so non-ROS streaming sources keep non-streaming behavior).
                    if getattr(self.data_stream, "is_shutdown", True):
                        self.logger.info("No more frames (data stream closed), finish mapping.")
                        return
                    # Transient idle: keep optimizing on existing keyframes.
                    if not self.train_with_frame(None):
                        return
                    continue
                if not self._step_one_frame(frame, frame_id):
                    return
                frame_id += 1  # increase frame_id when frame is not None and processed successfully
                pbar.update(1)
        finally:
            pbar.close()

    def _train_bounded(self) -> None:
        frame_indices = range(
            self.cfg.data.start_frame,
            self.cfg.data.end_frame,
            self.cfg.frame_downsample,
        )
        for frame_id in tqdm(frame_indices, desc="Mapping", ncols=120, leave=False):
            self.current_frame_idx = frame_id
            frame = self.fetch_one_frame()
            if frame is None:
                self.logger.info("No more valid frames, finish mapping.")
                return
            if not self._step_one_frame(frame, frame_id):
                return

    def _step_one_frame(self, frame: Frame, frame_id: int) -> bool:
        """Run insertion + key-frame update + training for one frame. Returns False if interrupted by callback."""
        t_frame_start = time.perf_counter()
        with self.timer_get_points:
            points = frame.get_points(to_world_frame=True, device=self.cfg.device)

        with self.timer_octree_insert:
            _, seen_voxels = self.insert_points_to_octree(points)

        with self.timer_key_frame_set_update:
            is_key_frame = self.update_key_frame_set(frame, seen_voxels)

        if is_key_frame:
            self.logger.info(f"Frame {frame_id} is selected as a key frame.")

        with self.timer_train_frame:
            if not self.train_with_frame(frame=frame):
                return False
        self.epoch += 1

        self._log_frame_timing(frame_id, t_frame_start, is_key_frame)

        if self.cfg.ckpt_interval > 0 and self.epoch % self.cfg.ckpt_interval == 0:
            self.save_model(f"epoch_{self.epoch:04d}.pth")
        return True

    def _log_frame_timing(self, frame_id: int, t_start: float, is_key_frame: bool) -> None:
        """Append one row to frame_timing.csv: end-to-end frame duration, wall
        clock since the first frame, and cumulative occupied-voxel count."""
        if self._frame_timing_fh is None:
            return
        now = time.perf_counter()
        if self._map_start_t is None:
            self._map_start_t = t_start
        dt_ms = (now - t_start) * 1e3
        wall_s = now - self._map_start_t
        try:
            # True octree fill = nodes actually stored in the voxel buffer. NOTE:
            # octree.voxels.shape[0] is the *capacity* (== init_voxel_num), so it
            # logged a constant. number_of_nodes is the live count of used rows,
            # i.e. exactly what init_voxel_num caps — use it to size init_voxel_num
            # (leave headroom; overflow at number_of_nodes == init_voxel_num).
            n_voxels = int(self.model.octree.sso.number_of_nodes)
        except Exception:
            n_voxels = -1
        self._frame_timing_fh.write(
            f"{frame_id},{wall_s:.6f},{dt_ms:.3f},{self.num_iterations},"
            f"{int(is_key_frame)},{n_voxels}\n"
        )
        self._frame_timing_fh.flush()

        if self._stage_timing_fh is not None:
            vals = ",".join(f"{t.t * 1e3:.3f}" for _, t in self._stage_timers)
            self._stage_timing_fh.write(
                f"{frame_id},{dt_ms:.3f},{int(is_key_frame)},{vals}\n")
            self._stage_timing_fh.flush()

    def _finalize_timing_logs(self) -> None:
        """Close the per-frame CSV and, if profiling, dump per-stage averages."""
        if self._frame_timing_fh is not None:
            try:
                self._frame_timing_fh.close()
            except Exception:
                pass
            self._frame_timing_fh = None
        if self._stage_timing_fh is not None:
            try:
                self._stage_timing_fh.close()
            except Exception:
                pass
            self._stage_timing_fh = None
        if self.cfg.profiling:
            stats = self.get_time_stats()
            stats["num_frames"] = self.epoch
            try:
                path = os.path.join(self.logger.log_dir, "stage_timing.json")
                with open(path, "w") as fh:
                    json.dump(stats, fh, indent=2)
                self.logger.info(f"Stage timing -> {path}")
            except Exception as exc:
                self.logger.info(f"Failed to write stage timing: {exc}")

    def fetch_one_frame(self) -> Optional[Frame]:
        frame = None
        if self.streaming:
            # Streaming source: index value is unused; loader blocks until a frame
            # is ready and returns None on shutdown. Skip frames with bad poses.
            while True:
                frame = self.data_stream[self.current_frame_idx]
                self.current_frame_idx += 1  # fetching counter for streaming source
                if frame is None:
                    return None
                if torch.all(frame.get_ref_pose().isfinite()):
                    return frame
        else:
            while self.current_frame_idx < self.cfg.data.end_frame:
                frame = self.data_stream[self.current_frame_idx]
                self.current_frame_idx += 1
                if not torch.all(frame.get_ref_pose().isfinite()):  # bad pose
                    continue
                break
            return frame

    @torch.no_grad()
    def insert_points_to_octree(self, points: torch.Tensor):
        voxels, seen_voxels = self.model.octree.insert_points(points)
        return voxels, seen_voxels

    @torch.no_grad()
    def find_voxel_indices(self, points: torch.Tensor):
        """
        Find the voxel indices for the given points.
        Args:
            points: (..., 3) points to find the voxel indices for

        Returns:
            (..., ) voxel indices for the given points, -1 if not exists
        """
        shape = points.shape
        voxel_indices = self.model.octree.find_voxel_indices(points.view(-1, 3), False)
        voxel_indices = voxel_indices.view(shape[:-1])
        return voxel_indices

    def update_key_frame_set(self, frame: Frame, seen_voxels: torch.Tensor) -> bool:
        return self.key_frame_set.add_key_frame(frame, seen_voxels)

    def select_key_frames(self) -> list[int]:
        return self.key_frame_set.select_key_frames()

    def train_with_frame(self, frame: Frame | None):
        self.num_iterations = self.cfg.num_iterations_per_frame
        if self.epoch < self.cfg.num_init_frames:
            self.num_iterations = self.cfg.init_frame_iterations

        if self.training_frame_start_callback is not None:
            if not self.training_frame_start_callback(self, frame):
                self.logger.info("Training interrupted by callback, exiting.")
                return False  # exit training

        with self.timer_select_key_frames:
            self.selected_key_frame_indices = self.key_frame_set.select_key_frames()
        with self.timer_sample_rays:
            rays_o_all, rays_d_all, depth_samples_all = self.key_frame_set.sample_rays(
                num_samples=self.cfg.num_rays_total,
                key_frame_indices=self.selected_key_frame_indices,
                current_frame=frame,
            )
            rays_o_all = rays_o_all.to(self.cfg.device)
            rays_d_all = rays_d_all.to(self.cfg.device)
            depth_samples_all = depth_samples_all.to(self.cfg.device)

            if self.cfg.extra_surface_sample:
                self.extra_surface_pcd = self.key_frame_set.sample_points(
                    ratio=1.0 / self.cfg.frame_downsample,
                    key_frame_indices=self.selected_key_frame_indices,
                    current_frame=frame,
                )

        with self.timer_generate_sdf_samples:
            self.samples = generate_sdf_samples(
                rays_d_all=rays_d_all,
                rays_o_all=rays_o_all,
                depth_samples_all=depth_samples_all,
                cfg=self.cfg.sample_rays,
                extra_surface_pcd=self.extra_surface_pcd,
                device=self.cfg.device,
            )

            mask = torch.ones_like(self.samples.sampled_xyz[..., 0], dtype=torch.bool)
            if not self.cfg.data.apply_bound:
                mask = mask & (self.samples.sampled_xyz >= self.bound_min).all(dim=-1)
                mask = mask & (self.samples.sampled_xyz <= self.bound_max).all(dim=-1)

        # self.samples.sampled_xyz: (n, m, 3)
        num_rays = self.samples.sampled_xyz.shape[0]
        if self.cfg.grad_method == "autodiff":
            self.samples.sampled_xyz.requires_grad_(True)
        else:
            with self.timer_compute_offset_points:
                (
                    offset_points_plus,
                    offset_points_minus,
                ) = self.compute_offset_points_for_finite_diff(self.samples.sampled_xyz)
            with self.timer_find_voxel_indices_offset_points:
                voxel_indices_plus = self.find_voxel_indices(offset_points_plus)  # (n, m, 3)
                voxel_indices_minus = self.find_voxel_indices(offset_points_minus)  # (n, m, 3)
        with self.timer_find_voxel_indices_sampled_xyz:
            voxel_indices = self.find_voxel_indices(self.samples.sampled_xyz)  # (n, m)
            mask = mask & (voxel_indices != -1)
            # assert voxel_indices.min() != -1, "voxel_indices has -1"

        bs = int(self.cfg.batch_size / self.samples.sampled_xyz.shape[1])

        if self.cfg.cuda_graph_training and self.cfg.grad_method != "autodiff":
            self._train_iterations_graphed(
                num_rays,
                voxel_indices,
                offset_points_plus,
                offset_points_minus,
                voxel_indices_plus,
                voxel_indices_minus,
                mask,
            )
            return True

        for _ in range(self.num_iterations):
            self.model.train()
            with self.timer_training_iteration:
                with torch.enable_grad():
                    self.optimizer.zero_grad()
                    sdf_pred_all = []
                    sdf_prior_all = []
                    sdf_grad_all = []
                    sdf_prior_grad_all = []
                    for i in range(0, num_rays, bs):
                        j = min(i + bs, num_rays)
                        points = self.samples.sampled_xyz[i:j]  # (b, m, 3)
                        voxel_indices_batch = voxel_indices[i:j]
                        _, sdf_prior, _, sdf_pred = self.model(points, voxel_indices_batch)
                        if self.cfg.grad_method == "autodiff":
                            sdf_grad = self.compute_sdf_grad_autodiff(points, sdf_pred)
                            sdf_prior_grad = self.compute_sdf_grad_autodiff(points, sdf_prior)
                        else:
                            sdf_grad, sdf_prior_grad = self.compute_sdf_grad_finite_difference(
                                points=points,
                                offset_points_plus=offset_points_plus[i:j],
                                offset_points_minus=offset_points_minus[i:j],
                                voxel_indices_plus=voxel_indices_plus[i:j],
                                voxel_indices_minus=voxel_indices_minus[i:j],
                            )[:2]

                        sdf_pred_all.append(sdf_pred)
                        sdf_prior_all.append(sdf_prior)  # (b, m)
                        sdf_grad_all.append(sdf_grad)  # (b, m, 3)
                        sdf_prior_grad_all.append(sdf_prior_grad)  # (b, m, 3)

                    if len(sdf_pred_all) == 1:
                        sdf_pred_all = sdf_pred_all[0]
                        sdf_prior_all = sdf_prior_all[0]
                        sdf_grad_all = sdf_grad_all[0]
                        sdf_prior_grad_all = sdf_prior_grad_all[0]
                    else:
                        sdf_pred_all = torch.cat(sdf_pred_all, dim=0)
                        sdf_prior_all = torch.cat(sdf_prior_all, dim=0)
                        sdf_grad_all = torch.cat(sdf_grad_all, dim=0)
                        sdf_prior_grad_all = torch.cat(sdf_prior_grad_all, dim=0)

                    if self.cfg.detect_nan:
                        self._log_nan("sdf_pred", sdf_pred_all)
                        self._log_nan("sdf_grad", sdf_grad_all)

                    loss, self.loss_dict = self.criterion(
                        pred_sdf=sdf_pred_all,
                        pred_prior=sdf_prior_all,
                        pred_grad=sdf_grad_all,
                        pred_prior_grad=sdf_prior_grad_all,
                        gt_sdf_perturb=self.samples.perturbation_sdf,
                        gt_sdf_stratified=self.samples.stratified_sdf,
                        positive_perturbation_mask=self.samples.positive_perturbation_mask,
                        perturb_eta=self.cfg.sample_rays.sigma_s,
                        valid_mask=mask,
                    )
                    loss.backward()
                    if self.cfg.detect_nan:
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    if self.cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    if self.cfg.detect_nan:
                        self._log_nan_grads()
                    self.optimizer.step()
            self.global_step += 1

            # Per-iteration loss logging forces a host sync + TensorBoard disk I/O
            # every frame. Gate it behind log_loss_interval (0 = off) so it doesn't
            # cost throughput during streaming map builds.
            if (self.cfg.log_loss_interval > 0
                    and self.global_step % self.cfg.log_loss_interval == 0):
                self.logger.info(f"loss_dict: {self.loss_dict}")
                for k, v in self.loss_dict.items():
                    self.logger.tb.add_scalar(f"loss/{k}", v, self.global_step)

            if self.training_iteration_end_callback is not None:
                self.training_iteration_end_callback(self)
        return True

    def _train_iterations_graphed(
        self,
        num_rays,
        voxel_indices,
        offset_points_plus,
        offset_points_minus,
        voxel_indices_plus,
        voxel_indices_minus,
        mask,
    ):
        """Run the training step via a captured CUDA graph (finite-difference only).

        The per-frame valid-ray count varies, so the frame's tensors are copied
        into fixed-shape static buffers (padded to num_rays_total; the tail is
        marked invalid via voxel_indices=-1 + valid_mask=False, which the octree
        forward and criterion already zero/exclude). The forward + FD gradient +
        criterion + backward + optimizer.step are captured once and replayed,
        collapsing ~400 per-iteration dispatches into a single launch. Falls back
        to eager (permanently, this run) if capture fails.
        """
        self.model.train()
        eps = self.cfg.finite_difference_eps
        R = self.cfg.num_rays_total
        m = self.samples.sampled_xyz.shape[1]
        V = min(int(num_rays), R)

        if self._graph_buffers is None:
            dev = self.cfg.device
            self._graph_buffers = {
                "xyz": torch.zeros(R, m, 3, device=dev),
                "vi": torch.full((R, m), -1, dtype=torch.long, device=dev),
                "op": torch.zeros(R, m, 3, 3, device=dev),
                "om": torch.zeros(R, m, 3, 3, device=dev),
                "vip": torch.full((R, m, 3), -1, dtype=torch.long, device=dev),
                "vim": torch.full((R, m, 3), -1, dtype=torch.long, device=dev),
                "gtp": torch.zeros(R, self.cfg.sample_rays.n_perturbed, device=dev),
                "gts": torch.zeros(R, self.cfg.sample_rays.n_stratified, device=dev),
                "pm": torch.zeros(R, self.cfg.sample_rays.n_perturbed, dtype=torch.bool, device=dev),
                "mask": torch.zeros(R, m, dtype=torch.bool, device=dev),
            }
        b = self._graph_buffers

        # Copy this frame's valid rows in; reset the padded tail to "invalid".
        with self.timer_graph_input_copy:
            b["vi"].fill_(-1); b["vip"].fill_(-1); b["vim"].fill_(-1); b["mask"].zero_()
            b["xyz"][:V].copy_(self.samples.sampled_xyz[:V])
            b["vi"][:V].copy_(voxel_indices[:V])
            b["op"][:V].copy_(offset_points_plus[:V])
            b["om"][:V].copy_(offset_points_minus[:V])
            b["vip"][:V].copy_(voxel_indices_plus[:V])
            b["vim"][:V].copy_(voxel_indices_minus[:V])
            b["gtp"][:V].copy_(self.samples.perturbation_sdf[:V])
            b["gts"][:V].copy_(self.samples.stratified_sdf[:V])
            b["pm"][:V].copy_(self.samples.positive_perturbation_mask[:V])
            b["mask"][:V].copy_(mask[:V])

        def _step():
            self.optimizer.zero_grad(set_to_none=False)
            with torch.enable_grad():
                _, sp, _, spred = self.model(b["xyz"], b["vi"])
                _, spp, _, splus = self.model(b["op"], b["vip"])
                _, spm, _, sminus = self.model(b["om"], b["vim"])
                grad = (splus - sminus) / (2 * eps)
                prior_grad = (spp - spm) / (2 * eps)
                loss, _ = self.criterion(
                    pred_sdf=spred, pred_prior=sp,
                    pred_grad=grad, pred_prior_grad=prior_grad,
                    gt_sdf_perturb=b["gtp"], gt_sdf_stratified=b["gts"],
                    positive_perturbation_mask=b["pm"],
                    perturb_eta=self.cfg.sample_rays.sigma_s,
                    valid_mask=b["mask"],
                )
                loss.backward()
                self.optimizer.step()
            return loss

        if self._cuda_graph is None:
            try:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        _step()
                torch.cuda.current_stream().wait_stream(stream)
                self._cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._cuda_graph):
                    self._graph_loss = _step()
                self.logger.info("cuda_graph_training: training step captured.")
            except Exception as exc:
                self.logger.info(
                    f"cuda_graph_training: capture failed ({exc}); falling back to eager."
                )
                self.cfg.cuda_graph_training = False
                self._cuda_graph = None
                for _ in range(self.num_iterations):
                    _step()
                    self.global_step += 1
                return

        with self.timer_training_iteration:
            for _ in range(self.num_iterations):
                self._cuda_graph.replay()
                self.global_step += 1
                if (self.cfg.log_loss_interval > 0
                        and self.global_step % self.cfg.log_loss_interval == 0):
                    self.logger.info(f"loss(total): {self._graph_loss.item():.4f}")
                if self.training_iteration_end_callback is not None:
                    self.training_iteration_end_callback(self)
        self.loss_dict = {"total_loss": self._graph_loss}

    def _log_nan(self, name: str, t: torch.Tensor) -> None:
        n = torch.isnan(t).sum().item()
        if n > 0:
            self.logger.info(f"[detect_nan] {name}: {n}/{t.numel()} NaN values, shape={tuple(t.shape)}, "
                             f"min={t[~torch.isnan(t)].min().item():.4f}, max={t[~torch.isnan(t)].max().item():.4f}")

    def _log_nan_grads(self) -> None:
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                n = torch.isnan(param.grad).sum().item()
                if n > 0:
                    self.logger.info(f"[detect_nan] grad/{name}: {n}/{param.grad.numel()} NaN values")

    @staticmethod
    def compute_sdf_grad_autodiff(points: torch.Tensor, pred_sdf: torch.Tensor):
        sdf_grad = torch.autograd.grad(
            outputs=pred_sdf,
            inputs=[points],
            grad_outputs=torch.ones_like(pred_sdf),
            create_graph=True,
            # retain_graph=True,
        )[0]
        return sdf_grad

    @torch.no_grad()
    def compute_offset_points_for_finite_diff(self, points: torch.Tensor):
        """
        Compute the offset points for finite difference gradient estimation.
        Args:
            points: (..., 3) points to compute the offset points for

        Returns:
            (..., 3, 3) tensor of points + offset
            (..., 3, 3) tensor of points - offset
        """
        eps = self.cfg.finite_difference_eps
        offset_points_plus = []
        offset_points_minus = []
        for i in range(3):
            points_plus = points.clone()
            points_plus[..., i] += eps  # (..., 3)
            offset_points_plus.append(points_plus)

            points_minus = points.clone()
            points_minus[..., i] -= eps  # (..., 3)
            offset_points_minus.append(points_minus)

        offset_points_plus = torch.stack(offset_points_plus, dim=-2)  # (..., 3, 3)
        offset_points_minus = torch.stack(offset_points_minus, dim=-2)  # (..., 3, 3)

        return offset_points_plus, offset_points_minus

    def compute_sdf_grad_finite_difference(
        self,
        points: torch.Tensor,
        offset_points_plus: Optional[torch.Tensor] = None,
        offset_points_minus: Optional[torch.Tensor] = None,
        voxel_indices_plus: Optional[torch.Tensor] = None,
        voxel_indices_minus: Optional[torch.Tensor] = None,
    ):
        """
        Compute the gradient of the SDF at the given points using finite difference.
        Args:
            points: (..., 3) points to compute the gradient for
            offset_points_plus: (..., 3, 3) tensor of points + offset, if None, will be computed
            offset_points_minus: (..., 3, 3) tensor of points - offset, if None, will be computed
            voxel_indices_plus: (..., 3) voxel indices for offset_points_plus, if None, will be computed
            voxel_indices_minus: (..., 3) voxel indices for offset_points_minus, if None, will be computed

        Returns:
            (..., 3) gradient of the SDF at the given points
            (..., 3) gradient of the SDF prior at the given points
            (..., 3, 3) offset_points_plus
            (..., 3, 3) offset_points_minus
            (..., 3) voxel_indices_plus
            (..., 3) voxel_indices_minus
        """
        eps = self.cfg.finite_difference_eps
        if offset_points_plus is None or offset_points_minus is None:
            offset_points_plus, offset_points_minus = self.compute_offset_points_for_finite_diff(points)
        voxel_indices_plus, sdf_prior_plus, _, sdf_plus = self.model(offset_points_plus, voxel_indices_plus)
        voxel_indices_minus, sdf_prior_minus, _, sdf_minus = self.model(offset_points_minus, voxel_indices_minus)

        grad = (sdf_plus - sdf_minus) / (2 * eps)
        prior_grad = (sdf_prior_plus - sdf_prior_minus) / (2 * eps)

        return (
            grad,
            prior_grad,
            offset_points_plus,
            offset_points_minus,
            voxel_indices_plus,
            voxel_indices_minus,
        )

    @torch.no_grad()
    def save_model(self, path: str):
        self.logger.log_ckpt(self.model.state_dict(), path)
        self.logger.info(f"Model saved to {path}.")

    def save_mesh(
        self,
        path: str,
        prior: bool = False,
        bound_min: Optional[list] = None,
        bound_max: Optional[list] = None,
        grid_resolution: Optional[float] = None,
        iso_value: Optional[float] = None,
    ) -> None:

        field = "sdf_prior" if prior else "sdf"
        bound_min = bound_min if bound_min is not None else self.cfg.bound_min
        bound_max = bound_max if bound_max is not None else self.cfg.bound_max
        grid_resolution = grid_resolution if grid_resolution is not None else self.cfg.mesh_resolution
        iso_value = iso_value if iso_value is not None else self.cfg.mesh_iso_value
        self.logger.info(
            f"Extracting mesh ({field}) with bound_min={bound_min}, bound_max={bound_max}, "
            f"grid_resolution={grid_resolution}, iso_value={iso_value}..."
        )

        [mesh] = self.evaluator.extract_mesh(
            bound_min=bound_min,
            bound_max=bound_max,
            grid_resolution=grid_resolution,
            fields=[field],
            iso_value=iso_value,
        )
        self.logger.log_mesh(mesh, path)
        self.logger.info(f"Mesh ({field}) saved to {path}.")

    def query_sdf(self, points: torch.Tensor, return_grad: bool = True, prior_only: bool = False) -> dict:
        """Forward the model on the given points. Returns dict with sdf and (optional) sdf_grad."""
        return self.evaluator.forward_model(
            self.model,
            points.to(self.cfg.device),
            get_grad=return_grad,
            auto_grad=True,
            prior_only=prior_only,
            device=self.cfg.device,
        )

    def get_time_stats(self) -> dict:
        time_stats = {
            "get_points": self.timer_get_points.average_t,
            "train_frame": self.timer_train_frame.average_t,
            "octree_insert": self.timer_octree_insert.average_t,
            "key_frame_set_update": self.timer_key_frame_set_update.average_t,
            "select_key_frames": self.timer_select_key_frames.average_t,
            "sample_rays": self.timer_sample_rays.average_t,
            "generate_sdf_samples": self.timer_generate_sdf_samples.average_t,
            "compute_offset_points": self.timer_compute_offset_points.average_t,
            "find_voxel_indices_offset_points": self.timer_find_voxel_indices_offset_points.average_t,
            "find_voxel_indices_sampled_xyz": self.timer_find_voxel_indices_sampled_xyz.average_t,
            "graph_input_copy": self.timer_graph_input_copy.average_t,
            "training_iteration": self.timer_training_iteration.average_t,
            # True => GPU per-stage times (cuda.synchronize); False => per-stage
            # CPU dispatch times (perf_counter, no GPU drains). The columns mean
            # different things depending on this flag, so record it.
            "cuda_sync": getattr(self.cfg, "profiling_sync", True),
        }
        return time_stats

    def evaluate(self, epoch_dir: Optional[str] = None):
        bound_min = self.cfg.bound_min
        bound_max = self.cfg.bound_max

        if self.cfg.save_mesh:
            mesh_prior, mesh = self.evaluator.extract_mesh(
                bound_min=bound_min,
                bound_max=bound_max,
                grid_resolution=self.cfg.mesh_resolution,
                fields=["sdf_prior", "sdf"],
                iso_value=self.cfg.mesh_iso_value,
            )
            if epoch_dir is not None:
                self.logger.log_mesh(mesh_prior, f"{epoch_dir}/mesh_prior.ply")
                self.logger.log_mesh(mesh, f"{epoch_dir}/mesh.ply")
            else:
                self.logger.log_mesh(mesh_prior, f"mesh_prior.ply")
                self.logger.log_mesh(mesh, f"mesh.ply")

        if self.cfg.save_slice:

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
            for axis in range(3):
                if self.cfg.slice_center is None:
                    pos = 0.5 * (bound_min[axis] + bound_max[axis])
                else:
                    pos = self.cfg.slice_center[axis]
                slice_result = self.evaluator.extract_slice(
                    axis=axis,
                    pos=pos,
                    resolution=self.cfg.mesh_resolution,
                    bound_min=bound_min,
                    bound_max=bound_max,
                )

                slice_config = slice_configs[axis]
                axis_name = slice_config["axis_name"]
                slice_bound = slice_result["slice_bound"].tolist()  # (bound_min, bound_max) for the two axes

                for slice_name in ["sdf_prior", "sdf_residual", "sdf"]:
                    slice_values = slice_result[slice_name].cpu().numpy()
                    plt.figure()
                    im = plt.imshow(
                        slice_values,
                        extent=(
                            slice_bound[0][0],
                            slice_bound[1][0],
                            slice_bound[0][1],
                            slice_bound[1][1],
                        ),
                        origin="lower",
                        cmap="jet",
                    )
                    plt.colorbar(im, shrink=0.8)
                    plt.xlabel(slice_config["xlabel"], fontsize=fontsize)
                    plt.ylabel(slice_config["ylabel"], fontsize=fontsize)
                    plt.title(f"At {axis_name} = {pos:.2f} m", fontsize=fontsize)
                    plt.tight_layout()
                    img_path = f"slice_{axis_name}_{slice_name}.png"
                    if epoch_dir is not None:
                        img_path = os.path.join(self.logger.misc_dir, epoch_dir, img_path)
                        os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    else:
                        img_path = os.path.join(self.logger.misc_dir, img_path)
                    plt.savefig(img_path, dpi=300)
                    plt.close()

        self.logger.info("Evaluation completed.")


def main():
    parser = TrainerConfig.get_argparser()
    cfg: TrainerConfig = parser.parse_args()
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
