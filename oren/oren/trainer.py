import os
import random
from typing import Callable, Optional

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
        if self.data_stream.bound_min is not None and self.data_stream.bound_max is not None:
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

        self.logger = BasicLogger(cfg.log_dir, cfg.exp_name, cfg.as_dict())

        self.epoch = 0
        self.global_step = 0
        self.num_iterations = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
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
        self.timer_octree_insert = GpuTimer("octree insert", enable=timer_on, verbose=verbose)
        self.timer_key_frame_set_update = GpuTimer("key frame set update", enable=timer_on, verbose=verbose)
        self.timer_train_frame = GpuTimer("train with frame", enable=timer_on, verbose=verbose)
        self.timer_select_key_frames = GpuTimer("select key frames", enable=timer_on, verbose=verbose)
        self.timer_sample_rays = GpuTimer("sample rays", enable=timer_on, verbose=verbose)
        self.timer_generate_sdf_samples = GpuTimer("generate sdf samples", enable=timer_on, verbose=verbose)
        self.timer_compute_offset_points = GpuTimer("compute offset points", enable=timer_on, verbose=verbose)
        self.timer_find_voxel_indices_offset_points = GpuTimer(
            "find voxel indices for offset points", enable=timer_on, verbose=verbose
        )
        self.timer_find_voxel_indices_sampled_xyz = GpuTimer(
            "find voxel indices for sampled_xyz", enable=timer_on, verbose=verbose
        )
        self.timer_training_iteration = GpuTimer("training iteration", enable=timer_on, verbose=verbose)

        self.training_iteration_end_callback: Callable[[Trainer], None] = None  # type: ignore
        self.training_frame_start_callback: Callable[[Trainer, Frame], bool] = None  # type: ignore
        self.training_end_callback: Callable[[Trainer], None] = None  # type: ignore

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
        for frame_id in tqdm(
            range(self.cfg.data.start_frame, self.cfg.data.end_frame),
            desc="Mapping",
            ncols=120,
            leave=False,
        ):
            frame = self.fetch_one_frame()
            if frame is None:
                self.logger.info("No more valid frames, finish mapping.")
                return
            if not self._step_one_frame(frame, frame_id):
                return

    def _step_one_frame(self, frame: Frame, frame_id: int) -> bool:
        """Run insertion + key-frame update + training for one frame. Returns False if interrupted by callback."""
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

        if self.cfg.ckpt_interval > 0 and self.epoch % self.cfg.ckpt_interval == 0:
            self.save_model(f"epoch_{self.epoch:04d}.pth")
        return True

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
                    self.optimizer.step()
            self.global_step += 1

            self.logger.info(f"loss_dict: {self.loss_dict}")
            for k, v in self.loss_dict.items():
                self.logger.tb.add_scalar(f"loss/{k}", v, self.global_step)

            if self.training_iteration_end_callback is not None:
                self.training_iteration_end_callback(self)
        return True

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
            "train_frame": self.timer_train_frame.average_t,
            "octree_insert": self.timer_octree_insert.average_t,
            "key_frame_set_update": self.timer_key_frame_set_update.average_t,
            "select_key_frames": self.timer_select_key_frames.average_t,
            "sample_rays": self.timer_sample_rays.average_t,
            "generate_sdf_samples": self.timer_generate_sdf_samples.average_t,
            "compute_offset_points": self.timer_compute_offset_points.average_t,
            "find_voxel_indices_offset_points": self.timer_find_voxel_indices_offset_points.average_t,
            "find_voxel_indices_sampled_xyz": self.timer_find_voxel_indices_sampled_xyz.average_t,
            "training_iteration": self.timer_training_iteration.average_t,
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
