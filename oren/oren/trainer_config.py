from dataclasses import dataclass, field
from typing import Optional

from oren.criterion import CriterionConfig
from oren.dataset.data_config import DataConfig
from oren.key_frame_set import KeyFrameSetConfig
from oren.model import SdfNetworkConfig
from oren.utils.config_abc import ConfigABC
from oren.utils.sampling import SampleRaysConfig


@dataclass
class TrainerConfig(ConfigABC):
    seed: int = 12345
    log_dir: str = "logs"
    exp_name: str = "oren"
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    bound_min: Optional[list[float]] = None
    bound_max: Optional[list[float]] = None
    key_frame_set: KeyFrameSetConfig = field(default_factory=KeyFrameSetConfig)
    model: SdfNetworkConfig = field(default_factory=SdfNetworkConfig)
    criterion: CriterionConfig = field(default_factory=CriterionConfig)
    num_init_frames: int = 3
    init_frame_iterations: int = 10
    num_iterations_per_frame: int = 1
    num_rays_total: int = 20480
    extra_surface_sample: bool = True
    frame_downsample: int = 100
    # Steps between per-iteration loss logging (logger.info + TensorBoard scalars),
    # which force a host sync + disk I/O each frame. 0 disables it entirely.
    log_loss_interval: int = 1
    # torch.compile the SDF model to cut per-kernel launch overhead (the workload is
    # launch-bound: a tiny MLP + many small kernels). compile_mode "default" uses
    # inductor fusion (safe). Avoid "reduce-overhead" (CUDA graphs) here: the octree
    # reassigns its buffers every insert, so captured graphs would replay stale ptrs.
    compile_model: bool = False
    compile_mode: str = "default"
    # Capture the per-frame training step (forward + finite-diff grad + criterion
    # + backward + optimizer.step) into a CUDA graph and replay it each frame.
    # The step is dispatch-bound (~400 tiny ops fighting the GIL under EdgeOS
    # contention); a graph collapses it to one launch. Requires finite_difference
    # grad. Rays are padded to num_rays_total (fixed shape) and masked. Falls back
    # to eager if capture fails.
    cuda_graph_training: bool = False
    # torch float32 matmul precision: "highest" (full fp32, default) | "high" (TF32
    # tensor cores — faster matmuls, slight precision loss) | "medium" (bf16).
    float32_matmul_precision: str = "highest"
    sample_rays: SampleRaysConfig = field(default_factory=SampleRaysConfig)
    batch_size: int = 204800
    lr: float = 0.01
    grad_method: str = "finite_difference"  # autodiff | finite_difference
    finite_difference_eps: float = 0.03
    final_iterations: int = 0  # number of iterations after all frames are processed, 0 means no extra iterations
    final_evaluate: bool = True  # whether to call evaluate() in the cleanup finally
    final_save_model: bool = True  # whether to write final.pth in the cleanup finally
    save_mesh: bool = True  # whether to save the final mesh
    mesh_resolution: float = 0.0125
    mesh_iso_value: float = 0.0
    clean_mesh: bool = True
    save_slice: bool = True
    slice_center: Optional[list] = None  # if None, use the center of the scene bounding box
    ckpt_interval: int = -1  # interval to save checkpoints, -1 means no intermediate checkpoints
    profiling: bool = False
    profiling_verbose: bool = False
    # When profiling, whether the per-stage timers call cuda.synchronize().
    # True  -> accurate GPU per-stage times, but ~10 syncs/frame (slow).
    # False -> per-stage CPU dispatch time via perf_counter, no GPU drains;
    #          cheap enough to attribute wall time at ~production speed. The
    #          right mode for a dispatch-bound workload (GPU mostly idle).
    profiling_sync: bool = True
    # Write one CSV row per processed frame (wall time, per-frame duration,
    # cumulative voxel count) to <log_dir>/frame_timing.csv for the mapping-speed
    # figure. Cheap (no per-stage CUDA syncs); independent of `profiling`.
    log_frame_timing: bool = False
    frozen_model_path: Optional[str] = None
    detect_nan: bool = False  # enable anomaly detection + per-tensor NaN logging (slow)
    grad_clip: float = 0.0  # gradient clipping max-norm; 0 disables
