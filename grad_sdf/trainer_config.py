from dataclasses import dataclass, field
from typing import Optional

from grad_sdf.criterion import CriterionConfig
from grad_sdf.dataset.data_config import DataConfig
from grad_sdf.key_frame_set import KeyFrameSetConfig
from grad_sdf.model import SdfNetworkConfig
from grad_sdf.node.ros_confg import RosConfig
from grad_sdf.utils.config_abc import ConfigABC
from grad_sdf.utils.sampling import SampleRaysConfig


@dataclass
class TrainerConfig(ConfigABC):
    seed: int = 12345
    log_dir: str = "logs"
    exp_name: str = "grad_sdf"
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    key_frame_set: KeyFrameSetConfig = field(default_factory=KeyFrameSetConfig)
    model: SdfNetworkConfig = field(default_factory=SdfNetworkConfig)
    criterion: CriterionConfig = field(default_factory=CriterionConfig)
    num_init_frames: int = 3
    init_frame_iterations: int = 10
    num_iterations_per_frame: int = 1
    num_rays_total: int = 20480
    extra_surface_sample: bool = True
    frame_downsample: int = 100
    sample_rays: SampleRaysConfig = field(default_factory=SampleRaysConfig)
    batch_size: int = 204800
    lr: float = 0.01
    grad_method: str = "finite_difference"  # autodiff | finite_difference
    finite_difference_eps: float = 0.03
    final_iterations: int = (
        0  # number of iterations after all frames are processed, 0 means no extra iterations
    )
    save_mesh: bool = True  # whether to save the final mesh
    mesh_resolution: float = 0.0125
    mesh_iso_value: float = 0.0
    clean_mesh: bool = True
    save_slice: bool = True
    slice_center: Optional[list] = (
        None  # if None, use the center of the scene bounding box
    )
    ckpt_interval: int = (
        -1
    )  # interval to save checkpoints, -1 means no intermediate checkpoints
    profiling: bool = False
    profiling_verbose: bool = False
    frozen_model_path: Optional[str] = None
    ros: RosConfig = field(default_factory=RosConfig)

