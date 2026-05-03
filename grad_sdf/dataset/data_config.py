from dataclasses import dataclass, field

from grad_sdf.utils.config_abc import ConfigABC


@dataclass
class DataConfig(ConfigABC):
    dataset_name: str = "newer_college"
    dataset_args: dict = field(
        default_factory=lambda: {
            "data_path": "data/newer_college-lidar",
            "max_depth": -1.0,
        }
    )
    start_frame: int = 0
    end_frame: int = -1
