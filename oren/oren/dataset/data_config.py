from dataclasses import dataclass, field
from typing import Optional

from oren.utils.config_abc import ConfigABC


@dataclass
class DataConfig(ConfigABC):
    dataset_name: str = "newer_college"
    dataset_args: dict = field(
        default_factory=lambda: {
            "data_path": "data/newer_college-lidar",
            "max_depth": -1.0,
        }
    )
    apply_bound: bool = False   # whether to apply bounding box cropping to the input data
    bound_min: Optional[list[float]] = None
    bound_max: Optional[list[float]] = None
    start_frame: int = 0
    end_frame: int = -1

    def __post_init__(self):
        super().__post_init__()
        # apply_bound, bound_min, bound_max are common config for dataset processing, we put
        # them in the top-level config and also pass them to dataset_args for backward
        # compatibility.

        self.dataset_args["apply_bound"] = self.apply_bound
        self.dataset_args["bound_min"] = self.bound_min
        self.dataset_args["bound_max"] = self.bound_max
