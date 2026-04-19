from dataclasses import dataclass

from grad_sdf.utils.config_abc import ConfigABC


@dataclass
class RosConfig(ConfigABC):
    depth_topic: str = '/quad/depth_img'
    pose_topic: str = '/quad/pose'
    max_pose_age_sec: float = 0.1
    no_data_timeout_sec: float = 20.0