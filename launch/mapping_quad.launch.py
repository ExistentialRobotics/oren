from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share_root = Path(get_package_share_directory("grad_sdf"))
    default_config_path = share_root / "configs" / "v2" / "quad-ros.yaml"
    if not default_config_path.exists():
        default_config_path = Path.cwd() / "configs" / "v2" / "quad-ros.yaml"

    config_path_arg = DeclareLaunchArgument(
        "config_path",
        default_value=default_config_path.as_posix(),
        description="Trainer config YAML path.",
    )

    config_path = LaunchConfiguration("config_path")

    mapping_node = Node(
        package="grad_sdf",
        executable="mapping_quad_node",
        name="grad_sdf_mapping_quad_node",
        output="screen",
        arguments=["--config", config_path],
    )

    return LaunchDescription(
        [
            config_path_arg,
            LogInfo(
                msg=[
                    "Starting grad_sdf mapping_quad_node with config: ",
                    config_path,
                ]
            ),
            mapping_node,
        ]
    )
