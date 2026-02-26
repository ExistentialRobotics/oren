#!/usr/bin/env python3
"""
grad-SDF Mapping Node for ROS 2 (Simulation)
This node subscribes to point cloud and odometry topics and performs online SDF mapping
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation as R


# Add grad-sdf to path
grad_sdf_path = str(Path(__file__).resolve().parents[4])
sys.path.insert(0, grad_sdf_path)

from grad_sdf.trainer_config import TrainerConfig
from grad_sdf.trainer_ros import Trainer_ros
from grad_sdf.frame import LiDARFrame


class GradSDFSimMappingNode(Node):
    """
    ROS 2 Node for online SDF mapping in simulation.
    Uses PointCloud2 and Odometry to build the map.
    """

    def __init__(self):
        super().__init__('grad_sdf_sim_mapping_node')

        # Parameters (topics + timeouts)
        self.declare_parameter('pointcloud_topic', '/a200_0000/sensors/lidar3d_0/points')
        self.declare_parameter('odom_topic', '/a200_0000/platform/odom')
        self.declare_parameter('max_pose_age_sec', 0.1)
        self.declare_parameter('no_data_timeout_sec', 5.0)

        self.pointcloud_topic = self.get_parameter('pointcloud_topic').get_parameter_value().string_value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.max_pose_age_sec = self.get_parameter('max_pose_age_sec').get_parameter_value().double_value
        self.no_data_timeout_sec = self.get_parameter('no_data_timeout_sec').get_parameter_value().double_value

        # State for auto-evaluate when bag playback ends
        self.last_pc_time = None  # rclpy.time.Time of last received point cloud
        self.evaluation_done = False  # Ensure evaluate runs only once

        # Latest odometry pose
        self.latest_pose_time = None  # float seconds
        self.latest_pose_np = None  # 4x4 numpy matrix

        # Load config
        parser = TrainerConfig.get_argparser()
        cfg, _ = parser.parse_known_args()
        self.cfg = cfg
        self.device = self.cfg.device

        # Initialize trainer
        self.get_logger().info('Initializing grad-SDF model...')
        self.trainer = Trainer(self.cfg)

        # Load poses if needed for logging (not used in sim)
        pose_path = self.cfg.data.dataset_args.get('pose_file_path', None)
        if pose_path:
            self.load_pose(pose_path)
            self.get_logger().info(f'Loaded {len(self.pose_data)} poses from GT file.')

        # Subscriptions
        self.pc_sub = self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self.pointcloud_callback,
            qos_profile_sensor_data
        )
        self.get_logger().info(f'Subscribed to pointcloud: {self.pointcloud_topic}')

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            odom_qos
        )
        self.get_logger().info(f'Subscribed to odometry: {self.odom_topic}')

        # State tracking
        self.frame_count = 0
        offset = self.cfg.data.dataset_args['offset']
        self.scene_offset = torch.tensor(offset)
        self.get_logger().info(f'Scene offset: {self.scene_offset}')
        self.get_logger().info('Node initialization complete, waiting for point cloud messages...')

        # Timer: periodically check for missing point cloud and auto-evaluate
        self.no_data_timer = self.create_timer(1.0, self.check_no_data_timeout)

    def load_pose(self, pose_file_path: str):
        df = pd.read_csv(pose_file_path, comment='#', header=None)
        timestamp = df.iloc[:, 0] + df.iloc[:, 1] / 1e9
        arr = np.column_stack([timestamp.values, df.iloc[:, 2:].values])
        self.pose_data = arr

    def odom_callback(self, msg: Odometry):
        try:
            t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = rot
            T[:3, 3] = [p.x, p.y, p.z]
            self.latest_pose_time = t
            self.latest_pose_np = T
        except Exception as e:
            self.get_logger().error(f'Error in odom_callback: {e}')

    def pointcloud_callback(self, msg: PointCloud2):
        self.get_logger().info(f'=== CALLBACK TRIGGERED === Message received, data size: {len(msg.data)} bytes')
        try:
            # Update last received point cloud time
            self.last_pc_time = self.get_clock().now()

            pc_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            self.get_logger().info(
                f'Point cloud timestamp: {pc_time:.6f} '
                f'(sec={msg.header.stamp.sec}, nanosec={msg.header.stamp.nanosec})'
            )

            if self.latest_pose_time is None or self.latest_pose_np is None:
                self.get_logger().warn('No odometry received yet, skipping point cloud.')
                return

            if abs(pc_time - self.latest_pose_time) > self.max_pose_age_sec:
                self.get_logger().warn(
                    f'Odometry too old for point cloud (dt={abs(pc_time - self.latest_pose_time):.3f}s). Skipping.'
                )
                return

            pose_mat = torch.tensor(self.latest_pose_np, dtype=torch.float32)

            points_np = np.frombuffer(
                msg.data,
                dtype=np.float32
            ).reshape(-1, msg.point_step // 4)[:, :3]
            points = torch.tensor(points_np, dtype=torch.float32)
            print(f'robot position: {pose_mat[:3, 3]}')
            self.process_frame(points, pose_mat)
        except Exception as e:
            self.get_logger().error(f'Error in pointcloud_callback: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def process_frame(self, points, pose):
        """
        Process a frame with points and pose using Trainer's methods
        """
        self.frame_count += 1
        self.get_logger().info(f'Processing frame {self.frame_count}...')

        # Create LiDARFrame with points in sensor frame and pose
        frame = LiDARFrame(
            fid=self.frame_count,
            pointcloud=points,
            offset=self.scene_offset,
            ref_pose=pose,
        )
        bound_min = torch.tensor(
            self.cfg.data.dataset_args['bound_min'],
            dtype=torch.float32,
        )
        bound_max = torch.tensor(
            self.cfg.data.dataset_args['bound_max'],
            dtype=torch.float32,
        )
        frame.apply_bound(bound_min, bound_max)

        # Get points in world frame
        points_world = frame.get_points(to_world_frame=True, device=self.cfg.device)

        # Insert points into octree (using Trainer's method)
        _, seen_voxels = self.trainer.insert_points_to_octree(points_world)

        # Update key frame set (using Trainer's method)
        is_key_frame = self.trainer.update_key_frame_set(frame, seen_voxels)
        if is_key_frame:
            self.get_logger().info(f'Frame {self.frame_count} is selected as a key frame.')

        self.trainer.train_with_frame(frame)
        self.trainer.epoch += 1

    def check_no_data_timeout(self):
        """
        Periodically check if no new point cloud has been received for long time.
        If no point cloud is received for longer than self.no_data_timeout_sec and
        at least one frame has been processed, auto-run evaluate and shut down.
        """
        # If no point cloud received yet, or already evaluated, do nothing
        if self.last_pc_time is None or self.evaluation_done:
            return

        now = self.get_clock().now()
        elapsed = (now - self.last_pc_time).nanoseconds / 1e9

        if elapsed > self.no_data_timeout_sec:
            self.get_logger().info(
                f'No point cloud received for {elapsed:.2f} seconds '
                f'(> {self.no_data_timeout_sec}s). Assuming bag playback finished, '
                'running evaluate() and shutting down.'
            )
            try:
                self.trainer.evaluate()
            except Exception as e:
                self.get_logger().error(f'Error during automatic evaluate: {e}')
            finally:
                self.evaluation_done = True
                # Trigger ROS shutdown to end spin (avoid duplicate shutdown)
                if rclpy.ok():
                    rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    node = GradSDFSimMappingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
        # evaluate on shutdown
        node.trainer.evaluate()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
