#!/usr/bin/env python3
"""
grad-SDF Mapping Node for ROS 2
This node subscribes to point cloud and pose topics and performs online SDF mapping
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation as R



# Add grad-sdf to path
grad_sdf_path = str(Path(__file__).resolve().parents[4])
sys.path.insert(0, grad_sdf_path)

from grad_sdf.trainer_config import TrainerConfig
from grad_sdf.trainer import Trainer
from grad_sdf.frame import LiDARFrame


class GradSDFMappingNode(Node):
    """
    ROS 2 Node for online SDF mapping using grad-SDF algorithm
    """

    def __init__(self):
        super().__init__('grad_sdf_mapping_node')

        self.pointcloud_topic = '/os_cloud_node/points'

        # State for auto-evaluate when bag playback ends
        self.last_pc_time = None  # Last received point cloud time (rclpy.time.Time)
        self.evaluation_done = False  # Ensure evaluate runs only once
        # Timeout: if no point cloud is received within this duration, assume bag finished
        self.no_data_timeout_sec = 5.0

        # Load config
        parser = TrainerConfig.get_argparser()
        cfg, _ = parser.parse_known_args()
        self.cfg = cfg
        self.device = self.cfg.device

        # Initialize trainer
        self.get_logger().info('Initializing grad-SDF model...')
        self.trainer = Trainer(self.cfg)

        # Load poses
        pose_path = self.cfg.data.dataset_args['pose_file_path']
        self.load_pose(pose_path)
        self.get_logger().info(f'Loaded {len(self.pose_data)} poses from GT file.')

        # Subscribe to point cloud
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.pc_sub = self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self.pointcloud_callback,
            qos_profile
        )
        self.get_logger().info(f'Subscribed to {self.pointcloud_topic}')

        # State tracking
        self.frame_count = 0
        offset = self.cfg.data.dataset_args['offset']
        self.scene_offset = torch.tensor(offset, device=self.device)
        self.get_logger().info(f'Scene offset: {self.scene_offset}')
        self.get_logger().info('Node initialization complete, waiting for point cloud messages...')

        # Timer: periodically check for missing point cloud and auto-evaluate
        self.no_data_timer = self.create_timer(1.0, self.check_no_data_timeout)

    def load_pose(self, pose_file_path: str):
        df = pd.read_csv(pose_file_path, comment='#', header=None)
        timestamp = df.iloc[:, 0] + df.iloc[:, 1] / 1e9
        arr = np.column_stack([timestamp.values, df.iloc[:, 2:].values])
        self.pose_data = arr

    def find_nearest_pose(self, query_time):
        """Find the nearest pose to the point cloud timestamp and return a 4x4 matrix."""
        times = self.pose_data[:, 0]
        idx = np.searchsorted(times, query_time)

        if idx == 0:
            best_idx = 0
        elif idx >= len(times):
            best_idx = len(times) - 1
        else:
            # Choose the closer one
            if abs(times[idx] - query_time) < abs(times[idx-1] - query_time):
                best_idx = idx
            else:
                best_idx = idx - 1

        # Reject if time difference is too large (e.g., > 0.1s means pose missing)
        if abs(times[best_idx] - query_time) > 0.1:
            return None

        # Extract x, y, z, qx, qy, qz, qw
        p = self.pose_data[best_idx, 1:]

        # Convert to 4x4 matrix
        T = np.eye(4)
        T[:3, 3] = p[0:3] # x, y, z
        rot = R.from_quat(p[3:7]).as_matrix() # qx, qy, qz, qw
        T[:3, :3] = rot

        return torch.tensor(T, dtype=torch.float32, device=self.device)

    def pointcloud_callback(self, msg):
        self.get_logger().info(f'=== CALLBACK TRIGGERED === Message received, data size: {len(msg.data)} bytes')
        try:
            # Update last received point cloud time
            self.last_pc_time = self.get_clock().now()

            pc_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            self.get_logger().info(f'Point cloud timestamp: {pc_time:.6f} (sec={msg.header.stamp.sec}, nanosec={msg.header.stamp.nanosec})')

            # Match pose
            pose_mat = self.find_nearest_pose(pc_time)
            if pose_mat is None:
                self.get_logger().warn(f'No matching pose found for point cloud at time {pc_time:.6f}')
                return

            self.get_logger().info(f'Matched pose successfully')

            points_np = np.frombuffer(
                msg.data,
                dtype=np.float32
            ).reshape(-1, msg.point_step // 4)[:, :3]
            points = torch.tensor(points_np, dtype=torch.float32, device=self.cfg.device)

            self.process_frame(points, pose_mat)
        except Exception as e:
            self.get_logger().error(f'Error in pointcloud_callback: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())


    def process_frame(self, points, pose):
        """
        Process a frame with points and pose using Trainer's methods

        Args:
            points: torch.Tensor of shape (N, 3) - points in sensor frame
            pose: torch.Tensor of shape (4, 4) - pose transformation matrix
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
        frame.apply_bound(self.cfg.data.dataset_args['bound_min'], self.cfg.data.dataset_args['bound_max'])

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

    node = GradSDFMappingNode()

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
