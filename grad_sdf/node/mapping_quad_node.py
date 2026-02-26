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
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation as R
import open3d as o3d



# Add grad-sdf to path
grad_sdf_path = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, grad_sdf_path)

from grad_sdf.trainer_config import TrainerConfig
from grad_sdf.trainer_ros import Trainer_ros
from grad_sdf.frame import DepthFrame


class GradSDFMappingNode(Node):
    """
    ROS 2 Node for online SDF mapping using grad-SDF algorithm
    """

    def __init__(self):
        super().__init__('grad_sdf_mapping_node')

        self.depth_topic = '/quad/depth_img'
        self.pose_topic = '/quad/depth_img_pose'

        # State for auto-evaluate when bag playback ends
        self.last_pc_time = None  # Last received point cloud time (rclpy.time.Time)
        self.evaluation_done = False  # Ensure evaluate runs only once
        # Timeout: if no point cloud is received within this duration, assume bag finished
        self.no_data_timeout_sec = 100.0

        # Load config
        parser = TrainerConfig.get_argparser()
        cfg, _ = parser.parse_known_args()
        self.cfg = cfg
        self.device = self.cfg.device

        # From provided camera info:
        # width: 212, height: 120
        # distortion_model: '', d: [] (no distortion)
        self.cam_intrinsic = torch.tensor(
            [
                [429.47845458984375, 0.0, 429.2593078613281],
                [0.0, 429.47845458984375, 243.30862426757812],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.bound_min = torch.tensor(cfg.data.dataset_args['bound_min'])
        self.bound_max = torch.tensor(cfg.data.dataset_args['bound_max'])
        self.scene_offset = torch.tensor(cfg.data.dataset_args['offset'])

        # Initialize trainer
        self.get_logger().info('Initializing grad-SDF model...')
        self.trainer = Trainer_ros(self.cfg)

        # Subscribe to point cloud
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile
        )
        self.get_logger().info(f'Subscribed to {self.depth_topic}')

        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            qos_profile
        )
        self.get_logger().info(f'Subscribed to {self.pose_topic}')

        # State tracking
        self.frame_count = 0
        self.get_logger().info(f'Scene offset: {self.scene_offset}')

        # Synchronization state
        self.latest_depth = None
        self.latest_pose = None
        self.latest_depth_time = None
        self.latest_pose_time = None
        self.sync_tolerance = 0.01  # 50ms tolerance for synchronization

        self.get_logger().info('Node initialization complete, waiting for point cloud messages...')

        # Timer: periodically check for missing point cloud and auto-evaluate
        self.no_data_timer = self.create_timer(1.0, self.check_no_data_timeout)

        self.points_trained = torch.tensor([], device='cpu')

    def depth_callback(self, msg: Image):
        self.get_logger().info(f'=== DEPTH CALLBACK === Message received, data size: {len(msg.data)} bytes')
        try:
            # Update last received point cloud time
            self.last_pc_time = self.get_clock().now()

            depth_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            # if depth_time > 158:
            #     self.get_logger().info(f'Depth image timestamp: {depth_time:.6f} is greater than 158')
            #     return
            self.get_logger().info(f'Depth image timestamp: {depth_time:.6f}')

            # Convert ROS Image to numpy array
            # Assuming depth image is 32FC1 (float32, single channel)
            depth_np = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)

            # Convert to torch tensor
            depth_tensor = torch.from_numpy(depth_np)

            # Store latest depth
            self.latest_depth = depth_tensor
            self.latest_depth_time = depth_time

            # self.get_logger().info(f'Depth image shape: {depth_tensor.shape}, range: [{depth_tensor.min():.2f}, {depth_tensor.max():.2f}]')

            # Try to process frame if pose is available
            self.try_process_synced_frame()

        except Exception as e:
            self.get_logger().error(f'Error in depth_callback: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def pose_callback(self, msg: PoseStamped):
        self.get_logger().info(f'=== POSE CALLBACK === Message received')
        try:
            # Update last received pose time
            pose_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            # if pose_time > 158:
            #     self.get_logger().info(f'Pose timestamp: {pose_time:.6f} is greater than 158')
            #     return
            self.get_logger().info(f'Pose timestamp: {pose_time:.6f}')

            # Extract position
            position = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])

            # Extract quaternion and convert to rotation matrix
            quat = np.array([
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ])
            rotation = R.from_quat(quat).as_matrix()

            # Build 4x4 transformation matrix
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = rotation
            pose_matrix[:3, 3] = position

            # Convert to torch tensor
            pose_tensor = torch.from_numpy(pose_matrix).float()

            # Store latest pose
            self.latest_pose = pose_tensor
            self.latest_pose_time = pose_time

            self.get_logger().info(f'Pose position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]')

            # Try to process frame if depth is available
            self.try_process_synced_frame()

        except Exception as e:
            self.get_logger().error(f'Error in pose_callback: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def try_process_synced_frame(self):
        """
        Try to process a frame if we have both depth and pose that are synchronized
        """
        if self.latest_depth is None or self.latest_pose is None:
            return

        if self.latest_depth_time is None or self.latest_pose_time is None:
            return

        # Check if timestamps are close enough
        time_diff = abs(self.latest_depth_time - self.latest_pose_time)
        if time_diff > self.sync_tolerance:
            self.get_logger().warn(f'Depth and pose timestamps differ by {time_diff:.3f}s, skipping frame')
            return

        self.get_logger().info(f'Processing synced frame (time diff: {time_diff:.4f}s)')

        # Process the frame
        self.process_frame(self.latest_depth, self.latest_pose)

        # Clear the buffers to avoid reprocessing
        self.latest_depth = None
        self.latest_pose = None


    def process_frame(self, depth_map, pose):
        """
        Process a frame with points and pose using Trainer's methods

        Args:
            depth_map: torch.Tensor of shape (H, W) - depth map
            pose: torch.Tensor of shape (4, 4) - pose transformation matrix
        """
        self.frame_count += 1
        self.get_logger().info(f'Processing frame {self.frame_count}...')
        # Check pose position is in bounds for each coordinate. Avoid ambiguous tensor comparison.
        pos_vec = pose[:3, 3]
        out_of_bound = ((pos_vec < self.bound_min) | (pos_vec > self.bound_max)).any()
        if out_of_bound:
            self.get_logger().warn(f'Pose position: {pos_vec} is out of bound, skipping frame {self.frame_count}')
            return


        # Create LiDARFrame with points in sensor frame and pose
        frame = DepthFrame(
            fid=self.frame_count,
            depth=depth_map,
            intrinsic=self.cam_intrinsic,
            offset=self.scene_offset,
            ref_pose=pose,
            max_depth=self.cfg.data.dataset_args['max_depth'],
            min_depth=self.cfg.data.dataset_args['min_depth'],
            device=self.cfg.device,
        )
        with self.trainer.timer_apply_bound:
            frame.project_to_bound(self.bound_min, self.bound_max)

        # Get points in world frame
        points_world = frame.get_points(to_world_frame=True, device=self.cfg.device)
        if points_world.numel() == 0 or points_world.shape[0] == 0:
            self.get_logger().warn(
                f'Frame {self.frame_count}: no valid points after filtering/bounds; skipping octree insert/train.'
            )
            return

        self.get_logger().info(
            f'points_min: {points_world.min(dim=0).values}, points_max: {points_world.max(dim=0).values}'
        )

        # Insert points into octree (using Trainer's method)
        _, seen_voxels = self.trainer.insert_points_to_octree(points_world)

        if seen_voxels.numel() != 0 or seen_voxels.shape[0] != 0:
            # Update key frame set (using Trainer's method)
            is_key_frame = self.trainer.update_key_frame_set(frame, seen_voxels)
            if is_key_frame:
                self.get_logger().info(f'Frame {self.frame_count} is selected as a key frame.')
            self.trainer.train_with_frame(frame)
            self.trainer.epoch += 1

            self.points_trained = torch.cat((self.points_trained, points_world[::100, :].to('cpu')), dim=0)
        else:
            self.get_logger().warn(f'Frame {self.frame_count}: no valid voxels after insertion; skipping train.')


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
                # Save the accumulated points as a ply file
                points_np = self.points_trained.numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_np)
                self.trainer.logger.log_point_cloud(pcd, "points_trained.ply")
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
        # Save the accumulated points as a ply file
        points_np = node.points_trained.numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        node.trainer.logger.log_point_cloud(pcd, "points_trained.ply")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
