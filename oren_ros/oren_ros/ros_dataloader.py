"""Streaming `DataLoader` that pulls (depth or LiDAR) frames + poses from ROS 2 topics.

Constructed by `RosTrainer` with a shared `rclpy.node.Node`; subscriptions and the no-data watchdog timer are
registered on that node. The trainer interacts with this loader through the same duck-typed interface as `replica.DataLoader` / `newer_college.DataLoader`:

- `bound_min`, `bound_max` (torch.Tensor) — for `Trainer.__init__` bound auto-set.
- `streaming = True` class attribute signals an unbounded streaming source (Python's built-in
    len() forbids negative returns, so a length-based sentinel isn't usable).
- `__getitem__(idx)` blocks until a synchronized frame is available, or returns `None` either when shutdown has been
requested OR when `idle_block_sec` elapsed without a new frame (so the trainer can keep optimizing existing keyframes
during sensor pauses). The trainer distinguishes the two cases via the `is_shutdown` property.

Pose source is selected by the `use_odom` parameter:
- `use_odom=True`: subscribe to `odom_topic` (nav_msgs/Odometry or geometry_msgs/PoseStamped per `odom_msg_type`) and
    synchronize with the sensor topic via `message_filters.ApproximateTimeSynchronizer` (slop = `sync_tolerance`).
- `use_odom=False`: look up the sensor pose at the message stamp from a `tf2_ros.Buffer` (target=`world_frame`,
    source=`sensor_frame` or the incoming `header.frame_id` if that param is empty).

`on_idle` is an optional callback invoked while `__getitem__` is blocked; the trainer wires it up to drain queued
service requests so they don't starve when no frames are arriving.
"""

import queue
import threading
import time
from typing import Callable, Optional

import message_filters
import numpy as np
import tf2_ros
import torch
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, PointCloud2

from oren.frame import DepthFrame, Frame, LiDARFrame

from oren_ros.ros_topic_param import RosTopicParam


class RosDataLoader:
    """Streaming data source for `Trainer`. See module docstring."""

    streaming = True

    def __init__(
        self,
        ros_node: Node,
        apply_bound: bool = False,
        bound_min: Optional[list[float]] = None,
        bound_max: Optional[list[float]] = None,
        intrinsics_fx: Optional[float] = None,
        intrinsics_fy: Optional[float] = None,
        intrinsics_cx: Optional[float] = None,
        intrinsics_cy: Optional[float] = None,
        min_depth: float = 0.0,
        max_depth: float = -1.0,
    ):
        self.node = ros_node
        self.apply_bound = apply_bound
        self.bound_min = bound_min
        self.bound_max = bound_max
        self.intrinsics = None
        self.min_depth = min_depth
        self.max_depth = max_depth

        if self.bound_min is not None:
            self.bound_min = torch.tensor(self.bound_min, dtype=torch.float32)
        if self.bound_max is not None:
            self.bound_max = torch.tensor(self.bound_max, dtype=torch.float32)

        if self.apply_bound:
            assert self.bound_min is not None, "bound_min is required when apply_bound is True"
            assert self.bound_max is not None, "bound_max is required when apply_bound is True"

        # Topic configuration is purely ROS-param driven. Defaults live on the
        # RosTopicParam dataclass; declare_ros_params seeds `<topic>.path` and
        # `<topic>.qos_*` from those defaults so users override at launch time
        # via `--ros-args -p depth_topic.path:=/foo`.
        self._depth_topic = RosTopicParam(path="/camera/depth")
        self._lidar_topic = RosTopicParam(path="/lidar/points")
        self._odom_topic = RosTopicParam(path="/odom")

        # ROS interface params (declared here, where they're used). Topic-name
        # / QoS params are declared by RosTopicParam in the modality branches
        # below — only the active topics are exposed.
        ros_node.declare_parameter("modality", "depth")  # "depth" | "lidar"
        ros_node.declare_parameter("depth_scale", 0.001)  # applied in _image_to_depth_tensor
        ros_node.declare_parameter("use_odom", True)
        ros_node.declare_parameter("odom_msg_type", "odometry")  # "odometry" | "pose_stamped"
        ros_node.declare_parameter("world_frame", "map")
        # Empty -> use the incoming sensor message's header.frame_id.
        ros_node.declare_parameter("sensor_frame", "")
        ros_node.declare_parameter("tf_lookup_timeout_sec", 0.1)
        ros_node.declare_parameter("sync_tolerance", 0.01)
        # If <= 0, wait indefinitely for frames.
        ros_node.declare_parameter("no_data_timeout_sec", 5.0)
        ros_node.declare_parameter("frame_queue_maxsize", 32)
        # If > 0, __getitem__ returns None after this many seconds of no frame
        # so the trainer can run train_with_frame(None) on existing keyframes.
        # 0 = block until a frame arrives or shutdown.
        ros_node.declare_parameter("idle_block_sec", 0.0)

        self._modality = ros_node.get_parameter("modality").value
        self._depth_scale = float(ros_node.get_parameter("depth_scale").value)
        self._use_odom = bool(ros_node.get_parameter("use_odom").value)
        self._sync_tol = float(ros_node.get_parameter("sync_tolerance").value)
        self._tf_timeout = float(ros_node.get_parameter("tf_lookup_timeout_sec").value)
        self._no_data_timeout = float(ros_node.get_parameter("no_data_timeout_sec").value)
        self._idle_block_sec = float(ros_node.get_parameter("idle_block_sec").value)
        self._world_frame = ros_node.get_parameter("world_frame").value
        self._sensor_frame = ros_node.get_parameter("sensor_frame").value
        maxsize = int(ros_node.get_parameter("frame_queue_maxsize").value)

        odom_type_str = ros_node.get_parameter("odom_msg_type").value
        if odom_type_str == "odometry":
            self._odom_msg_type = Odometry
        elif odom_type_str == "pose_stamped":
            self._odom_msg_type = PoseStamped
        else:
            raise ValueError(f"unknown odom_msg_type: {odom_type_str!r} (expected 'odometry' or 'pose_stamped')")

        # state
        self._frame_queue: queue.Queue[Frame] = queue.Queue(maxsize=maxsize)
        self._shutdown = threading.Event()
        self._frame_count = 0
        self._last_msg_time: Optional[float] = None  # wall-clock seconds
        self._lock = threading.Lock()

        # idle callback: invoked while __getitem__ is waiting for a frame
        self.on_idle: Optional[Callable[[], None]] = None

        # Subscribers + watchdog share a mutually-exclusive group so producer-side
        # callbacks (sync or tf paths) don't race the watchdog.
        self._sub_cbg = MutuallyExclusiveCallbackGroup()

        # tf path setup (only when not relying on a pose topic)
        self._tf_buffer: Optional[tf2_ros.Buffer] = None
        self._tf_listener: Optional[tf2_ros.TransformListener] = None
        if not self._use_odom:
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, ros_node)

        # Hold message_filters refs so they're not garbage-collected.
        self._mf_refs: tuple = ()

        if self._modality == "depth":
            assert (
                intrinsics_fx is not None
                and intrinsics_fy is not None
                and intrinsics_cx is not None
                and intrinsics_cy is not None
            ), "intrinsics_fx, _fy, _cx, _cy are required for depth modality"
            self.intrinsics = torch.tensor(
                [
                    [intrinsics_fx, 0.0, intrinsics_cx],
                    [0.0, intrinsics_fy, intrinsics_cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
            self._resolve_topic_param(self._depth_topic, "depth_topic")
            if self._use_odom:
                self._resolve_topic_param(self._odom_topic, "odom_topic")
                self._setup_synced(Image, self._depth_topic, self._odom_topic, self._on_synced_depth)
                ros_node.get_logger().info(
                    f"[RosDataLoader] depth+odom synced: {self._depth_topic.path} "
                    f"<-> {self._odom_topic.path} (slop={self._sync_tol:.3f}s)"
                )
            else:
                ros_node.create_subscription(
                    Image,
                    self._depth_topic.path,
                    self._depth_tf_cb,
                    self._depth_topic.get_qos(),
                    callback_group=self._sub_cbg,
                )
                ros_node.get_logger().info(
                    f"[RosDataLoader] depth+tf: {self._depth_topic.path} "
                    f"(lookup {self._world_frame}<-{self._sensor_frame or 'header.frame_id'})"
                )
        elif self._modality == "lidar":
            self._resolve_topic_param(self._lidar_topic, "lidar_topic")
            if self._use_odom:
                self._resolve_topic_param(self._odom_topic, "odom_topic")
                self._setup_synced(PointCloud2, self._lidar_topic, self._odom_topic, self._on_synced_lidar)
                ros_node.get_logger().info(
                    f"[RosDataLoader] lidar+odom synced: {self._lidar_topic.path} "
                    f"<-> {self._odom_topic.path} (slop={self._sync_tol:.3f}s)"
                )
            else:
                ros_node.create_subscription(
                    PointCloud2,
                    self._lidar_topic.path,
                    self._lidar_tf_cb,
                    self._lidar_topic.get_qos(),
                    callback_group=self._sub_cbg,
                )
                ros_node.get_logger().info(
                    f"[RosDataLoader] lidar+tf: {self._lidar_topic.path} "
                    f"(lookup {self._world_frame}<-{self._sensor_frame or 'header.frame_id'})"
                )
        else:
            raise ValueError(f"unknown modality: {self._modality!r} (expected 'depth' or 'lidar')")

        self._watchdog = None
        if self._no_data_timeout > 0:
            # 1 Hz watchdog
            self._watchdog = ros_node.create_timer(1.0, self._check_no_data_timeout, callback_group=self._sub_cbg)

    def _resolve_topic_param(self, param: RosTopicParam, prefix: str) -> None:
        param.declare_ros_params(self.node, prefix)
        param.update_from_ros_params(self.node, prefix)
        if not param.path:
            raise ValueError(f"ROS topic param {prefix!r} has empty path")

    def _setup_synced(self, sensor_type, sensor_topic: RosTopicParam, odom_topic: RosTopicParam, callback) -> None:
        self._sensor_sub = message_filters.Subscriber(
            self.node,
            sensor_type,
            sensor_topic.path,
            qos_profile=sensor_topic.get_qos(),
            callback_group=self._sub_cbg,
        )
        self._odom_sub = message_filters.Subscriber(
            self.node,
            self._odom_msg_type,
            odom_topic.path,
            qos_profile=odom_topic.get_qos(),
            callback_group=self._sub_cbg,
        )
        sync = message_filters.ApproximateTimeSynchronizer(
            [self._sensor_sub, self._odom_sub],
            queue_size=10,
            slop=self._sync_tol,
        )
        sync.registerCallback(callback)
        self._mf_refs = (self._sensor_sub, self._odom_sub, sync)

    # ------------------------- Trainer-facing interface -------------------------

    def __len__(self) -> int:
        # Built-in len() rejects negative returns. The Trainer uses the
        # `streaming = True` class attribute (above) to detect this loader.
        return 0

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown.is_set()

    def __getitem__(self, idx: int) -> Optional[Frame]:
        """Block until a frame is ready, transient idle expires, or shutdown.

        Returns:
            Frame if one is available.
            None if either (a) `idle_block_sec` elapsed without a new frame, or
            (b) the loader has been shut down. Caller distinguishes the two cases via `is_shutdown`.
        """
        deadline = time.monotonic() + self._idle_block_sec if self._idle_block_sec > 0 else None
        while not self._shutdown.is_set():
            try:
                return self._frame_queue.get(timeout=0.05)
            except queue.Empty:
                if self.on_idle is not None:
                    self.on_idle()
                if deadline is not None and time.monotonic() >= deadline:
                    return None  # transient idle
        return None  # shutdown

    def shutdown(self) -> None:
        self._shutdown.set()

    # -------------------------- Synced callbacks (use_odom=True) -----------------

    def _on_synced_depth(self, img_msg: Image, odom_msg) -> None:
        try:
            self._last_msg_time = time.monotonic()
            depth = _image_to_depth_tensor(img_msg)
            pose = _odom_msg_to_matrix(odom_msg)
            self._enqueue_depth(depth, pose)
        except Exception as e:
            self.node.get_logger().error(f"[RosDataLoader] _on_synced_depth: {e}")

    def _on_synced_lidar(self, cloud_msg: PointCloud2, odom_msg) -> None:
        try:
            self._last_msg_time = time.monotonic()
            points = _cloud_to_xyz_tensor(cloud_msg)
            pose = _odom_msg_to_matrix(odom_msg)
            self._enqueue_lidar(points, pose)
        except Exception as e:
            self.node.get_logger().error(f"[RosDataLoader] _on_synced_lidar: {e}")

    # -------------------------- TF callbacks (use_odom=False) -------------------

    def _depth_tf_cb(self, msg: Image) -> None:
        try:
            self._last_msg_time = time.monotonic()
            pose = self._lookup_pose(msg.header)
            if pose is None:
                return
            depth = _image_to_depth_tensor(msg, self._depth_scale)
            self._enqueue_depth(depth, pose)
        except Exception as e:
            self.node.get_logger().error(f"[RosDataLoader] _depth_tf_cb: {e}")

    def _lidar_tf_cb(self, msg: PointCloud2) -> None:
        try:
            self._last_msg_time = time.monotonic()
            pose = self._lookup_pose(msg.header)
            if pose is None:
                return
            points = _cloud_to_xyz_tensor(msg)
            self._enqueue_lidar(points, pose)
        except Exception as e:
            self.node.get_logger().error(f"[RosDataLoader] _lidar_tf_cb: {e}")

    def _lookup_pose(self, header) -> Optional[torch.Tensor]:
        source = self._sensor_frame or header.frame_id
        if not source:
            self.node.get_logger().warn(
                "[RosDataLoader] no source frame (empty header.frame_id and no sensor_frame param)"
            )
            return None
        try:
            tf = self._tf_buffer.lookup_transform(
                target_frame=self._world_frame,
                source_frame=source,
                time=Time.from_msg(header.stamp),
                timeout=Duration(seconds=self._tf_timeout),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ExtrapolationException,
            tf2_ros.ConnectivityException,
        ) as e:
            self.node.get_logger().warn(f"[RosDataLoader] tf lookup {self._world_frame}<-{source} failed: {e}")
            return None
        return _transform_stamped_to_matrix(tf)

    # ------------------------------- internals ----------------------------------

    def _enqueue_depth(self, depth: torch.Tensor, pose: torch.Tensor) -> None:
        with self._lock:
            self._frame_count += 1
            fid = self._frame_count
        frame = DepthFrame(
            fid=fid,
            depth=depth,
            intrinsic=self.intrinsics,
            ref_pose=pose,
            min_depth=self.min_depth if self.min_depth >= 0 else None,
            max_depth=self.max_depth if self.max_depth > 0 else None,
        )
        if self.apply_bound:
            frame.apply_bound(self.bound_min, self.bound_max)
        self._enqueue(frame)

    def _enqueue_lidar(self, points: torch.Tensor, pose: torch.Tensor) -> None:
        with self._lock:
            self._frame_count += 1
            fid = self._frame_count
        frame = LiDARFrame(fid=fid, pointcloud=points, ref_pose=pose)
        if self.apply_bound:
            frame.apply_bound(self.bound_min, self.bound_max)
        self._enqueue(frame)

    def _enqueue(self, frame: Frame) -> None:
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                _ = self._frame_queue.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            self._frame_queue.put_nowait(frame)
            self.node.get_logger().warn("[RosDataLoader] frame queue full; dropped oldest frame")

    def _check_no_data_timeout(self) -> None:
        if self._last_msg_time is None or self._shutdown.is_set():
            return
        elapsed = time.monotonic() - self._last_msg_time
        if elapsed > self._no_data_timeout:
            self.node.get_logger().info(
                f"[RosDataLoader] no data for {elapsed:.1f}s (> {self._no_data_timeout:.1f}s); " "shutting down stream."
            )
            self.shutdown()


# ------------------------------ helpers ---------------------------------------


def _quat_xyz_to_matrix(qx: float, qy: float, qz: float, qw: float, tx: float, ty: float, tz: float) -> torch.Tensor:
    rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot
    T[:3, 3] = [tx, ty, tz]
    return torch.from_numpy(T)


def _odom_msg_to_matrix(msg) -> torch.Tensor:
    if isinstance(msg, Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
    elif isinstance(msg, PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
    else:
        raise TypeError(f"unsupported odom msg type: {type(msg).__name__}")
    return _quat_xyz_to_matrix(q.x, q.y, q.z, q.w, p.x, p.y, p.z)


def _transform_stamped_to_matrix(tf: TransformStamped) -> torch.Tensor:
    t = tf.transform.translation
    q = tf.transform.rotation
    return _quat_xyz_to_matrix(q.x, q.y, q.z, q.w, t.x, t.y, t.z)


_DEPTH_ENCODING_DTYPES = {
    "8UC1": np.uint8,
    "mono8": np.uint8,
    "16UC1": np.uint16,
    "mono16": np.uint16,
    "32FC1": np.float32,
    "64FC1": np.float64,
}


def _image_to_depth_tensor(msg: Image, depth_scale: float) -> torch.Tensor:
    dtype = _DEPTH_ENCODING_DTYPES.get(msg.encoding)
    if dtype is None:
        raise ValueError(
            f"unsupported depth image encoding: {msg.encoding!r} "
            f"(supported: {sorted(_DEPTH_ENCODING_DTYPES)})"
        )
    if msg.is_bigendian and dtype != np.uint8:
        dtype = np.dtype(dtype).newbyteorder(">")
    arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
    # astype(copy=True) gives a writable, native-endian float32 array; scale to metres.
    return torch.from_numpy(arr.astype(np.float32, copy=True)) * depth_scale


def _cloud_to_xyz_tensor(msg: PointCloud2) -> torch.Tensor:
    points_np = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, msg.point_step // 4)[:, :3]
    return torch.from_numpy(points_np.copy())
