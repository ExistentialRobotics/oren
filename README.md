<div align="center">

# OREN: Octree Residual Network for Real-Time Euclidean Signed Distance Mapping

[![arXiv](https://img.shields.io/badge/arXiv-2510.18999-b31b1b.svg)](https://arxiv.org/abs/2510.18999)
[![License](https://img.shields.io/badge/license-see%20LICENSE-blue.svg)](LICENSE)

![OREN demo](assets/oren.gif)

</div>

**OREN** is a hybrid SDF reconstruction framework that combines gradient-augmented octree interpolation with an implicit neural residual to achieve efficient, continuous, non-truncated, and highly accurate Euclidean SDF mapping.

This repository provides:

- A trainer for the **Replica** and **Newer College** datasets.
- A **ROS 2** mapping node for live integration with rosbags and sensor streams.
- An interactive **GUI** for visualizing the SDF slice, octree structure, and camera poses during training.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [ROS 2 Mapping Node](#ros-2-mapping-node)
- [Docker](#docker)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Installation

### Prerequisites

- Ubuntu 24.04 (tested) or Arch Linux
- Python 3.12 (3.10 / 3.11 should also work)
- CUDA (tested with CUDA 12.8/13.2 and PyTorch 2.8.0/2.11.0)

### 1. Clone the repository

```bash
git clone --recursive https://github.com/ExistentialRobotics/oren.git
cd oren
```

### 2. Install system dependencies

**Ubuntu**

```bash
sudo apt install \
    cmake g++ ccache git \
    libeigen3-dev libyaml-cpp-dev libabsl-dev \
    python3-dev python3-pip pybind11-dev
```

**Arch Linux**

```bash
sudo pacman -S --needed \
    cmake gcc ccache git \
    eigen yaml-cpp abseil-cpp \
    python python-pip pybind11
```

### 3. Set up the Python environment

```bash
pip install pipenv  # or: sudo apt install pipenv
pipenv install
pipenv shell --verbose
```

If you prefer a different virtual-environment tool, you can install the dependencies listed in `Pipfile` / `requirements.txt` directly.

### 4. Build the bundled extensions

```bash
cd deps/pytorch3d
pip install --no-build-isolation --verbose .
cd ../..

cd deps/erl_geometry
pip install --no-build-isolation --verbose .
cd ../..
```
### 5. Install Oren
```bash
cd oren
pip install -e .
cd ..
```

> **Arch Linux note.** When building `pytorch3d` and `erl_geometry`, you may need to pin
> the toolchain to CUDA-compatible version, e.g. `gcc-14` / `g++-14`:
>
> ```bash
> CC=gcc-14 CXX=g++-14 NVCC_PREPEND_FLAGS='-ccbin g++-14' \
>     pip install --no-build-isolation --verbose .
> ```

## Dataset Preparation

### Replica

**0. Download the replica dataset** without RGBD data from [Hugging Face](https://huggingface.co/datasets/erl-ucsd/oren-datasets/resolve/main/replica.tar.gz).

**1. Rotate meshes and trajectories** so they align with the octree axes
([`replica_obb_rotation.py`](oren/oren/dataset/replica_obb_rotation.py)):

```bash
python oren/oren/dataset/replica_obb_rotation.py \
    --dataset-dir <replica_dataset_dir> \
    --output-dir <replica_preprocessed_path>
```

**2. Copy the camera parameters** into the preprocessed folder:

```bash
cp <replica_dataset_dir>/cam_params.json <replica_preprocessed_path>/cam_params.json
```

**3. (Optional) Augment with virtual upward-looking views**
([`replica_augment_views.py`](oren/oren/dataset/replica_augment_views.py)) to improve spatial coverage:

```bash
DATA_DIR=<replica_preprocessed_path> ./scripts/replica_augment_views.bash
```

### Newer College

We test OREN on the `quad-easy` sequence of the Newer College dataset. The original data is in ROS1 bag format and contains LiDAR motion distortion, which impacts the SDF learning quality.
We preprocess the data with our modified [DLIO](https://github.com/ExistentialRobotics/direct-lidar-inertial-odometry) and provide the processed data in a format compatible with our trainer. You can download it from [Hugging Face](https://huggingface.co/datasets/erl-ucsd/oren-datasets/resolve/main/newer-college-quad-rotated.tar.gz).

ROS2 bag format is also available for the same sequence, which can be directly used with our ROS2 mapping node. You can also download it from [Hugging Face](https://huggingface.co/datasets/erl-ucsd/oren-datasets/resolve/main/newer-college-quad-dlio-bag.tar.gz).

Newer College workflows are driven by `configs/trainer-newer_college.yaml`
and the rosbag launch file described in [ROS 2 Mapping Node](#ros-2-mapping-node).

## Training

### Replica

```bash
python oren/oren/trainer.py --config configs/replica.yaml
```

### GUI Trainer

The GUI trainer enables interactive visualization and monitoring of training, including the SDF slice, octree structure, camera poses, and more. It can be used with the same training configuration as the standard trainer, with additional options for GUI settings.

```bash
python oren/oren/gui_trainer.py \
    --gui-config configs/gui.yaml \
    --trainer-config configs/replica.yaml \
    --gt-mesh-path <replica_preprocessed_path>/room0_mesh.ply \
    --copy-scene-bound-to-gui
```

## ROS 2 Mapping Node

<div align="center">

![ROS2 Demo](assets/ros2-newer-college.gif)

</div>

The `oren_ros` package provides three executables:

- `mapping_node` — subscribes to LiDAR / depth + (optional) odometry, runs the trainer online.
- `sdf_query_node` — samples the learned SDF on a 2D grid and publishes it as a `grid_map_msgs/GridMap` (and optionally `sensor_msgs/PointCloud2`).
- `clock_node` — replays `/clock` from the bag's TF stamps so `use_sim_time` consumers stay in sync.

### Build and source

```bash
# from the repo root
colcon build --verbose --symlink-install
source install/setup.bash
```

### Launch with a rosbag

The single entrypoint is `mapping.launch.py`. It starts `mapping_node`, `sdf_query_node`, `clock_node`, and (optionally) RViz, then runs `ros2 bag play` after `bag_delay` seconds.

```bash
ros2 launch oren_ros mapping.launch.py \
    bag_path:=<newer_college_bag_path>
```

The default `trainer_config_path` resolves to the installed
`share/oren_ros/configs/trainer-ros.yaml` — override it to point at your own
trainer YAML (e.g. `trainer-ros-newer-college.yaml` shipped with the package):

```bash
ros2 launch oren_ros mapping.launch.py \
    bag_path:=<newer_college_bag_path> \
    trainer_config_path:=$(ros2 pkg prefix oren_ros)/share/oren_ros/configs/trainer-ros-newer-college.yaml \
    play_rate:=1.0 \
    bag_delay:=1.0 \
    rviz:=true \
    rviz_config:=$(ros2 pkg prefix oren_ros)/share/oren_ros/rviz/lidar.rviz
```

Launch arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `bag_path` | `ros2_bag` | Directory passed to `ros2 bag play`. |
| `trainer_config_path` | `share/oren_ros/configs/trainer-ros.yaml` | Trainer YAML consumed by `mapping_node`. |
| `play_rate` | `1.0` | `ros2 bag play -r` rate. |
| `bag_delay` | `1.0` | Seconds to wait before starting bag playback. |
| `use_sim_time` | `true` | Drives every node's clock from `/clock`. |
| `visualize_sdf` | `true` | Spawns `sdf_query_node` for live GridMap output. |
| `rviz` | `false` | Launches RViz alongside the mapping nodes. |
| `rviz_config` | _(empty)_ | Path to a `.rviz` file; empty opens RViz with its default layout. |

The launch file remaps `/robot/tf` → `/tf` and `/robot/tf_static` → `/tf_static` for the Newer College bag so `tf2_ros` sees the DLIO frames. Adjust this in `launch/mapping.launch.py` if your bag publishes TF on the default topics.

### Configuring topics, modality, and QoS

ROS-side parameters (sensor modality, topic names, QoS, sync tolerances, the SDF query grid) live in `oren_ros/configs/ros2-params.yaml` and are loaded automatically by the launch file. The most common knobs:

- `modality`: `lidar` or `depth`.
- `use_odom`: `true` syncs the sensor topic with `odom_topic` via `message_filters`; `false` looks up the sensor pose in the tf2 buffer at the message stamp.
- `lidar_topic` / `depth_topic` / `odom_topic`: full QoS profile per topic (preset + per-field overrides).
- `world_frame` / `sensor_frame`: tf2 lookup frames when `use_odom=false`.
- `sdf_query_node.*`: 2D grid resolution, query plane height, attached frame, publish rate, and which outputs (`publish_grid_map`, `publish_point_cloud`, `publish_gradient`) to enable.

Override individual parameters at launch time without editing the YAML:

```bash
ros2 launch oren_ros mapping.launch.py \
    bag_path:=<bag> \
    --ros-args -p oren_mapping_node.modality:=depth \
              -p sdf_query_node.resolution:=0.1
```

## Docker

### 1. Build the image

From the repo root:

```bash
./docker/build.bash
```

This produces the image `erl/oren:24.04`.

### 2. Run the container

The following starts a container with GPU, X11 display, and device access enabled:

```bash
docker run --privileged --restart always -t \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME:$HOME:rw \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    --gpus all \
    --runtime=nvidia \
    -e DISPLAY \
    --net=host \
    --detach \
    --hostname container-oren \
    --add-host=container-oren:127.0.0.1 \
    --name oren \
    erl/oren:24.04 \
    bash -l
```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{dai2026oren,
    title={OREN: Octree Residual Network for Real-Time Euclidean Signed Distance Mapping},
    author={Zhirui Dai and Qihao Qian and Tianxing Fan and Nikolay Atanasov},
    year={2026},
    eprint={2510.18999},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    url={https://arxiv.org/abs/2510.18999},
}
```

## Acknowledgement

- Our key-frame selection strategy builds on
  [H2-Mapping](https://github.com/Robotics-STAR-Lab/H2-Mapping).
- Our GUI is built with [Open3D](http://www.open3d.org/), drawing inspiration from
  [PIN-SLAM](https://github.com/PRBonn/PIN_SLAM).
