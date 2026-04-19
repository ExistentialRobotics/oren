$\nabla$-SDF: Learning Euclidean Signed Distance Functions Online with Gradient-Augmented Octree Interpolation and Neural Residual
=============

<p align="center">
<a href="https://github.com/ExistentialRobotics/grad-sdf/releases"><img src="https://img.shields.io/github/v/release/ExistentialRobotics/grad-sdf?label=version" /></a>
<a href="https://github.com/ExistentialRobotics/grad-sdf?tab=readme-ov-file#run-nabla-sdf"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
<a href="https://github.com/ExistentialRobotics/grad-sdf?tab=readme-ov-file#installation"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
<!-- <a href="https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/pan2024tro.pdf"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a> -->
<a href="https://github.com/ExistentialRobotics/grad-sdf/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>
</p>

This repository contains the code for the paper: **$\nabla$-SDF: Learning Euclidean Signed Distance Functions Online with Gradient-Augmented Octree Interpolation and Neural Residual**.

This branch targets running the grad-SDF mapping as a ROS 2 node.

$\nabla$-SDF is a hybrid SDF reconstruction framework that combines gradient-augmented octree interpolation with an implicit neural residual to achieve efficient, continuous non-truncated, and highly accurate Euclidean SDF mapping..

<div align="center">
  <img src="assets/grad-sdf.gif" width="600" alt="SDF Mapping Demo">
</div>

## Installation

### Prerequisites

- Ubuntu (24.04 tested) / Arch Linux
- Python 3.12 (3.10, 3.11 should also work)
- CUDA (tested with CUDA 12.8 and PyTorch 2.8.0)

### Steps

0. Clone the repository

    ```bash
    git clone --recursive https://github.com/ExistentialRobotics/grad-sdf.git
    cd grad-sdf
    ```

1. Setup pipenv environment

    ```bash
    pip install pipenv  # or sudo apt install pipenv
    pipenv install
    pipenv shell --verbose
    ```
    If you use other virtual environment tools, you can also install the dependencies by
    ```bash
    pip install -r requirements.txt
    ```

2. Install system dependencies
    - For Ubuntu
    ```bash
    sudo apt install \
        cmake \
        g++ \
        ccache \
        git \
        libeigen3-dev \
        libyaml-cpp-dev \
        libabsl-dev \
        python3-dev \
        python3-pip \
        pybind11-dev
    ```
    - For Arch Linux
    ```bash
    sudo pacman -S --needed \
        cmake \
        gcc \
        ccache \
        git \
        eigen \
        yaml-cpp \
        abseil-cpp \
        python \
        python-pip \
        pybind11
    ```

3. Install other dependencies

    ```bash
    pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@stable

    cd deps/tinycudann
    cmake . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j`nproc`
    cd bindings/torch
    python setup.py install
    cd ../../../..

    cd deps/sparse_octree
    python setup.py install
    cd ../..

    cd deps/erl_geometry
    pip install --no-build-isolation --verbose .
    cd ../..
    # for Arch Linux
    # CXX=/usr/bin/g++-14 pip install --no-build-isolation --verbose .
    ```

## ROS 2 (this branch)

### Build and install

```bash
# from repo root
colcon build --packages-select grad_sdf
source install/setup.bash
```

### Launch mapping node with rosbag

```bash
ros2 launch grad_sdf mapping_with_bag.launch.py \
  config_path:=/home/qihao/workplace/grad-sdf/configs/v2/quad-ros.yaml
```

Optional arguments:
```bash
ros2 launch grad_sdf mapping_with_bag.launch.py \
  bag_path:=/home/qihao/workplace/grad-sdf/data/newercollege-ros2 \
  config_path:=/home/qihao/workplace/grad-sdf/configs/v2/trainer.yaml \
  play_rate:=1.0 \
  bag_delay:=1.0
```


<!-- ## Docker

### 1. Build the image
First, build the Docker image (make sure you are in the project root):


Use the following command to start a container with GPU, X11 display, and device access enabled:
```bash
./docker/build.bash
```
This script will create the Docker image `erl/grad_sdf:24.04`.

### 2. Run the container
Use the following command to start a container with GPU, X11 display, and device access enabled:
```bash
docker run --privileged --restart always -t \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME:$HOME:rw \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    --workdir /workspace \
    --gpus all \
    --runtime=nvidia \
    -e DISPLAY \
    --net=host \
    --detach \
    --hostname container-grad_sdf \
    --add-host=container-grad_sdf:127.0.0.1 \
    --name grad_sdf \
    erl/grad_sdf:24.04 \
    bash -l
``` -->

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{dai2025nablasdf,
      title={{$\nabla$-SDF: Learning Euclidean Signed Distance Functions Online with Gradient-Augmented Octree Interpolation and Neural Residual}},
      author={Zhirui Dai and Qihao Qian and Tianxing Fan and Nikolay Atanasov},
      year={2025},
      eprint={2510.18999},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.18999},
}
```

## Acknowledgement

- We develop our key frame selection strategy based on [H2-Mapping](https://github.com/Robotics-STAR-Lab/H2-Mapping).
- We create the GUI based on [Open3D](http://www.open3d.org/) with inspirations from [PIN-SLAM](https://github.com/PRBonn/PIN_SLAM).
