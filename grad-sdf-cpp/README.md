# grad-sdf-cpp

LibTorch C++ scaffold that mirrors the Python `SdfNetwork` forward path:

- Octree prior lookup + gradient-augmented trilinear interpolation
- Residual MLP inference
- Final SDF prediction (`sdf_prior + sdf_residual`)

## 1) Export checkpoint bundle

Convert Python `state_dict` checkpoint into a tensor bundle that C++ can load deterministically.

```bash
python tools/export_cpp_bundle.py \
  --checkpoint models/model.pth \
  --trainer-config models/trainer.yaml \
  --output-dir models/cpp_bundle \
  --overwrite
```

This produces:

- `models/cpp_bundle/manifest.yaml`
- `models/cpp_bundle/trainer.yaml`
- `models/cpp_bundle/tensors/.../*.bin`

## 2) Build

You need PyTorch (LibTorch via pip wheel) and yaml-cpp available to CMake.

If you use a virtual environment, activate it first so `python` resolves to the same environment where `torch` is installed.

```bash
TORCH_CMAKE_PREFIX=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX"
cmake --build build --parallel
```

If `find_package(Torch)` still fails, pass `Torch_DIR` directly:

```bash
TORCH_DIR=$(python -c "import os, torch; print(os.path.join(torch.utils.cmake_prefix_path, 'Torch'))")
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTorch_DIR="$TORCH_DIR"
cmake --build build --parallel
```

## 3) Run demo inference

```bash
./build/grad_sdf_infer \
  --config models/cpp_bundle/trainer.yaml \
  --bundle models/cpp_bundle \
  --device cpu \
  --num-points 4096
```

## 4) Demonstrate sampling only

This runs the same point sampling logic (with bounds from config) and prints sampled mins/maxs + preview points,
without loading a model.

```bash
./build/grad_sdf_infer \
  --config models/trainer.yaml \
  --device cpu \
  --num-points 32 \
  --seed 0 \
  --sample-only
```

## Checkpoint compatibility note

This scaffold currently supports checkpoints where residual MLP weights are saved as dense linear layers:

- `residual.residual_net.0.weight`
- `residual.residual_net.0.bias`
- ...

If your checkpoint contains only `residual.residual_net.params` (flattened parameter format), the loader will reject it.
