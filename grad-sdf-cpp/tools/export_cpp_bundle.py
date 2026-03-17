#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent.parent
    models_dir = project_dir / "models"
    default_checkpoint = models_dir / "model.pth"
    default_trainer_config = models_dir / "trainer-ros.yaml"
    default_output_dir = models_dir / "bundle"

    parser = argparse.ArgumentParser(description="Export a grad-sdf checkpoint into a C++ tensor bundle.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(default_checkpoint),
        help="Path to model checkpoint (.pth) saved via torch.save(state_dict) "
        f"(default: {default_checkpoint})",
    )
    parser.add_argument(
        "--trainer-config",
        type=str,
        default=str(default_trainer_config),
        help=f"Path to trainer YAML config (default: {default_trainer_config})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir),
        help=f"Output directory for C++ bundle (default: {default_output_dir})",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists")
    return parser.parse_args()


def make_rel_tensor_path(key: str) -> Path:
    parts = key.split(".")
    return Path("tensors").joinpath(*parts).with_suffix(".bin")


def dtype_to_str(dtype: torch.dtype) -> str:
    mapping = {
        torch.float32: "float32",
        torch.float64: "float64",
        torch.int64: "int64",
        torch.int32: "int32",
        torch.int16: "int16",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.bool: "bool",
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype for export: {dtype}")
    return mapping[dtype]


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    trainer_config_path = Path(args.trainer_config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not trainer_config_path.exists():
        raise FileNotFoundError(f"Trainer config does not exist: {trainer_config_path}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint must deserialize to a state dict (dict-like object).")

    # Determine the actual number of voxels from the loaded tensors
    actual_num_voxels = None
    if "octree.sdf_priors" in state_dict:
        actual_num_voxels = state_dict["octree.sdf_priors"].shape[0]
    
    manifest = {
        "checkpoint": str(checkpoint_path),
        "trainer_config": "trainer.yaml",
        "tensors": {},
    }
    
    # Add voxel count metadata if we found it
    if actual_num_voxels is not None:
        manifest["metadata"] = {
            "octree_num_voxels": int(actual_num_voxels),
        }
        print(f"Detected octree with {actual_num_voxels} voxels")

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue

        rel_path = make_rel_tensor_path(key)
        abs_path = output_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        tensor = value.detach().cpu().contiguous()
        array = tensor.numpy()
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected numpy array for key {key}")

        array.tofile(abs_path)
        manifest["tensors"][key] = {
            "path": rel_path.as_posix(),
            "dtype": dtype_to_str(tensor.dtype),
            "shape": list(tensor.shape),
        }

    shutil.copy2(trainer_config_path, output_dir / "trainer.yaml")

    with open(output_dir / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=True)

    print(f"Exported {len(manifest['tensors'])} tensors to {output_dir}")

    if "residual.residual_net.params" in manifest["tensors"]:
        print(
            "WARNING: Found residual.residual_net.params (flattened tiny-cuda-nn style checkpoint).\n"
            "The current C++ scaffold expects dense torch.nn.Linear residual weights."
        )


if __name__ == "__main__":
    main()
