#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 LOG_DIR"
  echo "Example: $0 logs/IROS/replica/room0/2026-02-27-12-12-51"
  exit 1
fi

LOG_DIR="$1"

# Parse scene name from log path
# Assumes structure: logs/IROS/replica/<SCENE>/<TIMESTAMP>
SCENE_NAME="$(basename "$(dirname "$LOG_DIR")")"

CONFIG="$LOG_DIR/bak/config.yaml"
MODEL="$LOG_DIR/ckpt/final.pth"
MESH_DIR="$LOG_DIR/mesh"

# 1) SDF & gradient evaluation
PYTHONPATH="$(pwd)" python grad_sdf/evaluator_grad_sdf.py \
  --config "$CONFIG" \
  --model-path "$MODEL" \
  --sdf-and-grad-metrics \
  --test-set-dir "data/replica-3040/${SCENE_NAME}/test_set"

# 2) mesh.ply evaluation
PYTHONPATH="$(pwd)" python grad_sdf/evaluator_grad_sdf.py \
  --config "$CONFIG" \
  --model-path "$MODEL" \
  --mesh-metrics --pred-mesh-paths \
  "$MESH_DIR/mesh.ply" \
  "$MESH_DIR/mesh_prior.ply" \
  --gt-mesh-path "data/replica-3040/${SCENE_NAME}_mesh.ply"

