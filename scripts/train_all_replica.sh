#!/usr/bin/env bash
set -euo pipefail

# 进入项目根目录
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="$ROOT_DIR"
CONFIG_DIR="$ROOT_DIR/configs/replica"

echo "Project root: $ROOT_DIR"
echo "Config dir  : $CONFIG_DIR"
echo

SCENES=(
#   "room0.yaml"
  "room1.yaml"
  "room2.yaml"
  "office0.yaml"
  "office1.yaml"
  "office2.yaml"
  "office3.yaml"
  "office4.yaml"
)

for cfg in "${SCENES[@]}"; do
  echo "==========================================="
  echo "Training with config: ${CONFIG_DIR}/${cfg}"
  echo "==========================================="
  PYTHONPATH="${PYTHONPATH}" python grad_sdf/trainer.py \
    --config "${CONFIG_DIR}/${cfg}"
done

