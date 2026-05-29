#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${ROOT_DIR}/oren${PYTHONPATH:+:$PYTHONPATH}"
CONFIG_DIR="$ROOT_DIR/configs/replica"

echo "Project root : $ROOT_DIR"
echo "Config dir   : $CONFIG_DIR"
echo

SCENES=(
    "garage.yaml"
    "forest.yaml"
    "industrial.yaml"
    "warehouse.yaml"
)

for cfg in "${SCENES[@]}"; do
    echo "==========================================="
    echo "Training: ${CONFIG_DIR}/${cfg}"
    echo "==========================================="
    PYTHONPATH="${PYTHONPATH}" python oren/oren/trainer.py \
        --config "${CONFIG_DIR}/${cfg}" \
        "$@"
done
