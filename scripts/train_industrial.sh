#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${ROOT_DIR}/configs/replica/industrial.yaml"

echo "Project root : $ROOT_DIR"
echo "Config       : $CONFIG"
echo

PYTHONPATH="${ROOT_DIR}/oren${PYTHONPATH:+:$PYTHONPATH}" python oren/oren/trainer.py \
    --config "${CONFIG}" \
    "$@"
