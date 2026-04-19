#!/usr/bin/bash

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
DATA_DIR=${DATA_DIR:-${SCRIPT_DIR}/../data/NewerCollege_preprocessed}
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/../data/NewerCollege_preprocessed/test_set}
GRID_RESOLUTION=${GRID_RESOLUTION:-0.1}

PYTHONPATH="${SCRIPT_DIR}/.." python3 "${SCRIPT_DIR}/../grad_sdf/dataset/generate_test_set.py" \
    --pcd-path "${DATA_DIR}/all_points.ply" \
    --grid-resolution ${GRID_RESOLUTION} \
    --eps 0.03 \
    --near-surface-sdf-range -0.1 0.2 \
    --output-dir "${OUTPUT_DIR}" \
    --bound-min -19.5 -27.25 -13.79 \
    --bound-max 19.5 27.25 -0.79