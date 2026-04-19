#!/usr/bin/bash

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
DATA_DIR=${DATA_DIR:-${SCRIPT_DIR}/../data/Replica_preprocessed}

for scene in office0 office1 office2 office3 office4 room0 room1 room2; do
    echo "Generating test set for scene: ${scene}"

    PYTHONPATH="${SCRIPT_DIR}/.." python3 "${SCRIPT_DIR}/../grad_sdf/dataset/generate_test_set.py" \
        --mesh-path "${DATA_DIR}/${scene}_mesh.ply" \
        --grid-resolution 0.0125 \
        --eps 0.01 \
        --near-surface-sdf-range -0.1 0.2 \
        --output-dir "${DATA_DIR}/${scene}/test_set"

done