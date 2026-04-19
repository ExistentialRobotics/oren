#!/usr/bin/bash

set -e
set -x

SCRIPT_DIR=$(cd $(dirname $0); pwd)
DATA_DIR=${DATA_DIR:-${SCRIPT_DIR}/../data/Replica_preprocessed}

SCENE="room0"
CONFIG_PATH=${SCRIPT_DIR}/../../logs/replica_room0/2025-09-21-20-04-15/bak/config.yaml
MODEL_PATH=${SCRIPT_DIR}/../../logs/replica_room0/2025-09-21-20-04-15/ckpt/final.pth

# extract mesh
PYTHONPATH="${SCRIPT_DIR}/.." python3 "${SCRIPT_DIR}/../grad_sdf/evaluater_grad_sdf.py" \
    --config ${CONFIG_PATH} \
    --model-path ${MODEL_PATH} \
    --extract-mesh \
    --clean-mesh

# sdf & grad metrics
PYTHONPATH="${SCRIPT_DIR}/.." python3 "${SCRIPT_DIR}/../grad_sdf/evaluater_grad_sdf.py" \
    --config ${CONFIG_PATH} \
    --model-path ${MODEL_PATH} \
    --sdf-and-grad-metrics \
    --test-set-dir ${DATASET_DIR}/${SCENE}/test_set

# mesh metrics
PYTHONPATH="${SCRIPT_DIR}/.." python3 "${SCRIPT_DIR}/../grad_sdf/evaluater_grad_sdf.py" \
    --config "${CONFIG_PATH}" \
    --model-path "${MODEL_PATH}" \
    --mesh-metrics \
    --pred-mesh-paths "${SCRIPT_DIR}/../../logs/replica_room0/2025-09-21-20-04-15/eval/mesh_sdf.ply" \
        "${SCRIPT_DIR}/../../logs/replica_room0/2025-09-21-20-04-15/eval/mesh_sdf_prior.ply" \
    --gt-mesh-path "${DATASET_DIR}/${SCENE}_mesh.ply" \
    --f1-threshold 0.05 \
    --num-points 200000 \
    --seed 0