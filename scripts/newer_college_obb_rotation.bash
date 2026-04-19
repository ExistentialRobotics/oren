#!/usr/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
DATA_DIR=${DATA_DIR:-${SCRIPT_DIR}/../data/NewerCollege}
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/../data/NewerCollege_preprocessed}

PYTHON_PATH=${SCRIPT_DIR}/.. python ${SCRIPT_DIR}/../grad_sdf/dataset/newer_college_obb_rotation.py \
    --dataset-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR}
