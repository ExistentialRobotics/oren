#!/usr/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
SRC_DIR=$(cd ${SCRIPT_DIR}/..; pwd)
CONFIG_DIR=${SRC_DIR}/configs
REPLICA_DATA_DIR=${DATA_DIR:-${SRC_DIR}/data/Replica_preprocessed}

PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/oren/gui_trainer.py \
    --gui-config ${CONFIG_DIR}/gui.yaml \
    --trainer-config ${CONFIG_DIR}/replica_room0.yaml \
    --gt-mesh-path ${REPLICA_DATA_DIR}/room0_mesh.ply \
    --copy-scene-bound-to-gui
