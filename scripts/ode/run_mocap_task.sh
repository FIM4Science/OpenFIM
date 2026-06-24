#!/usr/bin/env bash
# run_mocap_task.sh <task>
# e.g. bash run_mocap_task.sh 09_short
#
# Designed to be called by srun inside a container.

set -euo pipefail

TASK="${1:?Usage: run_mocap_task.sh <task>}"
FIM_DIR="/home/sanchez/FIM"
FIMODE_DIR="/cephfs_projects/foundation_models/models/FIMODE"

source "$FIM_DIR/.venv/bin/activate"

mkdir -p "$FIMODE_DIR/mocap_${TASK}/checkpoints"
mkdir -p "$FIMODE_DIR/mocap_${TASK}/logging"
cp "$FIM_DIR/configs/train/ode/finetune_mocap_${TASK}.yaml" \
   "$FIMODE_DIR/mocap_${TASK}/finetune_parameters.yaml"

cd "$FIM_DIR"
python3 scripts/ode/finetune.py \
    --config   "configs/train/ode/finetune_mocap_${TASK}.yaml" \
    --device   cuda \
    --ckpt-dir "$FIMODE_DIR/mocap_${TASK}/checkpoints" \
    --run-dir  "$FIMODE_DIR/mocap_${TASK}/logging" \
    --local-ckpt "$FIMODE_DIR/pretrained"
