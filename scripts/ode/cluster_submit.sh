#!/usr/bin/env bash
# cluster_submit.sh — submit all MoCap finetuning jobs on the Lamarr cluster.
#
# Prerequisites:
#   1. FIM code cloned to ~/FIM/ (branch fim-ode-johannes)
#   2. Setup done: bash ~/FIM/experiments/cluster_setup.sh
#
# Run this on the cluster gateway INSIDE a tmux session:
#   tmux new-session -s fim
#   bash ~/FIM/experiments/cluster_submit.sh
#
# Each job gets its own tmux window. Monitor with:
#   tmux attach -t fim
# Cancel a job:
#   scancel <JOBID>   (get JOBID from: squeue -u $USER)
#
# Directory layout per task:
#   /cephfs_projects/foundation_models/models/FIMODE/mocap_<task>/
#       checkpoints/           — model checkpoints
#       logging/               — TensorBoard logs
#       model_architecture.txt — copy from FIMODE root
#       finetune_parameters.yaml

set -euo pipefail

FIM_DIR="$HOME/FIM"
VENV="$FIM_DIR/.venv/bin/activate"
CONTAINER_IMG="nvcr.io/ml2r/interactive_pytorch:23.12-py3"
LOG_DIR="$HOME/logs/mocap_finetune"

FIMODE_DIR="/cephfs_projects/foundation_models/models/FIMODE"
PRETRAINED_DIR="$FIMODE_DIR/pretrained"

SESSION="fim"

declare -a JOBS=("09_short" "09_long" "35_short" "35_long" "39_short" "39_long")

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -f "$VENV" ]; then
    echo "ERROR: venv not found. Run cluster_setup.sh first."
    exit 1
fi
if [ ! -f "$PRETRAINED_DIR/model.safetensors" ]; then
    echo "ERROR: pretrained model not found at $PRETRAINED_DIR."
    echo "Run cluster_setup.sh first."
    exit 1
fi
if [ ! -f "$FIMODE_DIR/model_architecture.txt" ]; then
    echo "ERROR: model_architecture.txt not found. Run cluster_setup.sh first."
    exit 1
fi

mkdir -p "$LOG_DIR"

# ── Create tmux session if needed ─────────────────────────────────────────────
tmux new-session -d -s "$SESSION" 2>/dev/null || true

# ── Submit one job per task ───────────────────────────────────────────────────
for i in "${!JOBS[@]}"; do
    JOB="${JOBS[$i]}"
    CONTAINER="fim_${JOB}"
    CONFIG="$FIM_DIR/experiments/configs/finetune_mocap_${JOB}.yaml"
    MODEL_DIR="$FIMODE_DIR/mocap_${JOB}"
    CKPT_DIR="$MODEL_DIR/checkpoints"
    RUN_DIR="$MODEL_DIR/logging"
    LOG="$LOG_DIR/ft_${JOB}.log"

    if [ "$i" -eq 0 ]; then
        tmux rename-window -t "${SESSION}:0" "$JOB" 2>/dev/null || true
    else
        tmux new-window -t "$SESSION" -n "$JOB"
    fi

    CMD="srun \
        --mem=64GB \
        -c 8 \
        --gres=gpu:1 \
        --container-name=${CONTAINER} \
        --container-image=${CONTAINER_IMG} \
        --export=ALL \
        -p GPU1 \
        bash -c '\
            set -e
            mkdir -p ${CKPT_DIR} ${RUN_DIR}
            cp ${FIMODE_DIR}/model_architecture.txt ${MODEL_DIR}/
            cp ${CONFIG} ${MODEL_DIR}/finetune_parameters.yaml
            source ${VENV}
            cd ${FIM_DIR}/experiments
            python3 fim-ode-finetune.py \
                --config  ${CONFIG} \
                --device  cuda \
                --ckpt-dir ${CKPT_DIR} \
                --run-dir  ${RUN_DIR} \
                --local-ckpt ${PRETRAINED_DIR} \
                2>&1 | tee ${LOG}
            echo \"=== ${JOB} DONE ===\"
        '"

    tmux send-keys -t "${SESSION}:${JOB}" "$CMD" Enter
    echo "Submitted: $JOB  →  $MODEL_DIR"
done

echo ""
echo "All 6 jobs submitted."
echo "  Monitor : tmux attach -t $SESSION"
echo "  Logs    : $LOG_DIR/"
echo "  Models  : $FIMODE_DIR/"
echo "  Queue   : squeue -u \$USER"
