#!/usr/bin/env bash
# cluster_setup.sh — one-time environment setup on the Lamarr cluster.
#
# Run this ONCE inside the interactive container (sanchez_pytorch_26 or similar)
# BEFORE submitting finetuning jobs.
#
# What it does:
#   1. Creates a shared venv with system-site-packages (inherits PyTorch)
#   2. Installs the FIM package
#   3. Downloads the pretrained FIM-ODE weights from HuggingFace
#   4. Writes model_architecture.txt to the FIMODE project directory
#
# Usage (inside the container, after cloning the repo):
#   bash ~/FIM/experiments/cluster_setup.sh

set -euo pipefail

FIM_DIR="$HOME/FIM"
VENV_DIR="$FIM_DIR/.venv"
FIMODE_DIR="/cephfs_projects/foundation_models/models/FIMODE"
PRETRAINED_DIR="$FIMODE_DIR/pretrained"

echo "=== FIM-ODE cluster setup ==="
echo "FIM dir    : $FIM_DIR"
echo "Venv       : $VENV_DIR"
echo "FIMODE dir : $FIMODE_DIR"
echo ""

# ── 1. Python venv ────────────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "[1/4] Venv already exists — skipping creation."
else
    echo "[1/4] Creating venv with system-site-packages ..."
    /bin/python -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ── 2. Install FIM package ────────────────────────────────────────────────────
echo "[2/4] Installing FIM package ..."
pip install --quiet -e "$FIM_DIR"

echo "      Verifying imports ..."
python -c "import torch; print(f'  torch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
python -c "import fim; print('  fim OK')"

# ── 3. Download pretrained FIM-ODE weights from HuggingFace ──────────────────
mkdir -p "$PRETRAINED_DIR"

if [ -f "$PRETRAINED_DIR/model.safetensors" ]; then
    echo "[3/4] Pretrained weights already present — skipping download."
else
    echo "[3/4] Downloading pretrained FIM-ODE from HuggingFace ..."
    python - << PYEOF
from huggingface_hub import hf_hub_download
import shutil, pathlib

repo      = "FIM4Science/fim-ode"
subfolder = "base_model/checkpoints/best-model"
dest      = pathlib.Path("$PRETRAINED_DIR")

for fname in ["config.json", "model.safetensors"]:
    src = hf_hub_download(repo, f"{subfolder}/{fname}")
    shutil.copy(src, dest / fname)
    print(f"  saved {fname}")
PYEOF
fi

# ── 4. Generate model_architecture.txt ───────────────────────────────────────
ARCH_FILE="$FIMODE_DIR/model_architecture.txt"

if [ -f "$ARCH_FILE" ]; then
    echo "[4/4] model_architecture.txt already exists — skipping."
else
    echo "[4/4] Generating model_architecture.txt ..."
    python - << PYEOF
import sys, json
sys.path.insert(0, "$FIM_DIR/src")
sys.path.insert(0, "$FIM_DIR/experiments")

from safetensors.torch import load_file
from fim.models.fim_ode import FIMODE as FimOdeonUnified
import pathlib

pretrained = pathlib.Path("$PRETRAINED_DIR")
with open(pretrained / "config.json") as f:
    import json
    config = json.load(f)

model = FimOdeonUnified(config)
n_params = sum(p.numel() for p in model.parameters())

with open("$ARCH_FILE", "w") as f:
    f.write(str(model))
    f.write(f"\n\nTotal trainable parameters: {n_params:,}\n")

print(f"  written to $ARCH_FILE  ({n_params:,} parameters)")
PYEOF
fi

echo ""
echo "Setup complete."
echo "  Pretrained model : $PRETRAINED_DIR"
echo "  Architecture     : $ARCH_FILE"
echo ""
echo "Next step: on the gateway inside tmux, run:"
echo "  bash $FIM_DIR/experiments/cluster_submit.sh"
