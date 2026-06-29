"""
Temporary evaluation script: zero-shot vs finetuned FIMODE on VDP-u, VDP-nu, FHN.

VDP tasks use a single evaluation protocol: encode the training window, integrate
forward, compute MSE on the held-out test window.

FHN uses TWO evaluation protocols that measure the same thing (MSE at the 12
missing interpolation points) but differ in how the context is presented:

  [segmented]  Context is split at gap boundaries into 2 padded segments with a
               boolean mask.  This is the paper / notebook protocol (eval_ref_fhn
               in helpers.py).  The model never sees a spurious large Δx/Δt jump
               across the gap, so this is the honest zero-shot comparison.
               Paper reports zero-shot FHN ≈ 0.40 with this method.

  [flat]       All 38 observed points are fed as one continuous trajectory without
               any mask.  This is the training/finetuning protocol (make_fhn_eval_fn
               in finetune.py, mode=full_trajectory).  The model sees observations
               on both sides of the gap in one shot, which leaks post-gap context
               and lowers the apparent zero-shot MSE (~0.21).  This is the correct
               protocol for evaluating finetuned models since they were adapted
               using this same context format.

Usage:
    python eval_tmp.py
    python eval_tmp.py --device cuda
    python eval_tmp.py --pretrained results/ode/pretrained/base_model/checkpoints/best-model
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT / "scripts" / "ode") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts" / "ode"))

from fim.models.ode import load_fim_ode_local

# Load data + eval functions from finetune.py via importlib (avoids hyphen in filename)
_ft_spec = importlib.util.spec_from_file_location(
    "fim_ode_ft", _ROOT / "scripts" / "ode" / "finetune.py"
)
_ft = importlib.util.module_from_spec(_ft_spec)
_ft_spec.loader.exec_module(_ft)

load_vdp_uniform    = _ft.load_vdp_uniform
load_vdp_nonuniform = _ft.load_vdp_nonuniform
load_fhn            = _ft.load_fhn
make_vdp_eval_fn    = _ft.make_vdp_eval_fn
make_fhn_eval_fn    = _ft.make_fhn_eval_fn   # flat context — finetuning protocol

# Load segmented FHN eval from helpers.py (paper / notebook protocol)
from helpers import preprocess_ref_fhn, eval_ref_fhn

DATA_DIR    = _ROOT / "data" / "ode" / "hedge_gp_odes_data"
PRETRAINED  = _ROOT / "results" / "ode" / "pretrained" / "base_model" / "checkpoints" / "best-model"
RESULTS_DIR = _ROOT / "results" / "ode"


def find_finetuned_checkpoints(task_prefix: str) -> List[Tuple[str, Path]]:
    """Return [(run_label, best_pt_path)] sorted newest first."""
    results = [
        (p.parent.parent.name, p)
        for p in sorted(RESULTS_DIR.glob(f"{task_prefix}*/checkpoints/best.pt"), reverse=True)
    ]
    results += [
        (p.parent.parent.name, p)
        for p in sorted(RESULTS_DIR.glob(f"{task_prefix}*/checkpoints/*_best.pt"), reverse=True)
    ]
    return results


def load_finetuned(base_model, ckpt_path: Path, device: str):
    """Load finetuned weights into a fresh copy of base_model."""
    model = copy.deepcopy(base_model)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt.get("epoch", "?")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--pretrained", default=str(PRETRAINED),
                        help="Local checkpoint dir with config.json + model.safetensors")
    args = parser.parse_args()

    device         = args.device
    pretrained_dir = Path(args.pretrained)

    print(f"Loading base model from {pretrained_dir} …")
    base_model = load_fim_ode_local(pretrained_dir, device=device)
    base_model.eval()
    print(f"  {sum(p.numel() for p in base_model.parameters()):,} parameters\n")

    col_w  = 36
    header = f"{'Run / checkpoint':<{col_w}}  {'MSE':>12}  {'Epoch':>6}"
    sep    = "-" * len(header)

    # ── VDP-u ────────────────────────────────────────────────────────────────
    for task, loader_fn, prefix in [
        ("vdp-u",  load_vdp_uniform,    "vdp-u_"),
        ("vdp-nu", load_vdp_nonuniform, "vdp-nu_"),
    ]:
        print(f"{'='*60}")
        print(f"  Task: {task}  [extrapolation MSE on held-out test window]")
        print(f"{'='*60}")
        print(header)
        print(sep)

        data    = loader_fn(DATA_DIR)
        eval_fn = make_vdp_eval_fn(data, device)

        with torch.no_grad():
            mse_zs = eval_fn(base_model)
        print(f"{'zero-shot (pretrained)':<{col_w}}  {mse_zs:12.6e}  {'—':>6}")

        for label, ckpt_path in find_finetuned_checkpoints(prefix):
            model_ft, epoch = load_finetuned(base_model, ckpt_path, device)
            with torch.no_grad():
                mse_ft = eval_fn(model_ft)
            ratio = mse_ft / mse_zs
            print(f"{label:<{col_w}}  {mse_ft:12.6e}  {str(epoch):>6}   ×{ratio:.3f}{'↓' if ratio < 1 else '↑'}")
        print()

    # ── FHN — segmented (paper protocol) ─────────────────────────────────────
    print(f"{'='*60}")
    print(f"  Task: fhn  [segmented context — paper / notebook protocol]")
    print(f"{'='*60}")
    print(f"  Context: 2 padded segments with boolean mask (no gap-crossing Δx/Δt).")
    print(f"  Honest zero-shot evaluation as reported in the paper (~0.40).")
    print(f"  Finetuned models skipped here — they were trained on flat context.\n")
    print(header)
    print(sep)

    ref = np.load(DATA_DIR / "fhn_interpolation.npz")
    ctx_trajs, ctx_times, ctx_mask, miss_mask, _, target_ys = preprocess_ref_fhn(ref)

    _, mse_zs_seg = eval_ref_fhn(
        base_model, device,
        ctx_trajs, ctx_times, ctx_mask,
        ref["full_ts"], ref["x0"], miss_mask, target_ys,
        label="zero-shot (pretrained)",
    )
    print()

    # ── FHN — flat (finetuning protocol) ─────────────────────────────────────
    print(f"{'='*60}")
    print(f"  Task: fhn  [flat 38-point context — finetuning protocol]")
    print(f"{'='*60}")
    print(f"  Context: all 38 observed points as one trajectory (includes both sides")
    print(f"  of the gap).  Lower zero-shot MSE (~0.21) — post-gap context leaks in.")
    print(f"  Correct protocol for finetuned models since training used this format.\n")
    print(header)
    print(sep)

    data_flat = load_fhn(DATA_DIR)
    eval_flat = make_fhn_eval_fn(data_flat, device)

    with torch.no_grad():
        mse_zs_flat = eval_flat(base_model)
    print(f"{'zero-shot (pretrained)':<{col_w}}  {mse_zs_flat:12.6e}  {'—':>6}")

    for label, ckpt_path in find_finetuned_checkpoints("fhn_"):
        model_ft, epoch = load_finetuned(base_model, ckpt_path, device)
        with torch.no_grad():
            mse_ft = eval_flat(model_ft)
        ratio = mse_ft / mse_zs_flat
        print(f"{label:<{col_w}}  {mse_ft:12.6e}  {str(epoch):>6}   ×{ratio:.3f}{'↓' if ratio < 1 else '↑'}")
    print()


if __name__ == "__main__":
    main()
