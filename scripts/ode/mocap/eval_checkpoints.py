"""
scripts/ode/mocap/eval_checkpoints.py
=======================================
Evaluate finetuned MoCap checkpoints on the cluster and print an MSE table.

Checkpoints are saved as ``{task_label}_best.pt`` files containing
``model_state_dict``.  We load the base FIMODE architecture (from
the local pretrained copy or HuggingFace) and inject the finetuned weights.

Usage (inside a GPU container on the cluster):
    cd ~/FIM
    python scripts/ode/mocap/eval_checkpoints.py [--device cuda]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

from fim.models.ode import load_fim_ode_hf, load_fim_ode_local


# ── project root on sys.path ──────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent  # scripts/ode/mocap/
_ROOT = _HERE.parent.parent.parent  # repo root

for _p in [str(_ROOT / "src"), str(_ROOT / "scripts" / "ode")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


_ft_path = _ROOT / "scripts" / "ode" / "finetune.py"
_ft_spec = importlib.util.spec_from_file_location("fim_ode_ft", _ft_path)
_ft = importlib.util.module_from_spec(_ft_spec)
_ft_spec.loader.exec_module(_ft)

load_mocap = _ft.load_mocap
make_mocap_test_eval_fn = _ft.make_mocap_test_eval_fn

FIMODE_DIR = Path("/cephfs_projects/foundation_models/models/FIMODE")

TASKS = [
    ("09", "short"),
    ("09", "long"),
    ("35", "short"),
    ("35", "long"),
    ("39", "short"),
    ("39", "long"),
]


def load_finetuned_model(base_model, ckpt_dir: Path, device: str):
    """Find the *_best.pt in ckpt_dir, load its weights into a copy of base_model."""
    import copy

    best_files = sorted(ckpt_dir.glob("*_best.pt"))
    if not best_files:
        raise FileNotFoundError(f"No *_best.pt found in {ckpt_dir}")
    ckpt_path = best_files[-1]  # take the latest if multiple exist
    print(f"    loading {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = copy.deepcopy(base_model)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    return model, epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mocap-dir", default=str(_ROOT / "data" / "mocap"))
    parser.add_argument(
        "--pretrained-dir", default=str(FIMODE_DIR / "pretrained"), help="Local copy of HF pretrained weights (falls back to HF download)"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--zeroshot", action="store_true", help="Also evaluate zero-shot baseline")
    args = parser.parse_args()

    mocap_dir = Path(args.mocap_dir)
    device = args.device

    # ── Load base model once ──────────────────────────────────────────────────
    pretrained_dir = Path(args.pretrained_dir)
    if pretrained_dir.exists() and (pretrained_dir / "model.safetensors").exists():
        print(f"Loading base model from {pretrained_dir} …")
        base_model = load_fim_ode_local(pretrained_dir, device=device)
    else:
        print("Loading base model from HuggingFace …")
        base_model = load_fim_ode_hf(device=device)
    base_model.eval()

    # ── Zero-shot baseline ────────────────────────────────────────────────────
    zs_mses = {}
    if args.zeroshot:
        print("\nEvaluating zero-shot …")
        for subj, var in TASKS:
            tag = f"{subj}_{var}"
            try:
                data = load_mocap(mocap_dir, subject=subj, variant=var)
                eval_fn = make_mocap_test_eval_fn(data, device)
                mse = eval_fn(base_model)
                zs_mses[tag] = mse
                print(f"  {tag:<15} zero-shot MSE = {mse:.5f}")
            except FileNotFoundError as e:
                print(f"  {tag:<15} data not found: {e}")
                zs_mses[tag] = float("nan")

    # ── Finetuned checkpoints ─────────────────────────────────────────────────
    ft_mses = {}
    ft_epochs = {}
    print("\nEvaluating finetuned checkpoints …")
    for subj, var in TASKS:
        tag = f"{subj}_{var}"
        ckpt_dir = FIMODE_DIR / f"mocap_{tag}" / "checkpoints"

        if not ckpt_dir.exists():
            print(f"  {tag:<15} checkpoint dir not found — skipping")
            ft_mses[tag] = float("nan")
            continue

        try:
            model, epoch = load_finetuned_model(base_model, ckpt_dir, device)
            data = load_mocap(mocap_dir, subject=subj, variant=var)
            eval_fn = make_mocap_test_eval_fn(data, device)
            mse = eval_fn(model)
            ft_mses[tag] = mse
            ft_epochs[tag] = epoch
            print(f"  {tag:<15} epoch {epoch:<4} finetuned MSE = {mse:.5f}")
            del model
        except Exception as e:
            print(f"  {tag:<15} ERROR: {e}")
            ft_mses[tag] = float("nan")

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    if args.zeroshot:
        print(f"{'Task':<15} {'Epoch':>6} {'Zero-shot':>12} {'Finetuned':>12} {'Improvement':>13}")
        print("-" * 63)
    else:
        print(f"{'Task':<15} {'Epoch':>6} {'Finetuned':>12}")
        print("-" * 36)
    for subj, var in TASKS:
        tag = f"{subj}_{var}"
        zs = zs_mses.get(tag, float("nan"))
        ft = ft_mses.get(tag, float("nan"))
        epoch = ft_epochs.get(tag, "—")
        ft_str = f"{ft:.5f}" if ft == ft else "not ready"
        if args.zeroshot:
            impr = f"{(zs - ft) / zs * 100:+.1f}%" if (zs == zs and ft == ft) else "—"
            zs_str = f"{zs:.5f}" if zs == zs else "—"
            print(f"{tag:<15} {str(epoch):>6} {zs_str:>12} {ft_str:>12} {impr:>13}")
        else:
            print(f"{tag:<15} {str(epoch):>6} {ft_str:>12}")


if __name__ == "__main__":
    main()
