"""
Infer hidden interaction network (influence matrix) between marks for a trained
FIM-Hawkes model and a given dataset.

For each inference path, we form the matrix with entries
  M_{k,j} = (1 / N_j) * sum_{i : m_i = j} (alpha_{k,i} - mu_{k,i}) / beta_{k,i}
where N_j is the number of events of type j in that path (if N_j = 0, the entry
is defined as 0). We compute this per path and then return the mean across all
paths.

Usage (single run):
python3 scripts/hawkes/fim_infer_network.py \
  --checkpoint <dir> \
  --dataset <easytpp/ID | short_name> \
  --run-dir <output_dir> \
  [--context-size <int>] [--inference-size <int>] [--max-num-events <int|-1>]

Usage (grid via YAML config):
python3 scripts/hawkes/fim_infer_network.py \
  --config scripts/hawkes/fim_infer_network.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from datasets import load_dataset


# ============================
# FIM-Hawkes import and setup
# ============================
try:
    from fim.models.hawkes import FIMHawkes, FIMHawkesConfig
except ImportError:
    print("Error: Could not import FIMHawkes. Ensure 'fim' is on PYTHONPATH.")
    raise

FIMHawkesConfig.register_for_auto_class()
FIMHawkes.register_for_auto_class("AutoModel")


def load_fimhawkes_with_proper_weights(checkpoint_path: str) -> FIMHawkes:
    return FIMHawkes.load_model(Path(checkpoint_path))


def detect_num_event_types_from_data(context_data_raw: Dict, inference_data_raw: Dict) -> int:
    unique = set()
    for s in context_data_raw.get("type_event", []):
        unique.update(int(t) for t in s)
    for s in inference_data_raw.get("type_event", []):
        unique.update(int(t) for t in s)
    return int(max(unique) + 1) if unique else 1


def _truncate_batch(data_dict: Dict, max_num_events: int | None) -> Dict:
    if max_num_events is None or max_num_events < 0:
        return data_dict
    out = {k: [] for k in ["time_since_start", "time_since_last_event", "type_event", "seq_len"]}
    for times, deltas, types, length in zip(
        data_dict["time_since_start"],
        data_dict["time_since_last_event"],
        data_dict["type_event"],
        data_dict["seq_len"],
    ):
        trunc = min(int(length), int(max_num_events))
        out["time_since_start"].append(times[:trunc])
        out["time_since_last_event"].append(deltas[:trunc])
        out["type_event"].append(types[:trunc])
        out["seq_len"].append(trunc)
    return out


def load_dataset_hf(dataset: str, context_size: int | None, inference_size: int | None) -> Tuple[Dict, Dict]:
    dataset_str = str(dataset)
    # If the provided dataset is a local path, load directly from files in that directory.
    ds_path = Path(dataset_str)
    if ds_path.exists():
        # First, support local tensor format: split subdirs with *.pt tensors
        def _exists_tensor_split(split_name: str) -> bool:
            sp = ds_path / split_name
            return (
                sp.exists() and (sp / "event_times.pt").exists() and (sp / "event_types.pt").exists() and (sp / "seq_lengths.pt").exists()
            )

        def _load_tensor_split(split_name: str) -> Dict:
            split_path = ds_path / split_name
            event_times = torch.load(split_path / "event_times.pt")  # [N, P, K, 1]
            event_types = torch.load(split_path / "event_types.pt")  # [N, P, K, 1]
            seq_lengths = torch.load(split_path / "seq_lengths.pt")  # [N, P]

            N, P, _ = int(event_times.shape[0]), int(event_times.shape[1]), int(event_times.shape[2])
            time_since_start: list[list[float]] = []
            time_since_last_event: list[list[float]] = []
            type_event: list[list[int]] = []
            seq_len: list[int] = []

            # Squeeze final dim if present
            if event_times.dim() == 4 and event_times.shape[-1] == 1:
                event_times_flat = event_times.squeeze(-1)
                event_types_flat = event_types.squeeze(-1)
            else:
                event_times_flat = event_times
                event_types_flat = event_types

            for n in range(N):
                for p in range(P):
                    L = int(seq_lengths[n, p].item())
                    times_t = event_times_flat[n, p, :L]
                    types_t = event_types_flat[n, p, :L]
                    times = [float(x) for x in times_t.tolist()]
                    types = [int(x) for x in types_t.tolist()]
                    deltas = [0.0] + [times[j] - times[j - 1] for j in range(1, len(times))]
                    time_since_start.append(times)
                    time_since_last_event.append(deltas)
                    type_event.append(types)
                    seq_len.append(L)

            return {
                "time_since_start": time_since_start,
                "time_since_last_event": time_since_last_event,
                "type_event": type_event,
                "seq_len": seq_len,
            }

        # Prefer tensor splits if present
        has_ctx_tensor = _exists_tensor_split("context")
        has_test_tensor = _exists_tensor_split("test") or _exists_tensor_split("val")
        if has_ctx_tensor and has_test_tensor:
            ctx_split = "context"
            inf_split = "test" if _exists_tensor_split("test") else "val"
            train_col = _load_tensor_split(ctx_split)
            test_col = _load_tensor_split(inf_split)

            def _take_first_n(col: Dict, n: int | None) -> Dict:
                if n is None:
                    return col
                n_eff = max(0, int(n))
                return {k: v[:n_eff] for k, v in col.items()}

            return _take_first_n(train_col, context_size), _take_first_n(test_col, inference_size)

        def _read_json_or_jsonl(file_path: Path) -> list | dict:
            try:
                # Try JSON first (single JSON object or list)
                return json.loads(file_path.read_text())
            except Exception:
                # Try JSON Lines
                try:
                    items = []
                    with file_path.open("r") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            items.append(json.loads(line))
                    return items
                except Exception as _:
                    raise

        def _to_columnar(obj: list | dict) -> Dict:
            # Expected keys
            keys = ["time_since_start", "time_since_last_event", "type_event", "seq_len"]
            if isinstance(obj, dict) and all(k in obj for k in keys):
                return {k: list(obj[k]) for k in keys}
            if isinstance(obj, list):
                col = {k: [] for k in keys}
                for ex in obj:
                    # Minimal robustness: compute seq_len if absent
                    tss = ex.get("time_since_start")
                    tsl = ex.get("time_since_last_event")
                    te = ex.get("type_event")
                    L = ex.get("seq_len")
                    if L is None:
                        L = int(len(te) if isinstance(te, (list, tuple)) else len(tss or []))
                    col["time_since_start"].append(list(tss))
                    col["time_since_last_event"].append(list(tsl))
                    col["type_event"].append(list(te))
                    col["seq_len"].append(int(L))
                return col
            raise ValueError("Unsupported dataset file structure; expected dict-of-lists or list-of-dicts.")

        def _load_split(split_names: list[str]) -> Dict:
            candidates = []
            for name in split_names:
                for ext in [".json", ".jsonl"]:
                    candidates.append(ds_path / f"{name}{ext}")
            for fp in candidates:
                if fp.exists():
                    return _to_columnar(_read_json_or_jsonl(fp))
            raise FileNotFoundError(f"Could not locate any of {[str(p) for p in candidates]} under {ds_path}")

        train_col = _load_split(["train", "context"])  # prefer train, fallback to context
        test_col = _load_split(["test", "valid", "validation", "eval", "inference"])  # prefer test

        def _take_first_n(col: Dict, n: int | None) -> Dict:
            if n is None:
                return col
            n_eff = max(0, int(n))
            return {k: v[:n_eff] for k, v in col.items()}

        return _take_first_n(train_col, context_size), _take_first_n(test_col, inference_size)

    # Otherwise, load from HuggingFace EasyTPP datasets
    dataset_id = dataset_str if dataset_str.startswith("easytpp/") else f"easytpp/{dataset_str}"
    train = load_dataset(dataset_id, split="train")
    test = load_dataset(dataset_id, split="test")
    eff_ctx = len(train) if context_size is None else int(context_size)
    eff_inf = len(test) if inference_size is None else int(inference_size)
    return train[:eff_ctx], test[:eff_inf]


@torch.no_grad()
def infer_interaction_matrix(
    model: FIMHawkes,
    context_data_raw: Dict,
    inference_data_raw: Dict,
    max_num_events: int | None,
    device: torch.device,
    inference_batch_size: int | None = 1,
) -> np.ndarray:
    # Truncate
    context_data_raw = _truncate_batch(context_data_raw, max_num_events)
    inference_data_raw = _truncate_batch(inference_data_raw, max_num_events)

    # Detect number of marks
    num_marks = detect_num_event_types_from_data(context_data_raw, inference_data_raw)
    model.config.max_num_marks = num_marks

    # Build context batch (pad to max length)
    num_ctx = len(context_data_raw.get("seq_len", []))
    max_ctx_len = max((len(s) for s in context_data_raw.get("time_since_start", [])), default=0)
    ctx = {"time_seqs": [], "type_seqs": [], "seq_len": []}
    for i in range(num_ctx):
        pad = max_ctx_len - len(context_data_raw["time_since_start"][i])
        ctx["time_seqs"].append(context_data_raw["time_since_start"][i] + [0] * pad)
        ctx["type_seqs"].append([int(t) for t in context_data_raw["type_event"][i]] + [0] * pad)
        ctx["seq_len"].append(int(context_data_raw["seq_len"][i]))
    if num_ctx == 0:
        # Fallback to an empty single context row
        ctx = {"time_seqs": [[0.0]], "type_seqs": [[0]], "seq_len": [1]}
        max_ctx_len = 1
        num_ctx = 1
    ctx = {k: torch.tensor(v, device=device) for k, v in ctx.items()}
    ctx["seq_non_pad_mask"] = torch.arange(max_ctx_len, device=device).expand(num_ctx, max_ctx_len) < ctx["seq_len"].unsqueeze(1)

    # Build inference batch (pad all sequences to same length)
    num_inf = len(inference_data_raw.get("seq_len", []))
    if num_inf == 0:
        return np.zeros((num_marks, num_marks), dtype=np.float32)
    max_inf_len = max(int(l) for l in inference_data_raw["seq_len"]) if num_inf > 0 else 0
    inf = {"time_seqs": [], "type_seqs": [], "seq_len": []}
    for i in range(num_inf):
        pad = max_inf_len - len(inference_data_raw["time_since_start"][i])
        inf["time_seqs"].append(inference_data_raw["time_since_start"][i] + [0] * pad)
        inf["type_seqs"].append([int(t) for t in inference_data_raw["type_event"][i]] + [0] * pad)
        inf["seq_len"].append(int(inference_data_raw["seq_len"][i]))
    inf = {k: torch.tensor(v, device=device) for k, v in inf.items()}
    inf["seq_non_pad_mask"] = torch.arange(max_inf_len, device=device).expand(num_inf, max_inf_len) < inf["seq_len"].unsqueeze(1)

    # Precompute enhanced context once
    precomp_ctx = {
        "context_event_times": ctx["time_seqs"].unsqueeze(0).unsqueeze(-1),
        "context_event_types": ctx["type_seqs"].unsqueeze(0).unsqueeze(-1),
        "context_seq_lengths": ctx["seq_non_pad_mask"].sum(dim=1).unsqueeze(0),
    }
    enhanced_context = model.encode_context(precomp_ctx)

    # Process inference paths in mini-batches to reduce memory
    running_sum = torch.zeros(num_marks, num_marks, device=device)
    total_paths = 0
    bsz = max(1, int(inference_batch_size or 1))

    for start in range(0, num_inf, bsz):
        end = min(num_inf, start + bsz)
        p_batch = end - start

        x = {
            "context_event_times": ctx["time_seqs"].unsqueeze(0).unsqueeze(-1),
            "context_event_types": ctx["type_seqs"].unsqueeze(0).unsqueeze(-1),
            "context_seq_lengths": ctx["seq_non_pad_mask"].sum(dim=1).unsqueeze(0),
            "inference_event_times": inf["time_seqs"][start:end].unsqueeze(0).unsqueeze(-1),
            "inference_event_types": inf["type_seqs"][start:end].unsqueeze(0).unsqueeze(-1),
            "inference_seq_lengths": inf["seq_non_pad_mask"][start:end].sum(dim=1).unsqueeze(0),
            "intensity_evaluation_times": torch.zeros(1, p_batch, 1, device=device),
            "precomputed_enhanced_context": enhanced_context,
            "num_marks": torch.tensor([num_marks], device=device),
        }

        model_out = model.forward(x)
        intensity_fn = model_out["intensity_function"]  # PiecewiseHawkesIntensity

        mu = intensity_fn.mu.squeeze(0)  # [M, P, L]
        alpha = intensity_fn.alpha.squeeze(0)
        beta = intensity_fn.beta.squeeze(0)

        event_types_b = inf["type_seqs"][start:end].to(dtype=torch.long)  # [P, L]
        seq_lengths_b = inf["seq_len"][start:end]  # [P]
        positions = torch.arange(event_types_b.shape[1], device=device).view(1, -1)
        valid_mask = positions < seq_lengths_b.unsqueeze(1)  # [P, L]

        eps = 1e-9
        contrib = (alpha - mu) / (beta + eps)  # [M, P, L]
        et_clamped = torch.clamp(event_types_b, 0, num_marks - 1)
        one_hot_src = torch.nn.functional.one_hot(et_clamped, num_classes=num_marks).to(contrib.dtype)  # [P, L, M]
        one_hot_src = one_hot_src * valid_mask.unsqueeze(-1)

        contrib_e = contrib.unsqueeze(0).unsqueeze(-1)  # [1, M, P, L, 1]
        src_mask_e = one_hot_src.unsqueeze(0).unsqueeze(0)  # [1, 1, P, L, M]
        sum_contrib = (contrib_e * src_mask_e).sum(dim=3)  # [1, M, P, M]

        counts = one_hot_src.sum(dim=1)  # [P, M]
        counts_e = counts.unsqueeze(0).unsqueeze(0)
        mean_per_path = torch.where(counts_e > 0, sum_contrib / (counts_e + eps), torch.zeros_like(sum_contrib))  # [1,M,P,M]

        running_sum = running_sum + mean_per_path.sum(dim=2).squeeze(0)
        total_paths += p_batch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_matrix = (running_sum / max(1, total_paths)).detach().cpu().numpy()
    return mean_matrix.astype(np.float32)


@torch.no_grad()
def infer_interaction_and_kernels(
    model: FIMHawkes,
    context_data_raw: Dict,
    inference_data_raw: Dict,
    max_num_events: int | None,
    device: torch.device,
    inference_batch_size: int | None = 1,
    kernel_num_points: int = 200,
    kernel_quantile: float = 0.9,
    kernel_time_scale_factor: float = 8.0,
):
    """
    Compute the interaction matrix as in infer_interaction_matrix and, in addition,
    the per-entry kernel functions phi_{k,j}(s) on a grid of lags s.

    Returns (matrix: [M,M], kernels: [M,M,S], s_grid: [S]).
    """

    # Truncate
    context_data_raw = _truncate_batch(context_data_raw, max_num_events)
    inference_data_raw = _truncate_batch(inference_data_raw, max_num_events)

    # Detect number of marks
    num_marks = detect_num_event_types_from_data(context_data_raw, inference_data_raw)
    model.config.max_num_marks = num_marks

    # Build context batch (pad to max length)
    num_ctx = len(context_data_raw.get("seq_len", []))
    max_ctx_len = max((len(s) for s in context_data_raw.get("time_since_start", [])), default=0)
    ctx = {"time_seqs": [], "type_seqs": [], "seq_len": []}
    for i in range(num_ctx):
        pad = max_ctx_len - len(context_data_raw["time_since_start"][i])
        ctx["time_seqs"].append(context_data_raw["time_since_start"][i] + [0] * pad)
        ctx["type_seqs"].append([int(t) for t in context_data_raw["type_event"][i]] + [0] * pad)
        ctx["seq_len"].append(int(context_data_raw["seq_len"][i]))
    if num_ctx == 0:
        ctx = {"time_seqs": [[0.0]], "type_seqs": [[0]], "seq_len": [1]}
        max_ctx_len = 1
        num_ctx = 1
    ctx = {k: torch.tensor(v, device=device) for k, v in ctx.items()}
    ctx["seq_non_pad_mask"] = torch.arange(max_ctx_len, device=device).expand(num_ctx, max_ctx_len) < ctx["seq_len"].unsqueeze(1)

    # Build inference batch (pad all sequences to same length)
    num_inf = len(inference_data_raw.get("seq_len", []))
    if num_inf == 0:
        zero = np.zeros((num_marks, num_marks), dtype=np.float32)
        s_grid_np = np.linspace(0.0, 1.0, int(max(2, kernel_num_points)), dtype=np.float32)
        kernels = np.zeros((num_marks, num_marks, s_grid_np.shape[0]), dtype=np.float32)
        return zero, kernels, s_grid_np
    max_inf_len = max(int(l) for l in inference_data_raw["seq_len"]) if num_inf > 0 else 0
    inf = {"time_seqs": [], "type_seqs": [], "seq_len": []}
    for i in range(num_inf):
        pad = max_inf_len - len(inference_data_raw["time_since_start"][i])
        inf["time_seqs"].append(inference_data_raw["time_since_start"][i] + [0] * pad)
        inf["type_seqs"].append([int(t) for t in inference_data_raw["type_event"][i]] + [0] * pad)
        inf["seq_len"].append(int(inference_data_raw["seq_len"][i]))
    inf = {k: torch.tensor(v, device=device) for k, v in inf.items()}
    inf["seq_non_pad_mask"] = torch.arange(max_inf_len, device=device).expand(num_inf, max_inf_len) < inf["seq_len"].unsqueeze(1)

    # Precompute enhanced context once
    precomp_ctx = {
        "context_event_times": ctx["time_seqs"].unsqueeze(0).unsqueeze(-1),
        "context_event_types": ctx["type_seqs"].unsqueeze(0).unsqueeze(-1),
        "context_seq_lengths": ctx["seq_non_pad_mask"].sum(dim=1).unsqueeze(0),
    }
    enhanced_context = model.encode_context(precomp_ctx)

    running_sum = torch.zeros(num_marks, num_marks, device=device)
    kernels_running_sum: torch.Tensor | None = None
    s_grid: torch.Tensor | None = None
    total_paths = 0
    bsz = max(1, int(inference_batch_size or 1))

    for start in range(0, num_inf, bsz):
        end = min(num_inf, start + bsz)
        p_batch = end - start

        x = {
            "context_event_times": ctx["time_seqs"].unsqueeze(0).unsqueeze(-1),
            "context_event_types": ctx["type_seqs"].unsqueeze(0).unsqueeze(-1),
            "context_seq_lengths": ctx["seq_non_pad_mask"].sum(dim=1).unsqueeze(0),
            "inference_event_times": inf["time_seqs"][start:end].unsqueeze(0).unsqueeze(-1),
            "inference_event_types": inf["type_seqs"][start:end].unsqueeze(0).unsqueeze(-1),
            "inference_seq_lengths": inf["seq_non_pad_mask"][start:end].sum(dim=1).unsqueeze(0),
            "intensity_evaluation_times": torch.zeros(1, p_batch, 1, device=device),
            "precomputed_enhanced_context": enhanced_context,
            "num_marks": torch.tensor([num_marks], device=device),
        }

        model_out = model.forward(x)
        intensity_fn = model_out["intensity_function"]
        mu = intensity_fn.mu.squeeze(0)  # [M,P,L]
        alpha = intensity_fn.alpha.squeeze(0)
        beta = intensity_fn.beta.squeeze(0)

        event_types_b = inf["type_seqs"][start:end].to(dtype=torch.long)  # [P,L]
        seq_lengths_b = inf["seq_len"][start:end]  # [P]
        positions = torch.arange(event_types_b.shape[1], device=device).view(1, -1)
        valid_mask = positions < seq_lengths_b.unsqueeze(1)  # [P,L]

        eps = 1e-9
        contrib = (alpha - mu) / (beta + eps)  # [M,P,L]
        et_clamped = torch.clamp(event_types_b, 0, num_marks - 1)
        one_hot_src = torch.nn.functional.one_hot(et_clamped, num_classes=num_marks).to(contrib.dtype)  # [P,L,M]
        one_hot_src = one_hot_src * valid_mask.unsqueeze(-1)

        contrib_e = contrib.unsqueeze(0).unsqueeze(-1)  # [1,M,P,L,1]
        src_mask_e = one_hot_src.unsqueeze(0).unsqueeze(0)  # [1,1,P,L,M]
        sum_contrib = (contrib_e * src_mask_e).sum(dim=3)  # [1,M,P,M]

        counts = one_hot_src.sum(dim=1)  # [P,M]
        counts_e = counts.unsqueeze(0).unsqueeze(0)
        mean_per_path = torch.where(counts_e > 0, sum_contrib / (counts_e + eps), torch.zeros_like(sum_contrib))  # [1,M,P,M]

        running_sum = running_sum + mean_per_path.sum(dim=2).squeeze(0)

        # Kernel computations
        if s_grid is None:
            inv_beta = 1.0 / (beta + eps)
            inv_beta_flat = inv_beta.reshape(-1)
            inv_beta_flat = inv_beta_flat[torch.isfinite(inv_beta_flat)]
            if inv_beta_flat.numel() == 0:
                s_max = 5.0
            else:
                try:
                    q = float(torch.quantile(inv_beta_flat, float(kernel_quantile)).item())
                except Exception:
                    q = float(inv_beta_flat.median().item())
                s_max = float(max(1e-3, min(100.0, kernel_time_scale_factor * q)))
            s_grid = torch.linspace(0.0, s_max, steps=int(max(2, kernel_num_points)), device=device)

        amplitude = alpha - mu  # [M,P,L]
        amplitude_s = amplitude.unsqueeze(-1) * torch.exp(-(beta.unsqueeze(-1)) * s_grid.view(1, 1, 1, -1))  # [M,P,L,S]
        amplitude_s = amplitude_s * valid_mask.unsqueeze(0).unsqueeze(-1)

        amplitude_s_e = amplitude_s.unsqueeze(-1)  # [M,P,L,S,1]
        src_mask_e = one_hot_src.unsqueeze(0).unsqueeze(3)  # [1,P,L,1,M]
        sum_contrib_s = (amplitude_s_e * src_mask_e).sum(dim=2)  # [M,P,S,M]

        counts = one_hot_src.sum(dim=1)  # [P,M]
        counts_e = counts.unsqueeze(0).unsqueeze(-1)  # [1,P,M,1]
        mean_per_path_s = torch.where(
            counts_e > 0,
            sum_contrib_s.permute(0, 1, 3, 2) / (counts_e + eps),  # [M,P,M,S]
            torch.zeros_like(sum_contrib_s.permute(0, 1, 3, 2)),
        )
        agg_s = mean_per_path_s.sum(dim=1)  # [M,M,S]
        kernels_running_sum = agg_s if kernels_running_sum is None else (kernels_running_sum + agg_s)

        total_paths += p_batch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_matrix = (running_sum / max(1, total_paths)).detach().cpu().numpy().astype(np.float32)
    kernels_mean = (
        (kernels_running_sum / max(1, total_paths)).detach().cpu().numpy().astype(np.float32)
        if kernels_running_sum is not None
        else np.zeros((num_marks, num_marks, int(max(2, kernel_num_points))), dtype=np.float32)
    )
    s_grid_np = (
        s_grid.detach().cpu().numpy().astype(np.float32)
        if s_grid is not None
        else np.linspace(0.0, 5.0, int(max(2, kernel_num_points)), dtype=np.float32)
    )
    return mean_matrix, kernels_mean, s_grid_np


def write_outputs(
    run_dir: Path, matrix: np.ndarray, dataset: str, checkpoint: str, kernels: np.ndarray | None = None, s_grid: np.ndarray | None = None
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    npy_path = run_dir / "interaction_matrix.npy"
    csv_path = run_dir / "interaction_matrix.csv"
    json_path = run_dir / "summary.json"

    np.save(npy_path, matrix)
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["k\\j"] + [str(j) for j in range(matrix.shape[1])])
        for k in range(matrix.shape[0]):
            writer.writerow([str(k)] + [f"{float(v):.6f}" for v in matrix[k]])

    payload = {
        "dataset": dataset,
        "model_checkpoint": checkpoint,
        "matrix_shape": list(matrix.shape),
        "matrix": matrix.tolist(),
    }
    if kernels is not None and s_grid is not None:
        payload["kernels_shape"] = list(kernels.shape)
        payload["s_grid"] = s_grid.tolist()
    json_path.write_text(json.dumps(payload, indent=2))

    # Heatmap visualization (best-effort; skip if matplotlib is unavailable)
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        m = matrix.shape[0]
        fig_size = (min(12, 0.4 * m + 2), min(10, 0.4 * m + 2))
        fig, ax = plt.subplots(figsize=fig_size)
        # Use a diverging colormap centered at 0 so negatives and positives differ,
        # with white at zero.
        sym = float(np.nanmax(np.abs(matrix)))
        if not np.isfinite(sym) or sym == 0.0:
            sym = 1.0
        norm = TwoSlopeNorm(vmin=-sym, vcenter=0.0, vmax=sym)
        im = ax.imshow(matrix, cmap="bwr", norm=norm, interpolation="nearest", aspect="auto")
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("influence", rotation=90, va="center")
        # Tick labels start at 1 for readability
        step = 1 if m <= 25 else max(1, m // 25)
        ticks = list(range(0, m, step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(t + 1) for t in ticks])
        ax.set_yticklabels([str(t + 1) for t in ticks])
        ax.set_xlabel("source mark j")
        ax.set_ylabel("target mark k")
        ax.set_title("Inferred Interaction Matrix (mean over paths)")
        plt.tight_layout()
        fig.savefig(run_dir / "interaction_matrix.png", dpi=200)
        fig.savefig(run_dir / "interaction_matrix.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    # Save kernel arrays and per-entry plots if provided
    if kernels is not None and s_grid is not None:
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            kernels_dir = run_dir / "kernels"
            kernels_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(kernels_dir / "kernels.npz", s=s_grid, phi=kernels)

            m = kernels.shape[0]
            for k in range(m):
                for j in range(m):
                    fig, ax = plt.subplots(figsize=(4.5, 3.2))
                    ax.plot(s_grid, kernels[k, j], lw=1.5)
                    ax.set_xlabel("time lag s")
                    ax.set_ylabel("phi_{k,j}(s)")
                    ax.set_title(f"Kernel k={k + 1} – j={j + 1}")
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    fig.savefig(kernels_dir / f"kernel_k{k + 1}_j{j + 1}.pdf")
                    plt.close(fig)
        except Exception:
            pass


def _is_valid_checkpoint_dir(p: Path) -> bool:
    try:
        return p.is_dir() and (
            (p / "config.json").exists()
            and (((p / "model-checkpoint.pth").exists()) or ((p / "model.safetensors").exists()) or ((p / "pytorch_model.bin").exists()))
        )
    except Exception:
        return False


def _run_single_grid(cfg: Dict) -> Tuple[str, str, Path, int]:
    dataset = str(cfg["dataset"])  # short id or easytpp/...
    checkpoint = Path(cfg["checkpoint"]) if cfg.get("checkpoint") else Path("")
    checkpoint_name = checkpoint.name or (str(cfg.get("checkpoint")) if cfg.get("checkpoint") else "scratch")
    results_root: Path = cfg["results_root"]

    base_name = f"{checkpoint_name}__{(dataset if dataset.startswith('easytpp/') else dataset.replace('/', '-'))}__network__ctx{cfg.get('context_size', 'all')}__inf{cfg.get('inference_size', 'all')}"
    result_dir = results_root / base_name
    result_dir.mkdir(parents=True, exist_ok=True)
    run_log = result_dir / "run.log"

    print(f"[NETWORK] ckpt={checkpoint_name} dataset={dataset} → {result_dir}", flush=True)

    rc = 0
    try:
        if not _is_valid_checkpoint_dir(checkpoint):
            raise FileNotFoundError(f"Invalid checkpoint directory: {checkpoint}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_fimhawkes_with_proper_weights(str(checkpoint))
        model.eval().to(device)

        ctx_raw, inf_raw = load_dataset_hf(dataset, cfg.get("context_size"), cfg.get("inference_size"))
        max_events = None if (cfg.get("max_num_events") is not None and int(cfg.get("max_num_events")) < 0) else cfg.get("max_num_events")
        matrix, kernels, s_grid = infer_interaction_and_kernels(
            model,
            ctx_raw,
            inf_raw,
            max_events,
            device,
            inference_batch_size=cfg.get("inference_batch_size", 16),
            kernel_num_points=int(cfg.get("kernel_num_points", 200)),
            kernel_quantile=float(cfg.get("kernel_quantile", 0.9)),
            kernel_time_scale_factor=float(cfg.get("kernel_time_scale_factor", 8.0)),
        )
        write_outputs(result_dir, matrix, dataset, str(checkpoint), kernels=kernels, s_grid=s_grid)
        rc = 0
    except Exception as e:
        rc = 1
        try:
            run_log.write_text(str(e))
        except Exception:
            pass

    status = "OK" if rc == 0 else f"FAIL(rc={rc})"
    print(f"[NETWORK END] ckpt={checkpoint_name} dataset={dataset} → {status}", flush=True)
    return dataset, checkpoint_name, result_dir, rc


def main():
    ap = argparse.ArgumentParser("Infer hidden mark interaction network for FIM-Hawkes")
    ap.add_argument("--config", type=str, default=None, help="YAML config for grid-style runs")
    ap.add_argument("--checkpoint", type=str, required=False)
    ap.add_argument("--dataset", type=str, required=False, help="HF id (easytpp/...) or short name like 'amazon'")
    ap.add_argument("--run-dir", type=str, required=False)
    ap.add_argument("--context-size", type=int, default=None)
    ap.add_argument("--inference-size", type=int, default=None)
    ap.add_argument("--max-num-events", type=int, default=100, help="-1 means no truncation")
    ap.add_argument("--kernel-num-points", type=int, default=200, help="Number of points for kernel s-grid")
    ap.add_argument("--kernel-quantile", type=float, default=0.9, help="Quantile of 1/beta to set s_max")
    ap.add_argument("--kernel-time-scale-factor", type=float, default=8.0, help="Multiplier for s_max from 1/beta")
    args = ap.parse_args()

    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        if not cfg.get("datasets"):
            raise SystemExit("[ERROR] Config must define 'datasets'.")
        if not cfg.get("checkpoints"):
            raise SystemExit("[ERROR] Config must define 'checkpoints' (at least one).")

        ts_root = datetime.now().strftime("%y%m%d-%H%M")
        base = {
            **{k: v for k, v in cfg.items() if k not in ["datasets", "checkpoints"]},
            "results_root": Path(cfg.get("results_root", "results/fim_network")) / ts_root,
        }
        datasets_list = cfg.get("datasets", [])
        checkpoints_list = cfg.get("checkpoints", [])
        run_confs = [{**base, "dataset": d, "checkpoint": ck} for d in datasets_list for ck in checkpoints_list]
        parallel = int(cfg.get("parallel", 1))
        base["results_root"].mkdir(parents=True, exist_ok=True)
        print(f"[GRID] Total network inferences: {len(run_confs)} | parallel={parallel}")
        if parallel <= 1:
            results = [_run_single_grid(rc) for rc in run_confs]
        else:
            with ThreadPoolExecutor(max_workers=parallel) as ex:
                futures = {ex.submit(_run_single_grid, rc): rc for rc in run_confs}
                results = [f.result() for f in as_completed(futures)]
        ok = sum(1 for _, _, _, rc in results if rc == 0)
        print(f"\nCompleted {ok}/{len(run_confs)} network inferences. Results are in {base['results_root']}")
        return

    # Single-run CLI path
    if not (args.checkpoint and args.dataset and args.run_dir):
        raise SystemExit("Provide --config or all of --checkpoint, --dataset, --run-dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_fimhawkes_with_proper_weights(args.checkpoint)
    model.eval().to(device)

    ctx_raw, inf_raw = load_dataset_hf(args.dataset, args.context_size, args.inference_size)
    max_events = None if (args.max_num_events is not None and args.max_num_events < 0) else args.max_num_events
    matrix, kernels, s_grid = infer_interaction_and_kernels(
        model,
        ctx_raw,
        inf_raw,
        max_events,
        device,
        inference_batch_size=16,
        kernel_num_points=args.kernel_num_points,
        kernel_quantile=args.kernel_quantile,
        kernel_time_scale_factor=args.kernel_time_scale_factor,
    )
    write_outputs(Path(args.run_dir), matrix, str(args.dataset), str(args.checkpoint), kernels=kernels, s_grid=s_grid)


if __name__ == "__main__":
    main()
