"""
Example usage:

conda activate model_training
python varying_context_size/varying_context_paths_experiment.py \
  --dataset-root /cephfs/users/berghaus/FoundationModels/FIM/data/synthetic_data/hawkes/EVAL_22D_2k_context_paths_100_inference_paths_const_base_exp_kernel_no_interactions \
  --checkpoint /cephfs/users/berghaus/FoundationModels/FIM/results/.ICLR_submission_model/checkpoints/best-model \
  --output-csv /cephfs/users/berghaus/FoundationModels/FIM/results/varying_context_size/const_base_exp_kernel_no_interactions_intensity_error_vs_context_paths.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# Ensure we can import fim.* when executed from repo root
try:
    from fim.models.hawkes import FIMHawkes
except Exception as e:
    raise RuntimeError(
        "Failed to import fim.models.hawkes.FIMHawkes. Please run this script from the repo root "
        "or set PYTHONPATH to include the 'src' directory."
    ) from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Vary FIM-Hawkes performance vs. number of context paths (synthetic dataset)")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="results/.ICLR_submission_model/checkpoints/best-model",
        help="Path to FIM-Hawkes checkpoint directory (with config.json + model-checkpoint.pth or model.safetensors)",
    )
    p.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Synthetic dataset root containing 'context/' and 'test/' folders with .pt tensors",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        default="results/varying_context_size/intensity_vs_context_paths.csv",
        help="Destination CSV for aggregated results",
    )
    p.add_argument(
        "--context-sizes",
        type=str,
        default=None,
        help="Comma-separated list of context sizes to evaluate (e.g., '2000,1900,1800,100'). "
        "If not provided, will infer max and step down by 100 to 100.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Force device; default selects cuda if available",
    )
    p.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optionally truncate sequences to this many events for both context and test (<=0 means no limit)",
    )
    return p.parse_args()


def load_split_tensors(split_dir: Path) -> Dict[str, torch.Tensor]:
    """
    Load required tensors for a split directory.
    Expected files (present in both 'context' and 'test'):
      - event_times.pt: [N, L, 1] or [N, P, L, 1]
      - event_types.pt: [N, L, 1] or [N, P, L, 1]
      - seq_lengths.pt: [N] or [N, P]
      - time_offsets.pt: [N] or [N, P]
    For 'test', also:
      - kernel_functions.pt: [N, M, M] (byte-encoded strings)
      - base_intensity_functions.pt: [N, M] (byte-encoded strings)
    """
    data: Dict[str, torch.Tensor] = {}

    def _load(name: str) -> torch.Tensor:
        path = split_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing required tensor: {path}")
        return torch.load(path)

    data["event_times"] = _load("event_times")
    data["event_types"] = _load("event_types")
    data["seq_lengths"] = _load("seq_lengths")
    # time_offsets may be absent in some synthetic exports; handle gracefully
    to_path = split_dir / "time_offsets.pt"
    data["time_offsets"] = torch.load(to_path) if to_path.exists() else None

    # Optional ground-truth functions (typically only in 'test')
    kf = split_dir / "kernel_functions.pt"
    bif = split_dir / "base_intensity_functions.pt"
    data["kernel_functions"] = torch.load(kf) if kf.exists() else None
    data["base_intensity_functions"] = torch.load(bif) if bif.exists() else None
    return data


def pad_to_max_len(
    times_list: List[List[float]], types_list: List[List[int]], lens_list: List[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_L = max(lens_list) if lens_list else 0
    padded_times = []
    padded_types = []
    for t_seq, ty_seq, L in zip(times_list, types_list, lens_list):
        pad = max_L - L
        padded_times.append(t_seq + [0.0] * pad)
        padded_types.append(ty_seq + [0] * pad)
    time_seqs = torch.tensor(padded_times, dtype=torch.float32) if padded_times else torch.zeros(0, max_L, dtype=torch.float32)
    type_seqs = torch.tensor(padded_types, dtype=torch.long) if padded_types else torch.zeros(0, max_L, dtype=torch.long)
    seq_len = torch.tensor(lens_list, dtype=torch.long) if lens_list else torch.zeros(0, dtype=torch.long)
    seq_non_pad_mask = (
        torch.arange(max_L).expand(len(lens_list), max_L) < seq_len.unsqueeze(1) if lens_list else torch.zeros(0, max_L, dtype=torch.bool)
    )
    return time_seqs, type_seqs, seq_len, seq_non_pad_mask


def to_lists_from_tensors(
    event_times: torch.Tensor,
    event_types: torch.Tensor,
    seq_lengths: torch.Tensor,
    max_events: Optional[int],
) -> Tuple[List[List[float]], List[List[int]], List[int], List[int], List[int], Tuple[int, int]]:
    """
    Convert [N, L, 1] or [N, P, L, 1] tensors into python lists per path.
    If shape is [N, P, L, 1], flatten the first two dims into one list of paths.
    """
    if event_times.dim() == 4:
        N, P, L, _ = event_times.shape
        times = event_times.reshape(N * P, L, 1)
        types = event_types.reshape(N * P, L, 1)
        lengths = seq_lengths.reshape(N * P)
        sample_indices = []
        path_indices = []
        for n in range(N):
            for p in range(P):
                sample_indices.append(n)
                path_indices.append(p)
        orig_np = (N, P)
    elif event_times.dim() == 3:
        N, L, _ = event_times.shape
        times = event_times
        types = event_types
        lengths = seq_lengths
        sample_indices = list(range(N))
        path_indices = [0 for _ in range(N)]
        orig_np = (N, 1)
    else:
        raise ValueError(f"Unexpected event_times shape: {tuple(event_times.shape)}")

    time_lists: List[List[float]] = []
    type_lists: List[List[int]] = []
    len_lists: List[int] = []
    for i in range(times.shape[0]):
        L_i = int(lengths[i].item())
        if max_events is not None and max_events > 0:
            L_i = min(L_i, int(max_events))
        t_seq = times[i, :L_i, 0].tolist()
        ty_seq = [int(x) for x in types[i, :L_i, 0].tolist()]
        time_lists.append([float(x) for x in t_seq])
        type_lists.append(ty_seq)
        len_lists.append(L_i)
    return time_lists, type_lists, len_lists, sample_indices, path_indices, orig_np


def smape_masked(pred: torch.Tensor, true: torch.Tensor, valid_mask: torch.Tensor) -> float:
    """
    pred/true: [M, L]
    valid_mask: [L] (True for valid events)
    """
    base_mask = valid_mask.unsqueeze(0).expand_as(pred)
    numerator = 2.0 * torch.abs(pred - true)
    denominator = torch.abs(pred) + torch.abs(true)
    # Only consider locations where denominator > 1e-8
    denom_mask = denominator > 1e-8
    mask = base_mask & denom_mask
    safe_den = torch.where(denom_mask, denominator, torch.ones_like(denominator))
    sm = (numerator / safe_den) * mask
    denom = mask.sum().clamp(min=1)
    return float(sm.sum().item() / float(denom))


def rmse_masked(pred: torch.Tensor, true: torch.Tensor, valid_mask: torch.Tensor) -> float:
    """
    pred/true: [M, L]
    valid_mask: [L]
    """
    mask = valid_mask.unsqueeze(0).expand_as(pred)
    se = ((pred - true) ** 2) * mask
    denom = mask.sum().clamp(min=1)
    return float(torch.sqrt(se.sum() / denom).item())


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load model
    model: FIMHawkes = FIMHawkes.load_model(Path(args.checkpoint))
    model.eval().to(device)

    dataset_root = Path(args.dataset_root)
    context_dir = dataset_root / "context"
    test_dir = dataset_root / "test"
    if not context_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Expected 'context' and 'test' subfolders under {dataset_root}")

    # Load splits
    context_data = load_split_tensors(context_dir)
    test_data = load_split_tensors(test_dir)

    # Derive lists for context paths (flattened) and keep per-sample indices
    ctx_times_list, ctx_types_list, ctx_lens_list, ctx_sample_idx, ctx_path_idx, (N_ctx, P_ctx) = to_lists_from_tensors(
        context_data["event_times"], context_data["event_types"], context_data["seq_lengths"], args.max_events
    )

    # Derive lists for test paths
    tst_times_list, tst_types_list, tst_lens_list, tst_sample_idx, tst_path_idx, (N_tst, P_tst) = to_lists_from_tensors(
        test_data["event_times"], test_data["event_types"], test_data["seq_lengths"], args.max_events
    )

    # Determine number of marks from base_intensity_functions tensor if available
    num_marks: Optional[int] = None
    bif_tensor = test_data.get("base_intensity_functions")
    if bif_tensor is not None:
        if bif_tensor.dim() == 2:
            # [N, M]
            num_marks = int(bif_tensor.shape[1])
        elif bif_tensor.dim() == 3:
            # [N, M, L_bytes]
            num_marks = int(bif_tensor.shape[1])
    if num_marks is None:
        # Fallback: infer from event_types
        max_type = 0
        for seq in tst_types_list or []:
            if seq:
                max_type = max(max_type, max(seq))
        num_marks = int(max_type + 1)
    model.config.max_num_marks = int(num_marks)

    # Build context sizes list (limit by paths-per-sample, not global flattened count)
    if isinstance(context_data.get("event_times"), torch.Tensor) and context_data["event_times"].dim() == 4:
        max_paths_per_sample = int(context_data["event_times"].shape[1])
    else:
        # Fall back to unique per-sample count in flattened view
        # Count how many entries belong to the first sample index
        first_sample = ctx_sample_idx[0] if ctx_sample_idx else 0
        max_paths_per_sample = sum(1 for s in ctx_sample_idx if s == first_sample)

    if args.context_sizes is not None:
        sizes = [int(s.strip()) for s in args.context_sizes.split(",") if s.strip()]
    else:
        max_ctx = max_paths_per_sample
        step = 100
        sizes = list(range(max_ctx, 99, -step))
        if sizes and sizes[-1] != 100:
            sizes.append(100)

    # Prepare CSV
    out_csv = Path(args.output_csv)
    ensure_dir(out_csv)
    if not out_csv.exists():
        with out_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["dataset_root", "checkpoint", "context_size", "num_test_paths", "smape_mean", "smape_std", "rmse_mean", "rmse_std"]
            )

    # Preload GT functions (decode lazily per test path)
    kernel_functions_tensor = test_data.get("kernel_functions")
    base_functions_tensor = test_data.get("base_intensity_functions")
    if kernel_functions_tensor is None or base_functions_tensor is None:
        raise FileNotFoundError("Ground-truth kernel/base intensity tensors not found in test split; cannot compute intensity metrics.")

    # Cache for precomputed enhanced context per (sample_idx, K)
    precomp_cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

    # For each K
    for K in sizes:
        K = int(min(K, max_paths_per_sample))
        if K <= 0:
            continue

        # Accumulators
        smape_vals: List[float] = []
        rmse_vals: List[float] = []

        # Iterate over test paths
        total_paths = len(tst_times_list)
        for i in range(total_paths):
            times_i = tst_times_list[i]
            types_i = tst_types_list[i]
            L_i = tst_lens_list[i]
            if L_i <= 0:
                continue
            sample_i = tst_sample_idx[i] if i < len(tst_sample_idx) else 0

            # Build or reuse context for this sample_i
            cache_key = (int(sample_i), int(K))
            bundle = precomp_cache.get(cache_key)
            if bundle is None:
                # Select only context paths belonging to this sample
                indices_for_sample = [idx for idx, s in enumerate(ctx_sample_idx) if s == sample_i]
                selected_indices = indices_for_sample[:K]
                sel_ctx_times = [ctx_times_list[j] for j in selected_indices]
                sel_ctx_types = [ctx_types_list[j] for j in selected_indices]
                sel_ctx_lens = [ctx_lens_list[j] for j in selected_indices]
                ctx_time_seqs, ctx_type_seqs, ctx_seq_len, ctx_seq_non_pad_mask = pad_to_max_len(sel_ctx_times, sel_ctx_types, sel_ctx_lens)
                precomp_ctx = {
                    "context_event_times": ctx_time_seqs.unsqueeze(0).unsqueeze(-1).to(device),
                    "context_event_types": ctx_type_seqs.unsqueeze(0).unsqueeze(-1).to(device),
                    "context_seq_lengths": ctx_seq_non_pad_mask.sum(dim=1).unsqueeze(0).to(device),
                }
                enhanced_context = model.encode_context(precomp_ctx)
                bundle = {**precomp_ctx, "enhanced_context": enhanced_context}
                precomp_cache[cache_key] = bundle
            enhanced_context = bundle["enhanced_context"]

            # Inference tensors (single path)
            inf_times = torch.tensor(times_i, dtype=torch.float32, device=device).view(1, 1, L_i, 1)
            inf_types = torch.tensor(types_i, dtype=torch.long, device=device).view(1, 1, L_i, 1)
            inf_lens = torch.tensor([[L_i]], dtype=torch.long, device=device)

            # Evaluation at the event times themselves
            eval_times = torch.tensor(times_i, dtype=torch.float32, device=device).view(1, 1, L_i)

            # Build forward input
            x = {
                "context_event_times": bundle["context_event_times"],
                "context_event_types": bundle["context_event_types"],
                "context_seq_lengths": bundle["context_seq_lengths"],
                "inference_event_times": inf_times,
                "inference_event_types": inf_types,
                "inference_seq_lengths": inf_lens,
                # We pass event times to evaluate intensities exactly at event positions
                "intensity_evaluation_times": eval_times,
                "num_marks": torch.tensor([num_marks], dtype=torch.long, device=device),
                # Provide precomputed context to skip recomputation
                "precomputed_enhanced_context": enhanced_context,
            }

            # Attach GT function tensors for this test path
            # Expect shapes: kernel_functions: [N, M, M], base_intensity_functions: [N, M]
            # Align index i with test path i
            # Select functions for the corresponding sample index
            if kernel_functions_tensor.dim() == 3:
                # [N, M, M]
                kf_i = kernel_functions_tensor[sample_i].unsqueeze(0)
            elif kernel_functions_tensor.dim() == 4:
                # [N, M, M, L_bytes]
                kf_i = kernel_functions_tensor[sample_i].unsqueeze(0)
            else:
                raise ValueError(f"Unexpected kernel_functions tensor shape: {tuple(kernel_functions_tensor.shape)}")
            if base_functions_tensor.dim() == 2:
                # [N, M]
                bif_i = base_functions_tensor[sample_i].unsqueeze(0)
            elif base_functions_tensor.dim() == 3:
                # [N, M, L_bytes]
                bif_i = base_functions_tensor[sample_i].unsqueeze(0)
            else:
                raise ValueError(f"Unexpected base_intensity_functions tensor shape: {tuple(base_functions_tensor.shape)}")
            x["kernel_functions"] = kf_i.to(device)
            x["base_intensity_functions"] = bif_i.to(device)

            # If time offsets are available for test, use it for targets
            if test_data.get("time_offsets") is not None:
                # time_offsets may be [N] or [N, P]; we use scalar per path i
                t_off = test_data["time_offsets"]
                if t_off.dim() == 2:
                    # [N, P]: use matching sample/path indices
                    path_i = tst_path_idx[i] if i < len(tst_path_idx) else 0
                    offset_val = float(t_off[sample_i, path_i].item())
                else:
                    # [N]: use matching sample index
                    offset_val = float(t_off[sample_i].item())
                x["inference_time_offsets"] = torch.tensor([[offset_val]], dtype=torch.float32, device=device)

            # Run forward (includes normalization + intensity function construction)
            out = model.forward(x)
            intensity_fn = out["intensity_function"]

            # Ensure we have predictions/targets at the provided eval times (original scale after _denormalize_output)
            # predicted_intensity_values: [B=1, M, P=1, L]
            y_pred = out["predicted_intensity_values"][0, :, 0, :]  # [M, L_all]

            # If targets not produced (should be, since we provided GT functions), compute via helper
            if "target_intensity_values" in out:
                y_true = out["target_intensity_values"][0, :, 0, :]  # [M, L_all]
            else:
                # Fallback path (should rarely trigger)
                # Compute targets at eval_times using helper; need normalized domain inputs
                # Use model utilities to keep consistency
                norm_consts = (
                    intensity_fn.norm_constants
                    if getattr(intensity_fn, "norm_constants", None) is not None
                    else torch.ones(1, device=device)
                )
                y_true = model.compute_target_intensity_values(
                    *model._decode_functions(kf_i, bif_i),
                    intensity_evaluation_times=eval_times,  # original scale, helper rescales internally
                    inference_event_times=inf_times,
                    inference_event_types=inf_types,
                    inference_seq_lengths=inf_lens.view(1, 1),
                    norm_constants=norm_consts,
                    num_marks=int(num_marks),
                    inference_time_offsets=x.get("inference_time_offsets", None),
                )[0, :, 0, :]  # [M, L]

            # Valid mask by sequence length
            # Forward evaluates at concatenated times: [orig_eval, event_times] -> take the last L_i positions
            y_pred = y_pred[:, -L_i:]
            y_true = y_true[:, -L_i:]
            valid_mask = torch.arange(L_i, device=device) < L_i

            smape_vals.append(smape_masked(y_pred, y_true, valid_mask))
            rmse_vals.append(rmse_masked(y_pred, y_true, valid_mask))

        # Aggregate
        smape_arr = np.array(smape_vals, dtype=np.float64) if smape_vals else np.zeros(0, dtype=np.float64)
        rmse_arr = np.array(rmse_vals, dtype=np.float64) if rmse_vals else np.zeros(0, dtype=np.float64)
        row = [
            str(dataset_root),
            str(args.checkpoint),
            int(K),
            int(len(smape_vals)),
            float(smape_arr.mean()) if smape_arr.size > 0 else 0.0,
            float(smape_arr.std(ddof=0)) if smape_arr.size > 0 else 0.0,
            float(rmse_arr.mean()) if rmse_arr.size > 0 else 0.0,
            float(rmse_arr.std(ddof=0)) if rmse_arr.size > 0 else 0.0,
        ]
        with out_csv.open("a", newline="") as fh:
            csv.writer(fh).writerow(row)


if __name__ == "__main__":
    main()
