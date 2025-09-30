"""
Long-Horizon Event Sequence Forecasting for FIM-Hawkes

Implements the evaluation methodology inspired by:
Zeng et al., "Interacting Diffusion Processes for Event Sequence Forecasting"

Key differences vs next-event prediction:
- For each test sequence of length L, we split into history (first L-N) and
  forecast horizon (last N). We autoregressively generate N future events
  without peeking at ground truth.
- We run the full generation loop multiple times (ensembles) and aggregate the
  trajectories (mean of inter-arrival times, majority vote for types).

CLI usage (invoked by the benchmark runner):
python /cephfs/users/berghaus/FoundationModels/FIM/scripts/hawkes/fim_long_horizon_prediction.py \
  --checkpoint results/FIM_Hawkes_10-22st_nll_mc_only_2000_paths_mixed_100_events_mixed-experiment-seed-10-dataset-dataset_kwargs-field_name_for_dimension_grouping-base_intensity_functions_09-23-1809/checkpoints/best-model \
  --dataset data/external/CDiff_dataset/taobao \
  --run-dir /cephfs/users/berghaus/FoundationModels/FIM/results/tmp \
  --forecast-horizon-size 10 \
  --context-size 2000 \
  --inference-size 1 \
  --plot-event-predictions \
  --plot-path-index 0
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Import the new OTD metric functions
from otd_metrics import get_distances_otd


# Lazy imports for plotting to avoid heavy deps unless requested
try:
    import matplotlib.pyplot as plt  # noqa: E402
except Exception:
    plt = None


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


# ============================
# Utilities: model + data IO
# ============================
def load_fimhawkes_with_proper_weights(checkpoint_path: str) -> FIMHawkes:
    # Delegate loading to the generic loader which reads config.json and model-checkpoint.pth
    return FIMHawkes.load_model(Path(checkpoint_path))


def load_local_dataset(dataset_path: str, split: str) -> Dict:
    split_path = Path(dataset_path) / split
    if not split_path.exists():
        raise FileNotFoundError(f"Split directory not found: {split_path}")

    event_times = torch.load(split_path / "event_times.pt")  # [N, P, K, 1]
    event_types = torch.load(split_path / "event_types.pt")  # [N, P, K, 1]
    seq_lengths = torch.load(split_path / "seq_lengths.pt")  # [N, P]
    time_offsets = torch.load(split_path / "time_offsets.pt")  # [N, P]

    N, P, K = event_times.shape[:3]

    time_since_start: List[List[float]] = []
    time_since_last_event: List[List[float]] = []
    type_event: List[List[int]] = []
    seq_len: List[int] = []
    per_path_offsets: List[float] = []

    for n in range(N):
        for p in range(P):
            L = int(seq_lengths[n, p].item())
            times = event_times[n, p, :L, 0].tolist()
            types = event_types[n, p, :L, 0].tolist()
            deltas = [0.0] + [times[j] - times[j - 1] for j in range(1, len(times))]
            time_since_start.append(times)
            time_since_last_event.append(deltas)
            type_event.append([int(t) for t in types])
            seq_len.append(L)
            per_path_offsets.append(float(time_offsets[n, p].item()))

    return {
        "time_since_start": time_since_start,
        "time_since_last_event": time_since_last_event,
        "type_event": type_event,
        "seq_len": seq_len,
        "time_offsets": per_path_offsets,
    }


def load_cdiff_dataset(dataset: str, split: str) -> Tuple[Dict, Optional[int]]:
    """Load CDiff_dataset pickle split and adapt to internal format.

    Returns (data_dict, num_marks) where data_dict has keys:
    - time_since_start: List[List[float]] (shifted so first event starts at 0)
    - time_since_last_event: List[List[float]] (recomputed from shifted start times)
    - type_event: List[List[int]]
    - seq_len: List[int]
    num_marks is d['dim_process'] when available.
    """
    base = Path(dataset)
    if not base.exists():
        # Try resolving against the repo's CDiff_dataset folder
        candidate = Path(__file__).resolve().parents[2] / "data" / "external" / "CDiff_dataset" / dataset
        if candidate.exists():
            base = candidate
    pkl_path = base / f"{split}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"CDiff dataset split not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    num_marks = d.get("dim_process") if isinstance(d, dict) else None
    seqs_raw = d.get(split)
    if not isinstance(seqs_raw, list):
        raise ValueError(f"Unexpected {split} content in {pkl_path}: {type(seqs_raw)}")

    time_since_start: List[List[float]] = []
    time_since_last_event: List[List[float]] = []
    type_event: List[List[int]] = []
    seq_len: List[int] = []

    for seq in seqs_raw:
        if not isinstance(seq, list) or len(seq) == 0:
            time_since_start.append([])
            time_since_last_event.append([])
            type_event.append([])
            seq_len.append(0)
            continue
        # Shift absolute times to start at zero
        first_time = float(seq[0].get("time_since_start", 0.0))
        times = [float(ev.get("time_since_start", 0.0)) - first_time for ev in seq]
        types = [int(ev.get("type_event", 0)) for ev in seq]
        # Recompute deltas from shifted times
        deltas = [0.0]
        for j in range(1, len(times)):
            deltas.append(float(times[j] - times[j - 1]))

        time_since_start.append(times)
        time_since_last_event.append(deltas)
        type_event.append(types)
        seq_len.append(len(times))

    return {
        "time_since_start": time_since_start,
        "time_since_last_event": time_since_last_event,
        "type_event": type_event,
        "seq_len": seq_len,
    }, (int(num_marks) if num_marks is not None else None)


def detect_num_event_types_from_data(context_data_raw: Dict, inference_data_raw: Dict) -> int:
    unique_types = set()
    for seq in context_data_raw.get("type_event", []):
        unique_types.update(seq)
    for seq in inference_data_raw.get("type_event", []):
        unique_types.update(seq)
    # Get the max type ID and add 1 for the number of classes (0-indexed)
    max_id = max(unique_types) if unique_types else -1
    return int(max_id + 1)


# ============================
# Metrics
# ============================
def calculate_rmse_x(pred_dtimes: torch.Tensor, true_dtimes: torch.Tensor) -> float:
    if pred_dtimes.numel() == 0:
        return 0.0
    se = (pred_dtimes - true_dtimes) ** 2
    return float(torch.sqrt(torch.mean(se)).item())


def calculate_smape(pred_dtimes: torch.Tensor, true_dtimes: torch.Tensor) -> float:
    p = pred_dtimes.abs()
    a = true_dtimes.abs()
    denom = p + a
    num = (pred_dtimes - true_dtimes).abs()
    term = torch.where(denom > 1e-9, num / denom, torch.zeros_like(denom))
    return float(200.0 * torch.mean(term).item())


def calculate_rmse_e(pred_types: torch.Tensor, true_types: torch.Tensor, num_marks: int) -> float:
    counts_pred = torch.bincount(pred_types, minlength=num_marks).float()
    counts_true = torch.bincount(true_types, minlength=num_marks).float()
    se = (counts_pred - counts_true) ** 2
    return float(torch.sqrt(torch.mean(se)).item())


def calculate_type_accuracy(pred_types: torch.Tensor, true_types: torch.Tensor) -> float:
    """Compute per-step event type accuracy between predicted and true types.

    If lengths differ, compare up to the minimum length.
    """
    if pred_types.numel() == 0 or true_types.numel() == 0:
        return 0.0
    m = min(int(pred_types.numel()), int(true_types.numel()))
    return float((pred_types[:m] == true_types[:m]).float().mean().item())


# ============================
# Sampling utilities
# ============================
@torch.no_grad()
def sample_next_event(
    model: FIMHawkes,
    context_batch: Dict[str, torch.Tensor],
    hist_times_1d: torch.Tensor,
    hist_types_1d: torch.Tensor,
    device: torch.device,
    precomputed_enhanced_context: Optional[Dict[str, torch.Tensor]] = None,
    num_marks: Optional[int] = None,
) -> Tuple[float, int]:
    L = int(hist_times_1d.shape[0])
    if L == 0:  # Handle empty history for first prediction
        t_last_tensor = torch.zeros(1, 1, 1, device=device)
    else:
        t_last_tensor = hist_times_1d[-1:].view(1, 1, 1)

    inf_times = hist_times_1d.view(1, 1, L, 1).to(device)
    inf_types = hist_types_1d.to(dtype=torch.long).view(1, 1, L, 1).to(device)
    inf_lengths = torch.tensor([[L]], device=device)

    x = {
        "context_event_times": context_batch["time_seqs"].unsqueeze(0).unsqueeze(-1).to(device),
        "context_event_types": context_batch["type_seqs"].unsqueeze(0).unsqueeze(-1).to(device),
        "context_seq_lengths": context_batch["seq_non_pad_mask"].sum(dim=1).unsqueeze(0).to(device),
        "inference_event_times": inf_times,
        "inference_event_types": inf_types,
        "inference_seq_lengths": inf_lengths,
        "intensity_evaluation_times": torch.zeros(1, 1, 1, device=device),
    }
    if precomputed_enhanced_context is not None:
        x["precomputed_enhanced_context"] = precomputed_enhanced_context
    if num_marks is not None:
        x["num_marks"] = torch.tensor([num_marks], device=device)

    model_out = model.forward(x)
    intensity_obj = model_out["intensity_function"]

    hist_times_sampler = hist_times_1d.view(1, L)
    hist_dtimes_sampler = torch.zeros_like(hist_times_sampler)
    if L > 1:
        hist_dtimes_sampler[:, 1:] = hist_times_sampler[:, 1:] - hist_times_sampler[:, :-1]
    hist_types_sampler = hist_types_1d.view(1, L)

    def intensity_fn_for_sampler(query_times, _ignored):
        return intensity_obj.evaluate(query_times).permute(0, 2, 3, 1)

    if getattr(model.event_sampler, "sampling_method", "thinning") == "inverse_transform":
        accepted_dtimes, weights = model.event_sampler.draw_next_time_one_step_inverse_transform(intensity_obj, compute_last_step_only=True)
    else:
        accepted_dtimes, weights = model.event_sampler.draw_next_time_one_step(
            time_seq=hist_times_sampler,
            time_delta_seq=hist_dtimes_sampler,
            event_seq=hist_types_sampler,
            intensity_fn=intensity_fn_for_sampler,
            compute_last_step_only=True,
        )

    raw_delta_samples = accepted_dtimes - t_last_tensor
    delta_samples = torch.clamp(raw_delta_samples, min=0.0)

    probs_weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
    flat_pw = probs_weights.view(-1)
    sample_idx = torch.multinomial(flat_pw, num_samples=1).item()
    sampled_delta = delta_samples.view(-1)[sample_idx].item()

    intensities_at_samples = intensity_obj.evaluate(accepted_dtimes)
    total_intensity = intensities_at_samples.sum(dim=1, keepdim=True)
    probs_types_all = intensities_at_samples / (total_intensity + 1e-9)
    probs_vec = probs_types_all.squeeze(0).squeeze(1)[:, sample_idx]
    probs_vec = probs_vec / (probs_vec.sum() + 1e-9)
    sampled_type = int(torch.multinomial(probs_vec, num_samples=1).item())

    return sampled_delta, sampled_type


@torch.no_grad()
def generate_long_horizon_forecasts(
    model: FIMHawkes,
    context_batch: Dict[str, torch.Tensor],
    init_history_times: torch.Tensor,
    init_history_types: torch.Tensor,
    horizon_n: int,
    num_ensembles: int,
    device: torch.device,
    precomputed_enhanced_context: Optional[Dict[str, torch.Tensor]] = None,
    num_marks: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    ensemble_dtimes: List[torch.Tensor] = []
    ensemble_types: List[torch.Tensor] = []

    for _ in range(num_ensembles):
        hist_times = init_history_times.clone().to(device)
        hist_types = init_history_types.clone().to(device)
        step_dtimes: List[float] = []
        step_types: List[int] = []
        for _ in range(horizon_n):
            dtime_next, type_next = sample_next_event(
                model,
                context_batch,
                hist_times,
                hist_types,
                device,
                precomputed_enhanced_context=precomputed_enhanced_context,
                num_marks=num_marks,
            )
            last_time = hist_times[-1] if hist_times.numel() > 0 else 0.0
            next_time = last_time + float(dtime_next)
            hist_times = torch.cat([hist_times, torch.tensor([next_time], device=device)])
            hist_types = torch.cat([hist_types, torch.tensor([int(type_next)], device=device)])
            step_dtimes.append(float(dtime_next))
            step_types.append(int(type_next))
        ensemble_dtimes.append(torch.tensor(step_dtimes, dtype=torch.float32, device=device))
        ensemble_types.append(torch.tensor(step_types, dtype=torch.long, device=device))

    dtimes_stack = torch.stack(ensemble_dtimes, dim=0)
    mean_dtimes = torch.mean(dtimes_stack, dim=0)
    types_stack = torch.stack(ensemble_types, dim=0)
    agg_types, _ = torch.mode(types_stack, dim=0)

    return {
        "predicted_event_dtimes": mean_dtimes.detach().cpu(),
        "predicted_event_types": agg_types.detach().cpu(),
    }


# ============================
# Runner
# ============================
def run_long_horizon_evaluation(
    model_checkpoint_path: str,
    dataset: str,
    forecast_horizon_size: int,
    num_ensemble_trajectories: int = 5,
    context_size: Optional[int] = None,
    inference_size: Optional[int] = None,
    max_num_events: Optional[int] = None,
    sample_index: int = 0,
    num_integration_points: int = 5000,
    sampling_method: Optional[str] = None,
    nll_method: Optional[str] = None,
    num_trials: int = 10,
    base_seed: int = 0,
    plot_intensity_predictions: bool = False,
    plot_event_predictions: bool = False,
    plot_path_index: int = 0,
) -> Dict:
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_fimhawkes_with_proper_weights(model_checkpoint_path)
    # Optional: override NLL method from caller
    if nll_method in ("closed_form", "monte_carlo"):
        try:
            nll_cfg = getattr(model.config, "nll", None)
            if isinstance(nll_cfg, dict):
                nll_cfg["method"] = nll_method
            elif nll_cfg is None:
                model.config.nll = {"method": nll_method}
            else:
                setattr(nll_cfg, "method", nll_method)
        except Exception:
            try:
                model.config.nll = {"method": nll_method}
            except Exception:
                pass
    model.eval().to(device)
    model.event_sampler.num_samples_boundary = 50
    if sampling_method in ("thinning", "inverse_transform"):
        model.event_sampler.sampling_method = sampling_method

    # Resolve CDiff_dataset path
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        # Allow dataset to be a short name like "amazon" resolving to repo data folder
        dataset_path = Path(__file__).resolve().parents[2] / "data" / "external" / "CDiff_dataset" / dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found or unsupported: {dataset}")

    # Load CDiff train/test splits, shifting times to start at zero
    train_dataset_dict, num_marks_ctx = load_cdiff_dataset(str(dataset_path), "train")
    test_dataset_dict, num_marks_inf = load_cdiff_dataset(str(dataset_path), "test")
    eff_ctx_size = len(train_dataset_dict["seq_len"]) if context_size is None else context_size
    eff_inf_size = len(test_dataset_dict["seq_len"]) if inference_size is None else inference_size

    def take_subset(d: Dict, size: int) -> Dict:
        return {k: (v[:size] if isinstance(v, list) else v) for k, v in d.items()}

    context_data_raw = take_subset(train_dataset_dict, eff_ctx_size)
    inference_data_raw = take_subset(test_dataset_dict, eff_inf_size)

    def _truncate_batch(data_dict: Dict) -> Dict:
        if max_num_events is None or max_num_events < 0:
            return data_dict
        out = {k: [] for k in ["time_since_start", "time_since_last_event", "type_event", "seq_len"]}
        for times, deltas, types, length in zip(
            data_dict["time_since_start"], data_dict["time_since_last_event"], data_dict["type_event"], data_dict["seq_len"]
        ):
            trunc = min(length, max_num_events)
            out["time_since_start"].append(times[:trunc])
            out["time_since_last_event"].append(deltas[:trunc])
            out["type_event"].append(types[:trunc])
            out["seq_len"].append(trunc)
        if "time_offsets" in data_dict:
            out["time_offsets"] = data_dict["time_offsets"]
        return out

    context_data_raw = _truncate_batch(context_data_raw)
    inference_data_raw = _truncate_batch(inference_data_raw)

    detected_num_marks = (
        num_marks_ctx
        if num_marks_ctx is not None
        else (num_marks_inf if num_marks_inf is not None else detect_num_event_types_from_data(context_data_raw, inference_data_raw))
    )
    model.config.max_num_marks = detected_num_marks

    num_ctx_seqs = len(context_data_raw.get("seq_len", []))
    if num_ctx_seqs > 0 and any(len(s) > 0 for s in context_data_raw["time_since_last_event"]):
        max_dtime_train = max(max(seq) for seq in context_data_raw["time_since_last_event"] if seq)
    else:
        max_dtime_train = 1.0
    model.event_sampler.dtime_max = float(max_dtime_train) * 1.2
    model.event_sampler.over_sample_rate = 5.0

    max_ctx_len = max((len(seq) for seq in context_data_raw["time_since_start"]), default=0)
    ctx_batch = {"time_seqs": [], "type_seqs": [], "seq_len": []}
    for i in range(len(context_data_raw["time_since_start"])):
        pad = max_ctx_len - len(context_data_raw["time_since_start"][i])
        ctx_batch["time_seqs"].append(context_data_raw["time_since_start"][i] + [0] * pad)
        ctx_batch["type_seqs"].append(context_data_raw["type_event"][i] + [0] * pad)
        ctx_batch["seq_len"].append(context_data_raw["seq_len"][i])
    ctx_batch = {k: torch.tensor(v, device=device) for k, v in ctx_batch.items()}
    ctx_batch["seq_non_pad_mask"] = torch.arange(max_ctx_len, device=device).expand(len(ctx_batch["seq_len"]), max_ctx_len) < ctx_batch[
        "seq_len"
    ].unsqueeze(1)

    with torch.no_grad():
        precomp_ctx_tensors = {
            "context_event_times": ctx_batch["time_seqs"].unsqueeze(0).unsqueeze(-1).to(device),
            "context_event_types": ctx_batch["type_seqs"].unsqueeze(0).unsqueeze(-1).to(device),
            "context_seq_lengths": ctx_batch["seq_non_pad_mask"].sum(dim=1).unsqueeze(0).to(device),
        }
        precomputed_enhanced_context = model.encode_context(precomp_ctx_tensors)

    otd_del_costs = [0.05, 0.5, 1, 1.5, 2, 3, 4]
    otd_trans_cost = 1.0

    # Containers for per-trial metrics (averaged over sequences per trial)
    trial_rmsex_plus: List[float] = []
    trial_smape: List[float] = []
    trial_rmse_e: List[float] = []
    trial_otd: List[float] = []
    trial_type_acc: List[float] = []
    num_eval_sequences: int = 0

    total_sequences = len(inference_data_raw.get("seq_len", []))
    for trial_idx in range(max(1, int(num_trials))):
        # Set deterministic but distinct random seeds for each trial
        seed_val = int(base_seed) + trial_idx
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            try:
                torch.cuda.manual_seed_all(seed_val)
            except Exception:
                pass

        rmsex_plus_sum, smape_sum, rmse_e_sum, otd_sum = 0.0, 0.0, 0.0, 0.0
        type_acc_sum = 0.0
        num_eval = 0

        for i in range(total_sequences):
            L = int(inference_data_raw["seq_len"][i])
            N = int(forecast_horizon_size)
            if L <= N:
                continue

            hist_times_list = inference_data_raw["time_since_start"][i][: L - N]
            hist_types_list = inference_data_raw["type_event"][i][: L - N]
            future_dtimes_list = inference_data_raw["time_since_last_event"][i][L - N : L]
            future_types_list = inference_data_raw["type_event"][i][L - N : L]

            hist_times = torch.tensor(hist_times_list, dtype=torch.float32, device=device)
            hist_types = torch.tensor([int(t) for t in hist_types_list], dtype=torch.long, device=device)
            true_future_dtimes = torch.tensor(future_dtimes_list, dtype=torch.float32)
            true_future_types = torch.tensor([int(t) for t in future_types_list], dtype=torch.long)

            forecasts = generate_long_horizon_forecasts(
                model,
                ctx_batch,
                hist_times,
                hist_types,
                N,
                num_ensemble_trajectories,
                device,
                precomputed_enhanced_context,
                detected_num_marks,
            )
            pred_dtimes = forecasts["predicted_event_dtimes"]
            pred_types = forecasts["predicted_event_types"]

            rmsex_plus_sum += calculate_rmse_x(pred_dtimes, true_future_dtimes)
            smape_sum += calculate_smape(pred_dtimes, true_future_dtimes)
            rmse_e_sum += calculate_rmse_e(pred_types, true_future_types, detected_num_marks)
            type_acc_sum += calculate_type_accuracy(pred_types, true_future_types)

            distances = get_distances_otd(
                pred_dt=[pred_dtimes],
                pred_type_result=[pred_types],
                gt_dt=[true_future_dtimes],
                gt_type_result=[true_future_types],
                num_classes=detected_num_marks,
                distance_del_cost=np.array(otd_del_costs),
                trans_cost=otd_trans_cost,
            )
            otd_sum += np.mean(distances[0])
            num_eval += 1

            # Optional: plotting for a single selected path index in the first trial
            if plot_event_predictions and plt is not None and trial_idx == 0 and i == int(plot_path_index):
                try:
                    _plot_next_event_predictions(
                        hist_times,
                        hist_types,
                        true_future_dtimes,
                        true_future_types,
                        pred_dtimes,
                        pred_types,
                        save_path=str(Path(f"event_predictions_long_horizon_seq_{i}.png")),
                    )
                except Exception as _e:
                    print(f"Warning: event prediction plotting failed for seq {i}: {_e}")
            if plot_intensity_predictions and trial_idx == 0 and i == int(plot_path_index):
                try:
                    _plot_intensity_for_sequence(
                        context_data_raw,
                        inference_data_raw,
                        i,
                        detected_num_marks,
                        model,
                        device,
                        save_path=str(Path(f"intensity_comparison_long_horizon_seq_{i}.png")),
                    )
                except Exception as _e:
                    print(f"Warning: intensity plotting failed for seq {i}: {_e}")

        # Keep track of how many sequences were actually evaluated (should be constant across trials)
        num_eval_sequences = num_eval

        if num_eval == 0:
            trial_rmsex_plus.append(0.0)
            trial_smape.append(0.0)
            trial_rmse_e.append(0.0)
            trial_otd.append(0.0)
        else:
            trial_rmsex_plus.append(rmsex_plus_sum / num_eval)
            trial_smape.append(smape_sum / num_eval)
            trial_rmse_e.append(rmse_e_sum / num_eval)
            trial_otd.append(otd_sum / num_eval)
            trial_type_acc.append(type_acc_sum / num_eval)
        if num_eval == 0:
            trial_type_acc.append(0.0)

    duration_seconds = float(time.time() - start_time)

    # Convert to numpy for simple mean/std
    t_rmsex_plus = np.array(trial_rmsex_plus, dtype=np.float64)
    t_smape = np.array(trial_smape, dtype=np.float64)
    t_rmse_e = np.array(trial_rmse_e, dtype=np.float64)
    t_otd = np.array(trial_otd, dtype=np.float64)
    t_type_acc = np.array(trial_type_acc, dtype=np.float64)

    return {
        "dataset": dataset,
        "model_checkpoint": model_checkpoint_path,
        "sampling_method": getattr(model.event_sampler, "sampling_method", "thinning"),
        "num_eval_sequences": int(num_eval_sequences),
        "duration_seconds": duration_seconds,
        "num_trials": int(num_trials),
        "base_seed": int(base_seed),
        "metrics": {
            "model": {
                "rmsex_plus": float(t_rmsex_plus.mean()) if t_rmsex_plus.size > 0 else 0.0,
                "rmsex_plus_std": float(t_rmsex_plus.std(ddof=0)) if t_rmsex_plus.size > 0 else 0.0,
                "smape": float(t_smape.mean()) if t_smape.size > 0 else 0.0,
                "smape_std": float(t_smape.std(ddof=0)) if t_smape.size > 0 else 0.0,
                "rmse_e": float(t_rmse_e.mean()) if t_rmse_e.size > 0 else 0.0,
                "rmse_e_std": float(t_rmse_e.std(ddof=0)) if t_rmse_e.size > 0 else 0.0,
                "otd": float(t_otd.mean()) if t_otd.size > 0 else 0.0,
                "otd_std": float(t_otd.std(ddof=0)) if t_otd.size > 0 else 0.0,
                "type_acc": float(t_type_acc.mean()) if t_type_acc.size > 0 else 0.0,
                "type_acc_std": float(t_type_acc.std(ddof=0)) if t_type_acc.size > 0 else 0.0,
            }
        },
    }


def _plot_next_event_predictions(
    hist_times: torch.Tensor,
    hist_types: torch.Tensor,
    true_future_dtimes: torch.Tensor,
    true_future_types: torch.Tensor,
    pred_dtimes: torch.Tensor,
    pred_types: torch.Tensor,
    save_path: Optional[str] = None,
):
    if plt is None:
        return
    # Build absolute times for true next events
    hist_times_np = hist_times.detach().cpu().numpy()
    last_hist_time = float(hist_times_np[-1]) if hist_times_np.size > 0 else 0.0
    true_future_dtimes_np = true_future_dtimes.detach().cpu().numpy()
    true_future_types_np = true_future_types.detach().cpu().numpy()
    pred_dtimes_np = pred_dtimes.detach().cpu().numpy()
    pred_types_np = pred_types.detach().cpu().numpy()

    # Absolute future times assuming autoregressive continuation
    true_next_abs = np.cumsum(true_future_dtimes_np) + last_hist_time
    pred_next_abs = np.cumsum(pred_dtimes_np) + last_hist_time

    n_steps = len(true_future_dtimes_np)
    height = max(4.0, min(14.0, 0.35 * n_steps + 2.0))
    fig, ax = plt.subplots(1, 1, figsize=(12, height))

    # Build color palette for types
    types_present = sorted(set(map(int, true_future_types_np.tolist())) | set(map(int, pred_types_np.tolist())))
    cmap = plt.get_cmap("tab20")
    colors = {t: cmap(i % 20) for i, t in enumerate(types_present)}

    y_rows = np.arange(n_steps)
    for y in y_rows:
        ax.axhline(y, color="#dddddd", linewidth=0.6, zorder=0)

    for idx in range(n_steps):
        t_true = float(true_next_abs[idx])
        t_pred = float(pred_next_abs[idx])
        type_true = int(true_future_types_np[idx])
        type_pred = int(pred_types_np[idx])
        y = y_rows[idx]

        ax.scatter(
            t_true,
            y,
            color=colors.get(type_true, "#1f77b4"),
            marker="o",
            edgecolors="k",
            linewidths=0.5,
            s=40,
            zorder=3,
        )
        ax.scatter(
            t_pred,
            y,
            color=colors.get(type_pred, "#ff7f0e"),
            marker="x",
            linewidths=1.0,
            s=50,
            zorder=3,
        )

        ax.text(t_true, y + 0.15, f"T{type_true}", fontsize=7, color="#333333", ha="center", va="bottom")
        ax.text(t_pred, y - 0.15, f"P{type_pred}", fontsize=7, color="#333333", ha="center", va="top")

    # Label rows with REAL transition indices: (L_hist → L_hist+1), ...
    hist_len = int(hist_times.shape[0])
    ax.set_yticks(y_rows)
    ax.set_yticklabels([f"{k}→{k + 1}" for k in range(hist_len, hist_len + n_steps)])
    ax.invert_yaxis()
    ax.set_xlabel("Event time")
    ax.set_ylabel("Step (k→k+1)")
    ax.set_title("Long-horizon predictions: rows = steps, circles = True, crosses = Predicted; color = Type")

    from matplotlib.lines import Line2D

    marker_handles = [
        Line2D([0], [0], marker="o", color="k", markerfacecolor="w", markersize=7, linestyle="None", label="True"),
        Line2D([0], [0], marker="x", color="k", markersize=7, linestyle="None", label="Predicted"),
    ]
    type_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markeredgecolor="k",
            markerfacecolor=colors[t],
            markersize=7,
            linestyle="None",
            label=f"Type {t}",
        )
        for t in types_present
    ]
    legend1 = ax.legend(handles=marker_handles, loc="upper right", title="Kind")
    ax.add_artist(legend1)
    ax.legend(handles=type_handles, loc="lower right", title="Types", ncol=min(3, len(type_handles)))
    fig.tight_layout()
    if save_path is None:
        save_path = "event_predictions_long_horizon.png"
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_intensity_for_sequence(
    context_data_raw: Dict,
    inference_data_raw: Dict,
    inf_idx: int,
    num_marks: int,
    model: FIMHawkes,
    device: torch.device,
    save_path: Optional[str] = None,
):
    # Defer heavy import
    from visualize_intensity_predictions import plot_intensity_comparison, prepare_batch_for_model

    # Build a single-sample with multiple paths: all context paths + one inference path
    all_times: List[List[float]] = []
    all_types: List[List[int]] = []
    all_lens: List[int] = []

    for t_seq, ty_seq, L in zip(context_data_raw["time_since_start"], context_data_raw["type_event"], context_data_raw["seq_len"]):
        all_times.append(list(map(float, t_seq)))
        all_types.append([int(x) for x in ty_seq])
        all_lens.append(int(L))

    # Append the chosen inference sequence as the last path
    all_times.append(list(map(float, inference_data_raw["time_since_start"][inf_idx])))
    all_types.append([int(x) for x in inference_data_raw["type_event"][inf_idx]])
    all_lens.append(int(inference_data_raw["seq_len"][inf_idx]))

    if len(all_lens) < 2:
        print("Skipping intensity plot: need at least 2 paths (context + inference).")
        return

    max_L = max(all_lens) if all_lens else 0
    times_padded: List[List[float]] = []
    types_padded: List[List[int]] = []
    for t_seq, c_seq, L in zip(all_times, all_types, all_lens):
        pad = max_L - L
        times_padded.append(t_seq + [0.0] * pad)
        types_padded.append(c_seq + [0] * pad)

    # Build tensors with shapes expected by the helper: [P_paths, L, 1] then helper adds batch dim
    event_times = torch.tensor(times_padded, dtype=torch.float32).unsqueeze(-1)
    event_types = torch.tensor(types_padded, dtype=torch.long).unsqueeze(-1)
    seq_lengths = torch.tensor(all_lens, dtype=torch.long)

    single_sample = {
        "event_times": event_times,
        "event_types": event_types,
        "seq_lengths": seq_lengths,
    }

    # The last path is the inference path
    inference_path_idx = len(all_lens) - 1
    model_data_vis = prepare_batch_for_model(single_sample, inference_path_idx=inference_path_idx, num_points_between_events=10)
    model_data_vis["num_marks"] = int(num_marks)

    # Move to device
    for key, val in model_data_vis.items():
        if torch.is_tensor(val):
            model_data_vis[key] = val.to(device)

    with torch.no_grad():
        model_output_vis = model(model_data_vis)

    if save_path is None:
        save_path = f"intensity_comparison_long_horizon_seq_{inf_idx}.png"
    plot_intensity_comparison(model_output_vis, model_data_vis, save_path=save_path, path_idx=0)


def main():
    ap = argparse.ArgumentParser("Run long-horizon FIM-Hawkes forecasting evaluation")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--forecast-horizon-size", type=int, required=True)
    ap.add_argument("--num-ensemble-trajectories", type=int, default=5)
    ap.add_argument("--context-size", type=int, default=None)
    ap.add_argument("--inference-size", type=int, default=None)
    ap.add_argument("--max-num-events", type=int, default=None)
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--num-integration-points", type=int, default=5000)
    ap.add_argument("--sampling-method", type=str, choices=["thinning", "inverse_transform"], default=None)
    ap.add_argument("--nll-method", type=str, choices=["closed_form", "monte_carlo"], default=None)
    ap.add_argument("--num-trials", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--plot-intensity-predictions", action="store_true")
    ap.add_argument("--plot-event-predictions", action="store_true")
    ap.add_argument("--plot-path-index", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    max_num_events = None if (args.max_num_events is not None and args.max_num_events < 0) else args.max_num_events

    try:
        result = run_long_horizon_evaluation(
            model_checkpoint_path=args.checkpoint,
            dataset=args.dataset,
            forecast_horizon_size=args.forecast_horizon_size,
            num_ensemble_trajectories=int(args.num_ensemble_trajectories),
            context_size=args.context_size,
            inference_size=args.inference_size,
            max_num_events=max_num_events,
            sample_index=args.sample_idx,
            num_integration_points=args.num_integration_points,
            sampling_method=args.sampling_method,
            nll_method=args.nll_method,
            num_trials=int(args.num_trials),
            base_seed=int(args.base_seed),
            plot_intensity_predictions=bool(args.plot_intensity_predictions),
            plot_event_predictions=bool(args.plot_event_predictions),
            plot_path_index=int(args.plot_path_index),
        )
        status = "OK"
    except Exception as e:
        import traceback

        result = {
            "dataset": args.dataset,
            "model_checkpoint": str(args.checkpoint),
            "status": "FAIL",
            "error": f"{e}\n{traceback.format_exc()}",
        }
        status = "FAIL"

    result.setdefault("status", status)
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    if status != "OK":
        sys.exit(1)


if __name__ == "__main__":
    main()
