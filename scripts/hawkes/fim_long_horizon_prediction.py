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
python3 scripts/hawkes/fim_long_horizon_prediction.py \
  --checkpoint <dir> --dataset <easytpp/...|local_path> --run-dir <dir> \
  [--context-size <int>] [--inference-size <int>] [--max-num-events <int|-1>] \
  [--sample-idx <int>] [--num-integration-points <int>] \
  --forecast-horizon-size <int> --num-ensemble-trajectories <int>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from safetensors import safe_open


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
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in checkpoint directory: {config_path}")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    if "model_type" not in config_dict:
        config_dict["model_type"] = "fimhawkes"

    config = FIMHawkesConfig.from_dict(config_dict)
    model = FIMHawkes(config)

    safetensors_path = checkpoint_path / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in checkpoint directory: {safetensors_path}")

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}

    model.load_state_dict(state_dict, strict=False)
    return model


def load_local_dataset(dataset_path: str, split: str) -> Dict:
    split_path = Path(dataset_path) / split
    if not split_path.exists():
        raise FileNotFoundError(f"Split directory not found: {split_path}")

    event_times = torch.load(split_path / "event_times.pt")  # [N, P, K, 1]
    event_types = torch.load(split_path / "event_types.pt")  # [N, P, K, 1]
    seq_lengths = torch.load(split_path / "seq_lengths.pt")  # [N, P]
    time_offsets = torch.load(split_path / "time_offsets.pt")  # [N, P]

    # For local sets, we evaluate per-path; flatten across paths for HF-like format
    N, P, K = event_times.shape[:3]

    # Convert each path into a separate sequence in list form
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
        "time_offsets": per_path_offsets,  # list[float] aligned with sequences
    }


def detect_num_event_types_from_data(context_data_raw: Dict, inference_data_raw: Dict) -> int:
    unique_types = set()
    for seq in context_data_raw.get("type_event", []):
        unique_types.update(seq)
    for seq in inference_data_raw.get("type_event", []):
        unique_types.update(seq)
    return max(1, int(len(unique_types) or 1))


# ============================
# Metrics
# ============================
def calculate_rmse_x(pred_dtimes: torch.Tensor, true_dtimes: torch.Tensor) -> float:
    """RMSEx+: RMSE on inter-arrival times of length N.
    Inputs: 1D tensors shape [N]."""
    if pred_dtimes.numel() == 0:
        return 0.0
    se = (pred_dtimes - true_dtimes) ** 2
    return float(torch.sqrt(torch.mean(se)).item())


def calculate_smape(pred_dtimes: torch.Tensor, true_dtimes: torch.Tensor) -> float:
    """sMAPE (%) on inter-arrival times of length N.
    sMAPE = (100/N) * sum(|p-a| / (|p|+|a|)) with 0 when both are 0."""
    p = pred_dtimes.abs()
    a = true_dtimes.abs()
    denom = p + a
    num = (pred_dtimes - true_dtimes).abs()
    # define term 0 when denom == 0
    term = torch.where(denom > 0, num / denom, torch.zeros_like(denom))
    return float(100.0 * torch.mean(term).item())


def calculate_rmse_e(pred_types: torch.Tensor, true_types: torch.Tensor, num_marks: int) -> float:
    """RMSEe on type count vectors over the N-step horizon.
    Inputs: 1D tensors length N with integer type ids; num_marks gives vector length.
    """
    counts_pred = torch.zeros(num_marks, dtype=torch.float32, device=pred_types.device)
    counts_true = torch.zeros(num_marks, dtype=torch.float32, device=true_types.device)
    for t in pred_types.tolist():
        if 0 <= int(t) < num_marks:
            counts_pred[int(t)] += 1.0
    for t in true_types.tolist():
        if 0 <= int(t) < num_marks:
            counts_true[int(t)] += 1.0
    se = (counts_pred - counts_true) ** 2
    return float(torch.sqrt(torch.mean(se)).item())


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
    """Sample a single next event (dtime, type) given full history.

    hist_times_1d, hist_types_1d: tensors of shape [L]
    Returns: (delta_time, event_type)
    """
    L = int(hist_times_1d.shape[0])

    # Build 4D tensors for the model forward
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

    # Prepare sequences for the sampler: absolute times and inter-arrival times
    hist_times = x["inference_event_times"].squeeze(0).squeeze(-1)  # [P=1, L]
    hist_dtimes = torch.zeros_like(hist_times)
    hist_dtimes[:, 1:] = hist_times[:, 1:] - hist_times[:, :-1]
    hist_types = x["inference_event_types"].squeeze(0).squeeze(-1)

    def intensity_fn_for_sampler(query_times, _ignored):
        # intensity_obj.evaluate -> [B, M, P, T]
        return intensity_obj.evaluate(query_times).permute(0, 2, 3, 1)  # [B, P, T, M]

    accepted_dtimes, weights = model.event_sampler.draw_next_time_one_step(
        time_seq=hist_times,
        time_delta_seq=hist_dtimes,
        event_seq=hist_types,
        intensity_fn=intensity_fn_for_sampler,
        compute_last_step_only=True,
    )

    # accepted_dtimes are absolute timestamps; get deltas relative to last observed
    t_last_tensor = hist_times[:, -1:].unsqueeze(-1)
    raw_delta_samples = accepted_dtimes - t_last_tensor
    # Robustness
    delta_samples = torch.clamp(raw_delta_samples, min=0.0)

    # Sample one candidate index according to weights
    probs_weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)  # [B=1, P=1, K]
    flat_pw = probs_weights.view(-1)
    sample_idx = torch.multinomial(flat_pw, num_samples=1).item()

    # Select sampled delta time (scalar)
    sampled_delta = delta_samples.view(-1)[sample_idx].item()

    # Compute event-type distribution at candidate times and sample a type
    intensities_at_samples = intensity_obj.evaluate(accepted_dtimes)  # [B, M, P, K]
    total_intensity = intensities_at_samples.sum(dim=1, keepdim=True)  # [B, 1, P, K]
    probs_types_all = intensities_at_samples / (total_intensity + 1e-9)
    # Grab the probabilities at selected sample index along K
    probs_types = probs_types_all.squeeze(0).squeeze(1)  # [M, K] when P=1
    probs_vec = probs_types[:, sample_idx]
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
    """Run S independent autoregressive forecasts of length N and aggregate.

    Returns a dict with aggregated 1D tensors of length N:
    - predicted_event_dtimes: mean of dtimes across ensembles
    - predicted_event_types: majority vote across ensembles (ties -> lowest id)
    """
    ensemble_dtimes: List[torch.Tensor] = []
    ensemble_types: List[torch.Tensor] = []

    for s in range(num_ensembles):
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
            # Append to history (absolute time grows by delta)
            last_time = hist_times[-1]
            next_time = last_time + float(dtime_next)
            hist_times = torch.cat([hist_times, torch.tensor([next_time], device=device)])
            hist_types = torch.cat([hist_types, torch.tensor([int(type_next)], device=device)])
            step_dtimes.append(float(dtime_next))
            step_types.append(int(type_next))
        ensemble_dtimes.append(torch.tensor(step_dtimes, dtype=torch.float32, device=device))
        ensemble_types.append(torch.tensor(step_types, dtype=torch.long, device=device))

    # Aggregate
    dtimes_stack = torch.stack(ensemble_dtimes, dim=0)  # [S, N]
    mean_dtimes = torch.mean(dtimes_stack, dim=0)
    types_stack = torch.stack(ensemble_types, dim=0)  # [S, N]
    # Majority vote per position
    agg_types: List[int] = []
    for k in range(horizon_n):
        vals = types_stack[:, k].tolist()
        counts: Dict[int, int] = {}
        for v in vals:
            counts[int(v)] = counts.get(int(v), 0) + 1
        # pick max count; tie broken by lower type id
        best_type = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        agg_types.append(int(best_type))

    return {
        "predicted_event_dtimes": mean_dtimes.detach().cpu(),
        "predicted_event_types": torch.tensor(agg_types, dtype=torch.long),
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
    num_integration_points: int = 5000,  # kept for API parity; sampler may use this internally
) -> Dict:
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_fimhawkes_with_proper_weights(model_checkpoint_path)
    model.eval().to(device)
    # Calibrate sampler bounds
    model.event_sampler.num_samples_boundary = 50

    # Resolve dataset source
    use_easytpp = isinstance(dataset, str) and str(dataset).startswith("easytpp/")

    if use_easytpp:
        train_dataset = load_dataset(dataset, split="train")
        test_dataset = load_dataset(dataset, split="test")
        effective_context_size = len(train_dataset) if context_size is None else context_size
        effective_inference_size = len(test_dataset) if inference_size is None else inference_size
        context_data_raw = train_dataset[:effective_context_size]
        inference_data_raw = test_dataset[:effective_inference_size]
    else:
        train_dataset_dict = load_local_dataset(dataset, "context")
        test_dataset_dict = load_local_dataset(dataset, "test")
        effective_context_size = len(train_dataset_dict["seq_len"]) if context_size is None else context_size
        effective_inference_size = len(test_dataset_dict["seq_len"]) if inference_size is None else inference_size

        # slice
        def take_subset(d: Dict, size: int) -> Dict:
            return {k: (v[:size] if isinstance(v, list) else v) for k, v in d.items()}

        context_data_raw = take_subset(train_dataset_dict, effective_context_size)
        inference_data_raw = take_subset(test_dataset_dict, effective_inference_size)

    # Optional truncation by max_num_events (applied to the whole sequences BEFORE splitting into history/future)
    def _truncate_batch(data_dict: Dict) -> Dict:
        if max_num_events is None:
            return data_dict
        out = {"time_since_start": [], "time_since_last_event": [], "type_event": [], "seq_len": []}
        for times, deltas, types, length in zip(
            data_dict["time_since_start"],
            data_dict["time_since_last_event"],
            data_dict["type_event"],
            data_dict["seq_len"],
        ):
            trunc = min(length, max_num_events)
            out["time_since_start"].append(times[:trunc])
            out["time_since_last_event"].append(deltas[:trunc])
            out["type_event"].append(types[:trunc])
            out["seq_len"].append(trunc)
        for key in ("time_offsets",):
            if key in data_dict:
                out[key] = data_dict[key]
        return out

    context_data_raw = _truncate_batch(context_data_raw)
    inference_data_raw = _truncate_batch(inference_data_raw)

    detected_num_marks = detect_num_event_types_from_data(context_data_raw, inference_data_raw)

    # Sampler bound based on context deltas
    num_context_sequences = len(context_data_raw.get("seq_len", []))
    if num_context_sequences > 0:
        max_dtime_train = (
            max(max(seq) for seq in context_data_raw["time_since_last_event"]) if context_data_raw["time_since_last_event"] else 1.0
        )
    else:
        max_dtime_train = 1.0
    model.event_sampler.dtime_max = float(max_dtime_train) * 1.2
    model.event_sampler.over_sample_rate = 5.0

    # Build context batch tensors once
    max_context_len = max((len(seq) for seq in context_data_raw["time_since_start"]), default=0)
    context_batch = {"time_seqs": [], "type_seqs": [], "seq_len": []}
    for i in range(len(context_data_raw["time_since_start"])):
        pad = max_context_len - len(context_data_raw["time_since_start"][i])
        context_batch["time_seqs"].append(context_data_raw["time_since_start"][i] + [0] * pad)
        context_batch["type_seqs"].append(context_data_raw["type_event"][i] + [0] * pad)
        context_batch["seq_len"].append(context_data_raw["seq_len"][i])
    context_batch["time_seqs"] = torch.tensor(context_batch["time_seqs"], device=device)
    context_batch["type_seqs"] = torch.tensor(context_batch["type_seqs"], device=device)
    context_batch["seq_len"] = torch.tensor(context_batch["seq_len"], device=device)
    context_batch["seq_non_pad_mask"] = torch.arange(max_context_len, device=device).expand(
        len(context_batch["seq_len"]), max_context_len
    ) < context_batch["seq_len"].unsqueeze(1)

    # Precompute enhanced context
    with torch.no_grad():
        precomp_ctx = {
            "context_event_times": context_batch["time_seqs"].unsqueeze(0).unsqueeze(-1).to(device),
            "context_event_types": context_batch["type_seqs"].unsqueeze(0).unsqueeze(-1).to(device),
            "context_seq_lengths": context_batch["seq_non_pad_mask"].sum(dim=1).unsqueeze(0).to(device),
        }
        precomputed_enhanced_context = model.encode_context(precomp_ctx)

    # Accumulate metrics over eligible sequences
    num_eval = 0
    rmsex_plus_sum = 0.0
    smape_sum = 0.0
    rmse_e_sum = 0.0

    total_sequences = len(inference_data_raw.get("seq_len", []))
    for i in range(total_sequences):
        L = int(inference_data_raw["seq_len"][i])
        N = int(forecast_horizon_size)
        if L <= N:
            continue  # skip too-short sequences

        # Split into history and future
        hist_times_list = inference_data_raw["time_since_start"][i][: L - N]
        hist_types_list = inference_data_raw["type_event"][i][: L - N]
        future_dtimes_list = inference_data_raw["time_since_last_event"][i][L - N : L]
        future_types_list = inference_data_raw["type_event"][i][L - N : L]

        # Convert to tensors
        hist_times = torch.tensor(hist_times_list, dtype=torch.float32, device=device)
        hist_types = torch.tensor([int(t) for t in hist_types_list], dtype=torch.long, device=device)
        true_future_dtimes = torch.tensor(future_dtimes_list, dtype=torch.float32)
        true_future_types = torch.tensor([int(t) for t in future_types_list], dtype=torch.long)

        forecasts = generate_long_horizon_forecasts(
            model,
            context_batch,
            hist_times,
            hist_types,
            horizon_n=N,
            num_ensembles=num_ensemble_trajectories,
            device=device,
            precomputed_enhanced_context=precomputed_enhanced_context,
            num_marks=detected_num_marks,
        )

        pred_dtimes = forecasts["predicted_event_dtimes"]  # cpu 1D
        pred_types = forecasts["predicted_event_types"]

        # Metrics
        rmsex_plus_sum += calculate_rmse_x(pred_dtimes, true_future_dtimes)
        smape_sum += calculate_smape(pred_dtimes, true_future_dtimes)
        rmse_e_sum += calculate_rmse_e(pred_types, true_future_types, detected_num_marks)
        num_eval += 1

    duration_seconds = float(time.time() - start_time)
    if num_eval == 0:
        avg_rmsex_plus = 0.0
        avg_smape = 0.0
        avg_rmse_e = 0.0
    else:
        avg_rmsex_plus = rmsex_plus_sum / num_eval
        avg_smape = smape_sum / num_eval
        avg_rmse_e = rmse_e_sum / num_eval

    result = {
        "dataset": dataset,
        "model_checkpoint": model_checkpoint_path,
        "num_eval_sequences": int(num_eval),
        "duration_seconds": duration_seconds,
        "metrics": {
            "model": {
                "rmsex_plus": float(avg_rmsex_plus),
                "smape": float(avg_smape),
                "rmse_e": float(avg_rmse_e),
            }
        },
    }
    return result


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
        )
        status = "OK"
    except Exception as e:
        result = {
            "dataset": args.dataset,
            "model_checkpoint": str(args.checkpoint),
            "status": "FAIL",
            "error": str(e),
        }
        status = "FAIL"

    result.setdefault("status", status)
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
