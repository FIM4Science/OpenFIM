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

import numpy as np
import torch
from datasets import load_dataset

# Import the new OTD metric functions
from otd_metrics import get_distances_otd
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
    return float(100.0 * torch.mean(term).item())


def calculate_rmse_e(pred_types: torch.Tensor, true_types: torch.Tensor, num_marks: int) -> float:
    counts_pred = torch.bincount(pred_types, minlength=num_marks).float()
    counts_true = torch.bincount(true_types, minlength=num_marks).float()
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
) -> Dict:
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_fimhawkes_with_proper_weights(model_checkpoint_path)
    model.eval().to(device)
    model.event_sampler.num_samples_boundary = 50

    use_easytpp = isinstance(dataset, str) and str(dataset).startswith("easytpp/")
    if use_easytpp:
        train_dataset = load_dataset(dataset, split="train")
        test_dataset = load_dataset(dataset, split="test")
        eff_ctx_size = len(train_dataset) if context_size is None else context_size
        eff_inf_size = len(test_dataset) if inference_size is None else inference_size
        context_data_raw = train_dataset[:eff_ctx_size]
        inference_data_raw = test_dataset[:eff_inf_size]
    else:
        train_dataset_dict = load_local_dataset(dataset, "context")
        test_dataset_dict = load_local_dataset(dataset, "test")
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

    detected_num_marks = detect_num_event_types_from_data(context_data_raw, inference_data_raw)
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

    num_eval = 0
    rmsex_plus_sum, smape_sum, rmse_e_sum, otd_sum = 0.0, 0.0, 0.0, 0.0
    otd_del_costs = [0.05, 0.5, 1, 1.5, 2, 3, 4]
    otd_trans_cost = 1.0

    total_sequences = len(inference_data_raw.get("seq_len", []))
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

    duration_seconds = float(time.time() - start_time)
    if num_eval == 0:
        avg_rmsex_plus, avg_smape, avg_rmse_e, avg_otd = 0.0, 0.0, 0.0, 0.0
    else:
        avg_rmsex_plus = rmsex_plus_sum / num_eval
        avg_smape = smape_sum / num_eval
        avg_rmse_e = rmse_e_sum / num_eval
        avg_otd = otd_sum / num_eval

    return {
        "dataset": dataset,
        "model_checkpoint": model_checkpoint_path,
        "num_eval_sequences": int(num_eval),
        "duration_seconds": duration_seconds,
        "metrics": {
            "model": {
                "rmsex_plus": float(avg_rmsex_plus),
                "smape": float(avg_smape),
                "rmse_e": float(avg_rmse_e),
                "otd": float(avg_otd),
            }
        },
    }


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


if __name__ == "__main__":
    main()
