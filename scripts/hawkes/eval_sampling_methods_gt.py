"""
Evaluate thinning vs inverse_transform using ground-truth Hawkes intensities on the synthetic dataset.

Usage (CUDA):
  bash -i -c 'conda activate model_training && python3 scripts/hawkes/eval_sampling_methods_gt.py \
    --data_root /cephfs/users/berghaus/FoundationModels/FIM/data/synthetic_data/hawkes/EVAL_10D_2k_context_paths_100_inference_paths_const_base_exp_kernel/test \
    --output results/gt_sampling_eval \
    --max_paths 10 --max_events 100 --num_samples 100 --device cuda'

Usage (CPU quick smoke test):
  bash -i -c 'conda activate model_training && python3 scripts/hawkes/eval_sampling_methods_gt.py \
    --data_root /cephfs/users/berghaus/FoundationModels/FIM/data/synthetic_data/hawkes/EVAL_10D_2k_context_paths_100_inference_paths_const_base_exp_kernel/test \
    --output results/gt_sampling_eval/_smoke \
    --max_paths 1 --max_events 20 --num_samples 5 --device cpu'

Arguments:
  --data_root   Path to the synthetic test folder containing: event_times.pt, event_types.pt,
                seq_lengths.pt, time_offsets.pt, base_intensity_functions.pt, kernel_functions.pt
  --output      Directory to write gt_sampling_comparison.json
  --max_paths   Optional cap on number of paths per batch (default: no cap)
  --max_events  Optional cap on events per path (default: no cap)
  --num_samples Number of samples per prefix for sampling-based estimators (default: 100)
  --device      'cuda' or 'cpu' (default: auto-detect)

The output JSON contains metrics for three methods: 'thinning', 'inverse_transform', and 'baseline'
(baseline predicts mean inter-event time and majority event type).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch


def _ensure_repo_src_on_path() -> None:
    this_file = Path(__file__).resolve()
    repo_src = this_file.parents[2] / "src"
    if str(repo_src) not in os.sys.path:
        os.sys.path.insert(0, str(repo_src))


_ensure_repo_src_on_path()

from fim.models.hawkes.piecewise_intensity import PiecewiseHawkesIntensity  # noqa: E402
from fim.models.hawkes.thinning import EventSampler  # noqa: E402
from fim.utils.helper import decode_byte_string  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate thinning vs inverse_transform using GT Hawkes intensities")
    p.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Synthetic dataset test folder containing *.pt tensors",
    )
    p.add_argument("--output", type=Path, required=True, help="Output directory for results JSON")
    p.add_argument("--max_paths", type=int, default=None, help="Optional limit of paths per batch to evaluate")
    p.add_argument("--max_events", type=int, default=None, help="Optional limit of events per path to evaluate")
    p.add_argument("--num_samples", type=int, default=100, help="Samples per prefix for sampling-based estimators")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_dataset(root: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    payload: Dict[str, torch.Tensor] = {}
    for name in [
        "event_times.pt",
        "event_types.pt",
        "seq_lengths.pt",
        "time_offsets.pt",
        "base_intensity_functions.pt",
        "kernel_functions.pt",
    ]:
        payload[name[:-3]] = torch.load(root / name, map_location=device)
    return payload


def decode_functions(
    base_funcs_bytes: torch.Tensor,
    kernel_funcs_bytes: torch.Tensor,
) -> Tuple[List[List[Callable]], List[List[List[Callable]]]]:
    # base_funcs_bytes expected shape: [B, M, L_bytes]
    # kernel_funcs_bytes expected shape: [B, M, M, L_bytes]
    B, M = base_funcs_bytes.shape[:2]
    base_funcs: List[List[Callable]] = []
    kernel_funcs: List[List[List[Callable]]] = []
    for b in range(B):
        base_row: List[Callable] = []
        kernel_row: List[List[Callable]] = []
        for i in range(M):
            base_row.append(eval(decode_byte_string(base_funcs_bytes[b, i])))
        for i in range(M):
            kij: List[Callable] = []
            for j in range(M):
                kij.append(eval(decode_byte_string(kernel_funcs_bytes[b, i, j])))
            kernel_row.append(kij)
        base_funcs.append(base_row)
        kernel_funcs.append(kernel_row)
    return base_funcs, kernel_funcs


@torch.no_grad()
def extract_params_numeric(
    base_funcs: List[List[Callable]],
    kernel_funcs: List[List[List[Callable]]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = len(base_funcs)
    M = len(base_funcs[0]) if B > 0 else 0
    mu = torch.zeros(B, M, device=device, dtype=torch.float64)
    alpha = torch.zeros(B, M, M, device=device, dtype=torch.float64)
    beta = torch.zeros(B, M, M, device=device, dtype=torch.float64)

    t_probe = torch.tensor([[0.0, 1.0, 2.0]], device=device, dtype=torch.float64)  # [1, 3]
    for b in range(B):
        for i in range(M):
            try:
                mu_val = base_funcs[b][i](t_probe)
                if isinstance(mu_val, (float, int)):
                    mu[b, i] = float(mu_val)
                else:
                    mu[b, i] = float(mu_val.reshape(-1)[0].item())
            except Exception:
                mu[b, i] = 0.0
        for i in range(M):
            for j in range(M):
                try:
                    k = kernel_funcs[b][i][j]
                    v0 = k(torch.tensor([0.0], device=device, dtype=torch.float64))
                    v1 = k(torch.tensor([1.0], device=device, dtype=torch.float64))
                    a0 = float(v0.reshape(-1)[0].item())
                    a1 = float(v1.reshape(-1)[0].item())
                    alpha[b, i, j] = max(a0, 0.0)
                    if a0 > 1e-12 and a1 > 0:
                        beta[b, i, j] = -math.log(a1 / a0)
                    else:
                        beta[b, i, j] = 0.0
                except Exception:
                    alpha[b, i, j] = 0.0
                    beta[b, i, j] = 0.0
    return mu.float(), alpha.float(), beta.float()


def make_gt_intensity_fn(
    eval_times_abs: torch.Tensor,
    hist_times_abs: torch.Tensor,
    hist_types: torch.Tensor,
    seq_len: int,
    mu_funcs: List[Callable],
    kernel_funcs_ij: List[List[Callable]],
) -> torch.Tensor:
    B, L_dummy, T = eval_times_abs.shape
    M = len(mu_funcs)
    out = torch.zeros(B, L_dummy, T, M, device=eval_times_abs.device, dtype=eval_times_abs.dtype)
    hist = hist_times_abs[:, :seq_len]
    types = hist_types[:, :seq_len]

    # Base intensity per mark i; decoded functions accept shape [P, T]
    eval_for_funcs = eval_times_abs.squeeze(0)  # [L_dummy, T] for B=1
    for i in range(M):
        mu_i = mu_funcs[i](eval_for_funcs)
        if isinstance(mu_i, (float, int)):
            base_vals = torch.full((L_dummy, T), float(mu_i), device=eval_times_abs.device, dtype=eval_times_abs.dtype)
        else:
            base_vals = mu_i
            if base_vals.dim() == 1:
                base_vals = base_vals.view(L_dummy, T)
        out[..., i] += base_vals.unsqueeze(0)

    if seq_len == 0:
        return torch.relu(out)

    # Kernel contributions φ_ij(t - t_k); decoded functions accept [P, T, L_hist]
    eval_exp = eval_times_abs.unsqueeze(-1)  # [B, L_dummy, T, 1]
    hist_exp = hist.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L_hist]
    dt = (eval_exp - hist_exp).squeeze(0)  # [L_dummy, T, L_hist]
    dt_mask = dt > 1e-12
    types_b = types.squeeze(0)  # [L_hist]
    for j in range(M):
        mask_j = (types_b == j).view(1, 1, -1)  # [1,1,L_hist]
        if not mask_j.any():
            continue
        for i in range(M):
            phi = kernel_funcs_ij[i][j]
            contrib = phi(dt)  # [L_dummy, T, L_hist]
            contrib = torch.where(dt_mask & mask_j, contrib, torch.zeros_like(contrib))
            summed = contrib.sum(dim=-1)  # [L_dummy, T]
            out[..., i] += summed.unsqueeze(0)
    return torch.relu(out)


def predict_next_with_thinning(
    sampler: EventSampler,
    hist_times: torch.Tensor,
    hist_types: torch.Tensor,
    seq_len: int,
    mu_funcs: List[Callable],
    kernel_funcs_ij: List[List[Callable]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    time_seq = hist_times[:, :seq_len]
    dtime = torch.zeros_like(time_seq)
    dtime[:, 1:] = time_seq[:, 1:] - time_seq[:, :-1]

    def intensity_fn(query_times: torch.Tensor, time_seq_unused: torch.Tensor) -> torch.Tensor:
        return make_gt_intensity_fn(query_times, hist_times, hist_types, seq_len, mu_funcs, kernel_funcs_ij)

    accepted_t, weights = sampler.draw_next_time_one_step(
        time_seq=time_seq,
        time_delta_seq=dtime,
        event_seq=hist_types[:, :seq_len],
        intensity_fn=intensity_fn,
        compute_last_step_only=True,
    )
    expected_t = (accepted_t * weights).sum(dim=-1).squeeze(-1)
    intens_at_expected = make_gt_intensity_fn(expected_t.view(-1, 1, 1), hist_times, hist_types, seq_len, mu_funcs, kernel_funcs_ij)
    type_pred = intens_at_expected.squeeze(0).squeeze(0).argmax(dim=-1)
    return expected_t.view(-1), type_pred


def predict_next_with_inverse_transform(
    sampler: EventSampler,
    hist_times: torch.Tensor,
    hist_types: torch.Tensor,
    seq_len: int,
    mu_vals: torch.Tensor,
    alpha_ij: torch.Tensor,
    beta_ij: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = hist_times.device
    dtype = hist_times.dtype
    B = hist_times.shape[0]
    M = int(mu_vals.shape[-1])
    t_last = hist_times[:, seq_len - 1] if seq_len > 0 else torch.zeros(B, device=device, dtype=dtype)

    # Occurrence counts for each source mark j in the history
    if seq_len > 0:
        counts = torch.bincount(hist_types[:, :seq_len].reshape(-1), minlength=M).to(device=device)
    else:
        counts = torch.zeros(M, device=device)

    # alpha_eff_i = mu_i + sum_j counts_j * alpha_ij[i,j]
    # beta_eff_i  = weighted average of beta_ij[i,j] with weights counts_j (fallback 1.0)
    if alpha_ij.dim() == 3:
        # [B, M, M] -> take first batch or broadcast
        alpha_mat = alpha_ij[0]
        beta_mat = beta_ij[0]
    else:
        alpha_mat = alpha_ij  # [M, M]
        beta_mat = beta_ij

    S = (alpha_mat * counts.view(1, -1)).sum(dim=1)  # [M]
    alpha_eff = mu_vals + S  # [M]

    denom = counts.sum()
    if denom > 1e-12:
        beta_eff = ((beta_mat * counts.view(1, -1)).sum(dim=1)) / denom
    else:
        beta_eff = torch.full_like(mu_vals, 1.0)

    event_times = t_last.view(B, 1, 1)
    mu_t = mu_vals.view(B, M, 1, 1)
    alpha_t = alpha_eff.view(B, M, 1, 1)
    beta_t = beta_eff.view(B, M, 1, 1)
    intensity_obj = PiecewiseHawkesIntensity(event_times=event_times, mu=mu_t, alpha=alpha_t, beta=beta_t)
    accepted_t, weights = sampler.draw_next_time_one_step_inverse_transform(intensity_obj, compute_last_step_only=True)
    expected_t = (accepted_t * weights).sum(dim=-1).squeeze(-1)
    return expected_t.view(-1), torch.zeros(B, dtype=torch.long, device=device)


def evaluate(
    data_root: Path,
    output_dir: Path,
    max_paths: int | None,
    max_events: int | None,
    num_samples: int,
    device: torch.device,
) -> Path:
    tensors = load_dataset(data_root, device)
    et = tensors["event_times"].squeeze(-1)
    ty = tensors["event_types"].squeeze(-1).long()
    sl = tensors["seq_lengths"].long()
    toff = tensors["time_offsets"]
    bf = tensors["base_intensity_functions"]
    kf = tensors["kernel_functions"]

    if max_paths is not None and et.shape[1] > max_paths:
        et = et[:, :max_paths]
        ty = ty[:, :max_paths]
        sl = sl[:, :max_paths]
        toff = toff[:, :max_paths]

    base_funcs, kernel_funcs = decode_functions(bf, kf)
    mu_vals, alpha_ij, beta_ij = extract_params_numeric(base_funcs, kernel_funcs, device)

    sampler_thin = EventSampler(num_sample=num_samples, num_exp=500, device=device)
    sampler_inv = EventSampler(num_sample=num_samples, device=device)
    sampler_inv.sampling_method = "inverse_transform"

    B, P, L = et.shape
    if max_events is not None and L > max_events:
        L = max_events
        et = et[:, :, :L]
        ty = ty[:, :, :L]
        sl = torch.clamp(sl, max=L)

    abs_times = et
    mae_thin = 0.0
    acc_thin = 0.0
    n_pred = 0
    mae_inv = 0.0
    acc_inv = 0.0

    # --- Baseline: global mean delta time and majority event type ---
    # Compute global delta times with masking of padding
    deltas = abs_times.clone()
    deltas[:, :, 1:] = abs_times[:, :, 1:] - abs_times[:, :, :-1]
    deltas[:, :, 0] = 0.0
    B_, P_, L_ = abs_times.shape
    positions = torch.arange(L_, device=device).view(1, 1, L_)
    # valid delta positions are 1..(seq_len-1)
    valid_delta_mask = (positions < sl.unsqueeze(-1)) & (positions >= 1)
    valid_deltas = deltas[valid_delta_mask]
    dtime_mean = float(valid_deltas.mean().item()) if valid_deltas.numel() > 0 else 0.0

    # Majority event type across valid events at positions >=1
    types_mask = (positions < sl.unsqueeze(-1)) & (positions >= 1)
    valid_types = ty[types_mask]
    if valid_types.numel() > 0:
        num_marks_guess = int(valid_types.max().item()) + 1
        binc = torch.bincount(valid_types.view(-1), minlength=num_marks_guess)
        majority_type = int(torch.argmax(binc).item())
    else:
        majority_type = 0
    mae_base = 0.0
    acc_base = 0.0

    for b in range(B):
        for p in range(P):
            seq_len = int(sl[b, p].item())
            if seq_len < 2:
                continue
            for prefix in range(1, seq_len):
                hist_times = abs_times[b : b + 1, p : p + 1, :prefix].reshape(1, prefix).to(device)
                hist_types = ty[b : b + 1, p : p + 1, :prefix].reshape(1, prefix).to(device)
                t_true = abs_times[b, p, prefix]
                type_true = ty[b, p, prefix]

                t_pred_thin, type_pred_thin = predict_next_with_thinning(
                    sampler_thin,
                    hist_times,
                    hist_types,
                    prefix,
                    base_funcs[b],
                    kernel_funcs[b],
                )
                err_t_thin = torch.abs(t_pred_thin - t_true).item()
                mae_thin += err_t_thin
                acc_thin += 1.0 if int(type_pred_thin.item()) == int(type_true.item()) else 0.0

                mu_row = mu_vals[b]
                alpha_row = alpha_ij[b]
                beta_row = beta_ij[b]
                t_pred_inv, _ = predict_next_with_inverse_transform(
                    sampler_inv,
                    hist_times,
                    hist_types,
                    prefix,
                    mu_row,
                    alpha_row,
                    beta_row,
                )
                err_t_inv = torch.abs(t_pred_inv - t_true).item()
                mae_inv += err_t_inv

                intens_at_inv = make_gt_intensity_fn(
                    t_pred_inv.view(1, 1, 1),
                    hist_times,
                    hist_types,
                    prefix,
                    base_funcs[b],
                    kernel_funcs[b],
                )
                type_pred_inv = intens_at_inv.squeeze().argmax().item()
                acc_inv += 1.0 if int(type_pred_inv) == int(type_true.item()) else 0.0

                n_pred += 1

                # Baseline prediction: last_time + mean_delta, majority type
                t_pred_base = hist_times[:, -1] + dtime_mean
                err_t_base = torch.abs(t_pred_base - t_true).item()
                mae_base += err_t_base
                acc_base += 1.0 if majority_type == int(type_true.item()) else 0.0

    mae_thin /= max(n_pred, 1)
    mae_inv /= max(n_pred, 1)
    acc_thin /= max(n_pred, 1)
    acc_inv /= max(n_pred, 1)
    mae_base /= max(n_pred, 1)
    acc_base /= max(n_pred, 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "gt_sampling_comparison.json"
    out_payload = {
        "dataset_root": str(data_root),
        "num_predictions": n_pred,
        "num_samples": num_samples,
        "metrics": {
            "thinning": {"mae_time": mae_thin, "type_accuracy": acc_thin},
            "inverse_transform": {"mae_time": mae_inv, "type_accuracy": acc_inv},
            "baseline": {"mae_time": mae_base, "type_accuracy": acc_base, "mean_dtime": dtime_mean, "majority_type": majority_type},
        },
    }
    out_path.write_text(json.dumps(out_payload, indent=2))
    return out_path


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_path = evaluate(
        data_root=args.data_root,
        output_dir=args.output,
        max_paths=args.max_paths,
        max_events=args.max_events,
        num_samples=args.num_samples,
        device=device,
    )
    print(f"Wrote results to: {out_path}")


if __name__ == "__main__":
    main()
