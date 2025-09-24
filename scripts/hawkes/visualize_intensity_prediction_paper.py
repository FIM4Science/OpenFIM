"""
CUDA_VISIBLE_DEVICES="" python scripts/hawkes/visualize_intensity_prediction_paper.py \
--checkpoint "results/FIM_Hawkes_10-22st_nll_mc_only_2000_paths_mixed_100_events_mixed-experiment-seed-10-dataset-dataset_kwargs-field_name_for_dimension_grouping-base_intensity_functions_09-22-1331/checkpoints/best-model"  \
--dataset "data/synthetic_data/hawkes/1k_3D_2k_paths_const_base_exp_kernel_no_interactions/test" \
--sample_idx 0 \
--path_idx 0


conda run -n model_training python scripts/hawkes/visualize_intensity_prediction_paper.py \
  --checkpoint "results/FIM_Hawkes_10-22st_nll_mc_only_2000_paths_mixed_100_events_mixed-experiment-seed-10-dataset-dataset_kwargs-field_name_for_dimension_grouping-base_intensity_functions_09-23-1809/checkpoints/best-model" \
  --dataset "data/synthetic_data/hawkes/1k_3D_2k_paths_const_base_exp_kernel_no_interactions/test" \
  --right_dataset "retweet" \
  --left_title "Synthetic Hawkes Process" \
  --right_title "Retweet Dataset" \
  --path_idx 0 \
  --sample_idx 0 \
  --save_path_comparison intensity_comparison_synth_vs_retweet.png
"""

#!/usr/bin/env python
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from fim.models.hawkes import FIMHawkes, FIMHawkesConfig


# Global plotting style: Computer Modern fonts and larger text to match LaTeX style
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 22,
    }
)


# Register FIMHawkes with transformers AutoConfig/AutoModel system
# This fixes the from_pretrained method by ensuring proper config loading
FIMHawkesConfig.register_for_auto_class()
FIMHawkes.register_for_auto_class("AutoModel")


def _move_to_device(obj, device):
    """Recursively move tensors in nested containers to the specified device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_move_to_device(v, device) for v in obj]
        return type(obj)(seq) if isinstance(obj, tuple) else seq
    return obj


def load_data_from_dir(dir_path: Path) -> dict:
    """
    Load all .pt files in a directory and return as a dict of tensors.
    """
    tensors = {}
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dir_path}")
    for file in dir_path.iterdir():
        if file.suffix == ".pt":
            try:
                # Use map_location to ensure tensors are loaded to CPU
                tensors[file.stem] = torch.load(file, map_location="cpu")
            except Exception as e:
                print(f"Could not load {file}: {e}")
    return tensors


def _is_cdiff_dataset_dir(dataset_dir: Path) -> bool:
    """Heuristic: directory with train.pkl/val.pkl/test.pkl like CDiff datasets."""
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return False
    expected_any = [dataset_dir / s for s in ("train.pkl", "val.pkl", "test.pkl")]
    return any(p.exists() for p in expected_any)


def _resolve_cdiff_dir(dataset: Path) -> Path:
    """Allow passing short names like 'retweet' by resolving into repo CDiff folder."""
    if _is_cdiff_dataset_dir(dataset):
        return dataset
    candidate = Path(__file__).resolve().parents[2] / "data" / "external" / "CDiff_dataset" / str(dataset)
    return candidate if _is_cdiff_dataset_dir(candidate) else dataset


def _load_cdiff_split(dataset_dir: Path, split: str) -> Dict[str, List]:
    """Load CDiff `<split>.pkl` and adapt to internal list-based format.

    Returns dict with keys: time_since_start, time_since_last_event, type_event, seq_len.
    All times are shifted so each sequence starts at 0.
    """
    pkl_path = dataset_dir / f"{split}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"CDiff dataset split not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    seqs_raw = d.get(split)
    if not isinstance(seqs_raw, list):
        raise ValueError(f"Unexpected '{split}' content in {pkl_path}: {type(seqs_raw)}")

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
        first_time = float(seq[0].get("time_since_start", 0.0))
        times = [float(ev.get("time_since_start", 0.0)) - first_time for ev in seq]
        types = [int(ev.get("type_event", 0)) for ev in seq]
        deltas = [0.0] + [float(times[j] - times[j - 1]) for j in range(1, len(times))]
        time_since_start.append(times)
        time_since_last_event.append(deltas)
        type_event.append(types)
        seq_len.append(len(times))

    return {
        "time_since_start": time_since_start,
        "time_since_last_event": time_since_last_event,
        "type_event": type_event,
        "seq_len": seq_len,
    }


def _choose_validation_split(dataset_dir: Path) -> str:
    """Prefer 'val' when available, otherwise fall back to 'dev'."""
    if (dataset_dir / "val.pkl").exists():
        return "val"
    if (dataset_dir / "dev.pkl").exists():
        return "dev"
    raise FileNotFoundError(f"Neither val.pkl nor dev.pkl found in {dataset_dir}")


def _detect_num_marks_from_lists(context_data: Dict[str, List], inference_data: Dict[str, List]) -> int:
    unique_types = set()
    for seq in context_data.get("type_event", []):
        unique_types.update(seq)
    for seq in inference_data.get("type_event", []):
        unique_types.update(seq)
    return int(max(unique_types) + 1) if unique_types else 1


def _pad_to_tensor(seqs: List[List[int | float]], dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad list-of-lists to tensor [P, Lmax] and return lengths [P]."""
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    Lmax = int(lengths.max().item()) if lengths.numel() > 0 else 0
    P = len(seqs)
    if Lmax == 0 or P == 0:
        return torch.zeros(P, 0, dtype=dtype), lengths
    out = torch.zeros(P, Lmax, dtype=dtype)
    for i, s in enumerate(seqs):
        if len(s) > 0:
            out[i, : len(s)] = torch.tensor(s, dtype=dtype)
    return out, lengths


def build_model_data_from_cdiff(dataset_dir: Path, val_index: int, max_context_paths: int = 2000) -> Dict[str, torch.Tensor]:
    """Construct model input using up to `max_context_paths` from train as context and
    a single sequence from val as the inference path.
    """
    dataset_dir = _resolve_cdiff_dir(dataset_dir)
    if not _is_cdiff_dataset_dir(dataset_dir):
        raise FileNotFoundError(f"Not a CDiff dataset directory: {dataset_dir}")

    train = _load_cdiff_split(dataset_dir, "train")
    val_split = _choose_validation_split(dataset_dir)
    val = _load_cdiff_split(dataset_dir, val_split)

    num_train = len(train["seq_len"]) if "seq_len" in train else 0
    num_val = len(val["seq_len"]) if "seq_len" in val else 0
    if num_val == 0:
        raise ValueError(f"Validation split is empty in {dataset_dir}")

    P_ctx = min(int(max_context_paths), int(num_train)) if num_train > 0 else 0
    val_index = int(val_index) % num_val

    # Prepare context tensors
    ctx_times, ctx_lengths = _pad_to_tensor(train["time_since_start"][:P_ctx], dtype=torch.float32)
    ctx_types, _ = _pad_to_tensor(train["type_event"][:P_ctx], dtype=torch.long)

    # Prepare inference tensors (single path from val)
    inf_times_list = val["time_since_start"][val_index]
    inf_types_list = val["type_event"][val_index]
    inf_times = torch.tensor(inf_times_list, dtype=torch.float32).view(1, -1)
    inf_types = torch.tensor([int(t) for t in inf_types_list], dtype=torch.long).view(1, -1)
    inf_lengths = torch.tensor([len(inf_times_list)], dtype=torch.long).view(1)

    # Align lengths for batching: model expects dims [B, P, L, 1]
    # We keep B=1. Context: [P_ctx, Lc] -> [1, P_ctx, Lc, 1]; Inference: [1, Li] -> [1, 1, Li, 1]
    model_data: Dict[str, torch.Tensor] = {}
    if P_ctx > 0:
        model_data["context_event_times"] = ctx_times.unsqueeze(0).unsqueeze(-1)
        model_data["context_event_types"] = ctx_types.unsqueeze(0).unsqueeze(-1)
        model_data["context_seq_lengths"] = ctx_lengths.view(1, -1)
    else:
        # Ensure at least empty context dims
        model_data["context_event_times"] = torch.zeros(1, 0, 0, 1, dtype=torch.float32)
        model_data["context_event_types"] = torch.zeros(1, 0, 0, 1, dtype=torch.long)
        model_data["context_seq_lengths"] = torch.zeros(1, 0, dtype=torch.long)

    model_data["inference_event_times"] = inf_times.unsqueeze(0).unsqueeze(-1)
    model_data["inference_event_types"] = inf_types.unsqueeze(0).unsqueeze(-1)
    model_data["inference_seq_lengths"] = inf_lengths.view(1, 1)

    # Evaluation times based on inference events
    model_data["intensity_evaluation_times"] = create_evaluation_times(
        model_data["inference_event_times"], model_data["inference_seq_lengths"], num_points_between_events=10
    )

    # Marks (types) count
    num_marks = _detect_num_marks_from_lists(train, val)
    model_data["num_marks"] = num_marks

    return model_data


def create_evaluation_times(inference_event_times, inference_seq_lengths, num_points_between_events=10):
    """
    Create evaluation times using actual event times and uniformly spaced points between consecutive events.

    This function combines:
    1. Actual event times for precise evaluation at event occurrences
    2. Uniformly spaced points between each consecutive pair of events for smooth plotting

    Args:
        inference_event_times: Event times tensor [B, P_inference, L, 1]
        inference_seq_lengths: Sequence lengths [B, P_inference]
        num_points_between_events: Number of uniformly spaced points between consecutive events

    Returns:
        evaluation_times_batch: Combined evaluation times [B, P_inference, max_combined_points]
    """
    B, P_inference, L, _ = inference_event_times.shape
    device = inference_event_times.device

    # Estimate maximum number of combined points (events + points between events)
    max_seq_len = inference_seq_lengths.max().item() if inference_seq_lengths.numel() > 0 else 0
    # Each interval between events gets num_points_between_events points, plus all event times
    max_combined_points = max_seq_len + (max_seq_len - 1) * num_points_between_events + num_points_between_events

    evaluation_times_batch = torch.zeros(B, P_inference, max_combined_points, device=device)

    for b in range(B):
        for p in range(P_inference):
            seq_len = inference_seq_lengths[b, p].item()
            if seq_len == 0:
                continue

            # Get actual event times for this path
            actual_event_times = inference_event_times[b, p, :seq_len, 0]

            if seq_len == 1:
                # If only one event, just use that event time and some points before it
                max_time = actual_event_times[0].item()
                if max_time > 0:
                    # Add some points before the first event
                    before_first = torch.linspace(0.0, max_time * 0.95, num_points_between_events, device=device)
                    combined_times = torch.cat([before_first, actual_event_times])
                else:
                    combined_times = actual_event_times
            else:
                # Multiple events: create points between consecutive events
                all_times = [torch.tensor([0.0], device=device)]  # Start from time 0

                for i in range(seq_len):
                    if i == 0:
                        # Points between 0 and first event
                        if actual_event_times[0] > 0:
                            between_points = torch.linspace(
                                0.0, actual_event_times[0].item(), num_points_between_events + 1, device=device
                            )[1:-1]  # Exclude endpoints
                            if len(between_points) > 0:
                                all_times.append(between_points)
                    else:
                        # Points between consecutive events
                        start_time = actual_event_times[i - 1].item()
                        end_time = actual_event_times[i].item()
                        if end_time > start_time:
                            between_points = torch.linspace(start_time, end_time, num_points_between_events + 1, device=device)[
                                1:-1
                            ]  # Exclude endpoints
                            if len(between_points) > 0:
                                all_times.append(between_points)

                    # Add the actual event time
                    all_times.append(actual_event_times[i : i + 1])

                # Add some points after the last event
                last_time = actual_event_times[-1].item()
                if last_time > 0:
                    after_last = torch.linspace(last_time, last_time * 1.05, num_points_between_events + 1, device=device)[1:]
                    all_times.append(after_last)

                # Combine all times
                combined_times = torch.cat(all_times)

            # Remove duplicates and sort
            combined_times_unique = torch.unique(combined_times, sorted=True)

            # Store in the batch tensor (pad with zeros if necessary)
            num_unique = len(combined_times_unique)
            if num_unique <= max_combined_points:
                evaluation_times_batch[b, p, :num_unique] = combined_times_unique
            else:
                # Truncate if too many points (shouldn't happen with our allocation)
                evaluation_times_batch[b, p, :] = combined_times_unique[:max_combined_points]

    return evaluation_times_batch


def prepare_batch_for_model(data_sample, inference_path_idx=0, num_points_between_events=10):
    """
    Prepare a single data sample for the model by splitting into context and inference paths.
    Uses all paths except the specified inference path for context.
    """
    # FIX: Ensure all tensors have a batch dimension of 1 before processing.
    # This resolves the IndexError for 1D tensors like seq_lengths.
    for key, value in data_sample.items():
        if torch.is_tensor(value):
            data_sample[key] = value.unsqueeze(0)

    # Get total number of paths
    total_paths = data_sample["event_times"].shape[1]

    if total_paths < 2:
        raise ValueError(f"Need at least 2 paths in sample ({total_paths}) to have both context and inference paths.")

    # Validate inference path index
    if inference_path_idx >= total_paths:
        print(f"Warning: inference_path_idx {inference_path_idx} >= total paths {total_paths}. Using path 0.")
        inference_path_idx = 0

    print(f"Total paths: {total_paths}, using path {inference_path_idx} for inference, remaining {total_paths - 1} for context")

    # Create masks for context and inference paths
    all_path_indices = torch.arange(total_paths)
    context_mask = all_path_indices != inference_path_idx
    inference_mask = all_path_indices == inference_path_idx

    context_indices = all_path_indices[context_mask]
    inference_indices = all_path_indices[inference_mask]

    # Split into context and inference paths
    event_times = data_sample["event_times"]
    event_types = data_sample["event_types"]
    seq_lengths = data_sample.get("seq_lengths")
    if seq_lengths is None:
        B, P, L, _ = event_times.shape
        seq_lengths = torch.full((B, P), L, device=event_times.device)

    model_data = {
        "context_event_times": event_times[:, context_indices],
        "context_event_types": event_types[:, context_indices],
        "context_seq_lengths": seq_lengths[:, context_indices],
        "inference_event_times": event_times[:, inference_indices],
        "inference_event_types": event_types[:, inference_indices],
        "inference_seq_lengths": seq_lengths[:, inference_indices],
    }

    model_data["intensity_evaluation_times"] = create_evaluation_times(
        model_data["inference_event_times"], model_data["inference_seq_lengths"], num_points_between_events=num_points_between_events
    )

    # Print info about evaluation times
    eval_times = model_data["intensity_evaluation_times"]
    for p in range(eval_times.shape[1]):
        valid_times = eval_times[0, p][eval_times[0, p] > 0]
        print(f"Path {p}: {len(valid_times)} evaluation points (events + points between events)")
        if len(valid_times) > 0:
            print(f"  Time range: {valid_times[0].item():.3f} to {valid_times[-1].item():.3f}")
            # Count how many are likely event times (close to actual events)
            if p < model_data["inference_event_times"].shape[1]:
                seq_len = model_data["inference_seq_lengths"][0, p].item()
                actual_events = model_data["inference_event_times"][0, p, :seq_len, 0]
                event_count = len(actual_events)
                print(f"  Includes {event_count} actual event times plus {len(valid_times) - event_count} points between events")

    for key in ["kernel_functions", "base_intensity_functions"]:
        if key in data_sample:
            model_data[key] = data_sample[key]

    if "base_intensity_functions" in model_data:
        model_data["num_marks"] = model_data["base_intensity_functions"].shape[1]
    else:
        model_data["num_marks"] = 1

    # If dataset provides per-path time offsets (absolute start times), pass the
    # offsets for the selected inference path so ground truth μ(t) is evaluated
    # at absolute time t'+offset while we plot over the shifted axis t'.
    if "time_offsets" in data_sample:
        offsets = data_sample["time_offsets"]  # shape [B, P] or [B, P, 1]
        # Ensure shape [B, P]
        if offsets.ndim == 3 and offsets.shape[-1] == 1:
            offsets = offsets.squeeze(-1)
        # Select offsets for the inference path(s); resulting shape [B, P_inference]
        model_data["inference_time_offsets"] = offsets[:, inference_indices]

    return model_data


def plot_intensity_comparison(model_output, model_data, save_path="intensity_comparison.png", path_idx=0):
    """
    Create vertically stacked plots comparing predicted and ground truth intensities.
    Uses scatter marks for events and smooth lines for intensity functions.
    """
    predicted_intensities = model_output["predicted_intensity_values"].detach().cpu().numpy()

    if "target_intensity_values" in model_output:
        target_intensities = model_output["target_intensity_values"].detach().cpu().numpy()
    else:
        target_intensities = None

    evaluation_times = model_data["intensity_evaluation_times"].detach().cpu().numpy()
    inference_event_times = model_data["inference_event_times"].detach().cpu().numpy()
    inference_event_types = model_data["inference_event_types"].detach().cpu().numpy()
    inference_seq_lengths = model_data["inference_seq_lengths"].detach().cpu().numpy()
    # Optional: per-path absolute-time offset (if events were shifted to 0)
    offsets_np = None
    if "inference_time_offsets" in model_data:
        off = model_data["inference_time_offsets"].detach().cpu().numpy()
        # Accept shapes [B,P] or [B,P,1]
        if off.ndim == 3 and off.shape[-1] == 1:
            off = off[..., 0]
        offsets_np = off

    B, M, P_inference, _ = predicted_intensities.shape

    # Marker cycle to give each mark type a distinct tick/shape
    marker_cycle = ["o", "s", "^", "D", "v", "P", "*", "X", "<", ">", "h", "8", "+", "x", "|", "_"]

    b = 0  # Always use first batch
    p = path_idx  # Use specified path index

    # Validate path index
    if p >= P_inference:
        print(f"Warning: path_idx {p} >= number of inference paths {P_inference}. Using path 0.")
        p = 0

    fig, axes = plt.subplots(M, 1, figsize=(16, 4.5 * M), sharex=True)
    if M == 1:
        axes = [axes]  # Ensure axes is always a list/array

    # Get all event times for the inference path to be plotted
    seq_len = inference_seq_lengths[b, p]
    all_event_times = inference_event_times[b, p, :seq_len, 0]
    all_event_types = inference_event_types[b, p, :seq_len, 0]

    for m in range(M):
        ax = axes[m]

        eval_times_p = evaluation_times[b, p]
        pred_intensity_p_m = predicted_intensities[b, m, p]

        # Filter out padded zeros and ensure proper alignment
        valid_mask = eval_times_p > 0
        sort_indices = None

        if valid_mask.any():
            eval_times_p = eval_times_p[valid_mask]
            pred_intensity_p_m = pred_intensity_p_m[valid_mask]

            # Ensure times are sorted and sort intensities accordingly
            sort_indices = np.argsort(eval_times_p)
            eval_times_p = eval_times_p[sort_indices]
            pred_intensity_p_m = pred_intensity_p_m[sort_indices]
        else:
            # If no valid times, create minimal plot
            eval_times_p = np.array([0.0])
            pred_intensity_p_m = np.array([0.0])

        # If offsets are available, shift the x-axis to absolute time for plotting
        if offsets_np is not None:
            offset_bp = offsets_np[b, p]
            eval_times_plot = eval_times_p + offset_bp
        else:
            eval_times_plot = eval_times_p

        # Plot smooth lines for intensity functions
        ax.plot(
            eval_times_plot,
            pred_intensity_p_m,
            color="#0072B2",
            linestyle="-",
            linewidth=2,
            label="FIM-PP (zero-shot)",
            alpha=1,
        )

        if target_intensities is not None:
            target_intensity_p_m = target_intensities[b, m, p]

            # Apply the same filtering and sorting to target intensities
            if valid_mask.any() and sort_indices is not None:
                target_intensity_p_m = target_intensity_p_m[valid_mask]
                target_intensity_p_m = target_intensity_p_m[sort_indices]
            else:
                target_intensity_p_m = np.array([0.0])

            ax.plot(
                eval_times_plot,
                target_intensity_p_m,
                color="black",
                linestyle="--",
                linewidth=1.8,
                label="Ground-Truth",
                alpha=1,
            )

        # Plot event scatter marks prominently
        # Events of the current mark, with a distinct marker for this mark type
        events_this_mark = all_event_times[all_event_types == m]
        if len(events_this_mark) > 0:
            event_intensities = []
            for event_time in events_this_mark:
                closest_idx = np.argmin(np.abs(eval_times_p - event_time))
                event_intensities.append(pred_intensity_p_m[closest_idx])

            if offsets_np is not None:
                events_plot = events_this_mark + offsets_np[b, p]
            else:
                events_plot = events_this_mark

            ax.scatter(
                events_plot,
                event_intensities,
                s=100,
                c="#CC79A7",
                marker=marker_cycle[m % len(marker_cycle)],
                label=None,
                zorder=10,
                edgecolors="#CC79A7",
                linewidth=1.5,
                alpha=1,
            )

        # Events of other marks, each with its own (grey) marker determined by mark type
        for k in range(M):
            if k == m:
                continue
            events_k = all_event_times[all_event_types == k]
            if len(events_k) == 0:
                continue

            other_event_intensities = []
            for event_time in events_k:
                closest_idx = np.argmin(np.abs(eval_times_p - event_time))
                other_event_intensities.append(pred_intensity_p_m[closest_idx])

            if offsets_np is not None:
                events_other_plot = events_k + offsets_np[b, p]
            else:
                events_other_plot = events_k

            ax.scatter(
                events_other_plot,
                other_event_intensities,
                s=60,
                c="gray",
                marker=marker_cycle[k % len(marker_cycle)],
                label=None,
                zorder=8,
                edgecolors="dimgray",
                linewidth=1.0,
                alpha=0.7,
            )

        ax.set_ylabel("Intensity", fontsize=14)
        ax.set_title(f"Intensity Function for Mark {m}", fontsize=14, fontweight="bold")

        # Axis aesthetics: slimmer spines/ticks, no grid, auto y-limits
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)
        ax.tick_params(axis="both", direction="out", width=0.5, length=2, pad=0.8)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
        ax.grid(False)
        ax.margins(x=0)

    axes[-1].set_xlabel("Time", fontsize=20)
    plt.suptitle(f"Hawkes Process Intensity Functions - Path {path_idx}", fontsize=22, fontweight="bold")
    # Unified legend at figure level (lines only)
    plt.draw()
    handles, labels = axes[0].get_legend_handles_labels()
    keep_labels = {"FIM-PP (zero-shot)", "Ground Truth"}
    uniq = []
    seen = set()
    for h, l in zip(handles, labels):
        if l in keep_labels and l not in seen:
            uniq.append((h, l))
            seen.add(l)

    # Add custom legend entries for mark symbols (three black markers)
    mark_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_cycle[i % len(marker_cycle)],
            linestyle="",
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=10,
        )
        for i in range(3)
    ]
    mark_labels = [f"Mark {i + 1}" for i in range(3)]

    combined_handles = [h for h, _ in uniq] + mark_handles
    combined_labels = [l for _, l in uniq] + mark_labels

    fig.legend(
        combined_handles,
        combined_labels,
        loc="upper center",
        bbox_to_anchor=[0.5, 0.995],
        ncols=len(combined_labels),
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.945])

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Also save as PDF for LaTeX-quality vector graphics
    try:
        pdf_path = str(Path(save_path).with_suffix(".pdf"))
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"Intensity comparison plot saved to {pdf_path}")
    except Exception as e:
        print(f"Warning: failed to save PDF version: {e}")
    plt.close()
    print(f"Intensity comparison plot saved to {save_path}")


def plot_two_datasets_side_by_side(
    left_output,
    left_data,
    right_output,
    right_data,
    left_title: str,
    right_title: str,
    save_path: str = "intensity_comparison_synthetic_vs_retweet.png",
    left_path_idx: int = 0,
    right_path_idx: int = 0,
):
    """
    Create a single figure with two columns: left and right datasets, each stacked by mark.
    """
    # Extract arrays (left)
    left_pred = left_output["predicted_intensity_values"].detach().cpu().numpy()
    left_eval_times = left_data["intensity_evaluation_times"].detach().cpu().numpy()
    left_inf_times = left_data["inference_event_times"].detach().cpu().numpy()
    left_inf_types = left_data["inference_event_types"].detach().cpu().numpy()
    left_inf_lengths = left_data["inference_seq_lengths"].detach().cpu().numpy()

    left_target = None
    if "target_intensity_values" in left_output:
        left_target = left_output["target_intensity_values"].detach().cpu().numpy()

    left_offsets_np = None
    if "inference_time_offsets" in left_data:
        off = left_data["inference_time_offsets"].detach().cpu().numpy()
        if off.ndim == 3 and off.shape[-1] == 1:
            off = off[..., 0]
        left_offsets_np = off

    # Extract arrays (right)
    right_pred = right_output["predicted_intensity_values"].detach().cpu().numpy()
    right_eval_times = right_data["intensity_evaluation_times"].detach().cpu().numpy()
    right_inf_times = right_data["inference_event_times"].detach().cpu().numpy()
    right_inf_types = right_data["inference_event_types"].detach().cpu().numpy()
    right_inf_lengths = right_data["inference_seq_lengths"].detach().cpu().numpy()

    right_target = None
    if "target_intensity_values" in right_output:
        right_target = right_output["target_intensity_values"].detach().cpu().numpy()

    right_offsets_np = None
    if "inference_time_offsets" in right_data:
        off = right_data["inference_time_offsets"].detach().cpu().numpy()
        if off.ndim == 3 and off.shape[-1] == 1:
            off = off[..., 0]
        right_offsets_np = off

    # Shapes
    _, M_left, P_left, _ = left_pred.shape
    _, M_right, P_right, _ = right_pred.shape
    M_global = max(M_left, M_right)

    # Marker cycle for distinct tick/shape per mark
    marker_cycle = ["o", "s", "^", "D", "v", "P", "*", "X", "<", ">", "h", "8", "+", "x", "|", "_"]

    # Create grid: rows = M_global (marks), cols = 3 with narrow spacer in the middle
    fig, axes = plt.subplots(
        M_global,
        3,
        figsize=(20, 3.0 * M_global),
        sharex=False,
        gridspec_kw={"width_ratios": [1.0, 0.002, 1.0]},
    )
    if M_global == 1:
        # Ensure 2D indexing
        axes = np.array([axes])

    # Hide the middle spacer column
    for r in range(M_global):
        axes[r, 1].axis("off")

    # Helper to plot one dataset into a column
    def _plot_dataset(col_idx: int, pred, eval_times, inf_times, inf_types, inf_lengths, target, offsets_np, path_idx, title):
        b = 0
        p = path_idx
        _, M, P, _ = pred.shape
        if p >= P:
            p = 0

        seq_len = inf_lengths[b, p]
        all_event_times = inf_times[b, p, :seq_len, 0]
        all_event_types = inf_types[b, p, :seq_len, 0]

        for m in range(M_global):
            col = 0 if col_idx == 0 else 2
            ax = axes[m, col]
            if m >= M:
                # No such mark in this dataset
                ax.axis("off")
                if m == 0:
                    ax.set_title(title, fontsize=16, fontweight="bold")
                continue

            eval_times_p = eval_times[b, p]
            pred_intensity_p_m = pred[b, m, p]
            valid_mask = eval_times_p > 0
            sort_indices = None
            if valid_mask.any():
                eval_times_p = eval_times_p[valid_mask]
                pred_intensity_p_m = pred_intensity_p_m[valid_mask]
                sort_indices = np.argsort(eval_times_p)
                eval_times_p = eval_times_p[sort_indices]
                pred_intensity_p_m = pred_intensity_p_m[sort_indices]
            else:
                eval_times_p = np.array([0.0])
                pred_intensity_p_m = np.array([0.0])

            # Shift to absolute time if offsets available
            if offsets_np is not None:
                eval_times_plot = eval_times_p + offsets_np[b, p]
            else:
                eval_times_plot = eval_times_p

            # Plot predicted intensity
            ax.plot(
                eval_times_plot,
                pred_intensity_p_m,
                color="#0072B2",
                linestyle="-",
                linewidth=3.5,
                label="FIM-PP (zero-shot)",
                alpha=1,
            )

            # Optional target
            if target is not None:
                target_intensity_p_m = target[b, m, p]
                if valid_mask.any() and sort_indices is not None:
                    target_intensity_p_m = target_intensity_p_m[valid_mask]
                    target_intensity_p_m = target_intensity_p_m[sort_indices]
                else:
                    target_intensity_p_m = np.array([0.0])
                ax.plot(
                    eval_times_plot,
                    target_intensity_p_m,
                    color="black",
                    linestyle="--",
                    linewidth=1.8,
                    label="Ground Truth",
                    alpha=1,
                )

            # Events of current mark (green) with distinct marker
            events_this_mark = all_event_times[all_event_types == m]
            if len(events_this_mark) > 0:
                event_intensities = []
                for event_time in events_this_mark:
                    closest_idx = np.argmin(np.abs(eval_times_p - event_time))
                    event_intensities.append(pred_intensity_p_m[closest_idx])
                if offsets_np is not None:
                    events_plot = events_this_mark + offsets_np[b, p]
                else:
                    events_plot = events_this_mark
                ax.scatter(
                    events_plot,
                    event_intensities,
                    s=100,
                    c="#CC79A7",
                    marker=marker_cycle[m % len(marker_cycle)],
                    label=None,
                    zorder=10,
                    edgecolors="#CC79A7",
                    linewidth=1.5,
                    alpha=1,
                )

            # Events of other marks (grey) with their own markers
            for k in range(M):
                if k == m:
                    continue
                events_k = all_event_times[all_event_types == k]
                if len(events_k) == 0:
                    continue
                other_event_intensities = []
                for event_time in events_k:
                    closest_idx = np.argmin(np.abs(eval_times_p - event_time))
                    other_event_intensities.append(pred_intensity_p_m[closest_idx])
                if offsets_np is not None:
                    events_other_plot = events_k + offsets_np[b, p]
                else:
                    events_other_plot = events_k
                ax.scatter(
                    events_other_plot,
                    other_event_intensities,
                    s=60,
                    c="gray",
                    marker=marker_cycle[k % len(marker_cycle)],
                    label=None,
                    zorder=8,
                    edgecolors="dimgray",
                    linewidth=1.0,
                    alpha=1,
                )

            ax.set_ylabel("Intensity", fontsize=15)
            if m == 0:
                ax.set_title(title, fontsize=18, fontweight="bold")
            if m == M_global - 1:
                ax.set_xlabel("Time", fontsize=15)
            # Axis aesthetics: slimmer spines/ticks, no grid, auto y-limits
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
            ax.tick_params(axis="both", direction="out", width=0.5, length=2, pad=0.8)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
            ax.grid(False)
            ax.margins(x=0)

    _plot_dataset(
        0,
        left_pred,
        left_eval_times,
        left_inf_times,
        left_inf_types,
        left_inf_lengths,
        left_target,
        left_offsets_np,
        left_path_idx,
        left_title,
    )
    _plot_dataset(
        1,
        right_pred,
        right_eval_times,
        right_inf_times,
        right_inf_types,
        right_inf_lengths,
        right_target,
        right_offsets_np,
        right_path_idx,
        right_title,
    )

    # Unified legend at figure level from the first row, left column
    plt.draw()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    keep_labels = {"FIM-PP (zero-shot)", "Ground Truth"}
    uniq = []
    seen = set()
    for h, l in zip(handles, labels):
        if l in keep_labels and l not in seen:
            uniq.append((h, l))
            seen.add(l)

    # Custom legend entries for three mark symbols (black markers) on top
    mark_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_cycle[i % len(marker_cycle)],
            linestyle="",
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=10,
        )
        for i in range(3)
    ]
    mark_labels = [f"Mark {i + 1}" for i in range(3)]

    combined_handles = [h for h, _ in uniq] + mark_handles
    combined_labels = [l for _, l in uniq] + mark_labels

    # Place legend centered above both columns; reserve extra top margin for it
    fig.legend(
        combined_handles,
        combined_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncols=len(combined_labels),
        fontsize=18,
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.945])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Also save as PDF for LaTeX-quality vector graphics
    try:
        pdf_path = str(Path(save_path).with_suffix(".pdf"))
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"Side-by-side intensity comparison saved to {pdf_path}")
    except Exception as e:
        print(f"Warning: failed to save PDF version: {e}")
    plt.close()
    print(f"Side-by-side intensity comparison saved to {save_path}")


def load_fimhawkes_with_proper_weights(checkpoint_path):
    """
    Load FIMHawkes model from a checkpoint directory using the generic AModel loader.

    Expects files: config.json and model-checkpoint.pth inside the checkpoint directory.
    """
    checkpoint_path = Path(checkpoint_path)
    return FIMHawkes.load_model(checkpoint_path)


def main(args):
    # Select device dynamically; do not hide CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_checkpoint = Path(args.checkpoint)
    dataset_dir = Path(args.dataset)

    print(f"Loading model from: {model_checkpoint}")
    print(f"Loading dataset from: {dataset_dir}")

    # Load model with proper weight loading (bypasses transformers issues)
    model = load_fimhawkes_with_proper_weights(model_checkpoint)
    model.eval()
    model.to(device)

    # Branch: CDiff datasets vs synthetic .pt directory datasets
    # If a right dataset is provided, generate a side-by-side figure
    if getattr(args, "right_dataset", None):
        # LEFT SIDE: args.dataset
        use_cdiff_left = _is_cdiff_dataset_dir(_resolve_cdiff_dir(dataset_dir))
        if use_cdiff_left:
            print("[Left] Detected CDiff dataset layout. Using train as context and val as inference.")
            print(f"[Left] Using validation index (inference path) from CDiff val split: {args.path_idx}")
            left_model_data = build_model_data_from_cdiff(dataset_dir, val_index=args.path_idx, max_context_paths=2000)
            left_effective_path_idx = 0
            try:
                num_marks = int(left_model_data.get("num_marks", 0))
                if num_marks > 0:
                    setattr(model.config, "max_num_marks", num_marks)
            except Exception:
                pass
        else:
            left_data = load_data_from_dir(dataset_dir)
            if not left_data:
                print("[Left] No data loaded. Exiting.")
                return
            print(f"[Left] Loaded data with keys: {list(left_data.keys())}")
            left_sample_idx = args.sample_idx
            sample_count = None
            for key, value in left_data.items():
                if torch.is_tensor(value):
                    sample_count = value.shape[0]
                    break
            if sample_count is not None and left_sample_idx >= sample_count:
                print(f"[Left] Warning: sample_idx {left_sample_idx} >= number of samples {sample_count}. Using sample 0.")
                left_sample_idx = 0
            print(f"[Left] Using sample index: {left_sample_idx}")
            single_sample_left = {}
            for key, value in left_data.items():
                if torch.is_tensor(value):
                    single_sample_left[key] = value[left_sample_idx]
                else:
                    single_sample_left[key] = value[left_sample_idx]
            try:
                left_model_data = prepare_batch_for_model(single_sample_left, args.path_idx, num_points_between_events=10)
            except ValueError as e:
                print(f"[Left] Error preparing batch: {e}")
                return
            left_effective_path_idx = args.path_idx

        # RIGHT SIDE: args.right_dataset (can be CDiff or synthetic)
        right_dataset_dir = Path(args.right_dataset)
        use_cdiff_right = _is_cdiff_dataset_dir(_resolve_cdiff_dir(right_dataset_dir))
        if use_cdiff_right:
            print("[Right] Detected CDiff dataset layout. Using train as context and val as inference.")
            print(f"[Right] Using validation index (inference path) from CDiff val split: {args.path_idx}")
            right_model_data = build_model_data_from_cdiff(right_dataset_dir, val_index=args.path_idx, max_context_paths=2000)
            right_effective_path_idx = 0
        else:
            right_data = load_data_from_dir(right_dataset_dir)
            if not right_data:
                print("[Right] No data loaded. Exiting.")
                return
            print(f"[Right] Loaded data with keys: {list(right_data.keys())}")
            right_sample_idx = args.sample_idx
            sample_count_r = None
            for key, value in right_data.items():
                if torch.is_tensor(value):
                    sample_count_r = value.shape[0]
                    break
            if sample_count_r is not None and right_sample_idx >= sample_count_r:
                print(f"[Right] Warning: sample_idx {right_sample_idx} >= number of samples {sample_count_r}. Using sample 0.")
                right_sample_idx = 0
            print(f"[Right] Using sample index: {right_sample_idx}")
            single_sample_right = {}
            for key, value in right_data.items():
                if torch.is_tensor(value):
                    single_sample_right[key] = value[right_sample_idx]
                else:
                    single_sample_right[key] = value[right_sample_idx]
            try:
                right_model_data = prepare_batch_for_model(single_sample_right, args.path_idx, num_points_between_events=10)
            except ValueError as e:
                print(f"[Right] Error preparing batch: {e}")
                return
            right_effective_path_idx = args.path_idx

        # Run model on both sides
        left_model_data = _move_to_device(left_model_data, device)
        right_model_data = _move_to_device(right_model_data, device)
        with torch.no_grad():
            left_output = model(left_model_data)
            right_output = model(right_model_data)

        # Titles
        left_title = args.left_title
        right_title = args.right_title

        # Plot and save
        plot_two_datasets_side_by_side(
            left_output,
            left_model_data,
            right_output,
            right_model_data,
            left_title,
            right_title,
            save_path=args.save_path_comparison,
            left_path_idx=left_effective_path_idx,
            right_path_idx=right_effective_path_idx,
        )
        return

    # Single-dataset mode (default)
    use_cdiff = _is_cdiff_dataset_dir(_resolve_cdiff_dir(dataset_dir))
    if use_cdiff:
        print("Detected CDiff dataset layout. Using train as context and val as inference.")
        # Interpret path_idx as the validation sequence index for CDiff datasets
        print(f"Using validation index (inference path) from CDiff val split: {args.path_idx}")
        model_data = build_model_data_from_cdiff(dataset_dir, val_index=args.path_idx, max_context_paths=2000)
        # For CDiff path selection, we have P_inference=1; force path_idx=0 for plotting
        effective_path_idx = 0
        # Align model config marks if possible
        try:
            num_marks = int(model_data.get("num_marks", 0))
            if num_marks > 0:
                setattr(model.config, "max_num_marks", num_marks)
        except Exception:
            pass
    else:
        data = load_data_from_dir(dataset_dir)
        if not data:
            print("No data loaded. Exiting.")
            return

        print(f"Loaded data with keys: {list(data.keys())}")

        # Use the specified sample index
        sample_idx = args.sample_idx

        # Validate sample index
        sample_count = None
        for key, value in data.items():
            if torch.is_tensor(value):
                sample_count = value.shape[0]
                break

        if sample_count is not None and sample_idx >= sample_count:
            print(f"Warning: sample_idx {sample_idx} >= number of samples {sample_count}. Using sample 0.")
            sample_idx = 0

        print(f"Using sample index: {sample_idx}")

        single_sample_data = {}
        for key, value in data.items():
            if torch.is_tensor(value):
                single_sample_data[key] = value[sample_idx]
            else:
                # Handle non-tensor data if necessary, e.g., lists of tensors
                single_sample_data[key] = value[sample_idx]

        try:
            model_data = prepare_batch_for_model(single_sample_data, args.path_idx, num_points_between_events=10)
        except ValueError as e:
            print(f"Error preparing batch: {e}")
            return
        effective_path_idx = args.path_idx

    print("Model input shapes:")
    for key, value in model_data.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape}")

    print(f"Using path index: {effective_path_idx}")

    # Ensure model inputs are on the same device as the model
    model_data = _move_to_device(model_data, device)

    with torch.no_grad():
        model_output = model(model_data)

    print(f"Model output keys: {list(model_output.keys())}")

    save_suffix = f"sample_{args.sample_idx}_path_{effective_path_idx}"
    save_path = f"intensity_comparison_{save_suffix}.png"
    plot_intensity_comparison(model_output, model_data, save_path, path_idx=effective_path_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Hawkes Process Intensity Functions.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/FIM_Hawkes_1-3st_optimized_mixed_rmse_norm_2000_paths_mixed_250_events_mixed-experiment-seed-10-dataset-dataset_kwargs-field_name_for_dimension_grouping-base_intensity_functions_07-16-1816/checkpoints/best-model",
        help="Path to the model checkpoint directory.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/synthetic_data/hawkes/1k_2D_1k_paths_diag_only_old_params/test",
        help="Path to the validation/test dataset directory.",
    )
    parser.add_argument(
        "--right_dataset",
        type=str,
        default=None,
        help="Optional second dataset to plot side-by-side (e.g., 'retweet').",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of the sample to visualize from the dataset (default: 0).",
    )
    parser.add_argument(
        "--path_idx",
        type=int,
        default=0,
        help="Index of the inference path to visualize (default: 0).",
    )
    parser.add_argument(
        "--left_title",
        type=str,
        default="Synthetic Hawkes Process",
        help="Title for the left subplot when comparing two datasets.",
    )
    parser.add_argument(
        "--right_title",
        type=str,
        default="Retweet Dataset",
        help="Title for the right subplot when comparing two datasets.",
    )
    parser.add_argument(
        "--save_path_comparison",
        type=str,
        default="intensity_comparison_synthetic_vs_retweet.png",
        help="Output path for the side-by-side comparison figure.",
    )
    args = parser.parse_args()
    main(args)
