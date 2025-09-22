"""
CUDA_VISIBLE_DEVICES="" python scripts/hawkes/visualize_intensity_prediction_paper.py \
--checkpoint "results/FIM_Hawkes_10-22st_nll_only_mc_2000_paths_mixed_100_events_mixed-experiment-seed-10-dataset-dataset_kwargs-field_name_for_dimension_grouping-base_intensity_functions_09-20-1214/checkpoints/best-model"  \
--dataset "data/synthetic_data/hawkes/1k_3D_2k_paths_const_base_exp_kernel_no_interactions/test" \
--sample_idx 0 \
--path_idx 0
"""

#!/usr/bin/env python
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from fim.models.hawkes import FIMHawkes, FIMHawkesConfig


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

    b = 0  # Always use first batch
    p = path_idx  # Use specified path index

    # Validate path index
    if p >= P_inference:
        print(f"Warning: path_idx {p} >= number of inference paths {P_inference}. Using path 0.")
        p = 0

    fig, axes = plt.subplots(M, 1, figsize=(15, 5 * M), sharex=True)
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
        ax.plot(eval_times_plot, pred_intensity_p_m, "b-", linewidth=2, label="Predicted Intensity", alpha=0.8)

        if target_intensities is not None:
            target_intensity_p_m = target_intensities[b, m, p]

            # Apply the same filtering and sorting to target intensities
            if valid_mask.any() and sort_indices is not None:
                target_intensity_p_m = target_intensity_p_m[valid_mask]
                target_intensity_p_m = target_intensity_p_m[sort_indices]
            else:
                target_intensity_p_m = np.array([0.0])

            ax.plot(eval_times_plot, target_intensity_p_m, "r--", linewidth=2, label="Ground Truth Intensity", alpha=0.8)

        # Plot event scatter marks prominently
        # Events of the current mark
        events_this_mark = all_event_times[all_event_types == m]
        if len(events_this_mark) > 0:
            # Get intensity values at event times for this mark
            event_intensities = []
            for event_time in events_this_mark:
                # Find closest evaluation time to get intensity value
                # Use shifted timeline for indexing; plotting x is shifted later if offset available
                closest_idx = np.argmin(np.abs(eval_times_p - event_time))
                event_intensities.append(pred_intensity_p_m[closest_idx])

            # Shift x for scatter if offsets are available
            if offsets_np is not None:
                events_plot = events_this_mark + offsets_np[b, p]
            else:
                events_plot = events_this_mark

            ax.scatter(
                events_plot,
                event_intensities,
                s=100,
                c="green",
                marker="o",
                label=f"Events (Mark {m})",
                zorder=10,
                edgecolors="darkgreen",
                linewidth=1.5,
                alpha=0.9,
            )

        # Events of other marks (smaller markers)
        events_other_marks = all_event_times[all_event_types != m]
        if len(events_other_marks) > 0:
            # Get intensity values at other event times
            other_event_intensities = []
            for event_time in events_other_marks:
                # Find closest evaluation time to get intensity value
                closest_idx = np.argmin(np.abs(eval_times_p - event_time))
                other_event_intensities.append(pred_intensity_p_m[closest_idx])

            if offsets_np is not None:
                events_other_plot = events_other_marks + offsets_np[b, p]
            else:
                events_other_plot = events_other_marks

            ax.scatter(
                events_other_plot, other_event_intensities, s=60, c="gray", marker="x", label="Events (Other Marks)", zorder=8, alpha=0.7
            )

        ax.set_ylabel("Intensity", fontsize=12)
        ax.set_title(f"Intensity Function for Mark {m}", fontsize=14, fontweight="bold")

        # Create a clean legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=10)

        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Time", fontsize=12)
    plt.suptitle(f"Hawkes Process Intensity Functions - Path {path_idx}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust for suptitle

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Intensity comparison plot saved to {save_path}")


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
    use_cdiff = _is_cdiff_dataset_dir(_resolve_cdiff_dir(dataset_dir))
    if use_cdiff:
        print("Detected CDiff dataset layout. Using train as context and val as inference.")
        model_data = build_model_data_from_cdiff(dataset_dir, val_index=args.sample_idx, max_context_paths=2000)
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
    args = parser.parse_args()
    main(args)
