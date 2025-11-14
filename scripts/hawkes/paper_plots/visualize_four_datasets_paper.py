"""
conda run -n model_training python scripts/hawkes/paper_plots/visualize_four_datasets_paper.py \
  --checkpoint "results/.ICLR_submission_model/checkpoints/best-model" \
  --datasets "data/synthetic_data/hawkes/1k_3D_2k_paths_Gamma_base_exp_kernel_sparse/test,data/synthetic_data/hawkes/1k_3D_2k_paths_poisson/test,data/synthetic_data/hawkes/10_3D_2k_context_paths_100_inference_paths_const_powerlaw/test,data/synthetic_data/hawkes/10_3D_2k_paths_const_base_rayleigh_kernel_sparse/test" \
  --sample_indices "20,1,5,3" \
  --path_idx 0 \
  --save_path intensity_comparison_four_datasets.png
"""

#!/usr/bin/env python
import argparse

# Import helper functions from the original script
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from fim.models.hawkes import FIMHawkes, FIMHawkesConfig


sys.path.insert(0, str(Path(__file__).parent))
from visualize_intensity_prediction_paper import (
    _move_to_device,
    load_data_from_dir,
    load_fimhawkes_with_proper_weights,
    prepare_batch_for_model,
)


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
FIMHawkesConfig.register_for_auto_class()
FIMHawkes.register_for_auto_class("AutoModel")


def _extract_title_from_path(dataset_path: str) -> str:
    """Extract a readable title from dataset path."""
    path = Path(dataset_path)
    path_str = str(path)

    # Check for specific dataset patterns and assign titles
    if "Gamma_base" in path_str or "gamma_base" in path_str.lower():
        return "Gamma Base-Intensity"
    elif "poisson" in path_str.lower():
        return "Poisson"
    elif "powerlaw" in path_str.lower():
        return "Powerlaw Kernel"
    elif "rayleigh" in path_str.lower():
        return "Rayleigh Kernel"

    # Fallback: Get the parent directory name
    parent_name = path.parent.name
    # Replace underscores with spaces and capitalize
    title = parent_name.replace("_", " ")
    # Make it more readable
    title = title.replace("1k", "1K").replace("2k", "2K").replace("10", "10")
    return title


def plot_four_datasets_grid(
    outputs: List[Dict],
    model_datas: List[Dict],
    titles: List[str],
    save_path: str = "intensity_comparison_four_datasets.png",
    path_idx: int = 0,
):
    """
    Create a 2x2 grid plot comparing 4 datasets.
    Each subplot shows all marks stacked vertically for one dataset.

    Args:
        outputs: List of 4 model outputs (each with predicted_intensity_values, optionally target_intensity_values)
        model_datas: List of 4 model data dicts (each with intensity_evaluation_times, inference_event_times, etc.)
        titles: List of 4 titles for each subplot
        save_path: Output file path
        path_idx: Path index to use for all datasets
    """
    if len(outputs) != 4 or len(model_datas) != 4 or len(titles) != 4:
        raise ValueError("Must provide exactly 4 outputs, 4 model_datas, and 4 titles")

    # Extract arrays for all datasets
    all_preds = []
    all_eval_times = []
    all_inf_times = []
    all_inf_types = []
    all_inf_lengths = []
    all_targets = []
    all_offsets = []

    for i in range(4):
        pred = outputs[i]["predicted_intensity_values"].detach().cpu().numpy()
        all_preds.append(pred)

        eval_times = model_datas[i]["intensity_evaluation_times"].detach().cpu().numpy()
        all_eval_times.append(eval_times)

        inf_times = model_datas[i]["inference_event_times"].detach().cpu().numpy()
        all_inf_times.append(inf_times)

        inf_types = model_datas[i]["inference_event_types"].detach().cpu().numpy()
        all_inf_types.append(inf_types)

        inf_lengths = model_datas[i]["inference_seq_lengths"].detach().cpu().numpy()
        all_inf_lengths.append(inf_lengths)

        if "target_intensity_values" in outputs[i]:
            target = outputs[i]["target_intensity_values"].detach().cpu().numpy()
        else:
            target = None
        all_targets.append(target)

        offsets_np = None
        if "inference_time_offsets" in model_datas[i]:
            off = model_datas[i]["inference_time_offsets"].detach().cpu().numpy()
            if off.ndim == 3 and off.shape[-1] == 1:
                off = off[..., 0]
            offsets_np = off
        all_offsets.append(offsets_np)

    # Find maximum number of marks across all datasets
    M_max = max([pred.shape[1] for pred in all_preds])

    # Marker cycle for distinct tick/shape per mark
    marker_cycle = ["o", "s", "^", "D", "v", "P", "*", "X", "<", ">", "h", "8", "+", "x", "|", "_"]

    # Create figure with GridSpec: 2 rows x 2 columns for datasets
    # Each dataset cell will contain M marks stacked vertically

    fig = plt.figure(figsize=(20, 4 * M_max))
    gs_main = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Helper function to plot one dataset into a subplot
    def _plot_dataset(row: int, col: int, pred, eval_times, inf_times, inf_types, inf_lengths, target, offsets_np, title):
        b = 0
        p = path_idx
        _, M, P, _ = pred.shape

        if p >= P:
            p = 0

        seq_len = inf_lengths[b, p]
        all_event_times = inf_times[b, p, :seq_len, 0]
        all_event_types = inf_types[b, p, :seq_len, 0]

        # Create nested GridSpec for marks within this dataset's cell
        # Get the position of the main cell by creating a temporary subplot
        temp_ax = fig.add_subplot(gs_main[row, col])
        bbox = temp_ax.get_position()
        temp_ax.remove()

        # Create M subplots for marks within this cell
        gs_marks = GridSpec(M, 1, figure=fig, left=bbox.x0, right=bbox.x1, top=bbox.y1, bottom=bbox.y0, hspace=0.3)

        mark_axes = []
        for m in range(M):
            ax = fig.add_subplot(gs_marks[m, 0])
            mark_axes.append(ax)

        for m in range(M):
            ax = mark_axes[m]

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
                linewidth=2,
                label="FIM-PP (zero-shot)" if m == 0 else None,
                alpha=1,
            )

            # Optional target
            target_intensity_p_m = None
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
                    label="Ground Truth" if m == 0 else None,
                    alpha=1,
                )

            # Events of current mark
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

            # Events of other marks
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

            ax.set_ylabel("Intensity", fontsize=12)
            if m == 0:
                ax.set_title(title, fontsize=16, fontweight="bold")
            if m == M - 1:
                ax.set_xlabel("Time", fontsize=14)

            # Set y-axis limits with padding
            # Collect all y-values (predicted and target if available)
            y_min = pred_intensity_p_m.min()
            y_max = pred_intensity_p_m.max()
            if target_intensity_p_m is not None and len(target_intensity_p_m) > 0:
                y_min = min(y_min, target_intensity_p_m.min())
                y_max = max(y_max, target_intensity_p_m.max())

            # Calculate padding based on range
            y_range = y_max - y_min
            if y_range > 0:
                # For Poisson dataset (detected by title containing "poisson"), use larger padding
                is_poisson = "poisson" in title.lower()
                padding_factor = 0.25 if is_poisson else 0.15  # 25% padding for Poisson, 15% for others
                padding = y_range * padding_factor
                y_min_padded = max(0, y_min - padding)  # Don't go below 0
                y_max_padded = y_max + padding
                ax.set_ylim(y_min_padded, y_max_padded)

            # Axis aesthetics
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
            ax.tick_params(axis="both", direction="out", width=0.5, length=2, pad=0.8)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
            ax.grid(False)
            ax.margins(x=0)

    # Plot each dataset
    # Grid positions: (0,0) top-left, (0,1) top-right, (1,0) bottom-left, (1,1) bottom-right
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i in range(4):
        row, col = positions[i]
        _plot_dataset(
            row,
            col,
            all_preds[i],
            all_eval_times[i],
            all_inf_times[i],
            all_inf_types[i],
            all_inf_lengths[i],
            all_targets[i],
            all_offsets[i],
            titles[i],
        )

    # Unified legend at figure level
    plt.draw()
    # Get handles from first subplot's first mark axis
    first_ax = fig.axes[0]  # First mark axis of first dataset
    handles, labels = first_ax.get_legend_handles_labels()
    keep_labels = {"FIM-PP (zero-shot)", "Ground Truth"}
    uniq = []
    seen = set()
    for h, l in zip(handles, labels):
        if l in keep_labels and l not in seen:
            uniq.append((h, l))
            seen.add(l)

    # Add custom legend entries for mark symbols
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
        for i in range(min(3, M_max))
    ]
    mark_labels = [f"Mark {i + 1}" for i in range(min(3, M_max))]

    combined_handles = [h for h, _ in uniq] + mark_handles
    combined_labels = [l for _, l in uniq] + mark_labels

    fig.legend(
        combined_handles,
        combined_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncols=len(combined_labels),
        fontsize=16,
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Also save as PDF
    try:
        pdf_path = str(Path(save_path).with_suffix(".pdf"))
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"Four-dataset intensity comparison saved to {pdf_path}")
    except Exception as e:
        print(f"Warning: failed to save PDF version: {e}")
    plt.close()
    print(f"Four-dataset intensity comparison saved to {save_path}")


def main(args):
    # Select device dynamically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_checkpoint = Path(args.checkpoint)

    print(f"Loading model from: {model_checkpoint}")

    # Load model
    model = load_fimhawkes_with_proper_weights(model_checkpoint)
    model.eval()
    model.to(device)

    # Parse datasets and sample indices
    dataset_paths = [Path(d.strip()) for d in args.datasets.split(",")]
    if len(dataset_paths) != 4:
        raise ValueError(f"Must provide exactly 4 datasets, got {len(dataset_paths)}")

    sample_indices = [int(s.strip()) for s in args.sample_indices.split(",")]
    if len(sample_indices) != 4:
        raise ValueError(f"Must provide exactly 4 sample indices, got {len(sample_indices)}")

    # Generate titles if not provided
    if args.titles:
        titles = [t.strip() for t in args.titles.split(",")]
        if len(titles) != 4:
            raise ValueError(f"Must provide exactly 4 titles, got {len(titles)}")
    else:
        titles = [_extract_title_from_path(str(d)) for d in dataset_paths]

    # Process each dataset
    outputs = []
    model_datas = []

    for i, (dataset_path, sample_idx) in enumerate(zip(dataset_paths, sample_indices)):
        print(f"\n[{i + 1}/4] Processing dataset: {dataset_path}")
        print(f"  Sample index: {sample_idx}")

        # Load data
        data = load_data_from_dir(dataset_path)
        if not data:
            print(f"  No data loaded. Skipping dataset {i + 1}.")
            continue

        print(f"  Loaded data with keys: {list(data.keys())}")

        # Validate sample index
        sample_count = None
        for key, value in data.items():
            if torch.is_tensor(value):
                sample_count = value.shape[0]
                break

        if sample_count is not None and sample_idx >= sample_count:
            print(f"  Warning: sample_idx {sample_idx} >= number of samples {sample_count}. Using sample 0.")
            sample_idx = 0

        # Extract single sample
        single_sample_data = {}
        for key, value in data.items():
            if torch.is_tensor(value):
                single_sample_data[key] = value[sample_idx]
            else:
                single_sample_data[key] = value[sample_idx]

        # Prepare batch
        try:
            model_data = prepare_batch_for_model(single_sample_data, args.path_idx, num_points_between_events=10)
        except ValueError as e:
            print(f"  Error preparing batch: {e}")
            raise

        # Update model config if needed
        if "base_intensity_functions" in model_data:
            num_marks = model_data["base_intensity_functions"].shape[1]
            try:
                setattr(model.config, "max_num_marks", num_marks)
            except Exception:
                pass

        # Move to device and run inference
        model_data = _move_to_device(model_data, device)
        with torch.no_grad():
            output = model(model_data)

        outputs.append(output)
        model_datas.append(model_data)

    if len(outputs) != 4:
        raise ValueError(f"Failed to process all 4 datasets. Only {len(outputs)} succeeded.")

    # Plot
    plot_four_datasets_grid(
        outputs,
        model_datas,
        titles,
        save_path=args.save_path,
        path_idx=args.path_idx,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 4 Hawkes Process Datasets in a 2x2 Grid.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/.ICLR_submission_model/checkpoints/best-model",
        help="Path to the model checkpoint directory.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated list of 4 dataset paths.",
    )
    parser.add_argument(
        "--sample_indices",
        type=str,
        default="0,0,0,0",
        help="Comma-separated list of 4 sample indices (one per dataset), default: '0,0,0,0'.",
    )
    parser.add_argument(
        "--path_idx",
        type=int,
        default=0,
        help="Path index to use for all datasets (default: 0).",
    )
    parser.add_argument(
        "--titles",
        type=str,
        default=None,
        help="Optional comma-separated list of 4 titles (auto-generated from folder names if not provided).",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="intensity_comparison_four_datasets.png",
        help="Output path for the figure.",
    )
    args = parser.parse_args()
    main(args)
