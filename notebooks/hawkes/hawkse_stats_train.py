# %%
# Configure matplotlib to avoid Type 1 fonts - MUST BE RUN FIRST
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from fim.data.dataloaders import DataLoaderFactory
import torch
from torch import Tensor


from datasets import Dataset, DatasetDict

# Set global font configuration to avoid Type 1 fonts
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Bitstream Vera Sans", "sans-serif"],
        "pdf.fonttype": 42,  # Use TrueType fonts instead of Type 1
        "ps.fonttype": 42,  # Use TrueType fonts instead of Type 1
        "axes.unicode_minus": False,  # Avoid issues with minus signs
        "figure.dpi": 300,  # High DPI for better quality
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

print("✅ Font configuration updated to avoid Type 1 fonts")


# %%
# Configuration for different dimensions
configs_by_dimension = {
    "1D": {
        "name": "HawkesDataLoader",
        "path": {
            "train": (
                "/cephfs_projects/foundation_models/data/PP/train/1k_1D_2k_paths_const_base_exp_kernel/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_1D_2k_paths_const_base_exp_kernel_no_interactions/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_1D_2k_paths_Gamma_base_exp_kernel/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_1D_2k_paths_const_base_rayleigh_kernel/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_1D_2k_paths_const_base_rayleigh_kernel_sparse/train",
            ),
        },
        "loader_kwargs": {
            "batch_size": 1,
            "num_workers": 0,
            "test_batch_size": 1,
            "variable_num_of_paths": True,
            "min_path_count": 2,
            "max_path_count": 2,
            "max_number_of_minibatch_sizes": 1,
            "variable_sequence_lens": {
                "train": True,
                "validation": False,
            },
            "min_sequence_len": 15,
            "max_sequence_len": 100,
            "full_len_ratio": 0.1,
            "num_inference_paths": 1,
            "num_inference_times": 2000,
        },
        "dataset_kwargs": {
            "files_to_load": {
                "base_intensity_functions": "base_intensity_functions.pt",
                "event_times": "event_times.pt",
                "event_types": "event_types.pt",
                "kernel_functions": "kernel_functions.pt",
            },
            "field_name_for_dimension_grouping": ["base_intensity_functions", "kernel_functions"],
        },
    },
    "5D": {
        "name": "HawkesDataLoader",
        "path": {
            "train": (
                "/cephfs_projects/foundation_models/data/PP/train/1k_5D_2k_paths_const_base_exp_kernel/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_5D_2k_paths_const_base_exp_kernel_no_interactions/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_5D_2k_paths_Gamma_base_exp_kernel/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_5D_2k_paths_const_base_rayleigh_kernel/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_5D_2k_paths_const_base_rayleigh_kernel_sparse/train",
            ),
        },
        "loader_kwargs": {
            "batch_size": 1,
            "num_workers": 0,
            "test_batch_size": 1,
            "variable_num_of_paths": True,
            "min_path_count": 2,
            "max_path_count": 2,
            "max_number_of_minibatch_sizes": 1,
            "variable_sequence_lens": {
                "train": True,
                "validation": False,
            },
            "min_sequence_len": 15,
            "max_sequence_len": 100,
            "full_len_ratio": 0.1,
            "num_inference_paths": 1,
            "num_inference_times": 2000,
        },
        "dataset_kwargs": {
            "files_to_load": {
                "base_intensity_functions": "base_intensity_functions.pt",
                "event_times": "event_times.pt",
                "event_types": "event_types.pt",
                "kernel_functions": "kernel_functions.pt",
            },
            "field_name_for_dimension_grouping": ["base_intensity_functions", "kernel_functions"],
        },
    },
    "10D": {
        "name": "HawkesDataLoader",
        "path": {
            "train": (
                "/cephfs_projects/foundation_models/data/PP/train/1k_10D_2k_paths_const_base_exp_kernel/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_10D_2k_paths_const_base_exp_kernel_no_interactions/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_10D_2k_paths_poisson/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_10D_2k_paths_Gamma_base_exp_kernel/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_10D_2k_paths_sin_base_exp_kernel/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_10D_2k_paths_const_base_exp_kernel_sparse/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_10D_2k_paths_Gamma_base_exp_kernel_sparse/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_10D_2k_paths_sin_base_exp_kernel_sparse/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_10D_2k_paths_const_base_rayleigh_kernel_sparse/train",
            ),
        },
        "loader_kwargs": {
            "batch_size": 1,
            "num_workers": 0,
            "test_batch_size": 1,
            "variable_num_of_paths": True,
            "min_path_count": 2,
            "max_path_count": 2,
            "max_number_of_minibatch_sizes": 1,
            "variable_sequence_lens": {
                "train": True,
                "validation": False,
            },
            "min_sequence_len": 15,
            "max_sequence_len": 100,
            "full_len_ratio": 0.1,
            "num_inference_paths": 1,
            "num_inference_times": 2000,
        },
        "dataset_kwargs": {
            "files_to_load": {
                "base_intensity_functions": "base_intensity_functions.pt",
                "event_times": "event_times.pt",
                "event_types": "event_types.pt",
                "kernel_functions": "kernel_functions.pt",
            },
            "field_name_for_dimension_grouping": ["base_intensity_functions", "kernel_functions"],
        },
    },
    "15D": {
        "name": "HawkesDataLoader",
        "path": {
            "train": (
                "/cephfs_projects/foundation_models/data/PP/train/1k_15D_2k_paths_const_base_exp_kernel/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_15D_2k_paths_const_base_exp_kernel_no_interactions/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_15D_2k_paths_poisson/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_15D_2k_paths_Gamma_base_exp_kernel/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_15D_2k_paths_sin_base_exp_kernel/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_15D_2k_paths_const_base_exp_kernel_sparse/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_15D_2k_paths_Gamma_base_exp_kernel_sparse/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_15D_2k_paths_sin_base_exp_kernel_sparse/train",
                "/cephfs_projects/foundation_models/data/PP/train/1k_15D_2k_paths_const_base_rayleigh_kernel/train",
            ),
        },
        "loader_kwargs": {
            "batch_size": 1,
            "num_workers": 0,
            "test_batch_size": 1,
            "variable_num_of_paths": True,
            "min_path_count": 2,
            "max_path_count": 2,
            "max_number_of_minibatch_sizes": 1,
            "variable_sequence_lens": {
                "train": True,
                "validation": False,
            },
            "min_sequence_len": 15,
            "max_sequence_len": 100,
            "full_len_ratio": 0.1,
            "num_inference_paths": 1,
            "num_inference_times": 2000,
        },
        "dataset_kwargs": {
            "files_to_load": {
                "base_intensity_functions": "base_intensity_functions.pt",
                "event_times": "event_times.pt",
                "event_types": "event_types.pt",
                "kernel_functions": "kernel_functions.pt",
            },
            "field_name_for_dimension_grouping": ["base_intensity_functions", "kernel_functions"],
        },
    },
    "22D": {
        "name": "HawkesDataLoader",
        "path": {
            "train": (
                "/cephfs_projects/foundation_models/data/PP/train/5k_22D_2k_paths_const_base_exp_kernel/train",
                "/cephfs_projects/foundation_models/data/PP/train/5k_22D_2k_paths_const_base_exp_kernel_no_interactions/train",
                "/cephfs_projects/foundation_models/data/PP/train/5k_22D_2k_paths_poisson/train",
                "/cephfs_projects/foundation_models/data/PP/train/5k_22D_2k_paths_Gamma_base_exp_kernel/train",
                "/cephfs_projects/foundation_models/data/PP/train/5k_22D_2k_paths_sin_base_exp_kernel/train",
                "/cephfs_projects/foundation_models/data/PP/train/5k_22D_2k_paths_const_base_exp_kernel_sparse/train",
                "/cephfs_projects/foundation_models/data/PP/train/5k_22D_2k_paths_const_base_rayleigh_kernel/train",
                "/cephfs_projects/foundation_models/data/PP/train/5k_22D_2k_paths_const_base_rayleigh_kernel_sparse/train",
            ),
        },
        "loader_kwargs": {
            "batch_size": 1,
            "num_workers": 0,
            "test_batch_size": 1,
            "variable_num_of_paths": True,
            "min_path_count": 2,
            "max_path_count": 2,
            "max_number_of_minibatch_sizes": 1,
            "variable_sequence_lens": {
                "train": True,
                "validation": False,
            },
            "min_sequence_len": 15,
            "max_sequence_len": 100,
            "full_len_ratio": 0.1,
            "num_inference_paths": 1,
            "num_inference_times": 2000,
        },
        "dataset_kwargs": {
            "files_to_load": {
                "base_intensity_functions": "base_intensity_functions.pt",
                "event_times": "event_times.pt",
                "event_types": "event_types.pt",
                "kernel_functions": "kernel_functions.pt",
            },
            "field_name_for_dimension_grouping": ["base_intensity_functions", "kernel_functions"],
        },
    },
}


# %%


def normalize_obs_grid(obs_grid: Tensor, seq_lengths: Tensor) -> tuple[Tensor, Tensor]:
    batch_indices = torch.arange(obs_grid.size(0), device=obs_grid.device).view(-1, 1).expand(-1, obs_grid.size(1))
    path_indices = torch.arange(obs_grid.size(1), device=obs_grid.device).view(1, -1).expand(obs_grid.size(0), -1)

    # Clamp seq_lengths to not exceed the actual sequence dimension size
    max_seq_len = obs_grid.size(2)
    clamped_seq_lengths = torch.clamp(seq_lengths, max=max_seq_len)

    max_times = obs_grid[batch_indices, path_indices, clamped_seq_lengths - 1]
    norm_constants = max_times.amax(dim=[1, 2])
    obs_grid_normalized = obs_grid / norm_constants.view(-1, 1, 1, 1)
    return obs_grid_normalized


def convert_to_easytpp_format(sample):
    # Extract tensors from the dictionary
    event_times = sample["event_times"][0]  # [P, L, 1]
    delta_times = sample["delta_times"][0]  # [P, L, 1]
    event_types = sample["event_types"][0]  # [P, L, 1]
    seq_lengths = sample["seq_lengths"][0]  # [P]
    dim_process = sample["dim_process"]

    # Get dimensions
    num_sequences = event_times.shape[0]

    # Initialize list to store the converted data
    easytpp_data = []

    # Process each sequence
    for i in range(num_sequences):
        seq_len = seq_lengths[i].item()

        # Extract valid events for this sequence
        times = event_times[i, :seq_len, 0].cpu().numpy().tolist()
        deltas = delta_times[i, :seq_len, 0].cpu().numpy().tolist()
        types = event_types[i, :seq_len, 0].cpu().numpy().tolist()

        # Create a single sequence entry
        sequence_entry = {
            "time_since_start": times,
            "time_since_last_event": deltas,
            "type_event": types,
            "seq_idx": i,
            "seq_len": seq_len,
            "dim_process": dim_process,
        }

        easytpp_data.append(sequence_entry)

    # Create a Hugging Face Dataset
    # dataset = Dataset.from_list(easytpp_data)
    return easytpp_data


# %%
def print_stats(dataset, split="test"):
    dataset = dataset[split]
    stats = {
        "num_sequences": len(dataset),
        "max_sequence_length": 0,
        "min_sequence_length": 1000000,
        "max_event_time": 0,
        "min_event_time": 1000000,
    }
    seq_lengths = [len(seq["time_since_last_event"]) for seq in dataset]
    times_since_last_event = [time for seq in dataset for time in seq["time_since_last_event"]]
    min_delta_event_time = min(times_since_last_event)
    max_delta_event_time = max(times_since_last_event)
    min_seq_length = min(seq_lengths)
    max_seq_length = max(seq_lengths)
    avg_seq_length = sum(seq_lengths) / len(seq_lengths)
    avg_delta_event_time = sum(times_since_last_event) / len(times_since_last_event)
    stats["max_sequence_length"] = max_seq_length
    stats["min_sequence_length"] = min_seq_length
    stats["num_sequences"] = len(seq_lengths)
    stats["avg_sequence_length"] = avg_seq_length
    stats["max_event_time"] = max_delta_event_time
    stats["min_event_time"] = min_delta_event_time
    stats["avg_event_time"] = avg_delta_event_time
    encoutered_marks = set()
    for seq in dataset:
        for mark in seq["type_event"]:
            if mark not in encoutered_marks:
                encoutered_marks.add(mark)
    stats["num_marks"] = len(encoutered_marks)
    pprint(stats)


# %%
import random

import seaborn as sns


# Configure seaborn to work with our font settings
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=0.8)


# --- Helper Function ---


def plot_stats(dataset, splits=["train", "validation", "test"]):
    # Set style for better-looking plots
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

    # Extract the data for each split
    split_data = {split: dataset[split] for split in splits if split in dataset}

    # Create subplots
    fig, axes = plt.subplots(len(split_data), 2, figsize=(15, 6 * len(split_data)))
    if len(split_data) == 1:
        axes = axes.reshape(1, -1)

    for i, split in enumerate(split_data.keys()):
        data = split_data[split]
        seq_len = [len(seq["time_since_last_event"]) for seq in data]
        time_since_last_event = [time for seq in data for time in seq["time_since_last_event"]]

        # Histogram of the seq_len
        axes[i, 0].hist(seq_len, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
        axes[i, 0].set_title(f"Sequence Length Distribution ({split})", fontsize=14, fontweight="bold")
        axes[i, 0].set_xlabel("Sequence Length", fontsize=12)
        axes[i, 0].set_ylabel("Frequency", fontsize=12)
        axes[i, 0].grid(True, alpha=0.3)

        # Histogram of the time_since_last_event
        axes[i, 1].hist(time_since_last_event, bins=50, edgecolor="black", alpha=0.7, color="darkseagreen")
        axes[i, 1].set_title(f"Time Since Last Event Distribution ({split})", fontsize=14, fontweight="bold")
        axes[i, 1].set_xlabel("Time Since Last Event", fontsize=12)
        axes[i, 1].set_ylabel("Frequency", fontsize=12)
        axes[i, 1].set_yscale("log")
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def get_max_time(dataset, split="train"):
    """
    Calculates the maximum event time ('time_since_start') across all trajectories
    in a split of a Hugging Face Dataset object.
    """
    max_t = 0
    data_split = dataset.get(split)
    if data_split is None:
        print(f"Warning: Split '{split}' not found in dataset.")
        return 1.0  # Default max time

    # Check if 'time_since_start' column exists
    if "time_since_start" not in data_split.column_names:
        print(f"Error: 'time_since_start' column not found in dataset split '{split}'. Cannot determine max time.")
        return 1.0  # Default max time

    for trajectory in data_split:
        times = trajectory.get("time_since_start")
        # Ensure times is a non-empty list or array
        if times and hasattr(times, "__len__") and len(times) > 0:
            # Check for potential nested lists or other issues if max fails
            try:
                current_max = np.max(times)
                if isinstance(current_max, (int, float)):  # Ensure it's a number
                    max_t = max(max_t, current_max)
            except (TypeError, ValueError) as e:
                print(f"Warning: Could not compute max time for a trajectory: {e}. Skipping.")
                print(f"Problematic 'time_since_start' value: {times}")

    return max_t if max_t > 0 else 1.0  # Avoid max_time being 0


# --- Professional color palette (colorblind-friendly) ---
PROFESSIONAL_COLORS = [
    "#2E86AB",  # Ocean Blue
    "#A23B72",  # Burgundy
    "#F18F01",  # Amber
    "#C73E1D",  # Red
    "#6A994E",  # Forest Green
    "#7209B7",  # Purple
    "#FFB500",  # Golden Yellow
    "#F72585",  # Hot Pink
    "#4CC9F0",  # Sky Blue
    "#8ECAE6",  # Light Blue
    "#FFB3BA",  # Light Pink
    "#BAFFC9",  # Light Green
    "#BAE1FF",  # Pale Blue
    "#FFFFBA",  # Light Yellow
    "#E6E6FA",  # Lavender
]


def get_professional_palette(n_colors):
    """Get a professional, colorblind-friendly palette with n_colors."""
    if n_colors <= len(PROFESSIONAL_COLORS):
        return PROFESSIONAL_COLORS[:n_colors]
    else:
        # Repeat colors if we need more than available
        return (PROFESSIONAL_COLORS * ((n_colors // len(PROFESSIONAL_COLORS)) + 1))[:n_colors]


# --- Visualization Functions ---


def plot_aggregate_rate(dataset, split="train", n_bins=50, max_time=None, ax=None):
    """
    Plots the aggregate event rate over time using a Hugging Face Dataset.
    Assumes 'time_since_start' contains absolute event times.

    Args:
        dataset (dict): Dictionary containing Hugging Face Dataset objects per split.
        split (str): The dataset split to use (e.g., 'train').
        n_bins (int): Number of time bins.
        max_time (float, optional): Maximum time for the plot range. If None, calculated from data.
        ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. If None, creates a new figure.
    """
    if ax is None:
        fig_created = True
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig_created = False

    data_split = dataset.get(split)
    if data_split is None:
        print(f"No data found for split '{split}'.")
        return

    num_trajectories = len(data_split)
    if num_trajectories == 0:
        print(f"No trajectories in split '{split}'.")
        return

    if "time_since_start" not in data_split.column_names:
        print(f"Error: 'time_since_start' column not found in dataset split '{split}'. Cannot plot aggregate rate.")
        return

    if max_time is None:
        max_time = get_max_time(dataset, split)

    # Define time bins
    bins = np.linspace(0, max_time, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    if bin_width == 0:  # Avoid division by zero if max_time is 0 or n_bins is 1
        print("Warning: Cannot calculate rate, time bin width is zero.")
        return

    # Count events in each bin
    counts = np.zeros(n_bins)

    for trajectory in data_split:
        times = trajectory.get("time_since_start", [])
        # Ensure times is a list/array of numbers
        if times and isinstance(times, (list, np.ndarray)):
            try:
                # Filter out non-numeric types if necessary, though HF datasets usually handle this
                numeric_times = [t for t in times if isinstance(t, (int, float))]
                if numeric_times:
                    hist, _ = np.histogram(numeric_times, bins=bins)
                    counts += hist
            except Exception as e:
                print(f"Warning: Could not process times for a trajectory: {e}")
                print(f"Problematic 'time_since_start': {times}")

    # Calculate rate (average events per trajectory per unit time)
    rate = counts / (num_trajectories * bin_width)

    # Plotting with professional styling
    ax.plot(
        bin_centers,
        rate,
        marker="o",
        linestyle="-",
        color=PROFESSIONAL_COLORS[0],
        linewidth=2,
        markersize=4,
        markerfacecolor="white",
        markeredgewidth=1.5,
        markeredgecolor=PROFESSIONAL_COLORS[0],
    )
    ax.set_title(f"Aggregate Event Rate ({split} split)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (from time_since_start)", fontsize=10)
    ax.set_ylabel("Avg. Events / Trajectory / Time Unit", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(0, max_time)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if fig_created:  # Only show if we created the figure
        plt.tight_layout()
        plt.show()


def plot_rate_per_mark(dataset, split="train", n_bins=50, max_time=None, ax=None):
    """
    Plots the event rate per mark over time using a Hugging Face Dataset.
    Assumes 'time_since_start' contains absolute event times and 'type_event' contains marks.

    Args:
        dataset (dict): Dictionary containing Hugging Face Dataset objects per split.
        split (str): The dataset split to use.
        n_bins (int): Number of time bins.
        max_time (float, optional): Maximum time for the plot range. If None, calculated from data.
        ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. If None, creates a new figure.
    """
    if ax is None:
        fig_created = True
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig_created = False

    data_split = dataset.get(split)
    if data_split is None:
        print(f"No data found for split '{split}'.")
        return

    num_trajectories = len(data_split)
    if num_trajectories == 0:
        print(f"No trajectories in split '{split}'.")
        return

    # Check for required columns
    required_cols = ["dim_process", "time_since_start", "type_event"]
    if not all(col in data_split.column_names for col in required_cols):
        print(f"Error: Dataset split '{split}' missing one or more required columns: {required_cols}")
        return

    # Infer number of marks (dim_process) - Assuming it's constant
    # Take it from the first row, ensure it's an integer
    try:
        dim_process = int(data_split[0]["dim_process"])
    except (ValueError, TypeError):
        print("Error: 'dim_process' feature is not a valid integer in the first row.")
        return

    if max_time is None:
        max_time = get_max_time(dataset, split)

    # Define time bins
    bins = np.linspace(0, max_time, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    if bin_width == 0:
        print("Warning: Cannot calculate rate, time bin width is zero.")
        return

    # Count events per mark in each bin
    counts_per_mark = np.zeros((dim_process, n_bins))

    for trajectory in data_split:
        times = np.array(trajectory.get("time_since_start", []))
        types = np.array(trajectory.get("type_event", []))

        if len(times) != len(types) or len(times) == 0:
            continue  # Skip if data is inconsistent or empty

        # Ensure types are integers
        try:
            types = types.astype(int)
        except (ValueError, TypeError):
            print("Warning: Could not convert 'type_event' to integers for a trajectory. Skipping.")
            continue

        for mark in range(dim_process):
            try:
                # Filter times corresponding to the current mark
                mark_times = times[types == mark]
                # Ensure mark_times contains numeric data before histogramming
                numeric_mark_times = [t for t in mark_times if isinstance(t, (int, float))]
                if numeric_mark_times:
                    hist, _ = np.histogram(numeric_mark_times, bins=bins)
                    counts_per_mark[mark, :] += hist
            except IndexError:
                print(f"Warning: Index error processing mark {mark}. Check data consistency (e.g., mark values vs dim_process).")
            except Exception as e:
                print(f"Warning: Could not process mark {mark} for a trajectory: {e}")

    # Calculate rates
    rates_per_mark = counts_per_mark / (num_trajectories * bin_width)

    # Plotting with professional styling
    colors = get_professional_palette(dim_process)
    for mark in range(dim_process):
        ax.plot(
            bin_centers,
            rates_per_mark[mark, :],
            marker="o",
            linestyle="-",
            label=f"Mark {mark}",
            color=colors[mark],
            linewidth=2,
            markersize=3,
            markerfacecolor="white",
            markeredgewidth=1,
            markeredgecolor=colors[mark],
        )

    ax.set_title(f"Event Rate per Mark ({split} split)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (from time_since_start)", fontsize=10)
    ax.set_ylabel("Avg. Events / Trajectory / Time Unit", fontsize=10)
    ax.legend(fontsize=8, frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(0, max_time)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if fig_created:  # Only show if we created the figure
        plt.tight_layout()
        plt.show()


def plot_raster_sample(dataset, split="train", n_samples=8, max_time=None, ax=None, seed=None):
    """
    Creates a raster plot for a sample of trajectories from a Hugging Face Dataset.
    Assumes 'time_since_start' and 'type_event' columns exist.

    Args:
        dataset (dict): Dictionary containing Hugging Face Dataset objects per split.
        split (str): The dataset split to use.
        n_samples (int): Number of trajectories to sample and plot.
        max_time (float, optional): Maximum time for the plot range. If None, calculated from data.
        ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. If None, creates a new figure.
        seed (int, optional): Random seed for reproducibility of sampling.
    """
    if ax is None:
        fig_created = True
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig_created = False

    data_split = dataset.get(split)
    if data_split is None:
        print(f"No data found for split '{split}'.")
        return

    num_trajectories = len(data_split)
    if num_trajectories == 0:
        print(f"No trajectories in split '{split}'.")
        return

    # Check for required columns
    required_cols = ["dim_process", "time_since_start", "type_event"]
    if not all(col in data_split.column_names for col in required_cols):
        print(f"Error: Dataset split '{split}' missing one or more required columns: {required_cols}")
        return

    if num_trajectories < n_samples:
        print(f"Warning: Requested {n_samples} samples, but only {num_trajectories} trajectories available.")
        n_samples = num_trajectories

    if n_samples == 0:
        return

    # Sample indices
    if seed is not None:
        random.seed(seed)
    sampled_indices = random.sample(range(num_trajectories), n_samples)

    # Select the sampled trajectories (more efficient than iterating and filtering)
    # Note: This creates a new dataset view, doesn't load all into memory at once
    sampled_data = data_split.select(sampled_indices)

    # Infer number of marks from the first sampled trajectory
    try:
        dim_process = int(sampled_data[0]["dim_process"])
    except (ValueError, TypeError):
        print("Error: 'dim_process' feature is not a valid integer in the first sampled row.")
        return

    if max_time is None:
        # Calculate max_time only from the sampled trajectories for efficiency
        max_t_sample = 0
        for traj in sampled_data:
            times = traj.get("time_since_start")
            if times and hasattr(times, "__len__") and len(times) > 0:
                try:
                    current_max = np.max(times)
                    if isinstance(current_max, (int, float)):
                        max_t_sample = max(max_t_sample, current_max)
                except (TypeError, ValueError):
                    pass  # Ignore errors here, focus is on getting a reasonable max
        max_time = max_t_sample if max_t_sample > 0 else 1.0

    colors = get_professional_palette(dim_process)

    # Plot events using scatter with enhanced styling
    event_times = []
    y_positions = []
    event_colors = []
    for i, trajectory in enumerate(sampled_data):  # Iterate over the selected sample
        times = trajectory.get("time_since_start", [])
        types = trajectory.get("type_event", [])

        if (
            not isinstance(times, (list, np.ndarray))
            or not isinstance(types, (list, np.ndarray))
            or len(times) != len(types)
            or len(times) == 0
        ):
            continue

        try:
            # Ensure types are integers for indexing colors
            types = np.array(types).astype(int)
            times = np.array(times)  # Ensure times are numpy array for potential filtering

            # Filter out non-numeric times just in case
            valid_indices = [idx for idx, t in enumerate(times) if isinstance(t, (int, float))]
            times = times[valid_indices]
            types = types[valid_indices]

            for t, type_ in zip(times, types):
                if 0 <= type_ < dim_process:  # Check if type is valid
                    event_times.append(t)
                    y_positions.append(i)  # Use sample index for y-position
                    event_colors.append(colors[type_])
                else:
                    print(f"Warning: Invalid event type {type_} encountered in sampled trajectory {i}. Skipping event.")

        except Exception as e:
            print(f"Warning: Error processing sampled trajectory {i}: {e}")

    # Using scatter for better color control per event
    if event_times:  # Only plot if there's data
        ax.scatter(event_times, y_positions, c=event_colors, marker="|", s=80, linewidths=2)
    else:
        print("No valid events found in the sample to plot.")

    ax.set_title(f"Raster Plot ({n_samples} Sampled Trajectories from {split} split)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (from time_since_start)", fontsize=10)
    ax.set_ylabel("Sampled Trajectory Index", fontsize=10)
    ax.set_yticks(range(n_samples))
    ax.set_ylim(-0.5, n_samples - 0.5)
    ax.set_xlim(0, max_time)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add a legend for marks with professional colors
    legend_elements = [
        plt.Line2D([0], [0], marker="|", color=colors[mark], linestyle="None", markersize=10, label=f"Mark {mark}", markeredgewidth=2)
        for mark in range(dim_process)
    ]
    ax.legend(
        handles=legend_elements,
        title="Event Marks",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
        title_fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    if fig_created:  # Only show if we created the figure
        plt.tight_layout()
        plt.show()


def plot_event_count_heatmap(dataset, split="train", n_bins=30, max_time=None, normalize=True, ax=None):
    """
    Plots a heatmap of event counts per mark over time using a Hugging Face Dataset.
    Assumes 'time_since_start', 'type_event', 'dim_process' columns exist.

    Args:
        dataset (dict): Dictionary containing Hugging Face Dataset objects per split.
        split (str): The dataset split to use.
        n_bins (int): Number of time bins.
        max_time (float, optional): Maximum time for the plot range. If None, calculated from data.
        normalize (bool): If True, normalize counts by the number of trajectories.
        ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. If None, creates a new figure.
    """
    if ax is None:
        fig_created = True
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig_created = False

    data_split = dataset.get(split)
    if data_split is None:
        print(f"No data found for split '{split}'.")
        return

    num_trajectories = len(data_split)
    if num_trajectories == 0:
        print(f"No trajectories in split '{split}'.")
        return

    # Check for required columns
    required_cols = ["dim_process", "time_since_start", "type_event"]
    if not all(col in data_split.column_names for col in required_cols):
        print(f"Error: Dataset split '{split}' missing one or more required columns: {required_cols}")
        return

    # Infer number of marks
    try:
        dim_process = int(data_split[0]["dim_process"])
    except (ValueError, TypeError):
        print("Error: 'dim_process' feature is not a valid integer in the first row.")
        return

    if max_time is None:
        max_time = get_max_time(dataset, split)

    # Define time bins
    bins = np.linspace(0, max_time, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # For labeling ticks

    # Initialize count matrix (marks x time_bins)
    heatmap_counts = np.zeros((dim_process, n_bins))

    # Populate the heatmap counts
    for trajectory in data_split:
        times = np.array(trajectory.get("time_since_start", []))
        types = np.array(trajectory.get("type_event", []))

        if len(times) != len(types) or len(times) == 0:
            continue

        try:
            types = types.astype(int)
            # Filter out non-numeric times
            valid_indices = [idx for idx, t in enumerate(times) if isinstance(t, (int, float))]
            times = times[valid_indices]
            types = types[valid_indices]

            for mark in range(dim_process):
                mark_times = times[types == mark]
                if len(mark_times) > 0:  # Only histogram if there are times for this mark
                    hist, _ = np.histogram(mark_times, bins=bins)
                    heatmap_counts[mark, :] += hist

        except Exception as e:
            print(f"Warning: Could not process trajectory for heatmap: {e}")

    # Normalize counts (optional)
    if normalize and num_trajectories > 0:
        heatmap_data = heatmap_counts / num_trajectories
        cbar_label = "Avg Events / Trajectory / Bin"
    else:
        heatmap_data = heatmap_counts
        cbar_label = "Total Events / Bin"

    # Plotting the heatmap with professional styling
    cmap = sns.color_palette("viridis", as_cmap=True)
    heatmap = sns.heatmap(heatmap_data, ax=ax, cmap=cmap, cbar_kws={"label": cbar_label, "shrink": 0.8}, linewidths=0.5, linecolor="white")

    # Set x-axis ticks and labels (show fewer labels for clarity)
    tick_positions = np.linspace(0, n_bins - 1, num=min(n_bins, 8), dtype=int)  # Show ~8 ticks
    ax.set_xticks(tick_positions + 0.5)  # Center ticks
    ax.set_xticklabels([f"{bin_centers[i]:.1f}" for i in tick_positions], rotation=45, ha="right")

    ax.set_yticks(np.arange(dim_process) + 0.5)
    ax.set_yticklabels([f"Mark {i}" for i in range(dim_process)], rotation=0)

    ax.set_xlabel("Time Bins (from time_since_start)", fontsize=10)
    ax.set_ylabel("Event Mark", fontsize=10)
    title = f"Event Count Heatmap ({split} split)"
    if normalize:
        title = f"Average {title}"
    ax.set_title(title, fontsize=12, fontweight="bold")

    if fig_created:  # Only show if we created the figure
        plt.tight_layout()
        plt.show()


def create_appendix_plots(dataset, split="train", save_path=None):
    """
    Creates a comprehensive set of plots for the appendix optimized for A4 page.

    Args:
        dataset (dict): Dictionary containing Hugging Face Dataset objects per split.
        split (str): The dataset split to use.
        save_path (str, optional): Path to save the plots. If None, plots are displayed.
    """
    # Ensure figures directory exists
    import os

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Set style for publication-quality plots (fonts already configured globally)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 12,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "grid.alpha": 0.4,
        }
    )

    data_split = dataset.get(split)
    if data_split is None:
        print(f"No data found for split '{split}'.")
        return

    # Get dataset dimensions
    try:
        dim_process = int(data_split[0]["dim_process"])
        seq_lengths = [len(seq["time_since_last_event"]) for seq in data_split]
        num_sequences = len(data_split)
    except Exception as e:
        print(f"Error getting dataset info: {e}")
        return

    print(f"Creating appendix plots for dataset with {dim_process}D processes, {num_sequences} sequences")

    # Create figure optimized for single column layout
    fig = plt.figure(figsize=(8.3, 20))

    # 1. Aggregate Rate (top)
    ax1 = plt.subplot(6, 1, 1)
    plot_aggregate_rate(dataset, split=split, ax=ax1)

    # 2. Rate per Mark
    ax2 = plt.subplot(6, 1, 2)
    plot_rate_per_mark(dataset, split=split, ax=ax2)

    # 3. Raster Plot
    ax3 = plt.subplot(6, 1, 3)
    plot_raster_sample(dataset, split=split, n_samples=8, ax=ax3, seed=42)

    # 4. Event Count Heatmap
    ax4 = plt.subplot(6, 1, 4)
    plot_event_count_heatmap(dataset, split=split, ax=ax4)

    # 5. Sequence Length Distribution
    ax5 = plt.subplot(6, 1, 5)
    ax5.hist(seq_lengths, bins=20, edgecolor="black", alpha=0.8, color=PROFESSIONAL_COLORS[1])
    ax5.set_title("Sequence Length Distribution", fontsize=10, fontweight="bold")
    ax5.set_xlabel("Sequence Length", fontsize=9)
    ax5.set_ylabel("Frequency", fontsize=9)
    ax5.grid(True, linestyle="--", alpha=0.4)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    # 6. Dataset Info and Event Type Distribution (bottom)
    ax6 = plt.subplot(6, 1, 6)

    # Event Type Distribution as bar chart
    event_counts = [0] * dim_process
    for seq in data_split:
        types = seq.get("type_event", [])
        for t in types:
            if isinstance(t, int) and 0 <= t < dim_process:
                event_counts[t] += 1

    colors = get_professional_palette(dim_process)
    bars = ax6.bar(range(dim_process), event_counts, color=colors, edgecolor="black", alpha=0.8, linewidth=1)
    ax6.set_title("Event Type Distribution", fontsize=10, fontweight="bold")
    ax6.set_xlabel("Event Type", fontsize=9)
    ax6.set_ylabel("Total Count", fontsize=9)
    ax6.set_xticks(range(dim_process))
    ax6.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    plt.suptitle(f"Synthetic Hawkes Process Dataset Analysis ({dim_process}D, {split} split)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    if save_path:
        plt.savefig(f"{save_path}_appendix_{dim_process}D_{split}.png", dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.savefig(f"{save_path}_appendix_{dim_process}D_{split}.pdf", bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"Plots saved to {save_path}_appendix_{dim_process}D_{split}.[png/pdf]")
    else:
        plt.show()


# %%
def calculate_comprehensive_stats(dataset, split="train"):
    """
    Calculate comprehensive statistics for a dataset split.
    Returns a dictionary with all relevant statistics.
    """
    data_split = dataset[split]

    # Basic counts
    num_sequences = len(data_split)

    # Sequence lengths
    seq_lengths = [len(seq["time_since_last_event"]) for seq in data_split]

    # Event times and deltas
    times_since_last_event = [time for seq in data_split for time in seq["time_since_last_event"]]
    times_since_start = [time for seq in data_split for time in seq["time_since_start"]]

    # Event types
    all_event_types = [event_type for seq in data_split for event_type in seq["type_event"]]
    unique_event_types = set(all_event_types)

    # Get dimension from first sequence
    dim_process = data_split[0]["dim_process"] if data_split else 0

    # Calculate statistics
    stats = {
        "dimension": dim_process,
        "num_sequences": num_sequences,
        "total_events": len(times_since_last_event),
        "num_event_types": len(unique_event_types),
        # Sequence length statistics
        "min_seq_length": min(seq_lengths) if seq_lengths else 0,
        "max_seq_length": max(seq_lengths) if seq_lengths else 0,
        "mean_seq_length": sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0,
        "std_seq_length": np.std(seq_lengths) if seq_lengths else 0,
        # Event time statistics
        "min_delta_time": min(times_since_last_event) if times_since_last_event else 0,
        "max_delta_time": max(times_since_last_event) if times_since_last_event else 0,
        "mean_delta_time": sum(times_since_last_event) / len(times_since_last_event) if times_since_last_event else 0,
        "std_delta_time": np.std(times_since_last_event) if times_since_last_event else 0,
        # Absolute time statistics
        "min_abs_time": min(times_since_start) if times_since_start else 0,
        "max_abs_time": max(times_since_start) if times_since_start else 0,
        "mean_abs_time": sum(times_since_start) / len(times_since_start) if times_since_start else 0,
        "std_abs_time": np.std(times_since_start) if times_since_start else 0,
        # Event type distribution
        "event_type_counts": {str(i): all_event_types.count(i) for i in range(dim_process)},
    }

    return stats


# %%
# Updated heatmap function with continuous styling
def plot_event_count_heatmap_continuous(dataset, split="train", n_bins=30, max_time=None, normalize=True, ax=None):
    """
    Plots a continuous heatmap of event counts per mark over time using a Hugging Face Dataset.
    Assumes 'time_since_start', 'type_event', 'dim_process' columns exist.

    Args:
        dataset (dict): Dictionary containing Hugging Face Dataset objects per split.
        split (str): The dataset split to use.
        n_bins (int): Number of time bins.
        max_time (float, optional): Maximum time for the plot range. If None, calculated from data.
        normalize (bool): If True, normalize counts by the number of trajectories.
        ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. If None, creates a new figure.
    """
    if ax is None:
        fig_created = True
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig_created = False

    data_split = dataset.get(split)
    if data_split is None:
        print(f"No data found for split '{split}'.")
        return

    num_trajectories = len(data_split)
    if num_trajectories == 0:
        print(f"No trajectories in split '{split}'.")
        return

    # Check for required columns
    required_cols = ["dim_process", "time_since_start", "type_event"]
    if not all(col in data_split.column_names for col in required_cols):
        print(f"Error: Dataset split '{split}' missing one or more required columns: {required_cols}")
        return

    # Infer number of marks
    try:
        dim_process = int(data_split[0]["dim_process"])
    except (ValueError, TypeError):
        print("Error: 'dim_process' feature is not a valid integer in the first row.")
        return

    if max_time is None:
        max_time = get_max_time(dataset, split)

    # Define time bins
    bins = np.linspace(0, max_time, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # For labeling ticks

    # Initialize count matrix (marks x time_bins)
    heatmap_counts = np.zeros((dim_process, n_bins))

    # Populate the heatmap counts
    for trajectory in data_split:
        times = np.array(trajectory.get("time_since_start", []))
        types = np.array(trajectory.get("type_event", []))

        if len(times) != len(types) or len(times) == 0:
            continue

        try:
            types = types.astype(int)
            # Filter out non-numeric times
            valid_indices = [idx for idx, t in enumerate(times) if isinstance(t, (int, float))]
            times = times[valid_indices]
            types = types[valid_indices]

            for mark in range(dim_process):
                mark_times = times[types == mark]
                if len(mark_times) > 0:  # Only histogram if there are times for this mark
                    hist, _ = np.histogram(mark_times, bins=bins)
                    heatmap_counts[mark, :] += hist

        except Exception as e:
            print(f"Warning: Could not process trajectory for heatmap: {e}")

    # Normalize counts (optional)
    if normalize and num_trajectories > 0:
        heatmap_data = heatmap_counts / num_trajectories
        cbar_label = "Avg Events / Trajectory / Bin"
    else:
        heatmap_data = heatmap_counts
        cbar_label = "Total Events / Bin"

    # Plotting the heatmap with continuous styling
    cmap = sns.color_palette("viridis", as_cmap=True)
    heatmap = sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=cmap,
        cbar_kws={"label": cbar_label, "shrink": 0.8},
        linewidths=0,  # Remove grid lines for continuous appearance
        square=False,  # Allow rectangular cells
        cbar=True,  # Ensure colorbar is shown
        vmin=0,  # Set minimum value for color scale
        vmax=None,
    )  # Let it auto-scale maximum

    # Set x-axis ticks and labels (show fewer labels for clarity)
    tick_positions = np.linspace(0, n_bins - 1, num=min(n_bins, 8), dtype=int)  # Show ~8 ticks
    ax.set_xticks(tick_positions + 0.5)  # Center ticks
    ax.set_xticklabels([f"{bin_centers[i]:.1f}" for i in tick_positions], rotation=45, ha="right")

    ax.set_yticks(np.arange(dim_process) + 0.5)
    ax.set_yticklabels([f"Mark {i}" for i in range(dim_process)], rotation=0)

    ax.set_xlabel("Time Bins (from time_since_start)", fontsize=10)
    ax.set_ylabel("Event Mark", fontsize=10)
    title = f"Event Count Heatmap ({split} split)"
    if normalize:
        title = f"Average {title}"
    ax.set_title(title, fontsize=12, fontweight="bold")

    if fig_created:  # Only show if we created the figure
        plt.tight_layout()
        plt.show()


# %%
# Create comprehensive analysis for all dimensions
import os

import pandas as pd


# Create output directories
os.makedirs("figures/hawkes_dataset", exist_ok=True)
os.makedirs("tables", exist_ok=True)

# Initialize results storage
all_stats = {}
all_plots_created = {}

print("Starting comprehensive analysis for all dimensions...")
print("=" * 60)

for dim_name, config in configs_by_dimension.items():
    print(f"\nProcessing {dim_name} dataset...")

    try:
        # Create dataloader for this dimension
        dataloader = DataLoaderFactory.create(**config)
        data_point = next(iter(dataloader.train_it))

        # Process data (same as before but for each dimension)
        num_samples = 5_000
        i = 0
        data = []

        for sample in dataloader.train_it:
            # if i >= num_samples:
            #     break
            # i += 1

            event_times = sample["context_event_times"]
            seq_lengths = sample["context_seq_lengths"]
            process_dim = sample["base_intensity_functions"].shape[1]

            # Normalize the event times
            normalized_event_times = normalize_obs_grid(event_times, seq_lengths)

            # Calculate delta times
            delta_times = normalized_event_times[:, :, 1:] - normalized_event_times[:, :, :-1]
            delta_times = torch.cat([torch.zeros_like(delta_times[:, :, :1]), delta_times], dim=2)

            # Create processed sample
            processed_sample = {
                "event_times": normalized_event_times,
                "delta_times": delta_times,
                "event_types": sample["context_event_types"],
                "seq_lengths": seq_lengths,
                "dim_process": process_dim,
            }
            data.append(processed_sample)

        # Convert to EasyTPP format
        converted_datasets = []
        for sample in data:
            converted_datasets.extend(convert_to_easytpp_format(sample))
        dataset = Dataset.from_list(converted_datasets)
        dataset = DatasetDict({"train": dataset})

        # Calculate statistics
        print(f"Calculating statistics for {dim_name}...")
        stats = calculate_comprehensive_stats(dataset, split="train")
        all_stats[dim_name] = stats

        # Create plots
        print(f"Creating plots for {dim_name}...")
        plot_path = f"figures/hawkes_dataset/{dim_name}"
        create_appendix_plots(dataset, split="train", save_path=plot_path)
        all_plots_created[dim_name] = plot_path

        print(f"✓ {dim_name} completed successfully")

    except Exception as e:
        print(f"✗ Error processing {dim_name}: {str(e)}")
        continue

print("\n" + "=" * 60)
print("Analysis completed!")
print(f"Processed {len(all_stats)} dimensions")
print(f"Created plots for {len(all_plots_created)} dimensions")


# %%
# Override the original heatmap function to use the continuous version
plot_event_count_heatmap = plot_event_count_heatmap_continuous

print("✅ Heatmap function updated to use continuous styling")


# %%
def create_latex_table(all_stats, save_path="tables/hawkes_dataset_stats.tex"):
    """
    Create a comprehensive LaTeX table with statistics for all dimensions.
    """
    # Prepare data for DataFrame
    table_data = []

    for dim_name, stats in all_stats.items():
        row = {
            "Dimension": dim_name,
            "Process Dim": stats["dimension"],
            "Sequences": stats["num_sequences"],
            "Total Events": stats["total_events"],
            "Event Types": stats["num_event_types"],
            "Seq Length (min)": f"{stats['min_seq_length']:.0f}",
            "Seq Length (max)": f"{stats['max_seq_length']:.0f}",
            "Seq Length (mean)": f"{stats['mean_seq_length']:.1f}",
            "Seq Length (std)": f"{stats['std_seq_length']:.1f}",
            "Delta Time (min)": f"{stats['min_delta_time']:.3f}",
            "Delta Time (max)": f"{stats['max_delta_time']:.3f}",
            "Delta Time (mean)": f"{stats['mean_delta_time']:.3f}",
            "Delta Time (std)": f"{stats['std_delta_time']:.3f}",
            "Abs Time (min)": f"{stats['min_abs_time']:.3f}",
            "Abs Time (max)": f"{stats['max_abs_time']:.3f}",
            "Abs Time (mean)": f"{stats['mean_abs_time']:.3f}",
            "Abs Time (std)": f"{stats['std_abs_time']:.3f}",
        }
        table_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Generate LaTeX table
    latex_table = df.to_latex(
        index=False,
        escape=False,
        column_format="l" + "c" * (len(df.columns) - 1),
        caption="Comprehensive Statistics for Synthetic Hawkes Process Datasets",
        label="tab:hawkes_dataset_stats",
        position="htbp",
    )

    # Add custom formatting
    latex_table = latex_table.replace("\\begin{table}[htbp]", "\\begin{table}[htbp]\n\\centering")
    latex_table = latex_table.replace(
        "\\toprule",
        "\\toprule\n\\multicolumn{1}{c}{\\textbf{Dimension}} & \\multicolumn{1}{c}{\\textbf{Process Dim}} & \\multicolumn{1}{c}{\\textbf{Sequences}} & \\multicolumn{1}{c}{\\textbf{Total Events}} & \\multicolumn{1}{c}{\\textbf{Event Types}} & \\multicolumn{4}{c}{\\textbf{Sequence Length}} & \\multicolumn{4}{c}{\\textbf{Delta Time}} & \\multicolumn{4}{c}{\\textbf{Absolute Time}} \\\\\n\\cmidrule(lr){6-9} \\cmidrule(lr){10-13} \\cmidrule(lr){14-17}\n& & & & & \\textbf{Min} & \\textbf{Max} & \\textbf{Mean} & \\textbf{Std} & \\textbf{Min} & \\textbf{Max} & \\textbf{Mean} & \\textbf{Std} & \\textbf{Min} & \\textbf{Max} & \\textbf{Mean} & \\textbf{Std} \\\\",
    )

    # Save to file
    with open(save_path, "w") as f:
        f.write(latex_table)

    print(f"LaTeX table saved to: {save_path}")
    return latex_table


# Create the LaTeX table
if all_stats:
    latex_table = create_latex_table(all_stats)
    print("\nLaTeX Table Preview:")
    print("=" * 80)
    print(latex_table)
else:
    print("No statistics available to create table.")


# %%
# Summary of created outputs
print("SUMMARY OF GENERATED OUTPUTS")
print("=" * 50)

print("\n📊 PLOTS CREATED:")
for dim_name, plot_path in all_plots_created.items():
    print(f"  {dim_name}: {plot_path}_appendix_{dim_name}_train.png")
    print(f"  {dim_name}: {plot_path}_appendix_{dim_name}_train.pdf")

print("\n📈 STATISTICS CALCULATED:")
for dim_name, stats in all_stats.items():
    print(f"  {dim_name}: {stats['num_sequences']} sequences, {stats['total_events']} total events")

print("\n📋 LATEX TABLE:")
print("  Saved to: tables/hawkes_dataset_stats.tex")

print("\n📁 OUTPUT DIRECTORIES:")
print("  figures/hawkes_dataset/ - Contains all plot files")
print("  tables/ - Contains LaTeX table file")

print("\n✅ All outputs successfully generated!")


# %%


# %%
