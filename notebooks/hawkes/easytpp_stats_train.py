# %%
# Configure matplotlib to avoid Type 1 fonts - MUST BE RUN FIRST
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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
from pprint import pprint

from datasets import load_dataset


# %%
# Configuration for EasyTPP datasets
easytpp_datasets = {
    "volcano": {
        "name": "volcano",
        "description": "Volcanic eruption events",
    },
    "taobao": {
        "name": "taobao",
        "description": "Taobao user behavior events",
    },
    "retweet": {
        "name": "retweet",
        "description": "Retweet cascade events",
    },
    "stackoverflow": {
        "name": "stackoverflow",
        "description": "StackOverflow question-answer events",
    },
    "taxi": {
        "name": "taxi",
        "description": "Taxi pickup events",
    },
    "amazon": {
        "name": "amazon",
        "description": "Amazon product review events",
    },
    "earthquake": {
        "name": "earthquake",
        "description": "Earthquake events",
    },
}


# %%
def normalize_times_to_1(dataset):
    """Normalize all times in the dataset to [0, 1] range."""
    normalized_dataset = {}

    # Process each split separately
    for split in dataset.keys():
        # Find max time for this specific split
        all_times = []
        for example in dataset[split]:
            all_times.extend(example["time_since_start"])

        max_time = max(all_times) if all_times else 1.0

        # Define normalization function for this split
        def normalize_example(example):
            return {
                "seq_len": example["seq_len"],
                "type_event": example["type_event"],
                "seq_idx": example["seq_idx"],
                "time_since_start": [time / max_time for time in example["time_since_start"]],
                "time_since_last_event": [time / max_time for time in example["time_since_last_event"]],
                "dim_process": example["dim_process"],
            }

        # Apply normalization while preserving Dataset type
        normalized_dataset[split] = dataset[split].map(normalize_example)

    return normalized_dataset


# %%
def print_stats(dataset, split="test"):
    """Print basic statistics for a dataset split."""
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
    min_delta_event_time = min(times_since_last_event) if times_since_last_event else 0
    max_delta_event_time = max(times_since_last_event) if times_since_last_event else 0
    min_seq_length = min(seq_lengths) if seq_lengths else 0
    max_seq_length = max(seq_lengths) if seq_lengths else 0
    avg_seq_length = sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0
    avg_delta_event_time = sum(times_since_last_event) / len(times_since_last_event) if times_since_last_event else 0
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
    return stats


# %%
import random

import seaborn as sns


# Configure seaborn to work with our font settings
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=0.8)

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


# %%
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


# %%
def plot_aggregate_rate(dataset, split="train", n_bins=50, max_time=None, ax=None):
    """
    Plots the aggregate event rate over time using a Hugging Face Dataset.
    Assumes 'time_since_start' contains absolute event times.
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


# %%
def plot_rate_per_mark(dataset, split="train", n_bins=50, max_time=None, ax=None):
    """
    Plots the event rate per mark over time using a Hugging Face Dataset.
    Assumes 'time_since_start' contains absolute event times and 'type_event' contains marks.
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


# %%
def plot_raster_sample(dataset, split="train", n_samples=8, max_time=None, ax=None, seed=None):
    """
    Creates a raster plot for a sample of trajectories from a Hugging Face Dataset.
    Assumes 'time_since_start' and 'type_event' columns exist.
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


# %%
def plot_event_count_heatmap(dataset, split="train", n_bins=30, max_time=None, normalize=True, ax=None):
    """
    Plots a heatmap of event counts per mark over time using a Hugging Face Dataset.
    Assumes 'time_since_start', 'type_event', 'dim_process' columns exist.
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
def create_appendix_plots(dataset, dataset_name, split="train", save_path=None):
    """
    Creates a comprehensive set of plots for the appendix optimized for A4 page.

    Args:
        dataset (dict): Dictionary containing Hugging Face Dataset objects per split.
        dataset_name (str): Name of the dataset for title.
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

    print(f"Creating appendix plots for {dataset_name} dataset with {dim_process}D processes, {num_sequences} sequences")

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

    plt.suptitle(
        f"EasyTPP Dataset Analysis - {dataset_name.title()} ({dim_process}D, {split} split)", fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    if save_path:
        plt.savefig(f"{save_path}_appendix_{dataset_name}_{split}.png", dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.savefig(f"{save_path}_appendix_{dataset_name}_{split}.pdf", bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"Plots saved to {save_path}_appendix_{dataset_name}_{split}.[png/pdf]")
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
# Create comprehensive analysis for all EasyTPP datasets
import os

import pandas as pd


# Create output directories
os.makedirs("figures/easytpp_dataset", exist_ok=True)
os.makedirs("tables", exist_ok=True)

# Initialize results storage
all_stats = {}
all_plots_created = {}

print("Starting comprehensive analysis for all EasyTPP datasets...")
print("=" * 60)

for dataset_name, config in easytpp_datasets.items():
    print(f"\nProcessing {dataset_name} dataset...")

    try:
        # Load dataset
        print(f"Loading {dataset_name} dataset...")
        dataset = load_dataset(f"easytpp/{dataset_name}")

        # Normalize times to [0, 1] range
        print(f"Normalizing times for {dataset_name}...")
        dataset = normalize_times_to_1(dataset)

        # Calculate statistics
        print(f"Calculating statistics for {dataset_name}...")
        stats = calculate_comprehensive_stats(dataset, split="train")
        all_stats[dataset_name] = stats

        # Create plots
        print(f"Creating plots for {dataset_name}...")
        plot_path = f"figures/easytpp_dataset/{dataset_name}"
        create_appendix_plots(dataset, dataset_name, split="train", save_path=plot_path)
        all_plots_created[dataset_name] = plot_path

        print(f"✓ {dataset_name} completed successfully")

    except Exception as e:
        print(f"✗ Error processing {dataset_name}: {str(e)}")
        continue

print("\n" + "=" * 60)
print("Analysis completed!")
print(f"Processed {len(all_stats)} datasets")
print(f"Created plots for {len(all_plots_created)} datasets")


# %%
def create_latex_table(all_stats, save_path="tables/easytpp_dataset_stats.tex"):
    """
    Create a comprehensive LaTeX table with statistics for all EasyTPP datasets.
    """
    # Prepare data for DataFrame
    table_data = []

    for dataset_name, stats in all_stats.items():
        row = {
            "Dataset": dataset_name.title(),
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
        caption="Comprehensive Statistics for EasyTPP Datasets",
        label="tab:easytpp_dataset_stats",
        position="htbp",
    )

    # Add custom formatting
    latex_table = latex_table.replace("\\begin{table}[htbp]", "\\begin{table}[htbp]\n\\centering")
    latex_table = latex_table.replace(
        "\\toprule",
        "\\toprule\n\\multicolumn{1}{c}{\\textbf{Dataset}} & \\multicolumn{1}{c}{\\textbf{Process Dim}} & \\multicolumn{1}{c}{\\textbf{Sequences}} & \\multicolumn{1}{c}{\\textbf{Total Events}} & \\multicolumn{1}{c}{\\textbf{Event Types}} & \\multicolumn{4}{c}{\\textbf{Sequence Length}} & \\multicolumn{4}{c}{\\textbf{Delta Time}} & \\multicolumn{4}{c}{\\textbf{Absolute Time}} \\\\\n\\cmidrule(lr){6-9} \\cmidrule(lr){10-13} \\cmidrule(lr){14-17}\n& & & & & \\textbf{Min} & \\textbf{Max} & \\textbf{Mean} & \\textbf{Std} & \\textbf{Min} & \\textbf{Max} & \\textbf{Mean} & \\textbf{Std} & \\textbf{Min} & \\textbf{Max} & \\textbf{Mean} & \\textbf{Std} \\\\",
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
for dataset_name, plot_path in all_plots_created.items():
    print(f"  {dataset_name}: {plot_path}_appendix_{dataset_name}_train.png")
    print(f"  {dataset_name}: {plot_path}_appendix_{dataset_name}_train.pdf")

print("\n📈 STATISTICS CALCULATED:")
for dataset_name, stats in all_stats.items():
    print(f"  {dataset_name}: {stats['num_sequences']} sequences, {stats['total_events']} total events")

print("\n📋 LATEX TABLE:")
print("  Saved to: tables/easytpp_dataset_stats.tex")

print("\n📁 OUTPUT DIRECTORIES:")
print("  figures/easytpp_dataset/ - Contains all plot files")
print("  tables/ - Contains LaTeX table file")

print("\n✅ All outputs successfully generated!")

# %%
