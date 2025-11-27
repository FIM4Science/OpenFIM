#!/usr/bin/env python3
"""
Verify and visualize patterns in the generated synthetic data.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_data(data_dir: Path, split: str = "train"):
    """Load event times and types from the specified directory."""
    split_dir = data_dir / split

    event_times = torch.load(split_dir / "event_times.pt")
    event_types = torch.load(split_dir / "event_types.pt")

    return event_times, event_types


def verify_taxi_pattern(event_types: torch.Tensor):
    """
    Verify the taxi pattern (alternating marks).

    Returns statistics about alternation rates.
    """
    N, P, K, _ = event_types.shape

    alternation_rates = []

    # Sample 100 random trajectories for analysis
    num_samples = min(100, N * P)

    for _ in range(num_samples):
        n = np.random.randint(0, N)
        p = np.random.randint(0, P)

        marks = event_types[n, p, :, 0].numpy()
        alternations = sum(1 for i in range(1, len(marks)) if marks[i] != marks[i - 1])
        alternation_rate = alternations / (len(marks) - 1) if len(marks) > 1 else 0
        alternation_rates.append(alternation_rate)

    return {
        "mean_alternation_rate": np.mean(alternation_rates),
        "std_alternation_rate": np.std(alternation_rates),
        "min_alternation_rate": np.min(alternation_rates),
        "max_alternation_rate": np.max(alternation_rates),
    }


def verify_amazon_pattern(event_times: torch.Tensor):
    """
    Verify the amazon pattern (periodic timing).

    Returns statistics about periodicity.
    """
    N, P, K, _ = event_times.shape

    cvs = []  # Coefficient of variation
    mean_intervals = []

    # Sample 100 random trajectories for analysis
    num_samples = min(100, N * P)

    for _ in range(num_samples):
        n = np.random.randint(0, N)
        p = np.random.randint(0, P)

        times = event_times[n, p, :, 0].numpy()
        inter_event_times = np.diff(times)

        if len(inter_event_times) > 0:
            mean_interval = np.mean(inter_event_times)
            std_interval = np.std(inter_event_times)
            cv = std_interval / mean_interval if mean_interval > 0 else float("inf")

            cvs.append(cv)
            mean_intervals.append(mean_interval)

    return {
        "mean_cv": np.mean(cvs),
        "std_cv": np.std(cvs),
        "mean_interval": np.mean(mean_intervals),
        "std_interval_across_trajs": np.std(mean_intervals),
    }


def plot_detailed_raster(event_times: torch.Tensor, event_types: torch.Tensor, num_samples: int, pattern_name: str, output_path: Path):
    """Create a detailed raster plot."""
    N, P, K, _ = event_times.shape
    M = int(event_types.max()) + 1

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Sample trajectories from the same process to show within-sample consistency
    sample_process = 0
    sample_indices = np.random.choice(P, num_samples, replace=False)

    # Define colors
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Plot 1: Raster plot
    ax = axes[0]
    for idx, traj_idx in enumerate(sample_indices):
        times = event_times[sample_process, traj_idx, :, 0].numpy()
        types = event_types[sample_process, traj_idx, :, 0].numpy()

        for t, mark in zip(times, types):
            ax.plot([t, t], [idx, idx + 0.8], color=colors[int(mark) % 10], linewidth=2)

    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Trajectory Index", fontsize=11)
    ax.set_title(f"Raster Plot - {pattern_name.upper()} Pattern ({num_samples} Paths from Same Sample)", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.5, num_samples)
    ax.grid(True, alpha=0.3)

    # Add legend
    handles = [plt.Line2D([0], [0], color=colors[m % 10], linewidth=2, label=f"Mark {m}") for m in range(M)]
    ax.legend(handles=handles, loc="upper right", title="Event Marks", fontsize=9)

    # Plot 2: Single trajectory detail
    ax = axes[1]
    detail_traj = sample_indices[0]
    times = event_times[sample_process, detail_traj, :30, 0].numpy()  # Show first 30 events
    types = event_types[sample_process, detail_traj, :30, 0].numpy()

    # Plot events as vertical lines
    for i, (t, mark) in enumerate(zip(times, types)):
        ax.plot([t, t], [0, 1], color=colors[int(mark) % 10], linewidth=3, alpha=0.7)
        ax.text(t, 1.1, str(int(mark)), ha="center", va="bottom", fontsize=8)

    # Plot inter-event times
    if len(times) > 1:
        inter_times = np.diff(times)
        mid_times = times[:-1] + inter_times / 2
        for mt, it in zip(mid_times, inter_times):
            ax.text(mt, -0.2, f"{it:.3f}", ha="center", va="top", fontsize=7, color="gray")

    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("", fontsize=11)
    ax.set_title(f"Detailed View - First 30 Events of Path {detail_traj}", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.4, 1.3)
    ax.set_xlim(times[0] - 0.05, times[-1] + 0.05)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved detailed raster plot to: {output_path}")
    plt.close()


def plot_pattern_analysis(event_times: torch.Tensor, event_types: torch.Tensor, pattern_name: str, output_path: Path):
    """Create analysis plots specific to the pattern type."""
    N, P, K, _ = event_times.shape

    if pattern_name == "taxi":
        # Analyze alternation patterns
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Collect alternation statistics from many trajectories
        alternation_rates = []
        for _ in range(200):
            n = np.random.randint(0, N)
            p = np.random.randint(0, P)
            marks = event_types[n, p, :, 0].numpy()
            alternations = sum(1 for i in range(1, len(marks)) if marks[i] != marks[i - 1])
            alternation_rate = alternations / (len(marks) - 1)
            alternation_rates.append(alternation_rate)

        # Plot histogram
        ax = axes[0]
        ax.hist(alternation_rates, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(alternation_rates), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(alternation_rates):.3f}")
        ax.set_xlabel("Alternation Rate", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Distribution of Alternation Rates", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot example sequence showing alternation
        ax = axes[1]
        n, p = 0, 0
        marks = event_types[n, p, :50, 0].numpy()
        times = event_times[n, p, :50, 0].numpy()

        M = int(event_types.max()) + 1
        colors_map = plt.cm.tab10(np.linspace(0, 1, 10))
        for i in range(len(marks)):
            color = colors_map[int(marks[i]) % 10]
            ax.scatter(i, marks[i], c=[color], s=100, zorder=3)
            if i > 0:
                ax.plot([i - 1, i], [marks[i - 1], marks[i]], "k-", alpha=0.3, linewidth=1)

        ax.set_xlabel("Event Index", fontsize=11)
        ax.set_ylabel("Mark", fontsize=11)
        ax.set_title("Example Alternation Sequence (First 50 Events)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, M - 0.5)

    elif pattern_name == "amazon":
        # Analyze periodicity - check if paths within same sample have same frequency
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Collect CV statistics
        cvs = []
        all_inter_times = []
        for _ in range(100):
            n = np.random.randint(0, N)
            p = np.random.randint(0, P)
            times = event_times[n, p, :, 0].numpy()
            inter_times = np.diff(times)
            all_inter_times.extend(inter_times.tolist())

            mean_interval = np.mean(inter_times)
            std_interval = np.std(inter_times)
            cv = std_interval / mean_interval
            cvs.append(cv)

        # Plot 1: CV distribution
        ax = axes[0, 0]
        ax.hist(cvs, bins=25, edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(cvs), color="red", linestyle="--", linewidth=2, label=f"Mean CV: {np.mean(cvs):.3f}")
        ax.set_xlabel("Coefficient of Variation (CV)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Distribution of CV (Lower = More Periodic)", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Inter-event time distribution
        ax = axes[0, 1]
        ax.hist(all_inter_times, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(all_inter_times), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(all_inter_times):.4f}")
        ax.set_xlabel("Inter-Event Time", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Distribution of Inter-Event Times", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Within-sample consistency - compare mean intervals across paths in same sample
        ax = axes[1, 0]
        sample_idx = 0
        mean_intervals_per_path = []
        for p in range(min(20, P)):
            times = event_times[sample_idx, p, :, 0].numpy()
            inter_times = np.diff(times)
            mean_intervals_per_path.append(np.mean(inter_times))

        ax.plot(range(len(mean_intervals_per_path)), mean_intervals_per_path, "o-", linewidth=2, markersize=6)
        ax.axhline(
            np.mean(mean_intervals_per_path),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(mean_intervals_per_path):.4f}",
        )
        ax.set_xlabel("Path Index (within same sample)", fontsize=11)
        ax.set_ylabel("Mean Inter-Event Time", fontsize=11)
        ax.set_title("Consistency Within Sample (should be similar)", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Between-sample variation - compare mean intervals across different samples
        ax = axes[1, 1]
        mean_intervals_per_sample = []
        for n in range(min(20, N)):
            times = event_times[n, 0, :, 0].numpy()  # Use first path of each sample
            inter_times = np.diff(times)
            mean_intervals_per_sample.append(np.mean(inter_times))

        ax.plot(range(len(mean_intervals_per_sample)), mean_intervals_per_sample, "s-", linewidth=2, markersize=6, color="green")
        ax.axhline(
            np.mean(mean_intervals_per_sample),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(mean_intervals_per_sample):.4f}",
        )
        ax.set_xlabel("Sample Index", fontsize=11)
        ax.set_ylabel("Mean Inter-Event Time", fontsize=11)
        ax.set_title("Variation Between Samples (should vary)", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved pattern analysis plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Verify and visualize pattern data")
    parser.add_argument("data_dir", type=str, help="Path to generated data directory")
    parser.add_argument("--pattern", type=str, required=True, choices=["taxi", "amazon"], help="Pattern type to verify")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of trajectories to plot (default: 10)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 80)
    print(f"Verifying {args.pattern.upper()} pattern data")
    print("=" * 80)
    print(f"Data directory: {data_dir}")

    # Load data
    print("\nLoading data...")
    event_times, event_types = load_data(data_dir, split="train")
    print(f"Loaded event_times shape: {event_times.shape}")
    print(f"Loaded event_types shape: {event_types.shape}")

    # Verify pattern
    print(f"\nAnalyzing {args.pattern} pattern...")
    if args.pattern == "taxi":
        stats = verify_taxi_pattern(event_types)
        print("\nAlternation Statistics:")
        print(f"  Mean alternation rate: {stats['mean_alternation_rate']:.3f} (expected ~0.85)")
        print(f"  Std alternation rate:  {stats['std_alternation_rate']:.3f}")
        print(f"  Min alternation rate:  {stats['min_alternation_rate']:.3f}")
        print(f"  Max alternation rate:  {stats['max_alternation_rate']:.3f}")
    elif args.pattern == "amazon":
        stats = verify_amazon_pattern(event_times)
        print("\nPeriodicity Statistics:")
        print(f"  Mean CV: {stats['mean_cv']:.4f} (lower = more periodic)")
        print(f"  Std CV:  {stats['std_cv']:.4f}")
        print(f"  Mean inter-event time: {stats['mean_interval']:.4f}")
        print(f"  Std across trajectories: {stats['std_interval_across_trajs']:.4f}")

    # Create visualizations
    print("\nCreating visualizations...")

    print("Creating detailed raster plot...")
    plot_detailed_raster(
        event_times,
        event_types,
        num_samples=args.num_samples,
        pattern_name=args.pattern,
        output_path=data_dir / f"detailed_raster_{args.pattern}.png",
    )

    print("Creating pattern analysis plot...")
    plot_pattern_analysis(
        event_times, event_types, pattern_name=args.pattern, output_path=data_dir / f"pattern_analysis_{args.pattern}.png"
    )

    print("\n" + "=" * 80)
    print("Verification complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
