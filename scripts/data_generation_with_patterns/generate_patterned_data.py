#!/usr/bin/env python3
"""
Generate synthetic event data with specific patterns for Hawkes process modeling.

This script generates event sequences that mimic patterns observed in real-world datasets:
- Taxi pattern: Marks alternate frequently with dominant types (using Dirichlet + Beta distributions)
- Amazon pattern: Periodic event clusters with power-law mark distribution (using Poisson + Zipf)

USAGE EXAMPLES:

# Generate Taxi pattern data (alternating marks)
python scripts/data_generation_with_patterns/generate_patterned_data.py \
    --pattern taxi \
    --N_processes_train 1000 \
    --N_processes_val 100 \
    --N_processes_test 100 \
    --P_trajectories_per_process 2000 \
    --K_events_per_trajectory 100 \
    --M_dimensions 5 \
    --output_name 1k_5D_2k_paths_alternating_pattern \
    --seed 42

# Generate Amazon pattern data (periodic clusters)
python scripts/data_generation_with_patterns/generate_patterned_data.py \
    --pattern amazon \
    --N_processes_train 1000 \
    --N_processes_val 100 \
    --N_processes_test 100 \
    --P_trajectories_per_process 2000 \
    --K_events_per_trajectory 100 \
    --M_dimensions 16 \
    --output_name 1k_16D_2k_paths_periodic_pattern \
    --seed 42

OUTPUT:
- Saves to: data/synthetic_data/hawkes/{output_name}/
- Creates train/val/test splits with .h5 and .pt files
- Generates metadata JSON and visualization plots
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# Add src to path to import DataSaver
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fim.data_generation.data_saver import DataSaver


def generate_taxi_pattern(
    N_processes: int,
    P_trajectories: int,
    K_events: int,
    M_dimensions: int,
    seed: int = 0,
    dirichlet_alpha: float = 0.1,  # Lower α → more concentrated on 2 marks
    beta_alternation: tuple = (20.0, 1.0),  # Higher → closer to 100% alternation
    mean_interarrival: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate event data with taxi pattern: marks alternate frequently with dominant marks.

    Mathematical Model:
    - Event times: Exponential inter-arrival distribution ~ Exp(λ)
    - Mark probabilities: Dirichlet distribution ~ Dir(α, ..., α) per process
      - Low α creates sparse/concentrated distributions (few dominant marks)
      - High α creates uniform distributions (all marks equally likely)
    - Alternation behavior: Beta distribution ~ Beta(a, b) for alternation probability
      - High a, low b → high alternation probability

    Args:
        N_processes: Number of processes (samples)
        P_trajectories: Number of trajectories per process
        K_events: Number of events per trajectory
        M_dimensions: Number of mark dimensions
        seed: Random seed
        dirichlet_alpha: Concentration parameter for Dirichlet distribution (default: 0.3)
                        Lower values create more concentrated/dominant marks
        beta_alternation: (a, b) parameters for Beta distribution controlling alternation
                         Default (8.5, 1.5) gives mean ≈ 0.85
        mean_interarrival: Mean time between events (default 0.03)

    Returns:
        event_times: [N, P, K, 1] array of event times
        event_types: [N, P, K, 1] array of event types
    """
    np.random.seed(seed)

    event_times = np.zeros((N_processes, P_trajectories, K_events, 1), dtype=np.float32)
    event_types = np.zeros((N_processes, P_trajectories, K_events, 1), dtype=np.int64)

    for n in range(N_processes):
        # Sample mark probabilities from Dirichlet distribution
        # Low alpha creates sparse distributions (few dominant marks)
        mark_probs = np.random.dirichlet([dirichlet_alpha] * M_dimensions)

        # Sample alternation probability from Beta distribution
        alternation_prob = np.random.beta(beta_alternation[0], beta_alternation[1])

        for p in range(P_trajectories):
            # Generate inter-arrival times using exponential distribution
            # Sample rate parameter from Gamma distribution for variability
            rate = np.random.gamma(2.0, mean_interarrival / 2.0)
            inter_arrivals = np.random.exponential(1.0 / rate, K_events)
            times = np.cumsum(inter_arrivals)
            times = times - times[0]  # Start at 0
            event_times[n, p, :, 0] = times

            # Generate marks using Dirichlet-based probabilities with alternation
            current_mark = np.random.choice(M_dimensions, p=mark_probs)
            event_types[n, p, 0, 0] = current_mark

            for k in range(1, K_events):
                if np.random.rand() < alternation_prob:
                    # Alternate: sample different mark from conditional distribution
                    # Create conditional probabilities (exclude current mark, renormalize)
                    conditional_probs = mark_probs.copy()
                    conditional_probs[current_mark] = 0
                    conditional_probs /= conditional_probs.sum()
                    current_mark = np.random.choice(M_dimensions, p=conditional_probs)
                else:
                    # Don't alternate: sample from original distribution (may stay same)
                    current_mark = np.random.choice(M_dimensions, p=mark_probs)

                event_types[n, p, k, 0] = current_mark

    return event_times, event_types


def generate_amazon_pattern(
    N_processes: int,
    P_trajectories: int,
    K_events: int,
    M_dimensions: int,
    seed: int = 0,
    period_range: tuple = (1.0, 2.0),  # Larger periods for realistic time scale
    noise_factor: float = 0.03,  # Low noise → lower CV, more periodic
    poisson_lambda: float = 3.0,
    zipf_exponent: float = 1.5,  # Power-law distribution with moderate skew
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate event data with amazon pattern: clusters of events occur at periodic times.

    Mathematical Model:
    - Base period: Uniform distribution ~ U(period_min, period_max) per process
    - Trigger times: t_i = i * T + ε, where ε ~ N(0, σ²) with σ = noise_factor * T
    - Cluster sizes: Poisson distribution ~ Poisson(λ) + 1 (shifted to ensure ≥1)
    - Event times within cluster: Normal distribution ~ N(trigger, (T * 0.15)²)
    - Mark probabilities: Zipf distribution ~ Zipf(s) where P(k) ∝ 1/k^s
      - Higher s creates more skewed distribution (few dominant marks)

    All trajectories within a process share the same period and mark distribution.

    Args:
        N_processes: Number of processes (samples)
        P_trajectories: Number of trajectories (paths) per process
        K_events: Number of events per trajectory
        M_dimensions: Number of mark dimensions
        seed: Random seed
        period_range: (min, max) range for base period (default: 0.05-0.15)
        noise_factor: Fraction of period for trigger noise std (default: 0.1)
        poisson_lambda: Mean parameter for Poisson cluster size distribution (default: 3.0)
        zipf_exponent: Exponent for Zipf distribution (default: 1.5)
                      Higher values create more skewed distributions

    Returns:
        event_times: [N, P, K, 1] array of event times
        event_types: [N, P, K, 1] array of event types
    """
    np.random.seed(seed)

    event_times = np.zeros((N_processes, P_trajectories, K_events, 1), dtype=np.float32)
    event_types = np.zeros((N_processes, P_trajectories, K_events, 1), dtype=np.int64)

    for n in range(N_processes):
        # Sample a base period for this PROCESS (same for all paths in this process)
        base_period = np.random.uniform(period_range[0], period_range[1])
        noise_std = base_period * noise_factor

        # Create Zipf-distributed mark probabilities (power-law distribution)
        # P(k) ∝ 1 / k^s where k is the rank (1, 2, 3, ...)
        # Sample exponent from a range to add variability between processes
        s = np.random.uniform(zipf_exponent * 0.8, zipf_exponent * 1.2)
        ranks = np.arange(1, M_dimensions + 1)
        mark_probs = 1.0 / (ranks**s)
        mark_probs = mark_probs / mark_probs.sum()  # Normalize

        for p in range(P_trajectories):
            all_times = []
            all_marks = []

            cluster_idx = 0
            while len(all_times) < K_events:
                # Periodic trigger point with Gaussian noise
                trigger_time = cluster_idx * base_period + np.random.normal(0, noise_std)
                trigger_time = max(0, trigger_time)

                # Sample cluster size from shifted Poisson distribution
                # Add 1 to ensure at least 1 event per cluster
                cluster_size = np.random.poisson(poisson_lambda) + 1
                cluster_size = min(cluster_size, K_events - len(all_times))  # Don't exceed K_events

                # Generate events within this cluster
                # Events are tightly clustered around the trigger time
                cluster_spread = base_period * 0.02  # Very tight clustering for lower CV

                for _ in range(cluster_size):
                    # Event time: normally distributed around trigger
                    event_time = trigger_time + np.random.normal(0, cluster_spread)
                    event_time = max(0, event_time)
                    all_times.append(event_time)

                    # Mark: sample from Zipf distribution (lower ranks more likely)
                    mark = np.random.choice(M_dimensions, p=mark_probs)
                    all_marks.append(mark)

                cluster_idx += 1

            # Take exactly K_events (in case we generated too many)
            all_times = all_times[:K_events]
            all_marks = all_marks[:K_events]

            # Sort by time to maintain chronological order
            sorted_indices = np.argsort(all_times)
            times = np.array([all_times[i] for i in sorted_indices], dtype=np.float32)
            marks = np.array([all_marks[i] for i in sorted_indices], dtype=np.int64)

            # Ensure strictly increasing times
            for k in range(1, len(times)):
                if times[k] <= times[k - 1]:
                    times[k] = times[k - 1] + 0.001

            # Pad if we didn't generate enough (rare edge case)
            if len(times) < K_events:
                last_time = times[-1] if len(times) > 0 else 0
                for k in range(len(times), K_events):
                    times = np.append(times, last_time + (k - len(times) + 1) * 0.01)
                    marks = np.append(marks, np.random.choice(M_dimensions, p=mark_probs))

            event_times[n, p, :, 0] = times
            event_types[n, p, :, 0] = marks

    return event_times, event_types


def create_metadata(args: argparse.Namespace, pattern: str) -> dict:
    """Create metadata dictionary for the generated dataset."""
    metadata = {
        "name": args.output_name,
        "pattern": pattern,
        "N_processes_train": args.N_processes_train,
        "N_processes_val": args.N_processes_val,
        "N_processes_test": args.N_processes_test,
        "P_trajectories_per_process": args.P_trajectories_per_process,
        "K_events_per_trajectory": args.K_events_per_trajectory,
        "M_dimensions": args.M_dimensions,
        "seed": args.seed,
    }

    # Add pattern-specific parameters
    if pattern == "taxi":
        metadata["distribution_parameters"] = {
            "mark_distribution": "Dirichlet",
            "dirichlet_alpha": 0.1,
            "alternation_distribution": "Beta",
            "beta_alternation": [20.0, 1.0],
            "interarrival_distribution": "Exponential",
            "mean_interarrival": 0.03,
        }
    elif pattern == "amazon":
        metadata["distribution_parameters"] = {
            "period_distribution": "Uniform",
            "period_range": [1.0, 2.0],
            "trigger_noise_distribution": "Normal",
            "noise_factor": 0.03,
            "cluster_size_distribution": "Poisson(λ)+1",
            "poisson_lambda": 3.0,
            "mark_distribution": "Zipf",
            "zipf_exponent": 1.5,
            "cluster_spread_factor": 0.02,
        }

    return metadata


def save_metadata(metadata: dict, output_dir: Path):
    """Save metadata as JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{metadata['name']}.json", "w") as f:
        json.dump(metadata, indent=2, fp=f)


def create_raster_plot(
    event_times: np.ndarray,
    event_types: np.ndarray,
    num_samples: int,
    pattern_name: str,
    output_dir: Path,
):
    """
    Create a raster plot visualization of sampled trajectories.

    Args:
        event_times: [N, P, K, 1] array of event times
        event_types: [N, P, K, 1] array of event types
        num_samples: Number of trajectories to sample and plot
        pattern_name: Name of the pattern for plot title
        output_dir: Directory to save the plot
    """
    N, P, K, _ = event_times.shape

    # Sample trajectories
    num_samples = min(num_samples, P)
    sample_process = 0  # Use first process
    sample_indices = np.random.choice(P, num_samples, replace=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Define colors for different marks
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, traj_idx in enumerate(sample_indices):
        times = event_times[sample_process, traj_idx, :, 0]
        types = event_types[sample_process, traj_idx, :, 0]

        for t, mark in zip(times, types):
            ax.plot([t, t], [idx, idx + 0.8], color=colors[int(mark) % 10], linewidth=2)

    ax.set_xlabel("Time (from time_since_start)", fontsize=12)
    ax.set_ylabel("Sampled Trajectory Index", fontsize=12)
    ax.set_title(f"Raster Plot ({num_samples} Sampled Trajectories from train split)\n{pattern_name.capitalize()} Pattern", fontsize=14)
    ax.set_ylim(-0.5, num_samples)
    ax.grid(True, alpha=0.3)

    # Add legend for marks
    M = int(event_types.max()) + 1
    handles = [plt.Line2D([0], [0], color=colors[m % 10], linewidth=2, label=f"Mark {m}") for m in range(M)]
    ax.legend(handles=handles, loc="upper right", title="Event Marks")

    plt.tight_layout()
    plot_path = output_dir / f"raster_plot_{pattern_name}_pattern.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved raster plot to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic event data with taxi or amazon patterns")
    parser.add_argument(
        "--pattern",
        type=str,
        required=True,
        choices=["taxi", "amazon"],
        help="Pattern type: 'taxi' (alternating marks) or 'amazon' (periodic timing)",
    )
    parser.add_argument(
        "--N_processes_train",
        type=int,
        default=100,
        help="Number of training processes (default: 100)",
    )
    parser.add_argument(
        "--N_processes_val",
        type=int,
        default=10,
        help="Number of validation processes (default: 10)",
    )
    parser.add_argument(
        "--N_processes_test",
        type=int,
        default=10,
        help="Number of test processes (default: 10)",
    )
    parser.add_argument(
        "--P_trajectories_per_process",
        type=int,
        default=100,
        help="Number of trajectories per process (default: 100)",
    )
    parser.add_argument(
        "--K_events_per_trajectory",
        type=int,
        default=100,
        help="Number of events per trajectory (default: 100)",
    )
    parser.add_argument(
        "--M_dimensions",
        type=int,
        default=2,
        help="Number of mark dimensions (default: 2)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Dataset name for saving (default: auto-generated from parameters)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Auto-generate output name if not provided
    if args.output_name is None:
        args.output_name = (
            f"{args.N_processes_train + args.N_processes_val + args.N_processes_test}_"
            f"{args.M_dimensions}D_{args.P_trajectories_per_process}_paths_"
            f"{args.K_events_per_trajectory}_events_{args.pattern}_pattern"
        )

    print("=" * 80)
    print(f"Generating {args.pattern.upper()} pattern data")
    print("=" * 80)
    print(f"Output name: {args.output_name}")
    print(f"N_processes: train={args.N_processes_train}, val={args.N_processes_val}, test={args.N_processes_test}")
    print(f"P_trajectories_per_process: {args.P_trajectories_per_process}")
    print(f"K_events_per_trajectory: {args.K_events_per_trajectory}")
    print(f"M_dimensions: {args.M_dimensions}")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    # Generate data for all splits
    total_processes = args.N_processes_train + args.N_processes_val + args.N_processes_test

    print(f"\nGenerating {total_processes} processes...")
    if args.pattern == "taxi":
        event_times, event_types = generate_taxi_pattern(
            N_processes=total_processes,
            P_trajectories=args.P_trajectories_per_process,
            K_events=args.K_events_per_trajectory,
            M_dimensions=args.M_dimensions,
            seed=args.seed,
        )
    elif args.pattern == "amazon":
        event_times, event_types = generate_amazon_pattern(
            N_processes=total_processes,
            P_trajectories=args.P_trajectories_per_process,
            K_events=args.K_events_per_trajectory,
            M_dimensions=args.M_dimensions,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown pattern: {args.pattern}")

    print(f"Generated event_times shape: {event_times.shape}")
    print(f"Generated event_types shape: {event_types.shape}")

    # Compute time offsets per process/path and shift sequences so the first event starts at 0
    # time_offsets[b, p] = original first timestamp of that path
    time_offsets_np = event_times[:, :, 0, 0].copy().astype(np.float32)
    event_times[:, :, :, 0] = event_times[:, :, :, 0] - time_offsets_np[:, :, None]

    # Create seq_lengths and time_offsets
    # seq_lengths: all trajectories have K_events
    seq_lengths = torch.full(
        (total_processes, args.P_trajectories_per_process),
        args.K_events_per_trajectory,
        dtype=torch.long,
    )

    # time_offsets: per-path shift we applied to start sequences at 0
    time_offsets = torch.tensor(time_offsets_np, dtype=torch.float32)

    print(f"Created seq_lengths shape: {seq_lengths.shape}")
    print(f"Created time_offsets shape: {time_offsets.shape}")

    # Prepare data dictionary for saving
    data = {
        "event_times": event_times,
        "event_types": event_types,
        "seq_lengths": seq_lengths,
        "time_offsets": time_offsets,
    }

    # Save data using DataSaver
    print("\nSaving data...")
    output_dir = Path("data/synthetic_data/hawkes") / args.output_name

    data_saver = DataSaver(
        process_type="hawkes",
        dataset_name=args.output_name,
        num_samples_train=args.N_processes_train,
        num_samples_val=args.N_processes_val,
        num_samples_test=args.N_processes_test,
        storage_format="h5",  # Save in h5 format
    )
    data_saver(data)

    # Also save as .pt files
    data_saver_pt = DataSaver(
        process_type="hawkes",
        dataset_name=args.output_name,
        num_samples_train=args.N_processes_train,
        num_samples_val=args.N_processes_val,
        num_samples_test=args.N_processes_test,
        storage_format="torch",  # Save in torch format
    )
    data_saver_pt(data)

    print(f"Data saved to: {output_dir}")

    # Save metadata
    print("\nSaving metadata...")
    metadata = create_metadata(args, args.pattern)
    save_metadata(metadata, output_dir)
    print(f"Metadata saved to: {output_dir / f'{args.output_name}.json'}")

    # Create visualization
    print("\nCreating raster plot visualization...")
    create_raster_plot(
        event_times[: args.N_processes_train],  # Use only training data
        event_types[: args.N_processes_train],
        num_samples=10,
        pattern_name=args.pattern,
        output_dir=output_dir,
    )

    print("\n" + "=" * 80)
    print("Data generation complete!")
    print("=" * 80)
    print(f"\nGenerated files in {output_dir}:")
    print("  - train/")
    print("    - event_times.h5, event_times.pt")
    print("    - event_types.h5, event_types.pt")
    print("    - seq_lengths.pt")
    print("    - time_offsets.h5, time_offsets.pt")
    print("  - val/ (same structure)")
    print("  - test/ (same structure)")
    print(f"  - {args.output_name}.json (metadata)")
    print(f"  - raster_plot_{args.pattern}_pattern.png")


if __name__ == "__main__":
    main()
