#!/usr/bin/env python3
"""
Script to find sequence length statistics (min, max, mean) for multiple EasyTPP datasets.

RESULTS SUMMARY:
================================================================================
Dataset                   Total Seqs   Min    Max    Mean
--------------------------------------------------------------------------------
RETWEET                   24000        50     264    108.75
TAOBAO                    2000         28     64     56.70
AMAZON                    9227         14     94     44.81
TAXI                      2000         36     38     37.04
STACKOVERFLOW             2203         41     101    64.81
================================================================================

DETAILED BREAKDOWN:

RETWEET (easytpp/retweet):
- Train: 20,000 sequences | Min: 50, Max: 264, Mean: 108.81
- Test: 2,000 sequences   | Min: 50, Max: 264, Mean: 109.23
- Validation: 2,000 sequences | Min: 50, Max: 264, Mean: 107.76
- OVERALL: 24,000 sequences | Min: 50, Max: 264, Mean: 108.75

TAOBAO (easytpp/taobao):
- Train: 1,300 sequences | Min: 28, Max: 64, Mean: 56.53
- Test: 500 sequences    | Min: 32, Max: 64, Mean: 56.91
- Validation: 200 sequences | Min: 31, Max: 64, Mean: 57.36
- OVERALL: 2,000 sequences | Min: 28, Max: 64, Mean: 56.70

AMAZON (easytpp/amazon):
- Train: 6,454 sequences | Min: 14, Max: 94, Mean: 44.68
- Test: 1,851 sequences  | Min: 14, Max: 94, Mean: 45.41
- Validation: 922 sequences | Min: 15, Max: 94, Mean: 44.46
- OVERALL: 9,227 sequences | Min: 14, Max: 94, Mean: 44.81

TAXI (easytpp/taxi):
- Train: 1,400 sequences | Min: 36, Max: 38, Mean: 37.04
- Test: 400 sequences    | Min: 36, Max: 38, Mean: 37.05
- Validation: 200 sequences | Min: 36, Max: 38, Mean: 37.02
- OVERALL: 2,000 sequences | Min: 36, Max: 38, Mean: 37.04

STACKOVERFLOW (easytpp/stackoverflow):
- Train: 1,401 sequences | Min: 41, Max: 101, Mean: 64.59
- Test: 401 sequences    | Min: 41, Max: 101, Mean: 66.13
- Validation: 401 sequences | Min: 41, Max: 101, Mean: 64.24
- OVERALL: 2,203 sequences | Min: 41, Max: 101, Mean: 64.81

KEY OBSERVATIONS:
- RETWEET: Longest sequences (up to 264 events), largest dataset (24K sequences)
- AMAZON: Shortest minimum sequences (14 events), second-largest dataset (9.2K sequences)
- TAXI: Very consistent sequence lengths (36-38 events) with minimal variation
- TAOBAO & STACKOVERFLOW: Mid-sized datasets with moderate sequence lengths
- Wide variation across datasets: from 14 (AMAZON) to 264 (RETWEET) events
"""

import numpy as np
from datasets import load_dataset


def calculate_dataset_statistics(dataset_name):
    """
    Load a dataset and calculate sequence length statistics across all splits.
    """
    print(f"\nProcessing dataset: {dataset_name}")
    print("-" * 50)

    # Try to load all common splits
    splits_to_check = ["train", "test", "validation"]
    all_lengths = []
    split_stats = {}

    for split in splits_to_check:
        try:
            dataset_split = load_dataset(dataset_name, split=split)

            # Get all sequence lengths for this split
            split_lengths = dataset_split["seq_len"]
            split_stats[split] = {
                "count": len(split_lengths),
                "min": min(split_lengths),
                "max": max(split_lengths),
                "mean": np.mean(split_lengths),
            }

            # Add to overall collection
            all_lengths.extend(split_lengths)

            print(f"{split.capitalize()} split: {len(dataset_split)} sequences")
            print(f"  Min: {split_stats[split]['min']}, Max: {split_stats[split]['max']}, Mean: {split_stats[split]['mean']:.2f}")

        except Exception:
            print(f"{split.capitalize()} split: Not available")
            continue

    if all_lengths:
        # Calculate overall statistics
        overall_stats = {"count": len(all_lengths), "min": min(all_lengths), "max": max(all_lengths), "mean": np.mean(all_lengths)}

        print(
            f"\nOVERALL ({overall_stats['count']} sequences): Min={overall_stats['min']}, Max={overall_stats['max']}, Mean={overall_stats['mean']:.2f}"
        )
        return overall_stats
    else:
        print("No data found!")
        return None


def main():
    """
    Calculate statistics for all specified datasets.
    """
    datasets = ["easytpp/retweet", "easytpp/taobao", "easytpp/amazon", "easytpp/taxi", "easytpp/stackoverflow"]

    print("SEQUENCE LENGTH STATISTICS FOR EASYTPP DATASETS")
    print("=" * 60)

    results = {}

    for dataset_name in datasets:
        try:
            stats = calculate_dataset_statistics(dataset_name)
            if stats:
                results[dataset_name] = stats
        except Exception as e:
            print(f"\nError loading {dataset_name}: {str(e)}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Dataset':<25} {'Total Seqs':<12} {'Min':<6} {'Max':<6} {'Mean':<8}")
    print("-" * 80)

    for dataset_name, stats in results.items():
        dataset_short = dataset_name.replace("easytpp/", "").upper()
        print(f"{dataset_short:<25} {stats['count']:<12} {stats['min']:<6} {stats['max']:<6} {stats['mean']:<8.2f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
