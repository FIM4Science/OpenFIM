#!/usr/bin/env python3
"""
Verify that for each CDiff dataset split (train/val/dev/test), every non-empty
sequence's first event time (time_since_start of first event) is zero.

Usage:
  python scripts/helpers/cdiff_verify_first_zero.py \
    --root /cephfs/users/berghaus/FoundationModels/FIM/data/external/CDiff_dataset
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    args = ap.parse_args()

    root = Path(args.root)
    splits = ["train", "val", "dev", "test"]
    any_bad = False

    for ddir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for split in splits:
            pkl = ddir / f"{split}.pkl"
            if not pkl.exists():
                continue
            with open(pkl, "rb") as f:
                d = pickle.load(f)
            seqs = d.get(split)
            if not isinstance(seqs, list):
                print(f"{ddir.name:12s} {split:5s} ERROR: split value is {type(seqs)} not list")
                any_bad = True
                continue
            total = len(seqs)
            nonempty = 0
            bad = 0
            examples: list[tuple[int, float]] = []
            for i, seq in enumerate(seqs):
                if not isinstance(seq, list) or len(seq) == 0:
                    continue
                nonempty += 1
                first = float(seq[0].get("time_since_start", 0.0))
                if abs(first) > 1e-9:
                    bad += 1
                    if len(examples) < 3:
                        examples.append((i, first))
            print(f"{ddir.name:12s} {split:5s} total={total:5d} nonempty={nonempty:5d} first!=0={bad:5d} examples={examples}")
            if bad > 0:
                any_bad = True

    print("\nOK" if not any_bad else "\nVIOLATIONS PRESENT")


if __name__ == "__main__":
    main()
