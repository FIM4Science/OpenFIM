"""MSE quantiles for Max (FIM-ODE) vs ODEFormer on ODE-bench.

Each JSON leaf is a length-122 vector with blocks:
  - rows 0..45:   1D systems (23 benchmarks × 2)
  - rows 46..101: 2D systems (28 × 2)
  - rows 102..121: 3D systems (10 × 2)

Reconstruction and generalization are **not** mixed. For each ODE state dimension
we print a markdown table (ODEFormer / FIM-ODE × reconstruction / generalization,
four rows) with 5th percentile, median, and 95th; values pool that block’s rows
across all ρ / σ leaves for the given task.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

_ROOT = Path(__file__).resolve().parent
_MAX_PATH = _ROOT / "results" / "max" / "all_mse_stats.json"
_ODEFORMER_PATH = _ROOT / "results" / "odeformer" / "all_mse_stats.json"
_MAX_TOP_KEY = "base_model_ckpt-None"
_ODEFORMER_TOP_KEY = "odeformer"

_TASK_ORDER = ("reconstruction", "generalization")

# Per-state-dimension flat row ranges in each leaf (0-based).
_DIMENSION_GROUPS: tuple[tuple[str, range], ...] = (
    ("For 1-dimensional ODEBench systems:  ", range(0, 46)),
    ("For 2-dimensional ODEBench systems:  ", range(46, 102)),
    ("For 3-dimensional ODEBench systems:  ", range(102, 122)),
)


def _iter_leaf_mse_lists_for_task(
    tree: dict[str, Any], task: str
) -> Iterator[list[float]]:
    """Yield each leaf MSE vector for one task (all ρ, σ)."""
    for rho in tree[task]:
        for sigma in tree[task][rho]:
            yield tree[task][rho][sigma]


def _collect_group_rows(
    tree: dict[str, Any], task: str, rows: range
) -> np.ndarray:
    """All MSE values for fixed flat indices over every leaf of that task."""
    out: list[float] = []
    for leaf in _iter_leaf_mse_lists_for_task(tree, task):
        for j in rows:
            out.append(float(leaf[j]))
    return np.asarray(out, dtype=np.float64)


def _summarize(arr: np.ndarray) -> tuple[float, float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    q_lo, med, q_hi = np.quantile(finite, [0.05, 0.5, 0.95])
    return float(q_lo), float(med), float(q_hi)


def _fmt_cell(x: float) -> str:
    """Format a scalar for the markdown table (match paper-style readability)."""
    if not np.isfinite(x):
        return "nan"
    ax = abs(x)
    if ax < 1e-2 or ax >= 1e4:
        return f"{x:.2e}"
    if ax < 1.0:
        return f"{x:.3g}"
    return f"{x:.2f}"


def _markdown_row(label: str, q05: float, med: float, q95: float) -> str:
    a, b, c = _fmt_cell(q05), _fmt_cell(med), _fmt_cell(q95)
    return f"| {label} | {a} | {b} | {c} |"


def _print_table_for_rows(
    max_tree: dict[str, Any],
    ode_tree: dict[str, Any],
    rows: range,
) -> None:
    """Emit one markdown table (header + 4 data rows) for the given row block."""
    print("| Model and Task | 0.05-quantile of MSE | Median MSE | 0.95-quantile of MSE |")
    print("| --------- | -------------------- | ---------- | -------------------- |")
    for task in _TASK_ORDER:
        ode_v = _collect_group_rows(ode_tree, task, rows)
        fim_v = _collect_group_rows(max_tree, task, rows)
        oq05, omed, oq95 = _summarize(ode_v)
        mq05, mmed, mq95 = _summarize(fim_v)
        print(_markdown_row(f"ODEFormer {task}", oq05, omed, oq95))
        print(_markdown_row(f"FIM-ODE {task}", mq05, mmed, mq95))


def main() -> None:
    with open(_MAX_PATH, encoding="utf-8") as f:
        max_tree = json.load(f)[_MAX_TOP_KEY]
    with open(_ODEFORMER_PATH, encoding="utf-8") as f:
        ode_tree = json.load(f)[_ODEFORMER_TOP_KEY]

    for dim_i, (dim_title, rows) in enumerate(_DIMENSION_GROUPS):
        if dim_i:
            print()
        print(f"{dim_title}")
        print()
        _print_table_for_rows(max_tree, ode_tree, rows)


if __name__ == "__main__":
    main()
