from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import MaxNLocator


try:
    # TensorBoard event reader
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:  # pragma: no cover
    EventAccumulator = None  # type: ignore


PRETRAINED_ROOT_DEFAULT = "/cephfs/users/berghaus/FoundationModels/FIM/results/fim_benchmark/251112-0936/_finetune"
SCRATCH_ROOT_DEFAULT = "/cephfs/users/berghaus/FoundationModels/FIM/results/fim_benchmark/251112-0937/_finetune"
SAVE_PATH_DEFAULT = "scripts/hawkes/paper_plots/val_loss_finetune_pretrained_vs_scratch.pdf"


def _latest_run_dir(dataset_root: Path) -> Optional[Path]:
    if not dataset_root.exists():
        return None
    # Pick latest (lexicographically) subdir as timestamped run
    subdirs = [p for p in dataset_root.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    subdirs.sort(key=lambda p: p.name)
    return subdirs[-1]


def _tensorboard_dir_for_dataset(run_root: Path, dataset: str) -> Optional[Path]:
    ds_dir = run_root / dataset
    latest = _latest_run_dir(ds_dir)
    if latest is None:
        return None
    tb_dir = latest / "tensorboard"
    return tb_dir if tb_dir.exists() else None


def _read_val_nll(tb_path: Path) -> Tuple[List[int], List[float]]:
    """
    Read (epoch, value) for tag 'val/nll' from a TensorBoard log directory or file.
    Falls back to 'validation/nll' or 'val/loss' if needed.
    """
    if EventAccumulator is None:
        raise RuntimeError("TensorBoard is not available. Install with `pip install tensorboard`.")
    # EventAccumulator can take a directory or a file path
    ea = EventAccumulator(str(tb_path), size_guidance={"scalars": 0})
    ea.Reload()
    candidate_tags = ["val/nll", "validation/nll", "val/loss"]
    scalars = None
    for tag in candidate_tags:
        try:
            s = ea.Scalars(tag)
            if s:
                scalars = s
                break
        except KeyError:
            continue
    if not scalars:
        return [], []
    # Steps are epochs as logged by fim_finetune.py
    steps = [int(ev.step) for ev in scalars if ev.value == ev.value]  # filter NaNs
    values = [float(ev.value) for ev in scalars if ev.value == ev.value]
    # Sort by step to ensure monotonic x
    paired = sorted(zip(steps, values), key=lambda t: t[0])
    if not paired:
        return [], []
    xs, ys = zip(*paired)
    return list(xs), list(ys)


def _apply_plot_style():
    # Okabe–Ito colorblind-friendly palette
    okabe_ito_colors = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ]
    plt.rcParams["axes.prop_cycle"] = cycler(color=okabe_ito_colors)
    # Computer Modern-like serif fonts and sizing
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "CMU Serif",
                "DejaVu Serif",
            ],
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


def plot_val_nll_row(
    pretrained_root: Path,
    scratch_root: Path,
    datasets: List[str],
    save_path: Path,
) -> None:
    _apply_plot_style()

    # Explicit styles to mirror the paper styling approach
    style_pretrained = {"color": "#009E73", "marker": "D", "label": "FIM-PP (pre-trained)", "linewidth": 2.0, "markersize": 6}
    style_scratch = {"color": "#D55E00", "marker": "o", "label": "Random Weight Initialization", "linewidth": 2.0, "markersize": 5}

    # Create a single-row, four-panel figure
    fig, axes = plt.subplots(1, len(datasets), figsize=(16, 4))
    if len(datasets) == 1:
        axes = [axes]

    display_name: Dict[str, str] = {
        "amazon": "Amazon",
        "stackoverflow": "Stack Overflow",
        "taobao": "Taobao",
        "taxi": "Taxi",
    }

    any_plotted = False
    for ax, ds in zip(axes, datasets):
        ax.set_title(display_name.get(ds, ds.capitalize()))
        # Locate TensorBoard directories
        tb_pre = _tensorboard_dir_for_dataset(pretrained_root, ds)
        tb_scr = _tensorboard_dir_for_dataset(scratch_root, ds)

        xs_pre: List[int] = []
        ys_pre: List[float] = []
        xs_scr: List[int] = []
        ys_scr: List[float] = []

        try:
            if tb_pre is not None:
                xs_pre, ys_pre = _read_val_nll(tb_pre)
        except Exception:
            xs_pre, ys_pre = [], []
        try:
            if tb_scr is not None:
                xs_scr, ys_scr = _read_val_nll(tb_scr)
        except Exception:
            xs_scr, ys_scr = [], []

        # Clip Taobao at 3000 epochs, Taxi at 2000 epochs
        if ds == "taobao":
            if xs_pre and ys_pre:
                clipped_pre = [(x, y) for x, y in zip(xs_pre, ys_pre) if x <= 3000]
                if clipped_pre:
                    xs_pre, ys_pre = zip(*clipped_pre)
                    xs_pre, ys_pre = list(xs_pre), list(ys_pre)
                else:
                    xs_pre, ys_pre = [], []
            if xs_scr and ys_scr:
                clipped_scr = [(x, y) for x, y in zip(xs_scr, ys_scr) if x <= 3000]
                if clipped_scr:
                    xs_scr, ys_scr = zip(*clipped_scr)
                    xs_scr, ys_scr = list(xs_scr), list(ys_scr)
                else:
                    xs_scr, ys_scr = [], []
        elif ds == "taxi":
            if xs_pre and ys_pre:
                clipped_pre = [(x, y) for x, y in zip(xs_pre, ys_pre) if x <= 2000]
                if clipped_pre:
                    xs_pre, ys_pre = zip(*clipped_pre)
                    xs_pre, ys_pre = list(xs_pre), list(ys_pre)
                else:
                    xs_pre, ys_pre = [], []
            if xs_scr and ys_scr:
                clipped_scr = [(x, y) for x, y in zip(xs_scr, ys_scr) if x <= 2000]
                if clipped_scr:
                    xs_scr, ys_scr = zip(*clipped_scr)
                    xs_scr, ys_scr = list(xs_scr), list(ys_scr)
                else:
                    xs_scr, ys_scr = [], []

        if xs_pre and ys_pre:
            ax.plot(xs_pre, ys_pre, **style_pretrained)
            any_plotted = True
        else:
            ax.text(0.5, 0.5, "No val points\n(pretrained)", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        if xs_scr and ys_scr:
            ax.plot(xs_scr, ys_scr, **style_scratch)
            any_plotted = True
        else:
            ax.text(0.5, 0.35, "No val points\n(scratch)", ha="center", va="center", transform=ax.transAxes, fontsize=10)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"NLL (validation)")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)
        ax.tick_params(axis="both", direction="out", width=0.5, length=2, pad=0.8)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
        ax.margins(x=0)
        if ds == "taobao":
            try:
                # Ensure a clean cutoff at 3000 on the x-axis
                ax.set_xlim(left=0, right=3000)
            except Exception:
                pass
        elif ds == "taxi":
            try:
                # Ensure a clean cutoff at 2000 on the x-axis
                ax.set_xlim(left=0, right=2000)
            except Exception:
                pass

    # Figure-level legend (deduplicate labels)
    handles_all, labels_all = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles_all += h
        labels_all += l
    unique: List[Tuple[object, str]] = []
    seen = set()
    for h, l in zip(handles_all, labels_all):
        if l not in seen:
            unique.append((h, l))
            seen.add(l)
    if unique:
        uh, ul = zip(*unique)
        import math

        n_items = len(ul)
        ncols = max(1, math.ceil(n_items / 3))
        legend = fig.legend(
            uh,
            ul,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.06),
            ncol=ncols,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            borderpad=0.2,
            handletextpad=0.9,
            borderaxespad=0.2,
            labelspacing=0.35,
            columnspacing=1.6,
        )
        frame = legend.get_frame()
        frame.set_linewidth(0.6)
        frame.set_edgecolor("black")
        frame.set_facecolor("white")

    plt.tight_layout(rect=[0, 0, 1, 0.86])
    plt.subplots_adjust(wspace=0.3, hspace=0.0)

    save_path = save_path.resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight", format="pdf")
    print(f"Saved figure to: {save_path}")
    if not any_plotted:
        print("Warning: No validation points were found to plot.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot validation NLL during fine-tuning: pretrained vs scratch, 4 datasets.")
    parser.add_argument(
        "--pretrained_root",
        type=str,
        default=PRETRAINED_ROOT_DEFAULT,
        help="Root directory containing pretrained fine-tune runs (per dataset subfolder).",
    )
    parser.add_argument(
        "--scratch_root",
        type=str,
        default=SCRATCH_ROOT_DEFAULT,
        help="Root directory containing scratch fine-tune runs (per dataset subfolder).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=["amazon", "stackoverflow", "taobao", "taxi"],
        help="Datasets to include (must match directory names).",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=SAVE_PATH_DEFAULT,
        help="Path to save the output PDF.",
    )
    args = parser.parse_args()

    plot_val_nll_row(
        pretrained_root=Path(args.pretrained_root),
        scratch_root=Path(args.scratch_root),
        datasets=list(args.datasets),
        save_path=Path(args.save_path),
    )


if __name__ == "__main__":
    main()
