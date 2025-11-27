import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
EASYTPP_ROOT_DEFAULT = "/cephfs/users/berghaus/FoundationModels/FIM/results/easytpp/checkpoints"
SAVE_PATH_DEFAULT = "scripts/hawkes/paper_plots/amazon_model_comparison.pdf"


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


def _read_easytpp_val_loss(csv_path: Path, yaml_path: Optional[Path] = None) -> Tuple[List[float], List[float]]:
    """
    Read validation loss from easytpp CSV file.
    CSV format: epoch,batch,neg_loglike,num_events
    Returns (epoch_equivalent, loss) tuples.
    Since CSV contains batch-level data within epochs, we convert batches to epoch-equivalent
    by normalizing batch numbers. We sample periodically to avoid too many points.
    """
    import yaml

    if not csv_path.exists():
        return [], []

    # Try to read max_epoch from YAML config
    max_epoch_from_config = None
    if yaml_path and yaml_path.exists():
        try:
            with open(yaml_path, "r") as f:
                config = yaml.safe_load(f)
                max_epoch_from_config = config.get("trainer_config", {}).get("max_epoch")
        except Exception:
            pass

    epochs_equiv: List[float] = []
    losses: List[float] = []

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            all_data = []
            max_batch = 0
            max_epoch = -1

            for row in reader:
                epoch = int(row["epoch"])
                batch = int(row["batch"])
                neg_loglike = float(row["neg_loglike"])
                all_data.append((epoch, batch, neg_loglike))
                max_batch = max(max_batch, batch)
                max_epoch = max(max_epoch, epoch)

            if not all_data:
                return [], []

            # If we have data from multiple epochs, use last batch per epoch
            # Otherwise, convert batches to epoch-equivalent scale
            if max_epoch > 0:
                # Multiple epochs: use last batch per epoch
                current_epoch = None
                last_loss = None

                for epoch, batch, neg_loglike in all_data:
                    if current_epoch is None:
                        current_epoch = epoch
                        last_loss = neg_loglike
                    elif epoch != current_epoch:
                        epochs_equiv.append(float(current_epoch))
                        losses.append(last_loss)
                        current_epoch = epoch
                        last_loss = neg_loglike
                    else:
                        last_loss = neg_loglike

                # Don't forget the last epoch
                if current_epoch is not None and last_loss is not None:
                    epochs_equiv.append(float(current_epoch))
                    losses.append(last_loss)
            else:
                # Single epoch: convert batches to epoch-equivalent
                # If batches go up to ~5000+, treat them as epochs directly (common for long training)
                # Otherwise use max_epoch from config or estimate
                if max_batch > 5000:
                    # Treat batches as epochs directly
                    sample_interval = max(1, len(all_data) // 200)  # Sample ~200 points max
                    for i, (epoch, batch, neg_loglike) in enumerate(all_data):
                        if i % sample_interval == 0 or i == len(all_data) - 1:
                            epochs_equiv.append(float(batch))
                            losses.append(neg_loglike)
                else:
                    # Use max_epoch from config if available, otherwise estimate
                    if max_epoch_from_config is not None:
                        num_epochs = float(max_epoch_from_config)
                    else:
                        num_epochs = 3.0  # Fallback to 3 epochs

                    batches_per_epoch = max_batch / num_epochs if num_epochs > 0 and max_batch > 0 else 1.0
                    sample_interval = max(1, len(all_data) // 200)  # Sample ~200 points max

                    for i, (epoch, batch, neg_loglike) in enumerate(all_data):
                        if i % sample_interval == 0 or i == len(all_data) - 1:
                            epoch_equiv = batch / batches_per_epoch if batches_per_epoch > 0 else float(epoch)
                            epochs_equiv.append(epoch_equiv)
                            losses.append(neg_loglike)
    except Exception as e:
        print(f"Warning: Failed to read CSV {csv_path}: {e}")
        return [], []

    return epochs_equiv, losses


def _discover_easytpp_models(checkpoints_root: Path) -> Dict[str, Tuple[Path, Path]]:
    """
    Discover easytpp models by scanning checkpoints directory.
    Returns dict mapping model_id to (csv_path, yaml_path) tuple.
    """
    import yaml

    model_map: Dict[str, Tuple[Path, Path]] = {}

    if not checkpoints_root.exists():
        return model_map

    for checkpoint_dir in checkpoints_root.iterdir():
        if not checkpoint_dir.is_dir():
            continue

        # Look for *_train_output.yaml files
        yaml_files = list(checkpoint_dir.glob("*_train_output.yaml"))
        if not yaml_files:
            continue

        yaml_file = yaml_files[0]
        csv_file = checkpoint_dir / "validation_loss.csv"

        if not csv_file.exists():
            continue

        try:
            with open(yaml_file, "r") as f:
                config = yaml.safe_load(f)
                model_id = config.get("base_config", {}).get("model_id")
                if model_id:
                    model_map[model_id] = (csv_file, yaml_file)
        except Exception as e:
            print(f"Warning: Failed to read YAML {yaml_file}: {e}")
            continue

    return model_map


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


def plot_amazon_comparison(
    pretrained_root: Path,
    scratch_root: Path,
    easytpp_root: Path,
    dataset: str,
    save_path: Path,
) -> None:
    _apply_plot_style()

    # Model display names
    model_display_names: Dict[str, str] = {
        "NHP": "NHP",
        "RMTPP": "RMTPP",
        "SAHP": "SAHP",
        "THP": "THP",
        "AttNHP": "AttNHP",
    }

    # Color and marker assignments
    # FIM-PP pretrained: #009E73 (green), FIM-PP scratch: #D55E00 (orange)
    # Remaining colors from Okabe-Ito palette for easytpp models
    model_styles: Dict[str, Dict[str, Any]] = {
        "FIM-PP (pre-trained)": {
            "color": "#009E73",
            "marker": "D",
            "linewidth": 1.0,
            "markersize": 6,
        },
        "FIM-PP (random initialization)": {
            "color": "#D55E00",
            "marker": "o",
            "linewidth": 1.0,
            "markersize": 5,
        },
        "NHP": {
            "color": "#E69F00",
            "marker": "s",
            "linewidth": 1.0,
            "markersize": 5,
        },
        "RMTPP": {
            "color": "#56B4E9",
            "marker": "^",
            "linewidth": 1.0,
            "markersize": 5,
        },
        "SAHP": {
            "color": "#F0E442",
            "marker": "v",
            "linewidth": 1.0,
            "markersize": 5,
        },
        "THP": {
            "color": "#0072B2",
            "marker": "p",
            "linewidth": 1.0,
            "markersize": 5,
        },
        "AttNHP": {
            "color": "#CC79A7",
            "marker": "*",
            "linewidth": 1.0,
            "markersize": 6,
        },
    }

    # Create single plot figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_title("Amazon")

    any_plotted = False

    # Read FIM-PP pretrained data
    tb_pre = _tensorboard_dir_for_dataset(pretrained_root, dataset)
    xs_pre: List[int] = []
    ys_pre: List[float] = []
    if tb_pre is not None:
        try:
            xs_pre, ys_pre = _read_val_nll(tb_pre)
        except Exception:
            xs_pre, ys_pre = [], []

    # Truncate at 3000 epochs
    max_epochs = 3000

    if xs_pre and ys_pre:
        # Clip data at max_epochs
        clipped_pre = [(x, y) for x, y in zip(xs_pre, ys_pre) if x <= max_epochs]
        if clipped_pre:
            xs_pre_clipped, ys_pre_clipped = zip(*clipped_pre)
            xs_pre_clipped, ys_pre_clipped = list(xs_pre_clipped), list(ys_pre_clipped)
            style = model_styles["FIM-PP (pre-trained)"].copy()
            style["label"] = "FIM-PP (pre-trained)"
            ax.plot(xs_pre_clipped, ys_pre_clipped, **style)
            any_plotted = True

    # Read FIM-PP scratch data
    tb_scr = _tensorboard_dir_for_dataset(scratch_root, dataset)
    xs_scr: List[int] = []
    ys_scr: List[float] = []
    if tb_scr is not None:
        try:
            xs_scr, ys_scr = _read_val_nll(tb_scr)
        except Exception:
            xs_scr, ys_scr = [], []

    if xs_scr and ys_scr:
        # Clip data at max_epochs
        clipped_scr = [(x, y) for x, y in zip(xs_scr, ys_scr) if x <= max_epochs]
        if clipped_scr:
            xs_scr_clipped, ys_scr_clipped = zip(*clipped_scr)
            xs_scr_clipped, ys_scr_clipped = list(xs_scr_clipped), list(ys_scr_clipped)
            style = model_styles["FIM-PP (random initialization)"].copy()
            style["label"] = "FIM-PP (random initialization)"
            ax.plot(xs_scr_clipped, ys_scr_clipped, **style)
            any_plotted = True

    # Read easytpp models
    easytpp_models = _discover_easytpp_models(easytpp_root)
    for model_id, (csv_path, yaml_path) in sorted(easytpp_models.items()):
        display_name = model_display_names.get(model_id, model_id)
        xs, ys = _read_easytpp_val_loss(csv_path, yaml_path)
        if xs and ys:
            # Clip data at max_epochs
            clipped = [(x, y) for x, y in zip(xs, ys) if x <= max_epochs]
            if clipped:
                xs_clipped, ys_clipped = zip(*clipped)
                xs_clipped, ys_clipped = list(xs_clipped), list(ys_clipped)
                style = model_styles.get(display_name, {}).copy()
                if not style:
                    # Fallback style if model not in predefined styles
                    style = {"color": "#000000", "marker": "x", "linewidth": 1.0, "markersize": 5}
                style["label"] = display_name
                ax.plot(xs_clipped, ys_clipped, **style)
                any_plotted = True

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"NLL (validation)")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    ax.tick_params(axis="both", direction="out", width=0.5, length=2, pad=0.8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
    ax.margins(x=0)
    # Set x-axis limit to max_epochs
    ax.set_xlim(left=0, right=max_epochs)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        import math

        n_items = len(labels)
        ncols = max(1, math.ceil(n_items / 3))
        legend = fig.legend(
            handles,
            labels,
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

    save_path = save_path.resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight", format="pdf")
    print(f"Saved figure to: {save_path}")
    if not any_plotted:
        print("Warning: No validation points were found to plot.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot validation NLL comparison for Amazon dataset: FIM-PP (pretrained/scratch) vs easytpp baselines."
    )
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
        "--easytpp_root",
        type=str,
        default=EASYTPP_ROOT_DEFAULT,
        help="Root directory containing easytpp checkpoints.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="amazon",
        help="Dataset name (must match directory names).",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=SAVE_PATH_DEFAULT,
        help="Path to save the output PDF.",
    )
    args = parser.parse_args()

    plot_amazon_comparison(
        pretrained_root=Path(args.pretrained_root),
        scratch_root=Path(args.scratch_root),
        easytpp_root=Path(args.easytpp_root),
        dataset=args.dataset,
        save_path=Path(args.save_path),
    )


if __name__ == "__main__":
    main()
