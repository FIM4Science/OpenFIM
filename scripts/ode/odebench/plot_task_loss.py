"""
Parse and plot task_loss.log files (epoch: loss per line).

Edit RUN_PATHS (and optionally SAVE_PATH, TITLE) below, then run:
  python experiments/plot_task_loss.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- Configure here: paths to run dirs or task_loss.log files ---
RUN_PATHS = [
    Path("models/vdp2/vdp2_01-29-0223"),
    #Path("models/vdp2/vdp2_01-29-0139"),
]
SAVE_PATH = Path("experiments/task_loss.pdf")  # or None to only show
TITLE = "Task loss vs epoch"


def parse_task_loss_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a task_loss.log file.

    Args:
        path: Path to task_loss.log (or directory containing it).

    Returns:
        epochs: 1D array of epoch numbers.
        losses: 1D array of loss values.
    """
    if path.is_dir():
        path = path / "task_loss.log"
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    epochs = []
    losses = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            try:
                epochs.append(int(parts[0].strip()))
                losses.append(float(parts[1].strip()))
            except ValueError:
                continue

    return np.array(epochs), np.array(losses)


def plot_task_loss(
    paths: list[Path],
    ax: plt.Axes | None = None,
    labels: list[str] | None = None,
    title: str = "Task loss vs epoch",
) -> plt.Figure:
    """Plot task loss from one or more task_loss.log files.

    Args:
        paths: Paths to task_loss.log files or directories containing them.
        ax: Optional axes. If None, a new figure is created.
        labels: Optional list of legend labels (one per path). Default: parent dir name.
        title: Plot title.

    Returns:
        The figure (so caller can save or show).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(paths), 1)))

    for i, p in enumerate(paths):
        epochs, losses = parse_task_loss_log(p)
        if len(epochs) == 0:
            continue
        label = labels[i] if labels is not None and i < len(labels) else p.parent.name
        ax.plot(epochs, losses, "o-", color=colors[i % len(colors)], label=label, markersize=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Task loss")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return fig


if __name__ == "__main__":
    fig = plot_task_loss(RUN_PATHS, title=TITLE)

    if SAVE_PATH is not None:
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(SAVE_PATH, bbox_inches="tight", dpi=150)
        print(f"Saved to {SAVE_PATH}")

    plt.show()
