from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fim import project_path
from fim.utils.sde.evaluation import save_fig


def drift(x_1, x_2):
    return np.stack([-(0.1 * x_1 - 2.0 * x_2), -(2.0 * x_1 + 0.1 * x_2)], axis=-1)


def diffusion(x_1, x_2):
    return np.ones_like(np.stack([x_1, x_2], axis=-1))


def create_fig(vector_field_func, scale, width, color, file_name):
    grid = np.mgrid[-2:2:7j, -2:2:7j]  # (2, X, X)

    x_1 = grid[0]
    x_2 = grid[1]

    vf_at_grid = vector_field_func(x_1, x_2)

    fig = plt.figure(figsize=(1.5, 1.5), dpi=300)
    ax = fig.add_axes(111)

    [x.set_linewidth(0.3) for x in ax.spines.values()]
    ax.tick_params(axis="both", direction="out", labelsize=4, width=0.5, length=2)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.quiver(x_1, x_2, vf_at_grid[..., 0], vf_at_grid[..., 1], scale_units="xy", scale=scale, width=width, color=color)

    save_fig(fig, Path(project_path), file_name)


if __name__ == "__main__":
    create_fig(drift, 5.5, 0.016, "black", "ex_vf_drift_black")
    create_fig(drift, 5.5, 0.016, "#0072B2", "ex_vf_drift_blue")
    create_fig(diffusion, 3, 0.014, "black", "ex_vf_diffusion_black")
    create_fig(diffusion, 3, 0.014, "#0072B2", "ex_vf_diffusion_blue")
