from pathlib import Path
from typing import Tuple, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from cycler import cycler
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import griddata

from dataProvider import FimDataloader

plt.rcParams.update({
    'axes.titlepad': 20,
    'figure.titlesize': 28,
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'image.cmap': 'viridis',
    'axes.prop_cycle': cycler('color', sns.color_palette("colorblind"))
})


def streamplot_with_alpha(axs: plt.Axes, x, y, u, v, alpha=0.5, **kwargs):
    stream = axs.streamplot(x, y, u, v, **kwargs)
    stream.lines.set_alpha(alpha)
    for child in plt.gca().get_children():
        if isinstance(child, matplotlib.patches.FancyArrowPatch):
            child.set_alpha(alpha)

    return stream


def contourplot(axs: plt.Axes, x: np.ndarray, y: np.ndarray, magnitude: np.ndarray,
                alphas: Tuple[float, float] = (0.9, 0.5), **kwargs):
    a_line, a_fill = alphas
    cont = axs.contour(x, y, magnitude, alpha=a_line, linewidths=5, **kwargs)
    contf = axs.contourf(x, y, magnitude, alpha=a_fill, **kwargs)

    return cont, contf


def interpolate_vector_field(fx: np.ndarray, coords: np.ndarray, scale: float = 1.0, resolution: int = 100):
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    x_min *= scale
    x_max *= scale

    y_min *= scale
    y_max *= scale

    xi = np.linspace(x_min, x_max, resolution)
    yi = np.linspace(y_min, y_max, resolution)
    xi_mg, yi_mg = np.meshgrid(xi, yi)

    u = griddata(coords, fx[:, 0], (xi_mg, yi_mg), method='cubic')
    v = griddata(coords, fx[:, 1], (xi_mg, yi_mg), method='cubic')

    cut_off_percent = 0.02
    n_rows, n_cols = xi_mg.shape
    trim_r = max(int(n_rows * cut_off_percent), 1)
    trim_c = max(int(n_cols * cut_off_percent), 1)

    # Slice all arrays to remove 10% from each edge
    xi_mg = xi_mg[trim_r:-trim_r, trim_c:-trim_c]
    yi_mg = yi_mg[trim_r:-trim_r, trim_c:-trim_c]
    u = u[trim_r:-trim_r, trim_c:-trim_c]
    v = v[trim_r:-trim_r, trim_c:-trim_c]

    return xi_mg, yi_mg, u, v


def plot_2d_trajectories(ax: plt.Axes, trajectories: np.ndarray):
    if trajectories is None:
        return

    for i in range(len(trajectories)):
        t = trajectories[i]
        # line, = ax.plot(t[:, 0], t[:, 1], color="w", alpha=0.3, linewidth=8)
        line, = ax.plot(t[:, 0], t[:, 1], alpha=0.9, color="white", linewidth=27, solid_capstyle='round', zorder=999)
        line, = ax.plot(t[:, 0], t[:, 1], alpha=0.5, linewidth=21, solid_capstyle='round', zorder=1000)
        ax.scatter(t[:, 0], t[:, 1], s=120, color="black", alpha=0.7, zorder=1002, edgecolors='lightgrey', linewidths=2 )
        # ax.scatter(t[:, 0], t[:, 1], s=100, color="white", alpha=0.7, zorder=1001)
        ax.scatter(t[0, 0], t[0, 1], alpha=0.95, s=500, marker="s", edgecolors='black', zorder=1003)
        ax.text(t[0, 0], t[0, 1], f"{i}", fontsize=18, ha='center', va='center', color='white', fontweight='bold', zorder=1004)


def plot_2d_flow_field(ax: plt.Axes, fig: plt.Figure, x, y, u, v, magnitude, trajectories, **kwargs):
    strm = streamplot_with_alpha(ax, x, y, u, v, density=1.5, linewidth=2.5, arrowsize=2.1, alpha=0.7, color=magnitude, **kwargs)
    plot_2d_trajectories(ax, trajectories)
    ax.set_title("Flow Field")

    fig.colorbar(strm.lines, ax=ax, label='Magnitude', location='right')

    for spine in ax.spines.values():
        spine.set_visible(False)


def calculate_magnitude(u: np.ndarray, v: np.ndarray):
    magnitude = np.sqrt(u ** 2 + v ** 2)
    return np.nan_to_num(magnitude, 0)


def plot_magnitude_filed(ax: plt.Axes, fig: plt.Figure, x, y, mag, **kwargs):
    cont, contf = contourplot(ax, x, y, mag, (0.9, 0.5), **kwargs)
    ax.set_title("Magnitude Field")

    fig.colorbar(contf, ax=ax, label='Magnitude', location='right')

    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_compare_vector_fields(fx: np.ndarray, pred_fx: np.ndarray, coords: np.ndarray,
                               trajectories: Optional[np.ndarray] = None,
                               pred_trajectories: Optional[np.ndarray] = None,
                               title_extension: Optional[str] = ""):
    factor = 1.0
    for i in range(10):
        try:
            fig, axs = plt.subplots(2, 2, figsize=(22, 20))

            x, y, u, v = interpolate_vector_field(fx, coords, scale=factor)
            mag = calculate_magnitude(u, v)

            _, _, pred_u, pred_v = interpolate_vector_field(pred_fx, coords, scale=factor)
            pred_mag = calculate_magnitude(pred_u, pred_v)

            # opts = {}
            # opts = {"norm": matplotlib.colors.Normalize(min(mag.min(), pred_mag.min()), max(mag.max(), pred_mag.max()))}
            opts = {"norm": matplotlib.colors.Normalize(mag.min(), mag.max())}

            plot_2d_flow_field(axs[0, 0], fig, x, y, u, v, mag, trajectories, **opts)
            plot_magnitude_filed(axs[0, 1], fig, x, y, mag, **opts)

            plot_2d_flow_field(axs[1, 0], fig, x, y, pred_u, pred_v, pred_mag, pred_trajectories, **opts)
            plot_magnitude_filed(axs[1, 1], fig, x, y, pred_mag, **opts)

            row_labels = ["Truth", "Prediction"]
            for i, label in enumerate(row_labels):
                fig.text(0.05, 0.75 - i * 0.45, label, va='center', ha='left', fontsize=22, weight='bold', rotation=90)

            fig.suptitle("Vector Field Visualization " + title_extension)

            return

        except ValueError as e:
            print("retrying", i)
            plt.close(fig)
            factor = 1 + torch.rand(1).item() * (i / 10)

    raise ValueError("Failed to plot after 10 retries") from e


# def save_all_figures_to_pdf(name: str):
#     with PdfPages(name) as pdf:
#         fig_nums = plt.get_fignums()
#         figs = [plt.figure(n) for n in fig_nums]
#         for fig in figs:
#             fig.savefig(pdf, format='pdf')
#             plt.close(fig)
#

def save_all_figures_to_pdf(name: str, close_figures: bool = True):
    with PdfPages(name) as pdf:
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]

        # Save all figures
        for fig in figs:
            pdf.savefig(fig)

        # Optionally close figures
        if close_figures:
            for fig in figs:
                plt.close(fig)



if __name__ == '__main__':
    path = Path(
        f"/home/teddev/PycharmProjects/pytorch_stuff/foundation_models_dynamical_systems/data_local/fim-data-more-path/0/data/processed/train/30k_drift_deg_3_ablation_studies/degree_and_monomial_survival_uniform/train/train_deg_2")
        # "/home/teddev/Downloads/val_deg_2")
    data = FimDataloader(path).load_dataset()
    assert data.is_loaded_and_dims_match
    data.filter_funcs_by_range((0, 1), idx_paths=(0, 8))

    fig, ax = plt.subplots(1, 1, figsize=(11, 10))

    x, y, u, v = interpolate_vector_field(data.fx[0], data.coordinates[0], scale=1.0)
    mag = calculate_magnitude(u, v)


    plot_2d_flow_field(ax, fig, x, y, u, v, mag, data.trajectories[0])
    # strm = streamplot_with_alpha(ax, x, y, u, v, alpha=1, density=0.9, arrowsize=2, linewidth=4, color=mag, zorder=1)
    # plot_2d_trajectories(ax, data.trajectories[0])
    # ax.set_title("Flow Field")
    #
    #
    # # cont, contf = contourplot(ax, x, y, mag, (0.3, 0.4), zorder=0)
    # # ax.set_title("Magnitude Field")
    #
    # # fig.colorbar(contf, ax=ax, label='Magnitude', location='right')
    #
    #
    # # fig.colorbar(strm.lines, ax=ax, label='Magnitude', location='right')
    #
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    plt.show()
