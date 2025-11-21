import json
from copy import copy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import optree

from fim import project_path
from fim.utils.sde.evaluation import save_fig


def load_system_results(all_results: list[dict], system_name: str, tau: float, exp: int, noise: float, apply_sqrt_to_diffusion: bool):
    """
    Extract results from one model on one system, defined by (system_name, tau, exp), from a list[dict] with all results from that model.

    Args:
        all_results (list[dict]): Loaded from some model_results.json.
        (system_name, tau, exp, noise): defines the results to extract.
        apply_sqrt_to_diffusion (bool): Some models (SparseGP, BISDE) return diffusion value under sqrt. Adapt it here for comparison.

    Return:
        all_exp_results (dict[np.ndarray]): Keys: locations, drift_at_locations, diffusion_at_locations, synthetic_paths.
    """
    print(system_name, tau, exp, noise)
    results_with_system = [result for result in all_results if result["name"] == system_name]
    results_with_tau = [result for result in results_with_system if result["tau"] == tau]
    results_with_noise = [result for result in results_with_tau if result["noise"] == noise]
    assert len(results_with_noise) == 1, f"Failed with {system_name=}, {tau=}, {exp=}, {noise=}. Got {len(results_with_noise)}"

    all_exp_results = results_with_noise[0]
    all_exp_results.pop("name", None)
    all_exp_results.pop("tau", None)
    all_exp_results.pop("noise", None)
    all_exp_results.pop("equations", None)

    all_exp_results = {k: np.array(v) for k, v in all_exp_results.items()}
    all_exp_results = optree.tree_map(lambda x: x[exp], all_exp_results)

    if apply_sqrt_to_diffusion is True:
        all_exp_results["diffusion_at_locations"] = np.sqrt(np.clip(all_exp_results["diffusion_at_locations"], a_min=0, a_max=np.inf))

    return all_exp_results


def plot_2D_paths(ax, values: np.ndarray, color: str, linestyle: str, linewidth: float, label: str):
    """
    Plot multiple paths into an axis.

    Args:
        values: Paths to plot. Shape: [P, T, 2]
        plot configs: ...
    """
    P = values.shape[0]
    for path in range(P):
        ax.plot(
            values[path, :, 0].reshape(-1),
            values[path, :, 1].reshape(-1),
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label if path == 0 else None,
        )


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    global_description = "poster_figure_damped_linear_paths"
    current_description = "develop"

    # data to load
    path_to_ksig_paths_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250325_synthetic_systems_5000_points_with_additive_noise/data/systems_ksig_reference_paths.json"
    )

    # results to plot
    # systems_to_plot = ["Double Well", "Wang", "Damped Linear", "Damped Cubic", "Duffing", "Glycosis", "Hopf"]
    system_to_plot = "Damped Linear"

    experiment_to_plot = 0  # we repeat each experiment 5 times, but reference paths stay the same, so choose first copy

    # paths setup
    plot_num_paths = 10

    # general plot config
    linewidth_vf = 1
    # linewidth_paths = 0.2
    loc_size_per_dim = 32
    loc_stride_length = 4

    gt_plot_config = {
        "color": "black",
        "linestyle": "solid",
        "label": "Ground-Truth",
        "linewidth_paths": 0.15,
    }

    tick_base = {
        "drift": {"x": 1, "y": 1},
        "diffusion": {"x": 1, "y": 1},
        "paths": {"x": 2, "y": 2},
    }

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / global_description / (time + "_" + current_description)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # load ground-truth paths
    all_paths: list[dict] = json.load(open(path_to_ksig_paths_json))
    all_systems_paths: dict = {all_paths[i].get("name"): copy(all_paths[i]) for i in range(len(all_paths))}
    system_paths = {k: v for k, v in all_systems_paths.items() if k == system_to_plot}
    system_paths = optree.tree_map(lambda x: np.array(x), system_paths, is_leaf=lambda x: isinstance(x, list))

    # create figure with 3 subplots per system: drift, diffusion, paths
    fig, axs = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(1.5, 1.54),
        dpi=300,
    )

    # configure axes
    [x.set_linewidth(0.3) for x in axs.spines.values()]
    axs.tick_params(axis="both", direction="out", labelsize=4, width=0.5, length=2)

    axs.xaxis.set_major_locator(plticker.MultipleLocator(base=tick_base["paths"]["x"]))
    axs.yaxis.set_major_locator(plticker.MultipleLocator(base=tick_base["paths"]["y"]))

    system_paths = system_paths[system_to_plot]
    paths = system_paths["real_paths"][experiment_to_plot]

    plot_2D_paths(
        axs,
        paths,
        color=gt_plot_config["color"],
        linewidth=gt_plot_config["linewidth_paths"],
        linestyle="solid",  # linestyle,
        label=None,
    )

    # save
    save_dir: Path = evaluation_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = "samples_of_single_system"
    save_fig(fig, save_dir, file_name)
