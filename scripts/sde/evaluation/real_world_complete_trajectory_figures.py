import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fim import project_path
from fim.utils.evaluation_sde import save_fig


def get_model_results_on_dataset(models_results: dict[list], dataset: str, apply_sqrt_to_diffusion: list) -> dict[dict]:
    """
    Extracts results of all models on a particular dataset.
    Converts loaded lists into arrays.
    If inference was on log-transformed data, reverse the transform.
    """
    models_results_of_dataset = {}

    for model_label, results in models_results.items():
        # should extract exactly one result for each model
        results_of_dataset = [r for r in results if r["name"] == dataset]
        assert len(results_of_dataset) == 1
        results = results_of_dataset[0]

        # convert to arrays
        results["locations"] = np.array(results["locations"])
        results["synthetic_paths"] = np.array(results["synthetic_paths"])
        results["drift_at_locations"] = np.array(results["drift_at_locations"])
        results["diffusion_at_locations"] = np.array(results["diffusion_at_locations"])

        if model_label in apply_sqrt_to_diffusion:
            results["diffusion_at_locations"] = np.sqrt(np.max(results["diffusion_at_locations"]))

        # reverse log-transform
        if results.get("transform") == "log":
            results["locations"] = np.exp(results["locations"])
            results["synthetic_paths"] = np.exp(results["synthetic_paths"])

            # ito formula applied to f(x) =df(x) = ddf(x) = exp(x)
            results["drift_at_locations"] = results["locations"] * (
                results["drift_at_locations"] + 1 / 2 * results["diffusion_at_locations"] ** 2
            )
            results["diffusion_at_locations"] = results["locations"] * results["diffusion_at_locations"]

        models_results_of_dataset.update({model_label: results})

    return models_results_of_dataset


def get_reference_data(data_paths: list, dataset: str) -> dict:
    """
    From all data, extract the dataset.
    Converts loaded lists into arrays.
    If log-transformed data, reverse the transform.
    """
    # should extract exactly one set of data
    reference_data = [d for d in data_paths if d["name"] == dataset]
    assert len(reference_data) == 1
    reference_data = reference_data[0]

    # convert to arrays
    reference_data["locations"] = np.array(reference_data["locations"])
    reference_data["obs_times"] = np.array(reference_data["obs_times"])
    reference_data["obs_values"] = np.array(reference_data["obs_values"])

    # reverse log-transform
    if reference_data.get("transform") == "log":
        reference_data["locations"] = np.exp(reference_data["locations"])
        reference_data["obs_values"] = np.exp(reference_data["obs_values"])

    return reference_data


def get_sample_paths_figure(
    reference_data: dict,
    models_results: dict[list],
    models_color: dict,
    title: str = None,
    figsize=(5, 5),
    alpha=0.75,
    linewidth=0.5,
    reference_label="Reference Paths",
):
    """
    Plot (set of) reference sample paths against paths from all models.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    obs_times = reference_data.get("obs_times")  # [1, P, T, 1]
    ref_obs_values = reference_data.get("obs_values")  # [1, P, T, 1]
    assert obs_times.shape == ref_obs_values.shape
    assert obs_times.ndim == 4

    _, P, T, _ = obs_times.shape

    P = ref_obs_values.shape[1]

    for p in range(P):
        ax.plot(obs_times[:, p].squeeze(), ref_obs_values[:, p].squeeze(), label=reference_label, color="black", linewidth=linewidth)

    for label, result in models_results.items():
        color = models_color[label]
        synthetic_paths = result.get("synthetic_paths")  # [1, P, T, 1]

        obs_times = obs_times.squeeze()

        synthetic_paths = synthetic_paths[..., :T, :]
        synthetic_paths = synthetic_paths.squeeze()

        assert obs_times.shape == synthetic_paths.shape, f"obs_times: {obs_times.shape}, synthetic_paths: {synthetic_paths.shape}"

        obs_times = obs_times.reshape(P, T)
        synthetic_paths = synthetic_paths.reshape(P, T)

        ax.plot(obs_times[p].squeeze(), synthetic_paths[p].squeeze(), label=label, color=color, linewidth=linewidth, alpha=alpha)

    ax.legend()
    ax.set_title(title)

    return fig


def get_vector_fields_figure(
    models_results: dict[list],
    models_color: dict,
    title=None,
    figsize=(5, 5),
    dpi=300,
    linewidth=2,
    alpha=0.75,
):
    """
    Plot infered vector fields from all models.
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    ax[0].set_title("Drift")
    ax[1].set_title("Diffusion")

    for label, result in models_results.items():
        color = models_color[label]
        locations = result.get("locations")  # [1, L, 1]
        drift = result.get("drift_at_locations")  # [1, L, 1]
        diffusion = result.get("diffusion_at_locations")  # [1, L, 1]

        assert locations.shape == drift.shape
        assert drift.shape == diffusion.shape

        ax[0].plot(locations.squeeze(), drift.squeeze(), label=label, color=color, linewidth=linewidth, alpha=alpha)
        ax[1].plot(locations.squeeze(), diffusion.squeeze(), label=label, color=color, linewidth=linewidth, alpha=alpha)

        ax[0].set_xlabel("Drift")
        ax[1].set_xlabel("Diffusion")

    ax[0].legend()

    fig.suptitle(title)

    return fig


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "real_world_complete_trajectory_figures"

    # How to name experiments
    experiment_descr = "fim_checkpoints_vs_BISDE"

    complete_trajectory_data_json = Path(
        "/home/seifner/repos/FIM/data/processed/test/20250506_real_world_with_5_fold_cross_validation/complete_paths.json"
    )

    datasets_to_evaluate: list[str] = [
        "wind",
        "oil",
        "fb",
        "tsla",
    ]

    models_jsons = {
        "FIM (05-03-2033)": Path(
            "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/real_world_complete_trajectory_vf_and_paths_evaluation/05101118_fim_fixed_attn_fixed_softmax_05-03-2033/model_paths.json",
        ),
        "FIM (05-06-2300)": Path(
            "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/real_world_complete_trajectory_vf_and_paths_evaluation/05101246_fim_fixed_attn_fixed_softmax_05-06-2300/model_paths.json",
        ),
        "BISDE(20250510, BISDE Library Functions)": Path(
            "/cephfs_projects/foundation_models/data/SDE/evaluation/20250506_real_world_with_5_fold_cross_validation/20250510_bisde_reeval_on_complete_trajectory_function_library_from_bisde_paper/bisde_real_world_reeval_results.json"
        ),
        "BISDE(20250510, (u^(0,..,3), exp(u), sin(u)) Library Functions)": Path(
            "/cephfs_projects/foundation_models/data/SDE/evaluation/20250506_real_world_with_5_fold_cross_validation/20250510_bisde_reeval_on_complete_trajectory_function_library_with_exps_and_sins/bisde_real_world_reeval_our_basis_results.json"
        ),
    }

    models_color = {
        "FIM (05-03-2033)": "#0072B2",
        "FIM (05-06-2300)": "blue",
        "BISDE(20250510, BISDE Library Functions)": "#CC79A7",
        "BISDE(20250510, (u^(0,..,3), exp(u), sin(u)) Library Functions)": "magenta",
    }

    apply_sqrt_to_diffusion = [
        "BISDE(20250510, BISDE Library Functions)",
        "BISDE(20250510, (u^(0,..,3), exp(u), sin(u)) Library Functions)",
    ]

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # load models results from jsons
    models_results: dict[list] = {model_label: json.load(open(model_json, "r")) for model_label, model_json in models_jsons.items()}

    # load complete trajectory data
    data_paths: list[dict] = json.load(open(complete_trajectory_data_json, "r"))

    # add locations to model_results
    def get_locations_of_dataset(data_paths: list[dict], dataset: str):
        data_from_dataset = [d for d in data_paths if d["name"] == dataset]
        assert len(data_from_dataset) == 1
        data_from_dataset = data_from_dataset[0]

        return data_from_dataset["locations"]

    models_results = {
        model_label: [result | {"locations": get_locations_of_dataset(deepcopy(data_paths), result["name"])} for result in model_results]
        for model_label, model_results in models_results.items()
    }

    for dataset in datasets_to_evaluate:
        reference_data = get_reference_data(deepcopy(data_paths), dataset)
        models_results_of_dataset = get_model_results_on_dataset(deepcopy(models_results), dataset, apply_sqrt_to_diffusion)

        # sample paths figure
        fig = get_sample_paths_figure(reference_data, models_results_of_dataset, models_color, title=dataset)

        save_dir: Path = evaluation_dir / "figures_sample_path"
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = dataset
        save_fig(fig, save_dir, file_name)

        plt.close(fig)

        # vector fields figure
        fig = get_vector_fields_figure(models_results_of_dataset, models_color, title=dataset)

        save_dir: Path = evaluation_dir / "figures_vector_fields"
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = dataset
        save_fig(fig, save_dir, file_name)
