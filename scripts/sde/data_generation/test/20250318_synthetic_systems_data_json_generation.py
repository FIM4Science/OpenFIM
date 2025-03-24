import json
from datetime import datetime
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import optree
import torch

from fim.data.datasets import get_file_paths
from fim.data.utils import load_file, load_h5
from fim.utils.evaluation_sde import NumpyEncoder, save_fig


def get_inference_data(
    system_data_dir: Path, initial_states: torch.Tensor, stride: int, noise: float, dt: float, target_length: int
) -> dict:
    """
    Subsample data from a large fine grid by strides. truncate to a target length.
    i.e. stride does not imply fewer points in trajectories.

    Args:
        system_data_dir (Path): Absolute path to directory containing pre-generated, long, fine-grid trajectories of one system.
        initial states (torch.Tensor): Set of initial states for path generation.
        stride (int): Stridelength to subsample the fine-grid trajectories with.
        noise (float): Relative additive noise added to observations of trajectories.
        dt (float): Time delta of pre-generated trajectories.
        target_length (int): Length of (subsampled and truncated) trajectories to return.

    Returns:
        data (dict): Subsampled and truncated trajectories of system.
                     Keys: obs_values, obs_times, tau, locations, drift_at_locations, diffusion_at_locations
    """
    # load and pad data to max dim
    files_to_load = {
        "obs_times": "obs_times.h5",
        "obs_values": "obs_values.h5",
        "locations": "locations.h5",
        "drift_at_locations": "drift_at_locations.h5",
        "diffusion_at_locations": "diffusion_at_locations.h5",
    }

    # load data from paths
    file_paths: dict[str, list[Path]] = get_file_paths(system_data_dir, files_to_load)
    data: dict[str, list[torch.Tensor]] = torch.utils._pytree.tree_map(load_file, file_paths)

    # load_file returns list of one tensor
    data: dict[str, torch.Tensor] = {k: v[0] for k, v in data.items()}

    # subsample observations by strides of provided length
    data["obs_values"] = data["obs_values"][:, :, ::stride, :]
    data["obs_times"] = data["obs_times"][:, :, ::stride, :]

    # truncate observations to target length
    data["obs_values"] = data["obs_values"][:, :, :target_length, :]
    data["obs_times"] = data["obs_times"][:, :, :target_length, :]

    # additive relative noise
    obs_range = 1 / 2 * (torch.amax(data["obs_values"], dim=-2, keepdim=True) - torch.amin(data["obs_values"], dim=-2, keepdim=True))
    data["obs_values"] = data["obs_values"] + noise * obs_range * torch.randn_like(data["obs_values"])
    data["noise"] = noise

    # time between two observations, based on data generating dt
    data["tau"] = stride * dt

    # initial states for path generation
    data["initial_states"] = initial_states

    return data


def get_ksig_reference_paths(system_data_dir: Path, sample_paths_count: int, sample_path_length: int) -> torch.Tensor:
    """
    Truncate and chunck pre-generated, long, fine-grid sample paths, into sample paths of sample_path_length.

    Args:
        system_data_dir (Path): Absolute path to directory containing pre-generated, long, fine-grid trajectories of one system.
        sample_paths_count (int): Number of sample paths to return.
        sample_path_length (int): Length of each path to return.

    Returns:
        paths (torch.Tensor): Shape [E, sample_paths_count, sample_path_length, D]
    """

    # load file
    paths = load_h5(system_data_dir / "obs_values.h5")  # [E, 1, L, D]

    # truncate one long path into required length
    paths = paths[:, :, : sample_paths_count * sample_path_length, :]  # [E, 1, sample_paths_count * sample_path_length, D]

    # reshape long path into sample_paths_count paths of lenght sample_path_length
    E, _, _, D = paths.shape
    paths = paths.reshape(E, sample_paths_count, sample_path_length, D)

    return paths


def get_system_data(all_systems_data: list[dict], system: str, tau: float, noise: float) -> dict:
    """
     From a list of all systems data, extract data of system with tau inter-observation times and relative noise.

    Args:
        all_systems_data (list[dict]): List of system data, each of which is a dict.
        system (str): Name of system data to extract.
        tau (float): Inter-observation time of data to extract.
        noise (float): Relative additive noise added to observations of trajectories.

    Return:
        data_of_system (dict): Keys: name, tau, obs_times, obs_values, locations, initial_states
    """
    data_of_system = [d for d in all_systems_data if (d["name"] == system and d["tau"] == tau and d["noise"] == noise)]

    # should contain exactly one data for each system and tau
    if len(data_of_system) == 1:
        data_of_system = data_of_system[0]

        # rename for easier inference for model
        data_of_system["obs_values"] = data_of_system.pop("observations")

        # add obs times based_on_tau
        B, M, T, _ = data_of_system["obs_values"].shape
        data_of_system["obs_times"] = tau * torch.ones((B, M, 1, 1)) * torch.arange(T).reshape(1, 1, T, 1)

        return data_of_system

    elif len(data_of_system) == 0:
        raise ValueError(f"Could not find data of system {system} and tau {tau} and noise perc {noise}.")

    else:
        raise ValueError(f"Found {len(data_of_system)} sets of data for system {system} and tau {tau}.")


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    save_data_dir: Path = Path("/cephfs/users/seifner/repos/FIM/data/processed/test/")
    subdir_label: str = "synthetic_systems_data_as_jsons"

    # systems in table of paper
    path_to_data = Path(
        "/home/seifner/repos/FIM/data/processed/test/20250129_opper_svise_wang_long_dense_for_density_ablation_5_realizations/"
    )
    path_to_ksig_reference_data = Path(
        "/home/seifner/repos/FIM/data/processed/test/20250129_opper_svise_wang_long_dense_for_density_ablation_5_realizations_KSIG_reference_paths"
    )
    systems_to_load: list[str] = [
        "Damped_Linear",
        "Damped_Cubic",
        "Duffing",
        "Glycosis",
        "Hopf",
        "Double_Well",
        "Wang",
        "Syn_Drift",
    ]

    subsampling_strides = [1, 5, 10, 100]
    noises = [0.0, 0.05, 0.1]
    target_length = 5000
    sample_paths_count = 100
    sample_path_length = 500
    dt = 0.002  # from data generation

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    time: str = str(datetime.now().strftime("%Y%m%d"))
    save_data_dir: Path = save_data_dir / (time + "_" + subdir_label)
    save_data_dir.mkdir(parents=True, exist_ok=True)

    # load all KSIG reference paths
    ksig_ref_obs_values = {system: load_h5(path_to_ksig_reference_data / system / "obs_values.h5") for system in systems_to_load}

    # truncate and reshape into sample_paths_count trajectories of length sample_path_length
    ksig_ref_obs_values = {
        system: ksig_values[:, :, : sample_paths_count * sample_path_length, :] for system, ksig_values in ksig_ref_obs_values.items()
    }
    ksig_ref_obs_values = {
        system: ksig_values.reshape(-1, sample_paths_count, sample_path_length, ksig_values.shape[-1])
        for system, ksig_values in ksig_ref_obs_values.items()
    }

    # load all KSIG reference paths
    ksig_ref_obs_values = {
        system: get_ksig_reference_paths(path_to_ksig_reference_data / system, sample_paths_count, sample_path_length)
        for system in systems_to_load
    }

    # reference paths for KSIG comparison
    reference_datasets = []
    for system, data in ksig_ref_obs_values.items():
        reference_datasets.append(
            {
                "name": system.replace("_", " "),
                "real_paths": data,
            }
        )

    print("Reference paths to save")
    pprint(optree.tree_map(lambda x: x.shape if isinstance(x, torch.Tensor) else x, reference_datasets))

    # extract data for inference of models for all systems and delta tau (strides)
    ksig_initial_states: dict = {system: paths[:, :, 0, :] for system, paths in ksig_ref_obs_values.items()}
    datasets = {
        (system, stride, noise, target_length): get_inference_data(
            path_to_data / system, ksig_initial_states[system], stride, noise, dt, target_length
        )
        for system in systems_to_load
        for stride in subsampling_strides
        for noise in noises
    }

    # inference dataset (including input paths, locations for vector field evaluation and initial states for sampling paths)
    inference_datasets = []
    for data_key, data in datasets.items():
        inference_datasets.append(
            {
                "name": data_key[0].replace("_", " "),
                "tau": data["tau"],
                "noise": data["noise"],
                "observations": data["obs_values"],
                "locations": data["locations"],
                "initial_states": data["initial_states"],
            }
        )

    print("Inference datasets to save")
    pprint(optree.tree_map(lambda x: x.shape if isinstance(x, torch.Tensor) else x, inference_datasets))

    # ground truth vector fields for comparison
    ground_truth_drift_diffusion = []
    for data_key, data in datasets.items():
        system, stride, noise, _ = data_key
        tau = data["tau"]
        if tau == 0.002 and stride == 1 and noise == 0.0:  # save only once
            ground_truth_drift_diffusion.append(
                {
                    "name": system.replace("_", " "),
                    "locations": data["locations"],
                    "drift_at_locations": data["drift_at_locations"],
                    "diffusion_at_locations": data["diffusion_at_locations"],
                }
            )

    print("Ground truth drift and diffusion to save")
    pprint(optree.tree_map(lambda x: x.shape if isinstance(x, torch.Tensor) else x, ground_truth_drift_diffusion))

    def _check_finite(x):
        if isinstance(x, torch.Tensor):
            assert torch.torch.isfinite(x).all().item()

    # save data in jsons
    optree.tree_map(_check_finite, (reference_datasets, inference_datasets), namespace="fimsde")

    reference_datasets, inference_datasets, ground_truth_drift_diffusion = optree.tree_map(
        lambda x: x.detach().to("cpu").numpy() if isinstance(x, torch.Tensor) else x,
        (reference_datasets, inference_datasets, ground_truth_drift_diffusion),
    )

    for data, filename in zip(
        [reference_datasets, inference_datasets, ground_truth_drift_diffusion],
        ["systems_ksig_reference_paths.json", "systems_observations_and_locations.json", "systems_ground_truth_drift_diffusion.json"],
    ):
        # Convert to JSON
        json_data = json.dumps(data, cls=NumpyEncoder)

        file: Path = save_data_dir / filename
        with open(file, "w") as file:
            file.write(json_data)

    # create plot with all noise levels
    exp = 0
    linewidth = 0.25

    colors = ["black", "r", "g"]
    taus = [stride * dt for stride in subsampling_strides]

    nrows = len(systems_to_load)
    ncols = len(taus)

    fig, axs = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows), tight_layout=True)

    for row in range(nrows):
        system = systems_to_load[row]
        axs[row, 0].set_ylabel(system)
        for col in range(ncols):
            tau = taus[col]

            if row == 0:
                axs[0, col].set_title(f"{tau=}")

            for i, (noise, color) in enumerate(zip(noises, colors)):
                system = system.replace("_", " ")
                system_data = get_system_data(inference_datasets, system, tau, noise)

                obs_values = system_data["obs_values"][exp].squeeze(0)  # [T, D]
                obs_times = system_data["obs_times"][exp].squeeze(0)  # [T, 1]
                D = obs_values.shape[-1]

                z_value = len(noises) - i
                if D == 2:
                    axs[row, col].plot(obs_values[:, 0], obs_values[:, 1], label=noise, color=color, linewidth=linewidth, zorder=z_value)

                if D == 1:
                    axs[row, col].plot(obs_times, obs_values, label=noise, color=color, linewidth=linewidth, zorder=z_value)

    axs[0, 0].legend()
    save_fig(fig, save_data_dir, "noise_comparison")
