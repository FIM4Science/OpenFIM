import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import synthetic_systems_helpers

from fim.utils.sde.evaluation import NumpyEncoder, save_fig


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    save_data_dir: Path = Path("/cephfs/users/seifner/repos/FIM/data/processed/test/")
    subdir_label: str = "synthetic_systems_data_as_jsons"

    # systems in table of paper
    path_to_data = Path(
        "/cephfs_projects/foundation_models/data/SDE/test/20250129_opper_svise_wang_long_dense_for_density_ablation_5_realizations_base_trajectories_for_icml_submission/"
    )
    path_to_ksig_reference_data = Path(
        "/cephfs_projects/foundation_models/data/SDE/test/20250129_opper_svise_wang_long_dense_for_density_ablation_5_realizations_KSIG_reference_paths_for_icml_submission/"
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
    observations_lengths = [5000]
    sample_paths_count = 100
    sample_path_length = 500
    dt = 0.002  # from data generation

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    time: str = str(datetime.now().strftime("%Y%m%d"))
    save_data_dir: Path = save_data_dir / (time + "_" + subdir_label)
    save_data_dir.mkdir(parents=True, exist_ok=True)

    reference_datasets, inference_datasets, ground_truth_drift_diffusion = synthetic_systems_helpers.prepare_synthetic_data(
        path_to_data,
        path_to_ksig_reference_data,
        systems_to_load,
        subsampling_strides,
        noises,
        observations_lengths,
        sample_paths_count,
        sample_path_length,
        dt,
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
                system_data = synthetic_systems_helpers.get_system_data(inference_datasets, system, tau, noise)

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
