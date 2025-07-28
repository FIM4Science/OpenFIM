import json
from datetime import datetime
from pathlib import Path

import synthetic_systems_helpers

from fim.utils.sde.evaluation import NumpyEncoder


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    save_data_dir: Path = Path("/cephfs/users/seifner/repos/FIM/data/processed/test/")
    subdir_label: str = "synthetic_systems_data_for_ablations_length_50_500_750_1000_2000_3000_4000_5000_50000"

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

    subsampling_strides = [1]
    noises = [0.0]
    observation_lengths = [50, 500, 750, 1000, 2000, 3000, 4000, 5000, 50000]
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
        observation_lengths,
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
