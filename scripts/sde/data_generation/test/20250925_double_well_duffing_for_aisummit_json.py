import json
from datetime import datetime
from pathlib import Path

import synthetic_systems_helpers

from fim.utils.sde.evaluation import NumpyEncoder


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    save_data_dir: Path = Path("/Users/patrickseifner/repos/FIM/data/processed/test/")
    subdir_label: str = "double_well_duffing_for_aisummit_json"

    # systems in table of paper
    path_to_data = Path("/Users/patrickseifner/repos/FIM/data/processed/test/20250925_double_well_duffing_for_aisummit")
    path_to_ksig_reference_data = Path("/Users/patrickseifner/repos/FIM/data/processed/test/20250925_double_well_duffing_for_aisummit")
    systems_to_load: list[str] = [
        "Duffing",
        "Double Well Const Diff",
        "Double Well State Dep Diff",
    ]

    subsampling_strides = [1]
    noises = [0.0]
    observation_lengths = [5000]
    sample_paths_count = 5
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
