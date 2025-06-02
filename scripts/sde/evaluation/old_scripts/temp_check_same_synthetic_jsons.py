import json
from pathlib import Path

import numpy as np


if __name__ == "__main__":
    # compare generated data
    old_data_dir = Path(
        "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250129_coarse_synthetic_systems_5000_points_data/"
    )
    old_data_paths_json = old_data_dir / "ksig_reference_paths.json"
    old_data_vector_fields_json = old_data_dir / "ground_truth_drift_diffusion.json"
    old_data_inference_json = old_data_dir / "systems_coarse_observations.json"

    new_data_dir = Path("/cephfs/users/seifner/repos/FIM/data/processed/test/20250320_synthetic_systems_data_as_jsons/")
    new_data_paths_json = new_data_dir / "systems_ksig_reference_paths.json"
    new_data_vector_fields_json = new_data_dir / "systems_ground_truth_drift_diffusion.json"
    new_data_inference_json = new_data_dir / "systems_observations_and_locations.json"

    old_data_paths: list = json.load(open(old_data_paths_json, "r"))
    old_data_vector_fields: list = json.load(open(old_data_vector_fields_json, "r"))
    old_data_inference: list = json.load(open(old_data_inference_json, "r"))

    new_data_paths: list = json.load(open(new_data_paths_json, "r"))
    new_data_vector_fields: list = json.load(open(new_data_vector_fields_json, "r"))
    new_data_inference: list = json.load(open(new_data_inference_json, "r"))

    assert old_data_paths == new_data_paths
    assert old_data_vector_fields == new_data_vector_fields
    assert old_data_inference == new_data_inference

    # compare model output (don't have to be exactly the same, probably because of some input precision somewhere)
    new_model_output_path = Path(
        "/home/seifner/repos/FIM/evaluations/synthetic_systems_vf_and_paths_evaluation/03202154_develop_new_script/model_paths.json"
    )

    old_model_reevaluated_output_path = Path(
        "/home/seifner/repos/FIM/evaluations/synthetic_systems_vf_and_paths_evaluation/03181419_checkpoint_before_script_change_paper_model/data_jsons/model_paths.json"
    )

    old_model_output_path = Path(
        "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250129_coarse_synthetic_systems_5000_points_data/20M_trained_even_longer_synthetic_paths.json"
    )

    new_model_output: list = json.load(open(new_model_output_path, "r"))
    old_model_reevaluated_output: list = json.load(open(old_model_reevaluated_output_path, "r"))
    old_model_output: list = json.load(open(old_model_output_path, "r"))

    for i in range(len(new_model_output)):
        new_model_output[i]["drift_at_locations"] = np.array(new_model_output[i]["drift_at_locations"])
        new_model_output[i]["diffusion_at_locations"] = np.array(new_model_output[i]["diffusion_at_locations"])
        new_model_output[i]["synthetic_paths"] = np.array(new_model_output[i]["synthetic_paths"])

        old_model_reevaluated_output[i]["drift_at_locations"] = np.array(old_model_reevaluated_output[i]["drift_at_locations"])
        old_model_reevaluated_output[i]["diffusion_at_locations"] = np.array(old_model_reevaluated_output[i]["diffusion_at_locations"])
        old_model_reevaluated_output[i]["synthetic_paths"] = np.array(old_model_reevaluated_output[i]["synthetic_paths"])

        old_model_output[i]["drift_at_locations"] = np.array(old_model_output[i]["drift_at_locations"])
        old_model_output[i]["diffusion_at_locations"] = np.array(old_model_output[i]["diffusion_at_locations"])
        old_model_output[i]["synthetic_paths"] = np.array(old_model_output[i]["synthetic_paths"])

    for i in range(len(new_model_output)):
        print("System: ", old_model_output[i]["name"], new_model_output[i]["name"])
        print("Tau: ", old_model_output[i]["tau"], new_model_output[i]["tau"])
        print(
            "Drift at locations: ",
            old_model_output[i]["drift_at_locations"][0, 100],
            old_model_reevaluated_output[i]["drift_at_locations"][0, 100],
            new_model_output[i]["drift_at_locations"][0, 100],
        )
        print(
            "Max. abs. difference: ", np.amax(np.abs(old_model_output[i]["drift_at_locations"] - new_model_output[i]["drift_at_locations"]))
        )
        print(
            "Diffusion at locations: ",
            old_model_output[i]["diffusion_at_locations"][0, 100],
            old_model_reevaluated_output[i]["diffusion_at_locations"][0, 100],
            new_model_output[i]["diffusion_at_locations"][0, 100],
        )
        print(
            "Max. abs. difference: ",
            np.amax(np.abs(old_model_output[i]["diffusion_at_locations"] - new_model_output[i]["diffusion_at_locations"])),
        )
        print(
            "Paths: ",
            old_model_output[i]["synthetic_paths"][0][0, 100],
            old_model_reevaluated_output[i]["synthetic_paths"][0][0, 100],
            new_model_output[i]["synthetic_paths"][0][0, 100],
        )
        print("\n")
