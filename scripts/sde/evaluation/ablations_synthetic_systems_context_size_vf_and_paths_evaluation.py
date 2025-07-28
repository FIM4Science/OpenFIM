import json
from datetime import datetime
from pathlib import Path
from pprint import pprint

import optree
import synthetic_systems_vf_and_paths_evaluation as synthetic_systems_helpers
import torch

from fim import project_path
from fim.utils.sde.evaluation import ModelEvaluation, NumpyEncoder


def save_evaluations_as_jsons(all_evaluations: list[ModelEvaluation], evaluation_dir: Path):
    model_outputs = []
    for model_evaluation in all_evaluations:
        name, tau, noise, observations_length = model_evaluation.dataloader_id
        name = name.replace("_", " ")

        model_outputs.append(
            {
                "name": name,
                "tau": tau,
                "noise": noise,
                "observations_length": observations_length,
                "synthetic_paths": model_evaluation.results.get("sample_paths"),
                "drift_at_locations": model_evaluation.results["estimated_concepts"].drift,
                "diffusion_at_locations": model_evaluation.results["estimated_concepts"].diffusion,
            }
        )
    print("Model outputs to save")
    pprint(optree.tree_map(lambda x: x.shape if isinstance(x, torch.Tensor) else x, model_outputs))

    assert len(model_dicts) == 1, "Only works for a single model right now"

    def _check_finite(x):
        if isinstance(x, torch.Tensor):
            assert torch.torch.isfinite(x).all().item()

    _check_finite(model_outputs)
    model_outputs = optree.tree_map(lambda x: x.detach().to("cpu").numpy() if isinstance(x, torch.Tensor) else x, model_outputs)

    # Convert to JSON
    json_data = json.dumps(model_outputs, cls=NumpyEncoder)

    # Write JSON data to a file
    evaluation_dir.mkdir(exist_ok=True, parents=True)

    file: Path = evaluation_dir / "model_paths.json"
    with open(file, "w") as file:
        file.write(json_data)


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "ablations_synthetics_systems_context_size_vf_and_paths_evaluation"

    # How to name experiments
    experiment_descr = "fim_synthetic_systems_50_500_750_1000_2000_3000_4000_5000_50000_obs"

    model_dicts = {
        "FIM_fixed_softmax_dim_epoch_139": {
            "checkpoint_dir": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/600k_drift_deg_3_diff_deg_2_delta_tau_fixed_linear_attn_softmax_no_extra_normalization_and_fix_in_residual_layer_05-06-2300/checkpoints/",
            "checkpoint_name": "epoch-139",
        },
    }

    results_to_load: list[str] = [
        "/cephfs/users/seifner/repos/FIM/evaluations/ablations_double_well_context_size_vf_and_paths_evaluation/07291358_fim_synthetic_systems_50_500_5000_50000_obs/model_evaluations/"
    ]

    # systems in table of paper
    path_to_inference_data_json = Path(
        "/cephfs/users/seifner/repos/FIM/data/processed/test/20250729_synthetic_systems_data_for_ablations_length_50_500_750_1000_2000_3000_4000_5000_50000/systems_observations_and_locations.json",
    )

    systems_to_load: list[str] = [
        "Damped Linear",
        "Damped Cubic",
        "Duffing",
        "Glycosis",
        "Hopf",
        "Double Well",
        "Wang",
        "Syn Drift",
    ]

    taus = [0.002]
    noises = [0.0]
    observations_lengths = [50, 500, 750, 1000, 2000, 3000, 4000, 5000, 50000]
    sample_paths_count = 100
    sample_path_steps = 500
    dt = 0.002

    save_model_data = True
    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    data = json.load(open(path_to_inference_data_json, "r"))

    datasets: dict = {
        (system, tau, noise, observations_length): synthetic_systems_helpers.get_system_data(data, system, tau, noise, observations_length)
        for system in systems_to_load
        for tau in taus
        for noise in noises
        for observations_length in observations_lengths
    }

    all_evaluations = synthetic_systems_helpers.evaluate_all_models(
        model_dicts, datasets, dt, sample_paths_count, sample_path_steps, results_to_load
    )
    synthetic_systems_helpers.save_evaluations(all_evaluations, evaluation_dir / "model_evaluations")

    if save_model_data is True:
        save_evaluations_as_jsons(all_evaluations, evaluation_dir)
