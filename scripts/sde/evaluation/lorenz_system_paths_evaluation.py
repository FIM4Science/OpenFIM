import itertools
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import optree
import torch
from torch import Tensor
from tqdm import tqdm

from fim import project_path
from fim.data.datasets import JsonSDEDataset
from fim.models.blocks import AModel
from fim.sampling.sde_path_samplers import fimsde_sample_paths_on_masked_grid
from fim.utils.sde.evaluation import (
    DataLoaderMap,
    ModelEvaluation,
    ModelMap,
    NumpyEncoder,
    dataloader_map_from_dict,
    load_evaluations,
    model_map_from_dict,
    save_evaluations,
)


def evaluate_latentsde_on_lorenz(
    model: AModel,
    reference_obs_times: Tensor,
    reference_obs_values: Tensor,
    train_data_label: str,
    inference_initial_states: Tensor,
    inference_locations: Tensor,
    inference_data_label: str,
    device: Optional[str] = None,
) -> list:
    """
    Sample paths and prior vector fields from a Latent SDE model, potentially with multiple sampling strategies.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    reference_obs_times.to(device)
    reference_obs_values.to(device)
    inference_locations.to(device)
    inference_initial_states.to(device)

    results = []

    # vector fields at locations, if possible
    if model.config.latent_size == 3 and model.config.learn_projection is False:
        drift_at_locations = model.h(None, inference_locations)  # [G, 3]
        diffusion_at_locations = model.g(None, inference_locations)  # [G, 3]

    else:
        drift_at_locations = None
        diffusion_at_locations = None

    # prior equation, posterior condition
    ctx, obs_times, _ = model.encode_inputs(reference_obs_times, reference_obs_values)
    posterior_initial_states, _, _ = model.sample_posterior_initial_condition(ctx[0])
    _, paths_post_init_cond = model.sample_from_prior_equation(posterior_initial_states, obs_times)
    results.append(
        {
            "synthetic_path": paths_post_init_cond,
            "drift_at_locations": drift_at_locations,
            "diffusion_at_locations": diffusion_at_locations,
            "sampling_label": "Prior Eq. in Latent Space",
            "initial_state_label": "Posterior Init. Cond. from Ref. Set.",
        }
    )

    # prior equation, prior initial condition
    prior_initial_states = model.sample_prior_initial_condition(reference_obs_values.shape[0])
    _, paths_prior_init_cond = model.sample_from_prior_equation(prior_initial_states, obs_times)
    results.append(
        {
            "synthetic_path": paths_prior_init_cond,
            "drift_at_locations": drift_at_locations,
            "diffusion_at_locations": diffusion_at_locations,
            "sampling_label": "Prior Eq. in Latent Space",
            "initial_state_label": "Prior Init. Cond.",
        }
    )

    # if model does not have output projection and latent size == 3, use reference initial states
    if model.config.learn_projection is False and model.config.latent_size == 3:
        _, paths_ref_init_states = model.sample_from_prior_equation(inference_initial_states, obs_times)
        results.append(
            {
                "synthetic_path": paths_ref_init_states,
                "drift_at_locations": drift_at_locations,
                "diffusion_at_locations": diffusion_at_locations,
                "sampling_label": "Prior Eq. in Data Space",
                "initial_state_label": "Reference Initial States",
            }
        )

    # add some model hyperparameters for monitoring
    for result in results:
        result.update(
            {
                "learn_projection": model.config.learn_projection,
                "latent_size": model.config.latent_size,
                "hidden_size": model.config.hidden_size,
                "context_size": model.config.context_size,
                "activation": model.config.activation,
                "train_data_label": train_data_label,
                "inference_data_label": inference_data_label,
            }
        )

    results = optree.tree_map(lambda x: x.to("cpu") if isinstance(x, torch.Tensor) else x, results)
    reference_obs_times.to("cpu")
    reference_obs_values.to("cpu")
    inference_initial_states.to("cpu")

    model.to("cpu")

    return results


def evaluate_fim_on_lorenz(
    model: AModel,
    train_obs_times: Tensor,
    train_obs_values: Tensor,
    train_data_label: str,
    inference_grid: Tensor,
    inference_initial_states: Tensor,
    inference_locations: Tensor,
    inference_data_label: str,
    device: Optional[str] = None,
):
    """
    Sample paths and vector fields from the equation inferred from the train dataset, with given initial states.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    dataset = {
        "obs_times": train_obs_times.unsqueeze(0),
        "obs_values": train_obs_values.unsqueeze(0),
        "locations": inference_locations.unsqueeze(0),
    }
    inference_grid = inference_grid.unsqueeze(0)
    inference_initial_states = inference_initial_states.unsqueeze(0)

    sample_paths, _ = fimsde_sample_paths_on_masked_grid(
        model,
        dataset,
        initial_states=inference_initial_states,
        grid=inference_grid,
        mask=torch.ones_like(inference_grid),
        solver_granularity=10,
    )  # [1, num_initial_states, T, 3]

    with torch.no_grad():
        estimated_concepts = model(dataset, training=False, return_losses=False)
        drift_at_locations = estimated_concepts.drift.to("cpu")
        diffusion_at_locations = estimated_concepts.diffusion.to("cpu")

    results = {
        "synthetic_path": sample_paths.squeeze(),
        "drift_at_locations": drift_at_locations.squeeze(0),
        "diffusion_at_locations": diffusion_at_locations.squeeze(0),
        "sampling_label": "Eq. from train set",
        "train_data_label": train_data_label,
        "inference_data_label": inference_data_label,
    }

    results = optree.tree_map(lambda x: x.to("cpu") if isinstance(x, torch.Tensor) else x, results)
    model.to("cpu")

    return results


def run_lorenz_evaluation(
    to_evaluate: list[ModelEvaluation],
    model_map: ModelMap,
    dataloader_map: DataLoaderMap,
    device: Optional[str] = None,
) -> list[ModelEvaluation]:
    """
    Evaluate model on a lorenz dataset.

    Args:
        evaluations (list[ModelEvaluation]): List of evaluations to run, specified by their model_id and dataloader_id
        model / dataloader map: Returning required dataloaders

    Return:
        evaluations (list[ModelEvaluation]): Input evaluations.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluations_with_results: list[ModelEvaluation] = []

    for evaluation in (pbar := tqdm(to_evaluate, total=len(to_evaluate), leave=False)):
        pbar.set_description(f"Model: {str(evaluation.model_id)}. Overall progress")

        model: AModel = model_map[evaluation.model_id]().to(torch.float)
        dataset: dict = dataloader_map[evaluation.dataloader_id]()
        print(f"\n Evaluating {evaluation.model_id} {evaluation.dataloader_id}")

        if model.config.model_type == "latentsde":
            results: list = evaluate_latentsde_on_lorenz(
                model,
                reference_obs_times=dataset["reference_obs_times"],
                reference_obs_values=dataset["reference_obs_values"],
                train_data_label=dataset["train_data_label"],
                inference_initial_states=dataset["inference_initial_states"],
                inference_locations=dataset["inference_locations"],
                inference_data_label=dataset["inference_data_label"],
                device=device,
            )  # contains results from multiple sampling strategies

            # copy evaluation setup for all results
            for result in results:
                evaluation_ = deepcopy(evaluation)
                evaluation_.results = result
                evaluations_with_results.append(evaluation_)

        else:
            evaluation.results = evaluate_fim_on_lorenz(
                model,
                train_obs_times=dataset["train_obs_times"],
                train_obs_values=dataset["train_obs_values"],
                train_data_label=dataset["train_data_label"],
                inference_grid=dataset["inference_grid"],
                inference_initial_states=dataset["inference_initial_states"],
                inference_locations=dataset["inference_locations"],
                inference_data_label=dataset["inference_data_label"],
                device=device,
            )
            evaluations_with_results.append(evaluation)

        del model

    return evaluations_with_results


def get_lorenz_datasets(train_data_label: str, train_data_jsons: Path, device=None, **inference_setups: dict[str, dict]):
    """
    Dataset for inference includes:
    - train data (for context of FIM)
    - inference data with grid and initial states
    - labels for train and inference sets
    - reference data (for posterior initial condition of LatentSDE)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset: dict = JsonSDEDataset._load_dict_from_json(
        train_data_jsons, keys_to_load={"train_obs_times": "obs_grid", "train_obs_values": "noisy_obs_values"}
    )
    train_dataset.update({"train_data_label": train_data_label})

    datasets = {}
    for inference_label, inference_jsons in inference_setups.items():
        inference_data: dict = JsonSDEDataset._load_dict_from_json(
            inference_jsons["inference_data"],
            keys_to_load={"inference_initial_states": "initial_states", "inference_grid": "obs_grid", "inference_locations": "locations"},
        )
        reference_data: dict = JsonSDEDataset._load_dict_from_json(
            inference_jsons["reference_data"],
            keys_to_load={
                "reference_obs_values": "noisy_obs_values",
                "reference_obs_times": "obs_grid",
            },
        )

        datasets.update(
            {
                (train_data_label, inference_label): train_dataset
                | inference_data
                | reference_data
                | {"inference_data_label": inference_label}
            }
        )

    datasets = torch.utils._pytree.tree_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, datasets)

    return datasets


def evaluate_all_models(dataset_descr: str, experiment_descr: str, model_dicts: dict, data_setups: dict, results_to_load: list[Path] = []):
    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[dict] = [get_lorenz_datasets(train_data_label=setup_label, **setup) for setup_label, setup in data_setups.items()]
    datasets: dict = {k: v for d in datasets for k, v in d.items()}

    # Get model_map to load models when they are needed
    model_map = model_map_from_dict(model_dicts)
    dataloader_map = dataloader_map_from_dict(datasets)

    # Load previous evaluations that don't need to be evaluated anymore
    loaded_evaluations: list[ModelEvaluation] = load_evaluations(results_to_load)

    # Evaluate models using the data they were trained on
    ids_to_evaluate = []
    for model_id, dataset_id in itertools.product(model_dicts.keys(), datasets.keys()):
        model_train_data_label = model_id[1]
        dataset_train_data_label = dataset_id[0]

        if model_train_data_label == dataset_train_data_label:
            ids_to_evaluate.append((model_id, dataset_id))

    all_evaluations: list[ModelEvaluation] = [ModelEvaluation(model_id, dataset_id) for model_id, dataset_id in ids_to_evaluate]
    to_evaluate: list[ModelEvaluation] = [evaluation for evaluation in all_evaluations if evaluation not in loaded_evaluations]

    # Create, run and save EvaluationConfig
    evaluated: list[ModelEvaluation] = run_lorenz_evaluation(to_evaluate, model_map, dataloader_map)
    all_evaluations: list[ModelEvaluation] = loaded_evaluations + evaluated
    save_evaluations(all_evaluations, evaluation_dir / "model_evaluations")

    # save results in json per model
    for model_id in model_map.keys():
        model_evals = [eval for eval in all_evaluations if eval.model_id == model_id]

        model_label, train_data_label = model_id

        model_outputs = []

        # save one json per model, train_data_label, context size
        for eval in model_evals:
            if eval.model_id == model_id:
                model_outputs.append(
                    {
                        "model_label": model_label,
                        "train_data_label": train_data_label,
                    }
                    | eval.results
                )

        # save outputs as json
        def _check_finite(x):
            if isinstance(x, torch.Tensor):
                assert torch.torch.isfinite(x).all().item()

        _check_finite(model_outputs)

        model_outputs = optree.tree_map(lambda x: x.detach().to("cpu").numpy() if isinstance(x, torch.Tensor) else x, model_outputs)
        json_data = json.dumps(model_outputs, cls=NumpyEncoder)

        # Write JSON data to a file
        json_dir = evaluation_dir / "model_paths"
        json_dir.mkdir(exist_ok=True, parents=True)

        file: Path = json_dir / (model_label + "_train_data_" + train_data_label + ".json")
        with open(file, "w") as file:
            file.write(json_data)


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "lorenz_system_vf_and_paths_evaluation"

    # How to name experiments
    experiment_descr = "latent_sde_context_1_with_vector_fields"

    model_dicts = {
        # ("fim_model_C_at_139_epochs_no_finetuning", "neural_sde_paper"): {
        #     "checkpoint_dir": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/600k_drift_deg_3_diff_deg_2_delta_tau_fixed_linear_attn_softmax_no_extra_normalization_and_fix_in_residual_layer_05-06-2300/checkpoints/",
        #     "checkpoint_name": "epoch-139",
        # },
        # ("lat_sde_context_1", "neural_sde_paper"): {
        #     "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/saved_results/20250617_lorenz_latent_sde_paper_setup/latent_sde_paper_model_context_size_1_06-16-2219/checkpoints",
        #     "checkpoint_name": "epoch-4999",
        # },
        ("lat_sde_context_100", "neural_sde_paper"): {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/saved_results/20250617_lorenz_latent_sde_paper_setup/latent_sde_paper_model_context_size_100_06-16-2219/checkpoints",
            "checkpoint_name": "epoch-4999",
        },
        # ("lat_sde_latent_3_context_1", "neural_sde_paper"): {
        #     "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/saved_results/20250617_lorenz_latent_sde_paper_setup/latent_sde_paper_model_latent_size_3_context_size_1_06-16-2219/checkpoints",
        #     "checkpoint_name": "epoch-4999",
        # },
        ("lat_sde_latent_3_context_100", "neural_sde_paper"): {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/saved_results/20250617_lorenz_latent_sde_paper_setup/latent_sde_paper_model_latent_size_3_context_size_100_06-16-2219/checkpoints",
            "checkpoint_name": "epoch-4999",
        },
        # ("lat_sde_latent_3_no_proj_context_1", "neural_sde_paper"): {
        #     "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/saved_results/20250617_lorenz_latent_sde_paper_setup/latent_sde_paper_model_latent_size_3_no_lin_proj_context_size_1_06-16-2230/checkpoints",
        #     "checkpoint_name": "epoch-4999",
        # },
        ("lat_sde_latent_3_no_proj_context_100", "neural_sde_paper"): {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/saved_results/20250617_lorenz_latent_sde_paper_setup/latent_sde_paper_model_latent_size_3_no_lin_proj_context_size_100_06-16-2219/checkpoints",
            "checkpoint_name": "epoch-4999",
        },
        # ("fim_finetune_1000_epochs_lr_1e-5", "neural_sde_paper"): {
        #     "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/saved_results/20250627_lorenz_fim_finetunings_all_1024_paths/fim_finetune_1024_paths_1024_points_1000_epochs_lr_1e-5_06-27-0540/checkpoints",
        #     "checkpoint_name": "epoch-999",
        # },
        # ("fim_finetune_1000_epochs_lr_1e-6", "neural_sde_paper"): {
        #     "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/saved_results/20250627_lorenz_fim_finetunings_all_1024_paths/fim_finetune_1024_paths_1024_points_1000_epochs_lr_1e-6_06-27-0818/checkpoints",
        #     "checkpoint_name": "epoch-999",
        # },
        # ("fim_finetune_2000_epochs_lr_1e-5", "neural_sde_paper"): {
        #     "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/saved_results/20250627_lorenz_fim_finetunings_all_1024_paths/fim_finetune_1024_paths_1024_points_2000_epochs_lr_1e-5_06-27-0831/checkpoints",
        #     "checkpoint_name": "epoch-1999",
        # },
        # ("fim_finetune_2000_epochs_lr_1e-6", "neural_sde_paper"): {
        #     "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/saved_results/20250627_lorenz_fim_finetunings_all_1024_paths/fim_finetune_1024_paths_1024_points_2000_epochs_lr_1e-6_06-27-0910/checkpoints",
        #     "checkpoint_name": "epoch-1999",
        # },
        # ("fim_finetune_200_epochs_lr_1e-5", "neural_sde_paper"): {
        #     "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/saved_results/20250627_lorenz_fim_finetunings_all_1024_paths/fim_finetune_1024_paths_1024_points_200_epochs_lr_1e-5_06-27-0535/checkpoints",
        #     "checkpoint_name": "epoch-199",
        # },
        # ("fim_finetune_200_epochs_lr_1e-6", "neural_sde_paper"): {
        #     "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/saved_results/20250627_lorenz_fim_finetunings_all_1024_paths/fim_finetune_1024_paths_1024_points_200_epochs_lr_1e-6_06-27-0814/checkpoints",
        #     "checkpoint_name": "epoch-199",
        # },
        # ("fim_finetune_500_epochs_lr_1e-5", "neural_sde_paper"): {
        #     "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/saved_results/20250627_lorenz_fim_finetunings_all_1024_paths/fim_finetune_1024_paths_1024_points_500_epochs_lr_1e-5_06-27-0803/checkpoints",
        #     "checkpoint_name": "epoch-499",
        # },
        # ("fim_finetune_500_epochs_lr_1e-6", "neural_sde_paper"): {
        #     "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/saved_results/20250627_lorenz_fim_finetunings_all_1024_paths/fim_finetune_1024_paths_1024_points_500_epochs_lr_1e-6_06-27-0600/checkpoints",
        #     "checkpoint_name": "epoch-499",
        # },
        # ("fim_train_from_scratch_lr_1e-5", "neural_sde_paper"): {
        #     "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/saved_results/20250627_lorenz_fim_finetunings_all_1024_paths/fim_train_from_scratch_1024_paths_1024_points_20000_epochs_lr_1e-5_07-01-1202/checkpoints",
        #     "checkpoint_name": "epoch-4999",
        # },
    }

    neural_sde_paper_path = Path(
        "/home/seifner/repos/FIM/data/processed/test/20250629_lorenz_system_with_vector_fields_at_locations/neural_sde_paper/set_0/"
    )
    neural_sde_github_path = Path(
        "/home/seifner/repos/FIM/data/processed/test/20250629_lorenz_system_with_vector_fields_at_locations/neural_sde_github/set_0/"
    )

    data_setups = {
        "neural_sde_paper": {
            "train_data_jsons": neural_sde_paper_path / "train_data.json",
            "(1,1,1)": {
                "inference_data": neural_sde_paper_path / "(1,1,1)_inference_data.json",
                "reference_data": neural_sde_paper_path / "(1,1,1)_reference_data.json",
            },
            "N(0,1)": {
                "inference_data": neural_sde_paper_path / "N(0,1)_inference_data.json",
                "reference_data": neural_sde_paper_path / "N(0,1)_reference_data.json",
            },
            "N(0,2)": {
                "inference_data": neural_sde_paper_path / "N(0,2)_inference_data.json",
                "reference_data": neural_sde_paper_path / "N(0,2)_reference_data.json",
            },
        },
        "neural_sde_github": {
            "train_data_jsons": neural_sde_github_path / "train_data.json",
            "(1,1,1)": {
                "inference_data": neural_sde_github_path / "(1,1,1)_inference_data.json",
                "reference_data": neural_sde_github_path / "(1,1,1)_reference_data.json",
            },
            "N(0,1)": {
                "inference_data": neural_sde_github_path / "N(0,1)_inference_data.json",
                "reference_data": neural_sde_github_path / "N(0,1)_reference_data.json",
            },
            "N(0,2)": {
                "inference_data": neural_sde_github_path / "N(0,2)_inference_data.json",
                "reference_data": neural_sde_github_path / "N(0,2)_reference_data.json",
            },
        },
    }

    results_to_load = []

    # --------------------------------------------------------------------------------------------------------------------------------- #

    evaluate_all_models(dataset_descr, experiment_descr, model_dicts, data_setups, results_to_load=results_to_load)
