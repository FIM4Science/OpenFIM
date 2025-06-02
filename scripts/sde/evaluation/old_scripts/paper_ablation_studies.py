import itertools
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import optree
import pandas as pd
import torch
from metrics_helpers import mean_bracket_std_agg, mean_plus_std_agg, save_table
from torch.utils.data import DataLoader
from tqdm import tqdm

from fim import project_path
from fim.data.datasets import PaddedFIMSDEDataset
from fim.models.blocks import AModel
from fim.models.sde import FIMSDE
from fim.utils.sde.evaluation import ModelEvaluation, ModelMap, load_evaluations, save_evaluations


def get_dataloader(data_dir: str) -> tuple[dict]:
    """
    Return DataLoaderInitializer for mocap forecasting data.

    Args:
        data_dir (Path): Absolute path to dir with subdirs of data.

    Returns:
        dataloder_dict, dataloader_map
    """
    data_dir: Path = Path(data_dir)
    data_dirs = list(data_dir.iterdir())
    data_dirs = [d for d in data_dirs if d.is_dir() is True]

    files_to_load = {
        "obs_times": "obs_times.h5",
        "obs_values": "obs_noisy_values.h5",
        "locations": "locations.h5",
        "drift_at_locations": "drift_at_locations.h5",
        "diffusion_at_locations": "diffusion_at_locations.h5",
    }

    dataset = PaddedFIMSDEDataset(
        data_dirs=data_dirs,
        batch_size=1,  # needs to be 1 to return losses accurately
        files_to_load=files_to_load,
        max_dim=3,
        # data_limit=200,
        shuffle_locations=False,
        shuffle_paths=False,
        shuffle_elements=False,
    )

    dataloader = DataLoader(
        dataset,
        drop_last=False,
        shuffle=False,
        batch_size=None,  # handled by iterable dataset
        num_workers=0,
    )

    return dataloader


def run_evaluations(
    to_evaluate: list[ModelEvaluation],
    model_map: ModelMap,
    dataloaders: dict,
) -> list[ModelEvaluation]:
    """
    Evaluate model on mocap data.

    Args:
        evaluations (list[ModelEvaluation]): List of evaluations to run, specified by their model_id and dataloader_id
        model / dataloader map: Returning required dataloaders
        num_sample_paths (int): number of model sample paths per trajectory

    Return:
        evaluations (list[ModelEvaluation]): Input evaluations.
    """
    evaluations_with_results: list[ModelEvaluation] = []

    for evaluation in (pbar := tqdm(to_evaluate, total=len(to_evaluate), leave=False)):
        pbar.set_description(f"Model: {str(evaluation.model_id)}. Overall progress")

        model: AModel = model_map[evaluation.model_id].to(torch.float)
        dataloader: list[DataLoader] = dataloaders[evaluation.dataloader_id]

        evaluation.results = evaluate_model(model, dataloader)
        evaluations_with_results.append(evaluation)

    return evaluations_with_results


def evaluate_model(model: AModel, dataloader: DataLoader):
    """
    Evaluate model on a synthetic test dataloader.
    Compute mse of drift and diffusion and the L1 objective.
    """
    model.eval()
    model.to("cuda")

    results = {
        "mse_drift": [],
        "mse_diffusion": [],
        "train_objective": [],
        "U_loss_scale": [],
        "l1": [],
        "l2": [],
        "l3": [],
        "dim": [],
    }

    for dataset in tqdm(dataloader):
        # evaluate model
        dataset = optree.tree_map(lambda x: x.to("cuda"), dataset)
        estimated_concepts, losses = model(dataset, training=False, return_losses=True)
        dataset, estimated_concepts, losses = optree.tree_map(
            lambda x: x.detach().to("cpu").numpy() if isinstance(x, torch.Tensor) else x,
            (dataset, estimated_concepts, losses),
            namespace="fimsde",
        )

        drift = estimated_concepts.drift  # [1, L, 3]
        diffusion = estimated_concepts.diffusion

        # compute metrics
        gt_drift = dataset["drift_at_locations"]  # [1, L, 3]
        gt_diffusion = dataset["diffusion_at_locations"]
        _, L, _ = gt_drift.shape

        dimension_mask = dataset["dimension_mask"]  # [1, L, 3], masking padded dimensions
        D = dimension_mask.sum(axis=-1)  # [1, L]

        se_drift = ((drift - gt_drift) ** 2 * dimension_mask).sum(axis=-1) / D  # [1, L]
        assert se_drift.shape == (1, L)

        se_diffusion = ((diffusion - gt_diffusion) ** 2 * dimension_mask).sum(axis=-1) / D  # [1, L]
        assert se_diffusion.shape == (1, L)

        l1 = se_drift + se_diffusion
        assert l1.shape == (1, L)

        # save metric result per equation for later processing
        results["l1"].append(np.array(l1.mean(axis=-1)))
        results["l2"].append(np.array(losses["losses"]["L2_KL_loss"]))
        results["l3"].append(np.array(losses["losses"]["L3_short_time_trans_log_likelihood_loss"]))
        results["mse_drift"].append(se_drift.mean(axis=-1))
        results["mse_diffusion"].append(se_diffusion.mean(axis=-1))
        results["train_objective"].append(losses["losses"]["loss"])
        results["U_loss_scale"].append(losses["losses"]["drift_log_loss_scale_per_location"])
        results["dim"].append(D[0, 0].sum())

    results["l1"] = np.concatenate(results["l1"])
    results["l2"] = np.array(results["l2"])
    results["l3"] = np.array(results["l3"])
    results["mse_drift"] = np.concatenate(results["mse_drift"])
    results["mse_diffusion"] = np.concatenate(results["mse_diffusion"])
    results["train_objective"] = np.array(results["train_objective"])
    results["U_loss_scale"] = np.array(results["U_loss_scale"])
    results["dim"] = np.array(results["dim"])

    return results


def metrics_tables(all_evaluations: list[ModelEvaluation], models_order: list[str]):
    """
    Turn results into tables per dataset, model and metric.
    Return tables of mean, std, mean + std, mean(std)
    """
    # to pandas dataframe for better handling
    rows = []
    for eval in all_evaluations:
        B = eval.results["l1"].shape[0]

        # with dim identifier
        for i in range(B):
            rows.append({"data": eval.dataloader_id, "model": eval.model_id} | optree.tree_map(lambda x: x[i], eval.results))

        # without dim identifier
        eval.results.pop("dim")
        for i in range(B):
            rows.append({"data": eval.dataloader_id, "model": eval.model_id, "dim": "all"} | optree.tree_map(lambda x: x[i], eval.results))

    cols = optree.tree_map(lambda *x: x, *rows)
    df = pd.DataFrame.from_dict(cols)

    # models have custom sorting
    df["model"] = pd.Categorical(df["model"], models_order)

    # count number of equations
    df_count_eqs = deepcopy(df[["data", "model", "mse_drift"]])
    df_count_eqs = df_count_eqs.groupby(["data", "model"]).size()
    df_count_eqs = df_count_eqs.unstack(0)

    # mean without Nans
    df_mean = df.groupby(["data", "dim", "model"]).agg(lambda x: str(round(x.dropna().mean(), 4)) if (len(x.dropna()) != 0) else "-")

    # std without Nans
    df_std = df.groupby(["data", "dim", "model"]).agg(lambda x: str(round(x.dropna().std(), 4)) if (len(x.dropna()) != 0) else "-")

    # mean and std in one cell; formatted as "mean $\pm$ std"
    df_mean_plus_std = df.groupby(["data", "dim", "model"]).agg(partial(mean_plus_std_agg, precision=4))

    # mean and std in one cell; formatted with bracket notation
    df_mean_bracket_std = df.groupby(["data", "dim", "model"]).agg(partial(mean_bracket_std_agg, precision=4))

    return df_count_eqs, df_mean, df_std, df_mean_plus_std, df_mean_bracket_std


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "ablation_studies"

    # How to name experiments
    experiment_descr = "size_and_drift_degree_ablation_at_1_epoch_and_500k_steps_10k_equations"

    models = {
        "30k_deg_4_drift_first_checkpoint": FIMSDE.load_model(
            "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/ablation_models/30k_drift_deg_4_diff_deg_2_5M_params_1_layer_enc_4_layer_128_hidden_size_03-24-1650/checkpoints/epoch-1"
        ),
        "30k_train_size_first_checkpoint": FIMSDE.load_model(
            "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/ablation_models/30k_drift_deg_3_diff_deg_2_5M_params_1_layer_enc_4_layer_128_hidden_size_03-16-1744/checkpoints/epoch-1"
        ).to("cpu"),
        "100k_train_size_first_checkpoint": FIMSDE.load_model(
            "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/ablation_models/100k_drift_deg_3_diff_deg_2_10M_params_1_layer_enc_4_layer_256_hidden_size_03-16-1737/checkpoints/epoch-1"
        ).to("cpu"),
        "600k_train_size_first_checkpoint": FIMSDE.load_model(
            "/home/seifner/repos/FIM/saved_results/20250124_20M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_delta_tau_1e-1_to_1e-3_2_layer_enc_8_layer_03-09-1844/checkpoints/epoch-1"
        ).to("cpu"),
        "30k_deg_4_drift_500k_steps": FIMSDE.load_model(
            "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/ablation_models/30k_drift_deg_4_diff_deg_2_5M_params_1_layer_enc_4_layer_128_hidden_size_03-24-1650/checkpoints/epoch-475"
        ),
        "30k_train_size_500k_steps": FIMSDE.load_model(
            "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/ablation_models/30k_drift_deg_3_diff_deg_2_5M_params_1_layer_enc_4_layer_128_hidden_size_03-16-1744/checkpoints/epoch-494"
        ).to("cpu"),
        "100k_train_size_500k_steps": FIMSDE.load_model(
            "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/ablation_models/100k_drift_deg_3_diff_deg_2_10M_params_1_layer_enc_4_layer_256_hidden_size_03-18-1123/checkpoints/epoch-159"
        ).to("cpu"),
        "600k_train_size_500k_steps": FIMSDE.load_model(
            "/home/seifner/repos/FIM/saved_results/20250124_20M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_delta_tau_1e-1_to_1e-3_2_layer_enc_8_layer_03-09-1844/checkpoints/epoch-70"
        ).to("cpu"),
    }

    dataloaders = {
        "10k_test_split": get_dataloader(
            "/cephfs_projects/foundation_models/data/SDE/train/20250316_100k_polynomials_drift_deg_3_diffusion_deg_2_100_paths_half_with_noise/test/"
        ),
    }
    results_to_load: list[str] = [
        "/home/seifner/repos/FIM/evaluations/ablation_studies/03261203_size_and_drift_degree_ablation_at_1_epoch_and_500k_steps_10k_equations/model_evaluations"
    ]

    # --------------------------------------------------------------------------------------------------------------------------------- #
    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Load previous evaluations that don't need to be evaluated anymore
    loaded_evaluations: list[ModelEvaluation] = load_evaluations(results_to_load)

    # Evaluate all models on all datasets
    all_evaluations: list[ModelEvaluation] = [
        ModelEvaluation(model_id, dataloader_id) for model_id, dataloader_id in itertools.product(models.keys(), dataloaders.keys())
    ]
    to_evaluate: list[ModelEvaluation] = [evaluation for evaluation in all_evaluations if evaluation not in loaded_evaluations]

    # Create, run and save evaluations
    evaluated: list[ModelEvaluation] = run_evaluations(to_evaluate, models, dataloaders)

    # Add loaded evaluations
    all_evaluations: list[ModelEvaluation] = loaded_evaluations + evaluated
    save_evaluations(all_evaluations, evaluation_dir / "model_evaluations")

    # create and save tables
    df_count_eqs, df_mean, df_std, df_mean_plus_std, df_mean_bracket_std = metrics_tables(all_evaluations, models_order=models.keys())

    save_table(df_count_eqs, evaluation_dir, "count_non_nans")
    save_table(df_mean, evaluation_dir, "mean")
    save_table(df_std, evaluation_dir, "std")
    save_table(df_mean_plus_std, evaluation_dir, "mean_plus_std")
    save_table(df_mean_bracket_std, evaluation_dir, "mean_bracket_std")
