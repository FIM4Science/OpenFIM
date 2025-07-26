#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import json
import logging
import warnings
from pathlib import Path

import click
import finetune_helpers
import latent_sde_helpers
import numpy as np
import torch
from evaluation.real_world_cross_validation_vf_and_paths_evaluation import evaluate_model as evaluate_model
from evaluation.real_world_cross_validation_vf_and_paths_evaluation import get_real_world_data

from fim import project_path
from fim.models import LatentSDE
from fim.models.blocks import AModel
from fim.utils.helper import load_yaml
from fim.utils.logging import RankLoggerAdapter, setup_logging
from fim.utils.sde.evaluation import NumpyEncoder


setup_logging()

warnings.filterwarnings("ignore", module="matplotlib")
logger = RankLoggerAdapter(logging.getLogger(__name__))


def get_train_data(inference_data_path: Path, dataset_label: str, split: int, num_train_subsplits: int, len_train_subsplits: int | None):
    """
    Load training data from json, normalize time, standardize values and extract batch_size and path_length_to_generate.
    """

    inference_data = json.load(open(inference_data_path, "r"))
    inference_data: dict = get_real_world_data(
        inference_data,
        dataset_label,
        split,
        expected_num_total_splits=5,
        obs_times_key="obs_times_split",
        obs_values_key="obs_values_split",
    )

    obs_times = inference_data.get("obs_times").squeeze(0)  # [4, T, 1]
    obs_values = inference_data.get("obs_values").squeeze(0)  # [4, T, 1]
    path_length_to_generate = inference_data["path_length_to_generate"]

    # split further into smaller trajectories for training
    T = obs_times.shape[-2]

    if len_train_subsplits is not None:
        num_train_subsplits = T // len_train_subsplits

    len_train_subsplits = T // num_train_subsplits
    T_trunc = len_train_subsplits * num_train_subsplits
    obs_times = obs_times[:, :T_trunc, :]
    obs_values = obs_values[:, :T_trunc, :]

    obs_times = np.concatenate(np.split(obs_times, indices_or_sections=num_train_subsplits, axis=-2), axis=-3)  # [B, T_, 1]
    obs_values = np.concatenate(np.split(obs_values, indices_or_sections=num_train_subsplits, axis=-2), axis=-3)  # [B, T_, 1]

    # normalize time
    obs_times = obs_times - obs_times[:, 0, :][:, None, :]
    obs_times_max = np.amax(obs_times)
    obs_times = obs_times / obs_times_max

    # standardize values
    obs_values_mean = obs_values.mean()
    obs_values_std = obs_values.std()
    obs_values = (obs_values - obs_values_mean) / obs_values_std

    dataset = {"obs_times": obs_times, "obs_values": obs_values}
    dataset = {k: torch.from_numpy(v).to(torch.float32) if isinstance(v, np.ndarray) else v for k, v in dataset.items()}

    batch_size = obs_times.shape[0]

    return dataset, batch_size, obs_times_max, obs_values_mean, obs_values_std, path_length_to_generate, len_train_subsplits


def get_reference_data(
    reference_data_path: Path,
    dataset_label: str,
    split: int,
    len_train_subsplits: int,
    path_length_to_generate: int,
    times_max: torch.Tensor,
    values_mean: torch.Tensor,
    values_std: torch.Tensor,
):
    """
    Load reference paths data, truncate them, and normalize / standardize the values according to train data.
    Build sampling_grid from information contained in reference path data
    Return identifying information for model output json.
    """

    reference_data = json.load(open(reference_data_path, "r"))
    reference_data: dict = get_real_world_data(
        reference_data,
        dataset_label,
        split,
        expected_num_total_splits=5,
        obs_times_key="obs_times",
        obs_values_key="obs_values",
    )
    reference_data = {k: torch.from_numpy(v).to(torch.float32) if isinstance(v, np.ndarray) else v for k, v in reference_data.items()}

    # apply time normalization
    obs_times = reference_data["obs_times"].squeeze(0)  # [10, T, 1]
    obs_times = obs_times[0, :, 0]  # regular grid, they are (after rescaling) all the same, [T]
    obs_times = (obs_times - obs_times[0]) / times_max

    # apply value standardization
    obs_values = reference_data["obs_values"].squeeze(0)  # [10, T, 1]
    obs_values = (obs_values - values_mean) / values_std

    # estimate posterior initial condition from trajectory of length model has been trained with
    obs_values = obs_values[:, :len_train_subsplits]
    obs_times = obs_times[:len_train_subsplits]  # now, by construction, regular grid in [0,1]

    # normalized sampling grid
    delta_tau = reference_data["delta_tau"]
    sampling_grid = torch.arange(path_length_to_generate) * delta_tau  # [G]
    sampling_grid = sampling_grid / times_max

    num_total_splits = reference_data["num_total_splits"]
    delta_tau = reference_data["delta_tau"]
    transform = reference_data["transform"]

    return obs_times, obs_values, sampling_grid, num_total_splits, delta_tau, transform


def train_latent_sde_on_real_world_cross_validation(
    inference_data_path: str,
    reference_data_path: str,
    epochs_to_evaluate: list[int] | None,
    dataset_label: str,
    split: int,
    base_config: Path,
    exp_name: str,
    num_train_subsplits: int,
    len_train_subsplits: int | None,  # if passed, num_train_subsplits is ignored
    **extra_configs,
) -> None:
    """
    Training config is loaded from file.
    Optionally overwrite some train configs / model attributes manually before training.
    Sample paths from some checkpoints of the trained model.
    """
    exp_name_with_dataset_split = f"{exp_name}_dataset_{dataset_label}_split_{split}"

    config = load_yaml(base_config)
    config = latent_sde_helpers.add_extra_latentsde_configs(config, exp_name_with_dataset_split, **extra_configs)

    dataset, batch_size, times_max, values_mean, values_std, path_length_to_generate, len_train_subsplits = get_train_data(
        inference_data_path, dataset_label, split, num_train_subsplits, len_train_subsplits
    )

    config["dataset"]["json_paths"]["train"] = dataset
    config["dataset"]["json_paths"]["test"] = dataset  # have not splits available
    config["dataset"]["json_paths"]["validation"] = dataset  # have not splits available

    config["dataset"]["batch_size"]["train"] = batch_size
    config["dataset"]["batch_size"]["test"] = batch_size
    config["dataset"]["batch_size"]["validation"] = batch_size

    trainer = latent_sde_helpers.train_latent_sde(config)
    finetune_helpers.add_model_type_to_checkpoints(trainer)

    checkpoint_dir = Path(project_path) / trainer.checkpointer.checkpoint_dir

    del trainer

    last_epoch = finetune_helpers.get_last_epoch(checkpoint_dir)

    if epochs_to_evaluate is None:
        epochs_to_evaluate = [last_epoch]

    else:
        if last_epoch not in epochs_to_evaluate:
            epochs_to_evaluate.append(last_epoch)

    for epoch in epochs_to_evaluate:
        get_real_world_vector_fields_and_paths(
            checkpoint_dir,
            epoch,
            reference_data_path,
            dataset_label,
            split,
            len_train_subsplits,
            path_length_to_generate,
            exp_name,
            times_max,
            values_mean,
            values_std,
        )


def get_real_world_vector_fields_and_paths(
    checkpoint_dir: Path,
    epoch: str,
    reference_data_path: Path,
    dataset_label: str,
    split: int,
    len_train_subsplits: int,
    path_length_to_generate: int,
    exp_name: str,
    times_max: torch.Tensor,
    values_mean: torch.Tensor,
    values_std: torch.Tensor,
):
    """
    Load checkpoint from epoch, sample paths and evaluate vector fields for the dataset and split.
    Save model outputs as json.
    """
    model_checkpoint_to_evaluate: Path = checkpoint_dir / epoch

    if model_checkpoint_to_evaluate.exists():
        obs_times, obs_values, sampling_grid, num_total_splits, delta_tau, transform = get_reference_data(
            reference_data_path, dataset_label, split, len_train_subsplits, path_length_to_generate, times_max, values_mean, values_std
        )

        model: LatentSDE = AModel.load_model(model_checkpoint_to_evaluate)
        model.eval()

        # sample from posterior initial condition and solve prior equation
        with torch.no_grad():
            ctx, obs_times, _ = model.encode_inputs(obs_times, obs_values)
            posterior_initial_states, _, _ = model.sample_posterior_initial_condition(ctx[0])
            _, sample_paths = model.sample_from_prior_equation(posterior_initial_states, sampling_grid)  # [10, grid_length, 1]

        # reverse normalization
        sample_paths = sample_paths * values_std + values_mean

        # save model outputs as json
        model_outputs = {
            "name": dataset_label,
            "split": split,
            "num_total_splits": num_total_splits,
            "delta_tau": delta_tau,
            "transform": transform,
            "synthetic_paths": sample_paths,
            "locations": None,
            "drift_at_locations": None,
            "diffusion_at_locations": None,
        }

        model_outputs = torch.utils._pytree.tree_map(
            lambda x: x.detach().to("cpu").numpy() if isinstance(x, torch.Tensor) else x, model_outputs
        )

        json_data = json.dumps(model_outputs, cls=NumpyEncoder)

        evaluation_path = Path(project_path) / "evaluations" / "real_world_cross_validation_vf_and_paths_evaluation" / exp_name
        evaluation_path.mkdir(exist_ok=True, parents=True)

        file_path: Path = evaluation_path / (f"{exp_name}_dataset_{dataset_label}_split_{split}_{epoch.replace('-', '_')}.json")

        with open(file_path, "w") as f:
            f.write(json_data)


if __name__ == "__main__":
    data_path = Path("/cephfs/users/seifner/repos/FIM/data/processed/test/20250726_real_world_with_5_fold_cross_validation_develop/")
    inference_data_path = data_path / "cross_val_inference_paths.json"
    reference_data_path = data_path / "cross_val_ksig_reference_paths.json"

    @click.command()
    @click.option("--base-config", "base_config", type=click.Path(exists=True), required=True)
    @click.option("--dataset-label", "dataset_label", type=str, required=True)
    @click.option("--split", "split", type=int, required=True)
    @click.option("--exp-name", "exp_name", type=str, required=True)
    @click.option("--num-train-subsplits", "num_train_subsplits", type=int, required=False, default=10)
    @click.option("--len-train-subsplits", "len_train_subsplits", type=int, required=False, default=None)
    @click.option("--seed", "seed", type=int, required=True)
    @click.option("--epochs", "epochs", type=int, required=False)
    @click.option("--epochs-to-evaluate", "epochs_to_evaluate", type=str, required=False)
    @click.option("--context-size", "context_size", type=int, required=False, default=None)
    @click.option("--hidden-size", "hidden_size", type=int, required=False, default=None)
    @click.option("--latent-size", "latent_size", type=int, required=False, default=None)
    @click.option("--activation", "activation", type=str, required=False, default=None)
    @click.option("--learn-projection", "learn_projection", type=bool, required=False, default=None)
    @click.option("--solver-dt", "solver_dt", type=float, required=False, default=None)
    @click.option("--kl-scale-end-step", "kl_scale_end_step", type=int, required=False, default=None)
    @click.option("--lr", "lr", type=float, default=None)
    @click.option("--lr-gamma", "lr_gamma", type=float, default=None)
    @click.option("--mse-objective", "mse_objective", type=bool, required=False, default=False)
    @click.option("--save-every", "save_every", type=int, required=False, default=100)
    def cli(epochs_to_evaluate: str = None, **kwargs):
        if epochs_to_evaluate is not None:
            # expect list of comma separated integers
            epochs_to_evaluate: list[str] = epochs_to_evaluate.split(",")
            epochs_to_evaluate = [f"epoch-{e}" for e in epochs_to_evaluate]

        train_latent_sde_on_real_world_cross_validation(inference_data_path, reference_data_path, epochs_to_evaluate, **kwargs)

    # pylint: disable=no-value-for-parameter
    cli()
