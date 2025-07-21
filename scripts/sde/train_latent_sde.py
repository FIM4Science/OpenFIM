#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import logging
import warnings
from pathlib import Path

import click
import finetune_helpers
import numpy as np
import torch
from evaluation.lorenz_system_paths_evaluation import evaluate_all_models

from fim import project_path
from fim.data.dataloaders import DataLoaderFactory
from fim.models.blocks import ModelFactory
from fim.trainers.trainer import Trainer, TrainerFactory
from fim.utils.helper import expand_params, load_yaml
from fim.utils.logging import RankLoggerAdapter, setup_logging


setup_logging()

warnings.filterwarnings("ignore", module="matplotlib")
logger = RankLoggerAdapter(logging.getLogger(__name__))


def sample_lorenz_paths_from_trained_model(trainer: Trainer, exp_name: str, train_data_label: str, test_data_setups: dict) -> None:
    """
    Sample Paths from trained model based on some test data.
    """
    checkpoint_dir = Path(project_path) / trainer.checkpointer.checkpoint_dir
    last_epoch = finetune_helpers.get_last_epoch(checkpoint_dir)

    model_dicts = {(exp_name, train_data_label): {"checkpoint_dir": checkpoint_dir, "checkpoint_name": last_epoch}}

    evaluate_all_models(
        dataset_descr="lorenz_system_vf_and_paths_evaluation",
        experiment_descr=exp_name,
        model_dicts=model_dicts,
        data_setups=test_data_setups,
    )


def train_latent_sde(
    test_data_setups: dict,
    base_config: Path,
    seed: int,
    exp_name: str,
    train_data_label: str,
    context_size: int,
    hidden_size: int,
    latent_size: int,
    activation: str,
    learn_projection: bool,
    solver_dt: float,
    kl_scale_end_step: int,
    lr: float,
    lr_gamma: float,
    sample_paths: bool,
) -> None:
    """
    Train Latent SDE model from config file.
    Optionally overwrite some configs manually.
    Sample paths from the trained model.
    """
    config = load_yaml(base_config)
    config["experiment"]["seed"] = seed
    config["experiment"]["name"] = exp_name

    if context_size is not None:
        config["model"]["context_size"] = context_size
    if hidden_size is not None:
        config["model"]["hidden_size"] = hidden_size
    if latent_size is not None:
        config["model"]["latent_size"] = latent_size
    if activation is not None:
        config["model"]["activation"] = activation
    if learn_projection is not None:
        config["model"]["learn_projection"] = learn_projection
    if solver_dt is not None:
        config["model"]["solver_dt"] = solver_dt
    if kl_scale_end_step is not None:
        kl_scheduler = config["trainer"]["schedulers"][0]
        kl_scheduler["end_step"] = kl_scale_end_step
        config["trainer"]["schedulers"] = (kl_scheduler,)
    if lr is not None:
        optimizer = config["optimizers"][0]["optimizer_d"]
        optimizer["lr"] = lr
        config["optimizers"] = ({"optimizer_d": optimizer},)
    if lr_gamma is not None:
        optimizer = config["optimizers"][0]["optimizer_d"]
        scheduler = optimizer["schedulers"][0]
        lr_scheduler = scheduler["schedulers"][0]
        lr_scheduler["gamma"] = lr_gamma
        scheduler["schedulers"] = (lr_scheduler,)
        optimizer["schedulers"] = (scheduler,)
        config["optimizers"] = ({"optimizer_d": optimizer},)

    gs_config = expand_params(config)[0]

    torch.manual_seed(int(gs_config.experiment.seed))
    torch.cuda.manual_seed(int(gs_config.experiment.seed))
    np.random.seed(int(gs_config.experiment.seed))
    torch.cuda.empty_cache()

    dataloader = DataLoaderFactory.create(**gs_config.dataset.to_dict())
    model = ModelFactory.create(gs_config.model.to_dict())

    trainer = TrainerFactory.create(gs_config.trainer.name, model=model, dataloader=dataloader, config=gs_config)
    trainer.train()

    if sample_paths is True:
        sample_lorenz_paths_from_trained_model(trainer, exp_name, train_data_label, test_data_setups)


if __name__ == "__main__":
    neural_sde_paper_path = Path(
        "/home/seifner/repos/FIM/data/processed/test/20250629_lorenz_system_with_vector_fields_at_locations/neural_sde_paper/set_0/"
    )
    neural_sde_github_path = Path(
        "/home/seifner/repos/FIM/data/processed/test/20250629_lorenz_system_with_vector_fields_at_locations/neural_sde_github/set_0/"
    )

    test_data_setups = {
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

    @click.command()
    @click.option("--base-config", "base_config", type=click.Path(exists=True), required=True)
    @click.option("--seed", "seed", type=int, required=True)
    @click.option("--exp-name", "exp_name", type=str, required=True)
    @click.option("--train-data-label", "train_data_label", type=str, required=True)
    @click.option("--context-size", "context_size", type=int, required=False, default=None)
    @click.option("--hidden-size", "hidden_size", type=int, required=False, default=None)
    @click.option("--latent-size", "latent_size", type=int, required=False, default=None)
    @click.option("--activation", "activation", type=str, required=False, default=None)
    @click.option("--learn-projection", "learn_projection", type=bool, required=False, default=None)
    @click.option("--solver-dt", "solver_dt", type=float, required=False, default=None)
    @click.option("--kl-scale-end-step", "kl_scale_end_step", type=int, required=False, default=None)
    @click.option("--lr", "lr", type=float, default=None)
    @click.option("--lr-gamma", "lr_gamma", type=float, default=None)
    @click.option("--sample-paths", "sample_paths", type=bool, required=False, default=True)
    def cli(**kwargs):
        train_latent_sde(test_data_setups, **kwargs)

    # pylint: disable=no-value-for-parameter
    cli()
