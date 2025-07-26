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
import latent_sde_helpers
from evaluation.lorenz_system_paths_evaluation import evaluate_all_models

from fim import project_path
from fim.trainers.trainer import Trainer
from fim.utils.helper import load_yaml
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
    exp_name: str,
    train_data_label: str,
    **extra_configs,
) -> None:
    """
    Train Latent SDE model from config file.
    Optionally overwrite some configs manually.
    Sample paths from the trained model.
    """
    config = load_yaml(base_config)
    config = latent_sde_helpers.add_extra_latentsde_configs(config, exp_name, **extra_configs)

    trainer = latent_sde_helpers.train_latent_sde(config)

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
    def cli(**kwargs):
        train_latent_sde(test_data_setups, **kwargs)

    # pylint: disable=no-value-for-parameter
    cli()
