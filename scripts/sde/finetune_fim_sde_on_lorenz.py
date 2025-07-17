#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import logging
import os
import warnings
from pathlib import Path

import click
import torch
from train_latent_sde import sample_paths_from_trained_model
from transformers import AutoConfig

from fim.data.dataloaders import DataLoaderFactory
from fim.models import FIMSDE
from fim.models.blocks import AModel, ModelFactory
from fim.trainers.trainer import TrainerFactory
from fim.trainers.utils import cleanup, clear_gpu_cache, get_accel_type, setup, setup_environ_flags
from fim.utils.helper import expand_params, load_yaml
from fim.utils.logging import RankLoggerAdapter, setup_logging


setup_logging()

warnings.filterwarnings("ignore", module="matplotlib")
logger = RankLoggerAdapter(logging.getLogger(__name__))


def finetune_fim(
    test_data_setups: dict,
    config: Path,
    seed: int,
    exp_name: str,
    train_data_label: str,
    detach_diffusion: bool,
    likelihood: bool,
    sampling_mse: bool,
    sampling_nll: bool,
    num_points: int,
    samples_count: int,
    samples_steps: int,
    em_steps: int,
    sample_paths: bool,
    epochs: int,
    save_every: int,
    lr: float,
    weight_decay: float,
    train_from_scratch: bool,
) -> None:
    """
    Training config is loaded from file.
    FIM is loaded from checkpoint.
    Optionally overwrite some train configs / model attributes manually before training.
    Sample paths from the finetuned model.
    """
    config = load_yaml(config)
    config["experiment"]["seed"] = seed
    config["experiment"]["name"] = exp_name

    optimizer = config["optimizers"][0]["optimizer_d"]
    optimizer["lr"] = lr
    optimizer["weight_decay"] = weight_decay
    config["optimizers"] = ({"optimizer_d": optimizer},)

    config["trainer"]["epochs"] = epochs
    config["trainer"]["save_every"] = save_every

    model_path = "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/600k_drift_deg_3_diff_deg_2_delta_tau_fixed_linear_attn_softmax_no_extra_normalization_and_fix_in_residual_layer_05-06-2300/checkpoints/epoch-139"

    if train_from_scratch is False:
        model: FIMSDE = AModel.load_model(model_path)

    else:
        model_config = AutoConfig.from_pretrained(Path(model_path) / "config.json")
        model = ModelFactory.create(model_config)

    model.finetune = True
    model.config.finetune = True

    model.finetune_samples_count = samples_count
    model.config.finetune_samples_count = samples_count

    model.finetune_samples_steps = samples_steps
    model.config.finetune_samples_steps = samples_steps

    model.finetune_em_steps = em_steps
    model.config.finetune_em_steps = em_steps

    model.finetune_detach_diffusion = detach_diffusion
    model.config.finetune_detach_diffusion = detach_diffusion

    model.finetune_on_likelihood = likelihood
    model.config.finetune_on_likelihood = likelihood

    model.finetune_on_sampling_mse = sampling_mse
    model.config.finetune_on_sampling_mse = sampling_mse

    model.finetune_on_sampling_nll = sampling_nll
    model.config.finetune_on_sampling_nll = sampling_nll

    model.finetune_num_points = num_points
    model.config.finetune_num_points = num_points

    gs_config = expand_params(config)[0]
    gs_config.model = model.config

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    setup(rank, world_size)

    # saved parameters are sharded, so need to use torchrun for now
    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
        device_map = gs_config.experiment.device_map
        if device_map == "auto":
            device_map = get_accel_type()

        dataloader = DataLoaderFactory.create(**gs_config.dataset.to_dict())
        trainer = TrainerFactory.create(gs_config.trainer.name, model=model, dataloader=dataloader, config=gs_config)
        trainer.train()

    cleanup()

    if sample_paths is True:
        sample_paths_from_trained_model(trainer, exp_name, train_data_label, test_data_setups)


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
    @click.option("--config", "config", type=click.Path(exists=True), required=True)
    @click.option("--seed", "seed", type=int, required=True)
    @click.option("--exp-name", "exp_name", type=str, required=True)
    @click.option("--train-data-label", "train_data_label", type=str, required=True)
    @click.option("--detach-diffusion", "detach_diffusion", type=bool, required=False, default=False)
    @click.option("--likelihood", "likelihood", type=bool, required=False, default=False)
    @click.option("--sampling-mse", "sampling_mse", type=bool, required=False, default=False)
    @click.option("--sampling-nll", "sampling_nll", type=bool, required=False, default=False)
    @click.option("--samples-count", "samples_count", type=int, required=False, default=1)
    @click.option("--samples-steps", "samples_steps", type=int, required=False, default=1)
    @click.option("--em-steps", "em_steps", type=int, required=False, default=1)
    @click.option("--num-points", "num_points", type=int, required=False, default=-1)
    @click.option("--sample-paths", "sample_paths", type=bool, required=False, default=True)
    @click.option("--epochs", "epochs", type=int, required=False, default=500)
    @click.option("--save-every", "save_every", type=int, required=False, default=100)
    @click.option("--lr", "lr", type=float, required=False, default=1e-5)
    @click.option("--weight-decay", "weight_decay", type=float, required=False, default=0.0)
    @click.option("--train-from-scratch", "train_from_scratch", type=bool, required=False, default=False)
    def cli(**kwargs):
        finetune_fim(test_data_setups, **kwargs)

    # pylint: disable=no-value-for-parameter
    cli()
