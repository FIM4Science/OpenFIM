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
from train_latent_sde import sample_lorenz_paths_from_trained_model

from fim.utils.helper import load_yaml
from fim.utils.logging import RankLoggerAdapter, setup_logging


setup_logging()

warnings.filterwarnings("ignore", module="matplotlib")
logger = RankLoggerAdapter(logging.getLogger(__name__))


def finetune_fim_on_lorenz(
    model_path: str,
    test_data_setups: dict,
    train_from_scratch: bool,
    base_config: Path,
    train_data_label: str,
    exp_name: str,
    sample_paths: bool,
    **extra_configs,
) -> None:
    """
    FIM is loaded from checkpoint.
    Training config is loaded from file.
    Optionally overwrite some train configs / model attributes manually before training.
    Sample paths from the finetuned model.
    """
    model = finetune_helpers.load_pretrained_model(model_path, train_from_scratch)
    base_config = load_yaml(base_config)

    config, model = finetune_helpers.add_extra_fimsde_configs(base_config, model, exp_name, **extra_configs)

    trainer = finetune_helpers.train_fimsde(model, config)
    finetune_helpers.add_model_type_to_checkpoints(trainer)

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

    model_path = "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/600k_drift_deg_3_diff_deg_2_delta_tau_fixed_linear_attn_softmax_no_extra_normalization_and_fix_in_residual_layer_05-06-2300/checkpoints/epoch-139"

    @click.command()
    @click.option("--base-config", "base_config", type=click.Path(exists=True), required=True)
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
        finetune_fim_on_lorenz(model_path, test_data_setups, **kwargs)

    # pylint: disable=no-value-for-parameter
    cli()
