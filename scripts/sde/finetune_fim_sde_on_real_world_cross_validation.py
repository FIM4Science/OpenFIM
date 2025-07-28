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
import numpy as np
import torch
from evaluation.real_world_cross_validation_vf_and_paths_evaluation import evaluate_model as evaluate_model
from evaluation.real_world_cross_validation_vf_and_paths_evaluation import get_real_world_data

from fim import project_path
from fim.models import FIMSDE
from fim.models.blocks import AModel
from fim.utils.helper import load_yaml
from fim.utils.logging import RankLoggerAdapter, setup_logging
from fim.utils.sde.evaluation import NumpyEncoder


setup_logging()

warnings.filterwarnings("ignore", module="matplotlib")
logger = RankLoggerAdapter(logging.getLogger(__name__))


def finetune_fim_on_real_world_cross_validation(
    model_path: str,
    inference_data_path: str,
    epochs_to_evaluate: list[int] | None,
    dataset_label: str,
    split: int,
    train_from_scratch: bool,
    base_config: Path,
    exp_name: str,
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

    all_data = json.load(open(inference_data_path, "r"))
    array_dataset: dict = get_real_world_data(all_data, dataset_label, split, expected_num_total_splits=5)
    tensor_dataset = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in array_dataset.items()}
    base_config["dataset"]["json_paths"]["train"] = tensor_dataset
    base_config["dataset"]["json_paths"]["test"] = tensor_dataset  # have not splits available
    base_config["dataset"]["json_paths"]["validation"] = tensor_dataset  # have not splits available

    exp_name_with_dataset_split = f"{exp_name}_dataset_{dataset_label}_split_{split}"
    config, model = finetune_helpers.add_extra_fimsde_configs(base_config, model, exp_name_with_dataset_split, **extra_configs)

    trainer = finetune_helpers.train_fimsde(model, config)
    finetune_helpers.add_model_type_to_checkpoints(trainer)

    checkpoint_dir = Path(project_path) / trainer.checkpointer.checkpoint_dir

    del model
    del trainer

    last_epoch = finetune_helpers.get_last_epoch(checkpoint_dir)

    if epochs_to_evaluate is None:
        epochs_to_evaluate = [last_epoch]

    else:
        if last_epoch not in epochs_to_evaluate:
            epochs_to_evaluate.append(last_epoch)

    epochs_to_evaluate.append("best-model")

    for epoch in epochs_to_evaluate:
        get_real_world_vector_fields_and_paths(checkpoint_dir, epoch, array_dataset, dataset_label, split, exp_name)


def get_real_world_vector_fields_and_paths(checkpoint_dir: Path, epoch: str, dataset: dict, dataset_label: str, split: int, exp_name: str):
    """
    Load checkpoint from epoch, sample paths and evaluate vector fields for the dataset and split.
    Save model outputs as json.
    """
    model_checkpoint_to_evaluate: Path = checkpoint_dir / epoch

    if model_checkpoint_to_evaluate.exists():
        model: FIMSDE = AModel.load_model(model_checkpoint_to_evaluate)

        evaluation_results = evaluate_model(model, dataset)
        model_outputs = {
            "name": dataset_label,
            "split": split,
            "num_total_splits": dataset["num_total_splits"],
            "delta_tau": dataset["delta_tau"],
            "transform": dataset["transform"],
            "synthetic_paths": evaluation_results["sample_paths"],
            "locations": dataset["locations"],
            "drift_at_locations": evaluation_results["estimated_concepts"].drift,
            "diffusion_at_locations": evaluation_results["estimated_concepts"].diffusion,
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
    inference_data_path = Path(
        "/cephfs_projects/foundation_models/data/SDE/test/20250506_real_world_with_5_fold_cross_validation/cross_val_inference_paths.json"
    )

    model_path = "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/600k_drift_deg_3_diff_deg_2_delta_tau_fixed_linear_attn_softmax_no_extra_normalization_and_fix_in_residual_layer_05-06-2300/checkpoints/epoch-139"

    @click.command()
    @click.option("--base-config", "base_config", type=click.Path(exists=True), required=True)
    @click.option("--dataset-label", "dataset_label", type=str, required=True)
    @click.option("--split", "split", type=int, required=True)
    @click.option("--exp-name", "exp_name", type=str, required=True)
    @click.option("--seed", "seed", type=int, required=True)
    @click.option("--epochs-to-evaluate", "epochs_to_evaluate", type=str, required=False)
    @click.option("--detach-diffusion", "detach_diffusion", type=bool, required=False, default=False)
    @click.option("--likelihood", "likelihood", type=bool, required=False, default=False)
    @click.option("--sampling-mse", "sampling_mse", type=bool, required=False, default=False)
    @click.option("--sampling-nll", "sampling_nll", type=bool, required=False, default=False)
    @click.option("--samples-count", "samples_count", type=int, required=False, default=1)
    @click.option("--samples-steps", "samples_steps", type=int, required=False, default=1)
    @click.option("--em-steps", "em_steps", type=int, required=False, default=1)
    @click.option("--num-points", "num_points", type=int, required=False, default=-1)
    @click.option("--epochs", "epochs", type=int, required=False, default=500)
    @click.option("--save-every", "save_every", type=int, required=False, default=100)
    @click.option("--lr", "lr", type=float, required=False, default=1e-5)
    @click.option("--weight-decay", "weight_decay", type=float, required=False, default=0.0)
    @click.option("--train-from-scratch", "train_from_scratch", type=bool, required=False, default=False)
    def cli(epochs_to_evaluate: str = None, **kwargs):
        if epochs_to_evaluate is not None:
            # expect list of comma separated integers
            epochs_to_evaluate: list[str] = epochs_to_evaluate.split(",")
            epochs_to_evaluate = [f"epoch-{e}" for e in epochs_to_evaluate]

        finetune_fim_on_real_world_cross_validation(model_path, inference_data_path, epochs_to_evaluate, **kwargs)

    # pylint: disable=no-value-for-parameter
    cli()
