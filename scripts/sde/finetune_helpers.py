import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoConfig

from fim import project_path
from fim.data.dataloaders import DataLoaderFactory
from fim.models import FIMSDE
from fim.models.blocks import AModel, ModelFactory
from fim.trainers.trainer import Trainer, TrainerFactory
from fim.trainers.utils import cleanup, clear_gpu_cache, get_accel_type, setup, setup_environ_flags
from fim.utils.helper import expand_params


def load_pretrained_model(model_path: str, train_from_scratch: bool) -> FIMSDE:
    """
    From a checkpoint path, load the model and pretrained weights, or a new instance of model.

    Args:
        model_path (str): Absolute path to `.../checkpoints/epoch-...`.
        train_from_scratch (bool): Flag to load pretrained weights.

    Returns:
        model (FIMSDE): Loaded instance of FIMSDE.
    """
    if train_from_scratch is False:
        model: FIMSDE = AModel.load_model(model_path)

    else:
        model_config = AutoConfig.from_pretrained(Path(model_path) / "config.json")
        model = ModelFactory.create(model_config)

    return model


def add_extra_fimsde_configs(
    config: dict,
    model: FIMSDE,
    exp_name: str,
    seed: int,
    detach_diffusion: bool,
    likelihood: bool,
    sampling_mse: bool,
    sampling_nll: bool,
    num_points: int,
    samples_count: int,
    samples_steps: int,
    em_steps: int,
    epochs: int,
    save_every: int,
    lr: float,
    weight_decay: float,
):
    """
    Adapt or add configs to a config dict (for trainer and dataloader) and model.
    """
    config["experiment"]["seed"] = seed
    config["experiment"]["name"] = exp_name

    optimizer = config["optimizers"][0]["optimizer_d"]
    optimizer["lr"] = lr
    optimizer["weight_decay"] = weight_decay
    config["optimizers"] = ({"optimizer_d": optimizer},)

    config["trainer"]["epochs"] = epochs
    config["trainer"]["save_every"] = save_every

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

    return config, model


def train_fimsde(model: FIMSDE, config: dict):
    """
    Train a FIMSDE model using trainer and dataloader configured by config.
    """
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

    return trainer


def add_model_type_to_checkpoints(trainer: Trainer) -> None:
    """
    Trainer does not add 'model_type' to config before saving.
    Loading checkpoints requires 'model_type'. Thus, add it to configs in all checkpoints.
    """
    checkpoint_dir = Path(project_path) / trainer.checkpointer.checkpoint_dir

    for checkpoint_name in [item.name for item in checkpoint_dir.iterdir() if item.is_dir()]:
        config_path = checkpoint_dir / checkpoint_name / "config.json"
        config = json.load(open(config_path, "r"))
        config["model_type"] = trainer.model.config.model_type
        json.dump(config, open(config_path, "w"))


def get_last_epoch(checkpoint_dir: Path):
    """
    Return latest checkpoint in a list of checkpoints (which are named `epoch-X`).
    """
    epoch_numbers = []
    for checkpoint_name in [item.name for item in checkpoint_dir.iterdir() if item.is_dir()]:
        match = re.match(r"^epoch-(\d+)$", checkpoint_name)
        if match:
            epoch_numbers.append(int(match.group(1)))

    last_epoch = f"epoch-{max(epoch_numbers)}"

    return last_epoch
