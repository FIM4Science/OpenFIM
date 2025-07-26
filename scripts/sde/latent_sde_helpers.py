import numpy as np
import torch

from fim.data.dataloaders import DataLoaderFactory
from fim.models.blocks import ModelFactory
from fim.trainers.trainer import Trainer, TrainerFactory
from fim.utils.helper import expand_params


def add_extra_latentsde_configs(
    config: dict,
    exp_name: str,
    seed: int,
    context_size: int,
    hidden_size: int,
    latent_size: int,
    activation: str,
    learn_projection: bool,
    solver_dt: float,
    kl_scale_end_step: int,
    lr: float,
    lr_gamma: float,
):
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

    return config


def train_latent_sde(config: dict) -> Trainer:
    """
    Train Latent SDE model from config dict loaded from yaml.
    """
    gs_config = expand_params(config)[0]

    torch.manual_seed(int(gs_config.experiment.seed))
    torch.cuda.manual_seed(int(gs_config.experiment.seed))
    np.random.seed(int(gs_config.experiment.seed))
    torch.cuda.empty_cache()

    dataloader = DataLoaderFactory.create(**gs_config.dataset.to_dict())
    model = ModelFactory.create(gs_config.model.to_dict())

    trainer = TrainerFactory.create(gs_config.trainer.name, model=model, dataloader=dataloader, config=gs_config)
    trainer.train()

    return trainer
