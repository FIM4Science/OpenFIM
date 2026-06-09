import argparse
import logging
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoModel

from fim.data.dataloaders import DataLoaderFactory
from fim.models.blocks import ModelFactory
from fim.trainers.trainer import TrainerFactory
#from fim.trainers.utils import cleanup, clear_gpu_cache, get_accel_type, setup, setup_environ_flags
from fim.utils.helper import GenericConfig, expand_params, load_yaml
from fim.utils.logging import RankLoggerAdapter, setup_logging



setup_logging()

warnings.filterwarnings("ignore", module="matplotlib")
logger = RankLoggerAdapter(logging.getLogger(__name__))



def train(configs: List[GenericConfig], resume: Path):
    for config in configs:
        if config.distributed.enabled:
            raise ValueError("This script is not for distributed training!")
        else:
            train_single(config, resume)


def train_single(config: GenericConfig, resume: Path):
    setup_logging()
    warnings.filterwarnings("ignore", module="matplotlib")
    logger = RankLoggerAdapter(logging.getLogger(__name__))
    logger.info("Starting Experiment: %s", config.experiment.name)

    torch.manual_seed(int(config.experiment.seed))
    torch.cuda.manual_seed(int(config.experiment.seed))
    np.random.seed(int(config.experiment.seed))
    torch.cuda.empty_cache()

    # device_map = config.experiment.device_map     # This is only for distributed training

    dataloader = DataLoaderFactory.create(**config.dataset.to_dict())
    if config.model.get("base_model", None) is None:
        logger.info("Creating model from scratch")
        model = ModelFactory.create(config.model.to_dict())
    else:
        logger.info(f"Loading model from {config.model.base_model}")
        model = AutoModel.from_pretrained(config.model.base_model)

    trainer = TrainerFactory.create(config.trainer.name, model=model, dataloader=dataloader, config=config, resume=resume)
    trainer.train()


if __name__ == "__main__":
    """ uv run python experiments/finetune.py -c /path/to/your_config.yaml """
    
    parser = argparse.ArgumentParser(description="Fine-tune model with config file")
    parser.add_argument(
        "-c",
        type=str,
        required=True,
        help="Path to the config YAML file"
    )
    
    args = parser.parse_args()
    
    print("Starting finetuning...")
    config = load_yaml(args.c)
    configs = expand_params(config)

    for cfg in configs:
        print(cfg.experiment.name)
        train_single(cfg, Path("models/base_model_for_finetuning/checkpoints"))
        print()


quit()
if __name__ == "__main__":
    from utils.clear_data_and_configs import clear_data_and_configs
    clear_data_and_configs(verbose=False)

    from data_gen_mocap import data_gen_mocap
    data_gen_mocap()

    from data_gen_vdp_fhn import data_gen_vdp_fhn
    data_gen_vdp_fhn()

    from config_gen import config_gen, generate_configs_for_different_noises
    generate_configs_for_different_noises()

    resume = Path("models/base_model_for_finetuning/checkpoints")
    #config_gen()


    """
    cfg_paths = [
        "experiments/mocap/mocap09long/configs/config_01.yaml",
        "experiments/mocap/mocap09short/configs/config_01.yaml",
        "experiments/mocap/mocap35long/configs/config_01.yaml",
        "experiments/mocap/mocap35short/configs/config_01.yaml",
        "experiments/mocap/mocap39long/configs/config_01.yaml",
        "experiments/mocap/mocap39short/configs/config_01.yaml",
    ]
    """

    #tasks = ["vdp1", "vdp2", "fhn", "mocap/mocap09long", "mocap/mocap09short", "mocap/mocap35long", "mocap/mocap35short", "mocap/mocap39long", "mocap/mocap39short"]
    tasks = ["vdp2"]


    cfg_paths = {task: [] for task in tasks}
    for task in tasks:
        cfg_dir = Path("experiments") / task / "configs"
        
        # Find all .yaml files in the config directory
        if cfg_dir.exists():
            yaml_files = sorted(cfg_dir.glob("*.yaml"))
            print(cfg_dir, ":", len(yaml_files), "configs")
            # Construct full relative paths (experiments/...)
            cfg_paths[task].extend([str(cfg_dir / f.name) for f in yaml_files])


    for task in tasks:
        cfg_path = cfg_paths[task][0]
        config = load_yaml(cfg_path)
        gs_configs = expand_params(config)

        print()
        print()
        print("NOW WE'RE GETTING TO TASK", task, "CONFIG (NOISE) ", 0)
        print()
        print()

        train(gs_configs, resume)
    
    quit()

    for task, task_cfg_paths in cfg_paths.items():
        for i, cfg_path in enumerate(task_cfg_paths):
            config = load_yaml(cfg_path)
            gs_configs = expand_params(config)

            print()
            print()
            print("NOW WE'RE GETTING TO TASK " + task, "CONFIG", i)
            print()
            print()

            train(gs_configs, resume)
    
    quit()
