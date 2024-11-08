from copy import deepcopy
from pathlib import Path
from typing import Optional

import click
import numpy as np
from fim.leftovers_from_old_library import save_in_yaml, create_class_instance, load_config

@click.command()
@click.option(
    "-c",
    "--config",
    "cfg_path",
    required=True,
    type=click.Path(exists=True),
    help="path to config file"
)
def generate_data(cfg_path: Path) -> None:
    cfg: dict = load_config(cfg_path)
    cfg_copy: dict = deepcopy(cfg)

    # prepare dataset path
    dataset_path: Path = Path("data/training_data/pp/" + cfg["dataset_path"])
    # Drop dataset_path from cfg
    cfg.pop("dataset_path")

    # save original config yaml
    save_in_yaml(cfg_copy, dataset_path, "config")

    # assemble datasets (assembler is of type Assembler)
    for dataset_label, assembler in cfg.items():
        assembler.update({"dataset_path": dataset_path})
        create_class_instance(assembler).assemble()


if __name__ == "__main__":
    generate_data()
