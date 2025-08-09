from pathlib import Path

import click
import yaml

from fim.data_generation.sde.dynamical_systems_to_files import save_dynamical_system_from_yaml


@click.command()
@click.option("--generation_label", "generation_label", type=str, required=True)
@click.option("--project_path", "project_path", type=str, required=True)
@click.option("--data_path", "data_path", type=str, required=True)
@click.option("--yaml_path", "yaml_path", type=str, required=True)
@click.option("--index", "index", type=int, required=True)
def main(generation_label: str, project_path: str, data_path: str, yaml_path: str, index: int):
    yaml_path = Path(yaml_path)
    if not yaml_path.is_absolute():
        yaml_path: Path = Path(project_path) / yaml_path

    with open(yaml_path, "r") as file:
        data_config = yaml.safe_load(file)

    configs_in_label: dict[str, list] = {
        "train": data_config.pop("train", None),
        "test": data_config.pop("test", None),
        "validation": data_config.pop("validation", None),
    }

    all_setups: list[tuple] = []

    for label, configs in configs_in_label.items():
        if configs is not None:
            for config in configs:
                all_setups.append((label, config))

    if index < len(all_setups):
        setup = all_setups[index]
        label, config = setup

        data_config[label] = [config]
        breakpoint

        tr_save_dir = Path(data_path) / "save_dynamical_system_tr" / generation_label
        tr_save_dir.mkdir(exist_ok=True, parents=True)

        global_save_dir = Path(data_path) / "processed" / "train" / generation_label
        tr_save_dir.mkdir(exist_ok=True, parents=True)

        save_dynamical_system_from_yaml(data_config, labels_to_use=[label], global_save_dir=global_save_dir, tr_save_dir=tr_save_dir)


if __name__ == "__main__":
    # dummy script to generate one setup in yaml, indicated by some index
    # # meant to be called by a bash script to run in different slum jobs in parallel
    main()
