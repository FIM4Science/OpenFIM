from copy import deepcopy
from pathlib import Path
from typing import Optional

import optree
import torch
import yaml

from fim import data_path, project_path
from fim.data.datasets import FIMSDEDatabatch
from fim.data.utils import save_h5
from fim.data_generation.sde.dynamical_systems import DynamicalSystem
from fim.data_generation.sde.dynamical_systems_sample import PathGenerator, set_up_a_dynamical_system


def get_data_from_dynamical_system(dynamical_system: DynamicalSystem, integration_config: dict, locations_params: dict) -> FIMSDEDatabatch:
    """
    Get samples from dynamical system by means of integration and location configs.
    """
    path_generator = PathGenerator(
        dataset_type="FIMSDEpDataset", system=dynamical_system, integrator_params=integration_config, locations_params=locations_params
    )
    data = path_generator.generate_paths()

    return data


def save_fimsdedatabatch_to_files(databatch: FIMSDEDatabatch, save_dir: Path) -> None:
    """
    Save FIMSDEDatabatch arguments into .h5 files.

    Args:
        databatch (FIMSDEDatabatch): Data to save.
        save_dir (Path): Absolute path to directory to save data at.
    """
    save_dir.mkdir(exist_ok=True, parents=True)

    save_h5(databatch.obs_values, save_dir / "obs_values.h5")
    if "obs_noisy_values" in dir(databatch):
        save_h5(databatch.obs_noisy_values, save_dir / "obs_noisy_values.h5")
    else:
        save_h5(databatch.obs_values, save_dir / "obs_noisy_values.h5")
    save_h5(databatch.obs_times, save_dir / "obs_times.h5")
    save_h5(databatch.obs_mask, save_dir / "obs_mask.h5")
    save_h5(databatch.locations, save_dir / "locations.h5")
    save_h5(databatch.drift_at_locations, save_dir / "drift_at_locations.h5")
    save_h5(databatch.diffusion_at_locations, save_dir / "diffusion_at_locations.h5")
    save_h5(torch.ones_like(databatch.locations), save_dir / "dimension_mask.h5")

    if not hasattr(databatch, "obs_mask") or databatch.obs_mask is None:
        databatch.obs_mask = torch.ones_like(databatch.obs_times)

    save_h5(databatch.obs_mask, save_dir / "obs_mask.h5")


def save_dynamical_system_from_yaml(
    yaml_path: str | Path | dict, labels_to_use: list[str], global_save_dir: str | Path, tr_save_dir=""
) -> None:
    """
    Generate dynamical systems from yaml and save them to disk.

    Args:
        yaml_path (str | Path | dict): Absolute or relative to project path, or content of a yaml.
        labels_to_use (list[str]): Which labels from yaml to generate and save.
        global_save_dir (str | Path): Absolute or relative to data path.
    """
    if isinstance(yaml_path, dict):
        data_config = yaml_path

    else:
        # prepare paths
        yaml_path = Path(yaml_path)
        if not yaml_path.is_absolute():
            yaml_path: Path = Path(project_path) / yaml_path

        # load general config
        with open(yaml_path, "r") as file:
            data_config = yaml.safe_load(file)

    global_save_dir = Path(global_save_dir)
    if not global_save_dir.is_absolute():
        global_save_dir: Path = Path(data_path) / global_save_dir

    global_save_dir.mkdir(parents=True, exist_ok=True)

    dataset_type = data_config["dataset_type"]
    all_integrator_params = data_config["integration"]
    locations_params = data_config["locations"]

    # loop over integration params, if applicable, adding its description to the data bulk name
    if not isinstance(all_integrator_params, list):
        all_integrator_params = [all_integrator_params]

    for label in labels_to_use:
        label_dir: Path = global_save_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        # save subconfig for reference
        with open(label_dir / "config.yaml", "w") as f:
            yaml.dump(data_config[label], f)

        print("WARNING: overlaps in data_bulk_name can lead to overwritten files.")
        for params in data_config[label]:
            # change bulk name and number of realizations per integrator
            params = deepcopy(params)
            num_realizations = int(params["num_realizations"])

            subset_label = params.get("data_bulk_name")
            subset_dir = label_dir / subset_label

            for integrator_params in all_integrator_params:
                integrator_params = deepcopy(integrator_params)

                integrator_label = integrator_params.pop("label", None)
                if integrator_label is None and len(all_integrator_params) > 1:
                    raise ValueError("Multiple integrators require passing a `label` with each of them.")

                # multiple integrators: add integrator label to save directory name
                if integrator_label is not None:
                    # percentage of num_realizations for current integrator
                    perc = integrator_params.pop("percentage", 1)
                    params["num_realizations"] = int(num_realizations * perc)

                    if perc != 1.0:
                        integrator_label = integrator_label + "_num_realiz_" + str(params["num_realizations"])

                    integrator_dir = subset_dir / integrator_label
                    params["data_bulk_name"] = subset_label + "_integrator_" + integrator_label

                else:
                    integrator_dir = deepcopy(subset_dir)

                print(f"Generating {subset_label} with integrator {integrator_label}")
                subset_data: FIMSDEDatabatch = set_up_a_dynamical_system(
                    dataset_type, params, integrator_params, locations_params, tr_save_dir, True
                )

                save_fimsdedatabatch_to_files(subset_data, integrator_dir)
                del subset_data

    # save data config yaml for reference
    with open(global_save_dir / "config.yaml", "w") as f:
        yaml.dump(data_config, f)


def split_databatch_into_paths(databatch: FIMSDEDatabatch, target_path_length: int, reset_time: Optional[bool] = True) -> FIMSDEDatabatch:
    """
    Cut long observation series generated from DynamicalSystem into shorter paths of some target length.

    Args:
        databatch (FIMSDEDatabatch): Data with .obs_values.shape == [B, P, T, D]
        target_path_length (int): (Maximal) length of one output path.
        reset_time (Optional[bool]): Reset first observation time to 0 after splitting.

    Returns:
        databatch (FIMSDEDatabatch): Input data with .obs_values.shape == [B, *, target_path_length, D]
    """
    padding_mask = torch.ones_like(databatch.obs_times).bool()

    def _split_and_pad(tensor: torch.Tensor):
        split_tensors: list[torch.Tensor] = torch.split(tensor, split_size_or_sections=target_path_length, dim=-2)
        split_tensors: list[torch.Tensor] = optree.tree_map(
            lambda x: torch.nn.functional.pad(x, (0, 0, target_path_length - x.size(2), 0), value=0), split_tensors
        )
        return torch.concatenate(split_tensors, dim=1)

    databatch.obs_values = _split_and_pad(databatch.obs_values)
    databatch.obs_times = _split_and_pad(databatch.obs_times)

    if hasattr(databatch, "obs_mask"):
        databatch.obs_mask = _split_and_pad(databatch.obs_mask)

    else:
        databatch.obs_mask = _split_and_pad(padding_mask)

    if reset_time is True:
        masked_obs_times = torch.where(databatch.obs_mask, databatch.obs_times, torch.inf)
        min_ = torch.amin(masked_obs_times, dim=-2, keepdims=True)
        databatch.obs_times = databatch.obs_times - min_

    return databatch
