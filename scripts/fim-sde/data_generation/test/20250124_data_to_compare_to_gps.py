from copy import deepcopy
from pathlib import Path

import torch

from fim import data_path
from fim.data.data_generation.dynamical_systems import (
    DynamicalSystem,
)
from fim.data.datasets import FIMSDEDatabatch
from fim.data_generation.sde.dynamical_systems_to_files import (
    get_data_from_dynamical_system,
)


# general config
STEPS_PER_DT = 1


class Wang2DSynthetic(DynamicalSystem):
    name_str: str = "Wang2DSynthetic"
    state_dim: int = 2

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        x, y = states[:, 0], states[:, 1]
        dxdt = x * (1 - x**2 - y**2) - y
        dydt = y * (1 - x**2 - y**2) + x
        return torch.stack([dxdt, dydt], dim=1)

    def diffusion(self, states, time, params):
        x, y = states[:, 0], states[:, 1]
        return torch.stack([torch.sqrt(1 + y**2), torch.sqrt(1 + x**2)], dim=1)

    def sample_drift_params(self, num_paths):
        return torch.zeros(num_paths, 2)

    def sample_diffusion_params(self, num_paths):
        return torch.zeros(num_paths, 2)

    def sample_initial_states(self, num_paths):
        return 1.5 * torch.ones(num_paths, 2)  # chosen by us as no available


def get_wang_two_dim_snythetic_model(time_step: float, num_steps: int, steps_per_dt: int):
    process_hyperparameters = {
        "name": "Wang2DSynthetic",
        "data_bulk_name": "wang_2D_synthetic",
        "redo": True,
        "num_realizations": 1,
        "drift_params": {},
        "diffusion_params": {},
        "initial_state": {},
    }

    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.025,
        "num_steps": 80000,
        "steps_per_dt": STEPS_PER_DT,
        "num_paths": 1,
        "num_locations": 1024,
        "stochastic": True,
    }

    locations_params = {
        "type": "regular_grid",
        "ranges": [[-4, 4], [-3, 3]],  # should be regular grid
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return Wang2DSynthetic(process_hyperparameters), integration_config, locations_params, config


class WangDoubleWell(DynamicalSystem):
    name_str: str = "WangDoubleWell"
    state_dim: int = 1

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        x = states
        return x - x**3

    def diffusion(self, states, time, params):
        x = states
        return torch.sqrt(1 + x**2)

    def sample_drift_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_diffusion_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_initial_states(self, num_paths):
        return 1 * torch.ones(num_paths, 1)


def get_wang_double_well():
    process_hyperparameters = {
        "name": "WangDoubleWell",
        "data_bulk_name": "wang_double_well",
        "redo": True,
        "num_realizations": 1,
        "drift_params": {},
        "diffusion_params": {},
        "initial_state": {},
    }

    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.04,
        "num_steps": 25000,
        "steps_per_dt": STEPS_PER_DT,
        "num_paths": 1,
        "num_locations": 1024,
        "stochastic": True,
    }

    locations_params = {
        "type": "regular_grid",
        "ranges": [[-4, 4]],  # should be regular grid
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return WangDoubleWell(process_hyperparameters), integration_config, locations_params, config


if __name__ == "__main__":
    save_dir = Path(data_path) / "processed" / "test" / "20241224_data_to_compare_to_gps"

    # wang two-dimensional synthetic model
    print("Wang: 2D")
    two_d_wang, integration_config, locations_params, config = get_wang_two_dim_snythetic_model()
    two_d_wang_data: FIMSDEDatabatch = get_data_from_dynamical_system(two_d_wang, integration_config, locations_params)

    two_d_wang_save_dir = save_dir / "two_d_wang_5000_points"
    # save_fimsdedatabatch_to_files(split_two_d_wang_data, two_d_wang_save_dir)
    # with open(two_d_wang_save_dir / "config.json", "w") as f:
    #     json.dump(config, f)
    #
    # # wang double well
    # print("Wang: Double Well")
    # double_well_wang, integration_config, locations_params, config = get_wang_double_well()
    # double_well_wang_data: FIMSDEDatabatch = get_data_from_dynamical_system(double_well_wang, integration_config, locations_params)
    # split_double_well_wang_data: FIMSDEDatabatch = split_databatch_into_paths(double_well_wang_data, target_path_length)
    #
    # double_well_wang_save_dir = save_dir / "double_well_wang_25000_points"
    # save_fimsdedatabatch_to_files(split_double_well_wang_data, double_well_wang_save_dir)
    # with open(double_well_wang_save_dir / "config.json", "w") as f:
    #     json.dump(config, f)
