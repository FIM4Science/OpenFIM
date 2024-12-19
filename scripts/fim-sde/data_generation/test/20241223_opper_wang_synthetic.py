import json
from copy import deepcopy
from pathlib import Path

import torch
from torch import Tensor

from fim import data_path
from fim.data.data_generation.dynamical_systems import (
    DoubleWellOneDimension,
    DynamicalSystem,
    Lorenz63System,
)
from fim.data.datasets import FIMSDEDatabatch
from fim.data_generation.sde.dynamical_systems_to_files import (
    get_data_from_dynamical_system,
    save_fimsdedatabatch_to_files,
    split_databatch_into_paths,
)


# general config
STEPS_PER_DT = 10


def get_lorenz_63_opper():
    process_hyperparameters = {
        "name": "Lorenz63System",
        "data_bulk_name": "lorenzt_63_opper",
        "redo": True,
        "num_realizations": 1,
        "drift_params": {
            "sigma": {
                "distribution": "fix",
                "fix_value": 10.0,
            },
            "beta": {
                "distribution": "fix",
                "fix_value": 2.66,
            },
            "rho": {
                "distribution": "fix",
                "fix_value": 28.0,
            },
        },
        "diffusion_params": {
            "constant_value": 1,
            "dimensions": 3,
        },
        "initial_state": {
            "distribution": "fix",
            "fix_value": [-8, 7, 27],
            "activation": None,
        },
    }

    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.2,
        "num_steps": 3000,
        "steps_per_dt": STEPS_PER_DT,
        "num_paths": 1,
        "num_locations": 1024,
        "stochastic": True,
    }

    locations_params = {
        "type": "regular_cube",
        "extension_perc": 0.3,
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return Lorenz63System(process_hyperparameters), integration_config, locations_params, config


class Opper2DSynthetic(DynamicalSystem):
    name_str: str = "Opper2DSynthetic"
    state_dim: int = 2

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        x, y = states[:, 0], states[:, 1]
        dxdt = x * (1 - x**2 - y**2) - y
        dydt = y * (1 - x**2 - y**2) + x
        return torch.stack([dxdt, dydt], dim=1)

    def diffusion(self, states, time, params):
        return torch.ones_like(states)

    def sample_drift_params(self, num_paths):
        return torch.zeros(num_paths, 2)

    def sample_diffusion_params(self, num_paths):
        return torch.zeros(num_paths, 2)

    def sample_initial_states(self, num_paths):
        return 1.5 * torch.ones(num_paths, 2)  # chosen by us as no available


def get_opper_two_dim_snythetic_model():
    process_hyperparameters = {
        "name": "Opper2DSynthetic",
        "data_bulk_name": "opper_2D_synthetic",
        "redo": True,
        "num_realizations": 1,
        "drift_params": {},
        "diffusion_params": {},
        "initial_state": {},
    }

    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.2,
        "num_steps": 10000,
        "steps_per_dt": STEPS_PER_DT,
        "num_paths": 1,
        "num_locations": 1024,
        "stochastic": True,
    }

    locations_params = {
        "type": "regular_cube",
        "extension_perc": 0.3,
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return Opper2DSynthetic(process_hyperparameters), integration_config, locations_params, config


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


def get_wang_two_dim_snythetic_model():
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
        "type": "regular_cube",
        "extension_perc": 0.3,
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return Wang2DSynthetic(process_hyperparameters), integration_config, locations_params, config


class DoubleWellConstantDiffusion(DoubleWellOneDimension):
    name_str: str = "DoubleWellConstantDiffusion"
    state_dim: int = 1

    def __init__(self, config: dict):
        super().__init__(config)

    def diffusion(self, states, time, params) -> Tensor:
        return params * torch.ones_like(states)

    def sample_diffusion_params(self, num_paths) -> Tensor:
        const = self.diffusion_params["constant"]
        return const * torch.ones(num_paths, 1)


def get_double_well_const_diffusion():
    process_hyperparameters = {
        "name": "DoubleWellConstantDiffusion",
        "data_bulk_name": "double_well_constant_diff",
        "redo": True,
        "num_realizations": 1,
        "drift_params": {
            "alpha": {
                "distribution": "fix",
                "fix_value": 4,
            },
            "beta": {
                "distribution": "fix",
                "fix_value": 4,
            },
        },
        "diffusion_params": {
            "constant": 1,
        },
        "initial_state": {
            "distribution": "fix",
            "fix_value": [0.0],  # opper initial state unknown
            "activation": None,
        },
    }

    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.002,
        "num_steps": 5000,
        "steps_per_dt": STEPS_PER_DT,
        "num_paths": 1,
        "num_locations": 1024,
        "stochastic": True,
    }

    locations_params = {
        "type": "regular_cube",
        "extension_perc": 0.3,
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return DoubleWellConstantDiffusion(process_hyperparameters), integration_config, locations_params, config


def get_double_well_state_dep_diffusion():
    process_hyperparameters = {
        "name": "DoubleWellOneDimension",
        "data_bulk_name": "double_well_state_dep_diff",
        "redo": True,
        "num_realizations": 1,
        "drift_params": {
            "alpha": {
                "distribution": "fix",
                "fix_value": 4,
            },
            "beta": {
                "distribution": "fix",
                "fix_value": 4,
            },
        },
        "diffusion_params": {
            "g1": {
                "distribution": "fix",
                "fix_value": 4,
            },
            "g2": {
                "distribution": "fix",
                "fix_value": 1.25,
            },
        },
        "initial_state": {
            "distribution": "fix",
            "fix_value": [0.0],  # opper initial state unknown
            "activation": None,
        },
    }

    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.002,
        "num_steps": 5000,
        "steps_per_dt": STEPS_PER_DT,
        "num_paths": 1,
        "num_locations": 1024,
        "stochastic": True,
    }

    locations_params = {
        "type": "regular_cube",
        "extension_perc": 0.3,
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return DoubleWellOneDimension(process_hyperparameters), integration_config, locations_params, config


if __name__ == "__main__":
    save_dir = Path(data_path) / "processed" / "test" / "20241223_opper_and_wang_cut_to_128_lenght_paths"
    target_path_length = 128

    # opper lorenz
    lorenz, integration_config, locations_params, config = get_lorenz_63_opper()
    lorenz_data: FIMSDEDatabatch = get_data_from_dynamical_system(lorenz, integration_config, locations_params)
    split_lorenz_data: FIMSDEDatabatch = split_databatch_into_paths(lorenz_data, target_path_length)

    lorenz_dir = save_dir / "lorenz_3000_points"

    save_fimsdedatabatch_to_files(split_lorenz_data, lorenz_dir)
    with open(lorenz_dir / "config.json", "w") as f:
        json.dump(config, f)

    # opper two-dimensional synthetic model
    two_d_opper, integration_config, locations_params, config = get_opper_two_dim_snythetic_model()
    two_d_opper_data: FIMSDEDatabatch = get_data_from_dynamical_system(two_d_opper, integration_config, locations_params)
    split_two_d_opper_data: FIMSDEDatabatch = split_databatch_into_paths(two_d_opper_data, target_path_length)

    two_d_opper_save_dir = save_dir / "two_d_opper_10000_points"
    save_fimsdedatabatch_to_files(split_two_d_opper_data, two_d_opper_save_dir)
    with open(two_d_opper_save_dir / "config.json", "w") as f:
        json.dump(config, f)

    # wang two-dimensional synthetic model
    two_d_wang, integration_config, locations_params, config = get_wang_two_dim_snythetic_model()
    two_d_wang_data: FIMSDEDatabatch = get_data_from_dynamical_system(two_d_wang, integration_config, locations_params)
    split_two_d_wang_data: FIMSDEDatabatch = split_databatch_into_paths(two_d_wang_data, target_path_length)

    two_d_wang_save_dir = save_dir / "two_d_wang_80000_points"
    save_fimsdedatabatch_to_files(split_two_d_wang_data, two_d_wang_save_dir)
    with open(two_d_wang_save_dir / "config.json", "w") as f:
        json.dump(config, f)

    # opper 1D double well constant diffusion
    double_well_const_diff, integration_config, locations_params, config = get_double_well_const_diffusion()
    double_well_data: FIMSDEDatabatch = get_data_from_dynamical_system(double_well_const_diff, integration_config, locations_params)
    split_double_well_data: FIMSDEDatabatch = split_databatch_into_paths(double_well_data, target_path_length)

    double_well_const_diff_save_dir = save_dir / "double_well_constant_diff_5000_points"
    save_fimsdedatabatch_to_files(split_double_well_data, double_well_const_diff_save_dir)
    with open(double_well_const_diff_save_dir / "config.json", "w") as f:
        json.dump(config, f)

    # opper 1D double well state dependent diffusion
    double_well_state_dep_diff, integration_config, locations_params, config = get_double_well_state_dep_diffusion()
    double_well_data: FIMSDEDatabatch = get_data_from_dynamical_system(double_well_state_dep_diff, integration_config, locations_params)
    split_double_well_data: FIMSDEDatabatch = split_databatch_into_paths(double_well_data, target_path_length)

    double_well_state_dep_diff_save_dir = save_dir / "double_well_state_dep_diff_5000_points"
    save_fimsdedatabatch_to_files(split_double_well_data, double_well_state_dep_diff_save_dir)
    with open(double_well_state_dep_diff_save_dir / "config.json", "w") as f:
        json.dump(config, f)
