import json
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
    save_fimsdedatabatch_to_files,
)


# general config
STEPS_PER_DT = 1000


class SignDiffusion(DynamicalSystem):
    name_str: str = "Sign"
    state_dim: int = 1

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        return torch.zeros_like(states)

    def diffusion(self, states, time, params):
        diffusion_level: float = self.diffusion_params.get("diffusion_level")
        return diffusion_level * torch.where(states == 0.0, -1 * torch.ones_like(states), torch.sign(states))

    def sample_drift_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_diffusion_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_initial_states(self, num_paths):
        return torch.zeros(num_paths, 1)


def get_sign_diffusion(num_paths, diffusion_level):
    process_hyperparameters = {
        "name": "SignDiffusion",
        "data_bulk_name": "difficult_sign_diff",
        "redo": True,
        "num_realizations": 1,
        "drift_params": {},
        "diffusion_params": {"diffusion_level": diffusion_level},
        "initial_state": {},
    }

    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.007812,
        "num_steps": 127,  # total 128 observations per path
        "steps_per_dt": STEPS_PER_DT,
        "num_paths": num_paths,
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

    return SignDiffusion(process_hyperparameters), integration_config, locations_params, config


class InverseDrift(DynamicalSystem):
    name_str: str = "InverseDrift"
    state_dim: int = 1

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        is_non_zero = torch.logical_not(torch.isclose(states, torch.zeros_like(states)))

        dxdt = -1 / (2 * states)
        dxdt = torch.where(is_non_zero, dxdt, torch.zeros_like(dxdt))
        return dxdt

    def diffusion(self, states, time, params):
        diffusion_level: float = self.diffusion_params.get("diffusion_level")
        return diffusion_level * torch.ones_like(states)

    def sample_drift_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_diffusion_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_initial_states(self, num_paths):
        return torch.zeros(num_paths, 1)


def get_inverse_drift(num_paths, diffusion_level):
    process_hyperparameters = {
        "name": "InverseDrift",
        "data_bulk_name": "inverse_drift",
        "redo": True,
        "num_realizations": 1,
        "drift_params": {},
        "diffusion_params": {"diffusion_level": diffusion_level},
        "initial_state": {},
    }

    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.007812,
        "num_steps": 127,  # total 128 observations per path
        "steps_per_dt": STEPS_PER_DT,
        "num_paths": num_paths,
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

    return InverseDrift(process_hyperparameters), integration_config, locations_params, config


class Exp(DynamicalSystem):
    name_str: str = "Exp"
    state_dim: int = 1

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        is_non_zero = torch.logical_not(torch.isclose(states, torch.zeros_like(states)))

        dxdt = torch.exp(-1 / states**2)
        dxdt = torch.where(is_non_zero, dxdt, 1)
        return dxdt

    def diffusion(self, states, time, params):
        diffusion_level: float = self.diffusion_params.get("diffusion_level")
        return diffusion_level * torch.ones_like(states)

    def sample_drift_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_diffusion_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_initial_states(self, num_paths):
        return torch.zeros(num_paths, 1)


def get_exp(num_paths, diffusion_level):
    process_hyperparameters = {
        "name": "Exp",
        "data_bulk_name": "exp",
        "redo": True,
        "num_realizations": 1,
        "drift_params": {},
        "diffusion_params": {"diffusion_level": diffusion_level},
        "initial_state": {},
    }

    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.007812,
        "num_steps": 127,  # total 128 observations per path
        "steps_per_dt": STEPS_PER_DT,
        "num_paths": num_paths,
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

    return Exp(process_hyperparameters), integration_config, locations_params, config


if __name__ == "__main__":
    save_dir = Path(data_path) / "processed" / "test" / "20250114_difficult_non_polynomial_equations"

    all_num_paths = [50, 100, 300]
    all_diffusion_levels = [0.1, 0.3, 0.5, 1.0]

    for num_paths in all_num_paths:
        print("Number of paths: ", str(num_paths))

        for diffusion_level in all_diffusion_levels:
            print("Diffusion Level: ", str(diffusion_level))

            current_save_dir = save_dir / ("num_paths_" + str(num_paths)) / ("diffusion_level_" + str(diffusion_level))

            # sign diffusion
            print("sign diffusion")
            sign_diffusion, integration_config, locations_params, config = get_sign_diffusion(num_paths, diffusion_level)
            sign_diffusion_data: FIMSDEDatabatch = get_data_from_dynamical_system(sign_diffusion, integration_config, locations_params)

            sign_diffusion_save_dir = current_save_dir / "sign_diffusion"
            save_fimsdedatabatch_to_files(sign_diffusion_data, sign_diffusion_save_dir)
            with open(sign_diffusion_save_dir / "config.json", "w") as f:
                json.dump(config, f)

            # inverse drift
            print("inverse drift")
            inverse_drift, integration_config, locations_params, config = get_inverse_drift(num_paths, diffusion_level)
            inverse_drift_data: FIMSDEDatabatch = get_data_from_dynamical_system(inverse_drift, integration_config, locations_params)

            inverse_drift_save_dir = current_save_dir / "inverse_drift"
            save_fimsdedatabatch_to_files(inverse_drift_data, inverse_drift_save_dir)
            with open(inverse_drift_save_dir / "config.json", "w") as f:
                json.dump(config, f)

            # exp
            print("exp")
            exp, integration_config, locations_params, config = get_exp(num_paths, diffusion_level)
            exp_data: FIMSDEDatabatch = get_data_from_dynamical_system(exp, integration_config, locations_params)

            exp_save_dir = current_save_dir / "exp_-1_xx"
            save_fimsdedatabatch_to_files(exp_data, exp_save_dir)
            with open(exp_save_dir / "config.json", "w") as f:
                json.dump(config, f)
