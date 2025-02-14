import json
from copy import deepcopy
from pathlib import Path

import torch

from fim import data_path
from fim.data.datasets import FIMSDEDatabatch
from fim.data_generation.sde.dynamical_systems import (
    DynamicalSystem,
    Lorenz63System,
)
from fim.data_generation.sde.dynamical_systems_to_files import (
    get_data_from_dynamical_system,
    save_fimsdedatabatch_to_files,
)


# general config
STEPS_PER_DT = 1000
NUM_PATHS = 500


class OrnsteinUhlenbeck(DynamicalSystem):
    name_str: str = "OrnsteinUhlenbeck"
    state_dim: int = 1

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        if time is None:  # s.t. evaluating at locations for data generation works
            time = torch.zeros_like(states[:, 0][:, None])

        mu, theta = params[:, 0][:, None], params[:, 1][:, None]
        dxdt = mu * time - theta * states
        return dxdt

    def diffusion(self, states, time, params):
        sigma = params
        return sigma

    def sample_drift_params(self, num_paths):
        mu = self.drift_params["mu"] * torch.ones(num_paths)
        theta = self.drift_params["theta"] * torch.ones(num_paths)
        return torch.stack([mu, theta], axis=-1)

    def sample_diffusion_params(self, num_paths):
        sigma = self.diffusion_params["sigma"] * torch.ones(num_paths, 1)
        return sigma

    def sample_initial_states(self, num_paths):
        return torch.randn(num_paths, 1)


def get_scdp_orn_uhl():
    process_hyperparameters = {
        "name": "OrnsteinUhlenbeck",
        "data_bulk_name": "cspd_orn_uhl",
        "redo": True,
        "num_realizations": 1,
        "drift_params": {"mu": 0.02, "theta": 0.1},
        "diffusion_params": {"sigma": 0.4},
        "initial_state": {},
    }

    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 1,
        "num_steps": 64,
        "steps_per_dt": STEPS_PER_DT,
        "num_paths": NUM_PATHS,
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

    return OrnsteinUhlenbeck(process_hyperparameters), integration_config, locations_params, config


class CIR(DynamicalSystem):
    name_str: str = "CIR"
    state_dim: int = 1

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        a, b = params[:, 0][:, None], params[:, 1][:, None]

        dxdt = a * (b - states)
        return dxdt

    def diffusion(self, states, time, params):
        sigma = params
        return sigma * torch.sqrt(states)

    def sample_drift_params(self, num_paths):
        a = self.drift_params["a"] * torch.ones(num_paths)
        b = self.drift_params["b"] * torch.ones(num_paths)
        return torch.stack([a, b], axis=-1)

    def sample_diffusion_params(self, num_paths):
        sigma = self.diffusion_params["sigma"] * torch.ones(num_paths, 1)
        return sigma

    def sample_initial_states(self, num_paths):
        return torch.abs(torch.randn(num_paths, 1))


def get_scdp_cir():
    process_hyperparameters = {
        "name": "OrnsteinUhlenbeck",
        "data_bulk_name": "cspd_cir",
        "redo": True,
        "num_realizations": 1,
        "drift_params": {"a": 1, "b": 1.2},
        "diffusion_params": {"sigma": 0.2},
        "initial_state": {},
    }

    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 1,
        "num_steps": 64,
        "steps_per_dt": STEPS_PER_DT,
        "num_paths": NUM_PATHS,
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

    return CIR(process_hyperparameters), integration_config, locations_params, config


def get_scdp_lorenz():
    process_hyperparameters = {
        "name": "Lorenz63System",
        "data_bulk_name": "lorenzt",
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
            "constant_value": 0,
            "dimensions": 3,
        },
        "initial_state": {
            "distribution": "normal",
            "mean": 0,
            "std_dev": 10,
            "activation": None,
        },
    }

    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.02,
        "num_steps": 100,
        "steps_per_dt": STEPS_PER_DT,
        "num_paths": NUM_PATHS,
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


class LotkaVolterra(DynamicalSystem):
    name_str: str = "LotkaVolterra"
    state_dim: int = 2

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        x, y = states[:, 0], states[:, 1]
        dxdt = 2 / 3 * x - 2 / 3 * x * y
        dydt = x * y - y
        return torch.stack([dxdt, dydt], dim=-1)

    def diffusion(self, states, time, params):
        return torch.zeros_like(states)

    def sample_drift_params(self, num_paths):
        return torch.ones(num_paths, 1)

    def sample_diffusion_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_initial_states(self, num_paths):
        return torch.abs(torch.randn(num_paths, 2))


def get_scdp_lotka_volterra():
    process_hyperparameters = {
        "name": "LotkaVolterra",
        "data_bulk_name": "cspd_lotka_volterra",
        "redo": True,
        "num_realizations": 1,
        "drift_params": {},
        "diffusion_params": {},
        "initial_state": {},
    }

    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.15625,
        "num_steps": 64,
        "steps_per_dt": STEPS_PER_DT,
        "num_paths": NUM_PATHS,
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

    return LotkaVolterra(process_hyperparameters), integration_config, locations_params, config


class Sink(DynamicalSystem):
    name_str: str = "Sink"
    state_dim: int = 2

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        A = torch.Tensor([[-4, 10], [-3, 2]]).to(states.device)
        return states @ A

    def diffusion(self, states, time, params):
        return torch.zeros_like(states)

    def sample_drift_params(self, num_paths):
        return torch.ones(num_paths, 1)

    def sample_diffusion_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_initial_states(self, num_paths):
        return torch.randn(num_paths, 2)


def get_scdp_sink():
    process_hyperparameters = {
        "name": "Sink",
        "data_bulk_name": "cspd_sink",
        "redo": True,
        "num_realizations": 1,
        "drift_params": {},
        "diffusion_params": {},
        "initial_state": {},
    }

    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.046875,
        "num_steps": 64,
        "steps_per_dt": STEPS_PER_DT,
        "num_paths": NUM_PATHS,
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

    return Sink(process_hyperparameters), integration_config, locations_params, config


if __name__ == "__main__":
    save_dir = Path(data_path) / "processed" / "test" / "20250106_scdp_synthetic_datasets"

    # ornstein uhlenbeck
    print("ornstein uhlenbeck")
    scdp_orn_uhl, integration_config, locations_params, config = get_scdp_orn_uhl()
    scdp_orn_uhl_data: FIMSDEDatabatch = get_data_from_dynamical_system(scdp_orn_uhl, integration_config, locations_params)

    scdp_orn_uhl_save_dir = save_dir / "orn_uhl_500_paths"
    save_fimsdedatabatch_to_files(scdp_orn_uhl_data, scdp_orn_uhl_save_dir)
    with open(scdp_orn_uhl_save_dir / "config.json", "w") as f:
        json.dump(config, f)

    # cir
    print("cir")
    scdp_cir, integration_config, locations_params, config = get_scdp_cir()
    scdp_cir_data: FIMSDEDatabatch = get_data_from_dynamical_system(scdp_cir, integration_config, locations_params)

    scdp_cir_save_dir = save_dir / "cir_500_paths"
    save_fimsdedatabatch_to_files(scdp_cir_data, scdp_cir_save_dir)
    with open(scdp_cir_save_dir / "config.json", "w") as f:
        json.dump(config, f)

    # lorenz
    print("lorenz")
    scdp_lorenz, integration_config, locations_params, config = get_scdp_lorenz()
    scdp_lorenz_data: FIMSDEDatabatch = get_data_from_dynamical_system(scdp_lorenz, integration_config, locations_params)

    scdp_lorenz_save_dir = save_dir / "lorenz_500_paths"
    save_fimsdedatabatch_to_files(scdp_lorenz_data, scdp_lorenz_save_dir)
    with open(scdp_lorenz_save_dir / "config.json", "w") as f:
        json.dump(config, f)

    # lotka_volterra
    print("lotka volterra")
    scdp_lotka_volterra, integration_config, locations_params, config = get_scdp_lotka_volterra()
    scdp_lotka_volterra_data: FIMSDEDatabatch = get_data_from_dynamical_system(scdp_lotka_volterra, integration_config, locations_params)

    scdp_lotka_volterra_save_dir = save_dir / "lotka_volterra_500_paths"
    save_fimsdedatabatch_to_files(scdp_lotka_volterra_data, scdp_lotka_volterra_save_dir)
    with open(scdp_lotka_volterra_save_dir / "config.json", "w") as f:
        json.dump(config, f)

    # sink
    print("sink")
    scdp_sink, integration_config, locations_params, config = get_scdp_sink()
    scdp_sink_data: FIMSDEDatabatch = get_data_from_dynamical_system(scdp_sink, integration_config, locations_params)

    scdp_sink_save_dir = save_dir / "sink_500_paths"
    save_fimsdedatabatch_to_files(scdp_sink_data, scdp_sink_save_dir)
    with open(scdp_sink_save_dir / "config.json", "w") as f:
        json.dump(config, f)
