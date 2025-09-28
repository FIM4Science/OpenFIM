import json
from copy import deepcopy
from pathlib import Path

import torch

from fim import data_path
from fim.data.datasets import FIMSDEDatabatch
from fim.data.utils import save_h5
from fim.data_generation.sde.dynamical_systems import (
    DoubleWellConstantDiffusion,
    DoubleWellOneDimension,
    DuffingOscillator,
)
from fim.data_generation.sde.dynamical_systems_to_files import (
    get_data_from_dynamical_system,
    save_fimsdedatabatch_to_files,
)


def get_double_well_const_diffusion(num_steps: int, num_realizations: int, num_paths: int):
    process_hyperparameters = {
        "name": "DoubleWellConstantDiffusion",
        "data_bulk_name": "double_well_constant_diff",
        "redo": True,
        "num_realizations": num_realizations,
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
        "num_steps": num_steps,
        "steps_per_dt": 1,
        "num_paths": num_paths,
        "num_locations": 1024,
        "stochastic": True,
    }

    locations_params = {
        "type": "regular_grid",
        "ranges": [[-2, 2]],
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return DoubleWellConstantDiffusion(process_hyperparameters), integration_config, locations_params, config


def get_double_well_state_dep_diffusion(num_steps: int, num_realizations: int, num_paths: int):
    process_hyperparameters = {
        "name": "DoubleWellOneDimension",
        "data_bulk_name": "double_well_state_dep_diff",
        "redo": True,
        "num_realizations": num_realizations,
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
        "num_steps": num_steps,
        "steps_per_dt": 1,
        "num_paths": num_paths,
        "num_locations": 1024,
        "stochastic": True,
    }

    locations_params = {
        "type": "regular_grid",
        "ranges": [[-2, 2]],
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return DoubleWellOneDimension(process_hyperparameters), integration_config, locations_params, config


def get_duffing_oscillator(num_steps: int, num_realizations: int, num_paths: int):
    process_hyperparameters = {
        "name": "DuffDuffingOscillator",
        "data_bulk_name": "duffing_oscillator",
        "redo": True,
        "num_realizations": num_realizations,
        "drift_params": {
            "alpha": {
                "distribution": "fix",
                "fix_value": 1.0,
            },
            "beta": {
                "distribution": "fix",
                "fix_value": 1.0,
            },
            "gamma": {
                "distribution": "fix",
                "fix_value": 0.35,
            },
        },
        "diffusion_params": {
            "g1": {
                "distribution": "fix",
                "fix_value": 1.0,
            },
            "g2": {
                "distribution": "fix",
                "fix_value": 1.0,
            },
        },
        "initial_state": {
            "distribution": "fix",
            "fix_value": [3, 2],
            "activation": None,
        },
    }

    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.002,
        "steps_per_dt": 1,
        "num_steps": num_steps,
        "num_paths": num_paths,
        "num_locations": 1024,
        "stochastic": True,
    }

    locations_params = {
        "type": "regular_grid",
        "ranges": [[-4, 4], [-4, 4]],
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return DuffingOscillator(process_hyperparameters), integration_config, locations_params, config


def sample_initial_states_from_fimsdedatabatch(databatch: FIMSDEDatabatch, num_initial_states: int):
    """
    Returns as set of initial states per realization and path, sampled from the observed on that path.
    """
    obs_values = databatch.obs_values  # [num_realizations, num_paths, num_steps, D]
    perm = torch.randperm(obs_values.shape[-2])
    initial_states = obs_values[:, :, perm, :][:, :, :num_initial_states, :]  # [num_realizations, num_paths, num_initial_states, D]
    return initial_states


if __name__ == "__main__":
    save_dir = Path(data_path) / "processed" / "test" / "20250925_double_well_duffing_for_aisummit"

    # test
    num_steps = 50000  # 100 paths * 500 obs for ksig
    num_paths = 1
    num_realizations = 5
    num_initial_states_from_realization = 1000

    # # opper 1D double well state dependent diffusion
    # print("Opper: Double Well, state dep diff")
    # double_well_state_dep_diff, integration_config, locations_params, config = get_double_well_state_dep_diffusion(
    #     num_steps, num_realizations, num_paths
    # )
    # double_well_data: FIMSDEDatabatch = get_data_from_dynamical_system(double_well_state_dep_diff, integration_config, locations_params)
    #
    # double_well_state_dep_diff_save_dir = save_dir / "Double_Well"
    # save_fimsdedatabatch_to_files(double_well_data, double_well_state_dep_diff_save_dir)
    # with open(double_well_state_dep_diff_save_dir / "config.json", "w") as f:
    #     json.dump(config, f)
    #
    # initial_states = sample_initial_states_from_fimsdedatabatch(double_well_data, num_initial_states_from_realization)
    # save_h5(initial_states, double_well_state_dep_diff_save_dir / "initial_states.h5")

    funcs = {
        "Duffing": get_duffing_oscillator,
        "Double Well Const Diff": get_double_well_const_diffusion,
        "Double Well State Dep Diff": get_double_well_state_dep_diffusion,
    }

    # SDE paths without additive noise
    for name, func in funcs.items():
        print(name)
        system, integration_config, locations_params, config = func(
            num_steps=num_steps, num_realizations=num_realizations, num_paths=num_paths
        )
        data = get_data_from_dynamical_system(system, integration_config, locations_params)
        save_fimsdedatabatch_to_files(data, save_dir / name)

        with open(save_dir / name / "config.json", "w") as f:
            json.dump(config, f)

        initial_states = sample_initial_states_from_fimsdedatabatch(data, num_initial_states_from_realization)
        save_h5(initial_states, save_dir / name / "initial_states.h5")
