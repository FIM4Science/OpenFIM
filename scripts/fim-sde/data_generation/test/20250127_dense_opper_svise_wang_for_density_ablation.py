import json
from copy import deepcopy
from pathlib import Path

import torch

from fim import data_path
from fim.data.data_generation.dynamical_systems import (
    DampedCubicOscillatorSystem,
    DampedLinearOscillatorSystem,
    DoubleWellConstantDiffusion,
    DoubleWellOneDimension,
    DuffingOscillator,
    HopfBifurcation,
    Opper2DSynthetic,
    SelkovGlycosis,
    Wang2DSynthetic,
    WangDoubleWell,
)
from fim.data.datasets import FIMSDEDatabatch
from fim.data.utils import save_h5
from fim.data_generation.sde.dynamical_systems_to_files import (
    get_data_from_dynamical_system,
    save_fimsdedatabatch_to_files,
)


def get_opper_two_dim_snythetic_model(num_steps: int, num_realizations: int, num_paths: int):
    process_hyperparameters = {
        "name": "Opper2DSynthetic",
        "data_bulk_name": "opper_2D_synthetic",
        "redo": True,
        "num_realizations": num_realizations,
        "drift_params": {},
        "diffusion_params": {},
        "initial_state": {},
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
        "ranges": [[-2, 2], [-2, 2]],
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return Opper2DSynthetic(process_hyperparameters), integration_config, locations_params, config


def get_wang_two_dim_snythetic_model(num_steps: int, num_realizations: int, num_paths: int):
    process_hyperparameters = {
        "name": "Wang2DSynthetic",
        "data_bulk_name": "wang_2D_synthetic",
        "redo": True,
        "num_realizations": num_realizations,
        "drift_params": {},
        "diffusion_params": {},
        "initial_state": {},
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
        "ranges": [[-4, 4], [-4, 4]],  # should be regular grid
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return Wang2DSynthetic(process_hyperparameters), integration_config, locations_params, config


def get_wang_double_well(num_steps: int, num_realizations: int, num_paths: int):
    process_hyperparameters = {
        "name": "WangDoubleWell",
        "data_bulk_name": "wang_double_well",
        "redo": True,
        "num_realizations": num_realizations,
        "drift_params": {},
        "diffusion_params": {},
        "initial_state": {},
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
        "ranges": [[-2, 2], [-2, 2]],
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return WangDoubleWell(process_hyperparameters), integration_config, locations_params, config


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


def get_damped_linear_oscillator(num_steps: int, num_realizations: int, num_paths: int):
    process_hyperparameters = {
        "name": "DampedLinearOscillatorSystem",
        "data_bulk_name": "damped_linear_oscillator",
        "redo": True,
        "num_realizations": num_realizations,
        "drift_params": {
            "damping": {
                "distribution": "fix",
                "fix_value": 0.1,
            },
            "alpha": {
                "distribution": "fix",
                "fix_value": 2.0,
            },
            "beta": {
                "distribution": "fix",
                "fix_value": 2.0,
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
            "fix_value": [2.5, -5],
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
        "ranges": [[-2, 2], [-2, 2]],
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return DampedLinearOscillatorSystem(process_hyperparameters), integration_config, locations_params, config


def get_damped_cubic_oscillator(num_steps: int, num_realizations: int, num_paths: int):
    process_hyperparameters = {
        "name": "DampedCubicOscillatorSystem",
        "data_bulk_name": "damped_cubic_oscillator",
        "redo": True,
        "num_realizations": num_realizations,
        "drift_params": {
            "damping": {
                "distribution": "fix",
                "fix_value": 0.1,
            },
            "alpha": {
                "distribution": "fix",
                "fix_value": 2.0,
            },
            "beta": {
                "distribution": "fix",
                "fix_value": 2.0,
            },
        },
        "initial_state": {
            "distribution": "fix",
            "fix_value": [0, -1],
            "activation": None,
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
        "ranges": [[-2, 2], [-2, 2]],
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return DampedCubicOscillatorSystem(process_hyperparameters), integration_config, locations_params, config


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


def get_selkov_glycolysis(num_steps: int, num_realizations: int, num_paths: int):
    process_hyperparameters = {
        "name": "SelkovGlycosis",
        "data_bulk_name": "selkov_glycolysis",
        "redo": True,
        "num_realizations": num_realizations,
        "drift_params": {
            "alpha": {
                "distribution": "fix",
                "fix_value": 0.08,
            },
            "beta": {
                "distribution": "fix",
                "fix_value": 0.08,
            },
            "gamma": {
                "distribution": "fix",
                "fix_value": 0.6,
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
            "fix_value": [0.7, 1.25],
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
        "ranges": [[-2, 4], [-2, 4]],
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return SelkovGlycosis(process_hyperparameters), integration_config, locations_params, config


def get_hopf_bifurcation(num_steps: int, num_realizations: int, num_paths: int):
    process_hyperparameters = {
        "name": "HopfBifurcation",
        "data_bulk_name": "hopf_bifurcation",
        "redo": True,
        "num_realizations": num_realizations,
        "drift_params": {
            "sigma": {
                "distribution": "fix",
                "fix_value": 0.5,
            },
            "beta": {
                "distribution": "fix",
                "fix_value": 0.5,
            },
            "rho": {
                "distribution": "fix",
                "fix_value": 1.0,
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
            "fix_value": [2, 2],
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
        "ranges": [[-2, 2], [-2, 2]],
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return HopfBifurcation(process_hyperparameters), integration_config, locations_params, config


def sample_initial_states_from_fimsdedatabatch(databatch: FIMSDEDatabatch, num_initial_states: int):
    """
    Returns as set of initial states per realization and path, sampled from the observed on that path.
    """
    obs_values = databatch.obs_values  # [num_realizations, num_paths, num_steps, D]
    perm = torch.randperm(obs_values.shape[-2])
    initial_states = obs_values[:, :, perm, :][:, :, :num_initial_states, :]  # [num_realizations, num_paths, num_initial_states, D]
    return initial_states


if __name__ == "__main__":
    save_dir = (
        Path(data_path)
        / "processed"
        / "test"
        / "20250129_opper_svise_wang_long_dense_for_density_ablation_5_realizations_KSIG_reference_paths"
    )

    # train
    # generate long paths with dt=0.002 that can be subsampled during ablation studies
    # num_steps = 5000 * 100  # want max. 100 strides
    # num_paths = 1
    # num_realizations = 5
    # num_initial_states_from_realization = 1000

    # test
    num_steps = 50000  # 100 paths * 500 obs for ksig
    num_paths = 1
    num_realizations = 5
    num_initial_states_from_realization = 1000

    # opper two-dimensional synthetic model
    print("Opper: 2D")
    two_d_opper, integration_config, locations_params, config = get_opper_two_dim_snythetic_model(num_steps, num_realizations, num_paths)
    two_d_opper_data: FIMSDEDatabatch = get_data_from_dynamical_system(two_d_opper, integration_config, locations_params)

    two_d_opper_save_dir = save_dir / "Syn_Drift"
    save_fimsdedatabatch_to_files(two_d_opper_data, two_d_opper_save_dir)
    with open(two_d_opper_save_dir / "config.json", "w") as f:
        json.dump(config, f)

    initial_states = sample_initial_states_from_fimsdedatabatch(two_d_opper_data, num_initial_states_from_realization)
    save_h5(initial_states, two_d_opper_save_dir / "initial_states.h5")

    # wang two-dimensional synthetic model
    print("Wang: 2D")
    two_d_wang, integration_config, locations_params, config = get_wang_two_dim_snythetic_model(num_steps, num_realizations, num_paths)
    two_d_wang_data: FIMSDEDatabatch = get_data_from_dynamical_system(two_d_wang, integration_config, locations_params)

    two_d_wang_save_dir = save_dir / "Wang"
    save_fimsdedatabatch_to_files(two_d_wang_data, two_d_wang_save_dir)
    with open(two_d_wang_save_dir / "config.json", "w") as f:
        json.dump(config, f)

    initial_states = sample_initial_states_from_fimsdedatabatch(two_d_wang_data, num_initial_states_from_realization)
    save_h5(initial_states, two_d_wang_save_dir / "initial_states.h5")

    # # wang double well
    # print("Wang: Double Well")
    # double_well_wang, integration_config, locations_params, config = get_wang_double_well(num_steps, num_realizations, num_paths)
    # double_well_wang_data: FIMSDEDatabatch = get_data_from_dynamical_system(double_well_wang, integration_config, locations_params)
    #
    # double_well_wang_save_dir = save_dir / "wang_double_well"
    # save_fimsdedatabatch_to_files(double_well_wang_data, double_well_wang_save_dir)
    # with open(double_well_wang_save_dir / "config.json", "w") as f:
    #     json.dump(config, f)

    # # opper 1D double well constant diffusion
    # print("Opper: Double Well, const diff")
    # double_well_const_diff, integration_config, locations_params, config = get_double_well_const_diffusion(num_steps, num_realizations, num_paths)
    # double_well_data: FIMSDEDatabatch = get_data_from_dynamical_system(double_well_const_diff, integration_config, locations_params)
    #
    # double_well_const_diff_save_dir = save_dir / "opper_double_well_constant_diff"
    # save_fimsdedatabatch_to_files(double_well_data, double_well_const_diff_save_dir)
    # with open(double_well_const_diff_save_dir / "config.json", "w") as f:
    #     json.dump(config, f)

    # opper 1D double well state dependent diffusion
    print("Opper: Double Well, state dep diff")
    double_well_state_dep_diff, integration_config, locations_params, config = get_double_well_state_dep_diffusion(
        num_steps, num_realizations, num_paths
    )
    double_well_data: FIMSDEDatabatch = get_data_from_dynamical_system(double_well_state_dep_diff, integration_config, locations_params)

    double_well_state_dep_diff_save_dir = save_dir / "Double_Well"
    save_fimsdedatabatch_to_files(double_well_data, double_well_state_dep_diff_save_dir)
    with open(double_well_state_dep_diff_save_dir / "config.json", "w") as f:
        json.dump(config, f)

    initial_states = sample_initial_states_from_fimsdedatabatch(double_well_data, num_initial_states_from_realization)
    save_h5(initial_states, double_well_state_dep_diff_save_dir / "initial_states.h5")

    funcs = {
        "Damped_Linear": get_damped_linear_oscillator,
        "Damped_Cubic": get_damped_cubic_oscillator,
        "Duffing": get_duffing_oscillator,
        "Glycosis": get_selkov_glycolysis,
        "Hopf": get_hopf_bifurcation,
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
