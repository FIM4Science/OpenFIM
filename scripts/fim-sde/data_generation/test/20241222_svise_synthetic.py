import json
from copy import deepcopy
from pathlib import Path

from fim import data_path
from fim.data.data_generation.dynamical_systems import (
    DampedCubicOscillatorSystem,
    DampedLinearOscillatorSystem,
    DuffingOscillator,
    HopfBifurcation,
    Lorenz63System,
    SelkovGlycosis,
)
from fim.data_generation.sde.dynamical_systems_to_files import get_data_from_dynamical_system, save_fimsdedatabatch_to_files


# general config
NUM_PATHS = 20
NUM_STEPS = 128
STEPS_PER_DT = 100
NUM_REALIZATIONS = 1
RELATIVE_DIFFUSION_SCALE = 0.01


def get_damped_linear_oscillator():
    process_hyperparameters = {
        "name": "DampedLinearOscillatorSystem",
        "data_bulk_name": "damped_linear_oscillator",
        "redo": True,
        "num_realizations": NUM_REALIZATIONS,
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
                "fix_value": 0.0,
            },
            "g2": {
                "distribution": "fix",
                "fix_value": 0.0,
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
        "time_length": 20,
        "steps_per_dt": STEPS_PER_DT,
        "num_steps": NUM_STEPS,
        "num_paths": NUM_PATHS,
        "num_locations": 1024,
        "relative_diffusion_scale": RELATIVE_DIFFUSION_SCALE,
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

    return DampedLinearOscillatorSystem(process_hyperparameters), integration_config, locations_params, config


def get_damped_cubic_oscillator():
    process_hyperparameters = {
        "name": "DampedCubicOscillatorSystem",
        "data_bulk_name": "damped_cubic_oscillator",
        "redo": True,
        "num_realizations": NUM_REALIZATIONS,
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
                "fix_value": 0.0,
            },
            "g2": {
                "distribution": "fix",
                "fix_value": 0.0,
            },
        },
    }

    integration_config = {
        "method": "EulerMaruyama",
        "time_length": 25,
        "steps_per_dt": STEPS_PER_DT,
        "num_steps": NUM_STEPS,
        "num_paths": NUM_PATHS,
        "num_locations": 1024,
        "relative_diffusion_scale": RELATIVE_DIFFUSION_SCALE,
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

    return DampedCubicOscillatorSystem(process_hyperparameters), integration_config, locations_params, config


def get_duffing_oscillator():
    process_hyperparameters = {
        "name": "DuffDuffingOscillator",
        "data_bulk_name": "duffing_oscillator",
        "redo": True,
        "num_realizations": NUM_REALIZATIONS,
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
                "fix_value": 0.0,
            },
            "g2": {
                "distribution": "fix",
                "fix_value": 0.0,
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
        "time_length": 20,
        "steps_per_dt": STEPS_PER_DT,
        "num_steps": NUM_STEPS,
        "num_paths": NUM_PATHS,
        "num_locations": 1024,
        "relative_diffusion_scale": RELATIVE_DIFFUSION_SCALE,
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

    return DuffingOscillator(process_hyperparameters), integration_config, locations_params, config


def get_selkov_glycolysis():
    process_hyperparameters = {
        "name": "SelkovGlycosis",
        "data_bulk_name": "selkov_glycolysis",
        "redo": True,
        "num_realizations": NUM_REALIZATIONS,
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
                "fix_value": 0.0,
            },
            "g2": {
                "distribution": "fix",
                "fix_value": 0.0,
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
        "time_length": 30,
        "steps_per_dt": STEPS_PER_DT,
        "num_steps": NUM_STEPS,
        "num_paths": NUM_PATHS,
        "num_locations": 1024,
        "relative_diffusion_scale": RELATIVE_DIFFUSION_SCALE,
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

    return SelkovGlycosis(process_hyperparameters), integration_config, locations_params, config


def get_lorenz_63():
    process_hyperparameters = {
        "name": "Lorenz63System",
        "data_bulk_name": "lorenzt_63",
        "redo": True,
        "num_realizations": NUM_REALIZATIONS,
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
            "constant_value": 0.5,
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
        "time_length": 10,
        "steps_per_dt": STEPS_PER_DT,
        "num_steps": NUM_STEPS,
        "num_paths": NUM_PATHS,
        "num_locations": 1024,
        "relative_diffusion_scale": RELATIVE_DIFFUSION_SCALE,
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


def get_hopf_bifurcation():
    process_hyperparameters = {
        "name": "HopfBifurcation",
        "data_bulk_name": "hopf_bifurcation",
        "redo": True,
        "num_realizations": NUM_REALIZATIONS,
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
                "fix_value": 0.0,
            },
            "g2": {
                "distribution": "fix",
                "fix_value": 0.0,
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
        "time_length": 20,
        "steps_per_dt": STEPS_PER_DT,
        "num_steps": NUM_STEPS,
        "num_paths": NUM_PATHS,
        "num_locations": 1024,
        "relative_diffusion_scale": RELATIVE_DIFFUSION_SCALE,
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

    return HopfBifurcation(process_hyperparameters), integration_config, locations_params, config


if __name__ == "__main__":
    ### Save 20 paths from each equation to disk
    funcs = {
        "damped_linear_oscillator": get_damped_linear_oscillator,
        "damped_cubic_oscillator": get_damped_cubic_oscillator,
        "duffing_oscillator": get_duffing_oscillator,
        "selkov_glycolysis": get_selkov_glycolysis,
        "lorenz_63": get_lorenz_63,
        "hopf_bifurcation": get_hopf_bifurcation,
    }

    for name, func in funcs.items():
        system, integration_config, locations_params, config = func()
        data = get_data_from_dynamical_system(system, integration_config, locations_params)
        save_dir = Path(data_path) / "processed" / "test" / "20241222_svise_1_perc_diffusion_no_additive_noise" / name
        save_fimsdedatabatch_to_files(data, save_dir)

        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f)
