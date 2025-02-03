import json
from copy import deepcopy
from pathlib import Path

from fim import data_path
from fim.data.data_generation.bisde_estimated_equations import (
    BISDEEst2DSynthetic,
    BISDEEstDoubleWell,
    BISDEEstFacebook,
    BISDEEstOil,
    BISDEEstTesla,
    BISDEEstWind,
    SINDyEstDoubleWell,
)
from fim.data.datasets import FIMSDEDatabatch
from fim.data_generation.sde.dynamical_systems_to_files import (
    get_data_from_dynamical_system,
    save_fimsdedatabatch_to_files,
)


# general config
STEPS_PER_DT = 10


def get_bisde_est_2D_synth():
    process_hyperparameters = {
        "name": "BISDEEst2DSynthetic",
        "data_bulk_name": "bisdeest2dsynth",
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

    return BISDEEst2DSynthetic(process_hyperparameters), integration_config, locations_params, config


def get_bisde_est_double_well():
    process_hyperparameters = {
        "name": "BISDEEstDoubleWell",
        "data_bulk_name": "bisdeestdouble_well",
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

    return BISDEEstDoubleWell(process_hyperparameters), integration_config, locations_params, config


def get_sindy_est_double_well():
    process_hyperparameters = {
        "name": "sindyEstDoubleWell",
        "data_bulk_name": "sindyestdouble_well",
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

    return SINDyEstDoubleWell(process_hyperparameters), integration_config, locations_params, config


def get_bisde_est_facebook():
    process_hyperparameters = {
        "name": "BISDEEstFacebook",
        "data_bulk_name": "bisdeestfacebook",
        "redo": True,
        "num_realizations": 1,
        "drift_params": {},
        "diffusion_params": {},
        "initial_state": {},
    }

    integration_config = {
        "method": "EulerMaruyama",
        # "time_length": 0.25,  # measured in years
        "time_step": 1 / (252 * 390),  # Minutes in trading years, per BISDE code
        "num_steps": 24959,  # total 24960  observations, initial state is first obs
        "steps_per_dt": STEPS_PER_DT,
        "num_paths": 1,
        "num_locations": 1024,
        "stochastic": True,
    }

    locations_params = {
        "type": "regular_grid",
        "ranges": [[227.37, 304.27]],  # regular grid in data range
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return BISDEEstFacebook(process_hyperparameters), integration_config, locations_params, config


def get_bisde_est_tesla():
    process_hyperparameters = {
        "name": "BISDEEstTesla",
        "data_bulk_name": "bisdeesttesla",
        "redo": True,
        "num_realizations": 1,
        "drift_params": {},
        "diffusion_params": {},
        "initial_state": {},
    }

    integration_config = {
        "method": "EulerMaruyama",
        # "time_length": 0.25,  # measured in years
        "time_step": 1 / (252 * 390),  # Minutes in trading years, per BISDE code
        "num_steps": 24959,  # total 24960  observations, initial state is first obs
        "steps_per_dt": STEPS_PER_DT,
        "num_paths": 1,
        "num_locations": 1024,
        "stochastic": True,
    }

    locations_params = {
        "type": "regular_grid",
        "ranges": [[218.17, 501.9742]],  # regular grid in data range
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return BISDEEstTesla(process_hyperparameters), integration_config, locations_params, config


def get_bisde_est_oil():
    process_hyperparameters = {
        "name": "BISDEEstoil",
        "data_bulk_name": "bisdeestoil",
        "redo": True,
        "num_realizations": 1,
        "drift_params": {},
        "diffusion_params": {},
        "initial_state": {},
    }

    integration_config = {
        "method": "EulerMaruyama",
        # "time_length": 0.416667,  # measured in years, 5 / 12
        "time_step": 1,  # Price change per day, per BISDE code
        "num_steps": 7920,  # total 7921 fluctuation observations, initial state is first obs
        "steps_per_dt": STEPS_PER_DT,
        "num_paths": 1,
        "num_locations": 1024,
        "stochastic": True,
    }

    locations_params = {
        "type": "regular_grid",
        "ranges": [[-14.76, 18.56]],  # regular grid in data range
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return BISDEEstOil(process_hyperparameters), integration_config, locations_params, config


def get_bisde_est_wind():
    process_hyperparameters = {
        "name": "BISDEEstWind",
        "data_bulk_name": "bisdeestWind",
        "redo": True,
        "num_realizations": 1,
        "drift_params": {},
        "diffusion_params": {},
        "initial_state": {},
    }

    integration_config = {
        "method": "EulerMaruyama",
        # "time_length": 0.5,  # measured in years
        "time_step": 1 / 6,  # wind chang per hour, measured every 10 minutes, per BISDE code
        "num_steps": 26201,  # total 26202 fluctuation observations, initial state is first obs
        "steps_per_dt": STEPS_PER_DT,
        "num_paths": 1,
        "num_locations": 1024,
        "stochastic": True,
    }

    locations_params = {
        "type": "regular_grid",
        "ranges": [[-5.7, 7.7]],  # regular grid in data range
    }

    config = deepcopy(
        {
            "process_hyperparameters": process_hyperparameters,
            "integration_config": integration_config,
            "locations_params": locations_params,
        }
    )

    return BISDEEstWind(process_hyperparameters), integration_config, locations_params, config


if __name__ == "__main__":
    save_dir = Path(data_path) / "processed" / "test" / "20250126_wang_estimated_equations"
    target_path_length = 128

    # # bisde two-dimensional synthetic model
    # print("BISDE 2D synthetic")
    # bisde_est_2D_synth, integration_config, locations_params, config = get_bisde_est_2D_synth()
    # bisde_est_2D_synth_data: FIMSDEDatabatch = get_data_from_dynamical_system(bisde_est_2D_synth, integration_config, locations_params)
    # split_bisde_est_2D_synth_data: FIMSDEDatabatch = split_databatch_into_paths(bisde_est_2D_synth_data, target_path_length)
    #
    # bisde_est_2D_synth_save_dir = save_dir / "bisde_est_2D_synth_80000_points_split_128_length"
    # save_fimsdedatabatch_to_files(split_bisde_est_2D_synth_data, bisde_est_2D_synth_save_dir)
    # with open(bisde_est_2D_synth_save_dir / "config.json", "w") as f:
    #     json.dump(config, f)
    #
    # # bisde_double_well
    # print("BISDE Double Well")
    # bisde_est_double_well, integration_config, locations_params, config = get_bisde_est_double_well()
    # bisde_est_double_well_data: FIMSDEDatabatch = get_data_from_dynamical_system(
    #     bisde_est_double_well, integration_config, locations_params
    # )
    # split_bisde_est_double_well_data: FIMSDEDatabatch = split_databatch_into_paths(bisde_est_double_well_data, target_path_length)
    #
    # bisde_est_double_well_save_dir = save_dir / "bisde_est_double_well_25000_points_split_128_length"
    # save_fimsdedatabatch_to_files(split_bisde_est_double_well_data, bisde_est_double_well_save_dir)
    # with open(bisde_est_double_well_save_dir / "config.json", "w") as f:
    #     json.dump(config, f)
    #
    # # sindy_double_well
    # print("SINDy Double Well")
    # sindy_est_double_well, integration_config, locations_params, config = get_sindy_est_double_well()
    # sindy_est_double_well_data: FIMSDEDatabatch = get_data_from_dynamical_system(
    #     sindy_est_double_well, integration_config, locations_params
    # )
    # split_sindy_est_double_well_data: FIMSDEDatabatch = split_databatch_into_paths(sindy_est_double_well_data, target_path_length)
    #
    # sindy_est_double_well_save_dir = save_dir / "sindy_est_double_well_25000_points_split_128_length"
    # save_fimsdedatabatch_to_files(split_sindy_est_double_well_data, sindy_est_double_well_save_dir)
    # with open(sindy_est_double_well_save_dir / "config.json", "w") as f:
    #     json.dump(config, f)

    # bisde_facebook
    print("BISDE Facebook")
    bisde_est_facebook, integration_config, locations_params, config = get_bisde_est_facebook()
    bisde_est_facebook_data: FIMSDEDatabatch = get_data_from_dynamical_system(bisde_est_facebook, integration_config, locations_params)

    bisde_est_facebook_save_dir = save_dir / "bisde_est_facebook"
    save_fimsdedatabatch_to_files(bisde_est_facebook_data, bisde_est_facebook_save_dir)
    with open(bisde_est_facebook_save_dir / "config.json", "w") as f:
        json.dump(config, f)

    # bisde_tesla
    print("BISDE tesla")
    bisde_est_tesla, integration_config, locations_params, config = get_bisde_est_tesla()
    bisde_est_tesla_data: FIMSDEDatabatch = get_data_from_dynamical_system(bisde_est_tesla, integration_config, locations_params)

    bisde_est_tesla_save_dir = save_dir / "bisde_est_tesla"
    save_fimsdedatabatch_to_files(bisde_est_tesla_data, bisde_est_tesla_save_dir)
    with open(bisde_est_tesla_save_dir / "config.json", "w") as f:
        json.dump(config, f)

    # bisde_oil
    print("BISDE oil")
    bisde_est_oil, integration_config, locations_params, config = get_bisde_est_oil()
    bisde_est_oil_data: FIMSDEDatabatch = get_data_from_dynamical_system(bisde_est_oil, integration_config, locations_params)

    bisde_est_oil_save_dir = save_dir / "bisde_est_oil"
    save_fimsdedatabatch_to_files(bisde_est_oil_data, bisde_est_oil_save_dir)
    with open(bisde_est_oil_save_dir / "config.json", "w") as f:
        json.dump(config, f)

    # bisde_wind
    print("BISDE wind")
    bisde_est_wind, integration_config, locations_params, config = get_bisde_est_wind()
    bisde_est_wind_data: FIMSDEDatabatch = get_data_from_dynamical_system(bisde_est_wind, integration_config, locations_params)

    bisde_est_wind_save_dir = save_dir / "bisde_est_wind"
    save_fimsdedatabatch_to_files(bisde_est_wind_data, bisde_est_wind_save_dir)
    with open(bisde_est_wind_save_dir / "config.json", "w") as f:
        json.dump(config, f)
