import itertools
import json
from functools import partial
from pathlib import Path

from fim import data_path
from fim.data.data_generation.dynamical_systems import Degree2Polynomial
from fim.data_generation.sde.dynamical_systems_to_files import get_data_from_dynamical_system, save_fimsdedatabatch_to_files


NUM_PATHS = 100
NUM_STEPS = 128
STEPS_PER_DT = 10
NUM_REALIZATIONS = 100
CHUNK_SIZE = 10000
REJECT_THRESHOLD = 100

STD_NORMAL_PARAM = {
    "distribution": "normal",
    "mean": 0.0,
    "std": 1.0,
}

ZERO_PARAM = {
    "distribution": "fix",
    "fix_value": 0.0,
}

STD_NORMAL_INIT = {
    "distribution": "normal",
    "mean": 0.0,
    "std_dev": 1.0,
    "activation": None,
}

UNIFORM_INIT = {
    "distribution": "uniform",
    "min": -10,
    "max": 10,
    "activation": None,
}

SCALE = {"sample_per_dimension": False, "distribution": "fix", "fix_value": 1.0}


INTEGRATION_CONFIG = {
    "method": "EulerMaruyama",
    "time_length": 10,
    "steps_per_dt": STEPS_PER_DT,
    "num_steps": NUM_STEPS,
    "num_paths": NUM_PATHS,
    "chunk_size": CHUNK_SIZE,
    "reject_threshold": REJECT_THRESHOLD,
    "num_locations": 1024,
    "stochastic": True,
}

LOCATION_PARAMS = {
    "type": "regular_cube",
    "extension_perc": 0.2,
}


def up_to_deg_polys(drift: int, diffusion: int, std_normal_init: bool, state_dim: int) -> tuple[Degree2Polynomial, dict]:
    """
    Helper function to set up a variety of up to degree 2 systems.
    """
    # create appropriate label (data_bulk_name)
    init = "std_normal_init" if std_normal_init is True else "unif_init"
    label = f"dim_{state_dim}_deg_drift_{drift}_deg_diff_{diffusion}_{init}"

    process_hyperparameters = {
        "name": "Degree2Polynomial",
        "data_bulk_name": label,
        "redo": True,
        "num_realizations": NUM_REALIZATIONS,
        "state_dim": state_dim,
        "drift_params": {
            "constant": STD_NORMAL_PARAM,
            "degree_1": STD_NORMAL_PARAM if drift >= 1 else ZERO_PARAM,
            "degree_2_squared": STD_NORMAL_PARAM if drift >= 2 else ZERO_PARAM,
            "degree_2_mixed": STD_NORMAL_PARAM if drift >= 2 else ZERO_PARAM,
            "scale": SCALE,
        },
        "diffusion_params": {
            "constant": STD_NORMAL_PARAM,
            "degree_1": STD_NORMAL_PARAM if diffusion >= 1 else ZERO_PARAM,
            "degree_2_squared": STD_NORMAL_PARAM if diffusion >= 2 else ZERO_PARAM,
            "degree_2_mixed": STD_NORMAL_PARAM if diffusion >= 2 else ZERO_PARAM,
            "scale": SCALE,
        },
        "initial_state": STD_NORMAL_INIT if std_normal_init is True else UNIFORM_INIT,
    }

    return Degree2Polynomial(process_hyperparameters), process_hyperparameters


if __name__ == "__main__":
    # gather all systems
    polynomial_systems = []

    drift_deg = [1, 2]
    diff_deg = [0, 1, 2]
    std_normal_init = [True, False]
    state_dim = [1, 2, 3]

    for drift_, diff_, init_, dim_ in itertools.product(drift_deg, diff_deg, std_normal_init, state_dim):
        polynomial_systems.append(partial(up_to_deg_polys, drift=drift_, diffusion=diff_, std_normal_init=init_, state_dim=dim_))

    # generate and save data under label given by data_bulk_name
    for poly_sys in polynomial_systems:
        system, config = poly_sys()
        name = config.get("data_bulk_name")
        print(f"Generating {name}")
        data = get_data_from_dynamical_system(system, INTEGRATION_CONFIG, LOCATION_PARAMS)
        save_dir = Path(data_path) / "processed" / "test" / "20241228_polynomial_test_sets_100_each_100_paths" / name
        save_fimsdedatabatch_to_files(data, save_dir)

        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f)
