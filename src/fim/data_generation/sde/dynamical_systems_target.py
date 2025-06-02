import torch

from fim.data.datasets import FIMSDEDatabatch, FIMSDEDatabatchTuple, FIMSDEDataset
from fim.data_generation.sde.dynamical_systems import (
    DampedCubicOscillatorSystem,
    DampedLinearOscillatorSystem,
    DoubleWellOneDimension,
    DuffingOscillator,
    HopfBifurcation,
    Lorenz63System,
    SelkovGlycosis,
)
from fim.data_generation.sde.dynamical_systems_sample import PathGenerator


def concat_name_tuple(tuples_list, MyTuple):
    # Initialize dictionaries to hold lists of tensors for each field
    concat_tensors = {field: [] for field in MyTuple._fields}

    # Populate the dictionaries with tensors from each named tuple
    for t in tuples_list:
        for field in t._fields:
            tensor_value = getattr(t, field)
            tensor_value = tensor_value.unsqueeze(0)
            concat_tensors[field].append(tensor_value)

    # Concatenate tensors for each field along the first dimension
    concatenated_tensors = {field: torch.cat(concat_tensors[field], dim=0) for field in concat_tensors}

    # Create a new named tuple with the concatenated tensors
    new_tuple = MyTuple(**concatenated_tensors)
    return new_tuple


def generate_lorenz(max_time_steps: int, max_num_paths: int):
    process_hyperparameters = {
        "name": "Lorenz63System",
        "data_bulk_name": "lorenz_theory",
        "redo": True,
        "num_realizations": 1,
        "observed_dimension": None,
        "drift_params": {
            "sigma": {
                "distribution": "fix",
                "fix_value": 10.0,
            },
            "beta": {
                "distribution": "fix",
                "fix_value": 2.66666666,
            },
            "rho": {
                "distribution": "fix",
                "fix_value": 28.0,
            },
        },
        "diffusion_params": {"constant_value": 1.0, "dimensions": 3},
        "initial_state": {"distribution": "fix", "fix_value": [-8.0, 7.0, 27.0], "activation": None},
    }
    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.01,
        "num_steps": max_time_steps,
        "num_paths": max_num_paths,
        "num_locations": 1024,
        "stochastic": True,
    }
    locations_params = {"type": "unit_cube"}

    dynamical_model = Lorenz63System(process_hyperparameters)
    path_generator = PathGenerator(
        dataset_type="FIMSDEpDataset", system=dynamical_model, integrator_params=integration_config, locations_params=locations_params
    )
    data = path_generator.generate_paths()

    return data


def generate_duffing(max_time_steps: int, max_num_paths: int):
    process_hyperparameters = {
        "name": "DuffingOscillator",
        "data_bulk_name": "duffing_theory",
        "redo": True,
        "num_realizations": 1,
        "observed_dimension": None,
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
        "diffusion_params": {"g1": {"distribution": "fix", "fix_value": 1.0}, "g2": {"distribution": "fix", "fix_value": 1.0}},
        "initial_state": {"distribution": "fix", "fix_value": [3.0, 2.0], "activation": None},
    }
    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.01,
        "num_steps": max_time_steps,
        "num_paths": max_num_paths,
        "num_locations": 1024,
        "stochastic": True,
    }
    locations_params = {"type": "unit_cube"}

    dynamical_model = DuffingOscillator(process_hyperparameters)
    path_generator = PathGenerator(
        dataset_type="FIMSDEpDataset", system=dynamical_model, integrator_params=integration_config, locations_params=locations_params
    )
    data = path_generator.generate_paths()

    return data


def generate_hopf(max_time_steps: int, max_num_paths: int):
    process_hyperparameters = {
        "name": "HopfBifurcation",
        "data_bulk_name": "hopf_theory",
        "redo": True,
        "num_realizations": 1,
        "observed_dimension": None,
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
        "diffusion_params": {"g1": {"distribution": "fix", "fix_value": 1.0}, "g2": {"distribution": "fix", "fix_value": 1.0}},
        "initial_state": {"distribution": "fix", "fix_value": [2.0, 2.0], "activation": None},
    }
    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.01,
        "num_steps": max_time_steps,
        "num_paths": max_num_paths,
        "num_locations": 1024,
        "stochastic": True,
    }
    locations_params = {"type": "unit_cube"}

    dynamical_model = HopfBifurcation(process_hyperparameters)
    path_generator = PathGenerator(
        dataset_type="FIMSDEpDataset", system=dynamical_model, integrator_params=integration_config, locations_params=locations_params
    )
    data = path_generator.generate_paths()

    return data


def generate_selkov(max_time_steps: int, max_num_paths: int):
    process_hyperparameters = {
        "name": "SelkovGlycosis",
        "data_bulk_name": "selkov_theory",
        "redo": True,
        "num_realizations": 1,
        "observed_dimension": None,
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
        "diffusion_params": {"g1": {"distribution": "fix", "fix_value": 1.0}, "g2": {"distribution": "fix", "fix_value": 1.0}},
        "initial_state": {"distribution": "fix", "fix_value": [0.7, 1.25], "activation": None},
    }
    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.01,
        "num_steps": max_time_steps,
        "num_paths": max_num_paths,
        "num_locations": 1024,
        "stochastic": True,
    }
    locations_params = {"type": "unit_cube"}

    dynamical_model = SelkovGlycosis(process_hyperparameters)
    path_generator = PathGenerator(
        dataset_type="FIMSDEpDataset", system=dynamical_model, integrator_params=integration_config, locations_params=locations_params
    )
    data = path_generator.generate_paths()

    return data


def generate_damped_cubic(max_time_steps: int, max_num_paths: int):
    process_hyperparameters = {
        "name": "DampedCubicOscillatorSystem",
        "data_bulk_name": "damped_cubic_theory",
        "redo": True,
        "num_realizations": 1,
        "observed_dimension": None,
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
        "diffusion_params": {"g1": {"distribution": "fix", "fix_value": 1.0}, "g2": {"distribution": "fix", "fix_value": 1.0}},
        "initial_state": {"distribution": "fix", "fix_value": [0.0, 1.0], "activation": None},
    }
    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.01,
        "num_steps": max_time_steps,
        "num_paths": max_num_paths,
        "num_locations": 1024,
        "stochastic": True,
    }
    locations_params = {"type": "unit_cube"}

    dynamical_model = DampedCubicOscillatorSystem(process_hyperparameters)
    path_generator = PathGenerator(
        dataset_type="FIMSDEpDataset", system=dynamical_model, integrator_params=integration_config, locations_params=locations_params
    )
    data = path_generator.generate_paths()

    return data


def generate_damped_linear(max_time_steps: int, max_num_paths: int):
    process_hyperparameters = {
        "name": "DampedLinearOscillatorSystem",
        "data_bulk_name": "damped_linear_theory",
        "redo": True,
        "num_realizations": 1,
        "observed_dimension": None,
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
        "diffusion_params": {"g1": {"distribution": "fix", "fix_value": 1.0}, "g2": {"distribution": "fix", "fix_value": 1.0}},
        "initial_state": {"distribution": "fix", "fix_value": [2.5, -5.0], "activation": None},
    }
    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.01,
        "num_steps": max_time_steps,
        "num_paths": max_num_paths,
        "num_locations": 1024,
        "stochastic": True,
    }
    locations_params = {"type": "unit_cube"}

    dynamical_model = DampedLinearOscillatorSystem(process_hyperparameters)
    path_generator = PathGenerator(
        dataset_type="FIMSDEpDataset", system=dynamical_model, integrator_params=integration_config, locations_params=locations_params
    )
    data = path_generator.generate_paths()

    return data


def generate_double_well(max_time_steps: int, max_num_paths: int):
    process_hyperparameters = {
        "name": "DampedLinearOscillatorSystem",
        "data_bulk_name": "damped_linear_theory",
        "redo": True,
        "num_realizations": 1,
        "observed_dimension": None,
        "drift_params": {
            "alpha": {
                "distribution": "fix",
                "fix_value": 0.1,
            },
            "beta": {
                "distribution": "fix",
                "fix_value": 2.0,
            },
        },
        "diffusion_params": {"g1": {"distribution": "fix", "fix_value": 4.0}, "g2": {"distribution": "fix", "fix_value": 1.25}},
        "initial_state": {
            "distribution": "normal",
            "mean": 0.0,
            "std_dev": 1.0,
            "activation": None,
        },
    }
    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.01,
        "num_steps": max_time_steps,
        "num_paths": max_num_paths,
        "num_locations": 1024,
        "stochastic": True,
    }
    locations_params = {"type": "unit_cube"}

    dynamical_model = DoubleWellOneDimension(process_hyperparameters)
    path_generator = PathGenerator(
        dataset_type="FIMSDEpDataset", system=dynamical_model, integrator_params=integration_config, locations_params=locations_params
    )
    data = path_generator.generate_paths()
    return data


def pad_from_dataset(data: FIMSDEDatabatch, sample_idx: int, dataset: FIMSDEDataset) -> FIMSDEDatabatchTuple:
    """
    performs the padding of one element of a FIMSDEpDataBulk using a dataset
    """
    # Get the tensor from the appropriate file
    obs_values = data.obs_values[sample_idx]
    obs_times = data.obs_times[sample_idx]
    diffusion_at_locations = data.diffusion_at_locations[sample_idx]
    drift_at_locations = data.drift_at_locations[sample_idx]
    locations = data.locations[sample_idx]

    # diffusion_parameters = data.diffusion_parameters[sample_idx]
    # drift_parameters = data.drift_parameters[sample_idx]

    # Pad and Obtain Mask of The tensors if necessary
    obs_values, obs_times = dataset._pad_obs_tensors(obs_values, obs_times)
    drift_at_locations, diffusion_at_locations, locations, mask = dataset._pad_locations_tensors(
        drift_at_locations, diffusion_at_locations, locations
    )

    # drift_parameters = dataset._pad_drift_params(drift_parameters)
    # diffusion_parameters = dataset._pad_diffusion_params(diffusion_parameters)

    return FIMSDEDatabatchTuple(
        obs_values=obs_values,
        obs_times=obs_times,
        diffusion_at_locations=diffusion_at_locations,
        drift_at_locations=drift_at_locations,
        locations=locations,
        dimension_mask=mask,
    )


def generate_all(max_time_steps: int, max_num_paths: int) -> FIMSDEDatabatchTuple:
    """
    creates a databatch with all the target data

    REVERSED_DYNAMICS_LABELS = {
        0: "LorenzSystem63",
        1: "HopfBifurcation",
        2: "DampedCubicOscillatorSystem",
        3: "SelkovGlycosis",
        4: "DuffingOscillator",
        5: "DampedLinearOscillatorSystem",
        6: "DoubleWellOneDimension"
    }
    """
    # generate all data
    lorenz_data = generate_lorenz(max_time_steps, max_num_paths)
    duffing_data = generate_duffing(max_time_steps, max_num_paths)
    hopf_data = generate_hopf(max_time_steps, max_num_paths)
    selkov_data = generate_selkov(max_time_steps, max_num_paths)
    damped_cubic_data = generate_damped_cubic(max_time_steps, max_num_paths)
    damped_linear_data = generate_damped_linear(max_time_steps, max_num_paths)
    double_well_data = generate_double_well(max_time_steps, max_num_paths)

    all_data = [lorenz_data, hopf_data, damped_cubic_data, selkov_data, duffing_data, damped_linear_data, double_well_data]

    # creates dataset since this object has all the padding functionality
    dataset = FIMSDEDataset(None, all_data)
    # as only one sample was create for each data set we use sample_idx = 0
    all_data_tuples = [pad_from_dataset(data=data, sample_idx=0, dataset=dataset) for data in all_data]
    # concat all the tuples
    data_tuple = concat_name_tuple(all_data_tuples, FIMSDEDatabatchTuple)
    return data_tuple
