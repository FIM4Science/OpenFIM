import torch

from fim.data.data_generation.dynamical_systems import (
    DampedCubicOscillatorSystem,
    DampedLinearOscillatorSystem,
    DoubleWellOneDimension,
    DuffingOscillator,
    HopfBifurcation,
    Lorenz63System,
    SelkovGlycosis,
)
from fim.data.data_generation.dynamical_systems_sample import PathGenerator
from fim.data.datasets import FIMSDEDatabatch, FIMSDEDatabatchTuple, FIMSDEDataset
from fim.models.config_dataclasses import FIMSDEConfig


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


def generate_lorenz(params: FIMSDEConfig):
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
        "num_steps": params.max_time_steps,
        "num_paths": params.max_num_paths,
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


def generate_duffing(params: FIMSDEConfig):
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
        "num_steps": params.max_time_steps,
        "num_paths": params.max_num_paths,
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


def generate_hopf(params: FIMSDEConfig):
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
        "num_steps": params.max_time_steps,
        "num_paths": params.max_num_paths,
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


def generate_selkov(params: FIMSDEConfig):
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
        "num_steps": params.max_time_steps,
        "num_paths": params.max_num_paths,
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


def generate_damped_cubic(params: FIMSDEConfig):
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
        "num_steps": params.max_time_steps,
        "num_paths": params.max_num_paths,
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


def generate_damped_linear(params: FIMSDEConfig):
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
        "num_steps": params.max_time_steps,
        "num_paths": params.max_num_paths,
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


def generate_double_well(params):
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
        "num_steps": params.max_time_steps,
        "num_paths": params.max_num_paths,
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
        # diffusion_parameters=diffusion_parameters,
        # drift_parameters=drift_parameters,
        # process_label=process_label,
        # process_dimension=process_dimension,
        dimension_mask=mask,
    )


def generate_all(params: FIMSDEConfig) -> FIMSDEDatabatchTuple:
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
    lorenz_data = generate_lorenz(params)
    duffing_data = generate_duffing(params)
    hopf_data = generate_hopf(params)
    selkov_data = generate_selkov(params)
    damped_cubic_data = generate_damped_cubic(params)
    damped_linear_data = generate_damped_linear(params)
    double_well_data = generate_double_well(params)

    all_data = [lorenz_data, hopf_data, damped_cubic_data, selkov_data, duffing_data, damped_linear_data, double_well_data]

    # creates dataset since this object has all the padding functionality
    dataset = FIMSDEDataset(None, all_data)
    # as only one sample was create for each data set we use sample_idx = 0
    all_data_tuples = [pad_from_dataset(data=data, sample_idx=0, dataset=dataset) for data in all_data]
    # concat all the tuples
    data_tuple = concat_name_tuple(all_data_tuples, FIMSDEDatabatchTuple)
    return data_tuple
