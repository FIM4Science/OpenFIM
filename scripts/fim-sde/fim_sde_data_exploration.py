import matplotlib.pyplot as plt

from fim.data.data_generation.dynamical_systems import Degree2Polynomial, DynamicalSystem, HybridDynamicalSystem, SelkovGlycosis
from fim.data.data_generation.dynamical_systems_sample import PathGenerator
from fim.data.datasets import FIMSDEDatabatch
from fim.utils.plots.sde_data_exploration_plots import (
    path_statistics,
    plot_1D_vector_field_in_axis,
    plot_2D_vector_field_in_axis,
    plot_3D_vector_field_in_axis,
    plot_paths_in_axis,
    show_paths_vector_fields_and_statistics,
)


def get_example_system(state_dim: int) -> tuple[DynamicalSystem, dict]:
    """
    Return example dynamical system and an integration config for data generation.
    """

    process_hyperparameters = {
        "name": "Degree2Polynomial",
        "data_bulk_name": "damped_linear_theory",
        "redo": True,
        "num_realizations": 5,
        "state_dim": state_dim,
        "show_equation": True,
        "observed_dimension": None,
        "drift_params": {},
        "diffusion_params": {
            "constant": {
                "distribution": "normal",
                "std": 1.0,
                "bernoulli_survival_rate": 0.5,
            },
            "degree_1": {
                "distribution": "normal",
                "std": 1.0,
                "bernoulli_survival_rate": 0.5,
            },
            "degree_2_squared": {
                "distribution": "normal",
                "std": 1.0,
                "bernoulli_survival_rate": 0.5,
            },
            "degree_2_mixed": {
                "distribution": "normal",
                "std": 1.0,
                "bernoulli_survival_rate": 0.5,
            },
            "scale": {
                "sample_per_dimension": True,
                "distribution": "uniform",
                "min": 0,
                "max": 1,
            },
        },
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
        "num_steps": 1000,
        "num_paths": 10,
        "num_locations": 512,
        "stochastic": True,
    }

    locations_params = {
        "type": "regular_cube",
        "extension_perc": 0.3,
    }

    return Degree2Polynomial(process_hyperparameters), integration_config, locations_params


def get_synthetic_data(state_dim: int) -> FIMSDEDatabatch:
    "Return example data with some dimension."

    dynamical_model, integration_config, locations_params = get_example_system(state_dim)

    path_generator = PathGenerator(
        dataset_type="FIMSDEpDataset", system=dynamical_model, integrator_params=integration_config, locations_params=locations_params
    )
    data = path_generator.generate_paths()

    return data


### Test plots of paths
def test_1D_plot_paths_in_axis() -> None:
    data = get_synthetic_data(state_dim=1)

    fig, ax = plt.subplots()
    plot_paths_in_axis(ax, data.obs_times[0], data.obs_values[0], color="red")
    plt.show()


def test_2D_plot_paths_in_axis() -> None:
    data = get_synthetic_data(state_dim=2)

    fig, ax = plt.subplots()
    plot_paths_in_axis(ax, data.obs_times[0], data.obs_values[0], color="red")
    plt.show()


def test_3D_plot_paths_in_axis() -> None:
    data = get_synthetic_data(state_dim=3)

    fig = plt.figure()
    ax = fig.add_axes(111, projection="3d")

    plot_paths_in_axis(ax, data.obs_times[0], data.obs_values[0], color="red")
    plt.show()


### Test plots of vector fields
def test_1D_plot_vector_field_in_axis() -> None:
    data = get_synthetic_data(state_dim=1)

    fig, ax = plt.subplots()
    plot_1D_vector_field_in_axis(ax, data.locations[0], data.diffusion_at_locations[0], color="red")
    plt.show()


def test_2D_plot_vector_field_in_axis() -> None:
    data = get_synthetic_data(state_dim=2)

    fig, ax = plt.subplots()
    plot_2D_vector_field_in_axis(ax, data.locations[0], data.diffusion_at_locations[0], color="red")
    plt.show()


def test_3D_plot_vector_field_in_axis() -> None:
    data = get_synthetic_data(state_dim=3)

    fig = plt.figure()
    ax = fig.add_axes(111, projection="3d")

    plot_3D_vector_field_in_axis(ax, data.locations[0], data.diffusion_at_locations[0], color="red")
    plt.show()


### Temporary place to test statistics
def test_path_statistics() -> None:
    data = get_synthetic_data(state_dim=1)
    path_statistics(data.obs_values)

    data = get_synthetic_data(state_dim=2)
    path_statistics(data.obs_values)

    data = get_synthetic_data(state_dim=3)
    path_statistics(data.obs_values)


### Test combination
def test_show_paths_vector_fields_and_statistics():
    dynamical_model, integration_config, locations_params = get_example_system(3)
    fig, stats_df = show_paths_vector_fields_and_statistics(dynamical_model, integration_config, locations_params)

    print(stats_df)
    plt.show()


### Temporary place to test HybridDynamicalSystem
def test_hybrid_dynamical_system() -> None:
    "Create paths from hybrid system with SelkovGlycosis as drift and Degree2Polynomial as diffusion."

    integration_config = {
        "method": "EulerMaruyama",
        "time_step": 0.01,
        "num_steps": 1000,
        "num_paths": 10,
        "num_locations": 512,
        "stochastic": True,
    }

    # Diffusion process
    diffusion_process_hyperparameters = {
        "name": "Degree2Polynomial",
        "data_bulk_name": "damped_linear_theory",
        "redo": True,
        "num_realizations": 5,
        "state_dim": 2,
        "show_equation": True,
        "observed_dimension": None,
        "drift_params": {},
        "diffusion_params": {
            "constant": {
                "distribution": "normal",
                "std": 1.0,
                "bernoulli_survival_rate": 0.5,
            },
            "degree_1": {
                "distribution": "normal",
                "std": 1.0,
                "bernoulli_survival_rate": 0.5,
            },
            "degree_2_squared": {
                "distribution": "normal",
                "std": 1.0,
                "bernoulli_survival_rate": 0.5,
            },
            "degree_2_mixed": {
                "distribution": "normal",
                "std": 1.0,
                "bernoulli_survival_rate": 0.5,
            },
        },
    }
    diffusion_dynamical_model = Degree2Polynomial(diffusion_process_hyperparameters)

    # Drift process
    drift_process_hyperparameters = {
        "name": "SelkovGlycosis",
        "data_bulk_name": "selkov_theory",
        "redo": True,
        "num_realizations": 5,
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
        "initial_state": {
            "distribution": "normal",
            "mean": 0.0,
            "std_dev": 1.0,
            "activation": None,
        },
    }
    drift_dynamical_model = SelkovGlycosis(drift_process_hyperparameters)

    # Hybrid system
    process_hyperparameters = {
        "name": "HybridDynamicalSystem",
        "data_bulk_name": "damped_linear_theory",
        "redo": True,
        "num_realizations": 5,
        "observed_dimension": None,
        "drift_dynamical_system": drift_dynamical_model,
        "diffusion_dynamical_system": diffusion_dynamical_model,
        "initial_state": {
            "distribution": "normal",
            "mean": 0.0,
            "std_dev": 1.0,
            "activation": None,
        },
    }

    locations_params = {
        "type": "regular_cube",
        "extension_perc": 0.3,
    }

    # generate data
    dynamical_model = HybridDynamicalSystem(process_hyperparameters)
    path_generator = PathGenerator(
        dataset_type="FIMSDEpDataset", system=dynamical_model, integrator_params=integration_config, locations_params=locations_params
    )
    path_generator.generate_paths()


if __name__ == "__main__":
    test_1D_plot_paths_in_axis()
    test_2D_plot_paths_in_axis()
    test_3D_plot_paths_in_axis()

    test_1D_plot_vector_field_in_axis()
    test_2D_plot_vector_field_in_axis()
    test_3D_plot_vector_field_in_axis()

    test_path_statistics()

    test_hybrid_dynamical_system()

    test_show_paths_vector_fields_and_statistics()
