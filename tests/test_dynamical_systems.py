import pytest

from fim.data_generation.sde.dynamical_systems import DYNAMICAL_SYSTEM_TO_MODELS
from fim.data_generation.sde.dynamical_systems_sample import PathGenerator


class TestPathGenerator:
    @pytest.fixture
    def dataset_type(self):
        return "FIMSDEpDataset"

    @pytest.fixture
    def integrator_params(self):
        return {
            "method": "EulerMaruyama",
            "time_length": 10,
            "steps_per_dt": 20,
            "num_steps": 128,
            "num_paths": 5,
            "num_locations": 256,
            "chunk_size": 50000,
            "reject_threshold": 100,
            "stochastic": True,
        }

    @pytest.fixture
    def locations_params(self):
        return {"type": "regular_cube", "extension_perc": 0.3}

    @pytest.fixture
    def system_params(self):
        return {
            "name": "Polynomials",
            "data_bulk_name": "train_deg_1",
            "num_realizations": 100,
            "observed_dimension": None,
            "state_dim": 1,
            "redo": False,
            "enforce_positivity": "abs",
            "max_degree_drift": 3,
            "max_degree_diffusion": 0,
            "observation_noise": {
                "relative": True,
                "distribution": {"name": "normal", "mean_of_mean": 0, "std_of_mean": 0, "mean_of_std": 0, "std_of_std": 0.1},
            },
            "drift_params": {
                "distribution": {"name": "normal", "std": 1},
                "degree_survival_rate": 0.5,
                "monomials_survival_distribution": {"name": "uniform", "min": 0.25, "max": 1},
                "scale": {"sample_per_dimension": False, "distribution": "uniform", "min": 0.0, "max": 2.0},
            },
            "diffusion_params": {
                "distribution": {"name": "normal", "std": 1.0},
                "degree_survival_rate": 0.5,
                "monomials_survival_distribution": {"name": "uniform", "min": 0.1, "max": 1.0},
                "scale": {"sample_per_dimension": False, "distribution": "uniform", "min": 0.0, "max": 2.0},
            },
            "initial_state": {"distribution": "normal", "mean": 0.0, "std_dev": 1.0, "survival_probability": None, "activation": None},
            "mask_sampler_params": {
                "name": "fim.data_generation.grid_samplers.BernoulliMaskSampler",
                "survival_probability": 0.5,
                "min_survival_count": 4,
            },
        }

    @pytest.fixture
    def dynamical_model(self, system_params):
        dynamical_system_name = system_params["name"]
        dynamical_model = DYNAMICAL_SYSTEM_TO_MODELS[dynamical_system_name](system_params)
        return dynamical_model

    def test_generate_paths(self, dataset_type, dynamical_model, integrator_params, locations_params):
        path_generator = PathGenerator(dataset_type, dynamical_model, integrator_params, locations_params)
        data = path_generator.generate_paths()

        assert data is not None
        assert data.obs_values.shape == data.obs_noisy_values.shape
        assert hasattr(data, "obs_mask")
        assert hasattr(data, "obs_values")
