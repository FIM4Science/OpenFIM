import pytest

from fim.data_generation.sde.dynamical_systems import DYNAMICAL_SYSTEM_TO_MODELS
from fim.data_generation.sde.dynamical_systems_sample import PathGenerator


# ### GPs no longer supported
# Assuming SDEGPsConfig and SDEGPDynamicalSystem are imported or defined in the same file.
# class TestInducingPointFunctions:
#     @pytest.fixture(autouse=True)
#     def setup(self):
#         """Setup fixtures for the test class."""
#         self.config = SDEGPsConfig()
#         self.integration_config = IntegrationConfig()
#
#         self.inducing_points = define_mesh_points(
#             total_points=self.config.number_of_inducing_points, n_dims=self.config.dimensions, ranges=self.config.inducing_point_ranges
#         )
#
#         self.X0 = torch.empty(
#             self.config.number_of_kernel_samples,
#             self.config.number_of_functions_per_kernel,
#             self.integration_config.num_paths,
#             self.config.dimensions,
#         ).uniform_(-1.0, 1.0)
#
#         # Assert the output shape is as expected.
#         self.expected_shape = (
#             self.config.number_of_kernel_samples,
#             self.config.number_of_functions_per_kernel,
#             self.integration_config.num_paths,
#             self.config.dimensions,
#         )
#
#     def test_scale_rbf(self):
#         inducing_function = ScaleRBF(self.config, self.inducing_points)
#         F = inducing_function(self.X0)
#         assert F.shape == self.expected_shape, f"Expected shape {self.expected_shape}, but got {F.shape}"
#
#
# class TestSDEGPDynamicalSystem:
#     """Test class for SDEGPDynamicalSystem."""
#
#     @pytest.fixture(autouse=True)
#     def setup(self):
#         """Setup fixtures for the test class."""
#         self.config = SDEGPsConfig(total_number_of_realizations=10, number_of_functions_per_kernel=2, number_of_kernels_per_file=2)
#         self.integration_config = IntegrationConfig()
#         self.system = SDEGPDynamicalSystem(self.config, self.integration_config)
#
#     def test_diffusion_shape(self):
#         """Test to check if the diffusion method produces the correct shape."""
#         X0 = torch.empty(
#             self.config.number_of_kernel_samples,
#             self.config.number_of_functions_per_kernel,
#             self.integration_config.num_paths,
#             self.config.dimensions,
#         ).uniform_(-1.0, 1.0)
#         D = self.system.diffusion(X0)
#         # Assert the output shape is as expected.
#         expected_shape = (
#             self.config.number_of_kernel_samples,
#             self.config.number_of_functions_per_kernel,
#             self.integration_config.num_paths,
#             self.config.dimensions,
#         )
#         assert D.shape == expected_shape, f"Expected shape {expected_shape}, but got {D.shape}"
#
#     def test_generate(self):
#         self.integration_config: IntegrationConfig
#         expected_shape = (
#             self.config.total_number_of_realizations,
#             self.integration_config.num_paths,
#             self.integration_config.num_steps + 1,
#             self.config.dimensions,
#         )
#
#         data: FIMSDEDatabatch = self.system.generate_paths()
#
#         assert data.obs_values.shape == expected_shape, f"Expected shape {expected_shape}, but got {data.obs_values.shape}"
#
#     def test_generate_and_save_from_yaml_file(self):
#         from fim import project_path
#
#         yaml_path = os.path.join(project_path, "tests", "resources", "config", "gp-sde-systems-hyperparameters.yaml")
#         dataset_type, experiment_name, train_studies, test_studies, validation_studies = define_dynamicals_models_from_yaml(
#             yaml_path, return_data=True
#         )
#
#         # Check that train_studies is a list
#         assert isinstance(train_studies, list), f"Expected train_studies to be a list, but got {type(train_studies)}"
#
#         # Check that all components of train_studies are strings
#         for study in train_studies:
#             assert isinstance(study, FIMSDEDatabatch), f"Expected element in train_studies to be of type str, but got {type(study)}"
#
#             # Check that train_studies is a list
#         assert isinstance(test_studies, list), f"Expected train_studies to be a list, but got {type(train_studies)}"
#
#         # Check that all components of train_studies are strings
#         for study in test_studies:
#             assert isinstance(study, FIMSDEDatabatch), f"Expected element in train_studies to be of type str, but got {type(study)}"
#
#     @pytest.mark.skip(
#         reason="There is some bug in loading loading the data. It is as if the paths, drifts and diffusions don't `belong` together."
#     )
#     def test_dataloader_from_yaml(self):
#         yaml_path = str(os.path.join("tests", "resources", "config", "gp-sde-systems-hyperparameters.yaml"))
#         dataloaders = FIMSDEDataloader(data_paths=yaml_path, data_type="theory", random_grid=False, random_paths=False, batch_size=32)
#         databatch: FIMSDEDatabatchTuple = next(dataloaders.train_it.__iter__())
#         print(databatch.drift_at_locations.shape)
#         assert dataloaders is not None


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
                "name": "fim.sampling.grid_samplers.BernoulliMaskSampler",
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
