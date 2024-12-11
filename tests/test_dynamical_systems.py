import os

import pytest
import torch

from fim.data.data_generation.gp_dynamical_systems import (
    IntegrationConfig,
    ScaleRBF,
    SDEGPDynamicalSystem,
    SDEGPsConfig,
    define_dynamicals_models_from_yaml,
)
from fim.data.dataloaders import FIMSDEDataloader
from fim.data.datasets import FIMSDEDatabatch, FIMSDEDatabatchTuple
from fim.models.gaussian_processes.utils import define_mesh_points


# Assuming SDEGPsConfig and SDEGPDynamicalSystem are imported or defined in the same file.
class TestInducingPointFunctions:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup fixtures for the test class."""
        self.config = SDEGPsConfig()
        self.integration_config = IntegrationConfig()

        self.inducing_points = define_mesh_points(
            total_points=self.config.number_of_inducing_points, n_dims=self.config.dimensions, ranges=self.config.inducing_point_ranges
        )

        self.X0 = torch.empty(
            self.config.number_of_kernel_samples,
            self.config.number_of_functions_per_kernel,
            self.integration_config.num_paths,
            self.config.dimensions,
        ).uniform_(-1.0, 1.0)

        # Assert the output shape is as expected.
        self.expected_shape = (
            self.config.number_of_kernel_samples,
            self.config.number_of_functions_per_kernel,
            self.integration_config.num_paths,
            self.config.dimensions,
        )

    def test_scale_rbf(self):
        inducing_function = ScaleRBF(self.config, self.inducing_points)
        F = inducing_function(self.X0)
        assert F.shape == self.expected_shape, f"Expected shape {self.expected_shape}, but got {F.shape}"


class TestSDEGPDynamicalSystem:
    """Test class for SDEGPDynamicalSystem."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup fixtures for the test class."""
        self.config = SDEGPsConfig(total_number_of_realizations=10, number_of_functions_per_kernel=2, number_of_kernels_per_file=2)
        self.integration_config = IntegrationConfig()
        self.system = SDEGPDynamicalSystem(self.config, self.integration_config)

    def test_diffusion_shape(self):
        """Test to check if the diffusion method produces the correct shape."""
        X0 = torch.empty(
            self.config.number_of_kernel_samples,
            self.config.number_of_functions_per_kernel,
            self.integration_config.num_paths,
            self.config.dimensions,
        ).uniform_(-1.0, 1.0)
        D = self.system.diffusion(X0)
        # Assert the output shape is as expected.
        expected_shape = (
            self.config.number_of_kernel_samples,
            self.config.number_of_functions_per_kernel,
            self.integration_config.num_paths,
            self.config.dimensions,
        )
        assert D.shape == expected_shape, f"Expected shape {expected_shape}, but got {D.shape}"

    def test_generate(self):
        self.integration_config: IntegrationConfig
        expected_shape = (
            self.config.total_number_of_realizations,
            self.integration_config.num_paths,
            self.integration_config.num_steps + 1,
            self.config.dimensions,
        )

        data: FIMSDEDatabatch = self.system.generate_paths()

        assert data.obs_values.shape == expected_shape, f"Expected shape {expected_shape}, but got {data.obs_values.shape}"

    def test_generate_and_save_from_yaml_file(self):
        from fim import project_path

        yaml_path = os.path.join(project_path, "tests", "resources", "config", "gp-sde-systems-hyperparameters.yaml")
        dataset_type, experiment_name, train_studies, test_studies, validation_studies = define_dynamicals_models_from_yaml(
            yaml_path, return_data=True
        )

        # Check that train_studies is a list
        assert isinstance(train_studies, list), f"Expected train_studies to be a list, but got {type(train_studies)}"

        # Check that all components of train_studies are strings
        for study in train_studies:
            assert isinstance(study, FIMSDEDatabatch), f"Expected element in train_studies to be of type str, but got {type(study)}"

            # Check that train_studies is a list
        assert isinstance(test_studies, list), f"Expected train_studies to be a list, but got {type(train_studies)}"

        # Check that all components of train_studies are strings
        for study in test_studies:
            assert isinstance(study, FIMSDEDatabatch), f"Expected element in train_studies to be of type str, but got {type(study)}"

    def test_dataloader_from_yaml(self):
        yaml_path = str(os.path.join("tests", "resources", "config", "gp-sde-systems-hyperparameters.yaml"))
        dataloaders = FIMSDEDataloader(data_paths=yaml_path, data_type="theory", random_grid=False, random_paths=False, batch_size=32)
        databatch: FIMSDEDatabatchTuple = next(dataloaders.train_it.__iter__())
        print(databatch.drift_at_locations.shape)
        assert dataloaders is not None
