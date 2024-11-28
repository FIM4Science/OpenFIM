from fim.data.data_generation.gp_dynamical_systems import SDEGPsConfig, SDEGPDynamicalSystem, ScaleRBF
from fim.models.gaussian_processes.utils import define_mesh_points

import pytest
import torch

# Assuming SDEGPsConfig and SDEGPDynamicalSystem are imported or defined in the same file.


class TestInducingPointFunctions:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup fixtures for the test class."""
        self.config = SDEGPsConfig()
        self.system = SDEGPDynamicalSystem(self.config)

        self.inducing_points = define_mesh_points(
            total_points=self.config.number_of_inducing_points, n_dims=self.config.dimensions, ranges=self.config.inducing_point_ranges
        )

        self.X0 = torch.empty(
            self.config.number_of_kernel_samples, self.config.number_of_functions_per_kernel, self.config.dimensions
        ).uniform_(-1.0, 1.0)

        # Assert the output shape is as expected.
        self.expected_shape = (self.config.number_of_kernel_samples, self.config.number_of_functions_per_kernel, self.config.dimensions)

    def test_scale_rbf(self):
        inducing_function = ScaleRBF(self.config, self.inducing_points)
        F = inducing_function(self.X0)
        assert F.shape == self.expected_shape, f"Expected shape {self.expected_shape}, but got {self.D.shape}"


class TestSDEGPDynamicalSystem:
    """Test class for SDEGPDynamicalSystem."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup fixtures for the test class."""
        self.config = SDEGPsConfig()
        self.system = SDEGPDynamicalSystem(self.config)

    def test_diffusion_shape(self):
        """Test to check if the diffusion method produces the correct shape."""
        X0 = torch.empty(self.config.number_of_kernel_samples, self.config.number_of_functions_per_kernel, self.config.dimensions).uniform_(
            -1.0, 1.0
        )

        D = self.system.diffusion(X0)

        # Assert the output shape is as expected.
        expected_shape = (self.config.number_of_kernel_samples, self.config.number_of_functions_per_kernel, self.config.dimensions)
        assert D.shape == expected_shape, f"Expected shape {expected_shape}, but got {D.shape}"
