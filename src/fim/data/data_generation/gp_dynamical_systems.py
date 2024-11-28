import torch
from torch import Tensor
import numpy as np
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from dataclasses import dataclass, field
import gpytorch.distributions as gpdst
from fim.models.gaussian_processes.utils import define_mesh_points

from typing import Tuple, List
from abc import ABC, abstractmethod


class MultivariateNormalWithJitter(gpdst.MultivariateNormal):
    """
    Defines a multivariate distribution with jittering to ensure positive definite covariance matrices.
    """

    def __init__(self, mean, covariance_matrix, epsilon=1e-7, max_retries=10, **kwargs):
        """
        :param mean: torch.Tensor
        :param covariance_matrix: torch.Tensor
        :param epsilon: float, initial jitter amount
        :param max_retries: int, maximum number of retries for jittering
        """
        global last_cov
        last_cov = covariance_matrix

        # Add jitter dynamically if necessary
        jitter = epsilon
        for attempt in range(max_retries):
            try:
                # Try initializing the parent class
                super().__init__(mean, covariance_matrix, **kwargs)
                break
            except RuntimeError as e:
                if "cholesky" in str(e).lower() or "positive definite" in str(e).lower():
                    covariance_matrix += jitter * torch.eye(covariance_matrix.shape[0], dtype=covariance_matrix.dtype)
                    jitter *= 10  # Exponentially increase jitter
                else:
                    raise
        else:
            raise ValueError("Failed to make covariance matrix positive definite after maximum retries.")

        self.inv_var = None  # Add other properties if needed

    def varinv(self):
        if self.inv_var is None:
            self.inv_var = self.covariance_matrix.inverse()
        return self.inv_var


@dataclass
class SDEGPsConfig:
    dimensions: int = 2

    # samples sizes
    number_of_kernel_samples: int = 99  # Fix: Added default value
    number_of_functions_per_kernel: int = 100
    number_of_kernels_per_file: int = 100
    total_number_of_paths: int = 10000

    # inducing points
    type_of_inducing_points: str = "random_uniform"
    number_of_inducing_points: int = 10
    inducing_point_ranges: list = field(default_factory=lambda: [(-1.0, 1.0), (-1.0, 1.0)])

    # kernels
    scale_kernel: bool = True
    drift_kernel_name: str = "ScaleRBF"
    diffusion_kernel_name: str = "ScaleRBF"

    kernel_sigma: dict = field(default_factory=lambda: {"name": "uniform", "min": 0.1, "max": 10.0})
    kernel_length_scale: dict = field(default_factory=lambda: {"name": "uniform", "min": 0.1, "max": 10.0})

    def __post_init__(self):
        self.total_number_of_paths = self.number_of_kernel_samples * self.number_of_functions_per_kernel


class InducingPointGPFunction(ABC):
    """

    This abatract class defines all objects requiered to sample functions from a GP prior
    with the use of inducing points. The functions to be sampled will be such that
    we generate number_of_kernel_samples and number_of_functions_per_kernel. This will
    be used to define the drift and the diffusion function.

    the children class should define how to sample kernel parameters and kernels
    this abstact class handles sampling and evaluation once kernels are defined

    """

    K_inducing_inducing_inv: Tensor = None
    inducing_functions: Tensor = None
    kernels: List[Kernel] = []

    def __init__(
        self,
        config: SDEGPsConfig,
        inducing_points: Tensor,
    ):
        self.config = config
        self.dimensions = config.dimensions
        self.num_inducing_points = config.number_of_inducing_points
        self.num_kernel_samples = config.number_of_kernel_samples
        self.num_functions_per_kernel = config.number_of_functions_per_kernel
        self.inducing_points = inducing_points
        assert self.inducing_points.size(1) == config.dimensions

        self.kernels = self.sample_and_set_kernels()
        self.get_inducing_function()

    @abstractmethod
    def sample_kernel_parameters(self):
        pass

    @abstractmethod
    def sample_kernel_parameter(self, param_dist):
        pass

    @abstractmethod
    def sample_and_set_kernels(self):
        pass

    def get_inducing_prior(self):
        """Sample functions using the inducing points and kernels."""
        if self.inducing_points is None:
            raise ValueError("Inducing points must be initialized before sampling inducing functions.")

        inducing_prior = []
        for kernel_list in self.kernels:
            prior_per_dimension = []
            for kernel in kernel_list:
                # Evaluate kernel on inducing points
                cov_matrix = kernel(self.inducing_points).evaluate()
                mean = torch.zeros(self.inducing_points.size(0), dtype=cov_matrix.dtype)
                # Ensure the covariance matrix is positive definite using jittering
                inducing_function = MultivariateNormalWithJitter(mean, cov_matrix, epsilon=1e-7)
                prior_per_dimension.append(inducing_function)
            inducing_prior.append(prior_per_dimension)
        return inducing_prior

    def get_inducing_function(self) -> Tuple[Tensor, Tensor]:
        """
        This function defines ONCE the elements requiered to sample
        functions from GP inducing points.

        1. we sample the hyperparatemers to obtain different kernels
        2. we evaluate the kernels at the inducing points
        3. we invert the covariance at inducing points
        4. we sample the functions in the prior

        All the values are kept during the existance of this object
        such as guarantee consistency and avoid computation

        return
        ------

        K_inducing_inducing_inv:Tensor (number_of_kernel_samples,
                                        number_of_functions_per_kernel,
                                        number_of_inducing_points,
                                        number_of_inducing_points,
                                        dimensions)

        inducing_functions : Tensor (number_of_kernel_samples,
                                     number_of_functions_per_kernel,
                                     number_of_inducing_points,
                                     number_of_inducing_points,
                                     dimensions)
        """
        self.sample_and_set_kernels()
        if self.inducing_functions is None and self.K_inducing_inducing_inv is None:
            inducing_prior = self.get_inducing_prior()
            inducing_functions = []
            K_inducing_inducing_inv = []
            for kernel_index in range(self.config.number_of_kernel_samples):
                K_inducing_inducing_per_dimension_inv = []
                inducing_functions_per_dimension = []
                for dimension_index in range(self.config.dimensions):
                    inducing_prior_per_dimension = inducing_prior[kernel_index][dimension_index]
                    # inverse kernel
                    K_inducing_inducing_per_dimension_inv.append(inducing_prior_per_dimension.varinv().unsqueeze(-1))
                    # inducing function
                    f_i = inducing_prior_per_dimension.sample(
                        sample_shape=torch.Size([self.config.number_of_functions_per_kernel])
                    ).unsqueeze(-1)
                    inducing_functions_per_dimension.append(f_i)

                K_inducing_inducing_per_dimension_inv = (
                    torch.concat(K_inducing_inducing_per_dimension_inv, dim=-1).unsqueeze(0).unsqueeze(0)
                )
                K_inducing_inducing_per_dimension_inv = K_inducing_inducing_per_dimension_inv.repeat(
                    1, self.config.number_of_functions_per_kernel, 1, 1, 1
                )
                inducing_functions_per_dimension = torch.concat(inducing_functions_per_dimension, dim=-1).unsqueeze(0)

                inducing_functions.append(inducing_functions_per_dimension)
                K_inducing_inducing_inv.append(K_inducing_inducing_per_dimension_inv)

            self.K_inducing_inducing_inv = torch.concat(K_inducing_inducing_inv, dim=0)
            self.inducing_functions = torch.concat(inducing_functions, dim=0)
        return self.K_inducing_inducing_inv, self.inducing_functions

    def evaluate_kernel_input_inducing(self, X0):
        """ "
        Args
        ----
            X0 (Tensor): [number_of_kernel_samples,number_of_functions_per_kernel,dimensions]

        Returns
        -------
        """
        self.sample_and_set_kernels()
        K_input_inducing = []
        for kernel_index in range(self.config.number_of_kernel_samples):
            K_input_inducing_per_dimension = []
            for dimension_index in range(self.config.dimensions):
                X0_per_kernel = X0[kernel_index, ...]
                multivariate_kernel: list = self.kernels[kernel_index]
                kernel_per_dimension: Kernel = multivariate_kernel[dimension_index]
                K_input_inducing_ = kernel_per_dimension.forward(X0_per_kernel, self.inducing_points).unsqueeze(0).unsqueeze(-1)
                K_input_inducing_per_dimension.append(K_input_inducing_)
            K_input_inducing_per_dimension = torch.concatenate(K_input_inducing_per_dimension, dim=-1)
            K_input_inducing.append(K_input_inducing_per_dimension)
        K_input_inducing = torch.concatenate(K_input_inducing)
        return K_input_inducing

    def __call__(self, X0):
        """
        X0
        """
        K_inducing_inducing_inv, inducing_functions = self.get_inducing_function()
        K_input_inducing = self.evaluate_kernel_input_inducing(X0)
        function_approximation = torch.einsum("kpid,kpijd,kpjd->kpd", K_input_inducing, K_inducing_inducing_inv, inducing_functions)
        return function_approximation


class ScaleRBF(InducingPointGPFunction):
    def __init__(self, config: SDEGPsConfig, inducing_points: Tensor):
        super().__init__(config, inducing_points)

    def sample_kernel_parameters(self):
        """Sample kernel hyperparameters from specified distributions."""
        sigma = self.sample_kernel_parameter(self.config.kernel_sigma)
        length_scale = self.sample_kernel_parameter(self.config.kernel_length_scale)
        return sigma, length_scale

    def sample_kernel_parameter(self, param_dist):
        """Sample values from a specified distribution. ONE PARAMETER"""
        if param_dist["name"] == "uniform":
            return np.random.uniform(param_dist["min"], param_dist["max"])
        else:
            raise ValueError(f"Unsupported distribution: {param_dist['name']}")

    def sample_and_set_kernels(self) -> List[Kernel]:
        """Construct the kernels after sampling its hyper parameters"""
        if len(self.kernels) == 0:
            """Sample multiple kernels based on the configuration."""
            for _ in range(self.num_kernel_samples):
                kernel_per_dimension = []
                for dimension in range(self.dimensions):
                    kernel_sigma, kernel_length_scale = self.sample_kernel_parameters()
                    kernel = ScaleKernel(RBFKernel(ard_num_dims=self.dimensions, requires_grad=True), requires_grad=True)
                    hypers = {
                        "raw_outputscale": torch.tensor(kernel_sigma),
                        "base_kernel.raw_lengthscale": torch.tensor(np.repeat(kernel_length_scale, self.dimensions)),
                    }
                    kernel = kernel.initialize(**hypers)
                    kernel_per_dimension.append(kernel)
                self.kernels.append(kernel_per_dimension)
        return self.kernels


KERNELS_FUNCTIONS = {"ScaleRBF": ScaleRBF}


class SDEGPDynamicalSystem:
    """ """

    def __init__(self, config: SDEGPsConfig):
        self.config = config
        self.dimensions = config.dimensions
        self.num_inducing_points = config.number_of_inducing_points
        self.num_kernel_samples = config.number_of_kernel_samples
        self.num_functions_per_kernel = config.number_of_functions_per_kernel
        self.kernels = []

        self.inducing_points = define_mesh_points(
            total_points=config.number_of_inducing_points, n_dims=config.dimensions, ranges=config.inducing_point_ranges
        )

        self.drift = KERNELS_FUNCTIONS[config.drift_kernel_name](config, inducing_points=self.inducing_points)
        self.diffusion = KERNELS_FUNCTIONS[config.diffusion_kernel_name](config, inducing_points=self.inducing_points)
