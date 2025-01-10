import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import gpytorch.distributions as gpdst
import numpy as np
import torch
import yaml
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from torch import Tensor

from fim.data.data_generation.dynamical_systems import DynamicalSystem
from fim.data.data_generation.dynamical_systems_sample import set_up_a_dynamical_system
from fim.data.datasets import FIMSDEDatabatch
from fim.models.gaussian_processes.utils import define_mesh_points


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
class IntegrationConfig:
    method: str = "EulerMaruyama"
    time_step: float = 0.01
    num_steps: int = 128
    num_paths: int = 30
    num_locations: int = 1024
    stochastic: bool = True


@dataclass
class SDEGPsConfig:
    redo: bool = False
    dimensions: int = 2
    type: str = "gps"
    file_name: str = "sdes_gps"
    # samples sizes
    number_of_kernel_samples: int = 50  # Fix: Added default value
    number_of_functions_per_kernel: int = 20
    number_of_kernels_per_file: int = 100
    total_number_of_realizations: int = 10000

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
    initial_state: dict = field(default_factory=lambda: {"name": "uniform", "min": 0.1, "max": 10.0})

    def __post_init__(self):
        self.total_number_of_realizations = self.number_of_kernel_samples * self.number_of_functions_per_kernel


class InducingPointGPFunction(ABC):
    """

    This abstract class defines all objects required to sample functions from a GP prior
    with the use of inducing points. The functions to be sampled will be such that
    we generate number_of_kernel_samples and number_of_functions_per_kernel. This will
    be used to define the drift and the diffusion function.

    the children class should define how to sample kernel parameters and kernels
    this abstract class handles sampling and evaluation once kernels are defined

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
        self.num_kernel_samples = config.number_of_kernel_samples
        self.num_functions_per_kernel = config.number_of_functions_per_kernel
        self.inducing_points = inducing_points
        self.num_inducing_points = inducing_points.size(0)

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
        This function defines ONCE the elements required to sample
        functions from GP inducing points.

        1. we sample the hyperparatemers to obtain different kernels
        2. we evaluate the kernels at the inducing points
        3. we invert the covariance at inducing points
        4. we sample the functions in the prior

        All the values are kept during the existence of this object
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
                number_of_functions_per_kernel, number_of_paths, dimensions = X0_per_kernel.shape
                X0_per_kernel = X0_per_kernel.reshape(number_of_functions_per_kernel * number_of_paths, dimensions)
                multivariate_kernel: list = self.kernels[kernel_index]
                kernel_per_dimension: Kernel = multivariate_kernel[dimension_index]
                K_input_inducing_ = kernel_per_dimension.forward(X0_per_kernel, self.inducing_points)
                K_input_inducing_ = K_input_inducing_.reshape(number_of_functions_per_kernel, number_of_paths, self.num_inducing_points)
                K_input_inducing_ = K_input_inducing_.unsqueeze(0).unsqueeze(-1)
                K_input_inducing_per_dimension.append(K_input_inducing_)
            K_input_inducing_per_dimension = torch.concatenate(K_input_inducing_per_dimension, dim=-1)
            K_input_inducing.append(K_input_inducing_per_dimension)
        K_input_inducing = torch.concatenate(K_input_inducing)
        return K_input_inducing

    def __call__(self, X0):
        """
        X0:  (number_of_kernel_samples,
              number_of_functions_per_kernel,
              number_of_paths,
              dimensions)
        """
        K_inducing_inducing_inv, inducing_functions = self.get_inducing_function()
        K_input_inducing = self.evaluate_kernel_input_inducing(X0)
        function_approximation = torch.einsum(
            "kfpid,kfpijd,kfpjd->kfpd", K_input_inducing, K_inducing_inducing_inv[:, :, None, :, :], inducing_functions[:, :, None, :, :]
        )
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

    def __init__(self, config: SDEGPsConfig, integration_config: IntegrationConfig):
        self.config = config
        self.integration_config = integration_config
        self.dimensions = config.dimensions
        self.num_inducing_points = config.number_of_inducing_points
        self.num_kernel_samples = config.number_of_kernel_samples
        self.num_functions_per_kernel = config.number_of_functions_per_kernel
        self.kernels = []

        self.num_steps = integration_config.num_steps
        self.num_paths = integration_config.num_paths
        self.dt = integration_config.time_step
        self.num_locations = integration_config.num_locations

        self.inducing_points = define_mesh_points(
            total_points=config.number_of_inducing_points, n_dims=config.dimensions, ranges=config.inducing_point_ranges
        )

        self.drift = KERNELS_FUNCTIONS[config.drift_kernel_name](config, inducing_points=self.inducing_points)
        self.diffusion = KERNELS_FUNCTIONS[config.diffusion_kernel_name](config, inducing_points=self.inducing_points)

    def generate_paths(self) -> FIMSDEDatabatch:
        """
        generate paths such that every realization of the parameters
        has a num of paths

        Returns
            DataBulk (FIMPOODEDataBulk|FIMSDEpDataBulk)
        """
        with torch.no_grad():
            states = self.sample_initial_states()
            # paths
            hidden_paths = torch.zeros(
                (
                    self.num_kernel_samples,
                    self.num_functions_per_kernel,
                    self.num_paths,
                    self.num_steps + 1,
                    self.dimensions,
                )
            )
            hidden_paths[:, :, :, 0, :] = states.clone()
            # times
            hidden_times = torch.linspace(0.0, self.num_steps * self.dt, self.num_steps + 1)
            hidden_times = hidden_times[None, None, None, :].repeat(
                self.num_kernel_samples, self.num_functions_per_kernel, self.num_paths, 1
            )  # [total_num_paths,max_diffusion_params]
            # go through iterator
            for step in range(self.num_steps):
                drift = self.drift(states)
                diffusion = self.diffusion(states)
                states = states + drift * self.dt + diffusion * torch.sqrt(torch.tensor(self.dt)) * torch.randn_like(states)
                hidden_paths[:, :, :, step + 1, :] = states.clone()

            hidden_times = hidden_times.unsqueeze(-1)
            return self.define_fim_sde_data(hidden_paths, hidden_times)

    def sample_initial_states(self):
        """ """
        states = torch.empty(
            self.config.number_of_kernel_samples,
            self.config.number_of_functions_per_kernel,
            self.integration_config.num_paths,
            self.config.dimensions,
        ).uniform_(-1.0, 1.0)
        return states

    def define_fim_sde_data(self, obs_values, obs_times) -> FIMSDEDatabatch:
        """
        Defines hyper cube and evaluates drift and diffusion there

        Args:
            data:  obs_values,obs_times,drift_parameters,diffusion_parameters
        """
        process_dimension = torch.full((obs_values.size(0), 1), self.dimensions)
        num_kernel_samples, num_functions_per_kernel, number_of_paths, num_steps, dimensions = obs_values.shape
        total_number_of_realizations = num_kernel_samples * num_functions_per_kernel

        locations = define_mesh_points(self.num_locations, dimensions)
        num_locations = locations.size(0)

        locations = locations[None, None, :, :].repeat((self.num_kernel_samples, self.num_functions_per_kernel, 1, 1))
        drift_at_locations = self.drift(locations)
        diffusion_at_locations = self.diffusion(locations)

        obs_times = obs_times.reshape(total_number_of_realizations, number_of_paths, num_steps, 1)
        obs_values = obs_values.reshape(total_number_of_realizations, number_of_paths, num_steps, dimensions)

        locations = locations.reshape(total_number_of_realizations, num_locations, dimensions)
        diffusion_at_locations = diffusion_at_locations.reshape(total_number_of_realizations, num_locations, dimensions)
        drift_at_locations = drift_at_locations.reshape(total_number_of_realizations, num_locations, dimensions)

        data = FIMSDEDatabatch(
            locations=locations,
            obs_times=obs_times,
            obs_values=obs_values,
            obs_noisy_values=None,  # for now
            obs_mask=None,  # for now
            diffusion_at_locations=diffusion_at_locations,
            drift_at_locations=drift_at_locations,
            # diffusion_parameters=diffusion_parameters,
            # drift_parameters=drift_parameters,
            # process_label=process_label,
            process_dimension=process_dimension,
        )
        return data


def set_up_a_gp_dynamical_system(
    dataset_type: str,
    params_yaml: dict,
    integrator_params: dict,
    experiment_dir: str,
    return_data: bool = True,
) -> DynamicalSystem | FIMSDEDatabatch:
    """
    Takes a dict of parameters from yaml and creates
    the dynamical system model and generate the data accordingly
    every time the data is generated it will be saved and will only be
    regenerated is so decided

    Args:
        -dataset_type (str): which type of dataset will be used
        -params_yaml (dict): dynamical system model parameters as dict
        -integrator_params: itegrator parameters
        -experiment_dir (str): where all the models data is saved
        -return_data (bool): if true returns the FIMSDEpDataBulk otherwise the model

    Returns
        SDEGPDynamicalSystem|FIMSDEpDataBulk
    """
    integrator_config = IntegrationConfig(**integrator_params)
    sdegp_config = SDEGPsConfig(**params_yaml)

    study_name_str = params_yaml.get("file_name", "default")
    redo_study = params_yaml.get("redo", False)
    study_path = Path(os.path.join(experiment_dir, study_name_str + ".tr"))

    # Create an instance of OneCompartmentModelParams with the loaded values
    dynamical_model = system = SDEGPDynamicalSystem(sdegp_config, integrator_config)
    if return_data:
        data: FIMSDEDatabatch
        # study data does not exist we generated again
        if not study_path.exists():
            data = dynamical_model.generate_paths()
            torch.save(data, study_path)
            return data
        else:
            # data exist but we must simulate again
            if redo_study:
                data = dynamical_model.generate_paths()
                torch.save(data, study_path)
                return data
            # data exist and we take it
            else:
                data = torch.load(study_path)
                return data
    return system


def define_dynamicals_models_from_yaml(
    yaml_file: str,
    return_data: bool = True,
) -> Tuple[
    str,
    List[DynamicalSystem | SDEGPDynamicalSystem | FIMSDEDatabatch],
    List[DynamicalSystem | SDEGPDynamicalSystem | FIMSDEDatabatch],
    List[DynamicalSystem | SDEGPDynamicalSystem | FIMSDEDatabatch],
]:
    """
    Function to load or generate different studies from a yaml file,
    this is the function that will allow the dataloader to get the data
    from the dynamic simulations

    Args:
        yaml_file: str of yaml file that contains a list of hyper parameters
        from different compartment models, one such hyperparameters allows the
        the set_up_a_study function (above) to generate one population study

        return_data: bool if false will return the dynamic system if true
                     will return the DataBulk object
    """
    from fim import data_path

    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)

    # check the experiment folder exist
    experiment_name = data["experiment_name"]
    experiment_dir = os.path.join(data_path, "processed", experiment_name)
    if not os.path.exists(experiment_dir):
        # Create the folder
        os.makedirs(experiment_dir)

    # data type
    dataset_type = data["dataset_type"]
    # integrator params
    integrator_params = data["integration"]
    # locations params
    locations_params = data["locations"]

    # generate the data
    train_studies: List[DynamicalSystem | SDEGPDynamicalSystem, FIMSDEDatabatch] = []
    test_studies: List[DynamicalSystem | SDEGPDynamicalSystem, FIMSDEDatabatch] = []
    validation_studies: List[DynamicalSystem | SDEGPDynamicalSystem, FIMSDEDatabatch] = []

    for params_yaml in data["train"]:
        if "SDEGPsConfig" in params_yaml.keys():
            params_yaml_ = params_yaml["SDEGPsConfig"]
            data_or_model = set_up_a_gp_dynamical_system(dataset_type, params_yaml_, integrator_params, experiment_dir, return_data)
        else:
            data_or_model = set_up_a_dynamical_system(
                dataset_type, params_yaml, integrator_params, locations_params, experiment_dir, return_data
            )

        train_studies.append(data_or_model)

    for params_yaml in data["test"]:
        if "SDEGPsConfig" in params_yaml.keys():
            params_yaml_ = params_yaml["SDEGPsConfig"]
            data_or_model = set_up_a_gp_dynamical_system(dataset_type, params_yaml_, integrator_params, experiment_dir, return_data)
        else:
            data_or_model = set_up_a_dynamical_system(
                dataset_type, params_yaml, integrator_params, locations_params, experiment_dir, return_data
            )
        test_studies.append(data_or_model)

    for params_yaml in data["validation"]:
        if "SDEGPsConfig" in params_yaml.keys():
            params_yaml_ = params_yaml["SDEGPsConfig"]
            data_or_model = set_up_a_gp_dynamical_system(dataset_type, params_yaml_, integrator_params, experiment_dir, return_data)
        else:
            data_or_model = set_up_a_dynamical_system(
                dataset_type, params_yaml, integrator_params, locations_params, experiment_dir, return_data
            )
        validation_studies.append(data_or_model)

    return (dataset_type, experiment_name, train_studies, test_studies, validation_studies)


def get_dynamicals_system_data_from_yaml(yaml_file: str, label: str) -> List[DynamicalSystem | SDEGPDynamicalSystem | FIMSDEDatabatch]:
    """
    Function to load or generate different dynamical system data from yaml.

    Args:
        yaml_file (str): path to yaml file that contains configs for dynamical system data
        label (str): name of data in yaml to generate

    Returns:
        systems_data (list[FIMSDEDatabatch]): data from all specified dynamical systems

    """
    from fim import data_path

    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)

    # check the experiment folder exist
    experiment_name = data["experiment_name"]
    experiment_dir = os.path.join(data_path, "processed", experiment_name)
    if not os.path.exists(experiment_dir):
        # Create the folder
        os.makedirs(experiment_dir)

    # data type
    dataset_type = data["dataset_type"]
    # integrator params
    integrator_params = data["integration"]
    # locations params
    locations_params = data["locations"]

    # generate the data
    systems_data: List[DynamicalSystem | SDEGPDynamicalSystem, FIMSDEDatabatch] = []

    for params_yaml in data[label]:
        if "SDEGPsConfig" in params_yaml.keys():
            params_yaml_ = params_yaml["SDEGPsConfig"]
            data = set_up_a_gp_dynamical_system(dataset_type, params_yaml_, integrator_params, experiment_dir, True)
        else:
            data = set_up_a_dynamical_system(dataset_type, params_yaml, integrator_params, locations_params, experiment_dir, True)
        systems_data.append(data)

    return systems_data
