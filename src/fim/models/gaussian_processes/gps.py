import os

# import DiagLazyVariable, ZeroLazyVariable
from dataclasses import dataclass

import numpy as np
import torch
from gpytorch import distributions as gpdst
from gpytorch.kernels import Kernel
from matplotlib import pyplot as plt
from torch.distributions import MultivariateNormal


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class SDEGPConfig:
    number_of_inducing_points: int = 5
    number_of_processes: int = 10
    windows_size: int = 1
    kernel_sigma: float = 1.0
    kernel_l: float = 1.0


class MV_Normal(gpdst.MultivariateNormal):
    def __init__(self, mean, covariance_matrix, **kwargs):
        global last_cov
        last_cov = covariance_matrix
        if torch.min(torch.abs(torch.diag(covariance_matrix))) < 1e-7:
            print("Jittering")
            covariance_matrix += 1e-7 * torch.eye(covariance_matrix.shape[0])
        super().__init__(mean, covariance_matrix, **kwargs)
        self.inv_var = None

    def varinv(self):
        if self.inv_var is None:
            self.inv_var = self.covariance_matrix.inverse()
        return self.inv_var


class multivariate_normal(gpdst.MultivariateNormal):
    """
    Defines a multivariate distribution and handles the jittering
    """

    def __init__(self, mean, covariance_matrix, epsilon=1e-7, **kwargs):
        """

        :param mean: torch tensor
        :param covariance_matrix: gpytorch kernel
        :param epsilon:
        """
        global last_cov
        last_cov = covariance_matrix
        if torch.min(torch.abs(torch.diag(covariance_matrix))) < epsilon:
            print("Jittering")
            covariance_matrix += epsilon * torch.eye(covariance_matrix.shape[0])
        super().__init__(mean, covariance_matrix, **kwargs)
        self.inv_var = None

    def varinv(self):
        if self.inv_var is None:
            self.inv_var = self.covariance_matrix.inverse()
        return self.inv_var


def calculate_posterior(test_input, train_output, train_input, kernel, Sigma=None):
    """
    :param test_input: torch tensor
    :param train_input: torch tensor
    :param kernel:
    :return: predictive_mean [number_of_points], predictive_variance [number_of_points]
    """
    """

    :param test_input: torch tensor
    :param train_input: torch tensor
    :param kernel:
    :return: predictive_mean [number_of_points], predictive_variance [number_of_points]
    """
    if len(train_output.shape) == 1:
        train_output = train_output.unsqueeze(0)

    K_train_train = kernel.forward(train_input, train_input)
    K_test_train = kernel.forward(test_input, train_input)
    K_test_test = kernel.forward(test_input, test_input, diag=True).double()

    if Sigma is None:
        K_train_train = K_train_train.evaluate()
    else:
        if isinstance(Sigma, float):
            K_train_train = K_train_train.evaluate() + torch.tensor(Sigma * np.eye(len(train_input)))
        elif isinstance(Sigma, torch.Tensor):
            K_train_train = K_train_train.evaluate() + Sigma

    prior_mean = torch.zeros_like(train_input)
    prior_normal = MV_Normal(prior_mean, K_train_train)

    K_train_train_inverse = prior_normal.varinv()
    kappa = K_test_train.evaluate().matmul(K_train_train_inverse).double()

    predictive_mean = kappa.matmul(train_output.double().T).double()

    predictive_variance = kappa.matmul(K_test_train.evaluate().T.double())
    predictive_variance = predictive_variance.diag()
    predictive_variance = K_test_test - predictive_variance

    return predictive_mean.T[0], predictive_variance


class white_noise_kernel(Kernel):
    def __init__(self, variance=torch.tensor(1e-3)):
        super(white_noise_kernel, self).__init__()
        self.register_buffer("variance", variance)

    def forward(self, x1, x2=None, diag=False, **params):
        return self.__call__(x1, x2, diag, **params)

    def __call__(self, x1, x2=None, diag=False, **params):
        if x2 is None:
            if diag:
                return self.variance * torch.ones(x1.size(0), device=x1.device)
            else:
                return self.variance * torch.eye(x1.size(0), device=x1.device)
        if x1.size() == x2.size() and torch.equal(x1, x2):
            return self.__call__(x1, diag=diag, **params)
        return torch.zeros((x1.size(0), x2.size(0)))


if __name__ == "__main__":
    from gpytorch.kernels import RBFKernel, ScaleKernel

    # ==============================================
    number_of_inducing_points = 5
    number_of_processes = 10
    windows_size = 1
    kernel_sigma = 1.0
    kernel_l = 1.0
    # ========================================================================
    # DEFINE AND INITIALIZE KERNEL
    kernel = ScaleKernel(RBFKernel(ard_num_dims=windows_size, requires_grad=True), requires_grad=True) + white_noise_kernel()
    hypers = {"raw_outputscale": torch.tensor(kernel_sigma), "base_kernel.raw_lengthscale": torch.tensor(np.repeat(kernel_l, windows_size))}
    kernel.kernels[0].initialize(**hypers)

    # SAMPLE DATA
    train_input = torch.tensor(np.linspace(0.0, 10.0, number_of_inducing_points))
    prior_variance = kernel(train_input, train_input).evaluate().float()
    prior_mean = torch.zeros((1, number_of_inducing_points))  # [number_of_batches,mean_dimension]
    distribution = MultivariateNormal(prior_mean, prior_variance)
    gp_sample = distribution.rsample().float()

    # ========================================================================
    # POSTERIOR
    test_input = torch.tensor(np.linspace(0.0, 10.0, 100))
    predictive_mean, predictive_variance = calculate_posterior(test_input, gp_sample, train_input, kernel)
    upper = predictive_mean + predictive_variance
    lower = predictive_mean - predictive_variance
    # PLOT
    f, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(train_input, gp_sample.detach().numpy().T, "o")
    ax.plot(test_input, predictive_mean.detach().numpy(), "r-")
    ax.fill_between(test_input.numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
    plt.show()
