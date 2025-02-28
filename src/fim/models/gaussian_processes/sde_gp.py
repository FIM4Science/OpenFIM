import sys
from typing import List, Tuple

import numpy as np
import torch
from torch import matmul as m


sys.path.insert(0, ".")

from dataclasses import dataclass

from gpytorch.kernels import Kernel, PolynomialKernel, RBFKernel, ScaleKernel
from matplotlib import pyplot as plt

from fim.models.gaussian_processes.sdes import DoubleWellDrift, MaxDiffusion


@dataclass
class SDEParams:
    a: float = 1.0  # Coefficient for x^4 term in the potential
    b: float = 1.0  # Coefficient for x^2 term in the potential
    sigma: float = 0.5  # Diffusion coefficient
    dt: float = 0.01  # Time step size
    n_steps: int = 3000  # Number of time steps
    x0: float = 1.0  # Initial condition
    num_paths: int = 10  # Number of paths to simulate


class DenseGPSDE:
    def __init__(
        self,
        dense_path_realization: torch.Tensor,
        dt: float,
        kernels: List[Kernel] = None,
        kernel_parameters: List[List[Tuple[float, float]]] = None,
        diffusion_sigma=1.0,
    ):
        self.dt = dt
        self.dense_path_realization = dense_path_realization
        self.dimensions = dense_path_realization.size(1)
        self.number_of_points = dense_path_realization.size(0)
        self.kernel_parameters = kernel_parameters
        self.diffusion_sigma = diffusion_sigma
        # define kernel
        if kernels is None:
            self.kernels = self.get_generic_kernel()
        else:
            assert len(kernels) == self.dimensions
            self.kernels = kernels

    def get_generic_kernel(self):
        kernels = []
        for j in range(self.dimensions):
            if self.kernel_parameters is None:
                kernel_l, kernel_sigma = 1.0, 1.0
            else:
                kernel_l, kernel_sigma = self.kernel_parameters[j][0], self.kernel_parameters[j][1]
            kernel = ScaleKernel(RBFKernel(ard_num_dims=self.dimensions, requires_grad=True), requires_grad=True)
            hypers = {
                "raw_outputscale": torch.tensor(kernel_sigma),
                "base_kernel.raw_lengthscale": torch.tensor(np.repeat(kernel_l, self.dimensions)),
            }
            kernel = kernel.initialize(**hypers)
            kernels.append(kernel)
        return kernels

    def gp_posterior(self, test_input, train_output, train_input, kernel, Sigma):
        K_nn = kernel.forward(train_input, train_input) + Sigma
        K_xn = kernel.forward(test_input, train_input)
        K_nn_inverse = torch.inverse(K_nn)
        kappa = m(K_xn, K_nn_inverse)
        f_x = m(kappa, train_output)
        # K_xx = kernel.forward(test_input, test_input, diag=True)
        # predictive_variance = kappa.matmul(K_test_train.evaluate().T.double())
        # predictive_variance = predictive_variance.diag()
        # predictive_variance = K_test_test - predictive_variance
        return f_x

    def __call__(
        self,
        X_test,
        diffusion=None,
    ):
        """
        Computes the posterior f_x
        """
        y = (self.dense_path_realization[1:, :] - self.dense_path_realization[:-1, :]) / self.dt
        if diffusion is None:
            y_tilde = ((self.dense_path_realization[1:, :] - self.dense_path_realization[:-1, :]) ** 2.0) / self.dt
        else:
            Sigma = diffusion(self.dense_path_realization[:-1]) / self.dt
        X_train = self.dense_path_realization[:-1]

        D = []
        F = []
        for j in range(self.dimensions):
            # Obtain diffusion
            if diffusion is None:
                sigma_d = torch.diag(self.diffusion_sigma * torch.ones(self.number_of_points - 1))
                d_x = self.gp_posterior(
                    test_input=X_train, train_output=y_tilde[:, j].unsqueeze(-1), train_input=X_train, kernel=self.kernels[j], Sigma=sigma_d
                )
                sigma = torch.diag(d_x.squeeze())
                D.append(d_x)
            else:
                sigma = torch.diag(Sigma[:, j])
            # Obtain drift
            f_x = self.gp_posterior(
                test_input=X_test, train_output=y[:, j].unsqueeze(-1), train_input=X_train, kernel=self.kernels[j], Sigma=sigma
            )
            F.append(f_x)
        F = torch.cat(F, dim=1)

        if diffusion is None:
            D = torch.cat(D, dim=1)
            return F, D
        else:
            return F


if __name__ == "__main__":
    from fim.models.gaussian_processes.utils import define_grid_ranges, define_mesh_points

    # gp_path = r"C:\Users\cesar\Desktop\Projects\DiffusiveGenerativeModelling\OurCodes\markov_bridges\data\raw\gps\2D.tr"
    # gp_path = r"C:\Users\cesar\Desktop\Projects\DiffusiveGenerativeModelling\OurCodes\markov_bridges\data\raw\gps\double_well.tr"
    gp_path = (
        r"C:\Users\cesar\Desktop\Projects\DiffusiveGenerativeModelling\OurCodes\markov_bridges\data\raw\gps\double_well_max_diffusion.tr"
    )

    data = torch.load(gp_path)
    dense_path_realization = data["dense_path_realization"]
    dt = data["dt"]
    sigma = data["sigma"]

    # drift = TwoDimensionalSynDrift()
    drift = DoubleWellDrift()
    # diffusion = ConstantDiffusion(sigma)
    diffusion = MaxDiffusion()

    # =============================================
    # WHERE TO EVALUATE THE FUNCTION
    # =============================================
    # where to evaluate the drift function
    dimensions = dense_path_realization.size(1)
    num_evaluation_points = 2000
    ranges_ = define_grid_ranges(dense_path_realization, ignore_percentatge=0.1)
    X_test = define_mesh_points(total_points=num_evaluation_points, n_dims=dimensions, ranges=ranges_)
    # ===============================================================================
    # j = 0
    # kernels_parameters = [(0.05,1.),(.1,1.)]
    # sde_gp = DenseGPSDE(dense_path_realization,
    #                    dt,
    #                    kernels=[PolynomialKernel(power=4)])
    # f_x = sde_gp(X_test,diffusion)
    # plt.plot(X_test[:,j].detach().numpy(),drift(X_test)[:,j].detach().numpy(),"r*")
    # plt.plot(X_test[:,j].detach().numpy(),f_x[:,j].detach().numpy(),"b*")
    # plt.show()

    # ===============================================================================
    j = 0
    # kernels_parameters = [(0.05,1.),(.1,1.)]
    sde_gp = DenseGPSDE(dense_path_realization, dt, kernels=[PolynomialKernel(power=4)], diffusion_sigma=10.0)
    f_x, D_x = sde_gp(X_test)
    plt.plot(X_test[:, j].detach().numpy(), drift(X_test)[:, j].detach().numpy(), "r*")
    plt.plot(X_test[:, j].detach().numpy(), f_x[:, j].detach().numpy(), "b*")
    plt.show()

    real_diffusion = diffusion(dense_path_realization[:-1])
    plt.plot(dense_path_realization[:-1, j].detach().numpy(), real_diffusion[:, j].detach().numpy(), "r*")
    plt.plot(dense_path_realization[:-1, j].detach().numpy(), torch.sqrt(torch.clamp(D_x[:, j], 0.0)).detach().numpy(), "b*")
    plt.show()
