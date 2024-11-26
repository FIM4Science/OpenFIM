import os
import sys
import torch
import numpy as np
from torch import matmul as m
from matplotlib import pyplot as plt
from fim.models.gaussian_processes.sdes import SDE,ConstantDiffusion,TwoDimensionalSynDrift,DoubleWellDrift
from torch.autograd.functional import jacobian
from typing import List,Tuple

from torch.distributions import MultivariateNormal

sys.path.insert(0,".")

from fim.models.gaussian_processes.utils import define_grid_ranges,define_mesh_points
from fim.models.gaussian_processes.gps import MultivariateNormal,white_noise_kernel
from fim.models.gaussian_processes.gps import white_noise_kernel
from gpytorch.kernels import Kernel,RBFKernel, ScaleKernel
import numpy as np

class SparseGPSDE:
    """
    """
    def __init__(
            self,
            dense_path_realization:torch.Tensor,
            dt:float,
            kernels:List[Kernel]=None,
            kernel_parameters:List[List[Tuple[float,float]]]=None,
            num_inducing_points:int=100,
            realization_as_inducing=False,
        ):
        """
        """
        self.dt = dt
        self.dense_path_realization = dense_path_realization
        self.dimensions = dense_path_realization.size(1)
        self.kernel_parameters = kernel_parameters

        # define kernel 
        if kernels is None:
            self.kernels = self.get_generic_kernel()
        else:
            self.kernels = kernels

        # define inducing points
        self.inducing_points = self.get_inducing_points(num_inducing_points,realization_as_inducing)

    def get_generic_kernel(self):
        kernels = []
        for j in range(self.dimensions):
            if self.kernel_parameters is None:
                kernel_l, kernel_sigma  = 1.,1.
            else:
                kernel_l, kernel_sigma = self.kernel_parameters[j][0],self.kernel_parameters[j][1]
            kernel = ScaleKernel(RBFKernel(ard_num_dims=self.dimensions, 
                                           requires_grad=True),
                                           requires_grad=True)
            hypers = {"raw_outputscale": torch.tensor(kernel_sigma),
                      "base_kernel.raw_lengthscale": torch.tensor(np.repeat(kernel_l, self.dimensions))}
            kernel = kernel.initialize(**hypers)
            kernels.append(kernel)
        return kernels

    def get_inducing_points(self,num_inducing_points,realization_as_inducing=False):
        if not realization_as_inducing:
            self.num_inducing_points = num_inducing_points
            ranges_ = define_grid_ranges(self.dense_path_realization,ignore_percentatge=0.)
            inducing_points = define_mesh_points(total_points=self.num_inducing_points,
                                                n_dims=self.dimensions,
                                                ranges=ranges_)
        else:
            inducing_points = self.dense_path_realization[:-1,:]
            self.num_inducing_points = self.dense_path_realization.size(0) - 1
        return inducing_points
        
    def inference(
            self,
            diffusion
        ):
        """
        """
        y = (self.dense_path_realization[1:,:] - self.dense_path_realization[:-1,:])/self.dt
        Diff = diffusion(self.dense_path_realization[:-1,:])

        j = 0
        self.F = []
        for j in range(self.dimensions):
            y_j = y[:,j].unsqueeze(-1)
            K_ns = self.kernels[j].forward(self.dense_path_realization[:-1,:], self.inducing_points)
            K_ss = self.kernels[j].forward(self.inducing_points, self.inducing_points) + torch.eye(self.num_inducing_points)*1e-3
            K_ss_inv = torch.inverse(K_ss)

            D_j_inv = torch.diag(1./Diff[:,j])
            pi_j = m(K_ns,K_ss_inv)
            Omega_j = m(pi_j.T,m(D_j_inv*self.dt,pi_j))

            A = torch.inverse(torch.eye(self.num_inducing_points) + m(Omega_j,K_ss)) 
            reference_f = m(pi_j.T,m(D_j_inv*self.dt,y_j))
            reference_f = m(A,reference_f)
            self.F.append(reference_f)

    def __call__(self,x):
        F = []
        for j in range(self.dimensions):
            K_xs = self.kernels[j].forward(x, self.inducing_points)
            f_x = m(K_xs,self.F[j])
            F.append(f_x)
        F = torch.cat(F,dim=1)
        return F

if __name__=="__main__":
    num_evaluation_points = 1000
    #=============================================
    # READ SDEs
    #=============================================
    #gp_path = r"C:\Users\cesar\Desktop\Projects\DiffusiveGenerativeModelling\OurCodes\markov_bridges\data\raw\gps\double_well.tr"
    gp_path = r"C:\Users\cesar\Desktop\Projects\DiffusiveGenerativeModelling\OurCodes\markov_bridges\data\raw\gps\2D.tr"
    data = torch.load(gp_path)
    dense_path_realization = data["dense_path_realization"]
    dimensions = dense_path_realization.size(1)
    sigma = data["sigma"]
    dt = data["dt"]

    drift = TwoDimensionalSynDrift()
    #drift = DoubleWellDrift()
    diffusion = ConstantDiffusion(sigma)
    #=============================================
    # SPARSE GP DENSE DRIFT ESTIMATION
    #=============================================
    # where to evaluate the drift function
    ranges_ = define_grid_ranges(dense_path_realization)
    evaluation_points = define_mesh_points(total_points=num_evaluation_points,
                                           n_dims=dimensions,
                                           ranges=ranges_)
    sparse_gp_sde = SparseGPSDE(dense_path_realization,dt=dt,num_inducing_points= 100)
    sparse_gp_sde.inference(diffusion)
    F = sparse_gp_sde(evaluation_points)
    f_x = F[:,1]
    # real drift for comparison
    real_drift = drift(evaluation_points)
    plt.plot(f_x.detach().numpy())
    plt.plot(real_drift[:,1].detach().numpy(),"r-")
    plt.show()

