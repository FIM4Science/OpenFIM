import torch
from torch import matmul as m
from fim.models.gaussian_processes.sdes import TwoDimensionalSynDrift, ConstantDiffusion
from fim.models.gaussian_processes.sde_sparse_gp import SparseGPSDE
from matplotlib import pyplot as plt
from torch.autograd.functional import jacobian
from fim.models.gaussian_processes.utils import define_grid_ranges, define_mesh_points
from torch.distributions import MultivariateNormal, Normal


def safe_inverse(matrix, epsilon=1e-6):
    # Check if the matrix is square
    if matrix.shape[-2] != matrix.shape[-1]:
        raise ValueError("Input matrix must be square for inversion.")

    batch_size = matrix.shape[0]
    n = matrix.shape[-1]

    # Create an identity matrix with the same device and add a batch dimension
    eye = torch.eye(n, device=matrix.device).unsqueeze(0).expand(batch_size, n, n)

    # Check if any diagonal elements are too small (approaching singularity) per batch
    diag_elements = torch.abs(torch.diagonal(matrix, dim1=-2, dim2=-1))
    min_diag = torch.min(diag_elements, dim=-1).values

    # Add jittering where needed
    jitter_mask = min_diag < epsilon
    if jitter_mask.any():
        print("Jittering: adding small epsilon to the diagonal of matrices where needed")
        matrix[jitter_mask] += epsilon * eye[jitter_mask]

    # Compute the inverse for all matrices in the batch
    return torch.inverse(matrix)


def where_mc_time(mc_times, observation_times):
    """
    mc_times: shape[nmc]
    observation_times: shape[n_observations]
    where_mc_time: shape[nmc] index of observations times where mc_times is located
    """
    n_observations = observation_times.size(0)
    right_ = mc_times[:, None] > observation_times[: n_observations - 1][None, :]
    left_ = mc_times[:, None] < observation_times[1:][None, :]
    where_mc_time_index = right_ * left_
    where_mc_time_index = torch.argmax(where_mc_time_index.float(), dim=1)
    return where_mc_time_index


def jacobian_of_drift(drift, inducing_points):
    """
    parameters
    ----------
    drift: R^D - > R^D
    inducing_points: [n,D]

    returns
    -------
    J: shape[n,D,D]
    """
    J = []
    for i in range(inducing_points.size(1)):

        def drift_i(x):
            return drift(x)[:, i]

        j_i = jacobian(drift_i, inducing_points)
        j_i = j_i[range(inducing_points.size(0)), range(inducing_points.size(0))]
        J.append(j_i.clone().unsqueeze(-1))
    J = torch.cat(J, dim=-1)
    return J


def jacobian_of_drift_finite_diferences(f, x, epsilon=1e-5):
    """
    Compute the Jacobian of the function f: R^3 -> R^3 using finite differences for a batch of points,
    without using explicit loops.

    Args:
    - f (callable): the function mapping R^3 to R^3
    - x (torch.Tensor): input tensor of shape (number_of_points, 3)
    - epsilon (float): small perturbation value for finite differences

    Returns:
    - J (torch.Tensor): Jacobian matrix for each point, shape (number_of_points, 3, 3)
    """
    num_points = x.shape[0]
    dimensions = x.shape[1]

    # Compute the function value for all points
    fx = f(x)  # Shape: (number_of_points, dimensions)

    # Prepare perturbations for all dimensions (broadcasting approach)
    perturbations = torch.eye(dimensions, device=x.device).unsqueeze(0) * epsilon  # Shape: (1, dimensions, dimensions)
    x_perturbed = x.unsqueeze(1) + perturbations  # Shape: (number_of_points, dimensions, dimensions)

    # Reshape x_perturbed to apply f
    x_perturbed_flat = x_perturbed.view(-1, dimensions)  # Shape: (number_of_points * dimensions, dimensions)

    # Compute f(x + epsilon * e_i) for all perturbed points
    fx_perturbed = f(x_perturbed_flat).view(num_points, dimensions, dimensions)  # Shape: (number_of_points, dimensions, dimensions)

    # Compute the Jacobian using finite differences
    J = (fx_perturbed - fx.unsqueeze(2)) / epsilon  # Shape: (number_of_points, dimensions, dimensions)

    return J


def OU(mc_times, observation_times, observations, diffusion, drift, nmc_x=100, evidence=False, EPSILON=1e-5):
    """
    OU Bridge functions
    """
    assert mc_times.max() < observation_times.max()
    time_nmc = mc_times.size(0)
    n_points = observations.size(0)
    dimensions = observations.size(1)

    diffusion_diagonal = diffusion(observations)
    drift_at_points = drift(observations)
    Gamma = jacobian_of_drift_finite_diferences(drift, observations)

    D = torch.zeros((n_points, dimensions, dimensions))
    E = torch.zeros((n_points, 2 * dimensions, 2 * dimensions))
    OI = torch.zeros((2 * dimensions, dimensions))
    OI[dimensions:, :] = torch.eye(dimensions)

    D[:, range(dimensions), range(dimensions)] = diffusion_diagonal
    E[:, :dimensions, :dimensions] = Gamma
    E[:, :dimensions:, dimensions:] = D
    E[:, dimensions:, dimensions:] = Gamma.transpose(2, 1)

    where_mc_time_index = where_mc_time(mc_times, observation_times)

    z_k = observations[where_mc_time_index]
    z_k1 = observations[where_mc_time_index + 1]

    f_k = drift_at_points[where_mc_time_index, :]
    E_k = E[where_mc_time_index, :, :]
    Gamma_k = Gamma[where_mc_time_index, :, :]
    Gamma_k_inv = safe_inverse(Gamma_k, EPSILON)
    D_k = D[where_mc_time_index, :, :]

    time_difference_k1 = observation_times[where_mc_time_index + 1, None, None] - mc_times[:, None, None]
    time_difference_k = mc_times[:, None, None] - observation_times[where_mc_time_index, None, None]
    time_difference_k = torch.clamp(time_difference_k, EPSILON)

    E_k1 = torch.matrix_exp(E_k * time_difference_k1)
    E_k1 = torch.matmul(E_k, OI)

    E_k = torch.matrix_exp(E_k * time_difference_k)
    E_k = torch.matmul(E_k, OI)

    A_s1 = E_k1[:, :dimensions, :]
    B_s1 = E_k1[:, dimensions:, :]

    A_s = E_k[:, :dimensions, :]
    B_s = E_k[:, dimensions:, :]

    S_s1 = torch.matmul(A_s1, safe_inverse(B_s1))
    S_s1_inv = safe_inverse(S_s1)

    S_s = torch.matmul(A_s, safe_inverse(B_s))
    S_s_inv = safe_inverse(S_s)

    if f_k.size(1) == 1:
        alpha_k = z_k + m(Gamma_k_inv, f_k.unsqueeze(-1))[:, 0, :]
    else:
        alpha_k = z_k + m(Gamma_k_inv, f_k.unsqueeze(-1)).squeeze()

    ME1 = torch.matrix_exp(-Gamma_k.transpose(2, 1) * time_difference_k1)
    ME2 = torch.matrix_exp(-Gamma_k * time_difference_k1)

    C_t = safe_inverse(m(ME1, m(S_s1_inv, ME2)) + S_s_inv)

    ME3 = m(C_t, m(ME1, S_s1_inv))
    a = z_k1[:, :, None] - alpha_k[:, :, None] + m(ME2, alpha_k[:, :, None])

    ME4 = m(C_t, S_s_inv)
    ME5 = torch.matrix_exp(-Gamma_k * time_difference_k)

    b = alpha_k[:, :, None] + m(ME5, (z_k[:, :, None] - alpha_k[:, :, None]))

    m_t = m(ME3, a) + m(ME4, b)
    if dimensions > 1:
        q_t = MultivariateNormal(m_t.squeeze(), C_t)
        monte_carlo_points = q_t.sample((nmc_x,))
    else:
        # m_t = torch.clamp(m_t,1e-5)
        C_t = torch.clamp(C_t, 1e-5)
        q_t = Normal(m_t.squeeze(), C_t.squeeze())
        monte_carlo_points = q_t.sample((nmc_x,)).unsqueeze(-1)

    # AVERAGE DRIFT
    nmc_x, time_nmc, _ = monte_carlo_points.shape
    monte_carlo_points_ = monte_carlo_points.reshape(nmc_x * time_nmc, -1)  # first index

    Gamma_k_ = Gamma_k.repeat((nmc_x, 1, 1))
    S_s1_inv_ = S_s1_inv.repeat((nmc_x, 1, 1))
    z_k_ = z_k.repeat((nmc_x, 1))
    z_k1_ = z_k1.repeat((nmc_x, 1))
    alpha_k_ = alpha_k.repeat((nmc_x, 1))
    f_k_ = f_k.repeat((nmc_x, 1))
    time_difference_k1_ = time_difference_k1.repeat((nmc_x, 1, 1))
    D_k_ = D_k.repeat((nmc_x, 1, 1))

    A = f_k_[:, :, None] - m(Gamma_k_, (monte_carlo_points_ - z_k_)[:, :, None])
    MEa = torch.matrix_exp(-Gamma_k_.transpose(2, 1) * time_difference_k1_)
    MEb = torch.matrix_exp(-Gamma_k_ * time_difference_k1_)
    B = m(D_k_, m(MEa, S_s1_inv_))
    C = z_k1_[:, :, None] - alpha_k_[:, :, None] - m(MEb, (monte_carlo_points_ - alpha_k_)[:, :, None])

    g_x = A + m(B, C)

    if dimensions > 1:
        A_x = q_t.log_prob(monte_carlo_points).reshape(-1)
    else:
        A_x = q_t.log_prob(monte_carlo_points.squeeze()).reshape(-1)

    return monte_carlo_points_, m_t, C_t, A_x, g_x


def maximization(x, monte_carlo_points_, inducing_points, kernels, A_x, g_x, diffusion, option_2=True, EPSILON=1e-4):
    F = []
    full_mc = monte_carlo_points_.size(0)
    dimensions = monte_carlo_points_.size(1)
    for dimension_index in range(dimensions):
        kernel = kernels[dimension_index]
        D_mc_diagonal = diffusion(monte_carlo_points_)
        D_j = D_mc_diagonal[:, dimension_index]

        # Kernels
        K_t_ind = kernel.forward(monte_carlo_points_, inducing_points)
        K_ind_ind = kernel.forward(inducing_points, inducing_points) + torch.eye(inducing_points.size(0)) * EPSILON
        K_ind_ind_inv = safe_inverse(K_ind_ind)

        if option_2:
            DA_j = (1.0 / D_j) * A_x
            DA_j = torch.diag(D_j)
            Lambda_s = m(K_t_ind.T, m(DA_j, K_t_ind)) / float(full_mc)
            Lambda_s = m(K_ind_ind_inv, m(Lambda_s, K_ind_ind_inv))
        else:
            Lambda_s = m(K_t_ind, K_t_ind.T)[range(full_mc), range(full_mc)]
            Lambda_s = Lambda_s * D_j * A_x
            Lambda_s = m(K_ind_ind_inv, K_ind_ind_inv) * Lambda_s.mean()

        b_x = A_x * g_x[:, dimension_index, 0]
        Db_j = (1.0 / D_j) * b_x
        b_integral = (K_t_ind * Db_j[:, None]).mean(axis=0)
        y_s = m(K_ind_ind_inv, b_integral[:, None])
        n_inducing = inducing_points.size(0)
        kx = kernel.forward(x, inducing_points)
        A = safe_inverse(torch.eye(n_inducing) + m(Lambda_s, K_ind_ind))
        f_x = m(m(kx, A), y_s)
        F.append(f_x)
    F = torch.cat(F, dim=1)
    return F


if __name__ == "__main__":
    nmc_x = 100
    nmc_t = 10
    num_evaluation_points = 100

    # =============================================
    # READ SDEs
    # =============================================
    # gp_path = r"C:\Users\cesar\Desktop\Projects\DiffusiveGenerativeModelling\OurCodes\markov_bridges\data\raw\gps\double_well.tr"

    gp_path = r"C:\Users\cesar\Desktop\Projects\DiffusiveGenerativeModelling\OurCodes\markov_bridges\data\raw\gps\2D.tr"
    data = torch.load(gp_path)
    dense_path_realization = data["dense_path_realization"]
    dense_path_realization = dense_path_realization[:100, :]
    dimensions = dense_path_realization.size(1)
    sigma = data["sigma"]
    dt = data["dt"]

    drift = TwoDimensionalSynDrift()
    # drift = DoubleWellDrift()
    diffusion = ConstantDiffusion(sigma)
    # =============================================
    # DEFINE SPARSE OBSERVATIONS
    # =============================================
    num_dense_steps_in_bridge = 20
    number_of_steps = dense_path_realization.size(0)
    dense_time = torch.arange(0, number_of_steps) * dt
    max_time = number_of_steps * dt
    observation_index = range(0, number_of_steps, num_dense_steps_in_bridge)
    sparse_observation_time = dense_time[observation_index]
    sparse_observations = dense_path_realization[observation_index]

    plt.plot(dense_time, dense_path_realization[:, 0], "b-")
    plt.plot(sparse_observation_time, sparse_observations[:, 0], "ro")
    plt.show()

    # =============================================
    # SPARSE GP DENSE DRIFT ESTIMATION
    # =============================================
    # where to evaluate the drift function
    ranges_ = define_grid_ranges(dense_path_realization)
    evaluation_points = define_mesh_points(total_points=num_evaluation_points, n_dims=dimensions, ranges=ranges_)
    sparse_gp_sde = SparseGPSDE(sparse_observations, dt=dt, num_inducing_points=100)
    sparse_gp_sde.inference(diffusion)
    # ========================================
    # EM STEPS
    # ========================================
    inducing_points = define_mesh_points(total_points=num_evaluation_points, n_dims=dimensions, ranges=ranges_)
    diffusion_0 = sparse_gp_sde
    mc_times = torch.rand((nmc_t,)) * sparse_observation_time.max() - dt
    monte_carlo_points_, m_t, C_t, A_x, g_x = OU(mc_times, sparse_observation_time, sparse_observations, diffusion, drift, nmc_x)

    F_x = maximization(evaluation_points, monte_carlo_points_, inducing_points, sparse_gp_sde.kernel, A_x, g_x)
    # plt.plot(F_x.shape())


# def OU(mc_times,observation_times,observations,diffusion,drift,x_nmc=100,t_nmc=100,evidence=False):
#     assert mc_times.max() < observation_times.max()
#     time_nmc = mc_times.size(0)
#     n_points = observations.size(0)
#     dimensions = observations.size(1)
#
#     diffusion_diagonal = diffusion(observations)
#     drift_at_points = drift(observations)
#     Gamma = jacobian_of_drift(drift,observations)
#
#     D = torch.zeros((n_points,dimensions,dimensions))
#     E = torch.zeros((n_points,2*dimensions,2*dimensions))
#     OI = torch.zeros((2*dimensions,dimensions))
#     OI[dimensions:,:] = torch.eye(dimensions)
#
#     D[:,range(dimensions),range(dimensions)] = diffusion_diagonal
#     E[:,:dimensions,:dimensions] = Gamma
#     E[:,:dimensions:,dimensions:] = D
#     E[:,dimensions:,dimensions:] = Gamma.transpose(2,1)
#
#     where_mc_time_index = where_mc_time(mc_times,observation_times)
#
#     z_k = observations[where_mc_time_index]
#     z_k1 = observations[where_mc_time_index+1]
#
#     f_k = drift_at_points[where_mc_time_index,:]
#     E_k = E[where_mc_time_index,:,:]
#     Gamma_k = Gamma[where_mc_time_index,:,:]
#     Gamma_k_inv = torch.inverse(Gamma_k)
#     D_k = D[where_mc_time_index,:,:]
#
#     time_difference_k1 = observation_times[where_mc_time_index+1,None,None]-mc_times[:,None,None]
#     time_difference_k = mc_times[:,None,None] - observation_times[where_mc_time_index,None,None]
#     time_difference_k = torch.clamp(time_difference_k,1e-3)
#
#     E_k1 = torch.matrix_exp(E_k*time_difference_k1)
#     E_k1 = torch.matmul(E_k,OI)
#
#     E_k = torch.matrix_exp(E_k*time_difference_k)
#     E_k = torch.matmul(E_k,OI)
#
#     A_s1 = E_k1[:,:dimensions,:]
#     B_s1 = E_k1[:,dimensions:,:]
#
#     A_s = E_k[:,:dimensions,:]
#     B_s = E_k[:,dimensions:,:]
#
#     S_s1 = torch.matmul(A_s1,torch.inverse(B_s1))
#     S_s1_inv = torch.inverse(S_s1)
#
#     S_s = torch.matmul(A_s,torch.inverse(B_s))
#     S_s_inv = torch.inverse(S_s)
#
#     if f_k.size(1) == 1:
#         alpha_k = z_k + m(Gamma_k_inv,f_k.unsqueeze(-1))[:,0,:]
#     else:
#         alpha_k = z_k + m(Gamma_k_inv,f_k.unsqueeze(-1)).squeeze()
#
#     ME1 = torch.matrix_exp(-Gamma_k.transpose(2,1)*time_difference_k1)
#     ME2 = torch.matrix_exp(-Gamma_k*time_difference_k1)
#
#     C_t = torch.inverse(m(ME1,m(S_s1_inv,ME2))+S_s_inv)
#
#     ME3 = m(C_t,m(ME1,S_s1_inv))
#     a = z_k1[:,:,None] - alpha_k[:,:,None] + m(ME2,alpha_k[:,:,None])
#
#     ME4 = m(C_t,S_s_inv)
#     ME5 = torch.matrix_exp(-Gamma_k*time_difference_k)
#
#     b = alpha_k[:,:,None] + m(ME5,(z_k[:,:,None]-alpha_k[:,:,None]))
#
#     m_t = m(ME3,a) + m(ME4,b)
#     if dimensions>1:
#         q_t = MultivariateNormal(m_t.squeeze(),C_t)
#         monte_carlo_points = q_t.sample((x_nmc,))
#     else:
#         q_t = Normal(m_t.squeeze(),C_t.squeeze())
#         monte_carlo_points = q_t.sample((x_nmc,)).unsqueeze(-1)
#
#     # AVERAGE DRIFT
#     x_nmc,time_nmc,_ = monte_carlo_points.shape
#     monte_carlo_points_ = monte_carlo_points.reshape(x_nmc*time_nmc,-1) # first index
#     full_mc = x_nmc*time_nmc
#
#     Gamma_k_ = Gamma_k.repeat((x_nmc,1,1))
#     S_s1_inv_ = S_s1_inv.repeat((x_nmc,1,1))
#     z_k_ = z_k.repeat((x_nmc,1))
#     z_k1_ = z_k1.repeat((x_nmc,1))
#     alpha_k_ = alpha_k.repeat((x_nmc,1))
#     f_k_ = f_k.repeat((x_nmc,1))
#     time_difference_k1_ = time_difference_k1.repeat((x_nmc,1,1))
#     D_k_ = D_k.repeat((x_nmc,1,1))
#
#     A = f_k_[:,:,None] - m(Gamma_k_,(monte_carlo_points_ - z_k_)[:,:,None])
#     MEa = torch.matrix_exp(-Gamma_k_.transpose(2,1)*time_difference_k1_)
#     MEb = torch.matrix_exp(-Gamma_k_*time_difference_k1_)
#     B = m(D_k_,m(MEa,S_s1_inv_))
#     C = z_k1_[:,:,None] - alpha_k_[:,:,None] - m(MEb,(monte_carlo_points_-alpha_k_)[:,:,None])
#
#     g_x = A + m(B,C)
#
#     if dimensions>1:
#         A_x = q_t.log_prob(monte_carlo_points).reshape(-1)
#     else:
#         A_x = q_t.log_prob(monte_carlo_points.squeeze()).reshape(-1)
#
#     return monte_carlo_points_,m_t,C_t,A_x,g_x
