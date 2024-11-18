import numpy as np
import click
import json
import matplotlib.pyplot as plt
import time
from pathlib import Path
import pandas as pd
import pickle
from tabulate import tabulate
import torch
from fim import models
from fim.utils.logging import RankLoggerAdapter, setup_logging
from fim.utils.helper import load_yaml
import logging



setup_logging()
logger = RankLoggerAdapter(logging.getLogger(__name__))

class Dataset:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.num_states = kwargs.get("num_states")
        self.ground_truth_num_states = kwargs.get("ground_truth_num_states", self.num_states)
        self.path = kwargs.get("path")
        self.observation_times_file = kwargs.get("observation_times_file")
        self.observation_values_file = kwargs.get("observation_values_file")
        self.time_normalization_factors_file = kwargs.get("time_normalization_factors_file")
        self.sequence_lengths_file = kwargs.get("sequence_lengths_file")
        self.ground_truth_intensity_matrices_file = kwargs.get("ground_truth_intensity_matrices_file")
        self.adjacency_matrices_file = kwargs.get("adjacency_matrices_file")
        self.ground_truth_initial_distributions_file = kwargs.get("ground_truth_initial_distributions_file")
        self.max_num_samples = kwargs.get("max_num_samples")
        self.predict_physical_quantities = kwargs.get("predict_physical_quantities", False)
        self.batch_size = kwargs.get("batch_size")

        if self.observation_times_file is None or self.observation_values_file is None or self.time_normalization_factors_file is None or self.sequence_lengths_file is None:
            raise ValueError("Missing required file!")

        with open(self.path + "/" + self.observation_times_file, "rb") as f:
            self.observation_times = pickle.load(f)
        with open(self.path + "/" + self.observation_values_file, "rb") as f:
            self.observation_values = pickle.load(f)
        with open(self.path + "/" + self.time_normalization_factors_file, "rb") as f:
            self.time_normalization_factors = pickle.load(f)
        with open(self.path + "/" + self.sequence_lengths_file, "rb") as f:
            self.sequence_lengths = pickle.load(f)
        if self.max_num_samples is not None:
            self.observation_times = self.observation_times[:self.max_num_samples]
            self.observation_values = self.observation_values[:self.max_num_samples]
            self.time_normalization_factors = self.time_normalization_factors[:self.max_num_samples]
            self.sequence_lengths = self.sequence_lengths[:self.max_num_samples]

        if self.ground_truth_intensity_matrices_file is not None:
            with open(self.path + "/" + self.ground_truth_intensity_matrices_file, "rb") as f:
                self.ground_truth_intensity_matrices = pickle.load(f)
        else:
            self.ground_truth_intensity_matrices = None

        if self.adjacency_matrices_file is not None:
            with open(self.path + "/" + self.adjacency_matrices_file, "rb") as f:
                self.adjacency_matrices = pickle.load(f)
        else:
            num_samples = self.observation_times.shape[0]
            self.adjacency_matrices = torch.Tensor(num_samples*[torch.ones((self.num_states, self.num_states))])

        if self.ground_truth_initial_distributions_file is not None:
            with open(self.path + "/" + self.ground_truth_initial_distributions_file, "rb") as f:
                self.ground_truth_initial_distributions = pickle.load(f)
        else:
            self.ground_truth_initial_distributions = None

def off_diagonal_elements_to_matrix(n, vector, set_diagonal_to_minus_row_sum=False):
    """
    Construct a matrix of size nxn from a vector of off-diagonal elements
    """
    upper_indices = torch.triu_indices(n, k=1)  # Get indices of upper triangular elements
    lower_indices = torch.tril_indices(n, k=-1)  # Get indices of lower triangular elements
    upper_values, lower_values = vector[:len(upper_indices[0])], vector[len(upper_indices[0]):]
    upper_matrix = torch.zeros((n, n))
    lower_matrix = torch.zeros((n, n))
    upper_matrix = upper_matrix.at[upper_indices].set(upper_values)
    lower_matrix = lower_matrix.at[lower_indices].set(lower_values)
    A = upper_matrix + lower_matrix
    if set_diagonal_to_minus_row_sum:
        A = A - torch.diag(torch.sum(A, axis=-1))
    return A

def get_stationary_distribution_and_relaxation_time(intensity_matrix):
    """
    Return the stationary distribution and relaxation time of the Markov jump process.
    """
    # Compute the stationary distribution by computing the left eigenvector of the intensity matrix with eigenvalue 0
    w, v = np.linalg.eig(
        intensity_matrix.T
    )  # We are working with the transpose of the intensity matrix here because we want the left eigenvector
    idx = np.where(np.isclose(w, 0, rtol=1e-03, atol=1e-05))[0][0]
    stationary_distribution = np.abs(v[:, idx] / np.sum(v[:, idx]))
    # Check if the stationary distribution is valid
    if not np.allclose(np.sum(stationary_distribution), 1, rtol=1e-03, atol=1e-05):
        logger.warning("The stationary distribution does not sum to 1: %s", stationary_distribution)
    # Compute the relaxation time by computing the inverse of the smallest non-zero eigenvalue
    idx = np.where(~np.isclose(w, 0, rtol=1e-03, atol=1e-05))[0]
    # Sort eigenvalues by the absolute value of the real part
    idx = idx[np.argsort(np.abs(np.real(w[idx])))]
    relaxation_time = 1 / np.abs(w[idx[0]])
    # Check if the relaxation time has a non-zero imaginary part which indicates that the solution is oscillating
    is_oscillating = ~np.isclose(np.imag(w[idx[0]]), 0)
    return stationary_distribution, relaxation_time, is_oscillating

def get_mean_first_passage_times(intensity_matrix):
    """
    Compute the mean first passage times as defined in https://arxiv.org/pdf/2305.19744.pdf appendix B.
    """
    num_states = intensity_matrix.shape[0]
    mean_first_passage_times = np.zeros((num_states, num_states))
    for j in range(num_states):
        A = np.delete(intensity_matrix, j, axis=0)
        A = np.delete(A, j, axis=1)
        x = np.linalg.solve(A, -np.ones(num_states - 1))
        p = 0
        for i in range(num_states):
            if i == j:
                continue
            mean_first_passage_times[i, j] = x[p]
            p += 1
    return mean_first_passage_times

def get_ordered_time_scales(intensity_matrix):
    """
    Compute the time scales as defined in https://arxiv.org/pdf/2305.19744.pdf appendix B.
    """
    w, v = np.linalg.eig(
        intensity_matrix.T
    )  # We are working with the transpose of the intensity matrix here because we want the left eigenvector
    w = np.real(w)
    idx = np.where(~np.isclose(w, 0, rtol=1e-03, atol=1e-05))[0]
    # Sort eigenvalues by the absolute value of the real part
    idx = idx[np.argsort(np.abs(np.real(w[idx])))]
    # Reverse the order of the eigenvalues
    idx = idx[::-1]
    return 1 / np.abs(w[idx])

def off_diagonal_elements_to_flattened_vector(matrix):
    """
    Extract off-diagonal elements of a matrix and flatten them into a vector.
    """
    n = matrix.shape[0]
    upper_indices = torch.triu_indices(n, k=1)  # Get indices of upper triangular elements
    lower_indices = torch.tril_indices(n, k=-1)  # Get indices of lower triangular elements
    upper_values, lower_values = matrix[upper_indices], matrix[lower_indices]
    vector = torch.concatenate([upper_values, lower_values], axis=-1)
    return vector

def off_diagonal_elements_to_flattened_vector_np(matrix):
    """
    Extract off-diagonal elements of a matrix and flatten them into a vector.
    """
    n = matrix.shape[0]
    upper_indices = np.triu_indices(n, k=1)  # Get indices of upper triangular elements
    lower_indices = np.tril_indices(n, k=-1)  # Get indices of lower triangular elements
    upper_values, lower_values = matrix[upper_indices], matrix[lower_indices]
    vector = np.concatenate([upper_values, lower_values], axis=-1)
    return vector

def intensity_matrix_to_dfr_parameters(num_states, intensity_matrix):
    """
    Compute the parameters V, r, b from the intensity matrix of the flashing ratchet model
    """
    if num_states != 6:
        raise NotImplementedError("Only 6 states are supported")
    r = (intensity_matrix[0,3] + intensity_matrix[1,4] + intensity_matrix[2,5] + intensity_matrix[3,0] + intensity_matrix[4,1] + intensity_matrix[5,2])/6
    b = (intensity_matrix[3,4] + intensity_matrix[3,5] + intensity_matrix[4,3] + intensity_matrix[4,5] + intensity_matrix[5,3] + intensity_matrix[5,4])/6
    V = (-2*np.log(intensity_matrix[0,1]) + -2*np.log(intensity_matrix[0,2])/2 + -2*np.log(intensity_matrix[1,0])/-1 + -2*np.log(intensity_matrix[1,2]) + -2*np.log(intensity_matrix[2,0])/-2 + -2*np.log(intensity_matrix[2,1])/-1)/6
    return V, r, b

def solve_homogenous_master_equation_exp(grids, p0, intensity_matrices):
    """
    Solve the master equation for the homogenous Markov jump process.
    We use the exact solution of the master equation and compute the matrix exponential.

    Input:
    grids: [num_samples, grid_length, 1],
    p0: [num_states],
    intensity_matrices: [num_samples, grid_length, num_states, num_states]
    """
    res = p0 @ torch.linalg.matrix_exp(intensity_matrices * grids[:, :, None])
    count_non_stationary = 0
    count_non_relaxed = 0

    for i, probabilities in enumerate(res):
        # Sanity checks:
        p = probabilities[-1]  # The last element of the solution is the solution at the end time
        intensity_matrix = intensity_matrices[i][0]  # !!! This currently only works for the homogenous case!
        # 1) Check if the elements of the solution sum to 1
        if not np.allclose(np.sum(p), 1, rtol=1e-03, atol=1e-05):
            logger.warning("The elements of the solution do not sum to 1: %s %s", p, intensity_matrix)
        stationary_distribution, relaxation_time, _ = get_stationary_distribution_and_relaxation_time(intensity_matrix)
        # 2) Check if the probabilities have reached the stationary distribution
        if not np.allclose(p, stationary_distribution, rtol=1e-01, atol=1e-03):
            # logger.warning("The probabilities have not reached the stationary distribution: %s %s", p, stationary_distribution)
            count_non_stationary += 1
        # 3) Check if the time is long enough
        if grids[i, -1] < relaxation_time:
            # logger.warning("The time is not long enough: %s %s", grids[i, -1], relaxation_time)
            count_non_relaxed += 1
    if count_non_stationary > 0:
        logger.warning("The probabilities have not reached the stationary distribution in %s cases", count_non_stationary)
    if count_non_relaxed > 0:
        logger.warning("The time is shorter than the relaxation time in %s cases", count_non_relaxed)

    return res

def get_solutions_of_master_eq(initial_distribution, intensity_matrix, end_time=5, grid_size=100):
    grid = np.linspace(0, end_time, grid_size).reshape(1,grid_size,1)
    intensity_matrix_grid = np.array([intensity_matrix for _ in range(grid_size)]).reshape(1,grid_size,6,6)
    return solve_homogenous_master_equation_exp(grid, initial_distribution, intensity_matrix_grid)

def compute_entropy_production(initial_distribution, intensity_matrix, end_time=5, grid_size=100):
    """
    Compute the entropy production according to https://iopscience.iop.org/article/10.1088/0034-4885/75/12/126001 eq. 128
    We return the cumulative entropy production.

    INPUTS:
    initial_distribution: [num_states]
    intensity_matrix: [num_states, num_states]
    """
    p_t = get_solutions_of_master_eq(initial_distribution, intensity_matrix, end_time, grid_size)[0]
    
    intensity_matrix = np.array(intensity_matrix)
    # Set diagonal of intensity matrix to zero
    np.fill_diagonal(intensity_matrix, 0)
    
    # Expand dimensions to match p_t
    expanded_intensity_matrix = intensity_matrix[None, :, :]
    
    # Compute the products for each pair and avoid division by zero
    p_n_m = p_t[:, :, None] * expanded_intensity_matrix
    p_m_n = p_t[:, None, :] * np.transpose(expanded_intensity_matrix, (0, 2, 1))
    
    # Compute log ratios, safely handling zero entries
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.log(p_n_m / p_m_n)
        log_ratio[np.isnan(log_ratio)] = 0  # Set NaNs resulting from 0/0 to 0
    
    # Compute entropy production for each time step
    entropy = np.sum(p_t[:, :, None] * expanded_intensity_matrix * log_ratio, axis=(1, 2))
    
    # Return the average entropy production
    return np.mean(entropy)

def compute_ground_truth_dfr_intensity_matrix(num_states, T, V, r, b):
    """
    Compute the ground truth intensity matrix for the flashing ratchet model, see https://arxiv.org/pdf/2305.19744.pdf appendix J.1
    """
    beta = 1 / T
    A = np.zeros((num_states, num_states))
    if num_states == 3:
        for i in range(num_states):
            for j in range(num_states):
                if i != j:
                    A[i, j] = np.exp(-beta * V * (j - i) / 2)
                    A[i, j] += b
        A /= 2  # Since we are averaging
    elif num_states == 6:
        rates_changing_potential = r * np.eye(3)
        rates_when_potential_off = b * (np.ones(3) - np.eye(3))

        V_array = np.array([0 * V, 1 * V, 2 * V])
        V_j = V_array[None, :]
        V_i = V_array[:, None]
        rates_when_potential_on = np.exp(-(V_j - V_i) / 2)
        # fill diagonal with zeros
        np.fill_diagonal(rates_when_potential_on, 0)

        A = np.concatenate(
            [
                np.concatenate([rates_when_potential_on, rates_changing_potential], axis=1),
                np.concatenate([rates_changing_potential, rates_when_potential_off], axis=1),
            ],
            axis=0,
        )
    else:
        raise NotImplementedError("Only 3 and 6 states are supported")
    for i in range(num_states):
        A[i, i] = -(np.sum(A[i, :]) - A[i, i])
    return A



class Model:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.path = kwargs.get("path")
        self.num_states = kwargs.get("num_states")
        self.evaluation_path_count = kwargs.get("evaluation_path_count")
        self.model_dict, self.state = models.FIMMJP.load_model(self.path)

    def __call__(self, dataset):
        """
        Evaluate the model on the dataset. If evaluation_path_count is None, evaluate on all paths. Otherwise, split the paths into batches of evaluation_path_count paths and return evaluation over all batches.

        Output shape:
        intensity_matrices: (num_samples, num_batches, num_states, num_states)
        confidences: (num_samples, num_batches, num_states, num_states)
        initial_distributions: (num_samples, num_batches, num_states)
        """
        num_samples = dataset.observation_times.shape[0]
        intensity_matrices = []
        confidences = []
        initial_distributions = []
        if self.evaluation_path_count is None:
            evaluation_path_count = dataset.observation_times.shape[1]
        else:
            evaluation_path_count = self.evaluation_path_count
        if evaluation_path_count >= dataset.observation_times.shape[1]:
            # We can evaluate on the full dataset
            if dataset.batch_size is None:
                dataset.batch_size = num_samples
            for i in range(0, num_samples, dataset.batch_size):
                observation_times = dataset.observation_times[i:i+dataset.batch_size]
                observation_values = dataset.observation_values[i:i+dataset.batch_size]
                sequence_lengths = dataset.sequence_lengths[i:i+dataset.batch_size]
                time_normalization_factors = dataset.time_normalization_factors[i:i+dataset.batch_size]
                adjacency_matrices = dataset.adjacency_matrices[i:i+dataset.batch_size]
                call_input = (observation_times, observation_values, sequence_lengths, time_normalization_factors, adjacency_matrices)
                train =  False 
                model_output, updated_states = self.model_dict["vmap_model_call"](self.state.variables, train, *call_input)
                embeddings, means, log_std, initial_distribution = model_output
                confidence = torch.exp(log_std)
                for i in range(observation_times.shape[0]):
                    intensity_matrix = off_diagonal_elements_to_matrix(self.num_states, means[i], set_diagonal_to_minus_row_sum=True)
                    intensity_matrices.append(intensity_matrix)
                    confidence_matrix = off_diagonal_elements_to_matrix(self.num_states, confidence[i], set_diagonal_to_minus_row_sum=False)
                    confidences.append(confidence_matrix)
                    initial_distributions.append(initial_distribution[i])
            intensity_matrices = torch.Tensor(intensity_matrices).reshape(len(intensity_matrices), 1, self.num_states, self.num_states)
            confidences = torch.Tensor(confidences).reshape(len(confidences), 1, self.num_states, self.num_states)
            initial_distributions = torch.Tensor(initial_distributions).reshape(len(initial_distributions), 1, self.num_states)
        else:
            # We now split the paths into different batches with evaluation_path_count paths in each batch
            num_batches_per_sample = dataset.observation_times.shape[1] // evaluation_path_count
            for i in range(num_samples):
                observation_times = dataset.observation_times[i, :num_batches_per_sample*evaluation_path_count].reshape(num_batches_per_sample, evaluation_path_count, dataset.observation_times.shape[2], dataset.observation_times.shape[3])
                observation_values = dataset.observation_values[i, :num_batches_per_sample*evaluation_path_count].reshape(num_batches_per_sample, evaluation_path_count, dataset.observation_times.shape[2], dataset.observation_times.shape[3])
                sequence_lengths = dataset.sequence_lengths[i, :num_batches_per_sample*evaluation_path_count].reshape(num_batches_per_sample, evaluation_path_count, 1)
                time_normalization_factors = torch.Tensor([dataset.time_normalization_factors[i]] * num_batches_per_sample)
                adjacency_matrices = torch.Tensor([dataset.adjacency_matrices[i]] * num_batches_per_sample)
                call_input = (observation_times, observation_values, sequence_lengths, time_normalization_factors, adjacency_matrices)
                train = False
                model_output, updated_states = self.model_dict["vmap_model_call"](self.state.variables, train, *call_input)
                embeddings, means, log_std, initial_distribution = model_output
                confidence = torch.exp(log_std)
                initial_distributions.append(initial_distribution)
                intensity_matrix_batch = [off_diagonal_elements_to_matrix(self.num_states, means[i], set_diagonal_to_minus_row_sum=True) for i in range(num_batches_per_sample)]
                intensity_matrices.append(intensity_matrix_batch)
                confidence_matrix_batch = [off_diagonal_elements_to_matrix(self.num_states, confidence[i], set_diagonal_to_minus_row_sum=False) for i in range(num_batches_per_sample)]
                confidences.append(confidence_matrix_batch)
            intensity_matrices = torch.Tensor(intensity_matrices)
            confidences = torch.Tensor(confidences)
            initial_distributions = torch.Tensor(initial_distributions)
            # Assert that the shapes are correct
            assert intensity_matrices.shape==(num_samples, num_batches_per_sample, self.num_states, self.num_states)
            assert confidences.shape==(num_samples, num_batches_per_sample, self.num_states, self.num_states)
            assert initial_distributions==(num_samples, num_batches_per_sample, self.num_states)
        return intensity_matrices, confidences, initial_distributions
    
def predict_physical_quantities(intensity_matrices, confidences, initial_distributions, ground_truth_num_states):
    # Compute the batch-wise means
    mean_intensity_matrices = torch.mean(intensity_matrices, axis=1)
    mean_confidences = torch.mean(confidences, axis=1)

    # Only select the valid states
    mean_intensity_matrices = mean_intensity_matrices[:, :ground_truth_num_states, :ground_truth_num_states]
    mean_confidences = mean_confidences[:, :ground_truth_num_states, :ground_truth_num_states]

    # set the diagonal to minus row sum
    for i in range(mean_intensity_matrices.shape[0]):
        for j in range(ground_truth_num_states):
            mean_intensity_matrices = mean_intensity_matrices.at[i, j, j].set(-(torch.sum(mean_intensity_matrices[i,j])-mean_intensity_matrices[i,j,j]))

    stationary_distributions, relaxation_times, mean_first_passage_times, ordered_time_scales = [], [], [], []
    for intensity_matrix in mean_intensity_matrices:
        stationary_distribution, relaxation_time, is_oscillating = get_stationary_distribution_and_relaxation_time(intensity_matrix)
        mean_first_passage_time = get_mean_first_passage_times(intensity_matrix)
        ordered_time_scale = get_ordered_time_scales(intensity_matrix)
        stationary_distributions.append(stationary_distribution)
        relaxation_times.append(relaxation_time)
        mean_first_passage_times.append(mean_first_passage_time)
        ordered_time_scales.append(ordered_time_scale)

    data = {
        "intensity_matrices": mean_intensity_matrices,
        "confidences": mean_confidences,
        "stationary_distributions": stationary_distributions,
        "relaxation_times": relaxation_times,
        "mean_first_passage_times": mean_first_passage_times,
        "ordered_time_scales": ordered_time_scales,
    }

    if initial_distributions is not None:
        initial_distributions = torch.mean(initial_distributions, axis=1)
        initial_distributions = initial_distributions[:, :ground_truth_num_states]
        data["initial_distributions"] = initial_distributions

    return data

def compute_rmse(off_diagonal_ground_truth, off_diagonal_prediction, num_states, ground_truth_num_states=None):
    """
    Compute the rmse of the off-diagonal elements of the intensity matrix. We only consider the off-diagonal elements that correspond to the intensity rates of a ground_truth_num_states-state system.
    If there are several batches, we compute the mean and std of the rmse over all batches.
    Input:
    off_diagonal_ground_truth: [num_states*num_states - num_states]
    off_diagonal_prediction: [num_batches, num_states*num_states - num_states]
    """
    if ground_truth_num_states is not None:
        # Only select the elements that correspond to intensity rates of a ground_truth_num_states-state system
        valid_entries = np.zeros((num_states,num_states), dtype=np.bool_)
        valid_entries[:ground_truth_num_states, :ground_truth_num_states] = True
        valid_entries = np.array(valid_entries)
        valid_entries = off_diagonal_elements_to_flattened_vector_np(valid_entries)
    else:
        valid_entries = np.ones_like(off_diagonal_ground_truth, dtype=np.bool_)
    # Apply mask to select valid entries
    off_diagonal_ground_truth = off_diagonal_ground_truth[valid_entries]
    off_diagonal_prediction = off_diagonal_prediction[:, valid_entries]

    # First compute all the rmses to get the std
    # Stack the off-diagonal elements of the intensity matrices to get the same shape as the prediction
    off_diagonal_ground_truth = torch.tile(off_diagonal_ground_truth, (off_diagonal_prediction.shape[0], 1))

    # Compute the error
    error = off_diagonal_ground_truth - off_diagonal_prediction

    # Compute the rmse
    mse = torch.mean(error**2, axis=-1)
    rmse = torch.sqrt(mse)

    # Compute the std over all batches
    rmse_std = torch.std(rmse, axis=0)

    # Now compute the rmse of the average over all batches
    off_diagonal_prediction_mean = torch.mean(off_diagonal_prediction, axis=0)
    error_mean = off_diagonal_ground_truth[0] - off_diagonal_prediction_mean
    rmse_mean = torch.sqrt(torch.mean(error_mean**2))

    return rmse_mean, rmse_std

def get_mean_confidence(confidences, num_states, ground_truth_num_states=None):
    """
    Input:
    confidences: [num_samples, num_batches, num_states, num_states]
    """
    # Compute off-diagonal confidences which has shape [num_samples, num_batches, num_states*num_states - num_states]
    off_diagonal_confidences = []
    for sample_idx in range(confidences.shape[0]):
        tmp = []
        for confidence_matrix in confidences[sample_idx]:
            tmp.append(off_diagonal_elements_to_flattened_vector(confidence_matrix))
        off_diagonal_confidences.append(tmp)
    off_diagonal_confidences = np.array(off_diagonal_confidences)

    # Select the valid entries
    if ground_truth_num_states is not None:
        valid_entries = np.zeros((num_states,num_states), dtype=np.bool_)
        valid_entries[:ground_truth_num_states, :ground_truth_num_states] = True
        valid_entries = np.array(valid_entries)
        valid_entries = off_diagonal_elements_to_flattened_vector_np(valid_entries)
    else:
        valid_entries = np.ones_like(off_diagonal_confidences[0][0], dtype=np.bool_)
    off_diagonal_confidences = off_diagonal_confidences[:, :, valid_entries]

    # First compute the mean entry-wise
    mean_confidences = np.mean(off_diagonal_confidences, axis=2)

    # Now compute the mean over all batches
    mean_confidences = np.mean(mean_confidences, axis=1)

    # Now compute the mean over all samples
    mean_confidence = np.mean(mean_confidences, axis=0)

    return mean_confidence

def compute_performance_metrics(dataset, ground_truth_intensity_matrices, intensity_matrices, confidences, initial_distributions, consider_initial_distributions=False):
    # Compute off-diagonal intensities which has shape [num_samples, num_batches, num_states*num_states - num_states]
    off_diagonal_intensities = []
    for sample_idx in range(intensity_matrices.shape[0]):
        tmp = []
        for intensity_matrix in intensity_matrices[sample_idx]:
            tmp.append(off_diagonal_elements_to_flattened_vector(intensity_matrix))
        off_diagonal_intensities.append(tmp)
    off_diagonal_intensities = np.array(off_diagonal_intensities)
    off_diagonal_ground_truths = [off_diagonal_elements_to_flattened_vector(intensity_matrix) for intensity_matrix in ground_truth_intensity_matrices]
    off_diagonal_ground_truths = np.array(off_diagonal_ground_truths)
    rmse_means = []
    rmse_stds = []
    for sample_idx in range(intensity_matrices.shape[0]):
        rmse_mean, rmse_std = compute_rmse(off_diagonal_ground_truths[sample_idx], off_diagonal_intensities[sample_idx], dataset.num_states, dataset.ground_truth_num_states)
        rmse_means.append(rmse_mean)
        rmse_stds.append(rmse_std)
    intensity_mean_rmse = np.mean(rmse_means)
    intensity_std_rmse = np.mean(rmse_stds)

    # Now get the mean confidence
    mean_confidence = get_mean_confidence(confidences, dataset.num_states, dataset.ground_truth_num_states)

    if consider_initial_distributions:
        if initial_distributions is None or dataset.ground_truth_initial_distributions is None:
            initial_distribution_mean_rmse = None
        else:
            # Compute the mean initial_distribution over the batches
            initial_distributions = np.mean(initial_distributions, axis=1)

            initial_distribution_rmses = []
            for i in range(len(initial_distributions)):
                initial_distribution_rmses.append(torch.sqrt(torch.mean(torch.square(initial_distributions[i] - dataset.ground_truth_initial_distributions[i]))))
            initial_distribution_rmses = torch.Tensor(initial_distribution_rmses)
            initial_distribution_mean_rmse = torch.mean(initial_distribution_rmses)
        return intensity_mean_rmse, intensity_std_rmse, mean_confidence, initial_distribution_mean_rmse
    return intensity_mean_rmse, intensity_std_rmse, mean_confidence

class Benchmark:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.dataset_configs = kwargs.get("dataset")
        self.model_configs = kwargs.get("model")
        self.datasets = [Dataset(**config) for config in self.dataset_configs]
        self.intensity_rate_metrics = kwargs.get("intensity_rate_metrics", [])

    def __call__(self):
        data = {}
        for dataset in self.datasets:
            data[dataset.name] = {}
            if dataset.ground_truth_intensity_matrices is not None and dataset.predict_physical_quantities:
                # Store the ground truth physical quantities if it is available
                ground_truth_intensity_matrices = np.array(dataset.ground_truth_intensity_matrices)
                intensity_matrices = ground_truth_intensity_matrices.reshape(ground_truth_intensity_matrices.shape[0], 1, ground_truth_intensity_matrices.shape[1], ground_truth_intensity_matrices.shape[2])
                confidences = np.zeros_like(intensity_matrices)
                if dataset.ground_truth_initial_distributions is not None:
                    ground_truth_initial_distributions = np.array(dataset.ground_truth_initial_distributions)
                    initial_distributions = ground_truth_initial_distributions.reshape(ground_truth_initial_distributions.shape[0], 1, ground_truth_initial_distributions.shape[1])
                else:
                    initial_distributions = None
                data[dataset.name]["ground_truth"] = predict_physical_quantities(intensity_matrices, confidences, initial_distributions, dataset.ground_truth_num_states)
        for model_config in self.model_configs:
            model = Model(**model_config)
            for dataset in self.datasets:
                intensity_matrices, confidences, initial_distributions = model(dataset)
                if dataset.ground_truth_intensity_matrices is not None:
                    intensity_mean_rmse, intensity_std_rmse, mean_confidence, initial_distribution_mean_rmse = compute_performance_metrics(dataset, dataset.ground_truth_intensity_matrices, intensity_matrices, confidences, initial_distributions, consider_initial_distributions=True)
                    data[dataset.name][model.name] = {"intensity_rate_metrics": {"rmse": {"mean": intensity_mean_rmse, "std": intensity_std_rmse}}, "mean_confidence": mean_confidence}
                    if initial_distribution_mean_rmse is not None:
                        data[dataset.name][model.name]["initial_distribution_metrics"] = {"rmse": {"mean": initial_distribution_mean_rmse}}
                else:
                    data[dataset.name][model.name] = {}
                if dataset.predict_physical_quantities:
                    data[dataset.name][model.name].update(predict_physical_quantities(intensity_matrices, confidences, initial_distributions, dataset.ground_truth_num_states))
                if "discrete_flashing_ratchet" in dataset.name or "DFR" in dataset.name:
                    if model.name != "ground_truth":
                        V_values, r_values, b_values = [], [], []
                        S = []
                        for i in range(intensity_matrices[0].shape[0]):
                            intensity_matrix = intensity_matrices[0][i]
                            initial_distribution = initial_distributions[0][i]
                            V, r, b = intensity_matrix_to_dfr_parameters(6, intensity_matrix)
                            V_values.append(V)
                            r_values.append(r)
                            b_values.append(b)
                            S.append(compute_entropy_production(initial_distribution, intensity_matrix))
                        data[dataset.name][model.name]["dfr_parameters"] = {"mean_V": np.mean(V_values), "mean_r": np.mean(r_values), "mean_b": np.mean(b_values), "std_V": np.std(V_values), "std_r": np.std(r_values), "std_b": np.std(b_values), "mean_entropy_production": np.mean(S), "std_entropy_production": np.std(S)}
            del model
        return data
    

def get_intensity_matrix_rmse(predicted_intensity_matrix, ground_truth_intensity_matrix):
    predicted_off_diagonal = off_diagonal_elements_to_flattened_vector(predicted_intensity_matrix)
    ground_truth_off_diagonal = off_diagonal_elements_to_flattened_vector(ground_truth_intensity_matrix)
    rmse = torch.sqrt(np.mean((predicted_off_diagonal - ground_truth_off_diagonal)**2))
    return rmse

def print_quantity(data, dataset, quantity):
    print(f"Dataset: {dataset}")
    for model in data[dataset]:
        print(f"Model: {model}")
        print(f"{quantity}: {data[dataset][model][quantity]}")
    print("")

def print_rates_rmse(data, file=None):
    if file is None:
        for dataset in data:
            has_model_with_intensity_rate_metrics = False
            for model in data[dataset]:
                if "intensity_rate_metrics" in data[dataset][model]:
                    has_model_with_intensity_rate_metrics = True
                    break
            if has_model_with_intensity_rate_metrics:
                df = pd.DataFrame(columns=["Model", "RMSE"])
                for model in data[dataset]:
                    if "intensity_rate_metrics" in data[dataset][model]:
                        df = pd.concat([df, pd.DataFrame({"Model": model, "RMSE": str(data[dataset][model]['intensity_rate_metrics']['rmse']["mean"]) + " +- " + str(data[dataset][model]['intensity_rate_metrics']['rmse']["std"]), "Confidence": data[dataset][model]['mean_confidence']}, index=[0])], ignore_index=True)
                print("")
                print("Dataset:", dataset)
                print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
                print("")
    else:
        with open(file, "w") as f:
            for dataset in data:
                has_model_with_intensity_rate_metrics = False
                for model in data[dataset]:
                    if "intensity_rate_metrics" in data[dataset][model]:
                        has_model_with_intensity_rate_metrics = True
                        break
                if has_model_with_intensity_rate_metrics:
                    f.write(f"Dataset: {dataset}\n")
                    df = pd.DataFrame(columns=["Model", "RMSE"])
                    for model in data[dataset]:
                        if "intensity_rate_metrics" in data[dataset][model]:
                            df = pd.concat([df, pd.DataFrame({"Model": model, "RMSE": str(data[dataset][model]['intensity_rate_metrics']['rmse']["mean"]) + " +- " + str(data[dataset][model]['intensity_rate_metrics']['rmse']["std"]), "Confidence": data[dataset][model]['mean_confidence']}, index=[0])], ignore_index=True)
                    f.write(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
                    f.write("\n\n")

            # Now do the same to save the latex tables
            for dataset in data:
                has_model_with_intensity_rate_metrics = False
                for model in data[dataset]:
                    if "intensity_rate_metrics" in data[dataset][model]:
                        has_model_with_intensity_rate_metrics = True
                        break
                if has_model_with_intensity_rate_metrics:
                    f.write(f"Dataset: {dataset}\n")
                    df = pd.DataFrame(columns=["Model", "RMSE"])
                    for model in data[dataset]:
                        if "intensity_rate_metrics" in data[dataset][model]:
                            df = pd.concat([df, pd.DataFrame({"Model": model, "RMSE": str(data[dataset][model]['intensity_rate_metrics']['rmse']["mean"]) + " +- " + str(data[dataset][model]['intensity_rate_metrics']['rmse']["std"]), "Confidence": data[dataset][model]['mean_confidence']}, index=[0])], ignore_index=True)
                    f.write(tabulate(df, headers='keys', tablefmt='latex_raw', showindex=False))
                    f.write("\n\n")

def compute_final_multi_state_score(data):
    """
    Compute the final score by using the weighted average of the RMSE on the synthetic test sets with the same weighting as the training distribution.
    We assume that 5k samples were used for states 2-5 and 25k for state 6.
    """
    try:
        rmse_mean_res = 0
        rmse_std_res = 0
        # Get the model name by the keys
        model_names = list(data[list(data.keys())[0]].keys())

        rmse_mean_results = []
        rmse_std_results = []
        for model_name in model_names:
            rmse_mean_res = 0
            rmse_std_res = 0
            for i in range(2,6):
                rmse_mean_res += data[f"{i}_st_10s_1%_noise_rand_300-samples-per-intensity"][model_name]["intensity_rate_metrics"]["rmse"]["mean"] * 5
                rmse_std_res += data[f"{i}_st_10s_1%_noise_rand_300-samples-per-intensity"][model_name]["intensity_rate_metrics"]["rmse"]["std"] * 5
            rmse_mean_res += data["6_st_10s_1%_noise_rand_300-samples-per-intensity"][model_name]["intensity_rate_metrics"]["rmse"]["mean"] * 25
            rmse_std_res += data["6_st_10s_1%_noise_rand_300-samples-per-intensity"][model_name]["intensity_rate_metrics"]["rmse"]["std"] * 25
            rmse_mean_res /= 45 
            rmse_std_res /= 45
            rmse_mean_results.append(rmse_mean_res)
            rmse_std_results.append(rmse_std_res)
        return model_names, rmse_mean_results, rmse_std_results
    except:
        print("Error in computing the final score.")
        return None

def compare_to_other_papers(num_states, data, file):
    """
    Compare all metrics for which we have results from other papers into a file.
    """
    np.set_printoptions(suppress=True)
    if "Toy_Protein_Folding" in data:
        with open(file, "a") as f:
            f.write("Toy Protein Folding\n")
            f.write("\n")
            f.write("   Intensity Matrix\n")
            f.write("   NeuralMJP: [[-0.028, 0.028], [0.085, -0.085]]\n")
            for model in data["Toy_Protein_Folding"]:
                if "intensity_matrices" in data["Toy_Protein_Folding"][model]:
                    np.set_printoptions(precision=3)
                    intensity_matrix = np.array(data["Toy_Protein_Folding"][model]["intensity_matrices"][0])
                    f.write(f"  {model}: {intensity_matrix}\n")
            f.write("\n")
            f.write("   Stationary Distribution:\n")
            f.write("   Mardt_et_al: [0.73, 0.27]\n")
            f.write("   NeuralMJP: [0.74, 0.26]\n")
            for model in data["Toy_Protein_Folding"]:
                    np.set_printoptions(precision=2)
                    stationary_distribution = np.array(data["Toy_Protein_Folding"][model]["stationary_distributions"][0])
                    f.write(f"  {model}: {stationary_distribution}\n")
            f.write("\n")

    if "Toy_Protein_Folding_neural_mjp_labels" in data:
        with open(file, "a") as f:
            f.write("Toy Protein Folding NeuralMJP Labels\n")
            f.write("\n")
            f.write("   Intensity Matrix\n")
            f.write("   NeuralMJP: [[-0.028, 0.028], [0.085, -0.085]]\n")
            for model in data["Toy_Protein_Folding_neural_mjp_labels"]:
                if "intensity_matrices" in data["Toy_Protein_Folding_neural_mjp_labels"][model]:
                    np.set_printoptions(precision=3)
                    intensity_matrix = np.array(data["Toy_Protein_Folding_neural_mjp_labels"][model]["intensity_matrices"][0])
                    f.write(f"  {model}: {intensity_matrix}\n")
            f.write("\n")
            f.write("   Stationary Distribution:\n")
            f.write("   Mardt_et_al: [0.73, 0.27]\n")
            f.write("   NeuralMJP: [0.74, 0.26]\n")
            for model in data["Toy_Protein_Folding_neural_mjp_labels"]:
                    np.set_printoptions(precision=2)
                    stationary_distribution = np.array(data["Toy_Protein_Folding_neural_mjp_labels"][model]["stationary_distributions"][0])
                    f.write(f"  {model}: {stationary_distribution}\n")
            f.write("\n")

    if "Two_Mode_Switching_System" in data:
        with open(file, "a") as f:
            f.write("Two Mode Switching System\n")
            f.write("\n")
            np.set_printoptions(precision=2)
            ground_truth_intensity_matrix = data["Two_Mode_Switching_System"]["ground_truth"]["intensity_matrices"][0]
            f.write(f"  Intensity Matrix\n")
            koehs_et_al_intensity_matrix = np.array([[-0.64, 0.64], [0.63, -0.63]])
            f.write(f"  Koehs_et_al: {koehs_et_al_intensity_matrix}\n")
            neural_mjp_intensity_matrix = np.array([[-0.19, 0.19], [0.36, -0.36]])
            f.write(f"  NeuralMJP: {neural_mjp_intensity_matrix}\n")
            for model in data["Two_Mode_Switching_System"]:
                if "intensity_matrices" in data["Two_Mode_Switching_System"][model]:
                    intensity_matrix = np.array(data["Two_Mode_Switching_System"][model]["intensity_matrices"][0])
                    f.write(f"  {model}: {intensity_matrix}\n")
            f.write("\n")
            f.write("   Intensity Rate RMSE\n")
            koehs_et_al_rmse = get_intensity_matrix_rmse(koehs_et_al_intensity_matrix, ground_truth_intensity_matrix)
            f.write(f"  Koehs_et_al: {koehs_et_al_rmse}\n")
            neural_mjp_rmse = get_intensity_matrix_rmse(neural_mjp_intensity_matrix, ground_truth_intensity_matrix)
            f.write(f"  NeuralMJP: {neural_mjp_rmse}\n")
            for model in data["Two_Mode_Switching_System"]:
                if "intensity_rate_metrics" in data["Two_Mode_Switching_System"][model]:
                    rmse = data["Two_Mode_Switching_System"][model]["intensity_rate_metrics"]["rmse"]
                    f.write(f"  {model}: {rmse}\n")
            f.write("\n")
    
    if "ion_channel_1s" in data:
        with open(file, "a") as f:
            f.write("Ion Channel 1s\n")
            f.write("\n")
            np.set_printoptions(precision=3)
            f.write("   Stationary Distributions\n")
            f.write("   Koehs_et_al: [0.180, 0.150, 0.671]\n")
            f.write("   NeuralMJP: [0.177, 0.095, 0.739]\n")
            for model in data["ion_channel_1s"]:
                if "stationary_distributions" in data["ion_channel_1s"][model]:
                    stationary_distribution = np.array(data["ion_channel_1s"][model]["stationary_distributions"][0])
                    f.write(f"  {model}: {stationary_distribution}\n")
            f.write("\n")
            f.write("   Mean first passage times\n")
            koehs_et_al_mfpt = np.array([[0, 0.068, 0.054], [0.133, 0, 0.033], [0.181, 0.092, 0]])
            f.write(f"  Koehs_et_al: {koehs_et_al_mfpt}\n")
            neural_mjp_mfpt = np.array([[0, 0.019, 0.031], [0.083, 0, 0.014], [0.119, 0.018, 0]])
            f.write(f"  NeuralMJP: {neural_mjp_mfpt}\n")
            for model in data["ion_channel_1s"]:
                if "mean_first_passage_times" in data["ion_channel_1s"][model]:
                    mfpt = np.array(data["ion_channel_1s"][model]["mean_first_passage_times"][0])
                    f.write(f"  {model}: {mfpt}\n")
            f.write("\n")

    if "ion_channel_1s_neural_mjp_labels" in data:
        with open(file, "a") as f:
            f.write("Ion Channel 1s NeuralMJP Labels\n")
            f.write("\n")
            np.set_printoptions(precision=3)
            f.write("   Stationary Distributions\n")
            f.write("   Koehs_et_al: [0.180, 0.150, 0.671]\n")
            f.write("   NeuralMJP: [0.177, 0.095, 0.739]\n")
            for model in data["ion_channel_1s_neural_mjp_labels"]:
                if "stationary_distributions" in data["ion_channel_1s_neural_mjp_labels"][model]:
                    stationary_distribution = np.array(data["ion_channel_1s_neural_mjp_labels"][model]["stationary_distributions"][0])
                    f.write(f"  {model}: {stationary_distribution}\n")
            f.write("\n")
            f.write("   Mean first passage times\n")
            koehs_et_al_mfpt = np.array([[0, 0.068, 0.054], [0.133, 0, 0.033], [0.181, 0.092, 0]])
            f.write(f"  Koehs_et_al: {koehs_et_al_mfpt}\n")
            neural_mjp_mfpt = np.array([[0, 0.019, 0.031], [0.083, 0, 0.014], [0.119, 0.018, 0]])
            f.write(f"  NeuralMJP: {neural_mjp_mfpt}\n")
            for model in data["ion_channel_1s_neural_mjp_labels"]:
                if "mean_first_passage_times" in data["ion_channel_1s_neural_mjp_labels"][model]:
                    mfpt = np.array(data["ion_channel_1s_neural_mjp_labels"][model]["mean_first_passage_times"][0])
                    f.write(f"  {model}: {mfpt}\n")
            f.write("\n")

    if "discrete_flashing_ratchet_shared_neural_mjp" in data:
        with open(file, "a") as f:
            f.write("Discrete Flashing Ratchet Shared\n")
            f.write("\n")
            np.set_printoptions(precision=2)
            ground_truth_intensity_matrix = data["discrete_flashing_ratchet_shared_neural_mjp"]["ground_truth"]["intensity_matrices"][0]
            f.write(f"  Intensity Matrix\n")
            neural_mjp_intensity_matrix = compute_ground_truth_dfr_intensity_matrix(6, 1, 0.95, 1.18, 1.14)
            f.write(f"  NeuralMJP: {neural_mjp_intensity_matrix}\n")
            for model in data["discrete_flashing_ratchet_shared_neural_mjp"]:
                if "intensity_matrices" in data["discrete_flashing_ratchet_shared_neural_mjp"][model]:
                    intensity_matrix = np.array(data["discrete_flashing_ratchet_shared_neural_mjp"][model]["intensity_matrices"][0])
                    f.write(f"  {model}: {intensity_matrix}\n")
            f.write("\n")
            f.write("   Intensity Rate RMSE\n")
            neural_mjp_rmse = get_intensity_matrix_rmse(neural_mjp_intensity_matrix, ground_truth_intensity_matrix)
            f.write(f"  NeuralMJP: {neural_mjp_rmse}\n")
            for model in data["discrete_flashing_ratchet_shared_neural_mjp"]:
                if "intensity_rate_metrics" in data["discrete_flashing_ratchet_shared_neural_mjp"][model]:
                    rmse = data["discrete_flashing_ratchet_shared_neural_mjp"][model]["intensity_rate_metrics"]["rmse"]
                    f.write(f"  {model}: {rmse}\n")
            f.write("\n")

    if "discrete_flashing_ratchet_regular_neural_mjp" in data:
        with open(file, "a") as f:
            f.write("Discrete Flashing Ratchet Regular\n")
            f.write("\n")
            np.set_printoptions(precision=2)
            ground_truth_intensity_matrix = data["discrete_flashing_ratchet_regular_neural_mjp"]["ground_truth"]["intensity_matrices"][0]
            f.write(f"  Intensity Matrix\n")
            neural_mjp_intensity_matrix = compute_ground_truth_dfr_intensity_matrix(6, 1, 1, 1.37, 1.36)
            f.write(f"  NeuralMJP: {neural_mjp_intensity_matrix}\n")
            for model in data["discrete_flashing_ratchet_regular_neural_mjp"]:
                if "intensity_matrices" in data["discrete_flashing_ratchet_regular_neural_mjp"][model]:
                    intensity_matrix = np.array(data["discrete_flashing_ratchet_regular_neural_mjp"][model]["intensity_matrices"][0])
                    f.write(f"  {model}: {intensity_matrix}\n")
            f.write("\n")
            f.write("   Intensity Rate RMSE\n")
            neural_mjp_rmse = get_intensity_matrix_rmse(neural_mjp_intensity_matrix, ground_truth_intensity_matrix)
            f.write(f"  NeuralMJP: {neural_mjp_rmse}\n")
            for model in data["discrete_flashing_ratchet_regular_neural_mjp"]:
                if "intensity_rate_metrics" in data["discrete_flashing_ratchet_regular_neural_mjp"][model]:
                    rmse = data["discrete_flashing_ratchet_regular_neural_mjp"][model]["intensity_rate_metrics"]["rmse"]
                    f.write(f"  {model}: {rmse}\n")
            f.write("\n")

    if "discrete_flashing_ratchet_irregular_neural_mjp" in data:
        with open(file, "a") as f:
            f.write("Discrete Flashing Ratchet Irregular\n")
            f.write("\n")
            np.set_printoptions(precision=2)
            ground_truth_intensity_matrix = data["discrete_flashing_ratchet_irregular_neural_mjp"]["ground_truth"]["intensity_matrices"][0]
            f.write(f"  Intensity Matrix\n")
            neural_mjp_intensity_matrix = compute_ground_truth_dfr_intensity_matrix(6, 1, 0.98, 1.11, 1.13)
            f.write(f"  NeuralMJP: {neural_mjp_intensity_matrix}\n")
            for model in data["discrete_flashing_ratchet_irregular_neural_mjp"]:
                if "intensity_matrices" in data["discrete_flashing_ratchet_irregular_neural_mjp"][model]:
                    intensity_matrix = np.array(data["discrete_flashing_ratchet_irregular_neural_mjp"][model]["intensity_matrices"][0])
                    f.write(f"  {model}: {intensity_matrix}\n")
            f.write("\n")
            f.write("   Intensity Rate RMSE\n")
            neural_mjp_rmse = get_intensity_matrix_rmse(neural_mjp_intensity_matrix, ground_truth_intensity_matrix)
            f.write(f"  NeuralMJP: {neural_mjp_rmse}\n")
            for model in data["discrete_flashing_ratchet_irregular_neural_mjp"]:
                if "intensity_rate_metrics" in data["discrete_flashing_ratchet_irregular_neural_mjp"][model]:
                    rmse = data["discrete_flashing_ratchet_irregular_neural_mjp"][model]["intensity_rate_metrics"]["rmse"]
                    f.write(f"  {model}: {rmse}\n")
            f.write("\n")

    if "ADP" in data:
        with open(file, "a") as f:
            f.write("ADP\n")
            f.write("\n")
            np.set_printoptions(precision=2)
            f.write(f"  Intensity Matrix\n")
            neural_mjp_intensity_matrix = np.array([[0, 53, 0.19, 7.9, 0.06, 0.02], [47, 0, 0.05, 12, 0.04, 0.01], [0.28, 0.13, 0, 17, 0.02, 0.04], [36, 26.9, 41, 0, 0.3, 0.01], [0.17, 0.2, 0.4, 0.2, 0, 3], [1.2, 1.8, 0.5, 0.7, 19, 0]])
            for i in range(6):
                neural_mjp_intensity_matrix[i, i] = -torch.sum(neural_mjp_intensity_matrix[i])
            f.write(f"  NeuralMJP: {neural_mjp_intensity_matrix}\n")
            for model in data["ADP"]:
                if "intensity_matrices" in data["ADP"][model]:
                    intensity_matrix = np.array(data["ADP"][model]["intensity_matrices"][0])
                    intensity_matrix *= 1000 # Convert to the same time scale as NeuralMJP
                    f.write(f"  {model}: {intensity_matrix}\n")
            f.write("\n")

    if "ADP_neural_mjp_labels" in data:
        with open(file, "a") as f:
            f.write("ADP NeuralMJP Labels\n")
            f.write("\n")
            np.set_printoptions(precision=2)
            f.write(f"  Intensity Matrix\n")
            neural_mjp_intensity_matrix = np.array([[0, 53, 0.19, 7.9, 0.06, 0.02], [47, 0, 0.05, 12, 0.04, 0.01], [0.28, 0.13, 0, 17, 0.02, 0.04], [36, 26.9, 41, 0, 0.3, 0.01], [0.17, 0.2, 0.4, 0.2, 0, 3], [1.2, 1.8, 0.5, 0.7, 19, 0]])
            for i in range(6):
                neural_mjp_intensity_matrix[i, i] = -torch.sum(neural_mjp_intensity_matrix[i])
            f.write(f"  NeuralMJP: {neural_mjp_intensity_matrix}\n")
            for model in data["ADP_neural_mjp_labels"]:
                if "intensity_matrices" in data["ADP_neural_mjp_labels"][model]:
                    intensity_matrix = np.array(data["ADP_neural_mjp_labels"][model]["intensity_matrices"][0])
                    intensity_matrix *= 1000 # Convert to the same time scale as NeuralMJP
                    f.write(f"  {model}: {intensity_matrix}\n")
            f.write("\n")

class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@click.command()
@click.option("-c", "--config", "cfg_path", required=True, type=click.Path(exists=True), help="path to config file")
def main(cfg_path: Path) -> None:
    cfg: dict = load_yaml(cfg_path,return_object=False)
    num_states = cfg["num_states"]

    benchmark = Benchmark(**cfg)
    data = benchmark()

    # print_quantity(data, "ion_channel", "stationary_distributions")
    # print_quantity(data, "discrete_flashing_ratchet", "intensity_rate_metrics")

    # Store data
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    dir = "results/mjp_evaluation"
    if num_states == 2:
        dir += "/two_states"
    elif num_states == 3:
        dir += "/three_states"
    elif num_states == 6:
        dir += "/six_states"
    else:
        raise NotImplementedError("Only implemented for 2, 3 and 6 states")
    Path(dir).mkdir(parents=True, exist_ok=True)
    name = benchmark.name
    with open(f"{dir}/{name}_{timestamp}.pkl", "wb") as f:
        pickle.dump(data, f)

    print_rates_rmse(data, file=f"{dir}/{name}_{timestamp}_rmse.txt")
    print_rates_rmse(data, file=None)

    compare_to_other_papers(num_states, data, file=f"{dir}/{name}_{timestamp}_comparison.txt")

    # Also store as JSON
    with open(f"{dir}/{name}_{timestamp}.json", "w") as f:
        json.dump(data, f, cls=json_serialize, indent=2)

    final_score = compute_final_multi_state_score(data)
    if final_score is not None:
        with open(f"{dir}/{name}_{timestamp}_final_score.txt", "w") as f:
            for i in range(len(final_score[0])):
                f.write(f"Model: {final_score[0][i]}\n")
                f.write(f"RMSE: {final_score[1][i]} +- {final_score[2][i]}\n")


if __name__ == "__main__":
    main()



