import torch
import numpy as np


def random_size_consecutive_locations(
    hidden_values,
    hidden_time,
    observation_time_params,
):
    """
    We just sample from a number of observation distribution and pick that
    number of consecutive events in a given data bulck per path, mask and lenght
    are provided and the returned sequences have padding

    returns
    -------
    obs_values,obs_times,obs_mask,obs_lenght
    """
    B, P, T, D = hidden_values.shape
    size_distribution = observation_time_params.get("size_distribution")
    max_samples = T
    # Step 1: Sample the number of observed points based on the chosen distribution
    if size_distribution == "poisson":
        av_num_observations = observation_time_params.get("av_num_observations", int(0.8 * T))
        num_observed = (
            torch.poisson(torch.full((B, P), float(av_num_observations))).clamp(1, T).int()
        )  # Poisson sampling, limited to [1, T]
    elif size_distribution == "uniform":
        # specify the range for uniform sampling
        low = observation_time_params.get("low")
        high = observation_time_params.get("high")
        num_observed = torch.randint(low, high + 1, (B, P))  # Uniform sampling between `low` and `high`
    else:
        raise ValueError("sampling_method must be either 'poisson' or 'uniform'")

    max_samples = num_observed.max().item()
    # Step 2: Randomly select start indices for each sequence in range [0, T - num_observed + 1]
    start_indices = torch.zeros(B, P)  # maximum valid range for start indices

    # Step 3: Generate indices for consecutive selection based on start and observed count
    # Create a tensor for the range of each selection
    range_tensor = torch.arange(T).expand(B, P, T)  # shape [B, P, T]
    selection_mask = (range_tensor >= start_indices.unsqueeze(-1)) & (
        range_tensor < (start_indices + num_observed).unsqueeze(-1)
    )  # mask of selected indices

    # Step 4 cut sizes
    selection_mask = selection_mask[:, :, :max_samples]
    selected_times = hidden_time[:, :, :max_samples, :]
    selected_values = hidden_values[:, :, :max_samples, :]

    # Step 5: Apply mask to extract values and pad to max observed
    selected_times = torch.where(selection_mask.unsqueeze(-1), selected_times, torch.tensor(0.0))
    selected_values = torch.where(selection_mask.unsqueeze(-1).expand(-1, -1, -1, D), selected_values, torch.tensor(0.0))

    # Step 6: Mask to indicate valid vs. padded entries
    mask = selection_mask.int()

    return selected_values, selected_times, mask, num_observed


# Define Mesh Points
def define_mesh_points(total_points=100, n_dims=1, ranges=[]) -> torch.Tensor:  # Number of dimensions
    """
    returns a points form the mesh defined in the range given the list ranges
    """
    # Calculate the number of points per dimension
    number_of_points = int(np.round(total_points ** (1 / n_dims)))
    if len(ranges) == n_dims:
        # Define the range for each dimension
        axes_grid = [torch.linspace(ranges[_][0], ranges[_][1], number_of_points) for _ in range(n_dims)]
    else:
        axes_grid = [torch.linspace(-1.0, 1.0, number_of_points) for _ in range(n_dims)]
    # Create a meshgrid for n dimensions
    meshgrids = torch.meshgrid(*axes_grid, indexing="ij")
    # Stack and reshape to get the observation points
    points = torch.stack(meshgrids, dim=-1).view(-1, n_dims)
    return points
