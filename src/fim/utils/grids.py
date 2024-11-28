from typing import Optional

import numpy as np
import torch
from torch import Tensor


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


def get_hypercube_boundaries(values: Tensor, extension_perc: Optional[float] = 0.0) -> tuple[Tensor, Tensor]:
    """
    Define boundaries of a hypercube surrounding list of values, optionally extended by some percentage.

    Args:
        values (Tensor): Tensor of values to compute boundaries from. Shape: [..., *, D]
        extension_perc (float): Increase size of cube by extension_perc.

    Returns:
        cube_min, cube_max (Tensor): Boundaries of extended hypercube, surrounding values in * dimension.  Shape: [..., D]
    """

    values_min, _ = torch.min(values, dim=-2)  # [..., D]
    values_max, _ = torch.max(values, dim=-2)
    values_range = values_max - values_min

    cube_min = values_min - (extension_perc / 2) * values_range
    cube_max = values_max + (extension_perc / 2) * values_range

    return cube_min, cube_max


def vmapped_linspace(x: Tensor, y: Tensor, steps: int) -> Tensor:
    """
    Regular grid between last dimension of input tensors.

    Args:
        x, y (Tensor): Tensors of same shape.
        steps (int): Number of points on grid.

    Returns:
        linspace (Tensor): Shape: x.shape + (steps, )
    """
    unit_grid = torch.linspace(0, 1, steps=steps)

    x = x[..., None]
    y = y[..., None]

    return x + (y - x) * unit_grid


def define_regular_surrounding_cube(num_points: int, paths_values: Tensor, extension_perc: Optional[float] = 0.0) -> Tensor:
    """
    Define regular points in a cube surrounding the observations of multiple paths.

    Args:
        num_points (int): Targeted number of points in cube.
        paths_values (Tensor): Observations of multiple paths. Shape [..., num_paths, num_obs, D]
        extension_perc (float): Increase size of cube by extension_perc.

    Returns:
        cube_points (Tensor): Points in a cube surrounding paths_values. Shape [..., num_points_realized, D]

        where num_points_realized is the maximum number of points in a regular grid cube that is smaller than num_points
    """

    D = paths_values.shape[-1]

    # define boundaries of cube
    paths_values = torch.flatten(paths_values, start_dim=-3, end_dim=-2)  # [..., *, D]
    cube_min, cube_max = get_hypercube_boundaries(paths_values, extension_perc)

    # num points in regular D-dimensional grid
    num_points_per_dim = int(np.round(num_points ** (1 / D)))

    # cartesian_prod and linspace are not vectorized
    vmapped_cartesian_prod = torch.vmap(torch.cartesian_prod)

    # points in cube are cartesian product of regular grids in each dimension
    grid_per_dim = [vmapped_linspace(cube_min[..., d], cube_max[..., d], steps=num_points_per_dim) for d in range(D)]

    if D == 1:
        cube_points = grid_per_dim[0].unsqueeze(-1)

    else:
        cube_points = vmapped_cartesian_prod(*grid_per_dim)  # [..., num_points_per_dim ** D, D]

    return cube_points


def define_random_surrounding_cube(num_points: int, paths_values: Tensor, extension_perc: Optional[float] = 0.0) -> Tensor:
    """
    Define points in a cube surrounding the observations of multiple paths sampled from the uniform distribution.

    Args:
        num_points (int): Number of points in cube.
        paths_values (Tensor): Observations of multiple paths. Shape [..., num_paths, num_obs, D]
        extension_perc (float): Increase size of cube by extension_perc.

    Returns:
        cube_points (Tensor): Points in a cube surrounding paths_values. Shape [..., num_points, D]
    """

    # define boundaries of cube
    paths_values = torch.flatten(paths_values, start_dim=-3, end_dim=-2)  # [..., *, D]
    cube_min, cube_max = get_hypercube_boundaries(paths_values, extension_perc)  # [..., D]
    cube_min, cube_max = cube_min.unsqueeze(-2), cube_max.unsqueeze(-2)  # [..., 1, D]

    # define size of hypercube
    D = paths_values.shape[-1]
    cube_size = paths_values.shape[:-2] + (num_points, D)  # [..., num_points, D]

    # random points by translated random unit cube
    unit_grid = torch.rand(size=cube_size)
    cube_points = cube_min + (cube_max - cube_min) * unit_grid  # [..., num_points, D]

    return cube_points
