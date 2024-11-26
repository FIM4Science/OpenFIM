import torch
import numpy as np

def define_grid_ranges(path_realization,ignore_percentatge=.2):
    """
    returns max and min values per dimensions

    ranges = list(tuple(min_val,max_val))
    """
    dimensions = path_realization.size(1)
    ranges = []
    for dimension_index in range(dimensions):
        min_ = path_realization[:,dimension_index].min()
        max_ = path_realization[:,dimension_index].max()
        gap = (max_ - min_)*ignore_percentatge
        range_ = (min_+gap,max_-gap)
        ranges.append(range_)
    return ranges

def define_mesh_points(total_points = 100,n_dims = 1, ranges=[]):  # Number of dimensions
    """
    returns a points form the mesh defined in the range given the list ranges
    """
    # Calculate the number of points per dimension
    number_of_points = int(np.round(total_points ** (1 / n_dims)))
    if  len(ranges) == n_dims:
    # Define the range for each dimension
        ranges = [torch.linspace(ranges[_][0], ranges[_][1], number_of_points) for _ in range(n_dims)]
    else:
        ranges = [torch.linspace(-2.0, 2.0, number_of_points) for _ in range(n_dims)]
    # Create a meshgrid for n dimensions
    meshgrids = torch.meshgrid(*ranges, indexing='ij')
    # Stack and reshape to get the observation points
    points = torch.stack(meshgrids, dim=-1).view(-1, n_dims)
    return points
