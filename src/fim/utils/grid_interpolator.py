from typing import Literal

import torch


class GridInterpolator:
    """A class for interpolating or finding closest values from a discrete grid.

    This class provides functionality to either interpolate values between grid points
    or find the closest grid point value for new evaluation locations.

    Args:
       grid_points (torch.Tensor): The grid points where the function is evaluated. Shape: [num_grid_points]
       values (torch.Tensor): The function values at grid points. Shape: [..., num_grid_points]
       mode (str): Either 'interpolate' or 'nearest'. If 'interpolate', performs linear interpolation.
                   If 'nearest', returns the value of the closest grid point.
    """

    def __init__(self, grid_points: torch.Tensor, values: torch.Tensor, mode: Literal["interpolate", "nearest"] = "interpolate"):
        self.grid_points = grid_points[:, :1]
        self.values = values.unsqueeze(0) if values.ndim == 1 else values
        self.mode = mode

        # Ensure grid points are sorted
        if not torch.all(grid_points[..., :-1] <= grid_points[..., 1:]):
            sorted_indices = torch.argsort(grid_points, dim=-1)
            self.grid_points = grid_points[..., sorted_indices]
            self.values = values[..., sorted_indices]

    def __call__(self, query_points: torch.Tensor) -> torch.Tensor:
        """Evaluate the function at new query points.

        Args:
           query_points (torch.Tensor): Points where to evaluate the function. Shape: [num_query_points]

        Returns:
           torch.Tensor: Interpolated or nearest values at query points. Shape: [..., num_query_points]
        """
        if query_points.ndim == 3:
            query_points = query_points.unsqueeze(1).repeat(1, self.values.shape[0], 1, 1)
        if self.mode == "interpolate":
            return self._interpolate(query_points)
        elif self.mode == "nearest":
            return self._nearest(query_points)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _find_exact_matches(self, query_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Find exact matches between query points and grid points.

        Returns:
            tuple: (exact_match_mask, exact_match_indices)
        """
        exact_match_mask = torch.zeros_like(query_points, dtype=torch.bool, device=query_points.device)

        exact_match_indices = []

        grid = self.grid_points.unsqueeze(1)
        for i, query in enumerate(query_points):
            query = query.unsqueeze(-1)
            exact_match = torch.isclose(query, grid, rtol=1e-5, atol=1e-8)

            exact_match_mask[i] = exact_match.any(dim=-1)
            indices = torch.nonzero(exact_match, as_tuple=True)
            indices = torch.stack((indices[0], indices[-1]), dim=-1)
            exact_match_indices.append(indices)

        return exact_match_mask, exact_match_indices

    def _interpolate(self, query_points: torch.Tensor) -> torch.Tensor:
        """Perform linear interpolation between grid points."""
        # First check for exact matches
        exact_match_mask, exact_match_indices = self._find_exact_matches(query_points)
        result = torch.zeros_like(query_points)
        # Handle exact matches
        if exact_match_mask.any():
            for i, indices in enumerate(exact_match_indices):
                result[i, exact_match_mask[i]] = self.values[indices[:, 0], indices[:, 1]]

        # Handle interpolation for non-exact matches
        if not exact_match_mask.all():
            # Check for out-of-bounds query points
            qp = query_points[~exact_match_mask]
            grid_points = self.grid_points[0, 0]
            if (qp < grid_points[0]).any() or (qp > grid_points[-1]).any():
                raise ValueError("Query points are out of bounds of the grid points.")

            # Find indices of grid points that bracket each query point
            indices = self._search_sorted(grid_points, qp)
            indices = torch.clamp(indices, 0, len(grid_points) - 1)

            # Get the grid points and values that bracket each query point
            x0 = grid_points[indices - 1]
            x1 = grid_points[indices]
            ix_values = torch.nonzero(~exact_match_mask, as_tuple=True)[1]
            y0 = self.values[ix_values, indices - 1]
            y1 = self.values[ix_values, indices]

            # Compute interpolation weights
            weights = (query_points[~exact_match_mask] - x0) / (x1 - x0)

            # Perform linear interpolation
            result[~exact_match_mask] = y0 + weights * (y1 - y0)

        return result

    def _nearest(self, query_points: torch.Tensor) -> torch.Tensor:
        """Find the value of the closest grid point."""
        # First check for exact matches
        exact_match_mask, exact_match_indices = self._find_exact_matches(query_points)

        # Initialize result tensor
        result = torch.zeros_like(query_points)

        # Handle exact matches
        if exact_match_mask.any():
            for i, indices in enumerate(exact_match_indices):
                result[i, exact_match_mask[i]] = self.values[indices[:, 0], indices[:, 1]]

        # Handle nearest neighbor for non-exact matches
        if not exact_match_mask.all():
            qp = query_points[~exact_match_mask]
            grid_points = self.grid_points[0, 0]
            if (qp < grid_points[0]).any() or (qp > grid_points[-1]).any():
                raise ValueError("Query points are out of bounds of the grid points.")

            # Find indices of closest grid points
            indices = self._search_sorted(grid_points, qp)
            indices = torch.clamp(indices, 0, len(grid_points) - 1)
            ix_values = torch.nonzero(~exact_match_mask, as_tuple=True)[1]
            # Get the values at the closest grid points
            result[~exact_match_mask] = self.values[ix_values, indices]

        return result

    def _search_sorted(self, points, query_points):
        # Compute pairwise distances between `query_points` and `points`
        distances = torch.abs(query_points.unsqueeze(-1) - points)
        # Find the index of the minimum distance
        closest_indices = distances.flip(dims=[-1]).argmin(dim=-1)
        closest_indices = distances.shape[-1] - 1 - closest_indices
        return closest_indices
