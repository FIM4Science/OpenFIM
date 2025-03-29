import logging
from typing import Literal

import torch

from .logging import RankLoggerAdapter


class KernelInterpolator:
    """A class for interpolating or finding closest values from a discrete grid.

    This class provides functionality to either interpolate values between grid points
    or find the closest grid point value for new evaluation locations.

    Args:
       grid_points (torch.Tensor): The grid points where the function is evaluated. Shape: [num_grid_points]
       values (torch.Tensor): The function values at grid points. Shape: [..., num_grid_points]
       mode (str): Either 'interpolate' or 'nearest'. If 'interpolate', performs linear interpolation.
                   If 'nearest', returns the value of the closest grid point.
       out_of_bounds_value (float): The value to return for query points that are out of bounds of the grid points.
    """

    def __init__(
        self,
        grid_points: torch.Tensor,
        values: torch.Tensor,
        mode: Literal["interpolate", "nearest"] = "interpolate",
        out_of_bounds_value: float = 0.0,
    ):
        self.grid_points = grid_points.float()
        self.values = values.float()
        self.mode = mode
        self.logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))
        # Ensure grid points are sorted
        if not torch.all(grid_points[..., :-1] <= grid_points[..., 1:]):
            sorted_indices = torch.argsort(grid_points, dim=-1)
            self.grid_points = grid_points[..., sorted_indices]
            self.values = values[..., sorted_indices]
        self.out_of_bounds_value = out_of_bounds_value
        self.num_markers = self.values.shape[0]

    def __call__(self, query_points: torch.Tensor) -> torch.Tensor:
        """Evaluate the function at new query points.

        Args:
           query_points (torch.Tensor): Points where to evaluate the function. Shape: [num_query_points]

        Returns:
           torch.Tensor: Interpolated or nearest values at query points. Shape: [..., num_query_points]
        """

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

        grid = self.grid_points.unsqueeze(1).unsqueeze(1)
        for i, query in enumerate(query_points):
            query = query.unsqueeze(-1)
            exact_match = torch.isclose(query, grid, rtol=1e-7, atol=1e-8)

            exact_match_mask[i] = exact_match.any(dim=-1)
            indices = torch.nonzero(exact_match, as_tuple=True)
            indices = torch.stack((indices[0], indices[-1]), dim=-1)
            exact_match_indices.append(indices)

        return exact_match_mask, exact_match_indices

    def _interpolate(self, query_points: torch.Tensor) -> torch.Tensor:
        """Perform linear interpolation between grid points.

        Args:
            query_points (torch.Tensor): Points where to evaluate the function. Shape: [B, M, L, N]

        Returns:
            torch.Tensor: Interpolated values at query points. Shape: [..., num_query_points]
        """
        # First check for exact matches
        exact_match_mask, exact_match_indices = self._find_exact_matches(query_points)
        result = torch.zeros_like(query_points, dtype=self.values.dtype, device=self.values.device)
        # Handle exact matches
        if exact_match_mask.any():
            for i, indices in enumerate(exact_match_indices):
                result[i, exact_match_mask[i]] = self.values[indices[:, 0], indices[:, 1]]

        if not exact_match_mask.all():
            # Check for out-of-bounds query points
            if (query_points < self.grid_points[:, 0].reshape(1, -1, 1, 1)).any() or (
                query_points > self.grid_points[:, -1].reshape(1, -1, 1, 1)
            ).any():
                self.logger.error(
                    f"Query points are out of bounds of the grid points. The kernel values for these points will be set to {self.out_of_bounds_value}."
                )
            # Find indices of grid points that bracket each query point
            indices = self._search_sorted(self.grid_points, query_points)
            indices = torch.clamp(indices, 1, self.grid_points.shape[-1] - 1)

            indices = indices.unsqueeze(-1)
            L, N = query_points.shape[2], query_points.shape[3]
            grid_points = self.grid_points.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            values = self.values.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            grid_points = grid_points.expand(query_points.shape[0], -1, L, N, -1)
            values = values.expand(query_points.shape[0], -1, L, N, -1)

            # Get the grid points and values that bracket each query point
            x0 = torch.gather(grid_points, -1, indices - 1).squeeze(-1)
            x1 = torch.gather(grid_points, -1, indices).squeeze(-1)
            y0 = torch.gather(values, -1, indices - 1).squeeze(-1)
            y1 = torch.gather(values, -1, indices).squeeze(-1)

            # Compute interpolation weights
            weights = (query_points - x0) / (x1 - x0)

            interpolated_values = y0 + weights * (y1 - y0)
            # Perform linear interpolation
            result[~exact_match_mask] = interpolated_values[~exact_match_mask]

        return result

        # # Handle interpolation for non-exact matches
        # for k in range(result.shape[1]):
        #     if not exact_match_mask[:, k].all():
        #         # Check for out-of-bounds query points
        #         qp = query_points[:, k, ~exact_match_mask[:, k]]
        #         grid_points = self.grid_points[0, 0]
        #         if (qp < grid_points[0]).any() or (qp > grid_points[-1]).any():
        #             self.logger.error(
        #                 f"Query points are out of bounds of the grid points. The kernel values for these points will be set to {self.out_of_bounds_value}."
        #             )

        #         # Find indices of grid points that bracket each query point
        #         indices = self._search_sorted(grid_points, qp)
        #         indices = torch.clamp(indices, 1, len(grid_points) - 1)

        #         # Get the grid points and values that bracket each query point
        #         x0 = grid_points[indices - 1]
        #         x1 = grid_points[indices]
        #         ix_values = torch.nonzero(~exact_match_mask[:, k], as_tuple=True)[0]
        #         y0 = self.values[ix_values, indices - 1]
        #         y1 = self.values[ix_values, indices]

        #         # Compute interpolation weights
        #         weights = (qp - x0) / (x1 - x0)

        #         # Calculate the interpolations
        #         interpolated_values = y0 + weights * (y1 - y0)

        #         # Perform linear interpolation
        #         result[:, k, ~exact_match_mask[:, k]] = torch.clamp(interpolated_values, min=0)

        # return result

    def _nearest(self, query_points: torch.Tensor) -> torch.Tensor:
        """Find the value of the closest grid point."""
        # First check for exact matches
        exact_match_mask, exact_match_indices = self._find_exact_matches(query_points)

        # Initialize result tensor
        result = torch.zeros_like(query_points, dtype=self.values.dtype, device=self.values.device)

        # Handle exact matches
        if exact_match_mask.any():
            for i, indices in enumerate(exact_match_indices):
                result[i, exact_match_mask[i]] = self.values[indices[:, 0], indices[:, 1]]

        # Handle nearest neighbor for non-exact matches
        if not exact_match_mask.all():
            if (query_points < self.grid_points[:, 0].reshape(1, -1, 1, 1)).any() or (
                query_points > self.grid_points[:, -1].reshape(1, -1, 1, 1)
            ).any():
                self.logger.error(
                    f"Query points are out of bounds of the grid points. The kernel values for these points will be set to {self.out_of_bounds_value}."
                )

            # Find indices of grid points that bracket each query point
            indices = self._search_sorted(self.grid_points, query_points)
            indices = torch.clamp(indices, 1, self.grid_points.shape[-1] - 1)
            indices = indices.unsqueeze(-1)
            L, N = query_points.shape[2], query_points.shape[3]
            values = self.values.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            values = values.expand(query_points.shape[0], -1, L, N, -1)

            y = torch.gather(values, -1, indices).squeeze(-1)
            # Get the values at the closest grid points
            result[~exact_match_mask] = y[~exact_match_mask]

        return result

    def _search_sorted(self, points, query_points):
        """Find the index of the closest grid point for each query point."""

        # Compute pairwise distances between `query_points` and `points`
        distances = torch.abs(query_points.unsqueeze(-1) - points.reshape(1, self.num_markers, 1, 1, -1))
        index_bias = torch.arange(distances.size(-1), 0, -1, dtype=distances.dtype, device=distances.device) * 1e-6
        # Find the index of the minimum distance
        distances = distances + index_bias
        closest_indices = distances.argmin(dim=-1)
        return closest_indices
