import pytest
import torch

from fim.utils.interpolator import KernelInterpolator


@pytest.fixture
def batch_grid():
    """Create a grid with batch dimensions for testing."""
    grid_points = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    ).reshape(3, 7)
    values = torch.tensor(
        [[0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0], [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0], [0.0, 1.0, 8.0, 27.0, 64.0, 125.0, 216.0]]
    )
    return grid_points, values


@pytest.fixture
def batch_grid_1_kernel():
    grid_points = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]).reshape(1, 7)
    values = torch.tensor([[0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0]]).reshape(1, 7)
    return grid_points, values


def test_interpolate_mode(batch_grid):
    """Test exact matches with batched values."""
    grid_points, values = batch_grid
    print(grid_points.shape, values.shape)
    interpolator = KernelInterpolator(grid_points, values, mode="interpolate")

    # Mix of exact and non-exact query points
    query_points = torch.tensor(
        [
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.5, 1.5, 1.5, 1.5],
                [2.0, 2.0, 2.0, 2.0],
                [2.5, 2.5, 2.5, 2.5],
                [3.0, 3.0, 3.0, 3.0],
                [6.0, 6.0, 6.0, 6.0],
            ],
            [
                [1.0, 1.5, 0.8, 1.5],
                [1.5, 1.5, 1.5, 1.5],
                [2.3, 2.0, 2.0, 2.3],
                [2.8, 2.0, 2.5, 2.5],
                [3.0, 2.0, 3.0, 3.0],
                [4.0, 2.0, 4.5, 6.0],
            ],
        ]
    ).reshape(2, 6, 4)
    result = interpolator(query_points)

    # Check shape
    assert result.shape == (2, 3, 6, 4)

    for i in range(1):
        for j in range(3):
            # Check exact matches
            assert torch.allclose(result[i, j, 0], values[j, 1].repeat(4))  # x = 1.0
            assert torch.allclose(result[i, j, 2], values[j, 2].repeat(4))  # x = 2.0
            assert torch.allclose(result[i, j, 4], values[j, 3].repeat(4))  # x = 3.0
            assert torch.allclose(result[i, j, 5], values[j, 6].repeat(4))  # x = 6.0

            if j < 2:
                # Check interpolated values
                assert torch.allclose(result[i, j, 1], torch.tensor(2.5).repeat(4))  # x = 1.5
                assert torch.allclose(result[i, j, 3], torch.tensor(6.5).repeat(4))  # x = 2.5
            else:
                assert torch.allclose(result[i, j, 1], torch.tensor(4.5).repeat(4))  # x = 1.5
                assert torch.allclose(result[i, j, 3], torch.tensor(17.5).repeat(4))  # x = 2.5

    for i in range(1, 2):
        for j in range(3):
            # Check exact matches
            if j < 2:
                assert torch.allclose(result[i, j, 0], torch.tensor([1.0, 2.5, 0.8, 2.5]))
                assert torch.allclose(result[i, j, 1], torch.tensor([2.5, 2.5, 2.5, 2.5]))
                assert torch.allclose(result[i, j, 2], torch.tensor([4.9, 4.0, 4.0, 4.9]))  # x = 2.0
                assert torch.allclose(result[i, j, 4], torch.tensor([9.0, 4.0, 9.0, 9.0]))  # x = 3.0
                assert torch.allclose(result[i, j, 5], torch.tensor([16.0, 4.0, 20.5, 36.0]))  # x = 6.0
            else:
                assert torch.allclose(result[i, j, 0], torch.tensor([1.0, 4.5, 0.8, 4.5]))
                assert torch.allclose(result[i, j, 1], torch.tensor([4.5, 4.5, 4.5, 4.5]))
                assert torch.allclose(result[i, j, 2], torch.tensor([10.1, 8.0, 8.0, 10.1]))  # x = 2.0
                assert torch.allclose(result[i, j, 3], torch.tensor([23.2, 8.0, 17.5, 17.5]))  # x = 2.5
                assert torch.allclose(result[i, j, 4], torch.tensor([27.0, 8.0, 27.0, 27.0]))  # x = 3.0
                assert torch.allclose(result[i, j, 5], torch.tensor([64.0, 8.0, 94.5, 216.0]))  # x = 6.0


def test_interpolate_mode_1_kernel(batch_grid_1_kernel):
    """Test exact matches with batched values."""
    grid_points, values = batch_grid_1_kernel
    print(grid_points.shape, values.shape)
    interpolator = KernelInterpolator(grid_points, values, mode="interpolate")

    # Mix of exact and non-exact query points
    query_points = torch.tensor(
        [
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.5, 1.5, 1.5, 1.5],
                [2.0, 2.0, 2.0, 2.0],
                [2.5, 2.5, 2.5, 2.5],
                [3.0, 3.0, 3.0, 3.0],
                [6.0, 6.0, 6.0, 6.0],
            ],
            [
                [1.0, 1.5, 0.8, 1.5],
                [1.5, 1.5, 1.5, 1.5],
                [2.3, 2.0, 2.0, 2.3],
                [2.8, 2.0, 2.5, 2.5],
                [3.0, 2.0, 3.0, 3.0],
                [4.0, 2.0, 4.5, 6.0],
            ],
        ]
    ).reshape(2, 6, 4)
    result = interpolator(query_points)

    # Check shape
    assert result.shape == (2, 1, 6, 4)

    for i in range(1):
        for j in range(1):
            # Check exact matches
            assert torch.allclose(result[i, j, 0], values[j, 1].repeat(4))  # x = 1.0
            assert torch.allclose(result[i, j, 2], values[j, 2].repeat(4))  # x = 2.0
            assert torch.allclose(result[i, j, 4], values[j, 3].repeat(4))  # x = 3.0
            assert torch.allclose(result[i, j, 5], values[j, 6].repeat(4))  # x = 6.0

            if j < 2:
                # Check interpolated values
                assert torch.allclose(result[i, j, 1], torch.tensor(2.5).repeat(4))  # x = 1.5
                assert torch.allclose(result[i, j, 3], torch.tensor(6.5).repeat(4))  # x = 2.5
            else:
                assert torch.allclose(result[i, j, 1], torch.tensor(4.5).repeat(4))  # x = 1.5
                assert torch.allclose(result[i, j, 3], torch.tensor(17.5).repeat(4))  # x = 2.5

    for i in range(1, 2):
        for j in range(1):
            # Check exact matches
            if j < 2:
                assert torch.allclose(result[i, j, 0], torch.tensor([1.0, 2.5, 0.8, 2.5]))
                assert torch.allclose(result[i, j, 1], torch.tensor([2.5, 2.5, 2.5, 2.5]))
                assert torch.allclose(result[i, j, 2], torch.tensor([4.9, 4.0, 4.0, 4.9]))  # x = 2.0
                assert torch.allclose(result[i, j, 4], torch.tensor([9.0, 4.0, 9.0, 9.0]))  # x = 3.0
                assert torch.allclose(result[i, j, 5], torch.tensor([16.0, 4.0, 20.5, 36.0]))  # x = 6.0
            else:
                assert torch.allclose(result[i, j, 0], torch.tensor([1.0, 4.5, 0.8, 4.5]))
                assert torch.allclose(result[i, j, 1], torch.tensor([4.5, 4.5, 4.5, 4.5]))
                assert torch.allclose(result[i, j, 2], torch.tensor([10.1, 8.0, 8.0, 10.1]))  # x = 2.0
                assert torch.allclose(result[i, j, 3], torch.tensor([23.2, 8.0, 17.5, 17.5]))  # x = 2.5
                assert torch.allclose(result[i, j, 4], torch.tensor([27.0, 8.0, 27.0, 27.0]))  # x = 3.0
                assert torch.allclose(result[i, j, 5], torch.tensor([64.0, 8.0, 94.5, 216.0]))  # x = 6.0


def test_nearest_mode(batch_grid):
    """Test nearest matches with batched values."""
    grid_points, values = batch_grid
    interpolator = KernelInterpolator(grid_points, values, mode="nearest")

    # Mix of exact and non-exact query points
    query_points = torch.tensor(
        [
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.5, 1.5, 1.5, 1.5],
                [2.0, 2.0, 2.0, 2.0],
                [2.5, 2.5, 2.5, 2.5],
                [3.0, 3.0, 3.0, 3.0],
                [6.0, 6.0, 6.0, 6.0],
            ],
            [
                [1.0, 1.5, 0.8, 1.5],
                [1.5, 1.5, 1.5, 1.5],
                [2.3, 2.0, 2.0, 2.3],
                [2.8, 2.0, 2.5, 2.5],
                [3.0, 2.0, 3.0, 3.0],
                [4.0, 2.0, 4.5, 6.0],
            ],
        ]
    ).reshape(2, 6, 4)
    result = interpolator(query_points)

    # Check shape
    assert result.shape == (2, 3, 6, 4)

    for i in range(1):
        for j in range(3):
            # Check exact matches
            assert torch.allclose(result[i, j, 0], values[j, 1].repeat(4))  # x = 1.0
            assert torch.allclose(result[i, j, 2], values[j, 2].repeat(4))  # x = 2.0
            assert torch.allclose(result[i, j, 4], values[j, 3].repeat(4))  # x = 3.0
            assert torch.allclose(result[i, j, 5], values[j, 6].repeat(4))  # x = 6.0

            if j < 2:
                # Check interpolated values
                assert torch.allclose(result[i, j, 1], torch.tensor(4.0).repeat(4))  # x = 1.5
                assert torch.allclose(result[i, j, 3], torch.tensor(9.0).repeat(4))  # x = 2.5
            else:
                assert torch.allclose(result[i, j, 1], torch.tensor(8.0).repeat(4))  # x = 1.5
                assert torch.allclose(result[i, j, 3], torch.tensor(27.0).repeat(4))  # x = 2.5

    for i in range(1, 2):
        for j in range(3):
            # Check exact matches
            if j < 2:
                assert torch.allclose(result[i, j, 0], torch.tensor([1.0, 4.0, 1.0, 4.0]))
                assert torch.allclose(result[i, j, 1], torch.tensor([4.0, 4.0, 4.0, 4.0]))
                assert torch.allclose(result[i, j, 2], torch.tensor([4.0, 4.0, 4.0, 4.0]))  # x = 2.0
                assert torch.allclose(result[i, j, 4], torch.tensor([9.0, 4.0, 9.0, 9.0]))  # x = 3.0
                assert torch.allclose(result[i, j, 5], torch.tensor([16.0, 4.0, 25, 36.0]))  # x = 6.0
            else:
                assert torch.allclose(result[i, j, 0], torch.tensor([1.0, 8.0, 1.0, 8.0]))
                assert torch.allclose(result[i, j, 1], torch.tensor([8.0, 8.0, 8.0, 8.0]))
                assert torch.allclose(result[i, j, 2], torch.tensor([8.0, 8.0, 8.0, 8.0]))  # x = 2.0
                assert torch.allclose(result[i, j, 3], torch.tensor([27.0, 8.0, 27.0, 27.0]))  # x = 2.5
                assert torch.allclose(result[i, j, 4], torch.tensor([27.0, 8.0, 27.0, 27.0]))  # x = 3.0
                assert torch.allclose(result[i, j, 5], torch.tensor([64.0, 8.0, 125.0, 216.0]))  # x = 6.0
