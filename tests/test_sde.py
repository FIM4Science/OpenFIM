from copy import deepcopy

import pytest
import torch
from torch import Tensor

from fim import test_data_path
from fim.models.blocks import ModelFactory
from fim.models.sde import (
    FIMSDE,
    FIMSDEConfig,
    backward_fill_masked_values,
    forward_fill_masked_values,
)
from fim.sampling.sde_path_samplers import (
    fimsde_euler_maruyama,
    fimsde_sample_paths,
    fimsde_sample_paths_by_dt_and_grid_size,
)
from fim.utils.helper import load_yaml


class TestMaskedFill:
    @pytest.fixture
    def data(self) -> tuple[Tensor]:
        observation_values = torch.tensor(
            [
                [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]],
                [[15, 16], [17, 18], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28]],
            ],
            dtype=torch.float,
        )

        observed_mask = torch.tensor(
            [
                [[False], [False], [True], [False], [False], [True], [True]],
                [[True], [False], [False], [True], [True], [False], [False]],
            ],
            dtype=torch.bool,
        )

        return observation_values, observed_mask

    @pytest.fixture
    def expected_forward_fill(self) -> Tensor:
        return torch.tensor(
            [
                [[5, 6], [5, 6], [5, 6], [5, 6], [5, 6], [11, 12], [13, 14]],
                [[15, 16], [15, 16], [15, 16], [21, 22], [23, 24], [23, 24], [23, 24]],
            ],
            dtype=torch.float,
        )

    @pytest.fixture
    def expected_backward_fill(self) -> Tensor:
        return torch.tensor(
            [
                [[5, 6], [5, 6], [5, 6], [11, 12], [11, 12], [11, 12], [13, 14]],
                [[15, 16], [21, 22], [21, 22], [21, 22], [23, 24], [23, 24], [23, 24]],
            ],
            dtype=torch.float,
        )

    def test_forward_fill(self, data: tuple[Tensor], expected_forward_fill: Tensor) -> None:
        observation_values, observed_mask = data

        forward_fill = forward_fill_masked_values(observation_values, observed_mask)

        assert torch.allclose(forward_fill, expected_forward_fill)

    def test_backward_fill(self, data: tuple[Tensor], expected_backward_fill: Tensor) -> None:
        observation_values, observed_mask = data

        backward_fill = backward_fill_masked_values(observation_values, observed_mask)

        assert torch.allclose(backward_fill, expected_backward_fill)


class TestModelPathsSampling:
    B: int = 10
    P: int = 3
    T: int = 50
    D: int = 3
    I: int = 7

    @pytest.fixture(scope="module")
    def results_dir(self, tmp_path_factory):
        return tmp_path_factory.mktemp("results")

    @pytest.fixture(scope="module")
    def model(self, results_dir) -> FIMSDE:
        TRAIN_CONF = test_data_path / "config" / "sde" / "sde_mini.yaml"
        config = load_yaml(TRAIN_CONF, True)
        model = ModelFactory.create(FIMSDEConfig(**config.model.to_dict()))

        return model

    @pytest.fixture(scope="module")
    def data(self) -> dict:
        obs_times = torch.linspace(0, 5, steps=self.T)
        obs_times = obs_times.reshape(1, 1, -1, 1)
        obs_times = torch.broadcast_to(obs_times, (self.B, self.P, self.T, 1))

        return {
            "obs_times": obs_times,
            "obs_values": torch.randn((self.B, self.P, self.T, self.D)),
        }

    @pytest.fixture(scope="module")
    def grid_data(self) -> tuple:
        initial_time_ = 0
        end_time_ = 2
        grid_size = 200

        initial_states = torch.randn((self.B, self.I, self.D))
        initial_time = initial_time_ * torch.ones((self.B, self.I, 1))
        end_time = end_time_ * torch.ones((self.B, self.I, 1))

        solver_granularity = 3

        return initial_time_, end_time_, grid_size, initial_states, initial_time, end_time, solver_granularity

    def test_fimsde_euler_maruyama(self, model: FIMSDE, data: dict, grid_data: tuple):
        initial_time_, end_time_, grid_size, initial_states, initial_time, end_time, solver_granularity = grid_data

        sample_paths, sample_paths_grid = fimsde_euler_maruyama(
            model, data, solver_granularity, initial_states, initial_time, grid_size, end_time
        )

        assert sample_paths.shape == (self.B, self.I, grid_size, self.D)
        assert sample_paths_grid.shape == (self.B, self.I, grid_size, 1)
        assert torch.allclose(initial_states, sample_paths[:, :, 0, :], rtol=0.01)
        assert torch.allclose(sample_paths_grid[:, :, 0, :], initial_time)
        assert torch.allclose(sample_paths_grid[:, :, -1, :], end_time)

    def test_fimsde_sample_paths_by_dt_and_grid_size(self, model: FIMSDE, data: dict, grid_data: tuple):
        initial_time_, end_time_, grid_size, initial_states, initial_time, end_time, solver_granularity = grid_data

        # sample with initial and end time for comparison
        sample_paths, sample_paths_grid = fimsde_euler_maruyama(
            model, deepcopy(data), solver_granularity, initial_states, initial_time, grid_size, end_time
        )

        # sample with dt
        dt = (end_time_ - initial_time_) / grid_size

        dt_sample_paths, dt_sample_paths_grid = fimsde_sample_paths_by_dt_and_grid_size(
            model, deepcopy(data), grid_size, solver_granularity, initial_states, initial_time, dt
        )

        # compare
        assert torch.allclose(dt_sample_paths_grid, sample_paths_grid, rtol=0.001)
        assert sample_paths.shape == dt_sample_paths.shape

    def test_fimsde_sample_paths_based_on_grid(self, model: FIMSDE, data: dict, grid_data: tuple):
        initial_time_, end_time_, grid_size, initial_states, initial_time, end_time, solver_granularity = grid_data

        # test sampling based on grid
        grid = torch.linspace(initial_time_, end_time_, steps=grid_size)
        grid = torch.broadcast_to(grid.view(1, 1, -1, 1), (self.B, self.I, grid_size, 1))

        sample_paths, sample_paths_grid = fimsde_sample_paths(model, data, initial_states=initial_states, grid=grid)

        assert torch.allclose(sample_paths_grid, grid, rtol=0.01)
        assert sample_paths.shape == (self.B, self.I, grid_size, self.D)

    def test_fimsde_sample_paths_no_initial_states(self, model: FIMSDE, data: dict, grid_data: tuple):
        initial_time_, end_time_, grid_size, initial_states, initial_time, end_time, solver_granularity = grid_data

        # test no initial states passed
        sample_paths, sample_paths_grid = fimsde_sample_paths(
            model, data, grid_size=grid_size, initial_time=initial_time, end_time=end_time
        )
        assert torch.allclose(sample_paths[:, :, 0, :], data["obs_values"][:, :, 0, :], rtol=0.01)

    def test_fimsde_sample_paths_subsampling_num_paths(self, model: FIMSDE, data: dict, grid_data: tuple):
        initial_time_, end_time_, grid_size, initial_states, initial_time, end_time, solver_granularity = grid_data

        # test subsampling number of paths
        num_paths = self.I // 2
        sample_paths, sample_paths_grid = fimsde_sample_paths(
            model, data, grid_size=grid_size, initial_time=initial_time, end_time=end_time, num_paths=num_paths
        )
        assert sample_paths.shape[1] == num_paths, f"Expected {num_paths} paths. Got {sample_paths.shape[1]}."

    def test_fimsde_sample_paths_interval_by_floats(self, model: FIMSDE, data: dict, grid_data: tuple):
        initial_time_, end_time_, grid_size, initial_states, initial_time, end_time, solver_granularity = grid_data

        # test specifying solver interval by floats
        sample_paths, sample_paths_grid = fimsde_sample_paths(
            model, data, grid_size=grid_size, initial_time=initial_time_, end_time=end_time_
        )

        assert torch.allclose(sample_paths_grid[:, :, 0, :], initial_time_ * torch.ones_like(sample_paths_grid[:, :, 0, :]))
        assert torch.allclose(sample_paths_grid[:, :, -1, :], end_time_ * torch.ones_like(sample_paths_grid[:, :, 0, :]))

    def test_fimsde_sample_paths_with_dt(self, model: FIMSDE, data: dict, grid_data: tuple):
        initial_time_, end_time_, grid_size, initial_states, initial_time, end_time, solver_granularity = grid_data

        # test specifying dt
        dt = (end_time_ - initial_time_) / grid_size
        sample_paths, sample_paths_grid = fimsde_sample_paths(
            model, data, grid_size=grid_size, initial_states=initial_states, initial_time=initial_time_, dt=dt
        )

        assert sample_paths.shape == (self.B, self.I, grid_size, self.D)
