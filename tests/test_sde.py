import pytest
import torch
from torch import Tensor

from fim.models.sde import FIMSDE, backward_fill_masked_values, forward_fill_masked_values


class TestFIMSDELoss:
    B: int = 2
    G: int = 10
    D: int = 4
    threshold: float = 1.0

    @pytest.fixture
    def vector_field(self) -> tuple[Tensor]:
        zeros = torch.zeros((self.B, self.G, self.D))
        ones = torch.ones((self.B, self.G, self.D))
        randn = torch.randn((self.B, self.G, self.D))

        return zeros, ones, randn

    @pytest.fixture
    def loss_at_locations(self) -> tuple[Tensor]:
        zeros = torch.zeros((self.B, self.G))
        ones = torch.ones((self.B, self.G))
        randn = torch.randn((self.B, self.G))

        return zeros, ones, randn

    def test_rmse_at_locations(self, vector_field: tuple[Tensor]) -> None:
        zeros, ones, randn = vector_field

        # reaches zero
        estimate = randn  # [B, G, D]
        target = randn  # [B, G, D]
        mask = torch.ones_like(target, dtype=bool)  # [B, G, D]

        rmse_at_locations = FIMSDE.rmse_at_locations(estimate, target, mask)  # [B, G, D]
        assert rmse_at_locations.ndim == 2
        assert torch.all(rmse_at_locations == torch.zeros_like(rmse_at_locations))

        # masking works -> compute loss twice: with masked values and with trunkated values -> they should be the same
        estimate = torch.concatenate([ones, randn], dim=-1)  # [B, G, 2*D]
        mask = torch.concatenate([ones, zeros], dim=-1).bool()  # [B, G, 2*D], first elements are just one, i.e. unmasked
        target = torch.concat([zeros, ones], dim=-1)  # [B, G, 2*D]

        rmse_with_mask = FIMSDE.rmse_at_locations(estimate, target, mask=mask)  # [B, G]
        rmse_with_trunctation = FIMSDE.rmse_at_locations(
            estimate[..., : self.D],
            target[..., : self.D],
            mask[..., : self.D],
        )  # [B, G]

        max_deviation = torch.amax(torch.abs(rmse_with_trunctation - rmse_with_mask))
        assert torch.allclose(rmse_with_trunctation, rmse_with_mask), "RMSEs at locations deviated by max. " + str(max_deviation)

    def test_gaussian_nll_at_locations(self, vector_field: tuple[Tensor]) -> None:
        zeros, ones, randn = vector_field

        # masking works -> compute loss twice: with masked values and with trunkated values -> they should be the same
        estimate = torch.concatenate([ones, randn], dim=-1)  # [B, G, 2*D]
        log_var_estimate = torch.concatenate([randn, ones], dim=-1)  # [B, G, 2*D]
        mask = torch.concatenate([ones, zeros], dim=-1).bool()  # [B, G, 2*D], first elements are just one, i.e. unmasked
        target = torch.concat([zeros, ones], dim=-1)  # [B, G, 2*D]

        nll_with_mask = FIMSDE.gaussian_nll_at_locations(estimate, log_var_estimate, target, mask)  # [B, G]
        nll_with_trunctation = FIMSDE.gaussian_nll_at_locations(
            estimate[..., : self.D], log_var_estimate[..., : self.D], target[..., : self.D], mask[..., : self.D]
        )  # [B, G]

        max_deviation = torch.amax(torch.abs(nll_with_trunctation - nll_with_mask))
        assert torch.allclose(nll_with_trunctation, nll_with_mask), "Gaussian NLLs at locations deviated by max. " + str(max_deviation)

    def test_filter_nans_from_vector_fields(self, vector_field: tuple[Tensor]) -> None:
        zeros, ones, randn = vector_field

        # test None input for log_var_estimate
        estimate = randn
        target = ones
        mask = ones

        filtered_estimate = FIMSDE.filter_nans_from_vector_fields(estimate, None, target, mask)[0]

        assert torch.allclose(filtered_estimate, estimate)

        # test if Nans get removed
        log_var_estimate = zeros

        estimate_nan_mask = torch.bernoulli(0.2 * ones).bool()  # 1 indicates Nan in input
        log_var_estimate_nan_mask = torch.bernoulli(0.2 * ones).bool()  # 1 indicates Nan in input
        target_nan_mask = torch.bernoulli(0.2 * ones).bool()  # 1 indicates Nan in input

        estimate_with_nan = torch.where(estimate_nan_mask, torch.nan, estimate)
        log_var_estimate_with_nan = torch.where(log_var_estimate_nan_mask, torch.nan, log_var_estimate)
        target_with_nan = torch.where(target_nan_mask, torch.nan, target)

        (
            filtered_estimate,
            filtered_log_var_estimate,
            filtered_target,
            are_finite_mask,
            _,
            _,
        ) = FIMSDE.filter_nans_from_vector_fields(estimate_with_nan, log_var_estimate_with_nan, target_with_nan, mask)

        assert torch.isnan(filtered_estimate).any().item() is False
        assert torch.isnan(filtered_log_var_estimate).any().item() is False
        assert torch.isnan(filtered_target).any().item() is False

        # test if values not Nans stay the same
        assert torch.allclose(estimate * are_finite_mask, filtered_estimate * are_finite_mask)
        assert torch.allclose(log_var_estimate * are_finite_mask, filtered_log_var_estimate * are_finite_mask)
        assert torch.allclose(target * are_finite_mask, filtered_target * are_finite_mask)

    def test_filter_loss_at_locations(self, loss_at_locations: tuple[Tensor]) -> None:
        zeros, ones, randn = loss_at_locations
        loss_nan_mask = torch.bernoulli(0.2 * ones).bool()  # 1 indicates Nan in loss

        # test non Nans
        loss = randn
        filter_mask, filter_perc = FIMSDE.filter_loss_at_locations(loss)
        assert filter_perc.item() == 0.0

        # test Nans
        loss_with_nan = torch.where(loss_nan_mask, torch.nan, randn)
        filter_mask, filter_perc = FIMSDE.filter_loss_at_locations(loss_with_nan)
        assert filter_perc.item() == loss_nan_mask.mean(dtype=torch.float32).item()

        # test threshold
        loss = randn
        filter_mask, filter_perc = FIMSDE.filter_loss_at_locations(loss, self.threshold)
        assert torch.all(torch.abs(filter_mask * loss) <= self.threshold)
        assert filter_perc.item() == (torch.abs(randn) > self.threshold).mean(dtype=torch.float32).item()

        # test threshold and Nan
        loss_with_nan = torch.where(loss_nan_mask, torch.nan, randn)
        filter_mask, filter_perc = FIMSDE.filter_loss_at_locations(loss_with_nan, self.threshold)
        assert torch.all(torch.abs(filter_mask * torch.logical_not(loss_nan_mask) * randn) <= self.threshold)
        assert filter_perc.item() == torch.logical_or((torch.abs(randn) > self.threshold), (loss_nan_mask)).mean(dtype=torch.float32).item()


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
