from typing import Callable

import einops
import pytest
import torch

from fim import test_data_path
from fim.models.blocks import ModelFactory
from fim.models.imputation_pointwise import (
    FIMImpPoint,
    FIMImpPointBase,
    FIMImpPointBaseConfig,
    ImputationConcepts,
    StaticWindowing,
    Windowing,
)
from fim.utils.helper import load_yaml


class TestFIMImpPoint:
    @pytest.fixture(scope="class")
    def shared_fim_imp_pointwise_base(self) -> FIMImpPointBase:

        train_config = test_data_path / "config" / "imputation" / "fim_imp_pointwise_base_mini_test.yaml"
        train_config = load_yaml(train_config, True)

        model_config = FIMImpPointBaseConfig(**train_config.model.to_dict())

        fim_imp_pointwise_base: FIMImpPointBase = ModelFactory.create(model_config)
        fim_imp_pointwise_base.eval()

        return fim_imp_pointwise_base

    @pytest.fixture(
        params=[
            pytest.param(None, id="no_windowing"),
            pytest.param(StaticWindowing(1, 0), id="static_single_window"),
            pytest.param(StaticWindowing(4, 0), id="static_multiple_windows_no_overlap"),
            pytest.param(StaticWindowing(5, 0.4), id="static_multiple_windows_with_overlap"),
        ]
    )
    def windowing(self, request) -> Windowing:
        return request.param

    @pytest.fixture(
        params=[
            pytest.param(None, id="none"),
            pytest.param(lambda obs, mask: obs, id="identity"),
        ]
    )
    def denoising_model(self, request) -> Callable:
        return request.param

    @pytest.fixture(  # (B, T, D, G)
        params=[
            pytest.param((1, 90, 3, 5), id="single_batch"),
            pytest.param((3, 90, 3, 5), id="multiple_batches"),
            pytest.param((3, 90, 1, 5), id="single_dimension"),
            pytest.param((4, 30, 8, 100), id="more_evals_than_obs"),
        ]
    )
    def data(self, request) -> tuple:

        shapes = request.param
        B, T, D, G = shapes

        obs_values = torch.randn(B, T, D)
        obs_times = einops.repeat(torch.linspace(0, 10, T).view(1, -1, 1), "1 T 1 -> b T 1", b=B)
        obs_mask = torch.zeros(B, T, 1)
        evaluation_times = einops.repeat(torch.linspace(-1, 11, G).view(1, -1, 1), "1 G 1 -> b G 1", b=B)

        return obs_values, obs_times, obs_mask, evaluation_times

    @pytest.fixture
    def fim_imp_pointwise(self, shared_fim_imp_pointwise_base: FIMImpPointBase, windowing: Windowing, denoising_model: Callable):
        return FIMImpPoint(fim_imp_pointwise_base=shared_fim_imp_pointwise_base, windowing=windowing, denoising_model=denoising_model)

    def test_forward(self, fim_imp_pointwise: FIMImpPoint, data: tuple):
        "Test FIMImpPoint forward output and its shapes under multiple windowings and denoisings."

        obs_values, obs_times, obs_mask, evaluation_times = data

        output = fim_imp_pointwise.forward(obs_times, obs_values, evaluation_times, obs_mask)

        assert isinstance(output, ImputationConcepts)

        B, _, D = obs_values.shape
        G = evaluation_times.shape[1]

        assert output.evaluation_times.shape == (B, G, 1)
        assert output.reconstructed_values.shape == (B, G, D)
        assert output.init_cond_mean.shape == (B, D)
        assert output.init_cond_log_std.shape == (B, D)
        assert output.vector_field_mean.shape == (B, G, D)
        assert output.vector_field_log_std.shape == (B, G, D)
        assert output.normalized is False

    @pytest.mark.parametrize(
        "sizes",  # (B, T, G)
        [
            pytest.param((1, 90, 5), id="single_batch"),
            pytest.param((3, 90, 5), id="multiple_batches"),
            pytest.param((4, 30, 100), id="more_evals_than_obs"),
        ],
    )
    def test_fallback_to_fim_imputation_pointwise_base(self, shared_fim_imp_pointwise_base: FIMImpPointBase, sizes: tuple):
        "Verify that, in single dimension data, with no windowing or denoising defined, FIMImpPoint is simply FIMImpPointBase."

        B, T, G = sizes

        obs_values = torch.randn(B, T, 1)
        obs_times = einops.repeat(torch.linspace(0, 10, T).view(1, -1, 1), "1 T 1 -> b T 1", b=B)
        obs_mask = torch.zeros(B, T, 1)
        evaluation_times = einops.repeat(torch.linspace(-1, 11, G).view(1, -1, 1), "1 G 1 -> b G 1", b=B)

        fim_imp_pointwise = FIMImpPoint(fim_imp_pointwise_base=shared_fim_imp_pointwise_base)  # no windowing, no denoising specified
        fim_imp_pointwise_output: ImputationConcepts = fim_imp_pointwise.forward(obs_times, obs_values, evaluation_times, obs_mask)

        obs_times, obs_values, obs_mask = FIMImpPoint.preprocess_inputs(obs_times, obs_values, obs_mask)

        fim_imp_pointwise_base_output = shared_fim_imp_pointwise_base(
            {
                "coarse_grid_observation_mask": obs_mask,
                "coarse_grid_noisy_sample_paths": obs_values,
                "coarse_grid_grid": obs_times,
                "fine_grid_grid": evaluation_times,
            },
            training=False,
        )

        vis = fim_imp_pointwise_base_output["visualizations"]

        fim_imp_pointwise_base_output = ImputationConcepts(
            evaluation_times=evaluation_times,
            reconstructed_values=vis["solution"]["learnt"],
            init_cond_mean=vis["init_condition"]["learnt"],
            init_cond_log_std=torch.log(vis["init_condition"]["certainty"]),
            vector_field_mean=vis["drift"]["learnt"],
            vector_field_log_std=torch.log(vis["drift"]["certainty"]),
            normalized=False,
        )

        assert fim_imp_pointwise_output == fim_imp_pointwise_base_output
