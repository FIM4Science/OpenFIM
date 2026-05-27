from copy import deepcopy

import pytest
import torch

from fim.models.imputation_pointwise import ImputationConcepts
from fim.models.sde import MinMaxNormalization, Standardization


class TestImputationConceptsBasics:
    SHAPE_CONFIGS = [
        (2, 5, 6, 10, 0),
        (3, 10, 17, 3, 4),  # (B, T, D, G, W)
    ]

    @pytest.fixture(params=SHAPE_CONFIGS, ids=["basic_grid", "extended_grid"])
    def imputation_dataset(self, request) -> tuple:
        """
        Generates random observation data and a filled ImputationConcepts
        instance based on the parametrized dimension constraints.
        """
        B, T, D, G, W = request.param

        obs_times_shape = (B, T, 1) if W == 0 else (B, W, T, 1)
        obs_values_shape = (B, T, D) if W == 0 else (B, W, T, D)
        init_shape = (B, D) if W == 0 else (B, W, D)
        eval_times_shape = (B, G, 1) if W == 0 else (B, W, G, 1)
        vfs_shape = (B, G, D) if W == 0 else (B, W, G, D)

        obs_times = torch.randn(obs_times_shape)
        obs_values = torch.randn(obs_values_shape)

        imputation_concepts = ImputationConcepts(
            evaluation_times=torch.randn(eval_times_shape),
            reconstructed_values=torch.randn(vfs_shape),
            init_cond_mean=torch.randn(init_shape),
            init_cond_log_std=torch.randn(init_shape),
            vector_field_mean=torch.randn(vfs_shape),
            vector_field_log_std=torch.randn(vfs_shape),
        )
        return obs_times, obs_values, imputation_concepts

    def test_assert_shapes(self, imputation_dataset: tuple) -> None:
        _, _, imputation_concepts = imputation_dataset
        imputation_concepts._assert_shape()

    def test_equality_and_allclose_wrapper(self, imputation_dataset: tuple) -> None:
        _, _, concepts_1 = imputation_dataset
        concepts_2 = deepcopy(concepts_1)

        assert concepts_1 == concepts_2

        concepts_2.reconstructed_values += 1e-7
        assert concepts_1 == concepts_2

        concepts_2.reconstructed_values += 1e-3
        assert concepts_1 != concepts_2

    def test_normalization_and_invertibility(self, imputation_dataset: tuple) -> None:
        obs_times, obs_vals, imputation_concepts = imputation_dataset

        values_norm = MinMaxNormalization(normalized_min=-1, normalized_max=1)
        values_norm_stats = values_norm.get_norm_stats(obs_vals)

        times_norm = MinMaxNormalization(normalized_min=0, normalized_max=1)
        times_norm_stats = times_norm.get_norm_stats(obs_times)

        original_concepts = deepcopy(imputation_concepts)

        imputation_concepts.normalize(values_norm, values_norm_stats, times_norm, times_norm_stats)
        assert imputation_concepts.normalized is True

        imputation_concepts.renormalize(values_norm, values_norm_stats, times_norm, times_norm_stats)
        assert imputation_concepts.normalized is False

        assert imputation_concepts == original_concepts


class TestImputationConceptsTransformationsValid:
    @pytest.mark.parametrize(
        "instance_normalization_class",
        [MinMaxNormalization, Standardization],
        ids=["min_max_normalization", "standardization"],
    )
    @pytest.mark.parametrize(
        "transformation",
        ["value", "time"],
        ids=["value_transf", "time_transf"],
    )
    @pytest.mark.parametrize(
        "as_double",
        [True, False],
        ids=["float64", "float32"],
    )
    def test_vector_field_transformation_validity(
        self,
        transformation: str,
        instance_normalization_class: object,
        as_double: bool,
    ) -> None:
        """
        Validates the algebraic correctness of ImputationConcepts transformations.
        Generates a synthetic analytical curve, applies the forward transformations,
        and reconstructs the trajectory via a deterministic integration step to ensure
        numerical and mathematical alignment.
        """
        num_realizations = 10
        num_steps = 5000  # High resolution minimizes Euler discretization error
        num_dimensions = 3
        time_horizon = 10.0

        # Create clean trajectory with analytical derivative: x(t) = 5 * sin(t) -> dx/dt = 5 * cos(t)
        obs_times = torch.linspace(0, time_horizon, num_steps).reshape(1, -1, 1)
        obs_times = obs_times.expand(num_realizations, -1, 1)

        unnormalized_values = 5 * torch.sin(obs_times).repeat(1, 1, num_dimensions)
        unnormalized_vf_mean = 5 * torch.cos(obs_times).repeat(1, 1, num_dimensions)
        unnormalized_vf_log_std = 2 * torch.log(torch.abs(torch.sin(obs_times)) + 0.1).repeat(1, 1, num_dimensions)

        if as_double:
            obs_times = obs_times.double()
            unnormalized_values = unnormalized_values.double()
            unnormalized_vf_mean = unnormalized_vf_mean.double()
            unnormalized_vf_log_std = unnormalized_vf_log_std.double()

        # Fit and apply normalization parameters
        if transformation == "value":
            norm = instance_normalization_class(normalized_min=-1, normalized_max=1)
            norm_stats = norm.get_norm_stats(unnormalized_values)

            integration_times = obs_times.clone()
            initial_state_transformed = norm.normalization_map(unnormalized_values[:, 0, :], norm_stats)
            expected_normalized_values = norm.normalization_map(unnormalized_values, norm_stats)

        else:
            norm = instance_normalization_class(normalized_min=0, normalized_max=1)
            norm_stats = norm.get_norm_stats(obs_times)

            integration_times = norm.normalization_map(obs_times, norm_stats)
            initial_state_transformed = unnormalized_values[:, 0, :].clone()
            expected_normalized_values = unnormalized_values

        # Build unnormalized ImputationConcepts container
        concepts = ImputationConcepts(
            evaluation_times=obs_times.clone(),
            reconstructed_values=unnormalized_values.clone(),
            init_cond_mean=unnormalized_values[:, 0, :].clone(),
            init_cond_log_std=torch.zeros_like(unnormalized_values[:, 0, :]),
            vector_field_mean=unnormalized_vf_mean.clone(),
            vector_field_log_std=unnormalized_vf_log_std.clone(),
        )

        # Execute Forward Normalization Map
        if transformation == "value":
            concepts._values_transformation(norm, norm_stats, normalize=True)
        else:
            concepts._times_transformation(norm, norm_stats, normalize=True)

        # Compute numerical integral via Euler step over the transformed fields
        reconstructed_normalized_values = [initial_state_transformed]
        current_state = initial_state_transformed

        for i in range(num_steps - 1):
            dt = integration_times[:, i + 1, :] - integration_times[:, i, :]
            v_mean = concepts.vector_field_mean[:, i, :]

            # Linear step integration
            current_state = current_state + v_mean * dt
            reconstructed_normalized_values.append(current_state)

        reconstructed_normalized_values = torch.stack(reconstructed_normalized_values, dim=1)

        # Check path integration alignment (Direction 1 vs Direction 2)
        atol = 1e-2
        rtol = 1e-2
        assert torch.allclose(reconstructed_normalized_values, expected_normalized_values, atol=atol, rtol=rtol), (
            f"Integrated normalized path deviates from directly transformed path under {transformation} transformation. "
            f"Max deviation: {torch.max(torch.abs(reconstructed_normalized_values - expected_normalized_values))}"
        )

        assert concepts.vector_field_log_std is not None

        # Self-consistency check via inverse processing (Invertibility Verification)
        concepts_backed = deepcopy(concepts)
        if transformation == "value":
            concepts_backed._values_transformation(norm, norm_stats, normalize=False)
            assert torch.allclose(concepts_backed.vector_field_log_std, unnormalized_vf_log_std, atol=1e-5, rtol=1e-5)
        else:
            concepts_backed._times_transformation(norm, norm_stats, normalize=False)
            assert torch.allclose(concepts_backed.vector_field_log_std, unnormalized_vf_log_std, atol=1e-5, rtol=1e-5)
