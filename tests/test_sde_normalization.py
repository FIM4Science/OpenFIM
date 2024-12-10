from copy import deepcopy

import pytest
import torch
from torch import Tensor

from fim.data.data_generation.dynamical_systems import Degree2Polynomial, DynamicalSystem, Lorenz63System
from fim.data.datasets import FIMSDEDatabatchTuple
from fim.models.sde import NormalizationStats, SDEConcepts


class TestNormalizationStats:
    @pytest.fixture
    def data(self) -> Tensor:
        return torch.tensor(
            [
                [[1, 2], [3, 4], [5, 6]],
                [[-1, -2], [-3, -4], [-5, -6]],
            ],
            dtype=torch.float,
        )

    @pytest.fixture
    def data_stats(self) -> tuple[Tensor]:
        data_min = torch.tensor([[1, 2], [-5, -6]], dtype=torch.float).reshape(2, 1, 2)
        data_max = torch.tensor([[5, 6], [-1, -2]], dtype=torch.float).reshape(2, 1, 2)

        return data_min, data_max

    @pytest.fixture
    def targets(self) -> tuple[float]:
        return -10.0, 100.0

    def test_squash_intermedate_dims(self, data: Tensor) -> None:
        norm_stats = NormalizationStats(data)

        # invariant for dim 3 tensors
        assert data.ndim == 3
        assert torch.all(data == norm_stats.squash_intermediate_dims(data)[0])

        # reshaping for larger dims works as well
        data = torch.repeat_interleave(data.unsqueeze(0), dim=0, repeats=10)
        data = torch.repeat_interleave(data.unsqueeze(0), dim=0, repeats=5)
        assert data.ndim == 5

        reshaped_data, original_shape = norm_stats.squash_intermediate_dims(data)
        assert reshaped_data.ndim == 3
        assert torch.all(data == reshaped_data.reshape(original_shape))

    def test_get_unnormalized_stats(self, data: Tensor, data_stats: tuple[Tensor]) -> None:
        data_min, data_max = data_stats

        norm_stats = NormalizationStats(data)
        min_, max_ = norm_stats.get_unnormalized_stats(data)

        assert torch.allclose(min_, data_min.squeeze(-2))
        assert torch.allclose(max_, data_max.squeeze(-2))

    def test_get_intervals_boundaries(self, data: Tensor, data_stats: tuple[Tensor], targets: tuple[float]) -> None:
        data_min, data_max = data_stats
        target_min, target_max = targets
        norm_stats = NormalizationStats(data, target_min, target_max)

        unnormalized_min, unnormalized_max, normalized_min, normalized_max = norm_stats.get_intervals_boundaries(data.shape)

        assert torch.allclose(unnormalized_min, data_min)
        assert torch.allclose(unnormalized_max, data_max)
        assert torch.allclose(normalized_min, target_min * torch.ones(2, 1, 2))
        assert torch.allclose(normalized_max, target_max * torch.ones(2, 1, 2))

    def test_normalization_map(self, data: Tensor, targets: tuple[Tensor]) -> None:
        target_min, target_max = targets
        norm_stats = NormalizationStats(data, target_min, target_max)

        # reaches its target
        unnormalized_min, unnormalized_max, normalized_min, normalized_max = norm_stats.get_intervals_boundaries(data.shape)
        transformed_data = norm_stats.normalization_map(data)

        assert torch.all(transformed_data.amin(dim=-2, keepdims=True) == target_min)
        assert torch.all(transformed_data.amax(dim=-2, keepdims=True) == target_max)

        # is invertible
        retransformed_data = norm_stats.inverse_normalization_map(transformed_data)
        assert torch.allclose(data, retransformed_data)

        # normalization first derivative is constant
        dummy_input = torch.randn_like(data)
        grad = norm_stats.normalization_map(dummy_input, derivative_num=1)

        expected_value = (normalized_max - normalized_min) / (unnormalized_max - unnormalized_min) * torch.ones_like(grad)
        assert torch.allclose(expected_value, grad)

        # normalization second derivative is zero
        dummy_input = torch.randn_like(data)
        grad_grad = norm_stats.normalization_map(dummy_input, derivative_num=2)

        expected_value = torch.zeros_like(dummy_input)
        assert torch.allclose(expected_value, grad_grad)

        # inverse normalization first derivative is constant
        dummy_input = torch.randn_like(data)
        grad = norm_stats.inverse_normalization_map(dummy_input, derivative_num=1)

        expected_value = (unnormalized_max - unnormalized_min) / (normalized_max - normalized_min) * torch.ones_like(grad)
        assert torch.allclose(expected_value, grad)

        # inverse normalization second derivative is zero
        dummy_input = torch.randn_like(data)
        grad_grad = norm_stats.inverse_normalization_map(dummy_input, derivative_num=2)

        expected_value = torch.zeros_like(dummy_input)
        assert torch.allclose(expected_value, grad_grad)


class TestSDEConceptsBasics:
    obs_vals_shape: tuple = (2, 5, 6)
    obs_times_shape: tuple = (2, 5, 1)
    locations_shape: tuple = (2, 10, 6)

    extended_obs_vals_shape: tuple = (2, 5, 10, 100, 6)
    extended_obs_times_shape: tuple = (2, 5, 10, 100, 1)
    extended_locations_shape: tuple = (2, 50, 2, 6)

    @pytest.fixture
    def data_without_var(self) -> tuple:
        obs_times = torch.randn(self.obs_times_shape)
        obs_vals = torch.randn(self.obs_vals_shape)
        locations = torch.randn(self.locations_shape)

        sde_concepts = SDEConcepts(
            locations=locations,
            drift=torch.randn(self.locations_shape),
            diffusion=torch.randn(self.locations_shape),
        )

        return obs_times, obs_vals, sde_concepts

    @pytest.fixture
    def data_with_var(self) -> tuple:
        obs_times = torch.randn(self.obs_times_shape)
        obs_vals = torch.randn(self.obs_vals_shape)
        locations = torch.randn(self.locations_shape)

        sde_concepts = SDEConcepts(
            locations=locations,
            drift=torch.randn(self.locations_shape),
            diffusion=torch.randn(self.locations_shape),
            log_var_drift=torch.randn(self.locations_shape),
            log_var_diffusion=torch.randn(self.locations_shape),
        )

        return obs_times, obs_vals, sde_concepts

    @pytest.fixture
    def data_with_extended_shapes(self) -> tuple:
        obs_times = torch.randn(self.extended_obs_times_shape)
        obs_vals = torch.randn(self.extended_obs_vals_shape)
        locations = torch.randn(self.extended_locations_shape)

        sde_concepts = SDEConcepts(
            locations=locations,
            drift=torch.randn(self.extended_locations_shape),
            diffusion=torch.randn(self.extended_locations_shape),
            log_var_drift=torch.randn(self.extended_locations_shape),
            log_var_diffusion=torch.randn(self.extended_locations_shape),
        )

        return obs_times, obs_vals, sde_concepts

    def test_from_dbt(self) -> None:
        obs_times = torch.randn(self.obs_times_shape)
        obs_values = torch.randn(self.obs_vals_shape)
        locations = torch.randn(self.locations_shape)
        drift_at_locations = torch.randn(self.locations_shape)
        diffusion_at_locations = torch.randn(self.locations_shape)
        dimension_mask = torch.randn(self.locations_shape)

        dbt = FIMSDEDatabatchTuple(
            obs_times=obs_times,
            obs_values=obs_values,
            locations=locations,
            drift_at_locations=drift_at_locations,
            diffusion_at_locations=diffusion_at_locations,
            dimension_mask=dimension_mask,
        )

        expected_sde_concepts = SDEConcepts(locations=locations, drift=drift_at_locations, diffusion=diffusion_at_locations)
        from_dbt_sde_concepts = SDEConcepts.from_dbt(dbt)

        assert expected_sde_concepts == from_dbt_sde_concepts

        # test None
        assert SDEConcepts.from_dbt(None) is None

        # test only one entry of dbt is None
        dbt_with_none = FIMSDEDatabatchTuple(
            obs_times=obs_times,
            obs_values=obs_values,
            locations=None,
            drift_at_locations=drift_at_locations,
            diffusion_at_locations=diffusion_at_locations,
            dimension_mask=dimension_mask,
        )

        assert SDEConcepts.from_dbt(dbt_with_none) is None

    def test_assert_shapes(self, data_without_var: tuple, data_with_var: tuple) -> None:
        sde_concepts_without_var: SDEConcepts = data_without_var[-1]
        sde_concepts_with_var: SDEConcepts = data_with_var[-1]

        sde_concepts_without_var._assert_shape()
        sde_concepts_with_var._assert_shape()

    def test_any_locations_num_can_be_normalized(self, data_with_extended_shapes: tuple, data_with_var: tuple) -> None:
        # interface to NormalizationStats should work for arbitrary intermediate dimensions
        obs_times, obs_vals, sde_concepts = data_with_extended_shapes

        states_norm_stats = NormalizationStats(obs_vals, normalized_min=-1, normalized_max=1)
        time_norm_stats = NormalizationStats(obs_times, normalized_min=0, normalized_max=1)

        # save sde_concepts for comparison
        original_sde_concepts = deepcopy(sde_concepts)

        sde_concepts.normalize(states_norm_stats, time_norm_stats)
        sde_concepts.renormalize(states_norm_stats, time_norm_stats)

        assert sde_concepts == original_sde_concepts
        assert sde_concepts.locations.ndim == 4
        assert sde_concepts.drift.ndim == 4
        assert sde_concepts.diffusion.ndim == 4

        # if B and D agree, should be able to apply (re)normalization to other shaped locations
        other_sde_concepts = data_with_var[-1]
        other_sde_concepts.normalize(states_norm_stats, time_norm_stats)
        other_sde_concepts.renormalize(states_norm_stats, time_norm_stats)

        assert other_sde_concepts.locations.ndim == 3
        assert other_sde_concepts.drift.ndim == 3
        assert other_sde_concepts.diffusion.ndim == 3

    def test_transforms_are_invertible(self, data_without_var: tuple, data_with_var: tuple) -> None:
        # without var
        obs_times, obs_vals, sde_concepts = data_without_var

        states_norm_stats = NormalizationStats(obs_vals, normalized_min=-1, normalized_max=1)
        time_norm_stats = NormalizationStats(obs_times, normalized_min=0, normalized_max=1)

        # save sde_concepts for comparison
        original_sde_concepts = deepcopy(sde_concepts)

        sde_concepts.normalize(states_norm_stats, time_norm_stats)
        sde_concepts.renormalize(states_norm_stats, time_norm_stats)

        assert sde_concepts == original_sde_concepts

        # with var
        obs_times, obs_vals, sde_concepts = data_with_var

        states_norm_stats = NormalizationStats(obs_vals, normalized_min=-1, normalized_max=1)
        time_norm_stats = NormalizationStats(obs_times, normalized_min=0, normalized_max=1)

        # save sde_concepts for comparison
        original_sde_concepts = deepcopy(sde_concepts)

        sde_concepts.normalize(states_norm_stats, time_norm_stats)
        sde_concepts.renormalize(states_norm_stats, time_norm_stats)

        assert sde_concepts == original_sde_concepts


# Several data generation configs for testing validity of SDEConcept normalization below
DEG_2_POLY_NORMAL_COEFFS_CONFIG = {
    "name": "Degree2Polynomial",
    "data_bulk_name": "damped_linear_theory",
    "redo": True,
    "num_realization": None,
    "state_dim": 3,
    "enforce_positivity": "clip",
    "drift_params": {
        "constant": {
            "distribution": "normal",
            "mean": 0.0,
            "std": 1.0,
        },
        "degree_1": {
            "distribution": "normal",
            "mean": 0.0,
            "std": 1.0,
        },
        "degree_2_squared": {
            "distribution": "normal",
            "mean": 0.0,
            "std": 1.0,
        },
        "degree_2_mixed": {
            "distribution": "normal",
            "mean": 0.0,
            "std": 1.0,
        },
    },
    "diffusion_params": {
        "constant": {
            "distribution": "normal",
            "mean": 0.0,
            "std": 1.0,
        },
        "degree_1": {
            "distribution": "normal",
            "mean": 0.0,
            "std": 1.0,
        },
        "degree_2_squared": {
            "distribution": "normal",
            "mean": 0.0,
            "std": 1.0,
        },
        "degree_2_mixed": {
            "distribution": "normal",
            "mean": 0.0,
            "std": 1.0,
        },
    },
}


DEG_1_POLY_NORMAL_COEFFS_CONFIG = {
    "name": "Degree2Polynomial",
    "data_bulk_name": "damped_linear_theory",
    "redo": True,
    "num_realization": None,
    "state_dim": 3,
    "enforce_positivity": "clip",
    "drift_params": {
        "constant": {
            "distribution": "normal",
            "mean": 0.0,
            "std": 1.0,
        },
        "degree_1": {
            "distribution": "normal",
            "mean": 0.0,
            "std": 1.0,
        },
        "degree_2_squared": {
            "distribution": "fix",
            "fix_value": 0.0,
        },
        "degree_2_mixed": {
            "distribution": "fix",
            "fix_value": 0.0,
        },
    },
    "diffusion_params": {
        "constant": {
            "distribution": "normal",
            "mean": 0.0,
            "std": 1.0,
        },
        "degree_1": {
            "distribution": "normal",
            "mean": 0.0,
            "std": 1.0,
        },
        "degree_2_squared": {
            "distribution": "fix",
            "fix_value": 0.0,
        },
        "degree_2_mixed": {
            "distribution": "fix",
            "fix_value": 0.0,
        },
    },
}

DEG_1_POLY_SMALL_COEFFS_CONFIG = {
    "name": "Degree2Polynomial",
    "data_bulk_name": "damped_linear_theory",
    "redo": True,
    "num_realization": None,
    "state_dim": 3,
    "enforce_positivity": "clip",
    "drift_params": {
        "constant": {
            "distribution": "normal",
            "mean": 0.0,
            "std": 0.01,
        },
        "degree_1": {
            "distribution": "normal",
            "mean": 0.0,
            "std": 0.01,
        },
        "degree_2_squared": {
            "distribution": "fix",
            "fix_value": 0.0,
        },
        "degree_2_mixed": {
            "distribution": "fix",
            "fix_value": 0.0,
        },
    },
    "diffusion_params": {
        "constant": {
            "distribution": "normal",
            "mean": 0.0,
            "std": 0.01,
        },
        "degree_1": {
            "distribution": "normal",
            "mean": 0.0,
            "std": 0.01,
        },
        "degree_2_squared": {
            "distribution": "fix",
            "fix_value": 0.0,
        },
        "degree_2_mixed": {
            "distribution": "fix",
            "fix_value": 0.0,
        },
    },
}

LORENZ_CONFIG = {
    "name": "Lorenz63System",
    "data_bulk_name": "lorenz_theory",
    "redo": True,
    "num_realization": None,
    "observed_dimension": None,
    "drift_params": {
        "sigma": {
            "distribution": "fix",
            "fix_value": 10.0,
        },
        "beta": {
            "distribution": "fix",
            "fix_value": 2.66666666,
        },
        "rho": {
            "distribution": "fix",
            "fix_value": 28.0,
        },
    },
    "diffusion_params": {
        "constant_value": 1.0,
        "dimensions": 3,
    },
}

lorenz_init_state = {
    "distribution": "fix",
    "fix_value": [-8.0, 7.0, 27.0],
    "activation": None,
}

normal_std_1_init_state = {
    "distribution": "normal",
    "mean": 0.0,
    "std_dev": 1.0,
    "activation": None,
}

normal_std_0_1_init_state = {
    "distribution": "normal",
    "mean": 0.0,
    "std_dev": 0.1,
    "activation": None,
}


@pytest.mark.skip(reason="Due to randomness in function sampling and numeric instabilities, some of these will fail.")
class TestSDEConceptsTransformationsValid:
    def _euler_maruyama(
        self,
        drift_fn: callable,
        diffusion_fn: callable,
        initial_states,
        drift_params,
        diffusion_params,
        obs_times,
        dWs,
        num_realizations,
        num_paths,
        num_steps,
    ):
        """
        EM with control over the dWs.
        """
        D = initial_states.shape[-1]

        # solve all paths of all equations in parallel
        initial_states = initial_states.reshape(num_realizations * num_paths, D)
        drift_params = drift_params.reshape(num_realizations * num_paths, *drift_params.shape[2:])
        diffusion_params = diffusion_params.reshape(num_realizations * num_paths, *diffusion_params.shape[2:])
        obs_times = obs_times.reshape(num_realizations * num_paths, num_steps, 1)
        dWs = dWs.reshape(num_realizations * num_paths, num_steps, D)

        # EM steps
        solutions = [initial_states]
        state = solutions[-1]

        for i in range(num_steps - 1):
            dt = obs_times[..., i + 1, :] - obs_times[..., i, :]

            drift = drift_fn(state, obs_times[..., i, :], drift_params)
            diffusion = diffusion_fn(state, obs_times[..., i, :], diffusion_params)

            state = state + drift * dt + diffusion * torch.sqrt(dt) * dWs[..., i, :]
            solutions.append(state)

        solutions = torch.stack(solutions, dim=-2)
        solutions = solutions.reshape(num_realizations, num_paths, num_steps, D)
        return solutions

    @staticmethod
    def _get_normalized_vector_field(
        system: DynamicalSystem,
        drift_or_diffusion: str,
        state_or_time: str,
        normalization_stat: NormalizationStats,
        num_realizations: int,
        num_paths: int,
        num_dimensions: int,
    ) -> callable:
        """
        From a dynamical system, get a vector field that implements the dynamical system in normalized space.
        """
        assert state_or_time in ["state", "time"]
        assert drift_or_diffusion in ["drift", "diffusion"]

        def normalized_vector_field(states, times, params):
            # renormalize state or time to apply unnormalized vector_field
            if state_or_time == "state":
                states = normalization_stat.inverse_normalization_map(states.reshape(num_realizations, num_paths, num_dimensions))
                states = states.reshape(-1, num_dimensions)

            else:
                times = normalization_stat.inverse_normalization_map(times.reshape(num_realizations, num_paths, 1))
                times = times.reshape(-1, 1)

            # apply unnormalized vector_field
            if drift_or_diffusion == "drift":
                vector_field_value = system.drift(states, times, params)
                vector_field_value = vector_field_value.reshape(num_realizations, num_paths, num_dimensions)

                sde_concept = SDEConcepts(
                    locations=torch.zeros_like(vector_field_value),
                    drift=vector_field_value,
                    diffusion=torch.zeros_like(vector_field_value),
                )

            else:
                vector_field_value = system.diffusion(states, times, params)
                vector_field_value = vector_field_value.reshape(num_realizations, num_paths, num_dimensions)

                sde_concept = SDEConcepts(
                    locations=torch.zeros_like(vector_field_value),
                    drift=torch.zeros_like(vector_field_value),
                    diffusion=vector_field_value,
                )

            # use SDEConcepts to normalize vector_field value
            if state_or_time == "state":
                sde_concept._state_transformation(normalization_stat, normalize=True)

            else:
                sde_concept._time_transformation(normalization_stat, normalize=True)

            vector_field_value = sde_concept.drift if drift_or_diffusion == "drift" else sde_concept.diffusion
            vector_field_value = vector_field_value.reshape(-1, num_dimensions)

            return vector_field_value

        return normalized_vector_field

    @pytest.mark.parametrize("num_steps", [10000], ids=lambda num: "num_steps_" + str(num))
    @pytest.mark.parametrize("time_horizon", [10.0], ids=lambda t: "time_horizon_" + str(round(t, 2)))
    @pytest.mark.parametrize("num_paths", [20], ids=lambda num: "num_paths_" + str(num))
    @pytest.mark.parametrize("num_realizations", [50], ids=lambda num: "num_reals_" + str(num))
    @pytest.mark.parametrize(
        "init_state_config",
        [
            lorenz_init_state,
            normal_std_1_init_state,
            normal_std_0_1_init_state,
        ],
        ids=[
            "lorenz_init_state",
            "normal_std_1_init_state",
            "normal_std_0.1_init_state",
        ],
    )
    @pytest.mark.parametrize(
        "as_double",
        [True, False],
        ids=[
            "float64",
            "float32",
        ],
    )
    @pytest.mark.parametrize(
        "system_descr, system_class, system_config",
        [
            ("deg_2_poly_normal_coeffs", Degree2Polynomial, DEG_2_POLY_NORMAL_COEFFS_CONFIG),
            ("deg_1_poly_normal_coeffs", Degree2Polynomial, DEG_1_POLY_NORMAL_COEFFS_CONFIG),
            ("deg_1_poly_0.01_std_coeffs", Degree2Polynomial, DEG_1_POLY_SMALL_COEFFS_CONFIG),
            ("lorenz", Lorenz63System, LORENZ_CONFIG),
        ],
        ids=[
            "deg_2_poly_normal_coeffs",
            "deg_1_poly_normal_coeffs",
            "deg_1_poly_0.01_std_coeffs",
            "lorenz",
        ],
    )
    @pytest.mark.parametrize(
        "transformation",
        [
            "state",
            "time",
        ],
        ids=[
            "state_transf",
            "time_transf",
        ],
    )
    def test_transformation(
        self,
        transformation: str,
        system_descr: str,
        system_class: object,
        system_config: dict,
        init_state_config: dict,
        as_double: float,
        num_realizations: int,
        num_paths: int,
        time_horizon: float,
        num_steps: int,
    ) -> tuple:
        """
        Test validity of (re)normalization transformations on SDEConcepts.
        Solve equations, normalize paths. Then normalize the equation and solve them.
        Control the dWs, then both normalized solutions agree.
        """
        assert transformation in ["state", "time"]

        # construct dynamical system
        system_config.update({"initial_state": init_state_config})
        system_config["num_realizations"] = num_realizations
        system = system_class(system_config)

        # sample equations and initial states
        initial_states = system.sample_initial_states(num_paths * num_realizations)  # [num_paths * num_realizations, D]
        drift_params = system.sample_drift_params(num_realizations)  # [num_realizations, D]
        diffusion_params = system.sample_diffusion_params(num_realizations)  # [num_realizations, D]

        # precision
        if as_double is True:
            initial_states = initial_states.double()
            drift_params = drift_params.double()
            diffusion_params = diffusion_params.double()

        # get solutions in original, unnormalized space
        initial_states = initial_states.reshape(num_realizations, num_paths, -1)  # [num_realizations, num_paths, D]
        drift_params = torch.repeat_interleave(drift_params.unsqueeze(1), dim=1, repeats=num_paths)  # [num_realizations, num_paths, ...]
        diffusion_params = torch.repeat_interleave(
            diffusion_params.unsqueeze(1), dim=1, repeats=num_paths
        )  # [num_realizations, num_paths, ...]

        obs_times = torch.linspace(0, time_horizon, num_steps).reshape(1, 1, -1, 1)
        obs_times = obs_times.expand(num_realizations, num_paths, -1, 1)  # [num_realizations, num_paths, T, 1]

        D = initial_states.shape[-1]
        dWs = torch.randn((num_realizations, num_paths, num_steps, D))  # [num_realizations, num_paths, T, D]

        unnormalized_solution = self._euler_maruyama(
            system.drift,
            system.diffusion,
            initial_states,
            drift_params,
            diffusion_params,
            obs_times,
            dWs,
            num_realizations,
            num_paths,
            num_steps,
        )
        assert unnormalized_solution.ndim == 4, "Expect solutions with 4 dimensions, i.e. shape [B, P, T, D], got " + str(
            unnormalized_solution.ndim
        )

        # already unnormalized solutions might contain Nans
        if torch.isnan(unnormalized_solution).any().item() is True:
            nan_count = torch.isnan(unnormalized_solution).any(dim=(1, 2, 3)).sum()
            assert False, "Found Nan in " + str(nan_count.item()) + " of " + str(num_realizations) + " unnormalized solutions."

        else:
            # apply transformation to inputs
            if transformation == "state":
                norm_stats = NormalizationStats(unnormalized_solution.reshape(num_realizations, -1, D), normalized_min=-1, normalized_max=1)
                initial_states = norm_stats.normalization_map(initial_states)

            else:
                norm_stats = NormalizationStats(obs_times.reshape(num_realizations, -1, 1), normalized_min=0, normalized_max=1)
                obs_times = norm_stats.normalization_map(obs_times.reshape(num_realizations, -1, 1)).reshape(
                    num_realizations, num_paths, -1, 1
                )

            # apply transformation to equations
            normalized_drift = self._get_normalized_vector_field(
                system, "drift", transformation, norm_stats, num_realizations, num_paths, D
            )
            normalized_diffusion = self._get_normalized_vector_field(
                system, "diffusion", transformation, norm_stats, num_realizations, num_paths, D
            )

            # solve transformed equation
            solution_in_normalized_space = self._euler_maruyama(
                normalized_drift,
                normalized_diffusion,
                initial_states,
                drift_params,
                diffusion_params,
                obs_times,
                dWs,
                num_realizations,
                num_paths,
                num_steps,
            )

            # normalize unnormalized_solution to compare against
            if transformation == "state":
                print(unnormalized_solution.shape)
                unnormalized_solution_normalized = norm_stats.normalization_map(unnormalized_solution.reshape(num_realizations, -1, D))
                unnormalized_solution_normalized = unnormalized_solution_normalized.reshape(num_realizations, num_paths, num_steps, D)

            else:  # solution values do not change under time transformation
                unnormalized_solution_normalized = unnormalized_solution

            # compare two solutions
            atol = 1e-3
            rtol = 1e-3

            max_deviation_in_norm_space = torch.amax(torch.abs(solution_in_normalized_space - unnormalized_solution_normalized)).item()
            assert torch.allclose(solution_in_normalized_space, unnormalized_solution_normalized, atol=atol, rtol=rtol, equal_nan=True), (
                "Normalized solutions deviate by max. "
                + str(max_deviation_in_norm_space)
                + ", more than atol="
                + str(atol)
                + ", rtol="
                + str(rtol)
                + "."
            )
