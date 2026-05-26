from typing import Any

import einops
import pytest
import torch
from torch import Tensor

from fim.models.imputation_pointwise import (
    StaticWindowing,
    Windowing,
    compress_contiguous_mask,
    get_balanced_windows,
    get_overlapping_window_slices,
    linear_windows_interpolation,
    no_windowing,
    scatter_contiguous_blocks,
)


class BasicWindowing(Windowing):
    """
    A basic implementation that dynamically splits and combines tensors based on the configuration's window count (W).
    """

    def __init__(self, window_count: int):
        self.window_count = window_count

    def get_windows_stats(self, obs_values: Tensor, obs_times: Tensor, obs_mask: Tensor, evaluation_times: Tensor) -> Any:
        _, T, _ = obs_values.shape
        _, G, _ = evaluation_times.shape

        assert T % self.window_count == 0, f"Total observations T ({T}) must be divisible by windows W ({self.window_count})"
        assert G % self.window_count == 0, f"Total evaluation points G ({G}) must be divisible by windows W ({self.window_count})"

        return {"T": T, "G": G, "T//W": T // self.window_count, "G//W": G // self.window_count}

    def split_obs(self, obs: Tensor, windows_stats: dict) -> Tensor:
        # Fixed pattern: Group (w k) on the left side to define how T splits
        return einops.rearrange(obs, "B (w k) D -> B w k D", w=self.window_count, k=windows_stats["T//W"])

    def split_evaluation_times(self, evaluation_times: Tensor, windows_stats: dict) -> Tensor:
        # Fixed typo: self.window_count (removed extra 's') and updated pattern
        return einops.rearrange(evaluation_times, "B (w l) 1 -> B w l 1", w=self.window_count, l=windows_stats["G//W"])

    def combine_evaluations(self, evaluations_windowed: Tensor, windows_stats: dict) -> Tensor:
        # Fixed pattern: Group (W L) on the right side to define how g is reconstructed
        return einops.rearrange(evaluations_windowed, "B W L D -> B (W L) D", W=self.window_count, L=windows_stats["G//W"])


class TestWindowingBaseSplitAndCombine:
    # (B, T, D, G, W)
    SHAPE_CONFIGS = [
        (2, 4, 3, 6, 2),  # Basic setup: 2 windows
        (4, 12, 8, 9, 3),  # Extended setup: 3 windows, larger dimensions
    ]

    @pytest.fixture(params=SHAPE_CONFIGS, ids=["basic_config", "extended_config"])
    def windowing_dataset(self, request) -> tuple:
        """
        Generates multi-dimensional input tensors and initializes the windowing pipeline configuration based on shape variables.
        """
        shapes = request.param
        B, T, D, G, W = shapes

        obs_values = torch.randn(B, T, 1)
        obs_times = einops.repeat(torch.linspace(0, 10, T).view(1, -1, 1), "1 T 1 -> b T 1", b=B)
        obs_mask = torch.zeros(B, T, 1)
        evaluation_times = einops.repeat(torch.linspace(-1, 11, G).view(1, -1, 1), "1 G 1 -> b G 1", b=B)

        # Note: target output dimensions for validation should match G shapes on combine
        reconstructed_values = torch.randn(B, W, G // W, D)
        vector_field_mean = torch.randn(B, W, G // W, D)
        vector_field_log_std = torch.randn(B, W, G // W, D)

        windowing = BasicWindowing(window_count=W)
        windows_stats = windowing.get_windows_stats(obs_values, obs_times, obs_mask, evaluation_times)

        return (
            windowing,
            windows_stats,
            (obs_values, obs_times, obs_mask, evaluation_times),
            (reconstructed_values, vector_field_mean, vector_field_log_std),
            request.param,
        )

    def test_split(self, windowing_dataset: tuple) -> None:
        windowing, stats, inputs, _, shapes = windowing_dataset
        _, T, _, G, W = shapes
        obs_values, obs_times, obs_mask, evaluation_times = inputs

        obs_values_w, obs_times_w, obs_mask_w, evaluation_times_w = windowing.split(
            obs_values, obs_times, obs_mask, evaluation_times, stats
        )

        assert torch.equal(obs_values_w, einops.rearrange(obs_values, "B (w k) D -> B w k D", w=W, k=T // W))
        assert torch.equal(obs_times_w, einops.rearrange(obs_times, "B (w k) 1 -> B w k 1", w=W, k=T // W))
        assert torch.equal(obs_mask_w, einops.rearrange(obs_mask, "B (w k) 1 -> B w k 1", w=W, k=T // W))
        assert torch.equal(evaluation_times_w, einops.rearrange(evaluation_times, "B (w l) 1 -> B w l 1", w=W, l=G // W))

    def test_combine(self, windowing_dataset: tuple) -> None:
        windowing, stats, inputs, outputs, shapes = windowing_dataset
        _, _, _, evaluation_times = inputs
        reconstructed_values, vector_field_mean, vector_field_log_std = outputs

        evaluation_times_split = windowing.split_evaluation_times(evaluation_times, stats)

        evaluation_times_c, reconstructed_values_c, vector_field_mean_c, vector_field_log_std_c = windowing.combine(
            evaluation_times_split, reconstructed_values, vector_field_mean, vector_field_log_std, stats
        )

        assert torch.equal(evaluation_times_c, einops.rearrange(evaluation_times_split, "B W L 1 -> B (W L) 1"))
        assert torch.equal(reconstructed_values_c, einops.rearrange(reconstructed_values, "B W L D -> B (W L) D"))
        assert torch.equal(vector_field_mean_c, einops.rearrange(vector_field_mean, "B W L D -> B (W L) D"))
        assert torch.equal(vector_field_log_std_c, einops.rearrange(vector_field_log_std, "B W L D -> B (W L) D"))

    def test_split_invalid_dimensions(self, windowing_dataset: tuple) -> None:
        windowing, stats, inputs, _, _ = windowing_dataset
        obs_values, obs_times, obs_mask, evaluation_times = inputs

        # Intentionally collapse obs_values into an invalid 2D shape [B, T]
        bad_obs_values = obs_values.squeeze(-1) if obs_values.shape[-1] == 1 else obs_values[:, :, 0]

        # Ensure the base class assert checks step in immediately
        with pytest.raises(AssertionError, match="Got 2"):
            windowing.split(bad_obs_values, obs_times, obs_mask, evaluation_times, stats)


BALANCED_WINDOW_EDGE_CASES = [
    pytest.param(
        {
            "grid_size": 100,
            "windows_count": 4,
            "overlap_percentage": 0.0,
            "exp_window": 25,
            "exp_stride": 25,
        },
        id="no_overlap",
    ),
    pytest.param(
        {
            "grid_size": 50,
            "windows_count": 1,
            "overlap_percentage": 0.0,
            "exp_window": 50,
            "exp_stride": 50,
        },
        id="single_window_no_overlap",
    ),
    pytest.param(
        {
            "grid_size": 50,
            "windows_count": 1,
            "overlap_percentage": 0.5,
            "exp_window": 50,
            "exp_stride": 25,
        },
        id="single_window_with_overlap",
    ),
    pytest.param(
        {
            "grid_size": 100,
            "windows_count": 2,
            "overlap_percentage": 0.5,
            "exp_window": 67,
            "exp_stride": 34,
        },
        id="two_window_with_overlap",
    ),
]


@pytest.mark.parametrize("config", BALANCED_WINDOW_EDGE_CASES)
def test_get_balanced_windows_edge_cases(config: dict) -> None:
    """Validates window and stride size calculations across standard and extreme configurations."""

    window_size, stride_size = get_balanced_windows(
        grid_size=config["grid_size"], windows_count=config["windows_count"], overlap_percentage=config["overlap_percentage"]
    )

    assert window_size == config["exp_window"]
    assert stride_size == config["exp_stride"]


OVERLAPPING_SLICE_CASES = [
    pytest.param(
        {
            "total_length": 100,
            "windows_count": 4,
            "window_size": 25,
            "stride_size": 25,
            "exp_slices": [[0, 25], [25, 50], [50, 75], [75, 100]],
        },
        id="even_split",
    ),
    pytest.param(
        {
            "total_length": 50,
            "windows_count": 1,
            "window_size": 50,
            "stride_size": 50,
            "exp_slices": [[0, 50]],
        },
        id="single_window_boundary",
    ),
    pytest.param(
        {
            "total_length": 100,
            "windows_count": 2,
            "window_size": 67,
            "stride_size": 34,
            "exp_slices": [[0, 67], [33, 100]],
        },
        id="imperfect_split_compensation",
    ),
]


@pytest.mark.parametrize("config", OVERLAPPING_SLICE_CASES)
def test_get_overlapping_window_slices(config: dict) -> None:
    """Ensures index slicing logic generates exact bounds and handles final window alignment."""

    slices = get_overlapping_window_slices(
        total_length=config["total_length"],
        windows_count=config["windows_count"],
        window_size=config["window_size"],
        stride_size=config["stride_size"],
    )

    expected_tensor = torch.tensor(config["exp_slices"], dtype=torch.long)

    assert torch.equal(slices, expected_tensor), f"Expected slices:\n{expected_tensor}\nGot:\n{slices}"

    assert slices[-1, 1] == config["total_length"], "The final window failed to snap to total_length!"


COMPRESS_MASK_CASES = [
    pytest.param(
        {
            "mask": [
                [
                    [1, 1, 0, 0],
                    [0, 1, 1, 0],
                ]
            ],
            "exp_slices": [
                [
                    [0, 2],
                    [1, 3],
                ]
            ],
        },
        id="single_batch_standard_blocks",
    ),
    pytest.param(
        {
            "mask": [
                [
                    [1, 1, 1],
                    [0, 0, 0],
                    [0, 0, 1],
                ]
            ],
            "exp_slices": [[[0, 3], [0, 0], [2, 3]]],
        },
        id="single_batch_full_single_and_empty",
    ),
    pytest.param(
        {
            "mask": [
                [
                    [0, 1, 1, 0],
                    [1, 1, 0, 0],
                ],
                [
                    [1, 1, 0, 0],
                    [0, 1, 1, 1],
                ],
            ],
            "exp_slices": [
                [
                    [1, 3],
                    [0, 2],
                ],
                [
                    [0, 2],
                    [1, 4],
                ],
            ],
        },
        id="two_batches_standard_blocks",
    ),
]


@pytest.mark.parametrize("config", COMPRESS_MASK_CASES)
def test_compress_contiguous_mask(config: dict) -> None:
    """Verifies slice indices extraction from various contiguous boolean configurations."""

    mask_tensor = torch.tensor(config["mask"], dtype=torch.bool)
    expected_tensor = torch.tensor(config["exp_slices"], dtype=torch.long)

    slices = compress_contiguous_mask(mask_tensor)

    assert torch.equal(slices, expected_tensor), f"Expected slices:\n{expected_tensor}\nGot:\n{slices}"


SCATTER_BLOCKS_CASES = [
    pytest.param(
        {
            "original_size": 5,
            "windows": [
                [
                    [[1.1], [1.2], [-1.3]],
                    [[2.1], [2.2], [2.3]],
                ]
            ],
            "original_slices": [
                [
                    [0, 2],
                    [2, 5],
                ]
            ],
            "exp_scattered": [
                [
                    [[1.1], [1.2], [0.0], [0.0], [0.0]],
                    [[0.0], [0.0], [2.1], [2.2], [2.3]],
                ]
            ],
            "exp_mask": [
                [
                    [[1], [1], [0], [0], [0]],
                    [[0], [0], [1], [1], [1]],
                ],
            ],
        },
        id="standard_sequential_scatter",
    ),
    pytest.param(
        {
            "original_size": 4,
            "windows": [
                [
                    [[1.0], [-1.1]],
                    [[-2.0], [-2.1]],
                ]
            ],
            "original_slices": [
                [
                    [1, 2],
                    [0, 0],
                ]
            ],
            "exp_scattered": [
                [
                    [[0.0], [1.0], [0.0], [0.0]],
                    [[0.0], [0.0], [0.0], [0.0]],
                ]
            ],
            "exp_mask": [
                [
                    [[0], [1], [0], [0]],
                    [[0], [0], [0], [0]],
                ]
            ],
        },
        id="zero_length_window",
    ),
    pytest.param(
        {
            "original_size": 4,
            "windows": [
                [
                    [[11.1], [-11.2]],
                    [[12.1], [12.2]],
                ],
                [
                    [[21.1], [21.2]],
                    [[22.1], [-22.2]],
                ],
            ],
            "original_slices": [
                [
                    [0, 1],
                    [1, 3],
                ],
                [
                    [2, 4],
                    [0, 1],
                ],
            ],
            #
            "exp_scattered": [
                [
                    [[11.1], [0.0], [0.0], [0.0]],
                    [[0.0], [12.1], [12.2], [0.0]],
                ],
                [
                    [[0.0], [0.0], [21.1], [21.2]],
                    [[22.1], [0.0], [0.0], [0.0]],
                ],
            ],
            #
            "exp_mask": [
                [
                    [[1], [0], [0], [0]],
                    [[0], [1], [1], [0]],
                ],
                [
                    [[0], [0], [1], [1]],
                    [[1], [0], [0], [0]],
                ],
            ],
        },
        id="multi_batch_independent_scatter",
    ),
]


@pytest.mark.parametrize("config", SCATTER_BLOCKS_CASES)
def test_scatter_contiguous_blocks(config: dict) -> None:
    """Validates data reconstruction mapping back onto the global coordinate grid."""

    windows = torch.tensor(config["windows"], dtype=torch.float32)
    slices = torch.tensor(config["original_slices"], dtype=torch.long)

    expected_scattered = torch.tensor(config["exp_scattered"], dtype=torch.float32)
    expected_mask = torch.tensor(config["exp_mask"], dtype=torch.float32)

    scat_out, mask_out = scatter_contiguous_blocks(windows=windows, original_slices=slices, original_size=config["original_size"])

    assert torch.allclose(scat_out, expected_scattered), f"Scattered values mismatch!\nGot:\n{scat_out}"
    assert torch.allclose(mask_out, expected_mask), f"Scattered masks mismatch!\nGot:\n{mask_out}"


INVERTIBLE_SCATTER = [
    pytest.param(
        {
            "original_size": 6,
            "mask": [
                [
                    [1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0],
                ],
                [
                    [1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 0],
                ],
            ],
        },
        id="multi_batch_overlapping_windows",
    ),
    pytest.param(
        {
            "original_size": 4,
            "mask": [
                [
                    [1, 1, 1, 1],
                    [0, 0, 0, 0],
                ]
            ],
        },
        id="full_and_empty_window_inversion",
    ),
]


@pytest.mark.parametrize("config", INVERTIBLE_SCATTER)
def test_mask_compression_and_scatter_are_inverses(config: dict) -> None:
    """Verifies that compressing a mask and scattering it back yields the original mask layout."""

    original_mask = torch.tensor(config["mask"], dtype=torch.bool)
    G = config["original_size"]

    slices = compress_contiguous_mask(original_mask)

    B, W, _ = slices.shape
    max_len = torch.max(slices[:, :, 1] - slices[:, :, 0]).item()
    dummy_windows = torch.ones((B, W, max_len, 1), dtype=torch.float32)

    _, scattered_mask = scatter_contiguous_blocks(windows=dummy_windows, original_slices=slices, original_size=G)

    expected_mask = original_mask.to(torch.float32).unsqueeze(-1)

    assert torch.equal(scattered_mask, expected_mask), (
        f"Round-trip failed!\nOriginal Mask:\n{expected_mask.squeeze(-1)}\nReconstructed Scattered Mask:\n{scattered_mask.squeeze(-1)}"
    )


LINEAR_INTERPOLATION_CASES = [
    pytest.param(
        {
            "windows_scattered": [
                [
                    [[10.0], [10.0], [9.0], [8.0], [7.0], [-1.0]],
                    [[-1.0], [2.0], [3.0], [4.0], [5.0], [5.0]],
                ]
            ],
            "windows_mask": [
                [
                    [[1], [1], [1], [1], [1], [0]],
                    [[0], [1], [1], [1], [1], [1]],
                ],
            ],
            "exp_combined": [
                [[10.0], [8.4], [6.6], [5.6], [5.4], [5.0]],
            ],
        },
        id="two_window_long_overlap",
    ),
    pytest.param(
        {
            "windows_scattered": [
                [
                    [[1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0]],
                    [[-2.0], [2.0], [2.0], [2.0], [2.0], [-2.0]],
                    [[-3.0], [-3.0], [-3.0], [3.0], [3.0], [3.0]],
                ]
            ],
            "windows_mask": [
                [
                    [[1], [1], [1], [0], [0], [0]],
                    [[0], [1], [1], [1], [1], [0]],
                    [[0], [0], [0], [1], [1], [1]],
                ],
            ],
            "exp_combined": [
                [[1.0], [1 + 1 / 3], [1 + 2 / 3], [2 + 1 / 3], [2 + 2 / 3], [3.0]],
            ],
        },
        id="three_windows",
    ),
    pytest.param(
        {
            "windows_scattered": [
                [
                    [
                        [1.0, 10.0],
                        [1.0, 10.0],
                        [1.0, 10.0],
                        [-1.0, -1.0],
                    ],
                    [
                        [-4.0, -4.0],
                        [4.0, 40.0],
                        [4.0, 40.0],
                        [4.0, 40.0],
                    ],
                ]
            ],
            "windows_mask": [
                [
                    [[1], [1], [1], [0]],
                    [[0], [1], [1], [1]],
                ],
            ],
            "exp_combined": [
                [
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                    [4.0, 40.0],
                ],
            ],
        },
        id="multi_dimensional_features",
    ),
    pytest.param(
        {
            "windows_scattered": [
                [
                    [[1], [1], [1], [-1]],
                    [[-4], [4], [4], [4]],
                ]
            ],
            "windows_mask": [
                [
                    [[1], [1], [1], [0]],
                    [[0], [1], [1], [1]],
                ],
            ],
            "exp_combined": [
                [[1.0], [2.0], [3.0], [4.0]],
            ],
        },
        id="two_window_with_overlap",
    ),
    pytest.param(
        {
            "windows_scattered": [
                [
                    [[1], [1], [-1], [-1]],
                    [[-2], [-2], [2], [2]],
                ]
            ],
            "windows_mask": [
                [
                    [[1], [1], [0], [0]],
                    [[0], [0], [1], [1]],
                ]
            ],
            "exp_combined": [
                [[1.0], [1.0], [2.0], [2.0]],
            ],
        },
        id="no_overlap",
    ),
    pytest.param(
        {
            "windows_scattered": [
                [
                    [[1], [1], [1], [-1]],
                    [[-4], [4], [4], [4]],
                ],
                [
                    [[1], [1], [1], [1]],
                    [[-2], [-2], [-2], [-2]],
                ],
            ],
            "windows_mask": [
                [
                    [[1], [1], [1], [0]],
                    [[0], [1], [1], [1]],
                ],
                [
                    [[1], [1], [1], [1]],
                    [[0], [0], [0], [0]],
                ],
            ],
            "exp_combined": [
                [[1.0], [2.0], [3.0], [4.0]],
                [[1.0], [1.0], [1.0], [1.0]],
            ],
        },
        id="multi_batch",
    ),
    pytest.param(
        {
            "windows_scattered": [
                [
                    [[1], [-11], [-1], [-1]],
                    [[-2], [-2], [2], [2]],
                ]
            ],
            "windows_mask": [
                [
                    [[1], [0], [0], [0]],
                    [[0], [0], [1], [1]],
                ]
            ],
            "exp_combined": [
                [[1.0], [0.0], [2.0], [2.0]],
            ],
        },
        id="with_gap",
    ),
]


@pytest.mark.parametrize("config", LINEAR_INTERPOLATION_CASES)
def test_linear_windows_interpolation(config: dict) -> None:
    """Validates that overlapping regions blend values via linear ramp weights."""

    scat = torch.tensor(config["windows_scattered"], dtype=torch.float32)
    mask = torch.tensor(config["windows_mask"], dtype=torch.float32)
    expected = torch.tensor(config["exp_combined"], dtype=torch.float32)

    combined = linear_windows_interpolation(scat, mask)

    assert combined.shape == expected.shape, f"Shape mismatch! Expected {expected.shape}, got {combined.shape}"
    assert torch.allclose(combined, expected, atol=1e-4), f"Interpolated math mismatch!\nGot:\n{combined}"


class TestStaticWindowingMethods:
    # Configurations covering: (B, T, D, G, windows_count, overlap_percentage)
    STATIC_CONFIGS = [
        pytest.param((3, 20, 4, 15, 3, 0.2), id="multi_batch_multi_dim_with_overlap"),
        pytest.param((2, 10, 1, 8, 1, 0.0), id="single_window_no_overlap"),
        pytest.param((2, 10, 2, 8, 1, 0.5), id="single_window_with_overlap"),
        pytest.param((5, 12, 16, 24, 4, 0.0), id="large_batch_high_dim_no_overlap"),
        pytest.param((1, 4069, 1, 2048, 32, 0.2), id="many_times_and_windows"),
    ]

    @pytest.fixture(params=STATIC_CONFIGS)
    def static_windowing_setup(self, request) -> tuple:
        """
        Sets up input tensors and structural windowing stats for a clean
        StaticWindowing evaluation.
        """

        B, T, D, G, W, overlap = request.param

        windowing = StaticWindowing(windows_count=W, overlap_percentage=overlap)

        obs_times = torch.arange(T, dtype=torch.float32).view(1, T, 1).repeat(B, 1, 1)
        obs_values = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, 1)

        evaluation_times = torch.linspace(-1, T + 1, steps=G).view(1, G, 1).repeat(B, 1, 1)

        stats = windowing.get_windows_stats(obs_values, obs_times, obs_mask, evaluation_times)

        return windowing, stats, obs_values, evaluation_times, D, G, W

    def test_split_obs(self, static_windowing_setup: tuple) -> None:
        """
        Verifies that split_obs cuts the global observation tensor into the
        expected window dimension and handles single-window paths safely.
        """

        windowing, stats, obs_values, _, D, _, W = static_windowing_setup
        B = obs_values.shape[0]
        K = stats.obs_windows_size

        obs_windowed = windowing.split_obs(obs_values, stats)

        assert obs_windowed.shape == (B, W, K, D)

        if W == 1:
            assert torch.equal(obs_windowed[:, 0], obs_values)
        else:
            assert not torch.all(obs_windowed == 0.0)  # some sensible windows are created

    def test_split_combine_inverse(self, static_windowing_setup: tuple) -> None:
        """
        Verifies that splitting and combining the evaluation_times yields again the evaluation_times.
        """

        windowing, stats, _, evaluation_times, _, _, _ = static_windowing_setup

        evaluation_times_split = windowing.split_evaluation_times(evaluation_times, stats)
        evaluation_times_combined = windowing.combine_evaluations(evaluation_times_split, stats)

        print(evaluation_times_combined.squeeze())

        assert torch.allclose(evaluation_times, evaluation_times_combined, atol=1e-3), (
            f"Max deviation {torch.amax(torch.abs(evaluation_times_combined - evaluation_times))}"
        )

    def test_split_evaluation_times_properties(self, static_windowing_setup: tuple) -> None:
        """
        Verifies that split_evaluation_times creates correct window coordinates
        without replicating the underlying slicing loops inside the test logic.
        """

        windowing, stats, _, evaluation_times, _, _, W = static_windowing_setup
        B = evaluation_times.shape[0]
        L = stats.max_eval_window_size

        # Override evaluation_times with a strict sequence [0, 1, 2, ...] per batch, tracking splitting and combining it easily
        G = evaluation_times.shape[1]
        evaluation_times = torch.arange(G, dtype=torch.float32).view(1, G, 1).repeat(B, 1, 1)
        global_max = G - 1
        global_min = 0.0

        eval_times_windowed = windowing.split_evaluation_times(evaluation_times, stats)

        assert eval_times_windowed.shape == (B, W, L, 1)

        if W == 1:
            assert torch.equal(eval_times_windowed[:, 0], evaluation_times)
            return

        for b in range(B):
            for w in range(W):
                window_slice = eval_times_windowed[b, w, :, 0]  # Shape [L]

                assert torch.all(window_slice >= global_min), "Values fell below global timeline boundary!"
                assert torch.all(window_slice <= global_max), "Values exceeded global timeline boundary!"

                diffs = torch.diff(window_slice)
                assert torch.all(diffs >= 0.0), f"Window {w} in batch {b} contains broken time sequences!"

                # Because evaluation_times[b, i, 0] == i, the value itself IS the original index!
                # If the values are [4.0, 5.0, 6.0], their differences must be exactly 1.0.
                valid_elements = window_slice[window_slice < global_max]
                if len(valid_elements) > 1:
                    assert torch.all(torch.diff(valid_elements) == 1.0), f"Window {w} in batch {b} has gaps or scrambled indices!"


@pytest.mark.parametrize(
    "sizes",  # (B, T, D, G)
    [
        pytest.param((1, 90, 3, 5), id="single_batch"),
        pytest.param((3, 90, 3, 5), id="multiple_batches"),
        pytest.param((3, 90, 1, 5), id="single_dimension"),
        pytest.param((4, 30, 8, 100), id="more_evals_than_obs"),
    ],
)
def test_no_windowing(sizes: tuple) -> None:
    """
    Verify that `no_windowing` implementation creates single window.
    """

    B, T, D, G = sizes

    obs_times = torch.arange(T, dtype=torch.float32).view(1, T, 1).repeat(B, 1, 1)
    obs_values = torch.randn(B, T, D)
    obs_mask = torch.zeros(B, T, 1)

    evaluation_times = torch.linspace(-1, T + 1, steps=G).view(1, G, 1).repeat(B, 1, 1)

    stats = no_windowing.get_windows_stats(obs_values, obs_times, obs_mask, evaluation_times)

    split_obs_values = no_windowing.split_obs(obs_values, stats)
    split_evaluation_times = no_windowing.split_evaluation_times(evaluation_times, stats)

    assert torch.allclose(obs_values.view(B, 1, T, D), split_obs_values)
    assert torch.allclose(evaluation_times.view(B, 1, G, 1), split_evaluation_times)
