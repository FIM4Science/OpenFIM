import torch

from fim.data.utils import split_into_variable_windows


# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long


def test_split_into_variable_windows():
    x = torch.arange(60).view(1, 60, 1).float()
    min_window_percentage = 0.1
    max_window_percentage = 0.3
    window_count = 5
    overlap = 0.2
    max_sequence_length = 60
    padding_value = 1

    windows, (overlap_size, padding_size_windowing_end) = split_into_variable_windows(
        x, min_window_percentage, max_window_percentage, window_count, overlap, max_sequence_length, padding_value
    )

    assert windows is not None
    assert isinstance(windows, torch.Tensor)
    assert windows.size(0) == window_count
    assert windows.size(2) == 1
    assert overlap_size > 0
    assert padding_size_windowing_end >= 0

    # Check if the first window has the correct size
    variable_window_size = windows[0].size(1) - overlap_size
    assert int(min_window_percentage * max_sequence_length) <= variable_window_size <= int(max_window_percentage * max_sequence_length)

    # Check if the remaining windows have the correct size
    fixed_window_size = (max_sequence_length - variable_window_size) // (window_count - 1)
    for i in range(1, window_count):
        assert windows[i].size(1) == fixed_window_size + overlap_size


def test_split_into_variable_windows_no_overlap():
    x = torch.arange(256).view(1, 256, 1).float()

    window_count = 5
    imputation_window_index = 3
    max_sequence_length = 256

    imputation_window_size = int(0.11 * max_sequence_length)
    windows = split_into_variable_windows(x, imputation_window_size, imputation_window_index, window_count, max_sequence_length)

    assert windows is not None
    assert isinstance(windows, torch.Tensor)
    assert windows.size(0) == window_count
    assert windows.size(2) == 1
