import torch

from fim.data.utils import split_into_variable_windows


# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long


def test_split_into_variable_windows():
    x = torch.arange(60).view(1, 60, 1).float()
    imputation_window_size = 20
    imputation_window_index = 3
    window_count = 5
    max_sequence_length = 60
    padding_value = 1

    windows = split_into_variable_windows(
        x, imputation_window_size, imputation_window_index, window_count, max_sequence_length, padding_value
    )

    assert windows is not None
    assert isinstance(windows, torch.Tensor)
    assert windows.size(0) == window_count
    assert windows.size(2) == 1


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
