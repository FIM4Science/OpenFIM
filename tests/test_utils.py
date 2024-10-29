# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import torch

from fim.data.utils import get_path_counts, split_into_variable_windows
from fim.models.utils import create_matrix_from_off_diagonal, create_padding_mask, get_off_diagonal_elements


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


def test_get_off_diagonal_elements():
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    expected = torch.tensor([2, 3, 4, 6, 7, 8], dtype=torch.float)
    result = get_off_diagonal_elements(matrix)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"


def test_get_off_diagonal_elements_batch():
    matrix = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]]], dtype=torch.float)
    expected = torch.tensor([[2, 3, 4, 6, 7, 8], [8, 7, 6, 4, 3, 2]], dtype=torch.float)
    result = get_off_diagonal_elements(matrix)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"


def test_get_off_diagonal_elements_single_element():
    matrix = torch.tensor([[1]], dtype=torch.float)
    expected = torch.tensor([], dtype=torch.float)
    result = get_off_diagonal_elements(matrix)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"


def test_get_off_diagonal_elements_non_square():
    matrix = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
    try:
        get_off_diagonal_elements(matrix)
    except AssertionError as e:
        assert str(e) == "The last two dimensions of the matrix must be square."
    else:
        assert False, "Expected an AssertionError, but none was raised"


def test_create_matrix_from_off_diagonal():
    off_diagonal_elements = torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.float)
    size = 3
    diagonal_value = 1.0
    expected = torch.tensor([[1, 2, 3], [4, 1, 5], [6, 7, 1]], dtype=torch.float)
    result = create_matrix_from_off_diagonal(off_diagonal_elements, size, diagonal_value)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"


def test_create_matrix_from_off_diagonal_batch():
    off_diagonal_elements = torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]], dtype=torch.float)
    size = 3
    diagonal_value = 0.0
    expected = torch.tensor([[[0, 2, 3], [4, 0, 5], [6, 7, 0]], [[0, 8, 9], [10, 0, 11], [12, 13, 0]]], dtype=torch.float)
    result = create_matrix_from_off_diagonal(off_diagonal_elements, size, diagonal_value)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"


def test_create_matrix_from_off_diagonal_single_element():
    off_diagonal_elements = torch.tensor([], dtype=torch.float)
    size = 1
    diagonal_value = 5.0
    expected = torch.tensor([[5]], dtype=torch.float)
    result = create_matrix_from_off_diagonal(off_diagonal_elements, size, diagonal_value)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"


def test_create_matrix_from_off_diagonal_non_square():
    off_diagonal_elements = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float)
    size = 4
    diagonal_value = 0.0
    try:
        create_matrix_from_off_diagonal(off_diagonal_elements, size, diagonal_value)
    except AssertionError as e:
        assert str(e) == "Number of off-diagonal elements does not match the expected size."
    else:
        assert False, "Expected an AssertionError, but none was raised"


def test_create_matrix_from_off_diagonal_sum_row():
    off_diagonal_elements = torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.float)
    size = 3
    diagonal_value = 1.0
    mode = "sum_row"
    expected = torch.tensor([[5, 2, 3], [4, 9, 5], [6, 7, 13]], dtype=torch.float)
    result = create_matrix_from_off_diagonal(off_diagonal_elements, size, diagonal_value, mode)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"


def test_create_padding_mask():
    mask_seq_lengths = torch.tensor([3, 5, 2])
    seq_len = 6
    expected = torch.tensor(
        [[False, False, False, True, True, True], [False, False, False, False, False, True], [False, False, True, True, True, True]]
    )
    result = create_padding_mask(mask_seq_lengths, seq_len)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"


def test_create_padding_mask_all_zeros():
    mask_seq_lengths = torch.tensor([0, 0, 0])
    seq_len = 4
    expected = torch.tensor([[True, True, True, True], [True, True, True, True], [True, True, True, True]])
    result = create_padding_mask(mask_seq_lengths, seq_len)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"


def test_create_padding_mask_all_full():
    mask_seq_lengths = torch.tensor([4, 4, 4])
    seq_len = 4
    expected = torch.tensor([[False, False, False, False], [False, False, False, False], [False, False, False, False]])
    result = create_padding_mask(mask_seq_lengths, seq_len)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"


def test_create_padding_mask_varied_lengths():
    mask_seq_lengths = torch.tensor([1, 2, 3, 4])
    seq_len = 4
    expected = torch.tensor(
        [[False, True, True, True], [False, False, True, True], [False, False, False, True], [False, False, False, False]]
    )
    result = create_padding_mask(mask_seq_lengths, seq_len)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"


def test_get_path_counts_basic():
    num_examples = 100
    minibatch_size = 10
    max_path_count = 30
    expected = torch.tensor([1, 4, 7, 10, 13, 16, 19, 22, 25, 28])
    result = get_path_counts(num_examples, minibatch_size, max_path_count)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"


def test_get_path_counts_with_remainder():
    num_examples = 105
    minibatch_size = 10
    max_path_count = 30
    expected = torch.tensor([1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 30])
    result = get_path_counts(num_examples, minibatch_size, max_path_count)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"


def test_get_path_counts_small_minibatch():
    num_examples = 100
    minibatch_size = 5
    max_path_count = 30
    expected = torch.tensor([1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28])
    result = get_path_counts(num_examples, minibatch_size, max_path_count)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"


def test_get_path_counts_not_enough_minibatches():
    num_examples = 10
    minibatch_size = 10
    max_path_count = 30
    try:
        get_path_counts(num_examples, minibatch_size, max_path_count)
    except ValueError as e:
        assert str(e) == "Not enough minibatches to distribute paths evenly. We have not implemented this case yet."
    else:
        assert False, "Expected a ValueError, but none was raised"


def test_get_path_counts_fill_paths_with_last_size():
    num_examples = 356
    minibatch_size = 32
    max_path_count = 300
    expected = torch.tensor([1, 31, 61, 91, 121, 151, 181, 211, 241, 271, 300, 300])
    result = get_path_counts(num_examples, minibatch_size, max_path_count)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"
