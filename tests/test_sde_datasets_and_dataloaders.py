from copy import deepcopy
from pathlib import Path

import optree
import torch

from fim.data.datasets import append_to_lists_in_dict, get_file_paths, get_subdict, pad_data_in_dict, shuffle_at_dim


class TestDatasetHelpers:
    def test_get_file_paths(self):
        dir_paths = ["path_1/subdir_1/", "path_2/subdir_2"]
        file_names = {"a": "a.h5", "b": "b.h5"}

        expected_result = {
            "a": [Path("path_1/subdir_1/a.h5"), Path("path_2/subdir_2/a.h5")],
            "b": [Path("path_1/subdir_1/b.h5"), Path("path_2/subdir_2/b.h5")],
        }

        result = get_file_paths(dir_paths, file_names)

        assert result == expected_result, f"Got {result}, expected {expected_result}"

    def test_get_subdict(self):
        d = {"a": "a", "b": "b", "c": "c"}
        keys = ["b", "c", "d"]

        expected_result = {"b": "b", "c": "c"}

        result = get_subdict(d, keys)

        assert result == expected_result, f"Got {result}, expected {expected_result}"

    def test_pad_data_in_dict(self):
        data = {
            "a": [torch.randn(2, 3, 4), torch.randn(2, 5, 4)],
            "b": [torch.randn(2, 3, 1)],
            "c": [torch.randn(2, 5, 1)],
        }

        keys_to_pad = ["a", "b"]

        # dim = -2
        padded_data = pad_data_in_dict(deepcopy(data), keys_to_pad, dim=-2)
        flattened_padded_tree = optree.tree_flatten(padded_data)[0]
        padded_shapes = [t.shape for t in flattened_padded_tree]

        expected_shapes = [(2, 5, 4), (2, 5, 4), (2, 5, 1), (2, 5, 1)]

        assert padded_shapes == expected_shapes, f"Got shapes {padded_shapes}."

        # dim = -1
        padded_data = pad_data_in_dict(deepcopy(data), keys_to_pad, dim=-1)
        flattened_padded_tree = optree.tree_flatten(padded_data)[0]
        padded_shapes = [t.shape for t in flattened_padded_tree]

        expected_shapes = [(2, 3, 4), (2, 5, 4), (2, 3, 4), (2, 5, 1)]

        assert padded_shapes == expected_shapes, f"Got shapes {padded_shapes}."

        # passing max_length
        padded_data = pad_data_in_dict(deepcopy(data), keys_to_pad, dim=-2, max_length=10)
        flattened_padded_tree = optree.tree_flatten(padded_data)[0]
        padded_shapes = [t.shape for t in flattened_padded_tree]

        expected_shapes = [(2, 10, 4), (2, 10, 4), (2, 10, 1), (2, 5, 1)]

        assert padded_shapes == expected_shapes, f"Got shapes {padded_shapes}."

    def test_shuffle_at_dim(self):
        # shuffle along dim 1
        arange_ = torch.arange(10)
        batched_arange_a = arange_.reshape(1, -1, 1).expand(5, 10, 2)
        batched_arange_b = arange_.reshape(1, -1, 1).expand(5, 10, 4)

        data = {
            "a": [batched_arange_a, batched_arange_a],
            "b": [batched_arange_b, batched_arange_b],
        }

        shuffled_data = shuffle_at_dim(deepcopy(data), dim=1)

        # reverse shuffle manually
        argsort_0 = torch.argsort(shuffled_data["a"][0], dim=1)
        argsort_1 = torch.argsort(shuffled_data["a"][1], dim=1)

        reverse_shuffled_data = {
            "a": [
                torch.take_along_dim(shuffled_data["a"][0], argsort_0, dim=1),
                torch.take_along_dim(shuffled_data["a"][1], argsort_1, dim=1),
            ],
            "b": [
                torch.take_along_dim(shuffled_data["b"][0], argsort_0[..., 0][..., None].expand(-1, -1, 4), dim=1),
                torch.take_along_dim(shuffled_data["b"][1], argsort_1[..., 0][..., None].expand(-1, -1, 4), dim=1),
            ],
        }

        for key in ["a", "b"]:
            for ind in [0, 1]:
                assert torch.allclose(
                    reverse_shuffled_data[key][ind], data[key][ind]
                ), f"Got {reverse_shuffled_data} for key {key} and ind {ind}."

        # shuffle along dim 0
        arange_ = torch.arange(5)
        batched_arange_a = arange_.reshape(-1, 1, 1).expand(5, 10, 2)
        batched_arange_b = arange_.reshape(-1, 1, 1).expand(5, 10, 4)

        data = {
            "a": [batched_arange_a, batched_arange_a],
            "b": [batched_arange_b, batched_arange_b],
        }

        shuffled_data = shuffle_at_dim(deepcopy(data), dim=1)

        # reverse shuffle manually
        argsort_0 = torch.argsort(shuffled_data["a"][0], dim=0)
        argsort_1 = torch.argsort(shuffled_data["a"][1], dim=0)

        reverse_shuffled_data = {
            "a": [
                torch.take_along_dim(shuffled_data["a"][0], argsort_0, dim=0),
                torch.take_along_dim(shuffled_data["a"][1], argsort_1, dim=0),
            ],
            "b": [
                torch.take_along_dim(shuffled_data["b"][0], argsort_0[..., 0][..., None].expand(-1, -1, 4), dim=0),
                torch.take_along_dim(shuffled_data["b"][1], argsort_1[..., 0][..., None].expand(-1, -1, 4), dim=0),
            ],
        }

        for key in ["a", "b"]:
            for ind in [0, 1]:
                assert torch.allclose(
                    reverse_shuffled_data[key][ind], data[key][ind]
                ), f"Got {reverse_shuffled_data} for key {key} and ind {ind}."

    def test_append_to_lists_in_dict(self):
        d = {
            "a": [0, 1],
            "b": [0, 1],
            "c": [0],
        }
        to_append = {"a": 2, "c": 1}
        expected_result = {
            "a": [0, 1, 2],
            "b": [0, 1],
            "c": [0, 1],
        }

        result = append_to_lists_in_dict(d, to_append)

        assert result == expected_result, f"Got {result}, expected {expected_result}"
