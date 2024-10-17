# pylint: disable=line-too-long
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring


import pytest
from datasets import Dataset, DownloadMode

from fim import test_data_path
from fim.data.datasets import FimDataset, HFDataset


class TestHFDataset:
    dataset_name = "nn5_daily"

    @pytest.fixture(scope="module")
    def test_dataset(self):
        return HFDataset(path="monash_tsf", ds_name=self.dataset_name, split="test", download_mode=DownloadMode.FORCE_REDOWNLOAD)

    def test_init(self):
        dataset = HFDataset(path="monash_tsf", ds_name=self.dataset_name)
        assert dataset.data is not None
        assert isinstance(dataset.data, Dataset)
        print(dataset)

    def test_init_with_optional_arguments(self):
        dataset = HFDataset(path="monash_tsf", ds_name=self.dataset_name, split="test")
        assert dataset.data is not None
        assert isinstance(dataset.data, Dataset)

    def test_init_with_invalid_split(self):
        with pytest.raises(ValueError):
            HFDataset(path="monash_tsf", ds_name=self.dataset_name, split="invalid_split")

    def test_init_with_download_mode(self):
        dataset = HFDataset(path="monash_tsf", ds_name=self.dataset_name, download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS, split="test")
        assert dataset.data is not None

        dataset = HFDataset(path="monash_tsf", ds_name=self.dataset_name, download_mode=DownloadMode.FORCE_REDOWNLOAD, split="test")
        assert dataset.data is not None

    def test_get_item(self):
        dataset = HFDataset(path="monash_tsf", ds_name="nn5_daily", split="test")
        assert dataset[0] is not None
        assert dataset[0].keys() == {"item_id", "target", "start", "feat_static_cat", "feat_dynamic_real", "seq_len"}


class TestFimDataset:
    def test_init(self):
        dataset = FimDataset(path=test_data_path / "data" / "mjp" / "train")
        assert dataset is not None
        assert isinstance(dataset.data, dict)
        assert dataset.data is not None
        assert "fine_grid_grid" in dataset.data
        assert len(dataset.data) == 9

    def test_get_item(self):
        dataset = FimDataset(path=test_data_path / "data" / "mjp" / "train")
        assert dataset[0] is not None
        assert dataset[0].keys() == {
            "fine_grid_grid",
            "fine_grid_masks",
            "fine_grid_adjacency_matrices",
            "fine_grid_initial_distributions",
            "fine_grid_intensity_matrices",
            "fine_grid_mask_seq_lengths",
            "fine_grid_noisy_sample_paths",
            "fine_grid_sample_paths",
            "fine_grid_time_normalization_factors",
        }
        assert len(dataset) == 2
        assert dataset[0]["fine_grid_grid"].shape == (300, 100, 1)
