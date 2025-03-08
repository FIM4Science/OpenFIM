# pylint: disable=line-too-long
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring


import pytest
from datasets import Dataset, DownloadMode

from fim import test_data_path
from fim.data.datasets import FIMDataset, HFDataset


class TestHFDataset:
    dataset_name = "nn5_daily"

    def test_init(self):
        dataset = HFDataset(path="monash_tsf", conf=self.dataset_name, trust_remote_code=True)
        assert dataset.data is not None
        assert isinstance(dataset.data, Dataset)
        print(dataset)

    def test_rename_column(self):
        new_column_names = {"time_since_last_event": "delta_time", "type_event": "event_type"}
        dataset = HFDataset(path="easytpp/volcano", trust_remote_code=True, rename_columns=new_column_names)
        columns = dataset.data.column_names
        assert "delta_time" in columns
        assert "event_type" in columns
        assert "time_since_last_event" not in columns
        assert "type_event" not in columns
        assert "seq_len" in columns

    def test_output_columns(self):
        output_columns = ["dim_process", "seq_idx", "seq_len"]
        dataset = HFDataset(path="easytpp/volcano", output_columns=output_columns, trust_remote_code=True)
        columns = dataset.data.column_names
        for column in output_columns:
            assert column in columns
        assert "time_since_last_event" not in columns
        assert "type_event" not in columns

    def test_init_with_optional_arguments(self):
        dataset = HFDataset(path="monash_tsf", conf=self.dataset_name, split="test", trust_remote_code=True)
        assert dataset.data is not None
        assert isinstance(dataset.data, Dataset)

    def test_init_with_invalid_split(self):
        with pytest.raises(ValueError):
            HFDataset(path="monash_tsf", conf=self.dataset_name, split="invalid_split", trust_remote_code=True)

    def test_init_with_download_mode(self):
        dataset = HFDataset(path="monash_tsf", conf=self.dataset_name, download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS, split="test")
        assert dataset.data is not None

        dataset = HFDataset(path="monash_tsf", conf=self.dataset_name, download_mode=DownloadMode.FORCE_REDOWNLOAD, split="test")
        assert dataset.data is not None

    def test_get_item(self):
        dataset = HFDataset(path="easytpp/volcano", split="test")
        assert dataset[0] is not None
        assert dataset[0].keys() == {"dim_process", "seq_idx", "seq_len", "time_since_last_event", "time_since_start", "type_event"}


class TestFimDataset:
    def test_init(self):
        dataset = FIMDataset(
            path=test_data_path
            / "data"
            / "mjp"
            / "5k_hom_mjp_4_st_10s_1%_noise_reg_300-samples-per-intensity_upscaled_with_initial_distribution/"
            / "train"
        )
        assert dataset is not None
        assert isinstance(dataset.data, dict)
        assert dataset.data is not None
        assert "fine_grid_grid" in dataset.data
        assert len(dataset.data) == 9
        assert len(dataset) == 400

    def test_init_list_of_paths(self):
        dataset = FIMDataset(
            path=[
                test_data_path
                / "data"
                / "mjp"
                / "5k_hom_mjp_4_st_10s_1%_noise_reg_300-samples-per-intensity_upscaled_with_initial_distribution"
                / "train",
                test_data_path
                / "data"
                / "mjp"
                / "25k_hom_mjp_6_st_10s_1%_noise_rand_300-samples-per-intensity_with_initial_distribution"
                / "test",
            ]
        )
        assert dataset is not None
        assert isinstance(dataset.data, dict)
        assert dataset.data is not None
        assert "fine_grid_grid" in dataset.data
        assert len(dataset.data) == 9
        assert len(dataset) == 800

    def test_get_item(self):
        dataset = FIMDataset(
            path=test_data_path
            / "data"
            / "mjp"
            / "5k_hom_mjp_4_st_10s_1%_noise_reg_300-samples-per-intensity_upscaled_with_initial_distribution"
            / "train"
        )
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
        assert len(dataset) == 400
        assert dataset[0]["fine_grid_grid"].shape == (300, 100, 1)
