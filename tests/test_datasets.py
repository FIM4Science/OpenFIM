# pylint: disable=line-too-long
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring


import pytest
from datasets import Dataset, DownloadMode

from fim.data.datasets import BaseDataset


class TestBaseTsDataset:
    @pytest.fixture(scope="module")
    def test_dataset(self):
        return BaseDataset(path="monash_tsf", ds_name="cif_2016", split="test", download_mode=DownloadMode.FORCE_REDOWNLOAD)

    def test_init(self):
        dataset = BaseDataset(path="monash_tsf", ds_name="cif_2016")
        assert dataset.data is not None
        assert isinstance(dataset.data, Dataset)
        print(dataset)

    def test_init_with_optional_arguments(self):
        dataset = BaseDataset(path="monash_tsf", ds_name="cif_2016", split="test")
        assert dataset.data is not None
        assert isinstance(dataset.data, Dataset)

    def test_init_with_invalid_split(self):
        with pytest.raises(ValueError):
            BaseDataset(path="monash_tsf", ds_name="cif_2016", split="invalid_split")

    def test_init_with_download_mode(self):
        dataset = BaseDataset(path="monash_tsf", ds_name="cif_2016", download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS, split="test")
        assert dataset.data is not None

        dataset = BaseDataset(path="monash_tsf", ds_name="cif_2016", download_mode=DownloadMode.FORCE_REDOWNLOAD, split="test")
        assert dataset.data is not None

    def test_get_item(self):
        dataset = BaseDataset(path="monash_tsf", ds_name="cif_2016", split="test")
        assert dataset[0] is not None
        assert dataset[0].keys() == {"item_id", "target", "start", "feat_static_cat", "feat_dynamic_real", "seq_len"}
