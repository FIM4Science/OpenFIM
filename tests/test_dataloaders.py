# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
import pytest
from datasets import DownloadMode
from fim.data.dataloaders import BaseDataLoader, BaseDataset
from torch.utils.data import DataLoader


class TestBaseDataLoader:
    path = "monash_tsf"
    name = "cif_2016"
    out_fields = ["target", "time_feat"]
    batch_size = 2

    @pytest.fixture(scope="module")
    def dataloader(self):
        BaseDataset(path="monash_tsf", name="cif_2016", split="test", download_mode=DownloadMode.FORCE_REDOWNLOAD)
        return BaseDataLoader(path=self.path, name=self.name, batch_size=self.batch_size, test_batch_size=8, output_fields=self.out_fields)

    def test_init(self, dataloader):
        assert dataloader.batch_size == self.batch_size
        assert dataloader.test_batch_size == 8
        assert dataloader.dataset_kwargs == {}
        assert dataloader.iter.keys() == {"train", "validation", "test"}
        assert dataloader.path == self.path
        assert dataloader.name == self.name
        assert dataloader.split is None
        assert len(dataloader.dataset) == 3
        assert "train" in dataloader.dataset
        # assert isinstance(dataloader.dataset["train"], BaseDataset)

    def test_train_it(self, dataloader):
        assert isinstance(dataloader.train_it, DataLoader)

    def test_validation_it(self, dataloader):
        assert isinstance(dataloader.validation_it, DataLoader)

    def test_test_it(self, dataloader):
        assert isinstance(dataloader.test_it, DataLoader)

    def test_n_train_batches(self, dataloader):
        assert dataloader.n_train_batches == len(dataloader.train_it)

    def test_n_validation_batches(self, dataloader):
        assert dataloader.n_validation_batches == len(dataloader.validation_it)

    def test_n_test_batches(self, dataloader):
        assert dataloader.n_test_batches == len(dataloader.test_it)

    def test_train_set_size(self, dataloader):
        assert dataloader.train_set_size == len(dataloader.train)

    def test_validation_set_size(self, dataloader):
        assert dataloader.validation_set_size == len(dataloader.validation)

    def test_test_set_size(self, dataloader):
        assert dataloader.test_set_size == len(dataloader.test)

    def test_getitem(self, dataloader):
        x = next(iter(dataloader.train_it))

        assert x is not None
        assert x.keys() == set(self.out_fields + ["seq_len"])
        assert x["target"].shape[0] == self.batch_size
        assert x["time_feat"].shape[0] == self.batch_size
        assert x["seq_len"].shape[0] == self.batch_size
        assert x["time_feat"].shape[-1] == 2
        assert x["target"].shape[-1] == 1
        assert x["target"].shape[1] == x["time_feat"].shape[1]
        assert x["target"].shape[1] == x["seq_len"].max()
