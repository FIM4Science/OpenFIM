from pathlib import Path

import h5py
import torch


class DataSaver:
    """
    Store the generated data in a file of a specified format.
    """

    def __init__(self, **kwargs) -> None:
        self.process_type = kwargs["process_type"]  # i.e., "hawkes", "mjp"
        self.dataset_name = kwargs["dataset_name"]
        self.num_samples_train = kwargs["num_samples_train"]
        self.num_samples_val = kwargs["num_samples_val"]
        self.num_samples_test = kwargs["num_samples_test"]
        self.storage_format = kwargs["storage_format"]  # i.e., "h5"

        # create dataset path
        self.train_data_path = Path("data/synthetic_data/" + self.process_type + "/" + self.dataset_name + "/train")
        self.val_data_path = Path("data/synthetic_data/" + self.process_type + "/" + self.dataset_name + "/val")
        self.test_data_path = Path("data/synthetic_data/" + self.process_type + "/" + self.dataset_name + "/test")
        if self.num_samples_train > 0:
            self.train_data_path.mkdir(parents=True, exist_ok=True)
        if self.num_samples_val > 0:
            self.val_data_path.mkdir(parents=True, exist_ok=True)
        if self.num_samples_test > 0:
            self.test_data_path.mkdir(parents=True, exist_ok=True)

    def __call__(self, data: dict):
        """
        Store the data.
        """
        # Make sure that all the values have the right number of samples
        for v in data.values():
            if isinstance(v, torch.Tensor):
                assert v.shape[0] == self.num_samples_train + self.num_samples_val + self.num_samples_test

        for k, v in data.items():
            if self.num_samples_train > 0:
                self._save_data(self.train_data_path, k, v[: self.num_samples_train])
            if self.num_samples_val > 0:
                self._save_data(self.val_data_path, k, v[self.num_samples_train : self.num_samples_train + self.num_samples_val])
            if self.num_samples_test > 0:
                self._save_data(self.test_data_path, k, v[self.num_samples_train + self.num_samples_val :])

    def _save_data(self, path: Path, name: str, data):
        """
        Save the data.
        """
        if self.storage_format == "h5":
            self._save_data_h5(path, name, data)
        elif self.storage_format == "torch":
            self._save_data_torch(path, name, data)
        else:
            raise ValueError("Unknown storage format.")

    def _save_data_h5(self, path: Path, name: str, data):
        """
        Save the data in h5 format.
        """
        with h5py.File(path / (name + ".h5"), "w") as f:
            f.create_dataset(name, data=data)

    def _save_data_torch(self, path: Path, name: str, data):
        """
        Save the data in torch format.
        """
        torch.save(data, path / (name + ".pt"))
