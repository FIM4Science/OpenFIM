# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# test why some batches take longer than others
import sys
import time
from itertools import islice

import numpy as np
import pytest
import torch
from datasets import DownloadMode
from torch.utils.data import DataLoader
from tqdm import tqdm

from fim import test_data_path
from fim.data.dataloaders import BaseDataLoader, DataLoaderFactory
from fim.utils.helper import expand_params, load_yaml


@pytest.mark.skip("Skip until we have a proper dataset")
class TestBaseDataLoader:
    path = "monash_tsf"
    name = "nn5_daily"
    out_fields = ["target"]
    batch_size = 2
    dataset_kwargs = {DownloadMode.FORCE_REDOWNLOAD: True}

    @pytest.fixture(scope="module")
    def dataloader(self):
        return BaseDataLoader(
            path=self.path, ds_name=self.name, batch_size=self.batch_size, test_batch_size=8, output_fields=self.out_fields
        )

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

        assert x["seq_len"].shape[0] == self.batch_size

        assert x["target"].shape[-1] == 1

        assert x["target"].shape[1] == x["seq_len"].max()


class TestFimDataLoader:
    @pytest.fixture
    def config(self):
        root_path = (
            test_data_path
            / "data"
            / "mjp"
            / "5k_hom_mjp_4_st_10s_1%_noise_reg_300-samples-per-intensity_upscaled_with_initial_distribution"
        )
        config = {
            "name": "FIMDataLoader",
            "path_collections": {"train": [root_path / "train"] * 2, "test": [root_path / "test"]},
            "loader_kwargs": {"num_workers": 16, "batch_size": 1, "test_batch_size": 2},
            # "split": "train"
            "dataset_kwargs": {
                "files_to_load": {
                    "observation_grid": "fine_grid_grid.pt",
                    "observation_values": "fine_grid_noisy_sample_paths.pt",
                    "mask_seq_lengths": "fine_grid_mask_seq_lengths.pt",
                    "time_normalization_factors": "fine_grid_time_normalization_factors.pt",
                    "intensity_matrices": "fine_grid_intensity_matrices.pt",
                    "adjacency_matrices": "fine_grid_adjacency_matrices.pt",
                    "initial_distributions": "fine_grid_initial_distributions.pt",
                },
                "data_limit": None,
            },
        }
        return config

    def test_load(self, config: dict):
        dataloader = DataLoaderFactory.create(**config)
        assert dataloader is not None
        assert dataloader.train_set_size == 800
        assert dataloader.test_set_size == 200

    def test_variable_num_of_paths(self, config: dict):
        config["loader_kwargs"]["max_path_count"] = 300
        config["loader_kwargs"]["max_number_of_minibatch_sizes"] = 10
        config["loader_kwargs"]["variable_num_of_paths"] = True
        config["loader_kwargs"]["batch_size"] = 32
        config["loader_kwargs"]["num_workers"] = 0
        dataloader = DataLoaderFactory.create(**config)
        assert dataloader is not None
        assert dataloader.train_set_size == 800
        assert dataloader.test_set_size == 200
        for _ in range(2):
            for ix, batch in enumerate(islice(dataloader.train_it, 10)):
                assert batch["observation_values"].shape[1] == ix * 30 + 1

    @pytest.mark.skip("Skip until we have a proper dataset")
    @pytest.mark.parametrize(
        "config_path", ["/home/koerner/FIM/configs/train/fim_ode.yaml", "/home/koerner/FIM/configs/train/decoderOnly_example.yaml"]
    )
    def test_loading_old(self, config_path):
        torch.manual_seed(4)

        config_inference = load_yaml(config_path)
        gs_configs_inference = expand_params(config_inference)

        dataloader = DataLoaderFactory.create(**gs_configs_inference[0].dataset.to_dict())
        batch_sizes = []
        times = [time.time()]
        for batch in tqdm(dataloader.train_it):
            batch_sizes.append(sys.getsizeof(batch))
            times.append(time.time())

        times = np.array(times)
        time_diff = times[1:] - times[:-1]

        import matplotlib.pyplot as plt

        plt.hist(batch_sizes)
        plt.title("batch sizes")
        plt.show()

        plt.hist(time_diff)
        plt.title("time differences")
        plt.show()
        plt.plot(time_diff)
        plt.title("time differences")
        plt.show()

        print("batch sizes:")
        print("mean: ", np.mean(batch_sizes))
        print("std: ", np.std(batch_sizes))

        print("time diffs:")
        print("mean: ", np.mean(time_diff))
        print("std: ", np.std(time_diff))

        # @pytest.mark.parametrize(
        #     "data_dir", ["/home/koerner/FIM/data/torch_2M_ode_chebyshev_max_deg_100_rbf_gp_2_5_and_2_10_length_128_avg_min_8/"]
        # )
        # def test_loading_pt(self, data_dir):
        #     import torch
        #     from torch.utils.data import DataLoader, Dataset

        #     # Custom Dataset
        #     class CustomDataset(Dataset):
        #         def __init__(self, data_path):
        #             self.data = torch.load(data_path)

        #         def __len__(self):
        #             return len(self.data["coarse_grid_sample_paths"])

        #         def __getitem__(self, idx):
        #             out = {k: (v[0][idx] if isinstance(v, tuple) else v[idx]) for k, v in self.data.items()}
        #             return out

        #     # Hyperparameters
        #     batch_size = 64  # You can set this to whatever value you want

        #     # Load Datasets
        #     train_dataset = CustomDataset(data_dir + "train.pt")
        #     test_dataset = CustomDataset(data_dir + "test.pt")
        #     val_dataset = CustomDataset(data_dir + "validation.pt")

        #     # DataLoaders
        #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        #     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        #     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        #     # Optional: Print out the number of batches for sanity check
        #     print(f"Number of batches in train_loader: {len(train_loader)}")
        #     print(f"Number of batches in val_loader: {len(val_loader)}")
        #     print(f"Number of batches in test_loader: {len(test_loader)}")

        #     batch_sizes = []
        #     times = [time.time()]
        #     for batch in tqdm(test_loader):
        #         batch_sizes.append(sys.getsizeof(batch))
        #         times.append(time.time())

        #     times = np.array(times)
        #     time_diff = times[1:] - times[:-1]

        #     import matplotlib.pyplot as plt

        #     plt.hist(batch_sizes)
        #     plt.title("batch sizes")
        #     plt.show()

        #     plt.hist(time_diff)
        #     plt.title("time differences")
        #     plt.show()
        #     plt.plot(time_diff)
        #     plt.title("time differences")
        #     plt.show()

        #     print("batch sizes:")
        #     print("mean: ", np.mean(batch_sizes))
        #     print("std: ", np.std(batch_sizes))

        #     print("time diffs:")
        #     print("mean: ", np.mean(time_diff))
        #     print("std: ", np.std(time_diff))

        # @pytest.mark.parametrize(
        #     "path", ["/home/koerner/FIM/data/torch_2M_ode_chebyshev_max_deg_100_rbf_gp_2_5_and_2_10_length_128_avg_min_8/"]
        # )
        # def test_dataloader(self, path):
        from fim.data.dataloaders import TimeSeriesDataLoaderTorch

        batch_size = 1024

        dl = TimeSeriesDataLoaderTorch(path=None, batch_size=batch_size, test_batch_size=batch_size)
        print(dl)
        batch_sizes = []
        times = [time.time()]
        for batch in tqdm(dl.train_it):
            batch_sizes.append(sys.getsizeof(batch))
            times.append(time.time())

        times = np.array(times)
        time_diff = times[1:] - times[:-1]

        import matplotlib.pyplot as plt

        plt.hist(batch_sizes)
        plt.title("batch sizes")
        plt.show()

        plt.hist(time_diff)
        plt.title("time differences")
        plt.show()

        print("batch sizes:")
        print("mean: ", np.mean(batch_sizes))
        print("std: ", np.std(batch_sizes))

        print("time diffs:")
        print("mean: ", np.mean(time_diff))
        print("std: ", np.std(time_diff))

        print("test")
        batch_sizes = []
        times = [time.time()]
        for batch in tqdm(dl.test_it):
            batch_sizes.append(sys.getsizeof(batch))
            times.append(time.time())

        times = np.array(times)
        time_diff = times[1:] - times[:-1]

        plt.hist(batch_sizes)
        plt.title("batch sizes")
        plt.show()

        plt.hist(time_diff)
        plt.title("time differences")
        plt.show()
        plt.plot(time_diff)
        plt.title("time differences")
        plt.show()

        print("batch sizes:")
        print("mean: ", np.mean(batch_sizes))
        print("std: ", np.std(batch_sizes))

        print("time diffs:")
        print("mean: ", np.mean(time_diff))
        print("std: ", np.std(time_diff))
