# test why some batches take longer than others
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from fim.data.dataloaders import DataLoaderFactory
from fim.utils.helper import expand_params, load_yaml


if True:
    torch.manual_seed(4)

    config_path = "/home/koerner/FIM/configs/train/fim_ode.yaml"
    # config_path = '/home/koerner/FIM/configs/train/decoderOnly_example.yaml'

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

# check with .pt
if False:
    data_dir ="/home/koerner/FIM/data/torch_2M_ode_chebyshev_max_deg_100_rbf_gp_2_5_and_2_10_length_128_avg_min_8/"
    import torch
    from torch.utils.data import DataLoader, Dataset

    # Custom Dataset
    class CustomDataset(Dataset):
        def __init__(self, data_path):
            self.data = torch.load(data_path)

        def __len__(self):
            return len(self.data["coarse_grid_sample_paths"])

        def __getitem__(self, idx):
            out = {k: (v[0][idx] if isinstance(v, tuple) else v[idx]) for k, v in self.data.items()}
            return out

    # Hyperparameters
    batch_size = 64  # You can set this to whatever value you want

    # Load Datasets
    train_dataset = CustomDataset(data_dir + "train.pt")
    test_dataset = CustomDataset(data_dir + "test.pt")
    val_dataset = CustomDataset(data_dir + "validation.pt")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optional: Print out the number of batches for sanity check
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")

    batch_sizes = []
    times = [time.time()]
    for batch in tqdm(test_loader):
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

# check with .pt


# test dataloader
if True:
    from fim.data.dataloaders import TimeSeriesDataLoader
    path = "/home/koerner/FIM/data/torch_2M_ode_chebyshev_max_deg_100_rbf_gp_2_5_and_2_10_length_128_avg_min_8/"

    batch_size = 1024

    dl = TimeSeriesDataLoader(path=path, batch_size=batch_size, test_batch_size=batch_size)
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
    plt.plot(time_diff)
    plt.title("time differences")
    plt.show()

    print("batch sizes:")
    print("mean: ", np.mean(batch_sizes))
    print("std: ", np.std(batch_sizes))

    print("time diffs:")
    print("mean: ", np.mean(time_diff))
    print("std: ", np.std(time_diff))

    batch_sizes = []
    times = [time.time()]
    for batch in tqdm(dl.test_it):
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
