import copy

import torch
import os
import pickle
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn.decomposition import PCA
from tqdm import tqdm

from typing import Literal

from utils.plot import plot_3d_paths, plot_2d_paths
from utils.helpers import plot_vector_field_and_trajectories
from utils.data_models import trajectory

# most of the implementation is gathered from
# https://github.com/hegdepashupati/gaussian-process-odes/blob/00443f515ce51f92d3d4423c832599b00289d16d/src/datasets/mocap.py#L30



class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return (x * self.std) + self.mean


class Data:
    def __init__(self, ys, ts, container_type="numpy"):
        if container_type == "numpy":
            self.ts = ts.astype(np.float32)
            self.ys = ys.astype(np.float32)
        elif container_type == "torch":
            self.ts = ts.to(dtype=torch.float32)
            self.ys = ys.to(dtype=torch.float32)
        else:
            raise ValueError("container_type must be either 'numpy' or 'torch'")

    def __len__(self):
        return self.ys.shape[0]

    def __getitem__(self, index):
        return self.ys[index, ...], self.ts

    def to_torch(self, device='cpu'):
        return Data(torch.from_numpy(self.ys).to(device=device),
                    torch.from_numpy(self.ts).to(device=device),
                    container_type="torch")

    def to_numpy(self):
        return Data(self.ys.cpu().numpy(),
                    self.ts.cpu().numpy(),
                    container_type="numpy")


class MocapDataset(object):
    def __init__(self, data_path="data/ode/mocap/",
                 subject="09",
                 dt=0.01,
                 pca_components=-1,
                 seqlen=50,
                 data_normalize=False,      # Why False???
                 pca_normalize=True):

        self.data_path = data_path
        self.dt = dt
        self.pca_components = pca_components
        self.data_normalize = data_normalize
        self.pca_normalize = pca_normalize

        self.seqlen = seqlen
        self.subject = subject

        assert subject in ["09", "35", "39"], "Wrong subject passed"
        fname = os.path.join(data_path, "mocap" + subject + ".npz") # e.g. "data/ode/mocap/mocap09.npz"
        mocap_data = np.load(fname) # <class 'numpy.lib.npyio.NpzFile'>

        xs_test = mocap_data["test"] # <class 'numpy.ndarray'>, shape (2,120,50) for mocap09.npz
        ts_test = dt * np.arange(0, xs_test.shape[1]) # [0.00 0.01 ... 1.19] for mocap09.npz

        xs_valid = mocap_data["validation"] # shape (2,120,50) for mocap09
        ts_valid = dt * np.arange(0, xs_valid.shape[1])

        xs_train = mocap_data["train"] # shape (6,120,50) for mocap09
        ts_train = dt * np.arange(0, xs_train.shape[1])

        # replace problematic zeros in some sensor dimensions by 1e-6
        xs_train = self.treat_zero_readings(xs_train)
        xs_valid = self.treat_zero_readings(xs_valid)
        xs_test = self.treat_zero_readings(xs_test)

        # compute data statistics (std, mean). Possibly normalize to zero mean, standard variance
        self.data_std = xs_train[:, :].std((0, 1), keepdims=True) + 1e-5    # (1,1,50)
        self.data_mean = xs_train[:, :].mean((0, 1), keepdims=True)         # (1,1,50)
        if data_normalize:
            # Save the normalization statistics
            self.data_normalize = Normalize(self.data_mean, self.data_std)
            # Perform normalization on trajectories (still 50-dimensional after this)
            xs_train, xs_valid, xs_test = self.data_normalize(xs_train), self.data_normalize(
                xs_valid), self.data_normalize(xs_test)
        else:
            self.data_normalize = None

        # Now actually do PCA
        if pca_components > 0:
            xs_train = self.build_pca(xs_train, train=True)
            xs_valid = self.build_pca(xs_valid, train=False)
            xs_test = self.build_pca(xs_test, train=False)

        if pca_normalize:
            pca_m = xs_train[:, :].mean((0, 1), keepdims=True)          # This is almost zero anyways (order of magnitued: e-15)    Shape (1,1,5)
            pca_s = xs_train[:, :].std((0, 1), keepdims=True) + 1e-5    # This is a lot bigger and actually necessary               Shape (1,1,5)
            self.pca_normalize = Normalize(pca_m, pca_s)                # Normalization object for post-PCA
            xs_train, xs_valid, xs_test = self.pca_normalize(xs_train), self.pca_normalize(
                xs_valid), self.pca_normalize(xs_test)
        else:
            self.pca_normalize = None

        # We create Data objects encoding the train, validation, and test trajectories. Notice that for training, we truncate at the seqlen-th observation
        if seqlen is None:
            print("seqlen is None!")
        self.trn = Data(ys=xs_train[:, :seqlen], ts=ts_train[:seqlen])
        self.val = Data(ys=xs_valid, ts=ts_valid)
        self.tst = Data(ys=xs_test, ts=ts_test)

    def treat_zero_readings(self, data):
        data[:, :, (24, 25, 31, 32)] = 1e-6
        return data

    def build_pca(self, x, train=False):
        N, T, D = x.shape   # (6,120,50) for mocap09 short training sequences
        x_stacked = np.vstack([x[i] for i in range(x.shape[0])])    # (N*T,D) = (720,50)
        if train:
            self.pca = PCA(n_components=self.pca_components)
            x_ = self.pca.fit_transform(x_stacked)
        else:
            x_ = self.pca.transform(x_stacked)
        x_ = np.concatenate([np.expand_dims(x_[i * T:(i + 1) * T], 0) for i in range(N)], 0)
        print(self.pca.explained_variance_ratio_)
        return x_

    def to_torch(self, device="cpu"):
        d = copy.deepcopy(self)

        d.trn = d.trn.to_torch(device=device)
        d.val = d.val.to_torch(device=device)
        d.tst = d.tst.to_torch(device=device)

        return d

    def to_numpy(self):
        d = copy.deepcopy(self)

        d.trn = d.trn.to_numpy()
        d.val = d.val.to_numpy()
        d.tst = d.tst.to_numpy()

        return d

    @staticmethod
    def load_from_pickle(path: Path) -> 'MocapDataset':
        with open(path, 'rb') as f:
            loaded = pickle.load(f)

        return loaded


class MocapDownloader:
    BASE_DATA_URL = "https://raw.githubusercontent.com/hegdepashupati/gaussian-process-odes/c084729817e09cb3910b45ec268eb4688e0a44f8/data/mocap"
    FILE_NAMES = ["mocap09.npz", "mocap35.npz", "mocap39.npz"]

    def __init__(self, data_path="data/ode/mocap/"):
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)

    def _get_raw_url(self, filename):
        return f"{self.BASE_DATA_URL}/{filename}"

    def _file_exists(self, filename):
        filepath = os.path.join(self.data_path, filename)
        return os.path.exists(filepath) and os.path.getsize(filepath) > 0

    def download(self):

        for filename in self.FILE_NAMES:
            if self._file_exists(filename):
                print(f"File {filename} already exists, skipping download.")
                continue

            url = self._get_raw_url(filename)
            filepath = os.path.join(self.data_path, filename)

            print(f"Downloading {filename}...")
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))

                with open(filepath, 'wb') as f:
                    if total_size == 0:
                        f.write(response.content)
                    else:
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))

                print(f"Successfully downloaded {filename}")

            except requests.exceptions.RequestException as e:
                print(f"Error downloading {filename}: {e}")
                # Clean up partial download
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise

    def download_if_needed(self):
        missing_files = [f for f in self.FILE_NAMES if not self._file_exists(f)]

        if not missing_files:
            print("All mocap data files are already downloaded.")
            return

        print(f"Missing files: {missing_files}")
        self.download()


class FimMocapDatasetBuilder:
    PCA_COMPONENTS = 3
    MOCAP_DATA_PATH = Path('data/ode/mocap/')
    DT = 0.1

    def build_train_dataset_for(self, subject: Literal["09", "35", "39"], seqlen: int):
        downloader = MocapDownloader()
        downloader.download_if_needed()

        data_pca = MocapDataset(data_path=str(self.MOCAP_DATA_PATH), subject=subject,
                                pca_components=self.PCA_COMPONENTS,
                                data_normalize=False, pca_normalize=True,
                                dt=self.DT, seqlen=seqlen)

        self._save_dataset(data_pca,
                           save_path=self.MOCAP_DATA_PATH / subject / f"seqlen{seqlen}",
                           save_dataset_object_additionally=True)

        return data_pca
    
    def _save_training_data(self, dataset: MocapDataset, save_path: Path, save_dataset_object_additionally: bool = False):
        train_traj = dataset.trn.ys[np.newaxis, ...]
        num_traj = train_traj.shape[1]
        ts_train = dataset.trn.ts[np.newaxis, np.newaxis, :, np.newaxis].repeat(num_traj, axis=1)
        mask = np.ones((*train_traj.shape[:-1], 1), dtype=bool)

        save_path.mkdir(parents=True, exist_ok=True)
        self._save_h5(save_path / "obs_times.h5", ts_train)
        self._save_h5(save_path / "obs_values.h5", train_traj)
        self._save_h5(save_path / "obs_mask.h5", mask)

        # hack for now -- In trajectory training, the locations don't matter, we just need to pass some for it to work.
        self._save_h5(save_path / "locations.h5", train_traj[0, :, :2, :])

        if save_dataset_object_additionally:
            with open(save_path / "mocap_dataset.pkl", 'wb') as f:
                pickle.dump(dataset, f)
    
    def _save_validation_data(self, dataset: MocapDataset, save_path: Path, save_dataset_object_additionally: bool = False):
        val_traj = dataset.val.ys[np.newaxis, ...]
        num_traj = val_traj.shape[1]
        ts_val = dataset.val.ts[np.newaxis, np.newaxis, :, np.newaxis].repeat(num_traj, axis=1)
        mask = np.ones((*val_traj.shape[:-1], 1), dtype=bool)

        save_path.mkdir(parents=True, exist_ok=True)
        self._save_h5(save_path / "obs_times.h5", ts_val)
        self._save_h5(save_path / "obs_values.h5", val_traj)
        self._save_h5(save_path / "obs_mask.h5", mask)

        # hack for now -- In trajectory training, the locations don't matter, we just need to pass some for it to work.
        self._save_h5(save_path / "locations.h5", val_traj[0, :, :2, :])

        if save_dataset_object_additionally:
            with open(save_path / "mocap_dataset.pkl", 'wb') as f:
                pickle.dump(dataset, f)
    
    def _save_testing_data(self, dataset: MocapDataset, save_path: Path, save_dataset_object_additionally: bool = False):
        test_traj = dataset.tst.ys[np.newaxis, ...]
        num_traj = test_traj.shape[1]
        ts_test = dataset.tst.ts[np.newaxis, np.newaxis, :, np.newaxis].repeat(num_traj, axis=1)
        mask = np.ones((*test_traj.shape[:-1], 1), dtype=bool)

        save_path.mkdir(parents=True, exist_ok=True)
        self._save_h5(save_path / "obs_times.h5", ts_test)
        self._save_h5(save_path / "obs_values.h5", test_traj)
        self._save_h5(save_path / "obs_mask.h5", mask)

        # hack for now -- In trajectory training, the locations don't matter, we just need to pass some for it to work.
        self._save_h5(save_path / "locations.h5", test_traj[0, :, :2, :])

        if save_dataset_object_additionally:
            with open(save_path / "mocap_dataset.pkl", 'wb') as f:
                pickle.dump(dataset, f)

    @staticmethod
    def _save_h5(path: Path, data: np.ndarray, dataset_name: str = 'data'):
        with h5py.File(path, 'w') as hf:
            hf.create_dataset(dataset_name, data=data)



def print_shapes_of_mocap_files():
    for fname in ["mocap09.npz", "mocap35.npz", "mocap39.npz"]:
        x = np.load("experiments/mocap/" + fname)
        print(fname+":")
        print("train shape:", x["train"].shape)
        print("test shape:", x["test"].shape)
        print("validation shape:", x["validation"].shape)
        print()


def data_gen_mocap():
    # Download the mocap09.npz, mocap35.npz, mocap39.npz from the GP-ODE repo (https://github.com/hegdepashupati/gaussian-process-odes/tree/main/data/mocap)
    downloader = MocapDownloader()
    downloader.download_if_needed()

    # For different experiments, the training trajectories have different lengths. These lengths come from the GP-ODE paper appendix: 
    experiments = {
        "Subject 09 short": 50,
        "Subject 09 long": 100,
        "Subject 35 short": 50,
        "Subject 35 long": 250,
        "Subject 39 short": 100,
        "Subject 39 long": 250,
    }

    # 5-dimensional PCA for all experiments:
    for experiment_name, sequence_length in experiments.items():
        subject: str = experiment_name.split()[1]       # "09", "35", or "39"
        long_short: str = experiment_name.split()[2]    # "long", "short"
        
        dataset = MocapDataset(pca_components=5, subject=subject, seqlen=sequence_length)      

        # training data
        FimMocapDatasetBuilder()._save_training_data(dataset,
                                    save_path=Path(f"experiments/mocap/mocap{subject}{long_short}/data/train/5d"),
                                    save_dataset_object_additionally=True)
        
        # testing data
        FimMocapDatasetBuilder()._save_testing_data(dataset,
                                    save_path=Path(f"experiments/mocap/mocap{subject}{long_short}/data/test/5d"),
                                    save_dataset_object_additionally=True)

        # validation data
        FimMocapDatasetBuilder()._save_validation_data(dataset,
                                    save_path=Path(f"experiments/mocap/mocap{subject}{long_short}/data/valid/5d"),
                                    save_dataset_object_additionally=True)
        
        ###########################################
        #  Now we save the 3d and 2d PCA slices:  #
        ###########################################

        for mode in ["train", "test", "valid"]:
            data_path = Path(f"experiments/mocap/mocap{subject}{long_short}/data") / mode

            # Create directories for 3d+2d splits if they don't exist
            (data_path / "3d+2d/3d").mkdir(parents=True, exist_ok=True)
            (data_path / "3d+2d/2d").mkdir(parents=True, exist_ok=True)
            
            with h5py.File(data_path / "5d/obs_values.h5", 'r') as f_5d:
                obs_values_5d: np.ndarray = f_5d["data"][:]       # (1,P,T,5)
                #print(obs_values_5d.shape)
                with h5py.File(data_path / "3d+2d/3d/obs_values.h5", 'w') as f:
                    f.create_dataset("data", data=obs_values_5d[:,:,:,:3])
                with h5py.File(data_path / "3d+2d/2d/obs_values.h5", 'w') as f:
                    f.create_dataset("data", data=obs_values_5d[:,:,:,3:])

            with h5py.File(data_path / "5d/locations.h5") as f:
                locations_5d: np.ndarray = f["data"][:]            # (P,2,5); 2 because of Max's weird hack
                #print(locations_5d.shape)
                with h5py.File(data_path / "3d+2d/3d/locations.h5", 'w') as f:
                    f.create_dataset("data", data=locations_5d[:,:,:3])
                with h5py.File(data_path / "3d+2d/2d/locations.h5", 'w') as f:
                    f.create_dataset("data", data=locations_5d[:,:,3:])

            with h5py.File(data_path / "5d/obs_mask.h5") as f:
                obs_mask: np.ndarray = f["data"][:]              # (1,P,T,1)
                #print(obs_mask_5d.shape)
                with h5py.File(data_path / "3d+2d/3d/obs_mask.h5", 'w') as f:
                    f.create_dataset("data", data=obs_mask)
                with h5py.File(data_path / "3d+2d/2d/obs_mask.h5", 'w') as f:
                    f.create_dataset("data", data=obs_mask)

            with h5py.File(data_path / "5d/obs_times.h5") as f:
                obs_times: np.ndarray = f["data"][:]                # (1,P,T,1)
                #print(obs_times_5d.shape)
                with h5py.File(data_path / "3d+2d/3d/obs_times.h5", 'w') as f:
                    f.create_dataset("data", data=obs_times)
                with h5py.File(data_path / "3d+2d/2d/obs_times.h5", 'w') as f:
                    f.create_dataset("data", data=obs_times)
            
            print(experiment_name + " " + mode + ": Created and saved all .h5 and .pkl files.")


def explore_mocap():
    dataset = MocapDataset(pca_components=5, subject="39", seqlen=None)

    ys = dataset.tst.ys   # (6,120,5)
    #print(ys.shape)

    ts = dataset.tst.ts   # (120,)
    #print(ts.shape)

    dimensions = [2,3,4]
    ctx_list = [trajectory(ys[i][:,[0,3,4]], ts) for i in range(ys.shape[0])]
    plot_3d_paths(ctx_list=ctx_list)
    
    ctx_list = [trajectory(ys[i][:,[3,4]], ts) for i in range(ys.shape[0])]
    plot_2d_paths(ctx_list=ctx_list)



if __name__ == "__main__":
    data_gen_mocap()
    #explore_mocap()
    
    """
        Shape of the data as used by GP-ODE (you get this output when you run print_shapes_of_mocap_files):

        mocap09.npz:
            train shape: (6, 120, 50)
            test shape: (2, 120, 50)
            validation shape: (2, 120, 50)

        mocap35.npz:
            train shape: (16, 300, 50)
            test shape: (4, 300, 50)
            validation shape: (3, 300, 50)

        mocap39.npz:
            train shape: (6, 300, 50)
            test shape: (2, 300, 50)
            validation shape: (2, 300, 50)
    """

        
    """
        Here's the data shapes after 5-dimensional PCA for the different experiments:

        PCA transformed training data shape of experiment Subject 09 short: (6, 50, 5)
        PCA transformed training data shape of experiment Subject 09 long: (6, 100, 5)
        PCA transformed training data shape of experiment Subject 35 short: (16, 50, 5)
        PCA transformed training data shape of experiment Subject 35 long: (16, 250, 5)
        PCA transformed training data shape of experiment Subject 39 short: (6, 100, 5)
        PCA transformed training data shape of experiment Subject 39 long: (6, 250, 5)
    """

    """ Relative feature importances:
        Subject 09:
            [0.52668012 0.23259408 0.18064054 0.01671045 0.00798557]
            top 3: 0.94
            top 5: 0.965
            4&5: 0.0246
            top 3 / top 5: 0.974
        Subject 35:
            [0.40852423 0.26171155 0.19500063 0.05930626 0.01727908]
            top 3: 0.865
            top 5: 0.942
            4&5: 0.0765
            top 3 / top 5: 0.918
        Subject 39:
            [0.75666713 0.08921361 0.07360449 0.04248909 0.00907802]
            top 3: 0.92
            top 5: 0.971
            4&5: 0.0514
            top 3 / top 5: 0.947
    """

    quit()

    # This is Max's code

    seqlen = 100
    subj = "09"
    dataset = MocapDataset(pca_components=5, subject=subj, seqlen=seqlen)
    x = dataset.build_pca(dataset.trn.ys, train=True)
    print(f"PCA transformed training data shape: {x.shape}")

    FimMocapDatasetBuilder()._save_dataset(dataset,
                                         save_path=Path(f"experiments/mocap/subject_{subj}/seqlen_{seqlen}"),
                                         save_dataset_object_additionally=True)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each trajectory with a different color
    colors = plt.cm.viridis(np.linspace(0, 1, x.shape[0]))
    for i in range(x.shape[0]):
        ax.plot(x[i, :, 0], x[i, :, 1], x[i, :, 2],
                color=colors[i], linewidth=2, alpha=0.7,
                label=f'Trajectory {i + 1}')
        # Mark start point
        ax.scatter(x[i, 0, 0], x[i, 0, 1], x[i, 0, 2],
                   color=colors[i], s=100, marker='o', edgecolors='black')

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_zlabel('PC3', fontsize=12)
    ax.set_title('Mocap Training Data - PCA Components (3D)', fontsize=14)

    # Set viewing angle to match reference image
    # ax.view_init(azim=100)

    ax.legend()
    plt.tight_layout()
    plt.show()
