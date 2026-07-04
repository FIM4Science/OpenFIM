"""
MoCap dataset class for FIMODE — adapted from:
  https://github.com/hegdepashupati/gaussian-process-odes/blob/main/src/datasets/mocap.py

Source .npz files (mocap09.npz, mocap35.npz, mocap39.npz) live in data/ode/mocap/.
Each file contains three splits
("train", "validation", "test") with shape (N, T, 50) — N trajectories of
T timesteps in 50D joint-angle space.

MocapDataset:
  - Applies PCA (pca_components=5) to reduce 50D → 5D.
  - Optionally normalizes both raw data and PCA-reduced data.
  - Exposes .trn, .val, .tst as Data objects with .ys and .ts arrays.
  - .pca and .pca_normalize are stored for inverse projection back to 50D.

This module is also imported by finetune.py's pickle shim (_load_mocap_pickle)
so the class names must match exactly.
"""

import os

import numpy as np
from sklearn.decomposition import PCA


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return (x * self.std) + self.mean


class Data:
    def __init__(self, ys, ts):
        self.ts = ts.astype(np.float32)
        self.ys = ys.astype(np.float32)

    def __len__(self):
        return self.ys.shape[0]

    def __getitem__(self, index):
        return self.ys[index, ...], self.ts


class MocapDataset:
    def __init__(
        self,
        data_path="data/ode/mocap/",
        subject="09",
        dt=0.01,
        pca_components=-1,
        seqlen=50,
        data_normalize=False,
        pca_normalize=True,
    ):
        self.data_path = data_path
        self.dt = dt
        self.seqlen = seqlen
        self.pca_components = pca_components
        self.data_normalize = data_normalize
        self.pca_normalize = pca_normalize

        assert subject in ["09", "35", "39"], f"Unknown subject: {subject}"
        fname = os.path.join(data_path, "mocap" + subject + ".npz")
        mocap_data = np.load(fname)

        xs_test = mocap_data["test"]
        ts_test = dt * np.arange(xs_test.shape[1])

        xs_valid = mocap_data["validation"]
        ts_valid = dt * np.arange(xs_valid.shape[1])

        xs_train = mocap_data["train"]
        ts_train = dt * np.arange(xs_train.shape[1])

        xs_train = self._treat_zero_readings(xs_train)
        xs_valid = self._treat_zero_readings(xs_valid)
        xs_test = self._treat_zero_readings(xs_test)

        self.data_std = xs_train.std((0, 1), keepdims=True) + 1e-5
        self.data_mean = xs_train.mean((0, 1), keepdims=True)

        if data_normalize:
            self.data_normalize = Normalize(self.data_mean, self.data_std)
            xs_train = self.data_normalize(xs_train)
            xs_valid = self.data_normalize(xs_valid)
            xs_test = self.data_normalize(xs_test)
        else:
            self.data_normalize = None

        if pca_components > 0:
            xs_train = self._build_pca(xs_train, train=True)
            xs_valid = self._build_pca(xs_valid, train=False)
            xs_test = self._build_pca(xs_test, train=False)

        if pca_normalize:
            pca_m = xs_train.mean((0, 1), keepdims=True)
            pca_s = xs_train.std((0, 1), keepdims=True) + 1e-5
            self.pca_normalize = Normalize(pca_m, pca_s)
            xs_train = self.pca_normalize(xs_train)
            xs_valid = self.pca_normalize(xs_valid)
            xs_test = self.pca_normalize(xs_test)
        else:
            self.pca_normalize = None

        self.trn = Data(ys=xs_train[:, :seqlen], ts=ts_train[:seqlen])
        self.val = Data(ys=xs_valid, ts=ts_valid)
        self.tst = Data(ys=xs_test, ts=ts_test)

    def _treat_zero_readings(self, data):
        data[:, :, (24, 25, 31, 32)] = 1e-6
        return data

    def _build_pca(self, x, train=False):
        N, T, D = x.shape
        x_stack = np.vstack([x[i] for i in range(N)])
        if train:
            self.pca = PCA(n_components=self.pca_components)
            x_pca = self.pca.fit_transform(x_stack)
        else:
            x_pca = self.pca.transform(x_stack)
        return np.stack([x_pca[i * T : (i + 1) * T] for i in range(N)])
