"""
This file generates data for the VDP and FHN experiments like in the GP-ODE paper: https://arxiv.org/abs/2309.09222
The reason we have this is to generate data with the exact same noise as in GP-ODE which we need for comparing properly.
"""


import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import h5py


class Data:
    def __init__(self, ys, ts):
        self.ts = ts.astype(np.float32)
        self.ys = ys.astype(np.float32)

    def __len__(self):
        return self.ys.shape[0]

    def __getitem__(self, index):
        return self.ys[index, ...], self.ts


class VanderPol(object):
    def __init__(self,
                 S_train=30, T_train=6.0,
                 S_test=None, T_test=None,
                 noise_var=0.1,
                 x0=np.array([[-1.5, 2.5]]),  # np.array([[-0.5, 2.5]]),
                 mu=0.5,
                 ):
        noise_rng = np.random.RandomState(121)
        init_rng = np.random.RandomState(123)
        S_test = S_test if S_test is not None else S_train
        T_test = T_test if T_test is not None else T_train

        self.xlim = (-3.5, 3.5)
        self.ylim = (-3.5, 3.5)

        self.mu = mu
        self.x0 = x0
        self.noise_var = noise_var
        self.new_x0 = self.x0 + init_rng.normal(size=(100, 2)) * 0.2

        xs_train, ts_train = self.generate_sequence(x0=self.x0, sequence_length=S_train, T=T_train)
        xs_test, ts_test = self.generate_sequence(x0=self.x0, sequence_length=S_test, T=T_test)
        xs_new_x0, ts_new_x0 = self.generate_sequence(x0=self.new_x0, sequence_length=S_train, T=T_train)

        xs_train = xs_train + noise_rng.normal(size=xs_train.shape) * (self.noise_var ** 0.5)

        self.trn = Data(ys=xs_train, ts=ts_train)
        self.tst = Data(ys=xs_test, ts=ts_test)
        self.tst_new_x0 = Data(ys=xs_new_x0, ts=ts_new_x0)

    def generate_sequence(self, x0, sequence_length, T):
        ts = np.linspace(0, 1.0, sequence_length) * T
        xs = []
        for _x0 in x0:
            xs.append(odeint(self.f, _x0, ts))
        return np.stack(xs), ts

    def f(self, y, t):
        dy = np.zeros(2)
        dy[0] = y[1]
        dy[1] = -y[0] + self.mu * y[1] * (1 - y[0] ** 2)
        return dy


class VanderPolNonUniform(object):
    def __init__(self,
                 S_train=25, T_train=7.0,
                 S_test=None, T_test=None,
                 noise_var=0.1,
                 x0=np.array([[-1.5, 2.5]]),
                 mu=0.5,
                 ):
        noise_rng = np.random.RandomState(121)
        ts_rng = np.random.RandomState(122)
        S_test = S_test if S_test is not None else S_train
        T_test = T_test if T_test is not None else T_train

        self.xlim = (-3.5, 3.5)
        self.ylim = (-3.5, 3.5)

        self.mu = mu
        self.x0 = x0
        self.noise_var = noise_var

        ts_train = self.generate_random_ts(sequence_length=S_train,
                                           time_range=(0.0, T_train),
                                           rng=ts_rng)
        ts_train[0] = 0.0
        ts_test = self.generate_random_ts(sequence_length=S_test,
                                          time_range=(T_train, T_test),
                                          rng=ts_rng)
        xs_train = self.generate_sequence(x0=self.x0, ts=ts_train)
        xs_test = self.generate_sequence(x0=self.x0, ts=np.insert(ts_test, 0, 0))[:, 1:]

        xs_train = xs_train + noise_rng.normal(size=xs_train.shape) * (self.noise_var ** 0.5)
        self.trn = Data(ys=xs_train, ts=ts_train)
        self.tst = Data(ys=xs_test, ts=ts_test)

    def generate_sequence(self, x0, ts):
        xs = []
        for _x0 in x0:
            xs.append(odeint(self.f, _x0, ts))
        return np.stack(xs)

    def f(self, y, t):
        dy = np.zeros(2)
        dy[0] = y[1]
        dy[1] = -y[0] + self.mu * y[1] * (1 - y[0] ** 2)
        return dy

    def generate_random_ts(self, sequence_length, time_range, rng):
        ts = np.sort(rng.random_sample(sequence_length)) * (time_range[1] - time_range[0]) + time_range[0]
        return ts


def plot_data_and_fhn():
    """Plot FHN integral curve and training data from experiments/fhn/data_gpode in the same diagram."""
    from pathlib import Path
    from scipy.integrate import solve_ivp

    from ODEs import FHN_ode

    data_dir = Path("experiments/fhn/data_gpode")

    # Load trajectory data: prefer obs_values.h5 (FIM convention), else from npz train_ys
    if (data_dir / "test_values.h5").exists():
        with h5py.File(data_dir / "test_values.h5", "r") as hf:
            obs_values = np.array(hf["data"])  # (1, 1, T, 2) or (1, T, 2)
        if (data_dir / "test_values.h5").exists():
            with h5py.File(data_dir / "test_values.h5", "r") as hf:
                obs_times = np.array(hf["data"]).squeeze()
        else:
            obs_times = None
    else:
        npz = np.load(data_dir / "fhn_interpolation.npz")
        obs_values = npz["test_ys"]   # (1, T, 2)
        obs_times = npz["test_ts"]    # (T,)

    # Squeeze to (T, 2) for a single trajectory
    while obs_values.ndim > 2:
        obs_values = obs_values.squeeze(0)
    ys_data = np.atleast_2d(obs_values)

    # FHN integral curve: integrate from first data point over data time range
    y0 = ys_data[0, :]
    if obs_times is not None:
        t_span = (float(obs_times.min()), float(obs_times.max()))
        t_eval = np.linspace(t_span[0], t_span[1], 200)
    else:
        t_span = (0.0, 10.0)
        t_eval = np.linspace(0.0, 10.0, 200)
    sol = solve_ivp(FHN_ode, t_span, y0, t_eval=t_eval, method="RK45")
    ys_fhn = sol.y.T  # (200, 2)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(ys_fhn[:, 0], ys_fhn[:, 1], "b-", label="FHN integral curve", linewidth=2)
    ax.plot(ys_data[:, 0], ys_data[:, 1], "ro", label="test_ys (data_gpode)", markersize=4)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("FHN dynamical system: integral curve and training data")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig("experiments/fhn/data_gpode/fhn_integral_curve_and_training_data.pdf")
    #plt.show()

#plot_data_and_fhn()


if __name__ == "__main__":
    """
    vdp = VanderPol(x0=np.array([[-1.5, 2.5]]), S_train=25, T_train=7., noise_var=0.05)
    xs, ts = vdp.trn.ys, vdp.trn.ts
    print(xs.shape, ts.shape)   # (1,25,2) (25,)
    plt.scatter(xs[0, :, 0], xs[0, :, 1], label="trajectory (x vs y)")
    plt.legend()
    plt.show()
    """

    vdp1_npz = np.load("experiments/vdp1/data_gpode/vdp1.npz")           # keys: train_ts, train_ys, test_ts, test_ys, obs_noise_var, x0, mu
    vdp2_npz = np.load("experiments/vdp2/data_gpode/vdp2.npz")
    fhn_small_npz = np.load("experiments/fhn/data_gpode/fhn_interpolation_small.npz")   # keys: 'full_ys', 'full_ts', 'train_ts', 'train_ys', 'interpolation_ts', 'interpolation_ys', 'interpolation_mask', 'obs_noise_var', 'x0'
    fhn_npz = np.load("experiments/fhn/data_gpode/fhn_interpolation.npz") # keys: 'full_ys', 'full_ts', 'train_ts', 'train_ys', 'interpolation_ts', 'interpolation_ys', 'interpolation_mask', 'obs_noise_var', 'x0'

    #print(list(vdp1_npz.keys()))
    #print(list(vdp2_npz.keys()))
    #print(list(fhn_small_npz.keys()))
    #print(list(fhn_npz.keys()))

    #print(vdp1_npz['obs_noise_var'])    # 0.05
    #print(vdp1_npz['x0'])               # array([[-1.5,  2.5]])
    #print(vdp1_npz['mu'])               # 0.5

    for task in ["vdp1", "vdp2"]:
        npz = np.load(f"experiments/{task}/data_gpode/{task}.npz")
        
        # turn them into .h5 files
        obs_times = npz['train_ts']        # (50,)
        obs_values = npz['train_ys']        # (1,50,2)
        obs_mask = np.ones_like(obs_times)       # (50,)

        obs_times = obs_times.reshape(1,1,50,1)
        obs_values = obs_values.reshape(1,1,50,2)
        obs_mask = obs_mask.reshape(1,1,50,1)

        #print(obs_times)

        with h5py.File(f"experiments/{task}/data_gpode/obs_times.h5", 'w') as hf:
            hf.create_dataset("data", data=obs_times)
        with h5py.File(f"experiments/{task}/data_gpode/obs_values.h5", 'w') as hf:
            hf.create_dataset("data", data=obs_values)
        with h5py.File(f"experiments/{task}/data_gpode/obs_mask.h5", 'w') as hf:
            hf.create_dataset("data", data=obs_mask)
        
        """ Lastly, must generate locations.h5 because the model always needs locations, even if unused. """
        with h5py.File(f"experiments/{task}/data_gpode/locations.h5", 'w') as hf:
            hf.create_dataset("data", data=np.ndarray((1,1,1)))   # we need at least one location
        
        """
        xs, ts = vdp1_npz['test_ys'], vdp1_npz['test_ts']
        print(xs.shape, ts.shape)   # (50,2) (50,)
        plt.scatter(xs[:,:,0], xs[:,:,1], label="trajectory (x vs y)")
        plt.legend()
        plt.show()
        """
    
    # print()
    # print(fhn_small_npz["full_ts"].shape)        # (25,)
    # print(fhn_small_npz["full_ys"].shape)        # (1,25,2)
    # print(fhn_small_npz["train_ts"].shape)       # (19,)
    # print(fhn_small_npz["train_ys"].shape)       # (1,19,2)
    # print(fhn_small_npz["interpolation_ts"].shape)       # (6,)
    # print(fhn_small_npz["interpolation_ys"].shape)       # (1,6,2)
    # print(fhn_small_npz["interpolation_mask"].shape)     # (25,)
    # print(fhn_small_npz["obs_noise_var"])                # 0.025
    # print(fhn_small_npz["x0"])                           # [-1. -1.]

    # print()
    # print(fhn_npz["full_ts"].shape)        # (50,)
    # print(fhn_npz["full_ys"].shape)        # (1,50,2)
    # print(fhn_npz["train_ts"].shape)       # (38,)
    # print(fhn_npz["train_ys"].shape)       # (1,38,2)
    # print(fhn_npz["interpolation_ts"].shape)       # (12,)
    # print(fhn_npz["interpolation_ys"].shape)       # (1,12,2)
    # print(fhn_npz["interpolation_mask"].shape)     # (50,)
    # print(fhn_npz["obs_noise_var"])                # 0.025
    # print(fhn_npz["x0"])                           # [-1. -1.]


    print()
    print("OBS")
    npz = np.load(f"experiments/fhn/data_gpode/fhn_interpolation_small.npz")
    obs_ys = npz["full_ys"]
    print(obs_ys.shape)
    mask = (obs_ys[..., 0] >= 0) & (obs_ys[..., 1] <= 0)   # shape (1, 25)
    print(mask.shape)
    obs_ys = obs_ys[~mask].reshape(1,1,20,2)
    print(obs_ys.shape)
    #print(obs_ys)

    obs_ts = npz["full_ts"][~mask[0,:]].reshape(1,1,20,1)
    print(obs_ts.shape)
    #print(obs_ts)

    obs_mask = np.ones_like(obs_ys)
    print(obs_mask.shape)

    print()
    print("TEST")
    test_ys = npz["interpolation_ys"][:,1:].reshape(1,1,5,2)      # I am throwing out the first point because it's the same as the last point of the observations!! Not sure if I should do this...
    print(test_ys.shape)
    test_ts = npz["interpolation_ts"][1:].reshape(1,1,5,1)
    print(test_ts.shape)
    test_mask = np.ones_like(test_ys)
    print(test_mask.shape)

    print()

    with h5py.File(f"experiments/fhn/data_gpode/obs_times.h5", 'w') as hf:
        hf.create_dataset("data", data=obs_ts)
    with h5py.File(f"experiments/fhn/data_gpode/obs_values.h5", 'w') as hf:
        hf.create_dataset("data", data=obs_ys)
    with h5py.File(f"experiments/fhn/data_gpode/obs_mask.h5", 'w') as hf:
        hf.create_dataset("data", data=obs_mask)
    
    with h5py.File(f"experiments/fhn/data_gpode/test_times.h5", 'w') as hf:
        hf.create_dataset("data", data=test_ts)
    with h5py.File(f"experiments/fhn/data_gpode/test_values.h5", 'w') as hf:
        hf.create_dataset("data", data=test_ys)
    with h5py.File(f"experiments/fhn/data_gpode/test_mask.h5", 'w') as hf:
        hf.create_dataset("data", data=test_mask)
    
    # Lastly, must generate locations.h5 because the model always needs locations, even if unused.
    with h5py.File(f"experiments/fhn/data_gpode/locations.h5", 'w') as hf:
        hf.create_dataset("data", data=np.ndarray((1,1,1)))   # we need at least one location

    """
    # generate .h5 files for fhn_interpolation_small (which was used in GP-ODE)
    npz = np.load(f"experiments/fhn/data_gpode/fhn_interpolation_small.npz")
        
    print(npz["interpolation_mask"])
    print(npz["interpolation_mask"].shape)

    # turn them into .h5 files
    obs_times = npz['train_ts']        # (19,)
    obs_values = npz['train_ys']        # (1,19,2)
    obs_mask = np.ones_like(obs_times)       # (19,)

    obs_times = obs_times.reshape(1,1,19,1)
    obs_values = obs_values.reshape(1,1,19,2)
    obs_mask = obs_mask.reshape(1,1,19,1)

    with h5py.File(f"experiments/fhn/data_gpode/obs_times.h5", 'w') as hf:
        hf.create_dataset("data", data=obs_times)
    with h5py.File(f"experiments/fhn/data_gpode/obs_values.h5", 'w') as hf:
        hf.create_dataset("data", data=obs_values)
    with h5py.File(f"experiments/fhn/data_gpode/obs_mask.h5", 'w') as hf:
        hf.create_dataset("data", data=obs_mask)
    
    test_times = npz['interpolation_ts']        # (6,)
    test_values = npz['interpolation_ys']        # (1,6,2)
    #print(test_values)
    test_mask = np.ones_like(test_times)       # (6,)

    test_times = test_times.reshape(1,1,6,1)
    test_values = test_values.reshape(1,1,6,2)
    test_mask = test_mask.reshape(1,1,6,1)

    #all_values = np.concatenate((test_values, obs_values), axis=2)
    #print(all_values.shape)
    #print(all_values.mean(axis=2))
    #print(all_values + np.array([[0., 0.065868]]))

    with h5py.File(f"experiments/fhn/data_gpode/test_times.h5", 'w') as hf:
        hf.create_dataset("data", data=test_times)
    with h5py.File(f"experiments/fhn/data_gpode/test_values.h5", 'w') as hf:
        hf.create_dataset("data", data=test_values)
    with h5py.File(f"experiments/fhn/data_gpode/test_mask.h5", 'w') as hf:
        hf.create_dataset("data", data=test_mask)
    
    
    # Lastly, must generate locations.h5 because the model always needs locations, even if unused.
    with h5py.File(f"experiments/fhn/data_gpode/locations.h5", 'w') as hf:
        hf.create_dataset("data", data=np.ndarray((1,1,1)))   # we need at least one location


    xs, ts = npz['full_ys'], npz['full_ts']
    #print(xs.shape, ts.shape)   # (50,2) (50,)
    plt.scatter(xs[:,:,0], xs[:,:,1], label="trajectory (x vs y)")
    plt.legend()
    plt.savefig("experiments/fhn/data_gpode/plot.pdf")
    #plt.show()
    """

    quit()
