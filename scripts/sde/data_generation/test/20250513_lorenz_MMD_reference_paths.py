import json
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import optree
import torch
import torchsde

from fim import data_path
from fim.utils.evaluation_sde import NumpyEncoder, save_fig


class StochasticLorenz(object):
    """Stochastic Lorenz attractor.
    Adapted from torchsde github repo:
    https://github.com/google-research/torchsde/blob/master/examples/latent_sde_lorenz.py
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, diffusion_label: str, a: Sequence = (10.0, 28.0, 8 / 3), b: Sequence = (0.15, 0.15, 0.15)):
        super(StochasticLorenz, self).__init__()
        self.a = a
        self.b = b
        if diffusion_label == "constant":
            self.g = self.g_constant

        elif diffusion_label == "linear":
            self.g = self.g_linear

        else:
            ValueError("'diffusion_label' must be in [constant, linear]")

    def f(self, t, y):
        x1, x2, x3 = torch.split(y, split_size_or_sections=(1, 1, 1), dim=1)
        a1, a2, a3 = self.a

        f1 = a1 * (x2 - x1)
        f2 = a2 * x1 - x2 - x1 * x3
        f3 = x1 * x2 - a3 * x3
        return torch.cat([f1, f2, f3], dim=1)

    def g_constant(self, t, y):
        x1, x2, x3 = torch.split(y, split_size_or_sections=(1, 1, 1), dim=1)
        b1, b2, b3 = self.b

        g1 = b1 * torch.ones_like(x1)
        g2 = b2 * torch.ones_like(x1)
        g3 = b3 * torch.ones_like(x1)
        return torch.cat([g1, g2, g3], dim=1)

    def g_linear(self, t, y):
        x1, x2, x3 = torch.split(y, split_size_or_sections=(1, 1, 1), dim=1)
        b1, b2, b3 = self.b

        g1 = x1 * b1
        g2 = x2 * b2
        g3 = x3 * b3
        return torch.cat([g1, g2, g3], dim=1)

    @torch.no_grad()
    def sample(self, x0, ts):
        xs = torchsde.sdeint(self, x0, ts)
        mean, std = torch.mean(xs, dim=(0, 1)), torch.std(xs, dim=(0, 1))
        xs.sub_(mean).div_(std)
        return xs


def lorenz_paths_figure(obs_values: np.ndarray):
    """
    obs_values: [P, T, 3]
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300, subplot_kw={"projection": "3d"})

    for i in range(obs_values.shape[0]):
        ax.plot(
            obs_values[i, :, 0],
            obs_values[i, :, 1],
            obs_values[i, :, 2],
            color="black",
            linewidth=0.5,
            label="Observed Paths" if i == 0 else None,
        )

    fig.legend()

    return fig


def save_as_json(data: list[dict], file_path: Path):
    # Convert to JSON
    json_data = json.dumps(data, cls=NumpyEncoder)

    with open(file_path, "w") as file:
        file.write(json_data)


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    # set save paths
    save_dir = Path("processed/test")
    subdir_label = "lorenz_system_mmd_reference_paths"

    if not save_dir.is_absolute():
        save_dir = Path(data_path) / save_dir

    t0 = 0
    t1 = 1
    time_series_length = 40
    time_series_count = 128

    num_dt_steps = 100

    loaded_inference_data = None

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    time: str = str(datetime.now().strftime("%Y%m%d%H%M%S"))
    save_data_dir: Path = save_dir / (time + "_" + subdir_label)
    save_data_dir.mkdir(parents=True, exist_ok=True)

    # dir for saving data as jsons
    data_jsons_dir = save_data_dir / "data_jsons"

    # generate data for inference, with initial states and grids
    grid = np.linspace(t0, t1, num=time_series_length)  # [T]

    if loaded_inference_data is None:
        inference_data = [
            {
                "initial_state_label": "sampled_normal_mean_0_std_1",
                "initial_states": np.random.normal(loc=0.0, scale=1.0, size=(time_series_count, 3)),
                "grid": grid,  # [T]
            },
            {
                "initial_state_label": "sampled_normal_mean_0_std_2",
                "initial_states": np.random.normal(loc=0.0, scale=2.0, size=(time_series_count, 3)),
                "grid": grid,  # [T]
            },
            {
                "initial_state_label": "fixed_at_1_1_1",
                "initial_states": np.ones(shape=(time_series_count, 3)),
                "grid": grid,  # [T]
            },
        ]

    else:
        inference_data: list[dict] = json.load(open(loaded_inference_data, "r"))  # same keys as paths setups

        # reduce to keys from above
        inference_data = [
            {
                "initial_state_label": d["initial_state_label"],
                "initial_states": d["initial_states"],
                "grid": grid,
            }
            for d in inference_data
        ]

    print("Inference data")
    pprint(optree.tree_map(lambda x: x.shape if isinstance(x, np.ndarray) else x, inference_data))
    print("\n")

    save_as_json(inference_data, save_data_dir / (time + "_lorenz_mmd_inference_data.json"))

    # generate reference paths for MMD
    reference_data = []

    for diffusion_label in ["constant", "linear"]:
        for data in inference_data:
            initial_state_label = data.get("initial_state_label")
            initial_states = data.get("initial_states")
            grid = data.get("grid")

            paths = StochasticLorenz(diffusion_label=diffusion_label).sample(torch.from_numpy(initial_states), torch.from_numpy(grid))
            paths = paths.numpy()  # [T, P, 3]
            paths = np.swapaxes(paths, 0, 1)  # [T, P, 3]

            reference_data.append(
                {
                    "diffusion_label": diffusion_label,
                    "paths": paths,
                    "initial_state_label": initial_state_label,
                    "grid": grid,
                }
            )

    print("Reference data:")
    pprint(optree.tree_map(lambda x: x.shape if isinstance(x, np.ndarray) else x, reference_data))
    print("\n")

    save_as_json(reference_data, save_data_dir / (time + "_lorenz_mmd_reference_data.json"))

    # plot paths for validation
    for data in reference_data:
        fig = lorenz_paths_figure(data.get("paths"))
        descr = "init_state_" + data.get("initial_state_label") + "_diff_type_" + data.get("diffusion_label")
        save_fig(fig, save_data_dir / "sample_paths_figures", descr)
