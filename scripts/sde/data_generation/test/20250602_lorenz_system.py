import json
from datetime import datetime
from pathlib import Path
from pprint import pprint

import torch
import torchsde
from torch import Tensor
from tqdm import tqdm

from fim import data_path
from fim.utils.grids import define_mesh_points
from fim.utils.sde.evaluation import NumpyEncoder


def save_as_json(data: dict, save_dir: Path, filename: str):
    data = torch.utils._pytree.tree_map(lambda x: x.detach().numpy() if isinstance(x, Tensor) else x, data)
    json_data = json.dumps(data, cls=NumpyEncoder)

    file: Path = save_dir / (filename + ".json")

    save_dir.mkdir(exist_ok=True, parents=True)
    with open(file, "w") as file:
        file.write(json_data)


def _check_finite(x: Tensor) -> None:
    if isinstance(x, Tensor):
        assert torch.isfinite(x).all().item()


def _pprint_dict_with_shapes(d: dict) -> None:
    pprint(torch.utils._pytree.tree_map(lambda x: x.shape if isinstance(x, Tensor) else x, d))


class StochasticLorenz(object):
    """
    Adapted class `StochasticLorenz` from: https://github.com/google-research/torchsde/blob/master/examples/latent_sde_lorenz.py
    to constant and linear diffusion terms.
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, diffusion_mode: str, diffusion_constants: tuple):
        super(StochasticLorenz, self).__init__()
        self.a = (10.0, 28.0, 8 / 3)

        if diffusion_mode == "linear":
            self.g = self.g_linear

        elif diffusion_mode == "constant":
            self.g = self.g_constant

        else:
            raise ValueError(f"Got {diffusion_mode}")

        assert len(diffusion_constants) == 3, f"Got {len(diffusion_constants)}"
        self.diffusion_constants = diffusion_constants

    def f(self, t, y):
        x1, x2, x3 = torch.split(y, split_size_or_sections=(1, 1, 1), dim=1)
        a1, a2, a3 = self.a

        f1 = a1 * (x2 - x1)
        f2 = a2 * x1 - x2 - x1 * x3
        f3 = x1 * x2 - a3 * x3
        return torch.cat([f1, f2, f3], dim=1)

    def g_linear(self, t, y):
        x1, x2, x3 = torch.split(y, split_size_or_sections=(1, 1, 1), dim=1)
        b1, b2, b3 = self.diffusion_constants

        g1 = x1 * b1
        g2 = x2 * b2
        g3 = x3 * b3
        return torch.cat([g1, g2, g3], dim=1)

    def g_constant(self, t, y):
        x1, x2, x3 = torch.split(y, split_size_or_sections=(1, 1, 1), dim=1)
        b1, b2, b3 = self.diffusion_constants

        g1 = torch.ones_like(x1) * b1
        g2 = torch.ones_like(x2) * b2
        g3 = torch.ones_like(x3) * b3
        return torch.cat([g1, g2, g3], dim=1)

    @torch.no_grad()
    def sample(self, x0, ts, noise_std, normalize):
        """Sample data for training. Store data normalization constants if necessary."""
        xs = torchsde.sdeint(self, x0, ts)
        if normalize:
            mean, std = torch.mean(xs, dim=(0, 1)), torch.std(xs, dim=(0, 1))
            xs.sub_(mean).div_(std).add_(torch.randn_like(xs) * noise_std)
        return xs


def sample_initial_states(label, num_traj, device):
    """
    Sample initial states for lorenz system by label.
    """
    if label == "N(0,1)":
        return torch.randn(num_traj, 3, device=device)

    elif label == "N(0,2)":
        return 2 * torch.randn(num_traj, 3, device=device)

    elif label == "(1,1,1)":
        return torch.ones(num_traj, 3, device=device)

    else:
        raise ValueError("Initial states label not recognized.")


def get_data_from_setup(
    set_id: int,
    initial_states: Tensor,  # [B, 3]
    diffusion_mode: str,
    diffusion_constants: tuple,
    t0: float,
    t1: float,
    num_obs: int,
    noise_std: float,
    locations: Tensor,  # [G, 3]
    initial_states_label: str,
    mean: Tensor | None = None,  # [3]
    std: Tensor | None = None,  # [3]
    device: str = "cpu",
):
    """
    Simulate Lorenz data from the specified setup.
    Standardize data with (inferred) mean and std.
    Add gaussian noise to standardized trajectories.
    Extract vector fields on locations grid.


    Return:
        data (dict): Contains generated trajectories and identifiers.
        mean, std (Tensor): To apply same standardization to other splits.
    """
    num_traj = initial_states.shape[0]

    # solutions of lorenz system
    lorenz = StochasticLorenz(diffusion_mode, diffusion_constants)
    obs_grid = torch.linspace(t0, t1, steps=num_obs, device=device)  # [T]
    sol = torchsde.sdeint(lorenz, initial_states, obs_grid)  # [T, num_traj, 3]
    sol = torch.transpose(sol, 0, 1)  # [num_traj, T, 3]

    drift_at_locations = lorenz.f(None, locations)  # [G, 3]
    diffusion_at_locations = lorenz.g(None, locations)  # [G, 3]

    # normalize observations
    if mean is None and std is None:
        mean = torch.mean(sol, dim=(0, 1))
        std = torch.std(sol, dim=(0, 1))

    else:
        if mean is None or std is None:
            raise ValueError(f"Must pass mean and std, got {type(mean)} and {type(std)}.")

        else:
            mean = mean.to(device)
            std = std.to(device)

    clean_obs_values = (sol - mean) / std
    locations = (locations - mean) / std

    # ito's formula
    drift_at_locations = drift_at_locations / std
    diffusion_at_locations = diffusion_at_locations / std

    noisy_obs_values = clean_obs_values + noise_std * torch.randn_like(clean_obs_values)

    obs_grid = torch.broadcast_to(obs_grid.reshape(1, -1, 1), (num_traj, num_obs, 1))

    # check shapes
    assert obs_grid.shape == (num_traj, num_obs, 1), f"Got {obs_grid.shape}"
    assert clean_obs_values.shape == (num_traj, num_obs, 3), f"Got {clean_obs_values.shape}"
    assert noisy_obs_values.shape == (num_traj, num_obs, 3), f"Got {noisy_obs_values.shape}"
    assert mean.shape == (3,), f"Got {mean.shape}"
    assert std.shape == (3,), f"Got {std.shape}"
    assert locations.shape == drift_at_locations.shape == diffusion_at_locations.shape

    data = {
        "set_id": set_id,
        "diffusion_mode": diffusion_mode,
        "diffusion_constants": diffusion_constants,
        "noise_std": noise_std,
        "delta_tau": (t1 - t0) / (num_obs - 1),
        "obs_grid": obs_grid,
        "clean_obs_values": clean_obs_values,
        "noisy_obs_values": noisy_obs_values,
        "initial_states_label": initial_states_label,
        "normalizing_mean": mean,
        "normalizing_std": std,
        "locations": locations,
        "drift_at_locations": drift_at_locations,
        "diffusion_at_locations": diffusion_at_locations,
    }

    data = torch.utils._pytree.tree_map(lambda x: x.to("cpu") if isinstance(x, Tensor) else x, data)
    torch.utils._pytree.tree_map(_check_finite, data)

    return data, mean, std


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    save_dir = Path("processed/test")
    subdir_label = "lorenz_system_with_vector_fields_at_locations"

    num_sets = 5
    num_traj_train = 1024
    num_traj_val = 128
    num_traj_ref = 128

    train_init_states = "N(0,1)"
    all_ref_init_states = ["N(0,1)", "N(0,2)", "(1,1,1)"]

    # data min is approx. [-20, -30, -3.5]
    # data max is approx. [20, 30, 50]
    locations_min = [-30, -40, -10]
    locations_max = [30, 40, 60]
    locations_size_per_axis = 20

    setups = {
        "neural_sde_paper": {
            "diffusion_mode": "constant",
            "diffusion_constants": (0.15, 0.15, 0.15),
            "num_obs": 41,
            "t0": 0,
            "t1": 1,
            "noise_std": 0.01,
        },
        "neural_sde_github": {
            "diffusion_mode": "linear",
            "diffusion_constants": (0.1, 0.28, 0.3),
            "num_obs": 100,
            "t0": 0,
            "t1": 2,
            "noise_std": 0.01,
        },
    }

    # --------------------------------------------------------------------------------------------------------------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not save_dir.is_absolute():
        save_dir = Path(data_path) / save_dir
    time: str = str(datetime.now().strftime("%Y%m%d"))
    save_data_dir: Path = save_dir / (time + "_" + subdir_label)
    save_data_dir.mkdir(parents=True, exist_ok=True)

    locations = define_mesh_points(total_points=locations_size_per_axis**3, n_dims=3, ranges=list(zip(locations_min, locations_max)))
    # [locations_size_per_axis**3,3]

    for setup_label, setup in tqdm(setups.items(), desc="Setup", total=len(setups), leave=False):
        for set_id in tqdm(range(num_sets), desc="Set", total=num_sets, leave=False):
            setup_set_data_dir = save_data_dir / setup_label / (f"set_{set_id}")

            train_initial_states = sample_initial_states(train_init_states, num_traj_train, device)
            train_data, train_mean, train_std = get_data_from_setup(
                set_id,
                train_initial_states,
                **setup,
                locations=locations,
                initial_states_label=train_init_states,
                device=device,
            )
            save_as_json(train_data, setup_set_data_dir, "train_data")

            if set_id == 0:
                print(f"{setup_label} train data shapes:")
                _pprint_dict_with_shapes(train_data)

            val_initial_states = sample_initial_states(train_init_states, num_traj_val, device)
            val_data, _, _ = get_data_from_setup(
                set_id,
                val_initial_states,
                **setup,
                locations=locations,
                mean=train_mean,
                std=train_std,
                initial_states_label=train_init_states,
                device=device,
            )
            save_as_json(val_data, setup_set_data_dir, "validation_data")

            for ref_init_states in tqdm(all_ref_init_states, desc="Reference Data", total=len(all_ref_init_states), leave=False):
                ref_initial_states = sample_initial_states(ref_init_states, num_traj_ref, device)
                ref_data, _, _ = get_data_from_setup(
                    set_id,
                    ref_initial_states,
                    **setup,
                    locations=locations,
                    mean=train_mean,
                    std=train_std,
                    initial_states_label=ref_init_states,
                    device=device,
                )
                save_as_json(ref_data, setup_set_data_dir, ref_init_states + "_" + "reference_data")

                inference_data = ref_data
                inference_data.pop("noisy_obs_values")
                inference_data.pop("drift_at_locations")
                inference_data.pop("diffusion_at_locations")
                clean_obs_values = inference_data.pop("clean_obs_values")  # [B, T, 3]
                inference_data["initial_states"] = clean_obs_values[:, 0, :]
                save_as_json(inference_data, setup_set_data_dir, ref_init_states + "_" + "inference_data")
