import json
import pickle
from datetime import datetime
from pathlib import Path
from pprint import pprint

import numpy as np
import optree

from fim import data_path
from fim.utils.sde.evaluation import NumpyEncoder


def process_single_trajectory(name: str, data_path: Path, delta_tau: float, transform: str, num_locations: int, **kwargs) -> dict:
    """
    Return data from single trajectory as one-path + batched data (s.t. it can be loaded by dataloader).

    Args:
        name (str): Name of system.
        data_path (Path): Path to .pickle containing array of shape [T].
        delta_tau (float): Time delta between two observations, for artificial time grid.
        transform (str): Transform applied to data. Must be in ["None", "fluctuations", "log"]

    Returns:
        dict with keys:
            name (str): Name of system.
            delta_tau (float): Time delta between two observations.
            transform (str): Transform applied to data.
            obs_times (np.ndarray): Regular observation times in [0, max_time]. Shape: [1, 1, T or T-1, 1]
            obs_values (np.ndarray): Transformed content of file path. Shape: [1, 1, T or T-1, 1]
            locations (np.ndarray): Regular grid in the observed data range. Shape: [1, num_locations, 1]
            initial_state (np.ndarray): Initial state of observations, to optionally generate path from. Shape: [1, 1, 1]
            path_length_to_generate (int): Length of obs_values, to generate path with just as many points.
    """

    assert transform in ["None", "fluctuations", "log"], f"Got {transform}."

    data = pickle.load(open(data_path, "rb"))  # [T]
    assert data.ndim == 1

    # remove nan values
    data = data[np.logical_not(np.isnan(data))]

    # create artificial time
    T = data.shape[0]
    times = delta_tau * np.arange(T)

    obs_times = times.reshape(1, 1, T, 1)
    obs_values = data.reshape(1, 1, T, 1)

    # apply an optional transformation to observed values
    if transform == "fluctuations":
        obs_times = obs_times[:, :, :-1, :]
        obs_values = obs_values[:, :, 1:, :] - obs_values[:, :, :-1, :]

    elif transform == "log":
        assert np.all(obs_values > 0.0)
        obs_values = np.log(obs_values)

    initial_states = obs_values[:, :, 0, :]

    # for plotting vector fields in observation range
    locations = np.linspace(start=np.amin(obs_values), stop=np.amax(obs_values), num=num_locations).reshape(1, -1, 1)

    return {
        "name": name,
        "delta_tau": delta_tau,
        "transform": transform,
        "obs_times": obs_times,
        "obs_values": obs_values,
        "locations": locations,
        "initial_states": initial_states,
        "path_length_to_generate": obs_values.shape[-2],
    }


def get_cross_validation_splits(complete_path_data: dict, num_total_splits: int, num_initial_states_in_val_split: int):
    """
    Split data into num_total_splits parts, leaving one split our for KSIG reference.
    Return splits as (potentially) separate paths and also concatenated, for BISDE evaluation.
    Chunk validation split into smaller trajectories for KSIG computation and return their initial states for generation.

    Args:
        complete_path_data (dict): Output of `process_single_trajectory`.
        num_total_splits (int): Number of data splits for cross validation.
        num_initial_states_in_val_split (int): Number of chunks to split validation split into.

    Returns:
        cross val inference paths: dicts with keys:
            name (str): Name of system
            transform (str): Transform applied to data.
            delta_tau (float): Time delta between two observations.
            split (int): Label identifying split. In (1, ...,num_total_splits).
            num_total_splits (int): Number of splits for reference.
            obs_times/values/mask for inference, either as separate paths (before and after validation split), or concated.
                obs_times/values_separate (np.ndarray): Shape: [1, 1 or 2, M, 1]
                obs_mask (np.ndarray): [1, 1 or 2, M, 1]
                obs_times/values_concat (np.ndarray): Shape: [1, 1, K, 1]
            locations (np.ndarray): Regular grid in the observed data range. Shape: [1, num_locations, 1]
            initial_states (np.ndarray): First observation of validation split. Shape: [1, num_initial_states_in_val_split, 1]
            path_length_to_generate (int): Length of validation split

        cross validation ksig reference paths: dicts with keys:
            name (str): Name of system.
            transform (str): Transform applied to data.
            delta_tau (float): Time delta between two observations.
            split (int): Label identifying split. In (1, ...,num_total_splits).
            num_total_splits (int): Number of splits for reference.
            locations (np.ndarray): Regular grid in the observed data range. Shape: [1, num_locations, 1]
            obs_times/values (np.ndarray): Times and in validation split. [1, num_initial_states_in_val_split, N, 1],
    """

    obs_times = complete_path_data.get("obs_times")  # [1, 1, T, 1]
    obs_values = complete_path_data.get("obs_values")  # [1, 1, T, 1]

    # remove last points for equal split
    T = obs_times.shape[-2]
    T = (T // num_total_splits) * num_total_splits
    obs_times = obs_times[..., :T, :]
    obs_values = obs_values[..., :T, :]

    # split into equal parts
    obs_times_split = np.split(obs_times, indices_or_sections=num_total_splits, axis=-2)
    obs_values_split = np.split(obs_values, indices_or_sections=num_total_splits, axis=-2)

    # gather data per validation split
    cross_val_inference_paths = []
    cross_val_ksig_reference_paths = []

    for split in range(num_total_splits):
        # ksig reference paths
        obs_times_at_val_split = obs_times_split[split]  # [1, 1, M, 1]
        obs_values_at_val_split = obs_values_split[split]

        # chunk into smaller paths
        M = obs_times_at_val_split.shape[-2]
        M_ = M // num_initial_states_in_val_split

        obs_times_at_val_split = obs_times_at_val_split[:, :, : (M_ * num_initial_states_in_val_split), :]
        obs_values_at_val_split = obs_values_at_val_split[:, :, : (M_ * num_initial_states_in_val_split), :]

        obs_times_at_val_split = obs_times_at_val_split.reshape(1, num_initial_states_in_val_split, M_, 1)
        obs_values_at_val_split = obs_values_at_val_split.reshape(1, num_initial_states_in_val_split, M_, 1)

        # to generate paths to compare against ...at_val_split
        initial_states = obs_values_at_val_split[:, :, 0, :]  # [1, num_initial_states_in_val_split, 1]

        # observations passed to models for inference
        if split == 0:
            # keep copy of split sequences of train split
            obs_times_at_train_split = obs_times_split[1:]
            obs_values_at_train_split = obs_values_split[1:]

            # one continuous path after the validation split
            obs_times_concat = np.concatenate(obs_times_at_train_split, axis=-2)
            obs_values_concat = np.concatenate(obs_values_at_train_split, axis=-2)

            obs_times_separate = obs_times_concat
            obs_values_separate = obs_values_concat
            obs_mask_separate = np.ones_like(obs_values_separate)

        elif split == num_total_splits - 1:
            # keep copy of split sequences of train split
            obs_times_at_train_split = obs_times_split[:-1]
            obs_values_at_train_split = obs_values_split[:-1]

            # one continuous path before the validation split
            obs_times_concat = np.concatenate(obs_times_at_train_split, axis=-2)
            obs_values_concat = np.concatenate(obs_values_at_train_split, axis=-2)

            obs_times_separate = obs_times_concat
            obs_values_separate = obs_values_concat
            obs_mask_separate = np.ones_like(obs_values_separate)

        else:
            # keep copy of split sequences of train split
            obs_times_at_train_split = obs_times_split[:split] + obs_times_split[split + 1 :]
            obs_values_at_train_split = obs_values_split[:split] + obs_values_split[split + 1 :]

            # combine the paths before and after the validation split
            obs_times_before_val_split = np.concatenate(obs_times_split[:split], axis=-2)
            obs_values_before_val_split = np.concatenate(obs_values_split[:split], axis=-2)
            obs_mask_before_val_split = np.ones_like(obs_times_before_val_split)

            obs_times_after_val_split = np.concatenate(obs_times_split[split + 1 :], axis=-2)
            obs_values_after_val_split = np.concatenate(obs_values_split[split + 1 :], axis=-2)
            obs_mask_after_val_split = np.ones_like(obs_times_after_val_split)

            # option 1: simply concatenate observations before and after split
            obs_times_concat = np.concatenate([obs_times_before_val_split, obs_times_after_val_split], axis=-2)
            obs_values_concat = np.concatenate([obs_values_before_val_split, obs_values_after_val_split], axis=-2)

            # option 2: keep paths before and after gap for validation separate
            T_before = obs_times_before_val_split.shape[-2]
            T_after = obs_times_after_val_split.shape[-2]
            T_padded = max(T_before, T_after)

            obs_times_before_val_split = np.pad(obs_times_before_val_split, pad_width=((0, 0), (0, 0), (0, T_padded - T_before), (0, 0)))
            obs_values_before_val_split = np.pad(obs_values_before_val_split, pad_width=((0, 0), (0, 0), (0, T_padded - T_before), (0, 0)))
            obs_mask_before_val_split = np.pad(obs_mask_before_val_split, pad_width=((0, 0), (0, 0), (0, T_padded - T_before), (0, 0)))

            obs_times_after_val_split = np.pad(obs_times_after_val_split, pad_width=((0, 0), (0, 0), (0, T_padded - T_after), (0, 0)))
            obs_values_after_val_split = np.pad(obs_values_after_val_split, pad_width=((0, 0), (0, 0), (0, T_padded - T_after), (0, 0)))
            obs_mask_after_val_split = np.pad(obs_mask_after_val_split, pad_width=((0, 0), (0, 0), (0, T_padded - T_after), (0, 0)))

            obs_times_separate = np.concatenate([obs_times_before_val_split, obs_times_after_val_split], axis=-3)
            obs_values_separate = np.concatenate([obs_values_before_val_split, obs_values_after_val_split], axis=-3)
            obs_mask_separate = np.concatenate([obs_mask_before_val_split, obs_mask_after_val_split], axis=-3)

        # build and save dicts for current split
        cross_val_inference_paths.append(
            {
                "name": complete_path_data.get("name"),
                "delta_tau": complete_path_data.get("delta_tau"),
                "transform": complete_path_data.get("transform"),
                "split": split,
                "num_total_splits": num_total_splits,
                "obs_times_separate": obs_times_separate,
                "obs_values_separate": obs_values_separate,
                "obs_mask_separate": obs_mask_separate.astype(bool),
                "obs_times_concat": obs_times_concat,
                "obs_values_concat": obs_values_concat,
                "obs_times_split": np.concatenate(obs_times_at_train_split, axis=-3),
                "obs_values_split": np.concatenate(obs_values_at_train_split, axis=-3),
                "initial_states": initial_states,
                "locations": complete_path_data.get("locations"),
                "path_length_to_generate": M_,
            }
        )

        cross_val_ksig_reference_paths.append(
            {
                "name": complete_path_data.get("name"),
                "delta_tau": complete_path_data.get("delta_tau"),
                "transform": complete_path_data.get("transform"),
                "split": split,
                "num_total_splits": num_total_splits,
                "obs_times": obs_times_at_val_split,
                "obs_values": obs_values_at_val_split,
                "locations": complete_path_data.get("locations"),
            }
        )

    return cross_val_inference_paths, cross_val_ksig_reference_paths


def _check_finite(x) -> None:
    """
    Helper function to check finiteness of arrays in a nested structure
    """
    if isinstance(x, np.ndarray):
        assert np.isfinite(x).all().item()


def _pprint_dict_with_shapes(d: dict) -> None:
    pprint(optree.tree_map(lambda x: x.shape if isinstance(x, np.ndarray) else x, d))


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    # set save paths
    save_dir = Path("processed/test")
    subdir_label = "real_world_with_5_fold_cross_validation_develop"

    if not save_dir.is_absolute():
        save_dir = Path(data_path) / save_dir

    # set data paths
    base_data_dir = Path("/cephfs_projects/foundation_models/data/SDE/raw/BISDE_datasets")

    if not base_data_dir.is_absolute():
        base_data_dir = Path(data_path) / base_data_dir

    # process data
    datasets_configs = [
        {
            "name": "wind",
            "delta_tau": 1 / 6,
            "transforms": ["fluctuations"],
            "data_path": base_data_dir / "wind" / "wind_speeds.pickle",
        },
        {
            "name": "oil",
            "delta_tau": 1,
            "transforms": ["fluctuations"],
            "data_path": base_data_dir / "oil" / "oil_prices.pickle",
        },
        {
            "name": "fb",
            "delta_tau": 1 / (252 * 390),
            "transforms": ["log"],
            "data_path": base_data_dir / "stonks" / "fb_stock_price.pickle",
        },
        {
            "name": "tsla",
            "delta_tau": 1 / (252 * 390),
            "transforms": ["log"],
            "data_path": base_data_dir / "stonks" / "tsla_stock_price.pickle",
        },
    ]

    num_total_splits = 5
    num_initial_states_in_val_split = 10
    num_locations = 1024

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    time: str = str(datetime.now().strftime("%Y%m%d"))
    save_data_dir: Path = save_dir / (time + "_" + subdir_label)
    save_data_dir.mkdir(parents=True, exist_ok=True)

    # get full data of each system per transform
    complete_paths: list[dict] = []

    print("Processing complete data.")
    for dataset_config in datasets_configs:
        transforms: list[str] = dataset_config.pop("transforms")

        if not isinstance(transforms, list):
            transforms = [transforms]

        for transform in transforms:
            transformed_data = process_single_trajectory(transform=transform, num_locations=num_locations, **dataset_config)
            _pprint_dict_with_shapes(transformed_data)
            print("\n")

            complete_paths.append(transformed_data)

    # split data for cross validation
    cross_val_inference_paths: list[dict] = []
    cross_val_ksig_reference_paths: list[dict] = []

    print("\n\nProcessing cross validation splits.")
    for complete_path_data in complete_paths:
        inference_paths, reference_paths = get_cross_validation_splits(
            complete_path_data, num_total_splits, num_initial_states_in_val_split
        )
        cross_val_inference_paths = cross_val_inference_paths + inference_paths
        cross_val_ksig_reference_paths = cross_val_ksig_reference_paths + reference_paths

        print("Inference data:")
        _pprint_dict_with_shapes(inference_paths)
        print("Reference data:")
        _pprint_dict_with_shapes(reference_paths)
        print("\n")

    # save data in jsons
    optree.tree_map(_check_finite, (complete_paths, cross_val_inference_paths, cross_val_ksig_reference_paths), namespace="fimsde")

    for data, filename in zip(
        [complete_paths, cross_val_inference_paths, cross_val_ksig_reference_paths],
        ["complete_paths.json", "cross_val_inference_paths.json", "cross_val_ksig_reference_paths.json"],
    ):
        # Convert to JSON
        json_data = json.dumps(data, cls=NumpyEncoder)

        file: Path = save_data_dir / filename
        with open(file, "w") as file:
            file.write(json_data)
