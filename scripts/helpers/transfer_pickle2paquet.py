"""
Transfer data from pickle to paquet data and from shared storage to local one.
Needed for data conversion for faster data set generation in data loader.

Takes test set from /cephfs directory and splits it into test and validation sets by taking every second element.
"""

import pickle
from typing import Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch


def read_pickle2torch(file_path: str) -> Union[torch.FloatTensor, tuple[torch.FloatTensor]]:
    with open(file_path, "rb") as f:
        jax_data = pickle.load(f)
        # convert Jax -> numpy (need copy to make it writeable) -> torch
        if isinstance(jax_data, tuple):
            # needed for concepts
            return (
                torch.from_numpy(np.asarray(jax_data[0]).copy()) if jax_data[0] is not None else None,
                torch.from_numpy(np.asarray(jax_data[1]).copy()) if jax_data[1] is not None else None,
            )

        return torch.from_numpy(np.asarray(jax_data).copy())  # type: torch.FloatTensor


def read_data(path, split):
    coarse_grid_concept_values = read_pickle2torch(path + split + "/" + "coarse_grid_concept_values.pickle")
    fine_grid_noisy_sample_paths = read_pickle2torch(path + split + "/" "fine_grid_noisy_sample_paths.pickle")
    coarse_grid_grid = read_pickle2torch(path + split + "/" "coarse_grid_grid.pickle")
    coarse_grid_observation_mask = read_pickle2torch(path + split + "/" "coarse_grid_observation_mask.pickle")
    fine_grid_grid = read_pickle2torch(path + split + "/" "fine_grid_grid.pickle")
    coarse_grid_sample_paths = read_pickle2torch(path + split + "/" "coarse_grid_sample_paths.pickle")

    data = {
        "fine_grid_noisy_sample_paths": fine_grid_noisy_sample_paths,
        "fine_grid_concept_values": coarse_grid_concept_values,
        "coarse_grid_grid": coarse_grid_grid,
        "coarse_grid_observation_mask": coarse_grid_observation_mask,
        "fine_grid_grid": fine_grid_grid,
        "coarse_grid_sample_paths": coarse_grid_sample_paths,
    }

    data["coarse_grid_observation_mask"] = ~data["coarse_grid_observation_mask"].bool()
    data["fine_grid_sample_paths"] = data["coarse_grid_sample_paths"]
    if len(data["fine_grid_concept_values"]) > 1:
        data["fine_grid_concept_values"] = data["fine_grid_concept_values"][0]

    print(
        "avg. number of masked values: "
        + str(torch.mean(torch.sum(data["coarse_grid_observation_mask"], dim=1).float()).item()),
        flush=True,
    )

    # get indices of obs_mask rows where sum of dim 1  = 128
    indices = torch.where(torch.sum(data["coarse_grid_observation_mask"], dim=1) == 128)[0].tolist()
    print(f"number of dropped observations: {len(indices)}", flush=True)

    indices.insert(-1, 0)
    indices.append(None)
    for k, v in data.items():
        data[k] = torch.concat([v[s1 + 1 : s2] for s1, s2 in zip(indices, indices[1:])], dim=0)

    return data


def save_data(data: dict, split: str, target_path: str):
    df = pd.DataFrame(
        {
            "coarse_grid_grid": data["coarse_grid_grid"].tolist(),
            "coarse_grid_sample_paths": data["coarse_grid_sample_paths"].tolist(),
            "coarse_grid_observation_mask": data["coarse_grid_observation_mask"].tolist(),
            "fine_grid_grid": data["fine_grid_grid"].tolist(),
            "fine_grid_sample_paths": data["fine_grid_sample_paths"].tolist(),
            "fine_grid_concept_values": data["fine_grid_concept_values"].tolist(),
        }
    )

    print("Data converted to pandas DataFrame.", flush=True)
    # convert to paquet data
    table = pa.Table.from_pandas(df)
    pq.write_table(table, target_path + split + ".parquet")
    print(f"Data {split} saved as parquet file.", flush=True)


data_source_path = (
    "/cephfs_projects/foundation_models/data/2M_ode_chebyshev_max_deg_100_rbf_gp_2_5_and_2_10_length_128_avg_min_8/"
)
data_destination_path = "data/2M_ode_chebyshev_max_deg_100_rbf_gp_2_5_and_2_10_length_128_avg_min_8/"
split = "train"
data_train = read_data(data_source_path, split)
save_data(data_train, split, target_path=data_destination_path)

#####

split = "test"
data = read_data(data_source_path, split)

# split into test and validation
data_test, data_val = {}, {}
for k, v in data.items():
    data_test[k] = v[::2]
    data_val[k] = v[1::2]

print("Data split into test and validation.", flush=True)
print("Saving test data...", flush=True)
save_data(data_test, "test", target_path=data_destination_path)

print("Saving validation data...", flush=True)
save_data(data_val, "validation", target_path=data_destination_path)
