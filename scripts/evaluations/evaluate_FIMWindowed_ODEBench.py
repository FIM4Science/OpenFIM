""" "
- ODEBench settings:
    - Hyperparameters:
        - Bernoulli_sampling parameter: rho : {0_0, 0_5}, -> output_field = f"observation_mask_p_{rho}"
        - noise parameter sigma  {0.0, 0.1, 0.2, 0.3, 0.4, 0.5} -> output_field = f"fine_grid_noisy_sample_paths_sigma_{sigma}"
        - dimension: {1,2,3,4} -> subfolder = f"equations_of_dimension_{dim}"
- Model:
    - fimbase model + checkpoint
    - window size + overlap

- proceedings:
    - forward pass FIMWindowed
        - get predictions for each sample
    - compute metrics for each sample (need: target & prediction)
        - use script
    - save
        - save metrics (avg & raw) (jsonl)
        - save predictions (jsonl)
"""

import itertools
import json
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from fim.data.dataloaders import TimeSeriesDataLoaderTorch
from fim.models.FIM_models import FIMWindowed
from fim.utils.helper import load_yaml
from fim.utils.metrics import compute_metrics


torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

device_map = "cuda:0" if torch.cuda.is_available() else "cpu"


rho = "0_0"
sigma = "0.0"
dim = 1

show_plots = False

data_base_dir = "/cephfs_projects/foundation_models/data/ode_bench_preprocessed"


def load_ODEBench_as_torch(directory: str) -> dict:
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pickle"):
            key, _, _ = filename.rpartition(".")
            with open(os.path.join(directory, filename), "rb") as f:
                data[key] = pickle.load(f)

    # convert numpy arrays to torch tensors
    for key, value in data.items():
        if isinstance(value, tuple):
            value = value[0]
        if isinstance(value, np.ndarray):
            data[key] = torch.tensor(value, dtype=torch.float64)
        else:
            raise TypeError(f"Expected numpy array, got {type(value)}")
        if "mask" in key:
            data[key] = ~data[key].bool()
    return data


def evaluate_hyperparameter_setting(model, dim: int, sigma: str, rho: str, output_folder_base: str):
    output_path = output_folder_base + f"dim{dim}_sigma{sigma}_rho{rho}"
    os.makedirs(output_path, exist_ok=True)

    torch.manual_seed(0)
    dl = TimeSeriesDataLoaderTorch(
        path=data_base_dir + "/",
        ds_name=f"equations_of_dimension_{dim}",
        split="test",
        batch_size=1,
        test_batch_size=1,
        dataset_kwargs={"loading_function": load_ODEBench_as_torch},
        output_fields=[
            "fine_grid_drift",
            "fine_grid_grid",
            f"fine_grid_noisy_sample_paths_sigma_{sigma}",
            "fine_grid_sample_paths",
            f"observation_mask_p_{rho}",
            "id",
        ],
    )

    sample_path_prediction = []
    sample_paths_target = []
    times_prediction = []
    times_target = []
    observation_times = []
    observation_values = []
    observation_mask = []
    ids = []

    key_mapping = {
        "fine_grid_drift": "fine_grid_concept_values",
        "fine_grid_grid": "coarse_grid_grid",
        f"fine_grid_noisy_sample_paths_sigma_{sigma}": "coarse_grid_noisy_sample_paths",
        "fine_grid_sample_paths": "fine_grid_sample_paths",
        f"observation_mask_p_{rho}": "coarse_grid_observation_mask",
        "id": "id",
    }

    for batch in dl.test_it:
        # prepare batch
        batch = {key_mapping.get(k, k): v.to(device_map) for k, v in batch.items()}
        batch["fine_grid_grid"] = batch["coarse_grid_grid"]
        batch.pop("fine_grid_concept_values")

        with torch.no_grad():
            output = model(batch)

            times_prediction.append(output["observation_times"].cpu())
            sample_path_prediction.append(output["learnt_solution_paths"].cpu())
            times_target.append(batch["fine_grid_grid"].cpu())
            sample_paths_target.append(batch["fine_grid_sample_paths"].cpu())
            ids.append(batch["id"].cpu())
            observation_times.append(batch["coarse_grid_grid"].cpu())
            observation_values.append(batch["coarse_grid_noisy_sample_paths"].cpu())
            observation_mask.append(batch["coarse_grid_observation_mask"].cpu())

    sample_path_prediction = torch.cat(sample_path_prediction, dim=0)
    sample_paths_target = torch.cat(sample_paths_target, dim=0)
    times_prediction = torch.cat(times_prediction, dim=0)
    times_target = torch.cat(times_target, dim=0)
    observation_times = torch.cat(observation_times, dim=0)
    observation_values = torch.cat(observation_values, dim=0)
    observation_mask = torch.cat(observation_mask, dim=0)
    ids = torch.cat(ids, dim=0)

    # print(times_target.shape)

    # compute metrics
    metrics = compute_metrics(sample_path_prediction, sample_paths_target)

    # save metrics
    json.dump(metrics, open(f"{output_path}/metrics.json", "w"))

    # save predictions
    torch.save(
        {
            "sample_paths_target": sample_paths_target,
            "sample_path_prediction": sample_path_prediction,
            "times_target": times_target,
            "times_prediction": times_prediction,
            "observation_times": observation_times,
            "observation_values": observation_values,
            "observation_mask": observation_mask,
            "ids": ids,
        },
        f"{output_path}/predictions.pt",
    )

    # plot 9 samples
    sample_ids = list(range(9))
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    dim = 0
    for i, ax in enumerate(axes.flatten()):
        ax.plot(
            times_target[sample_ids[i], :, dim],
            sample_paths_target[sample_ids[i], :, dim],
            label="target",
            linestyle="--",
        )
        ax.plot(
            times_prediction[sample_ids[i], :, dim], sample_path_prediction[sample_ids[i], :, dim], label="prediction"
        )
    ax.legend()

    fig.suptitle(f"ODEBench predictions for sigma={sigma}, rho={rho}")
    plt.tight_layout()
    # plt.savefig(f"{output_path}/predictions.png")
    plt.savefig("data.png")
    if show_plots:
        plt.show()
    else:
        plt.close()


def print_dict_as_table(data_dict):
    # Extract the header from the first item
    headers = list(next(iter(data_dict.values())).keys())

    # Print the table header
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

    print(header_row)
    print(separator_row)

    # Print each row
    for key, values in data_dict.items():
        row = key + " | " + " | ".join(f"{values[h]:.4}" for h in headers) + " |"
        print(row)


def get_overview_all_metrics(
    print: bool = True,
):
    metrics: dict = {}
    for dim, sigma, rho in itertools.product(dims, sigmas, rhos):
        key = f"dim{dim}_sigma{sigma}_rho{rho}"
        metrics[key] = json.load(open(f"reports/FIMWindowed/ODEBench/{key}/metrics.json"))
    if print:
        print_dict_as_table(metrics)
    else:
        return metrics


def get_table_1(print_table: bool = True):
    """R2 score > 90% for different sigma & rho, averaged over dims"""
    metrics = get_overview_all_metrics(print=False)
    table1 = defaultdict(int)
    for dim, sigma, rho in itertools.product(dims, sigmas, rhos):
        key = f"dim{dim}_sigma{sigma}_rho{rho}"
        new_key = f"sigma{sigma}_rho{rho}"
        table1[new_key] += metrics[key]["r2_score_above0.9"]

    for key in table1.keys():
        table1[key] /= len(dims)
    if print_table:
        print("| " + " | ".join(table1.keys()) + " |")
        print("| " + " | ".join(["---"] * len(table1.keys())) + " |")
        print("| " + " | ".join(f"{v:.4}" for v in table1.values()) + " |")

        print("\n\n")
        table1_keys = ["sigma0.0_rho0_0", "sigma0.0_rho0_5", "sigma0.05_rho0_0", "sigma0.05_rho0_5"]
        print("| " + " | ".join(table1_keys) + " |")
        print("| " + " | ".join(["---"] * len(table1_keys)) + " |")
        print("| " + " | ".join(f"{table1[k]:.4}" for k in table1_keys) + " |")
    else:
        return table1


if __name__ == "__main__":
    config_dir = "/home/koerner/FIM/configs/inference/fim_windowed.yaml"
    config = load_yaml(config_dir)

    # denoising_model_name = config["model"].get("denoising_model", {}).get("name", None)

    dims = [1, 2, 3, 4]
    sigmas = ["0.0", "0.05"]  # "0.01", "0.02", "0.03", "0.04", ]
    rhos = ["0_0", "0_5"]
    model_names = [
        "results/fim_ode_noisy_MinMax-experiment-seed-10_08-23-1331/checkpoints/best-model/model-checkpoint.pth",
        "results/fim_ode_noisy_RevIN-0-1-experiment-seed-10_08-23-1833/checkpoints/best-model/model-checkpoint.pth",
    ]
    window_counts = [8, 12, 16, 20]
    # overlaps = [0.1, 0.2, 0.3, 0.4, 0.5]

    for window_count, model_path in tqdm(
        itertools.product(window_counts, model_names),
        desc="models & window counts",
        total=len(window_counts) * len(model_names),
        leave=False,
        position=1,
    ):
        config["model"]["window_count"] = window_count
        config["model"]["fim_base"] = model_path
        # if denoising_model_name is not None:
        #     config["model"]["denoising_model"]["name"] = denoising_model_name

        model = FIMWindowed(**config["model"])
        model.to(device_map)
        model.eval()

        model_abbr = model_path  # config["model"]["fim_base"]
        if "MinMax" in model_abbr:
            model_abbr = "MinMax"
        elif "RevIN" in model_abbr:
            model_abbr = "RevIN"
        else:
            model_abbr = None
        # window_count = config["model"]["window_count"]
        overlap = config["model"]["overlap"]
        output_folder_base = (
            f"reports/FIMWindowed/ODEBench/{window_count}windows_{int(100*overlap)}%overlap/{model_abbr}/"
        )
        os.makedirs(output_folder_base, exist_ok=True)
        # print(f"Running evaluation for window_count={window_count}, model={model_abbr}")
        # use itertools.product to loop over all hyperparameters
        for dim, sigma, rho in tqdm(
            itertools.product(dims, sigmas, rhos),
            total=len(dims) * len(sigmas) * len(rhos),
            desc=f"ODEBench settings for wc={window_count}, model={model_abbr}",
            leave=False,
        ):
            # tqdm.write(f"Running evaluation for dim={dim}, sigma={sigma}, rho={rho}")
            # print(f"Running evaluation for dim={dim}, sigma={sigma}, rho={rho}")
            evaluate_hyperparameter_setting(model, dim, sigma, rho, output_folder_base)

        print(f"{model_abbr}")
        print(f'windows: {window_count} overlap: {config["model"]["overlap"]}')

        # get_table_1(print_table=True)
        # metrics = get_overview_all_metrics(print=False)
        # print("r2_score_above0.9")
        # for k, v in metrics.items():
        #     print("|" + k + "|" + f"{v['r2_score_above0.9']:.4}" + "|")
