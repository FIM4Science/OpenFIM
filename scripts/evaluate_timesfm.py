import os
import pickle
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timesfm
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

from fim.utils.metrics import compute_metrics


access_token = os.getenv("HUGGINGFACE_TOKEN")

load_dotenv()

# Suppress all warnings
warnings.simplefilter("ignore")

verbose = True
base_date = datetime(2021, 1, 1)


def load_data(directory: str) -> dict:
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pickle"):
            key, _, _ = filename.rpartition(".")
            with open(os.path.join(directory, filename), "rb") as f:
                data[key] = pickle.load(f)

    return data


def datetime2float(date: datetime) -> float:
    date = date.astype("M8[ns]").astype(datetime)
    delta = date - base_date
    return delta.days + delta.seconds / (60 * 60 * 24) + delta.microseconds / 86400 / 1e6


def float2datetime(x: float) -> timedelta:
    return base_date + timedelta(days=x)


def build_dfs(data: dict, horizon_len: int = 128, sigma: float = 0.0, mask_p: int = 0.0) -> tuple[pd.DataFrame]:
    """
    prepare data for model input + prepare ground truth prediction data for evaluation.
    df has columns 'unique_id', ds (time stamps in datetime!), 'values'"""

    df_input, df_output = [], []

    for sample_id in range(len(data["dim"])):
        dimensions = data["dim"][sample_id]
        mask = data[f"observation_mask_p_{str(mask_p).replace('.', '_')}"][sample_id].squeeze()
        # split mask into context and horizon
        mask_context = mask[:-horizon_len]

        for dim in range(dimensions):
            # split into context and horizon
            y_context = data[f"fine_grid_noisy_sample_paths_sigma_{sigma}"][sample_id, :-horizon_len, dim]
            y_horizon = data[f"fine_grid_noisy_sample_paths_sigma_{sigma}"][sample_id, -horizon_len:, dim]
            # convert time stamps to datetime
            ds_context = np.array([float2datetime(ts) for ts in data["fine_grid_grid"][sample_id][:-horizon_len].squeeze()])
            ds_horizon = np.array([float2datetime(ts) for ts in data["fine_grid_grid"][sample_id][-horizon_len:].squeeze()])

            # mask out values and grid
            y_context = y_context[mask_context == 1]
            # y_horizon = y_horizon[mask_horizon == 1]
            ds_context = ds_context[mask_context == 1]
            # ds_horizon = ds_horizon[mask_horizon == 1]

            # store in df
            df_input.append(
                pd.DataFrame(
                    {
                        "unique_id": f"series_{sample_id}__{dim}",
                        "series_id": f"series_{sample_id}",
                        "ds": ds_context,
                        "values": y_context,
                    }
                )
            )
            df_output.append(
                pd.DataFrame(
                    {
                        "unique_id": f"series_{sample_id}__{dim}",
                        "series_id": f"series_{sample_id}",
                        "ds": ds_horizon,
                        "values": y_horizon,
                    }
                )
            )

    df_input = pd.concat(df_input)
    df_output = pd.concat(df_output)

    return df_input, df_output


def present_data_nicely():
    import json
    import re
    from collections import defaultdict

    def compute_average_mean(data_dict):
        average_means = {}

        for noise_level, dim_data in data_dict.items():
            total_means = defaultdict(float)
            count = defaultdict(int)

            for dim, metrics in dim_data.items():
                for metric, values in metrics.items():
                    total_means[metric] += values["mean"]
                    count[metric] += 1

            average_means[noise_level] = {metric: total_means[metric] / count[metric] for metric in total_means}

        return average_means

    def parse_data(data):
        mask_p_0 = defaultdict(dict)
        mask_p_5 = defaultdict(dict)

        current_dict = None
        current_noise = None
        current_dim = None

        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue

            # Identify which mask_p we are processing
            if line.startswith("mask_p: 0.0"):
                current_dict = mask_p_0
            elif line.startswith("mask_p: 0.5"):
                current_dict = mask_p_5
            elif line.startswith("dim:"):
                dim_noise_match = re.match(r"dim: (\d+), noise: ([0-9.]+)", line)
                if dim_noise_match:
                    current_dim = int(dim_noise_match.group(1))
                    current_noise = float(dim_noise_match.group(2))
                    current_dict[f"noise_{current_noise}"][f"dim_{current_dim}"] = {}
            else:
                if current_dict is not None and current_noise is not None and current_dim is not None:
                    metric_match = re.match(r"(\w+)\s+([\d.-]+),([\d.-]+)", line)
                    if metric_match:
                        metric_name = metric_match.group(1)
                        mean_value = float(metric_match.group(2))
                        std_value = float(metric_match.group(3))
                        current_dict[f"noise_{current_noise}"][f"dim_{current_dim}"][metric_name] = {
                            "mean": mean_value,
                            "std": std_value,
                        }

        return mask_p_0, mask_p_5

    data_dir = "results/timesfm/ode_bench/metrics_summary.txt"
    with open(data_dir, "r") as file:
        data = file.read()

    mask_p_0, mask_p_5 = parse_data(data)

    # Compute average mean for mask_p_0 and mask_p_5
    average_mean_mask_p_0 = compute_average_mean(mask_p_0)
    average_mean_mask_p_5 = compute_average_mean(mask_p_5)

    # Print results
    print("Average mean for mask_p_0:")
    for noise_level, averages in average_mean_mask_p_0.items():
        print(f"{noise_level}: {json.dumps(averages, indent=4)}")

    print("\nAverage mean for mask_p_5:")
    for noise_level, averages in average_mean_mask_p_5.items():
        print(f"{noise_level}: {json.dumps(averages, indent=4)}")


def visualize_random_samples(result_dir, mask_p, dim, sigma, df_input, df_output, prediction_df):
    vis_data = []
    for unique_id in prediction_df["unique_id"].unique():
        predictions = prediction_df[prediction_df["unique_id"] == unique_id]["timesfm"].values
        ground_truth = df_output[df_output["unique_id"] == unique_id]["values"].values
        context = df_input[df_input["unique_id"] == unique_id]["values"].values

        grid_context = df_input[df_input["unique_id"] == unique_id]["ds"].values.astype(float)

        grid_horizon = df_output[df_output["unique_id"] == unique_id]["ds"].values.astype(float)

        vis_data.append(
            {
                "prediction": predictions,
                "ground_truth": ground_truth,
                "context": context,
                "grid_context": grid_context,
                "grid_horizon": grid_horizon,
            }
        )

    plots = (4, 4)
    fig, axes = plt.subplots(*plots, figsize=(20, 20))
    # sample 16 series
    series_ids = np.random.choice(len(vis_data), plots[0] * plots[1])
    for i, ax in enumerate(axes.flat):
        sample = vis_data[i]

        ax.plot(sample["grid_context"], sample["context"], label="context")
        ax.plot(sample["grid_horizon"], sample["ground_truth"], label="ground truth")
        ax.plot(sample["grid_horizon"], sample["prediction"], label="prediction")
        ax.set_title(f"series {series_ids[i]}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(result_dir + f"{mask_p}_{dim}_{sigma}.png")


if __name__ == "__main__":
    data_base_dir = "/cephfs_projects/foundation_models/data/ode_bench_preprocessed"
    result_base_dir = "results/timesfm/ode_bench/"

    if "tfm" not in globals():
        login(access_token)
        # load model + checkpoint
        tfm = timesfm.TimesFm(
            context_len=512,
            horizon_len=128,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend="cuda",
        )
        tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

    for freq in tqdm(["H", "W", "Y"], desc="freq", position=0, leave=True):
        result_dir = result_base_dir + f"freq_{freq}/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_dir_summary = result_dir + "metrics_summary.csv"

        for mask_p in tqdm([0.0, 0.5], desc="mask_p", position=1, leave=True):
            for dim in tqdm(range(1, 4), desc="dim", position=2, leave=True):
                subfolder = f"equations_of_dimension_{dim}"

                for sigma in tqdm([0.0, 0.01, 0.02, 0.03, 0.04, 0.05], desc="sigma", position=3, leave=True):
                    data: dict = load_data(os.path.join(data_base_dir, subfolder))
                    df_input, df_output = build_dfs(data, sigma=sigma, mask_p=mask_p)

                    prediction_df = tfm.forecast_on_df(df_input, freq=freq)
                    # add series_id
                    prediction_df["series_id"] = prediction_df["unique_id"].apply(lambda x: x.split("__")[0])

                    # compute metrics
                    metrics = []
                    for unique_series_id in prediction_df["series_id"].unique():
                        # ie select over all dimensions of a single series
                        predictions = prediction_df[prediction_df["series_id"] == unique_series_id]["timesfm"].values
                        ground_truth = df_output[df_output["series_id"] == unique_series_id]["values"].values
                        # reshape to (horizon_len, n_ims)
                        predictions = predictions.reshape(-1, dim)
                        ground_truth = ground_truth.reshape(-1, dim)

                        metrics.append(
                            {
                                "unique_id": unique_series_id,
                                "freq": freq,
                                "dim": dim,
                                "mask_p": mask_p,
                                "sigma": sigma,
                                **compute_metrics(predictions, ground_truth),
                            }
                        )
                        # break

                    metrics_single_config = pd.DataFrame(metrics)

                    metrics_single_config.to_csv(os.path.join(result_dir, f"metrics_{mask_p}_{dim}_{sigma}.csv"), index=False)

                    # visualization
                    # visualize_random_samples(result_dir, mask_p, dim, sigma, df_input, df_output, prediction_df)
            #         break
            #     break
            # break
        dfs = []
        for file in os.listdir(result_dir):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(result_dir, file))
                dfs.append(df)

        metrics_summary = pd.concat(dfs)
        metrics_summary.to_csv(result_dir_summary, index=False)


print("done")
