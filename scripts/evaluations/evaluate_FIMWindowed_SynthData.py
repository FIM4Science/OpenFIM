import itertools
import json
import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from fim.data.dataloaders import TimeSeriesDataLoaderTorch
from fim.models.FIM_models import FIMWindowed
from fim.utils.helper import load_yaml
from fim.utils.metrics import compute_metrics


torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

device_map = "cuda:0" if torch.cuda.is_available() else "cpu"


show_plots = False

data_base_dir = "data/FIMImputation/torch_500K_ode_centere_restricted_length_256_with_per_gps_no_imputation_mask/"


def evaluate_hyperparameter_setting(
    model,
    output_folder_base: str,
):
    output_path = output_folder_base
    os.makedirs(output_path, exist_ok=True)

    torch.manual_seed(0)
    dl = TimeSeriesDataLoaderTorch(
        path=data_base_dir,
        split="test",
        batch_size=1024,
        test_batch_size=1024,
        output_fields=[
            "fine_grid_grid",
            "fine_grid_concept_values",
            "fine_grid_sample_paths",
            "coarse_grid_grid",
            "coarse_grid_noisy_sample_paths",  #  "coarse_grid_sample_paths"  or "coarse_grid_noisy_sample_paths"
            "coarse_grid_observation_mask",
        ],
    )

    sample_path_prediction = []
    sample_paths_target = []
    times_prediction = []
    times_target = []
    observation_times = []
    observation_values = []
    observation_mask = []

    for batch in dl.test_it:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device_map).to(torch.float64)
            else:
                for ki, vi in v.items():
                    batch[k][ki] = vi.to(device_map).to(torch.float64)

        with torch.no_grad():
            output = model(batch)

            times_prediction.append(output["observation_times"].cpu())
            sample_path_prediction.append(output["learnt_solution_paths"].cpu())
            times_target.append(batch["fine_grid_grid"].cpu())
            sample_paths_target.append(batch["fine_grid_sample_paths"].cpu())
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
        ax.plot(times_prediction[sample_ids[i], :, dim], sample_path_prediction[sample_ids[i], :, dim], label="prediction")
    ax.legend()

    fig.suptitle("predictions")
    plt.tight_layout()
    # plt.savefig(f"{output_path}/predictions.png")
    plt.savefig("data.png")
    if show_plots:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    config_dir = "configs/inference/fim_windowed.yaml"
    config = load_yaml(config_dir)

    denoising_model_name = config["model"].get("denoising_model", {}).get("name", None)

    model_names = [
        "results/FIMODE/fim_ode_noisy_MinMax-experiment-seed-10_08-23-1331/checkpoints/best-model/model-checkpoint.pth",
        "results/FIMODE/fim_ode_noisy_RevIN-0-1-experiment-seed-10_08-23-1833/checkpoints/best-model/model-checkpoint.pth",
    ]
    window_counts = [4, 8, 16]  # , 12, 16, 20]
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
        if denoising_model_name is not None:
            config["model"]["denoising_model"]["name"] = denoising_model_name

        model = FIMWindowed(**config["model"])
        model.to(device_map)
        model.eval()

        model_abbr = model_path  # config["model"]["fim_base"]
        if "MinMax" in model_abbr:
            model_abbr = "MinMax_SavGol"
        elif "RevIN" in model_abbr:
            model_abbr = "RevIN_SavGol"
        else:
            model_abbr = None
        # window_count = config["model"]["window_count"]
        overlap = config["model"]["overlap"]
        output_folder_base = f"reports/FIMWindowed/SynthData_500k/{window_count}windows_{int(100*overlap)}%overlap/{model_abbr}/"
        os.makedirs(output_folder_base, exist_ok=True)
        evaluate_hyperparameter_setting(model, output_folder_base)

        print(f"{model_abbr}")
        print(f'windows: {window_count} overlap: {config["model"]["overlap"]}')

        # get_table_1(print_table=True)
        # metrics = get_overview_all_metrics(print=False)
        # print("r2_score_above0.9")
        # for k, v in metrics.items():
        #     print("|" + k + "|" + f"{v['r2_score_above0.9']:.4}" + "|")
