"""
1. create dataloader
2. load model [from list of dirs]
3. evaluate model on test set -> save ground truth, observations, predictions
4. compute metrics
"""

import json
import os

import torch
from tqdm import tqdm

from fim.data.dataloaders import TimeSeriesDataLoaderTorch
from fim.models.models import FIMODE
from fim.models.utils import load_model_from_checkpoint
from fim.utils.metrics import compute_metrics


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def evaluate_one_configuration(model_checkpoint_path: str, dl: TimeSeriesDataLoaderTorch, output_base_dir: str):
    if "MinMax" in model_checkpoint_path:
        model_abbr = "MinMax"
    elif "RevIN" in model_checkpoint_path:
        "RevIN"
    else:
        raise ValueError("Unknown model type")

    output_path = output_base_dir + model_abbr + "/"
    os.makedirs(output_path, exist_ok=True)

    model = load_model_from_checkpoint(model_checkpoint_path, module=FIMODE, for_eval=True)

    predictions = {}
    for id, batch in tqdm(enumerate(dl.test_it), desc=f"Evaluating {model_abbr}"):
        if id == 0:
            predictions = model(batch)["visualizations"]
        else:
            output = model(batch)["visualizations"]
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    predictions[k] = torch.cat([predictions[k], v], dim=0)
                else:
                    for ki, vi in v.items():
                        predictions[k][ki] = torch.cat([predictions[k][ki], vi], dim=0)

    torch.save(
        predictions,
        output_path + "predictions.pth",
    )

    # compute metrics
    target_sample_paths = predictions["solution"]["target"]
    pred_sample_paths = predictions["solution"]["sample_paths"]
    metrics = compute_metrics(pred_sample_paths, target_sample_paths)

    json.dump(metrics, open(output_path + "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    data_path = "data/torch_2M_ode_chebyshev_max_deg_100_rbf_gp_2_5_and_2_10_length_128_avg_min_8/"

    output_base_dir = "reports/FIMBase/SynthData/"

    batch_size = 4096

    torch.manual_seed(0)
    dl = TimeSeriesDataLoaderTorch(
        path=data_path,
        split="test",
        batch_size=batch_size,
        test_batch_size=batch_size,
        output_fields=[
            "fine_grid_grid",
            "fine_grid_concept_values",
            "fine_grid_sample_paths",
            "coarse_grid_grid",
            "coarse_grid_noisy_sample_paths",  #  "coarse_grid_sample_paths"  or "coarse_grid_noisy_sample_paths"
            "coarse_grid_observation_mask",
        ],
    )

    model_chkpts = [
        "results/fim_ode_noisy_MinMax-experiment-seed-10_08-23-1331/checkpoints/best-model/model-checkpoint.pth"
    ]
    for model_chkpt in model_chkpts:
        evaluate_one_configuration(model_chkpt, dl, output_base_dir)
