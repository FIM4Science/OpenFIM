"""Evaluate FIMImputation on synthetic data. Compute metrics and save predictions."""

import json
import os

import torch
from tqdm import tqdm

from fim.data.dataloaders import TimeSeriesDataLoaderTorch
from fim.models.fim_imputation import FIMImputation
from fim.models.utils import load_model_from_checkpoint
from fim.utils.metrics import compute_metrics


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def evaluate_one_configuration(
    model_checkpoint_path: str, dl: TimeSeriesDataLoaderTorch, output_base_dir: str, model_abbr: str = None
):
    # check if model checkpoint path ends with pth or pt
    if not (model_checkpoint_path.endswith(".pth") or model_checkpoint_path.endswith(".pt")):
        raise ValueError("Model checkpoint path has to point to a .pth or .pt file!")

    if model_abbr is None:
        # if "MinMax" in model_checkpoint_path:
        #     model_abbr = "MinMax"
        # elif "RevIN" in model_checkpoint_path:
        #     "RevIN"
        # else:
        #     raise ValueError("Unknown model type")
        model_abbr = model_checkpoint_path.split("/")[-4].split("_")[-1]

    output_path = output_base_dir + model_abbr + "/"
    os.makedirs(output_path, exist_ok=True)

    model = load_model_from_checkpoint(model_checkpoint_path, module=FIMImputation, for_eval=True)

    predictions = {}
    try:
        loader = dl.test_it
    except KeyError:
        loader = dl.train_it
    for id, batch in tqdm(enumerate(loader), desc=f"Evaluating {model_abbr}", total=len(loader)):
        if id == 0:
            with torch.no_grad():
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
    target_sample_paths = predictions["imputation_window"]["target"]
    pred_sample_paths = predictions["imputation_window"]["learnt"]
    metrics = compute_metrics(pred_sample_paths, target_sample_paths)

    json.dump(metrics, open(output_path + "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    # data_path = "data/20k_ImputationDummy/"
    data_path = "data/FIMImputation/torch_500K_ode_centere_restricted_length_256_with_per_gps_no_imputation_mask/"

    output_base_dir = "reports/FIMImputation/SynthData/all/"
    # output_base_dir = "reports/FIMImpuation/DummyData_20k/"

    batch_size = 4096

    torch.manual_seed(0)
    dl = TimeSeriesDataLoaderTorch(
        path=data_path,
        split="test",
        batch_size=batch_size,
        test_batch_size=batch_size,
        dataset_name="fim.data.datasets.TimeSeriesImputationDatasetTorch",
        output_fields=[""],
        loader_kwargs={"num_workers": 8},
        dataset_kwargs={
            "output_fields_fimbase": [
                "fine_grid_grid",
                "fine_grid_concept_values",
                "fine_grid_sample_paths",
                "coarse_grid_grid",
                "coarse_grid_noisy_sample_paths",  #  "coarse_grid_sample_paths"  or "coarse_grid_noisy_sample_paths"
                "coarse_grid_observation_mask",
            ],
            "debugging_data_range": 2000,
            "window_count": 5,
            "overlap": 0,
            "imputation_mask": (False, False, True, False, False),
            "max_sequence_length": 256,
        },
    )

    model_chkpts = [
        # "results/FIMImputation/SynthData_all_5w_MinMax_MinMax_nllh_sfvGlobNorm_LRcosAn_3encBlocks_fp16-experiment-seed-10_09-08-2131/checkpoints/best-model/model-checkpoint.pth",
        # "results/FIMImputation/SynthData_all_5w_MinMax_MinMax_nllh_sfvGlobNorm_LRcosAn_4encBlocks-experiment-seed-4_09-09-1541/checkpoints/best-model/model-checkpoint.pth",
        # "results/FIMImputation/SynthData_all_3w_MinMax_MinMax_nllh_sfvGlobNorm_LRcosAn_4encBlocks-experiment-seed-4_09-09-1549/checkpoints/best-model/model-checkpoint.pth",
        "results/FIMImputation/SynthData_all_5w_MinMax_MinMax_nllh_sfvGlobNorm_LRcosAn_4encBlocks-experiment-seed-4_09-13-1636/checkpoints/best-model/model-checkpoint.pth",
        # "results/FIMImputation/SynthData_all_3w_MinMax_MinMax_nllh_sfvGlobNorm_LRcosAn_4encBlocks-experiment-seed-4_09-13-1635/checkpoints/best-model/model-checkpoint.pth",
    ]
    model_abbrs = [
        # "09-08-2131_epoch-_test",
        # "09-09-1541_epoch-_test",
        # "09-09-1549_epoch-_test",
        "09-13-1636_epoch-369_test",
        # "09-13-1635_epoch-_test",
    ]
    for model_chkpt, model_abbr in zip(model_chkpts, model_abbrs):
        evaluate_one_configuration(model_chkpt, dl, output_base_dir, model_abbr=model_abbr)
        # model = load_model_from_checkpoint(model_chkpt, module=FIMImputation, for_eval=True)

    print("done")
