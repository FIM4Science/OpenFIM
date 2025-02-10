from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import torch

from fim.utils.experiment_files import ExperimentsFiles
from fim.models.hawkes import FIMHawkes

DATASET_DIR = Path("data/synthetic_data/hawkes/10K_5_st_hawkes_exp_500_paths_100_events/test")
EXPERIMENT_DIR = Path("results/FIM_Hawkes_1k_5_st_exp_500_paths_100_events_bigger_model-experiment-seed-10_02-10-1551")
num_batches = 5

def load_pt_in_dir(dir_path: Path):
    """
    Load all .pt files in a directory and return as a dict of tensors.
    """
    tensors = {}
    for file in dir_path.iterdir():
        if file.suffix == ".pt":
            tensors[file.stem] = torch.load(file)
    return tensors

def plot_model_predictions_and_true_values(model_predictions, data):
    for (k,v) in model_predictions.items():
        if isinstance(v, torch.Tensor):
            model_predictions[k] = v.detach().cpu().numpy()
    for (k,v) in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().cpu().numpy()
    B, M, T = model_predictions["predicted_kernel_values"].shape
    predicted_kernel_function = model_predictions["predicted_kernel_values"] * np.exp(-model_predictions["predicted_kernel_decay"][:,:,None] * data["kernel_grids"]) # + model_predictions["predicted_base_intensity"]
    ground_truth_kernel_function = data["kernel_evaluations"] # + data["base_intensities"]
    # Use B as rows and M as columns
    fig, axs = plt.subplots(B, M, figsize=(10, 10))
    for b in range(B):
        for m in range(M):
            axs[b, m].plot(data["kernel_grids"][b,m], predicted_kernel_function[b, m], label="Model")
            axs[b, m].plot(data["kernel_grids"][b,m], ground_truth_kernel_function[b, m], label="Ground Truth")
            axs[b, m].legend()    
    plt.savefig("model_vs_ground_truth.png")

if EXPERIMENT_DIR.exists():
    experiment_files = ExperimentsFiles(experiment_dir=EXPERIMENT_DIR)
    checkpoint_path = experiment_files.get_lightning_checkpoint_path("best")
    model = FIMHawkes.load_model(checkpoint_path)
    data = load_pt_in_dir(DATASET_DIR)
    for (k,v) in data.items():
        data[k] = v[:num_batches]
    model_predictions = model(data)
    plot_model_predictions_and_true_values(model_predictions, data)