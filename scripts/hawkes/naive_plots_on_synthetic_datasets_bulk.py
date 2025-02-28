from pathlib import Path

import torch
from matplotlib import pyplot as plt

from fim.models.hawkes import FIMHawkes
from fim.utils.experiment_files import ExperimentsFiles


DATASET_DIR = Path("data/synthetic_data/hawkes/1K_25_st_hawkes_exp_1000_paths_100_events/test")
EXPERIMENT_DIR = Path("results/FIM_Hawkes_bulk_exp_1000_paths_mixed_100_events_mixed-experiment-seed-10_02-13-1627")
num_samples = 5


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
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().cpu().numpy()

    B, M, T = data["kernel_evaluations"].shape
    predicted_kernel_function = model_predictions
    ground_truth_kernel_function = data["kernel_evaluations"] + data["base_intensities"][:, :, None]

    # Define scaling factors
    width_per_subplot = 3  # Adjust as needed
    height_per_subplot = 3  # Adjust as needed

    figsize = (width_per_subplot * M, height_per_subplot * B)
    fig, axs = plt.subplots(B, M, figsize=figsize, squeeze=False)

    for b in range(B):
        for m in range(M):
            axs[b, m].plot(data["kernel_grids"][b, m], predicted_kernel_function[b, m], label="Model")
            axs[b, m].plot(data["kernel_grids"][b, m], ground_truth_kernel_function[b, m], label="Ground Truth")
            axs[b, m].legend()
            axs[b, m].tick_params(axis="both", which="major", labelsize=8)  # Optional: adjust tick label size

    plt.tight_layout()
    plt.savefig("model_vs_ground_truth.png", dpi=300)
    plt.close()


def perform_inference_for_all_marks(model, data):
    B, M, T = data["kernel_evaluations"].shape
    res = []
    for i in range(num_samples):
        tmp = []
        for m in range(M):
            new_data = {
                "event_times": data["event_times"][i][None],
                "event_types": torch.where(
                    data["event_types"][i][None] == m,
                    torch.zeros_like(data["event_types"][i][None]),
                    torch.ones_like(data["event_types"][i][None]),
                ),
                "kernel_grids": data["kernel_grids"][i, m][None, None],
            }
            model_predictions = model(new_data)
            predicted_kernel_function = (
                model_predictions["predicted_kernel_values"] + model_predictions["predicted_base_intensity"][:, :, None]
            )
            tmp.append(predicted_kernel_function[0].detach())
        res.append(torch.stack(tmp))
    return torch.stack(res).squeeze(2).detach().cpu().numpy()


if EXPERIMENT_DIR.exists():
    experiment_files = ExperimentsFiles(experiment_dir=EXPERIMENT_DIR)
    checkpoint_path = experiment_files.get_lightning_checkpoint_path("best")
    model = FIMHawkes.load_model(checkpoint_path)
    model.eval()
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
    data = load_pt_in_dir(DATASET_DIR)
    for k, v in data.items():
        data[k] = v[:num_samples].to("cuda") if torch.is_tensor(v) else v[:num_samples]
    model_predictions = perform_inference_for_all_marks(model, data)
    plot_model_predictions_and_true_values(model_predictions, data)
