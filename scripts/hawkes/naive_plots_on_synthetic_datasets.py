from pathlib import Path

import torch
from matplotlib import pyplot as plt

from fim.models.hawkes import FIMHawkes
from fim.utils.experiment_files import ExperimentsFiles


DATASET_DIR = Path("data/synthetic_data/hawkes/5k_3_st_hawkes_mixed_2000_paths_250_events/test")
EXPERIMENT_DIR = Path(
    # "results/FIM_Hawkes_1-3_st_new_loss_delta_t_norm_exp_2000_paths_mixed_250_events_mixed-experiment-seed-10-dataset-dataset_kwargs-field_name_for_dimension_grouping-base_intensities_03-19-1037/checkpoints/best-model"
    "results/FIM_Hawkes_1-3_st_weird_norm_exp_2000_paths_mixed_250_events_mixed-experiment-seed-10-dataset-dataset_kwargs-field_name_for_dimension_grouping-base_intensities_03-22-1420/checkpoints/best-model"
)
num_samples = 10
start_idx = 0
SEQ_LEN = 250
NUM_PATHS = 2000


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
    for k, v in model_predictions.items():
        if isinstance(v, torch.Tensor):
            model_predictions[k] = v.detach().cpu().numpy()
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().cpu().numpy()
    B, M, T = model_predictions["predicted_kernel_values"].shape
    predicted_kernel_function = model_predictions["predicted_kernel_values"] + model_predictions["predicted_base_intensity"][:, :, None]
    ground_truth_kernel_function = data["kernel_evaluations"] + data["base_intensities"][:, :, None]

    # Define scaling factors
    width_per_subplot = 3  # Adjust as needed
    height_per_subplot = 3  # Adjust as needed

    figsize = (width_per_subplot * M, height_per_subplot * B)
    fig, axs = plt.subplots(B, M, figsize=figsize, squeeze=False)

    kernel_rmse = torch.sqrt(torch.mean(torch.tensor((predicted_kernel_function - ground_truth_kernel_function)) ** 2, dim=-1))
    print("RMSE: ", torch.mean(kernel_rmse))
    for b in range(B):
        for m in range(M):
            axs[b, m].plot(data["kernel_grids"][b, m], predicted_kernel_function[b, m], label="Model")
            axs[b, m].plot(data["kernel_grids"][b, m], ground_truth_kernel_function[b, m], label="Ground Truth")
            # Also plot the base intensities as -- lines
            axs[b, m].axhline(data["base_intensities"][b, m], color="black", linestyle="--", label="Base Intensity")
            # Also predicted base intensity
            axs[b, m].axhline(
                model_predictions["predicted_base_intensity"][b, m], color="red", linestyle="--", label="Predicted Base Intensity"
            )
            axs[b, m].legend()
            axs[b, m].tick_params(axis="both", which="major", labelsize=8)  # Optional: adjust tick label size

    plt.tight_layout()
    plt.savefig("model_vs_ground_truth.png", dpi=300)
    plt.close()


if EXPERIMENT_DIR.exists():
    experiment_files = ExperimentsFiles(experiment_dir=EXPERIMENT_DIR)
    # checkpoint_path = experiment_files.get_lightning_checkpoint_path("best")
    model = FIMHawkes.from_pretrained(EXPERIMENT_DIR)
    model.eval()
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
    data = load_pt_in_dir(DATASET_DIR)
    for k, v in data.items():
        data[k] = v[start_idx : start_idx + num_samples].to(model.device) if torch.is_tensor(v) else v[:num_samples]
    data["event_times"] = data["event_times"][:, :NUM_PATHS, :SEQ_LEN]
    data["event_types"] = data["event_types"][:, :NUM_PATHS, :SEQ_LEN]
    model_predictions = model(data)
    plot_model_predictions_and_true_values(model_predictions, data)
