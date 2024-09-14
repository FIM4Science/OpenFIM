import json
import os
import pickle
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch

from fim.data.utils import make_multi_dim, make_single_dim
from fim.models.fim_imputation import FIMImputation
from fim.models.utils import load_model_from_checkpoint
from fim.utils.metrics import compute_metrics


data_dir = "/cephfs_projects/foundation_models/data/neurips_baseline_comparison_data/cmu_mocap_43_preprocessed_imputation_percentage_0_2/"
pca_component_count = 3


def load_data(directory: str) -> dict:
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pickle"):
            key, _, _ = filename.rpartition(".")
            with open(os.path.join(directory, filename), "rb") as f:
                v = pickle.load(f)
            if isinstance(v, dict):
                v = (torch.tensor(v["eigenvectors"]), torch.tensor(v["eigenvalues"]))
            else:
                v = torch.tensor(v)
            data[key] = v
    return data


def repeat_for_dim(t, process_dim):
    return torch.concat([t for _ in range(process_dim)], dim=0)


def pad_window(x, target_shape):
    """Helper function: pad window with 0 to target shape"""
    padding = torch.zeros(target_shape)
    padding[: x.size(0)] = x
    return padding


def process_5_windows(
    x: torch.Tensor, sample_id: int, target_shape: Tuple[int, int], left_mid, left_end, right_start, right_mid
):
    return torch.stack(
        [
            # left context
            pad_window(x[sample_id, :left_mid], target_shape),
            pad_window(
                x[sample_id, left_mid:left_end],
                target_shape,
            ),
            # right context
            pad_window(
                x[
                    sample_id,
                    right_start:right_mid,
                ],
                target_shape,
            ),
            pad_window(x[sample_id, right_mid:], target_shape),
        ],
        dim=0,
    )


def process_3_windows(x: torch.Tensor, sample_id: int, target_shape: Tuple[int, int], left_end, right_start):
    return torch.stack(
        [
            # left context
            pad_window(x[sample_id, :left_end], target_shape),
            # rigth context
            pad_window(
                x[sample_id, right_start:],
                target_shape,
            ),
        ],
        dim=0,
    )


def prepare_data(data, windows: int = 3, max_length_window: int = 128, value_key: str = "observation_values"):
    """
    tranform data to the correct batch format: need to separate locations & observations, split observations into windows.

    Args:
        data: dict
        windows: int, the window count the model was trained on, 3 or 5 including the imputation window
        max_length_window: int, the maximum length of the windows
        value_key: str, the key of the values in the data dict either 'observation_values' (PCA reduced) or 'high_dim_trajectory' (original data)

    Returns:
        batch: dict, the batch for the model
            with observations (values, times, mask). shape: (batch_size*process_dim, wc-1, max_length, 1)
            with locations (values, times). shape: (batch_size*process_dim, max_length_locations, 1)
        imp_mask: torch.Tensor, the mask of the imputation window (for visualization)
    """
    obs_grid = data["observation_grid"]
    obs_values_pca = data["observation_values"] # in pca space
    loc_values_pca_data = data["high_dim_reconst_from_3_pca"] # high dim (reconstructed from pca)
    obs_values_high_dim = data["high_dim_trajectory"] # high dim (original data)
    obs_mask = data["observation_mask"].bool()

    imp_mask = data["imputation_mask"].bool()

    # get locations (time stamps of imputation window) pad with last time stamp so that all have same size
    loc_grid = []
    loc_values_pca = []
    loc_values_high_dim = []
    initial_conditions_pca = []
    max_loc_size = imp_mask.sum(dim=1).max().item()
    for sample_id in range(obs_grid.size(0)):
        loc = obs_grid[sample_id][imp_mask[sample_id]]
        loc_val_pca = obs_values_pca[sample_id][imp_mask[sample_id].flatten()]
        loc_val_high_dim = obs_values_high_dim[sample_id][imp_mask[sample_id].flatten()]
        # pad with last value to have same size
        if (cur_len := loc.size(0)) < max_loc_size:
            loc = torch.cat([loc, loc[-1].repeat(max_loc_size - cur_len)])
            loc_val_pca = torch.cat([loc_val_pca, loc_val_pca[-1].repeat(max_loc_size - cur_len, 1)])
            loc_val_high_dim = torch.cat([loc_val_high_dim, loc_val_high_dim[-1].repeat(max_loc_size - cur_len, 1)])
        loc_grid.append(loc)
        loc_values_pca.append(loc_val_pca)
        loc_values_high_dim.append(loc_val_high_dim)

        # initial conditions in PCA space, location value are in the "from pca reconstructed" space
        initial_conditions_pca.append(obs_values_pca[sample_id][imp_mask[sample_id].flatten()][0])

    loc_grid = torch.stack(loc_grid).unsqueeze(-1)
    loc_values_pca = torch.stack(loc_values_pca)
    loc_values_high_dim = torch.stack(loc_values_high_dim)
    initial_conditions_pca = torch.stack(initial_conditions_pca)

    #
    # get observation windows
    # Go through time series and extract observations. (left and right of imputation window)
    #
    start_index_impu_window = imp_mask.int().argmax(dim=1)
    start_index_right_context = start_index_impu_window + imp_mask.int().sum(dim=1)

    context_values_pca = []
    context_values_high_dim = []
    context_obs_mask = []
    context_times = []
    for sample_id in range(obs_grid.size(0)):
        context_size = start_index_impu_window[sample_id].item()

        if windows == 5:
            # note: need to split left and right window each into two parts
            mid_point = context_size // 2
            left_mid, left_end = mid_point, context_size
            right_start, right_mid = (
                start_index_right_context[sample_id],
                start_index_right_context[sample_id] + mid_point,
            )
            values_pca = process_5_windows(
                obs_values_pca,
                sample_id,
                (max_length_window, obs_values_pca.size(-1)),
                left_mid,
                left_end,
                right_start,
                right_mid,
            )
            values_high_dim = process_5_windows(
                obs_values_high_dim,
                sample_id,
                (max_length_window, obs_values_high_dim.size(-1)),
                left_mid,
                left_end,
                right_start,
                right_mid,
            )
            times = process_5_windows(
                obs_grid, sample_id, (max_length_window, obs_grid.size(-1)), left_mid, left_end, right_start, right_mid
            )
            mask = process_5_windows(
                obs_mask, sample_id, (max_length_window, obs_mask.size(-1)), left_mid, left_end, right_start, right_mid
            )
        elif windows == 3:
            values_pca = process_3_windows(
                obs_values_pca,
                sample_id,
                (max_length_window, obs_values_pca.size(-1)),
                context_size,
                start_index_right_context[sample_id],
            )
            values_high_dim = process_3_windows(
                obs_values_high_dim,
                sample_id,
                (max_length_window, obs_values_high_dim.size(-1)),
                context_size,
                start_index_right_context[sample_id],
            )
            times = process_3_windows(
                obs_grid,
                sample_id,
                (max_length_window, obs_grid.size(-1)),
                context_size,
                start_index_right_context[sample_id],
            )
            mask = process_3_windows(
                obs_mask,
                sample_id,
                (max_length_window, obs_mask.size(-1)),
                context_size,
                start_index_right_context[sample_id],
            )
        context_values_pca.append(values_pca)
        context_values_high_dim.append(values_high_dim)
        context_times.append(times)
        context_obs_mask.append(mask)
    context_values_pca = torch.stack(context_values_pca)
    context_values_high_dim = torch.stack(context_values_high_dim)
    context_obs_mask = torch.stack(context_obs_mask)
    context_times = torch.stack(context_times)

    batch_pca = {
        "location_times": repeat_for_dim(loc_grid, 3).float(),
        "target_sample_path": (l:=make_single_dim(loc_values_pca).float()),
        "initial_conditions":  l[:, 0], # make_single_dim(initial_conditions_pca).float(),
        "observation_values": make_single_dim(context_values_pca).float(),
        "observation_mask": ~(repeat_for_dim(context_obs_mask, 3).bool()),
        "observation_times": repeat_for_dim(context_times, 3).float(),
    }
    batch_high_dim = {
        "location_times": repeat_for_dim(loc_grid, 50).float(),
        "target_sample_path": (l := make_single_dim(loc_values_high_dim).float()),
        "initial_conditions": l[:, 0],
        "observation_values": make_single_dim(context_values_high_dim).float(),
        "observation_mask": ~(repeat_for_dim(context_obs_mask, 50).bool()),
        "observation_times": repeat_for_dim(context_times, 50).float(),
    }

    return batch_pca, batch_high_dim, imp_mask


def visualize_samples(model_out, batch, sample_ids: list = [0, 10, 15], title="", save_dir=None):
    """
    Visualizes the observations, targets, and learnt imputation windows for the given sample IDs.

    Parameters:
    model_out (dict): Output from the model, including observations and imputation window data.
    batch (dict): The input batch data (not used but can be extended later if needed).
    sample_ids (list): List of 3 sample IDs to visualize.
    title (str): Title for the entire figure (optional).
    """
    # Extract relevant data from model output
    obs_grid = model_out["visualizations"]["observations"]["times"].detach()
    obs_values = model_out["visualizations"]["observations"]["values"].detach()
    obs_mask = (~model_out["visualizations"]["observations"]["mask"]).detach()
    loc_grid = model_out["visualizations"]["imputation_window"]["locations"].detach()

    # Initialize the figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)

    # Iterate over each of the sample IDs to create a plot in the corresponding subplot
    for idx, sample_id in enumerate(sample_ids):
        ax = axes[idx]  # Select the subplot

        # Plot observed data points (with different colors for each dimension)
        for i in range(obs_values.size(1)):
            ax.scatter(
                obs_grid[sample_id, i][obs_mask[sample_id, i]],
                obs_values[sample_id, i][obs_mask[sample_id, i]],
                label=f"observed (dim {i})",
                color=["blue", "green", "red", "black"][i],  # Keep the original colors
            )

        # Plot the target and learnt imputation curves
        ax.plot(
            loc_grid[sample_id, :, 0],
            model_out["visualizations"]["imputation_window"]["target"][sample_id, :, 0].detach().numpy(),
            c="black",
            linestyle="--",
            label="target",
        )
        ax.plot(
            loc_grid[sample_id, :, 0],
            model_out["visualizations"]["imputation_window"]["learnt"][sample_id, :, 0].detach().numpy(),
            c="orange",
            label="learnt",
        )

        # Set labels, legend, and title for each subplot
        ax.set_title(f"Sample {sample_id}")
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        # remove spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Adjust layout and display/save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the suptitle
    if save_dir is not None:
        plt.savefig(
            f"{save_dir}{title}_samples_{'_'.join(map(str, sample_ids))}.png"
        )  # Save the figure with all subplots
        print(f"Saved figure to {save_dir}{title}_samples_{'_'.join(map(str, sample_ids))}.png")
    plt.show()
    plt.close()


def evaluate_model(
    model,
    batch_pca,
    batch_high_dim,
    process_dim: int = 3,
    pca_params: Tuple[torch.Tensor, torch.Tensor] = None,
    output_path_base: Optional[str] = None,
) -> Tuple[dict, dict]:
    """
    Args:
        model: torch.nn.Module, the model to evaluate
        batch: dict, the batch for the model
        target_sample_paths: torch.Tensor, the target sample paths in 50 dim, ground truth for metric evaluation
        target_sample_paths_pca: torch.Tensor, the target sample paths as reconstructed from pca, ground truth for metric evaluation
        process_dim: int, the process dimension
        pca_params: Tuple[torch.Tensor, torch.Tensor], the pca params to project back to high dim
        output_path_base: str, the output path to save the metrics and plots. If None, nothing is saved

    Returns:
        metrics_high_dim: dict, the metrics for the high dim data
        metrics_pca: dict, the metrics for the pca data (reconstructed)
    """

    model_out_high_dim = model(batch_high_dim)
    visualize_samples(model_out_high_dim, batch_high_dim, title="high dim", save_dir=output_path_base)
    pred_sample_paths_high_dim = make_multi_dim(
        model_out_high_dim["visualizations"]["imputation_window"]["learnt"], batch_size=43, process_dim=50
    )  # [43, wlen_loc, 50]
    target_sample_paths = make_multi_dim(
        model_out_high_dim["visualizations"]["imputation_window"]["target"], batch_size=43, process_dim=50
    )
    metrics_high_dim = compute_metrics(pred_sample_paths_high_dim, target_sample_paths)

    # same for pca
    model_out_pca = model(batch_pca)
    visualize_samples(model_out_pca, batch_pca, title="PCA", save_dir=output_path_base)
    # make prediction multi dim
    pred_sample_paths = make_multi_dim(
        model_out_pca["visualizations"]["imputation_window"]["learnt"], batch_size=43, process_dim=3
    )  # [43, wlen_loc, 3]
    # revert PCA before computing metrics
    eigenvectors, eigenvalues = pca_params
    pred_sample_paths_high_dim = (
        pred_sample_paths
        * torch.sqrt(eigenvalues[..., :pca_component_count][..., None, :])
        @ torch.transpose(eigenvectors[..., :pca_component_count], dim0=-1, dim1=-2)
    )  # [43, wlen_loc, 50]
    target_sample_paths_pca = make_multi_dim(
        model_out_high_dim["visualizations"]["imputation_window"]["target"], batch_size=43, process_dim=50
    )
    metrics_pca = compute_metrics(pred_sample_paths_high_dim, target_sample_paths_pca)
    # visualize_samples(model_out_pca, batch_pca, 'PCA reconstructed')

    # save metrics
    if output_path_base is not None:
        with open(os.path.join(output_path_base, "metrics_high_dim.json"), "w") as f:
            json.dump(metrics_high_dim, f, indent=2)
        with open(os.path.join(output_path_base, "metrics_pca.json"), "w") as f:
            json.dump(metrics_pca, f, indent=2)

    return metrics_high_dim, metrics_pca


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    wc = 3  # 3 # 5
    pca = True

    # model_checkpoint_path = "results/FIMImputation/SynthData_all_5w_MinMax_MinMax_nllh_sfvGlobNorm_LRcosAn_4encBlocks-experiment-seed-4_09-13-1636/checkpoints/best-model/model-checkpoint.pth"
    # model_abbr: str = "5w_09-13-1636-epoch-369"
    model_checkpoint_path = "results/FIMImputation/SynthData_all_3w_MinMax_MinMax_nllh_sfvGlobNorm_LRcosAn_4encBlocks-experiment-seed-4_09-13-1635/checkpoints/best-model/model-checkpoint.pth"
    model_abbr: str = "3w_09-13-1635"

    output_path_base = f"reports/FIMImputation/mocap/{model_abbr}/"
    os.makedirs(output_path_base, exist_ok=True)

    data = load_data(data_dir)
    max_sequence_length = 64 if wc == 5 else 128
    value_key = "observation_values" if pca else "high_dim_trajectory"
    process_dim = 3 if pca else 50
    pca_params = data["pca_params"] if pca else None

    batch_pca, batch_high_dim, imp_mask = prepare_data(
        data, windows=wc, max_length_window=max_sequence_length, value_key=value_key
    )

    # sample_id = 25
    # for i in range(batch["observation_values"].size(1)):
    #     plt.scatter(
    #                 batch["observation_times"][sample_id, i][~batch["observation_mask"][sample_id, i]],
    #                 batch["observation_values"][sample_id, i][~batch["observation_mask"][sample_id, i]],
    #                 label="observed",
    #             )
    # plt.plot(
    #     batch['location_times'][sample_id], batch['target_sample_path'][sample_id], color='black', linestyle='--', label='target'
    # )
    # plt.show()

    model = load_model_from_checkpoint(model_checkpoint_path, module=FIMImputation, for_eval=True)

    metrics_high_dim, metrics_pca = evaluate_model(
        model,
        batch_pca,
        batch_high_dim,
        process_dim=process_dim,
        pca_params=pca_params,
        output_path_base=output_path_base,
    )
    print("Metrics for: ")
    print("model: ", model_checkpoint_path.split("/")[2])
    print("PCA: ")

    print(json.dumps(metrics_pca, indent=2))

    print("High Dim: ")
    print(json.dumps(metrics_high_dim, indent=2))
