import json
import math
import os
import pickle
from typing import Optional, Tuple

import torch
from tqdm import tqdm

from fim.models import ModelFactory
from fim.utils.metrics import compute_metrics


device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
pca_component_count = 3

model_config = {
    "name": "FIM_imputation_windowed",
    "fim_imputation": "/home/cvejoski/Projects/FoundationModels/FIM/results/FIMImputation/SynthData_all_5w_MinMax_MinMax_nllh_sfvGlobNorm_LRcosAn_4encBlocks-experiment-seed-4_09-22-2323/checkpoints/best-model/model-checkpoint.pth",
    # "fim_imputation": "results/FIMImputation/SynthData_all_5w_MinMax_MinMax_nllh_sfvGlobNorm_LRcosAn_4encBlocks-experiment-seed-4_09-22-2323/checkpoints/best-model/model-checkpoint.pth",
    # "fim_imputation": "results/FIMImputation/SynthData_all_5w_MinMax_MinMax_nllh_sfvGlobNorm_LRcosAn_4encBlocks-experiment-seed-4_09-13-1636/checkpoints/best-model/model-checkpoint.pth",
    # "fim_imputation": "results/FIMImputation/SynthData_all_3w_MinMax_MinMax_nllh_sfvGlobNorm_LRcosAn_4encBlocks-experiment-seed-4_09-13-1635/checkpoints/best-model/model-checkpoint.pth",
    "denoising_model": None,
}
data_config = {
    "data_dir": "/cephfs_projects/foundation_models/data/neurips_baseline_comparison_data/navier_stokes_preprocessed_with_imputation_sets_and_train_test_split/split_single_long/",
    "window_count": (wc := 5),
    "max_length_window": 40 if wc == 5 else 40,
    "overlap_locations": 0,
}


def load_data_navier_stokes(directory: str) -> dict:
    data = {}
    import numpy as np

    def jax_to_torch(jax_array):
        if isinstance(jax_array, dict):
            return {k: torch.tensor(np.array(v)) for k, v in jax_array.items()}
        return torch.tensor(np.array(jax_array))

    data["observation_grid"] = jax_to_torch(pickle.load(open(os.path.join(directory, "time.pickle"), "rb")))
    data["observation_values"] = jax_to_torch(pickle.load(open(os.path.join(directory, "u_and_v_time_coefficients.pickle"), "rb")))
    data["high_dim_trajectory"] = jax_to_torch(pickle.load(open(os.path.join(directory, "u_and_v.pickle"), "rb")))
    data["imputation_mask"] = jax_to_torch(pickle.load(open(os.path.join(directory, "imputation_mask_0_2_perc.pickle"), "rb")))
    data["observation_mask"] = jax_to_torch(pickle.load(open(os.path.join(directory, "observation_mask_0_2_perc.pickle"), "rb")))
    data["pca_params"] = jax_to_torch(pickle.load(open(os.path.join(directory, "u_and_v_pca_params.pickle"), "rb")))
    B, T = data["observation_values"].size(0), data["observation_values"].size(1)
    data["high_dim_trajectory"] = data["high_dim_trajectory"].reshape(B, T, -1)

    return data


def pad_window(x, target_shape):
    """Helper function: pad window with 0 to target shape"""
    padding = torch.zeros(target_shape)
    padding[: x.size(0)] = x
    return padding


def process_5_windows(x: torch.Tensor, sample_id: int, target_shape: Tuple[int, int], left_mid, left_end, right_start, right_mid):
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


def prepare_data(data_dir: str, window_count: int = 3, max_length_window: int = 128, overlap_locations: int = 0):
    """
    load data, tranform data to the correct batch format: need to separate locations & observations, split observations into windows.

    Args:
        data: dict
        window_count: int, the window count the model was trained on, 3 or 5 including the imputation window
        max_length_window: int, the maximum length of the windows

    Returns:
        batch_high_dim: dict, the batch for the model on high dimensionality data
            with observations (values, times, mask). shape: (batch_size, wc-1, max_length, pocess_dim)
            with locations (values, times). shape: (batch_size, max_length_locations, pocess_dim)
        batch_pca: dict, the batch for the model on pca reduced data
        pca_params: torch.Tensor, the mask of the imputation window (for visualization)
    """
    # load data
    data = load_data_navier_stokes(data_dir)

    obs_grid = data["observation_grid"]
    obs_values_pca = data["observation_values"]  # in pca space
    obs_values_high_dim = data["high_dim_trajectory"]  # high dim (original data)
    obs_mask = data["observation_mask"].bool()

    imp_mask = data["imputation_mask"].bool()

    start_index_impu_window = imp_mask.int().argmax(dim=1)
    start_index_right_context = start_index_impu_window + imp_mask.int().sum(dim=1)

    # get locations (time stamps of imputation window) pad with last time stamp so that all have same size
    loc_grid = []
    loc_values_pca = []
    loc_values_high_dim = []
    initial_conditions_pca = []
    padding_mask_locations = []
    max_loc_size = imp_mask.sum(dim=1).max().item() + 2 * overlap_locations

    for sample_id in range(obs_grid.size(0)):
        left_extend = max(0, start_index_impu_window[sample_id].item() - overlap_locations)
        right_extend = min(obs_grid.size(1), start_index_right_context[sample_id].item() + overlap_locations)

        loc = obs_grid[sample_id][left_extend:right_extend]
        loc_val_pca = obs_values_pca[sample_id][left_extend:right_extend]
        loc_val_high_dim = obs_values_high_dim[sample_id][left_extend:right_extend]

        mask = torch.zeros(max_loc_size, dtype=bool)
        # pad with last value to have same size
        padding_size = max_loc_size - loc.size(0)
        if padding_size > 0:
            loc = torch.cat([loc, loc[-1].repeat(padding_size, 1)])
            loc_val_pca = torch.cat([loc_val_pca, loc_val_pca[-1].repeat(padding_size, 1)])
            loc_val_high_dim = torch.cat([loc_val_high_dim, loc_val_high_dim[-1].repeat(padding_size, 1)])
            mask[-padding_size:] = True
        loc_grid.append(loc)
        loc_values_pca.append(loc_val_pca)
        loc_values_high_dim.append(loc_val_high_dim)
        padding_mask_locations.append(mask)

        # initial conditions in PCA space, location value are in the "from pca reconstructed" space
        initial_conditions_pca.append(loc_val_pca[0])

    loc_grid = torch.stack(loc_grid)
    loc_values_pca = torch.stack(loc_values_pca)
    loc_values_high_dim = torch.stack(loc_values_high_dim)
    initial_conditions_pca = torch.stack(initial_conditions_pca)
    padding_mask_locations = torch.stack(padding_mask_locations)

    #
    # get observation windows
    # Go through time series and extract observations. (left and right of imputation window)
    #
    context_values_pca = []
    context_values_high_dim = []
    context_obs_mask = []
    context_times = []
    for sample_id in range(obs_grid.size(0)):
        context_size = max(0, start_index_impu_window[sample_id].item() - overlap_locations)

        if window_count == 5:
            # note: need to split left and right window each into two parts
            mid_point = context_size // 2
            left_mid, left_end = mid_point, context_size
            right_start, right_mid = (
                start_index_right_context[sample_id] + overlap_locations,
                start_index_right_context[sample_id] + mid_point + overlap_locations,
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
        elif window_count == 3:
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
        "location_times": loc_grid.float(),
        "target_sample_path": (l := loc_values_pca.float()),
        "initial_conditions": l[:, 0],
        "observation_values": context_values_pca.float(),
        "observation_mask": ~(context_obs_mask.bool()),
        "observation_times": context_times.float(),
        "padding_mask_locations": padding_mask_locations,
    }
    batch_high_dim = {
        "location_times": loc_grid.float(),
        "target_sample_path": (l := loc_values_high_dim.float()),
        "initial_conditions": l[:, 0],
        "observation_values": context_values_high_dim.float(),
        "observation_mask": ~(context_obs_mask.bool()),
        "observation_times": context_times.float(),
        "padding_mask_locations": padding_mask_locations,
    }

    return batch_high_dim, batch_pca, data["pca_params"]


def evaluate_configuration(model, data, output_path, pca_params: Optional[tuple] = None, individual_eval_each_dim=False):
    """
    Evaluate the model on the given batch and save the results.

    Args:
        model: the model to evaluate
        batch: the batch to evaluate
        output_path: the path to save the results
        pca: flag indicating if data is in pca space and needs to be transformed back to high dim
    """
    # Reshape the batch tensors from BxTxD to B*DxTx1

    if individual_eval_each_dim:
        B, D = data["observation_values"].size(0), data["observation_values"].size(-1)
        data["location_times"] = data["location_times"].repeat_interleave(D, dim=0)
        data["target_sample_path"] = data["target_sample_path"].permute(0, 2, 1).reshape(B * D, -1, 1)
        data["initial_conditions"] = data["initial_conditions"].reshape(B * D, 1)
        data["observation_values"] = data["observation_values"].permute(0, 3, 1, 2).reshape(B * D, 4, 40, 1)
        data["observation_mask"] = data["observation_mask"].repeat_interleave(D, dim=0)
        data["observation_times"] = data["observation_times"].repeat_interleave(D, dim=0)
        data["padding_mask_locations"] = data["padding_mask_locations"].repeat_interleave(D, dim=0)

    model.eval()
    with torch.no_grad():
        if individual_eval_each_dim:
            batch_size = 1024 * 5
            num_batches = math.ceil(data["observation_values"].size(0) / batch_size)
            outputs = []
            for i in tqdm(range(num_batches), total=num_batches, desc="Evaluating"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, data["observation_values"].size(0))
                batch_data = {k: v[start_idx:end_idx].to(device_map) for k, v in data.items()}
                batch_output = model(batch_data)
                def move_to_cpu(data):
                    if isinstance(data, dict):
                        return {k: move_to_cpu(v) for k, v in data.items()}
                    elif isinstance(data, torch.Tensor):
                        return data.cpu()
                    else:
                        return data

                batch_output = move_to_cpu(batch_output)
                outputs.append(batch_output)

            def recursive_cat(dicts, key=None):
                if isinstance(dicts[0], dict):
                    return {k: recursive_cat([d[k] for d in dicts], k) for k in dicts[0].keys()}
                else:
                    if dicts[0].dim() == 3:
                        t = torch.cat(dicts, dim=0).reshape(10, -1, 8).permute(0, 2, 1)
                        if key == "locations":
                            return t[:, :, 0, None]
                        return t
                    elif dicts[0].dim() == 2:
                        return torch.cat(dicts, dim=0).reshape(10, -1, 8).permute(0, 2, 1)[:, :, 0]
                    elif dicts[0].dim() == 4:
                        t = torch.cat(dicts, dim=0).reshape(10, -1, 4, 40).permute(0, 2, 3, 1)
                        if key in ["mask", "times"]:
                            return t[:, :, :, 0, None]
                        return t

            output = recursive_cat(outputs)

        else:
            data = {k: v.to(device_map) for k, v in data.items()}
            output = model(data)

        target_sample_paths = output["imputation_window"]["target"]
        pred_sample_paths = output["imputation_window"]["learnt"]

    # reconstruct from pca space if necessary
    if pca_params is not None:
        right_eigenvectors = pca_params.get("right_eigenvectors")
        data_mean = pca_params.get("data_mean")
        pca_components_count = pred_sample_paths.size(-1)

        # this is similar to above, reverting the PCA by multiplying with the inverse (which is just the transpose) of the projection matrix
        inverse_base_change = torch.transpose(right_eigenvectors, 1, 2)

        # only consider the 38 components
        inverse_base_change = inverse_base_change[..., :pca_components_count, :]  # [..., components_count, D]

        pred_sample_paths_high_dim_pca = pred_sample_paths[..., :pca_components_count]  # [..., components_count, D]
        # base change from 38 to high dimensional
        pred_sample_paths_high_dim_pca = pred_sample_paths_high_dim_pca @ inverse_base_change  # [..., T, D]
        # adjust for centering of data
        pred_sample_paths_high_dim_pca = pred_sample_paths_high_dim_pca + data_mean

        target_sample_paths_high_dim_pca = target_sample_paths[..., :pca_components_count]  # [..., components_count, D]
        # base change from 38 to high dimensional
        target_sample_paths_high_dim_pca = target_sample_paths_high_dim_pca @ inverse_base_change  # [..., T, D]
        # adjust for centering of data
        target_sample_paths_high_dim_pca = target_sample_paths_high_dim_pca + data_mean

        pred_sample_paths = pred_sample_paths_high_dim_pca
        target_sample_paths = target_sample_paths_high_dim_pca

    # compute metrics
    metrics = compute_metrics(
        pred_sample_paths,
        target_sample_paths,
        mask=output["imputation_window"]["padding_mask_locations"],
    )

    # save metrics
    json.dump(metrics, open(output_path + "metrics.json", "w"), indent=2)

    # save predictions
    torch.save(
        output,
        output_path + "predictions.pth",
    )


if __name__ == "__main__":
    model = ModelFactory.create(**model_config, device_map=device_map)
    model.to(device_map)
    model_abbr = model_config["fim_imputation"].split("/")[-4].split("_")[-1]
    output_dir_base = f"reports/FIMImputation/NavierStokes/{model_abbr}_overlap0/"
    os.makedirs(output_dir_base, exist_ok=True)

    print("saving at ", output_dir_base)
    batch_high_dim, batch_pca, pca_params = prepare_data(**data_config)
    # evaluate_configuration(model, batch_pca, output_dir_base + "pca_", pca_params=pca_params)
    evaluate_configuration(model, batch_high_dim, output_dir_base + "high_dim_", individual_eval_each_dim=True)

    # visualize
    # import matplotlib.pyplot as plt

    # sample_id = 0
    # dim = 0
    # for w in range(4):
    #     plt.scatter(
    #         batch_high_dim["observation_times"][sample_id][w][:, dim].numpy(),
    #         batch_high_dim["observation_values"][sample_id][w][:, dim].numpy(),
    #         label=f"obs_{w}",
    #     )
    # plt.plot(
    #     batch_high_dim["location_times"][sample_id][:, dim].numpy(),
    #     batch_high_dim["target_sample_path"][sample_id][:, dim].numpy(),
    #     label="target",
    # )

    # plt.show()
