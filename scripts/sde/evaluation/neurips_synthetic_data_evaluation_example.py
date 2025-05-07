# import json
# from pathlib import Path
# from typing import Optional
#
# import matplotlib.pyplot as plt
# import numpy as np
# import optree
# import torch
#
# from fim.models.sde import FIMSDE
# from fim.pipelines.sde_sampling_from_model import fimsde_sample_paths_on_masked_grid
# from scripts.sde.evaluation.lab_visits_synthetic_equations_figure import plot_1D_paths, plot_2D_paths, plot_2D_vf
#
#
# def get_system_data(all_systems_data: list[dict], system: str, tau: float, noise: float) -> dict:
#     """
#      From a list of all systems data, extract data of system with tau inter-observation times and relative noise.
#
#     Args:
#         all_systems_data (list[dict]): List of system data, each of which is a dict.
#         system (str): Name of system data to extract.
#         tau (float): Inter-observation time of data to extract.
#         noise (float): Relative additive noise added to observations of trajectories.
#
#     Return:
#         data_of_system (dict): Keys: name, tau, obs_times, obs_values, locations, initial_states
#     """
#     data_of_system = [d for d in all_systems_data if (d["name"] == system and d["tau"] == tau and d["noise"] == noise)]
#
#     # should contain exactly one data for each system and tau
#     if len(data_of_system) == 1:
#         data_of_system = {k: np.array(v) if isinstance(v, list) else v for k, v in data_of_system[0].items()}
#
#         # rename for easier inference for model
#         if "observations" in data_of_system:
#             data_of_system["obs_values"] = data_of_system.pop("observations")
#
#         # add obs times based_on_tau
#         B, M, T, _ = data_of_system["obs_values"].shape
#         data_of_system["obs_times"] = tau * np.ones((B, M, 1, 1)) * np.arange(T).reshape(1, 1, T, 1)
#
#         return data_of_system
#
#     elif len(data_of_system) == 0:
#         raise ValueError(f"Could not find data of system {system} and tau {tau} and noise perc {noise}.")
#
#     else:
#         raise ValueError(f"Found {len(data_of_system)} sets of data for system {system} and tau {tau}.")
#
#
# def evaluate_model(
#     model: FIMSDE,
#     dataset: dict,
#     sample_paths: Optional[bool],
#     sample_paths_count: int,
#     dt: float,
#     sample_path_steps: int,
# ):
#     model.eval()
#
#     results = {}
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)
#
#     dataset = optree.tree_map(
#         lambda x: torch.from_numpy(x).to(torch.float32).to(device) if isinstance(x, np.ndarray) else x, dataset, namespace="fimsde"
#     )
#
#     initial_states = dataset.get("initial_states")
#     D = dataset["initial_states"].shape[-1]
#
#     grid = (torch.arange(sample_path_steps) * dt).view(1, 1, -1, 1)
#     grid = torch.broadcast_to(grid, (initial_states.shape[0], sample_paths_count, sample_path_steps, 1)).to(device)
#
#     if sample_paths is True:
#         sample_paths, sample_paths_grid = fimsde_sample_paths_on_masked_grid(
#             model,
#             dataset,
#             grid=grid,
#             mask=torch.ones_like(grid),
#             initial_states=initial_states,
#             solver_granularity=1,
#         )
#
#         results.update(
#             {
#                 "sample_paths": sample_paths,
#                 "sample_paths_grid": sample_paths_grid,
#             }
#         )
#
#     # get vector fields at locations
#     estimated_concepts = model(dataset, training=False, return_losses=False)
#     results.update({"estimated_concepts": estimated_concepts})
#
#     # reduce outputs to dimensionality of original problem
#     estimated_concepts.drift = estimated_concepts.drift[..., :D]
#     estimated_concepts.diffusion = estimated_concepts.diffusion[..., :D]
#     if "sample_paths" in results.keys():
#         results["sample_paths"] = results["sample_paths"][..., :D]
#
#     results = optree.tree_map(lambda x: x.detach().to("cpu"), results, namespace="fimsde")
#
#     return results
#
#
# if __name__ == "__main__":
#     # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
#
#     model = {
#         "checkpoint_dir": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/600k_drift_deg_3_diff_deg_2_delta_tau_fixed_linear_attn_softmax_no_extra_normalization_and_fix_in_residual_layer_05-06-2300/checkpoints/",
#         "checkpoint_name": "epoch-139",
#     }
#
#     # systems in table of paper
#     path_to_inference_data_json = Path(
#         "/cephfs_projects/foundation_models/data/SDE/evaluation/20250325_synthetic_systems_5000_points_with_additive_noise/data/systems_observations_and_locations.json"
#     )
#
#     systems_to_load: list[str] = [
#         "Damped Linear",
#         "Damped Cubic",
#         "Duffing",
#         "Glycosis",
#         "Hopf",
#         "Double Well",
#         "Wang",
#         "Syn Drift",
#     ]
#
#     taus = [0.002, 0.01, 0.02, 0.2]
#     noises = [0.0, 0.05, 0.1]
#     sample_paths = True
#     # sample_paths_count = 100
#     sample_paths_count = 10
#     dt = 0.002
#     sample_path_steps = 500
#
#     #########################################################
#     # load_model
#     checkpoint_path: Path = model.get("checkpoint_dir") / model.get("checkpoint_name")
#     model = FIMSDE.load_model(model_path=checkpoint_path)
#
#     # prepare all evaluations
#     data = json.load(open(path_to_inference_data_json, "r"))
#     datasets: dict = {
#         (system, tau, noise): get_system_data(data, system, tau, noise) for system in systems_to_load for tau in taus for noise in noises
#     }
#
#     # this iterates over all evaluations. maybe we just do it for one or so
#     for dataset in datasets:
#         # sample paths from model for one dataset
#         # also evalute vector fields
#         dataset_results = evaluate_model(model, dataset, sample_paths, sample_paths_count, dt, sample_path_steps)
#
#         # this should roughly print a figure with 3 subplots: drift, diffusion, paths
#
#         # general plot config
#         linewidth_vf = 1
#         linewidth_paths = 0.2
#         loc_size_per_dim = 32
#         loc_stride_length = 4
#
#         gt_plot_config = {
#             "color": "black",
#             "linestyle": "solid",
#             "label": "Ground-truth",
#         }
#
#         fimsde_plot_config = {
#             "color": "#0072B2",
#             "linestyle": "solid",
#             "label": "FIM-SDE",
#         }
#
#         fig, axs = plt.subplots(
#             nrows=len(systems_to_plot),
#             ncols=3,
#             figsize=(4.5, 1.5 * len(systems_to_plot)),
#             gridspec_kw={"width_ratios": [1, 1, 1]},
#             dpi=300,
#             tight_layout=True,
#         )
#
#         D = locations.shape[-1]
#
#         if D == 1:
#             axs[0].plot(locations.squeeze(), drift.squeeze(), color=color, linewidth=linewidth_vf, linestyle=linestyle, label=label)
#             axs[1].plot(locations.squeeze(), diffusion.squeeze(), color=color, linewidth=linewidth_vf, linestyle=linestyle, label=label)
#
#             if label in ["Ground-truth", "FIM-SDE"]:
#                 plot_1D_paths(
#                     axs[2],
#                     times,
#                     paths,
#                     color=color,
#                     linewidth=linewidth_paths,
#                     linestyle=linestyle,
#                     label=label,
#                 )
#
#             return None, None
#
#         else:
#             drift_quiver = plot_2D_vf(axs[0], locations, drift, color=color, linestyle=linestyle, scale=drift_quiver_scale, label=label)
#             diff_quiver = plot_2D_vf(axs[1], locations, diffusion, color=color, linestyle=linestyle, scale=diff_quiver_scale, label=label)
#
#             if label in ["Ground-truth", "FIM-SDE"]:
#                 plot_2D_paths(
#                     axs[2],
#                     paths,
#                     color=color,
#                     linewidth=linewidth_paths,
#                     linestyle=linestyle,
#                     label=label,
#                 )
