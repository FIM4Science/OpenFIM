from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import Tensor

from fim.data.datasets import FIMSDEDatabatch
from fim.data_generation.sde.dynamical_systems import DynamicalSystem
from fim.data_generation.sde.dynamical_systems_sample import PathGenerator
from fim.models.sde import MinMaxNormalization, SDEConcepts


def get_dynamical_system_data(
    system: DynamicalSystem,
    integration_config: dict,
    locations_params: dict,
    normalize: Optional[bool] = False,
    return_params: Optional[bool] = False,
) -> FIMSDEDatabatch:
    """
    Generate dynamical system data and optionally normalize it.

    Args:
        system (DynamicalSystem): dynamical system to sample paths from
        integration_config (dict): solver configuration
        locations_params (dict): hypercube locations configuration
        normalize (bool): return instance normalized dynamical systems data
        return_params (bool): return tensors with params from dynamical system

    Returns:
        data (FIMSDEDatabatch): Data generated from dynamical system, via integration and location configs.
    """
    path_generator = PathGenerator(
        dataset_type="FIMSDEpDataset", system=system, integrator_params=integration_config, locations_params=locations_params
    )

    if return_params is True:
        data, drift_params, diffusion_params = path_generator.generate_paths(return_params=True)

    else:
        data: FIMSDEDatabatch = path_generator.generate_paths(return_params=False)

    # to dict
    data: dict = {
        "obs_times": data.obs_times,
        "obs_values": data.obs_values,
        "obs_mask": data.obs_mask,
        "locations": data.locations,
        "diffusion_at_locations": data.diffusion_at_locations,
        "drift_at_locations": data.drift_at_locations,
    }
    if normalize is True:
        values_norm = MinMaxNormalization(normalized_min=-1, normalized_max=1)
        values_norm_stats = values_norm.get_norm_stats(data["obs_values"])
        data["obs_values"] = values_norm.normalization_map(data["obs_values"], values_norm_stats)

        if data["locations"] is not None:
            data["locations"] = values_norm.normalization_map(data["locations"], values_norm_stats)

        times_norm = MinMaxNormalization(normalized_min=0, normalized_max=1)
        times_norm_stats = times_norm.get_norm_stats(data["obs_times"])
        data["obs_times"] = times_norm.normalization_map(data["obs_times"], times_norm_stats)

        target_concepts: SDEConcepts | None = SDEConcepts.from_dict(data)
        target_concepts.normalize(values_norm, values_norm_stats, times_norm, times_norm_stats)

        data["drift_at_locations"] = target_concepts.drift
        data["diffusion_at_locations"] = target_concepts.diffusion

    if return_params is True:
        return data, drift_params, diffusion_params

    else:
        return data


def show_paths_vector_fields_and_statistics(
    system: DynamicalSystem,
    integration_config: dict,
    locations_params: dict,
    fig_config: Optional[dict] = {},
    paths_plt_config: Optional[dict] = {},
    vector_field_plt_config: Optional[dict] = {},
    normalize: Optional[bool] = False,
) -> tuple[plt.Figure, pd.DataFrame]:
    """
    Sample paths from a system and plot them, including the vector fields. Also return some statistics of paths.

    Args:
        system (DynamicalSystem): dynamical system to sample paths from
        integration_config (dict): solver configuration
        locations_params (dict): hypercube locations configuration
        fig_plt_config (dict): configs passed to figure
        paths_plt_config (dict): configs passed to paths plotting
        vector_field_plt_config (dict): configs passed to vector field plotting

    Returns:
        fig (plt.Figure): Figure with two or three subplots: paths, drift and diffusion
        stats_df (pd.Dataframe): Dataframe with some statistics of paths
    """

    data: FIMSDEDatabatch = get_dynamical_system_data(system, integration_config, locations_params, normalize)

    num_realizations, _, _, D = data.obs_values.shape

    # set up figure
    default_fig_config = {"dpi": 300, "figsize": (12, 4 * num_realizations)}

    if (figsize := fig_config.get("figsize")) is not None:  # can pass single value as figsize
        if not isinstance(figsize, tuple):
            fig_config["figsize"] = (3 * figsize, figsize * num_realizations)

        else:
            fig_config["figsize"] = figsize

    fig_config = default_fig_config | fig_config

    # set up subplots
    fig, axs = plt.subplots(num_realizations, 3, subplot_kw={"projection": "3d" if D == 3 else None}, **fig_config)
    axs = axs.reshape(num_realizations, 3)

    # fill subplots
    for realization in range(num_realizations):
        plot_paths_in_axis(axs[realization, 0], data.obs_times[realization], data.obs_values[realization], **paths_plt_config)
        plot_vector_field_in_axis(
            axs[realization, 1], data.locations[realization], data.drift_at_locations[realization], **vector_field_plt_config
        )
        plot_vector_field_in_axis(
            axs[realization, 2], data.locations[realization], data.diffusion_at_locations[realization], **vector_field_plt_config
        )

    # axis config
    tick_params = {"labelsize": 4, "length": 3}
    for ax in axs.reshape(-1):
        ax.tick_params(**tick_params)
        [x.set_linewidth(0.5) for x in ax.spines.values()]

    axs[0, 0].set_title("Paths", fontsize=7)
    axs[0, 1].set_title("Drift", fontsize=7)
    axs[0, 2].set_title("Diffusion", fontsize=7)

    # statistics
    stats_df = path_statistics(data.obs_values[0])

    return fig, stats_df


def plot_paths_in_axis(
    ax: plt.Axes,
    paths_times: Tensor,
    paths_values: Tensor,
    paths_mask: Optional[Tensor] = None,
    cmap: Optional[str] = None,
    init_states_marker_size: Optional[int] = 2,
    paths_label: Optional[str] = None,
    initial_states_label: Optional[str] = None,
    **plt_config,
) -> None:
    """
    Plot paths into provided axis using passed plot configuration.
    If cmap is passed, color each path differently according to cmap.

    Args:
        ax (plt.Axes): Axis to plot paths into using passed plotting config
        paths_times (Tensor): Times of paths from one system. Shape: [num_paths, num_obs, 1]
        paths_values (Tensor): Values of paths from one system. Shape: [num_paths, num_obs, state_dim]
        paths_mask (Optional[Tensor]): True indicates where observation is valid.
        init_states_marker_size (float): size of marker for initial state of each path
        paths_label (str): Label for plotted paths.
        initial_states_label_label (str): Label for plotted initial states.
        plt_config (dict): kwargs passed to plt.plot().
    """
    # handle no mask passed
    if paths_mask is None:
        paths_mask = torch.ones_like(paths_times).bool()

    # update default plotting config if not passed
    default_plt_config = {"linewidth": 0.5}
    plt_config = default_plt_config | plt_config

    # init states plotting config
    init_states_plt_config = {"color": "black", "marker": "o", "markersize": init_states_marker_size, "linestyle": "None"}

    num_paths, num_obs, state_dim = paths_values.shape

    # define color per path
    if cmap is not None:
        cmap = plt.get_cmap(cmap)
        colors = cmap(torch.linspace(0, 1, num_paths))

    elif (color := plt_config.get("color")) is not None:
        colors = num_paths * [color]

    else:
        colors = num_paths * ["black"]

    # paths, path times are trunkated because sometimes model paths are passed here and they might be fewer than paths_times
    for path_num, (path_times, path_values, path_mask, color) in enumerate(
        zip(
            paths_times[: paths_values.shape[0], : paths_values.shape[1]],
            paths_values,
            paths_mask[: paths_values.shape[0], : paths_values.shape[1]],
            colors,
        )
    ):
        plt_config.update({"color": color})

        path_mask = path_mask.squeeze()

        if state_dim == 1:
            ax.plot(path_times[path_mask], path_values[path_mask], label=paths_label if path_num == 0 else None, **plt_config)
            ax.plot(
                path_times[path_mask][0],
                path_values[path_mask][0],
                label=initial_states_label if path_num == 0 else None,
                **init_states_plt_config,
            )

        elif state_dim == 2:
            ax.plot(path_values[path_mask][:, 0], path_values[path_mask][:, 1], label=paths_label if path_num == 0 else None, **plt_config)
            ax.plot(
                path_values[path_mask][0, 0],
                path_values[path_mask][0, 1],
                label=initial_states_label if path_num == 0 else None,
                **init_states_plt_config,
            )

        elif state_dim == 3:
            ax.plot(
                path_values[path_mask][:, 0],
                path_values[path_mask][:, 1],
                path_values[path_mask][:, 2],
                label=paths_label if path_num == 0 else None,
                **plt_config,
            )
            ax.plot(
                path_values[path_mask][0, 0],
                path_values[path_mask][0, 1],
                path_values[path_mask][0, 2],
                label=initial_states_label if path_num == 0 else None,
                **init_states_plt_config,
            )

        else:
            raise ValueError("Only plot paths of dimension <= 3. Got " + str(state_dim) + ".")


def plot_vector_field_in_axis(ax: plt.Axes, locations: Tensor, vector_field: Tensor, **plt_config) -> None:
    """
    Plot vector field into provided axis using passed plot configuration.

    Args:
        ax (plt.Axes): Axis to plot paths into using passed plotting config
        locations (Tensor): Set of points where vector field is evaluated at. Shape: [num_locations, state_dim]
        vector_field (Tensor): Values of vector field at locations. Shape: [num_locations, state_dim]
        plt_config (dict): kwargs passed to plt.plot().
    """

    default_plt_config = {"color": "black"}
    plt_config = default_plt_config | plt_config

    num_obs, state_dim = locations.shape

    if state_dim == 1:
        plot_1D_vector_field_in_axis(ax, locations, vector_field, **plt_config)

    elif state_dim == 2:
        plot_2D_vector_field_in_axis(ax, locations, vector_field, **plt_config)

    elif state_dim == 3:
        plot_3D_vector_field_in_axis(ax, locations, vector_field, **plt_config)

    else:
        raise ValueError("Only plot vector fields of dimension <= 3. Got " + str(state_dim) + ".")


def plot_1D_vector_field_in_axis(ax: plt.Axes, locations: Tensor, vector_field: Tensor, **plt_config) -> None:
    """
    Plot vector field into provided axis using passed plot configuration.

    Args:
        ax (plt.Axes): Axis to plot paths into using passed plotting config
        locations (Tensor): Set of points where vector field is evaluated at. Shape: [num_locations, 1]
        vector_field (Tensor): Values of vector field at locations. Shape: [num_locations, 1]
        plt_config (dict): kwargs passed to plt.plot().
    """

    assert locations.shape == vector_field.shape
    assert locations.shape[-1] == 1
    assert locations.ndim == 2

    locations = locations.reshape(-1)
    vector_field = vector_field.reshape(-1)

    # sort inputs if necessary
    permutation = torch.argsort(locations, axis=0)
    locations = locations[permutation]
    vector_field = vector_field[permutation]

    # plot vector field as function
    ax.plot(locations, vector_field, **plt_config)


def plot_2D_vector_field_in_axis(ax: plt.Axes, locations: Tensor, vector_field: Tensor, **plt_config) -> None:
    """
    Plot vector field into provided axis using passed plot configuration.

    Args:
        ax (plt.Axes): Axis to plot paths into using passed plotting config
        locations (Tensor): Set of points where vector field is evaluated at. Shape: [num_locations, 2]
        vector_field (Tensor): Values of vector field at locations. Shape: [num_locations, 2]
        plt_config (dict): kwargs passed to plt.plot().
    """

    assert locations.shape == vector_field.shape
    assert locations.shape[-1] == 2
    assert locations.ndim == 2

    locations, vector_field = remove_zero_vector_fields(locations, vector_field)

    # plot vector field as quiver plot
    x_locations, y_locations = locations[:, 0], locations[:, 1]
    x_vector_field, y_vector_field = vector_field[:, 0], vector_field[:, 1]

    ax.quiver(x_locations, y_locations, x_vector_field, y_vector_field, **plt_config)


def plot_3D_vector_field_in_axis(ax: plt.Axes, locations: Tensor, vector_field: Tensor, **plt_config) -> None:
    """
    Plot vector field into provided axis using passed plot configuration.

    Args:
        ax (plt.Axes): Axis to plot paths into using passed plotting config
        locations (Tensor): Set of points where vector field is evaluated at. Shape: [num_locations, 3]
        vector_field (Tensor): Values of vector field at locations. Shape: [num_locations, 3]
        plt_config (dict): kwargs passed to plt.plot().
    """

    assert locations.shape == vector_field.shape
    assert locations.shape[-1] == 3
    assert locations.ndim == 2

    locations, vector_field = remove_zero_vector_fields(locations, vector_field)

    # plot vector field as quiver plot
    x_locations, y_locations, z_locations = locations[:, 0], locations[:, 1], locations[:, 2]
    x_vector_field, y_vector_field, z_vector_field = vector_field[:, 0], vector_field[:, 1], vector_field[:, 2]

    ax.quiver(x_locations, y_locations, z_locations, x_vector_field, y_vector_field, z_vector_field, length=0.05, **plt_config)


def remove_zero_vector_fields(locations: Tensor, vector_field: Tensor) -> tuple[Tensor]:
    """
    Removes locations where vector_field is zero.

    Args:
        locations (Tensor): Shape [num_locations, D]
        vector_field (Tensor): Shape [num_locations, D]

    Returns:
        locations (Tensor): Shape [num_non_zero_locations, D]
        vector_field (Tensor): Shape [num_non_zero_locations, D]

        where num_non_zero_locations are the number of locations where the vector field is non-zero
    """
    vector_norm = torch.sqrt(torch.sum(vector_field**2, dim=-1))  # [num_locations]
    locations = locations[vector_norm != 0.0]
    vector_field = vector_field[vector_norm != 0.0]

    return locations, vector_field


def path_statistics(paths_values: Tensor) -> pd.DataFrame:
    """
    Gather some statistis from path, per dimension and also norm of values.
    Include: mean, std, min, max, percentage larger than +-10.

    Args:
        paths_values (Tensor): Paths: Shape [num_paths, num_obs, D]

    Returns:
        stats_df (pd.DataFrame): Dataframe with mean, std, min, max, percentage larger than +-10, per dimension and norm of values.
    """

    D = paths_values.shape[-1]
    paths_values = paths_values.reshape(-1, D)  # [num_paths * num_obs, D]

    paths_df = pd.DataFrame(data=paths_values, columns=["Dim " + str(i) for i in range(D)])
    paths_df["Norm"] = torch.sqrt(torch.sum(paths_values**2, dim=-1))

    _min = paths_df.min()
    _max = paths_df.max()
    _mean = paths_df.mean()
    _std = paths_df.std()
    _perc = (paths_df.abs() > 10).mean()

    stats_df = pd.DataFrame({"Mean": _mean, "Std": _std, "Min": _min, "Max": _max, "|.|>10": _perc})

    return stats_df
