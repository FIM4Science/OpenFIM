from fim.data.data_generation.dynamical_systems_target import generate_all
from fim.data.datasets import FIMSDEDatabatchTuple
from fim.models import FIMSDEConfig
from fim.models.sde import FIMSDE
from fim.pipelines.sde_pipelines import FIMSDEPipeline
from fim.utils.helper import select_dimension_for_plot
from fim.utils.plots.sde_estimation_plots import (
    plot_1d_vf_real_and_estimation,
    plot_2d_vf_real_and_estimation,
    plot_3d_vf_real_and_estimation,
)


def test_plot_1d(target_data: FIMSDEDatabatchTuple):
    model_config = FIMSDEConfig()
    model = FIMSDE(model_config)
    pipeline = FIMSDEPipeline(model)
    pipeline_output = pipeline(target_data)

    selected_data = select_dimension_for_plot(
        1,
        target_data.dimension_mask,
        target_data.obs_times,
        target_data.obs_values,
        target_data.locations,
        pipeline_output.drift_at_locations_estimator,
        pipeline_output.diffusion_at_locations_estimator,
        target_data.drift_at_locations,
        target_data.diffusion_at_locations,
        pipeline_output.path,
    )

    (
        obs_times,
        obs_values,
        locations,
        drift_at_locations_real,
        diffusion_at_locations_real,
        drift_at_locations_estimation,
        diffusion_at_locations_estimation,
        paths_estimation,
    ) = selected_data
    plot_1d_vf_real_and_estimation(
        locations,
        drift_at_locations_real,
        drift_at_locations_estimation,
        diffusion_at_locations_real,
        diffusion_at_locations_estimation,
        show=True,
    )


def test_plot_2d(target_data: FIMSDEDatabatchTuple):
    model_config = FIMSDEConfig()
    model = FIMSDE(model_config)
    pipeline = FIMSDEPipeline(model)
    pipeline_output = pipeline(target_data)

    selected_data = select_dimension_for_plot(
        2,
        target_data.dimension_mask,
        target_data.obs_times,
        target_data.obs_values,
        target_data.locations,
        pipeline_output.drift_at_locations_estimator,
        pipeline_output.diffusion_at_locations_estimator,
        target_data.drift_at_locations,
        target_data.diffusion_at_locations,
        pipeline_output.path,
    )

    (
        obs_times,
        obs_values,
        locations,
        drift_at_locations_real,
        diffusion_at_locations_real,
        drift_at_locations_estimation,
        diffusion_at_locations_estimation,
        paths_estimation,
    ) = selected_data
    plot_2d_vf_real_and_estimation(
        locations,
        drift_at_locations_real,
        drift_at_locations_estimation,
        diffusion_at_locations_real,
        diffusion_at_locations_estimation,
        show=True,
    )


def test_plot_3d(target_data: FIMSDEDatabatchTuple):
    model_config = FIMSDEConfig()
    model = FIMSDE(model_config)
    pipeline = FIMSDEPipeline(model)
    pipeline_output = pipeline(target_data)

    selected_data = select_dimension_for_plot(
        3,
        target_data.dimension_mask,
        target_data.obs_times,
        target_data.obs_values,
        target_data.locations,
        pipeline_output.drift_at_locations_estimator,
        pipeline_output.diffusion_at_locations_estimator,
        target_data.drift_at_locations,
        target_data.diffusion_at_locations,
        pipeline_output.path,
    )

    (
        obs_times,
        obs_values,
        locations,
        drift_at_locations_real,
        diffusion_at_locations_real,
        drift_at_locations_estimation,
        diffusion_at_locations_estimation,
        paths_estimation,
    ) = selected_data
    plot_3d_vf_real_and_estimation(
        locations,
        drift_at_locations_real,
        drift_at_locations_estimation,
        diffusion_at_locations_real,
        diffusion_at_locations_estimation,
        your_fixed_x_value=0.1,
        show=True,
    )


if __name__ == "__main__":
    target_data = generate_all(128, 50)
    # test_plot_1d(target_data)
    test_plot_2d(target_data)
    # test_plot_3d(target_data)
