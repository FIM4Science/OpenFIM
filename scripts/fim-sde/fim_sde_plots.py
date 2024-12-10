from fim.data.config_dataclasses import FIMDatasetConfig
from fim.models.config_dataclasses import FIMSDEConfig
from fim.models.sde import FIMSDE
from fim.pipelines.sde_pipelines import FIMSDEPipeline
from fim.utils.helper import select_dimension_for_plot
from fim.utils.plots.sde_estimation_plots import (
    plot_1d_vf_real_and_estimation,
    plot_2d_vf_real_and_estimation,
    plot_3d_vf_real_and_estimation,
)


def test_plot_1d():
    model_config = FIMSDEConfig()
    data_config = FIMDatasetConfig()
    model = FIMSDE(model_config, data_config)
    databatch_target = model.target_data
    pipeline = FIMSDEPipeline(model)
    pipeline_output = pipeline(databatch_target)

    selected_data = select_dimension_for_plot(
        1,
        databatch_target.dimension_mask,
        databatch_target.locations,
        pipeline_output.drift_at_locations_estimator,
        pipeline_output.diffusion_at_locations_estimator,
        databatch_target.drift_at_locations,
        databatch_target.diffusion_at_locations,
        index_to_select=0,
    )

    locations, drift_at_locations_real, diffusion_at_locations_real, drift_at_locations_estimation, diffusion_at_locations_estimation = (
        selected_data
    )
    plot_1d_vf_real_and_estimation(
        locations,
        drift_at_locations_real,
        drift_at_locations_estimation,
        diffusion_at_locations_real,
        diffusion_at_locations_estimation,
        show=True,
    )


def test_plot_2d():
    model_config = FIMSDEConfig()
    data_config = FIMDatasetConfig()
    model = FIMSDE(model_config, data_config)
    databatch_target = model.target_data
    pipeline = FIMSDEPipeline(model)
    pipeline_output = pipeline(databatch_target)

    selected_data = select_dimension_for_plot(
        2,
        databatch_target.dimension_mask,
        databatch_target.locations,
        pipeline_output.drift_at_locations_estimator,
        pipeline_output.diffusion_at_locations_estimator,
        databatch_target.drift_at_locations,
        databatch_target.diffusion_at_locations,
        index_to_select=0,
    )

    locations, drift_at_locations_real, diffusion_at_locations_real, drift_at_locations_estimation, diffusion_at_locations_estimation = (
        selected_data
    )
    plot_2d_vf_real_and_estimation(
        locations,
        drift_at_locations_real,
        drift_at_locations_estimation,
        diffusion_at_locations_real,
        diffusion_at_locations_estimation,
        show=True,
    )


def test_plot_3d():
    model_config = FIMSDEConfig()
    data_config = FIMDatasetConfig()
    model = FIMSDE(model_config, data_config)
    databatch_target = model.target_data
    pipeline = FIMSDEPipeline(model)
    pipeline_output = pipeline(databatch_target)

    selected_data = select_dimension_for_plot(
        3,
        databatch_target.dimension_mask,
        databatch_target.locations,
        pipeline_output.drift_at_locations_estimator,
        pipeline_output.diffusion_at_locations_estimator,
        databatch_target.drift_at_locations,
        databatch_target.diffusion_at_locations,
        index_to_select=0,
    )

    locations, drift_at_locations_real, diffusion_at_locations_real, drift_at_locations_estimation, diffusion_at_locations_estimation = (
        selected_data
    )
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
    test_plot_1d()
    test_plot_2d()
    test_plot_3d()
