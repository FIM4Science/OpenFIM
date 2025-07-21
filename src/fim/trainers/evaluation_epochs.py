from abc import ABC, abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import optree
import torch
from torch import Tensor

from fim.data.dataloaders import BaseDataLoader
from fim.models.blocks import AModel
from fim.models.sde import SDEConcepts
from fim.sampling.sde_path_samplers import fimsde_sample_paths, fimsde_sample_paths_on_masked_grid
from fim.trainers.utils import TrainLossTracker
from fim.utils.sde.vector_fields_and_paths_plots import (
    plot_1d_vf_real_and_estimation,
    plot_2d_vf_real_and_estimation,
    plot_3d_vf_real_and_estimation,
    plot_paths,
)


class EvaluationEpoch(ABC):
    model: AModel
    dataloader: BaseDataLoader
    loss_tracker: TrainLossTracker
    debug_mode: bool

    # accelerator handling
    local_rank: int
    accel_type: str
    use_mixeprecision: bool
    is_accelerator: bool
    auto_cast_type: torch.dtype

    def __init__(
        self,
        model: AModel,
        dataloader: BaseDataLoader,
        loss_tracker: TrainLossTracker,
        debug_mode: bool,
        local_rank: int,
        accel_type: str,
        use_mixeprecision: bool,
        is_accelerator: bool,
        auto_cast_type: torch.dtype,
        **kwargs,
    ):
        self.model: AModel = model
        self.dataloader: BaseDataLoader = dataloader
        self.loss_tracker: TrainLossTracker = loss_tracker

        # accelerator handling
        self.debug_mode: bool = debug_mode
        self.local_rank: int = local_rank
        self.accel_type: str = accel_type
        self.use_mixeprecision: bool = use_mixeprecision
        self.is_accelerator: bool = is_accelerator
        self.auto_cast_type: torch.dtype = auto_cast_type

    @abstractmethod
    def __call__(self, epoch: int) -> dict:
        """
        Run evaluation epoch and return a dict with stats to log.
        Returned dict looks something like {"losses": losses_dict, "figures": figures_dict}, where:
            losses_dict: maps labels to zero-dim. tensors
            figures: maps labels to plt.figures
        """
        raise NotImplementedError("The __call__ method is not implemented in your class!")


class TestEvaluationEpoch(EvaluationEpoch):
    def __init__(
        self,
        model,
        dataloader,
        loss_tracker,
        debug_mode,
        local_rank,
        accel_type,
        use_mixeprecision,
        is_accelerator,
        auto_cast_type,
        **kwargs,
    ):
        super().__init__(
            model, dataloader, loss_tracker, debug_mode, local_rank, accel_type, use_mixeprecision, is_accelerator, auto_cast_type
        )

    def __call__(self, epoch: int) -> dict:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(torch.linspace(0, 1, 10), torch.linspace(-2, 1, 10))

        return {"losses": {"dumm_loss": torch.sum(torch.zeros(1))}, "figures": {"dummy": fig}}


class SDEEvaluationPlots(EvaluationEpoch):
    def __init__(
        self,
        model,
        dataloader,
        loss_tracker,
        debug_mode,
        local_rank,
        accel_type,
        use_mixeprecision,
        is_accelerator,
        auto_cast_type,
        **kwargs,
    ):
        super().__init__(
            model, dataloader, loss_tracker, debug_mode, local_rank, accel_type, use_mixeprecision, is_accelerator, auto_cast_type
        )

        # which dataloader to take plotting data from
        iterator_name: str = kwargs.get("iterator_name", "test")
        if iterator_name == "validation":
            self.dataloader = dataloader.validation_it
        elif iterator_name == "test":
            self.dataloader = dataloader.test_it
        elif iterator_name == "evaluation":
            self.dataloader = dataloader.evaluation_it

        # how often to plot
        self.plot_frequency: int = kwargs.get("plot_frequency", 1)

        # how many paths to show maximal
        self.plot_paths_count: int = kwargs.get("plot_paths_count")

    @staticmethod
    def plot_example_of_dim(
        data: dict,
        estimated_concepts: SDEConcepts,
        sample_paths: Tensor,
        sample_grid: Tensor,
        dimension: int,
        plot_paths_count: Optional[int],
    ) -> tuple[plt.figure]:
        """
        Plot data and model estimates of random example from data.
        """
        # select batch elements of dimension
        has_dim: Tensor[bool] = data["dimension_mask"].sum(dim=-1)[:, 0].long() == dimension  # [B]
        if not torch.any(has_dim).item():
            return None, None

        input_data: tuple = (data, estimated_concepts, sample_paths, sample_grid)
        input_data = optree.tree_map(lambda x: x[has_dim], input_data, namespace="fimsde")

        # extract example at random index
        B = optree.tree_flatten(input_data)[0][0].shape[0]
        index = torch.randint(size=(1,), low=0, high=B).item()
        input_data = optree.tree_map(lambda x: x[index], input_data, namespace="fimsde")

        # truncate last dimension to dimension
        input_data = optree.tree_map(lambda x: x[..., :dimension] if x.shape[-1] >= dimension else x, input_data, namespace="fimsde")

        # prepare for plotting
        input_data = optree.tree_map(lambda x: x.detach().cpu(), input_data, namespace="fimsde")

        if dimension == 1:
            vf_plotting_func = plot_1d_vf_real_and_estimation
        elif dimension == 2:
            vf_plotting_func = plot_2d_vf_real_and_estimation
        elif dimension == 3:
            vf_plotting_func = plot_3d_vf_real_and_estimation
        else:

            def vf_plotting_func(*args, **kwargs):
                return None

        data, estimated_concepts, sample_paths, sample_grid = input_data

        fig_vf = vf_plotting_func(
            data.get("locations"),
            data.get("drift_at_locations"),
            estimated_concepts.drift,
            data.get("diffusion_at_locations"),
            estimated_concepts.diffusion,
            show=False,
        )

        # select paths to plot
        P = data["obs_times"].shape[0]
        perm = torch.randperm(P)
        plot_paths_count = P if plot_paths_count is None else min(plot_paths_count, P)

        fig_paths = plot_paths(
            dimension,
            data["obs_times"][perm][:plot_paths_count],
            data["obs_values"][perm][:plot_paths_count],
            sample_paths[perm][:plot_paths_count],
            sample_grid[perm][:plot_paths_count],
        )

        return fig_vf, fig_paths

    @torch.no_grad()
    def __call__(self, epoch: int) -> dict:
        """
        Plot ground-truth and estimated vector fields and paths for all available dimensions.
        """
        self.model.eval()

        if epoch % self.plot_frequency != 0:
            return {}

        else:
            # find example for all dimensions
            max_dim = 3

            all_dims: tuple[int] = []

            all_examples = []
            all_estimated_concepts = []
            all_sample_paths = []
            all_sample_paths_grid = []

            for dim in range(1, max_dim + 1):
                for batch in iter(self.dataloader):
                    batch = optree.tree_map(lambda x: x.to(self.model.device), batch, namespace="sde")

                    dim_in_batch = (batch["dimension_mask"].sum(dim=-1) == dim)[:, 0]  # [B]

                    # select first element in batch with dim
                    if dim_in_batch.any().item() is True:
                        example_of_dim: dict = optree.tree_map(lambda x: x[dim_in_batch][0][None], batch)
                        all_examples.append(example_of_dim)
                        all_dims.append(dim)

                        break

                example_of_dim: dict = optree.tree_map(lambda x: x.to(self.model.device), example_of_dim)

                # don't need vector fields on paths
                example_of_dim.pop("obs_values_clean", None)
                example_of_dim.pop("drift_at_obs_values", None)
                example_of_dim.pop("diffusion_at_obs_values", None)

                # get concepts and samples from example_of_dim
                with torch.no_grad():
                    with torch.amp.autocast(
                        self.accel_type,
                        enabled=self.use_mixeprecision and self.is_accelerator,
                        dtype=self.auto_cast_type,
                    ):
                        grid_size = example_of_dim["obs_times"].shape[-2]
                        max_steps = 1024
                        solver_granularity = max_steps // grid_size

                        estimated_concepts: SDEConcepts = self.model(example_of_dim, training=False)
                        sample_paths, sample_paths_grid = fimsde_sample_paths(
                            self.model, example_of_dim, grid=example_of_dim["obs_times"], solver_granularity=solver_granularity
                        )
                all_estimated_concepts.append(estimated_concepts)
                all_sample_paths.append(sample_paths)
                all_sample_paths_grid.append(sample_paths_grid)

            # Create figures from model outputs and samples
            figures = {}
            for index, dim in enumerate(all_dims):
                fig_vf, fig_paths = self.plot_example_of_dim(
                    all_examples[index],
                    all_estimated_concepts[index],
                    all_sample_paths[index],
                    all_sample_paths_grid[index],
                    dimension=dim,
                    plot_paths_count=self.plot_paths_count,
                )

                dim_figures = {f"Vector_Field_{str(dim)}D": fig_vf, f"Paths_{str(dim)}D": fig_paths}
                figures.update(dim_figures)

            return {"figures": figures}


class LorenzEvaluationEpoch(EvaluationEpoch):
    def __init__(
        self,
        model,
        dataloader,
        loss_tracker,
        debug_mode,
        local_rank,
        accel_type,
        use_mixeprecision,
        is_accelerator,
        auto_cast_type,
        **kwargs,
    ):
        super().__init__(
            model, dataloader, loss_tracker, debug_mode, local_rank, accel_type, use_mixeprecision, is_accelerator, auto_cast_type
        )

        self.model_type = kwargs.get("model_type")

        # which dataloader to take plotting data from
        iterator_name: str = kwargs.get("iterator_name", "test")
        if iterator_name == "validation":
            self.dataloader = dataloader.validation_it
        elif iterator_name == "test":
            self.dataloader = dataloader.test_it
        elif iterator_name == "evaluation":
            self.dataloader = dataloader.evaluation_it

        # how often to plot
        self.plot_frequency: int = kwargs.get("plot_frequency", 1)

    def __call__(self, epoch: int) -> dict:
        if epoch % self.plot_frequency != 0:
            return {}

        else:
            self.model.eval()

            batch = next(iter(self.dataloader))
            batch = optree.tree_map(lambda x: x.to(self.model.device), batch)
            obs_times = batch["obs_times"]
            obs_values = batch["obs_values"]

            with torch.no_grad():
                with torch.amp.autocast(
                    self.accel_type,
                    enabled=self.use_mixeprecision and self.is_accelerator,
                    dtype=self.auto_cast_type,
                ):
                    if self.model_type == "latentsde":
                        ctx, obs_times, _ = self.model.encode_inputs(obs_times, obs_values)
                        posterior_initial_states, _, _ = self.model.sample_posterior_initial_condition(ctx[0])
                        _, paths_post_init_cond = self.model.sample_from_prior_equation(posterior_initial_states, obs_times)

                        prior_initial_states = self.model.sample_prior_initial_condition(obs_values.shape[0])
                        _, paths_prior_init_cond = self.model.sample_from_prior_equation(prior_initial_states, obs_times)

                    elif self.model_type == "fimsde":
                        paths_post_init_cond = None
                        paths_prior_init_cond, _ = fimsde_sample_paths_on_masked_grid(
                            self.model,
                            batch,
                            obs_times,
                            torch.ones_like(obs_times),
                            initial_states=obs_values[:, :, 0, :],
                            solver_granularity=5,
                        )

            # Create figure from paths
            figures = {}

            paths_prior_init_cond = paths_prior_init_cond.to("cpu").detach().to(torch.float32)
            paths_prior_init_cond = paths_prior_init_cond.squeeze()

            obs_values = obs_values.to("cpu").detach().to(torch.float32)
            obs_values = obs_values.squeeze()

            if paths_post_init_cond is not None:
                paths_post_init_cond = paths_post_init_cond.to("cpu").detach().to(torch.float32)
                paths_post_init_cond = paths_post_init_cond.squeeze()

            fig = plt.Figure(figsize=(5, 5), dpi=300)
            ax = fig.add_axes(111, projection="3d")
            ax.set_axis_off()

            for i in range(obs_values.shape[0]):
                ax.plot(
                    obs_values[i, ..., 0],
                    obs_values[i, ..., 1],
                    obs_values[i, ..., 2],
                    color="black",
                    linestyle="dashed",
                    label="Observations" if i == 0 else None,
                    linewidth=0.2,
                )
                ax.plot(
                    paths_prior_init_cond[i, ..., 0],
                    paths_prior_init_cond[i, ..., 1],
                    paths_prior_init_cond[i, ..., 2],
                    color="#0072B2",
                    linestyle="solid",
                    label="Prior Eq." if i == 0 else None,
                    linewidth=0.2,
                )
                if paths_post_init_cond is not None:
                    ax.plot(
                        paths_post_init_cond[i, ..., 0],
                        paths_post_init_cond[i, ..., 1],
                        paths_post_init_cond[i, ..., 2],
                        color="#CC79A7",
                        linestyle="solid",
                        label="Prior Eq. - Post. Init. Cond." if i == 0 else None,
                        linewidth=0.2,
                    )

            ax.legend()

            figures["sample_paths"] = fig

            return {"figures": figures}


class RealWorldEvaluationEpoch(EvaluationEpoch):
    def __init__(
        self,
        model,
        dataloader,
        loss_tracker,
        debug_mode,
        local_rank,
        accel_type,
        use_mixeprecision,
        is_accelerator,
        auto_cast_type,
        **kwargs,
    ):
        super().__init__(
            model, dataloader, loss_tracker, debug_mode, local_rank, accel_type, use_mixeprecision, is_accelerator, auto_cast_type
        )

        self.model_type = kwargs.get("model_type")

        # which dataloader to take plotting data from
        iterator_name: str = kwargs.get("iterator_name", "test")
        if iterator_name == "validation":
            self.dataloader = dataloader.validation_it
        elif iterator_name == "test":
            self.dataloader = dataloader.test_it
        elif iterator_name == "evaluation":
            self.dataloader = dataloader.evaluation_it

        # how often to plot
        self.plot_frequency: int = kwargs.get("plot_frequency", 1)

    def __call__(self, epoch: int) -> dict:
        if epoch % self.plot_frequency != 0:
            return {}

        else:
            self.model.eval()

            batch = next(iter(self.dataloader))
            batch = optree.tree_map(lambda x: x.to(self.model.device).to(torch.bfloat16), batch)

            with torch.no_grad():
                with torch.amp.autocast(
                    self.accel_type,
                    enabled=self.use_mixeprecision and self.is_accelerator,
                    dtype=self.auto_cast_type,
                ):
                    # get concepts and samples from example_of_dim
                    estimated_concepts: SDEConcepts = self.model(batch, training=False)

            locations = batch["locations"].to("cpu").to(torch.float32).squeeze()
            drift = estimated_concepts.drift.detach().to("cpu").to(torch.float32).squeeze()[:, 0]  # dimension padded to 3
            diffusion = estimated_concepts.diffusion.detach().to("cpu").to(torch.float32).squeeze()[:, 0]  # dimension padded to 3

            figures = {}

            fig = plt.Figure(figsize=(7, 5), dpi=300, tight_layout=True)
            ax_drift = fig.add_axes(111)
            ax_diffusion = ax_drift.twinx()  # instantiate a second Axes that shares the same x-axis

            ax_drift.set_ylabel("Drift", color="r")
            ax_drift.plot(locations, drift, color="r", label="Model Drift")

            ax_diffusion.set_ylabel("Diffusion", color="tab:blue")
            ax_diffusion.plot(locations, diffusion, color="tab:blue", label="Model Diffusion")

            fig.legend()

            figures["vector_fields"] = fig

            return {"figures": figures}


class HawkesEvaluationPlots(EvaluationEpoch):
    """
    Evaluation epoch class for Hawkes processes that creates TensorBoard plots
    comparing target and predicted intensity values for all marks.
    """

    def __init__(
        self,
        model,
        dataloader,
        loss_tracker,
        debug_mode,
        local_rank,
        accel_type,
        use_mixeprecision,
        is_accelerator,
        auto_cast_type,
        **kwargs,
    ):
        super().__init__(
            model, dataloader, loss_tracker, debug_mode, local_rank, accel_type, use_mixeprecision, is_accelerator, auto_cast_type
        )

        # which dataloader to take plotting data from
        iterator_name: str = kwargs.get("iterator_name", "validation")
        if iterator_name == "validation":
            self.data_iterator = dataloader.validation_it
        elif iterator_name == "test":
            self.data_iterator = dataloader.test_it
        elif iterator_name == "evaluation":
            self.data_iterator = dataloader.evaluation_it
        else:
            self.data_iterator = dataloader.validation_it

        # how often to plot
        self.plot_frequency: int = kwargs.get("plot_frequency", 1)

        # which inference path to plot (default: first one)
        self.inference_path_idx: int = kwargs.get("inference_path_idx", 0)

    @staticmethod
    def create_intensity_plots(
        target_intensities: Tensor,
        predicted_intensities: Tensor,
        evaluation_times: Tensor,
        num_marks: int,
        inference_path_idx: int = 0,
    ) -> dict:
        """
        Create plots comparing target and predicted intensities for all marks.

        Args:
            target_intensities: Target intensity values [B, M, P_inference, L_inference]
            predicted_intensities: Predicted intensity values [B, M, P_inference, L_inference]
            evaluation_times: Times at which intensities are evaluated [B, P_inference, L_inference]
            num_marks: Number of marks to plot (avoids plotting zero functions)
            inference_path_idx: Which inference path to plot

        Returns:
            Dictionary of figure names to matplotlib figures
        """
        figures = {}

        # Select first batch element and specified inference path
        target_int = target_intensities[0, :, inference_path_idx, :].detach().cpu()  # [M, L_inference]
        pred_int = predicted_intensities[0, :, inference_path_idx, :].detach().cpu()  # [M, L_inference]
        eval_times = evaluation_times[0, inference_path_idx, :].detach().cpu()  # [L_inference]

        # Create individual plots for each mark
        for mark_idx in range(num_marks):
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create scatter plots for target and predicted intensity for this mark
            ax.scatter(
                eval_times,
                target_int[mark_idx],
                color="blue",
                s=60,
                alpha=0.7,
                label=f"Target Mark {mark_idx}",
                marker="o",
                edgecolors="darkblue",
            )
            ax.scatter(
                eval_times,
                pred_int[mark_idx],
                color="red",
                s=60,
                alpha=0.7,
                label=f"Predicted Mark {mark_idx}",
                marker="^",
                edgecolors="darkred",
            )

            ax.set_xlabel("Time")
            ax.set_ylabel("Intensity")
            ax.set_title(f"Intensity Comparison for Mark {mark_idx}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            figures[f"Intensity_Mark_{mark_idx}"] = fig

        # Create combined plot with all marks
        fig_combined, ax_combined = plt.subplots(figsize=(12, 8))

        colors = plt.cm.tab10(range(num_marks))
        markers_target = ["o", "s", "D", "v", "^", "<", ">", "p", "*", "h"]
        markers_pred = ["^", "v", "X", "<", ">", "P", "d", "8", "H", "+"]

        for mark_idx in range(num_marks):
            # Use different markers for target vs predicted, same color for same mark
            ax_combined.scatter(
                eval_times,
                target_int[mark_idx],
                color=colors[mark_idx],
                s=60,
                alpha=0.8,
                marker=markers_target[mark_idx % len(markers_target)],
                label=f"Target Mark {mark_idx}",
                edgecolors="black",
                linewidths=0.5,
            )
            ax_combined.scatter(
                eval_times,
                pred_int[mark_idx],
                color=colors[mark_idx],
                s=40,
                alpha=0.6,
                marker=markers_pred[mark_idx % len(markers_pred)],
                label=f"Predicted Mark {mark_idx}",
                edgecolors="black",
                linewidths=0.5,
            )

        ax_combined.set_xlabel("Time")
        ax_combined.set_ylabel("Intensity")
        ax_combined.set_title("Intensity Comparison for All Marks")
        ax_combined.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax_combined.grid(True, alpha=0.3)
        plt.tight_layout()

        figures["Intensity_All_Marks"] = fig_combined

        return figures

    @torch.no_grad()
    def __call__(self, epoch: int) -> dict:
        """
        Create intensity plots for target vs predicted values during validation.
        """
        self.model.eval()

        if epoch % self.plot_frequency != 0:
            return {}

        # Get a batch from validation data
        try:
            batch = next(iter(self.data_iterator))
        except StopIteration:
            return {}

        # Move batch to device
        batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Check if we have the required keys for intensity computation
        required_keys = [
            "context_event_times",
            "context_event_types",
            "inference_event_times",
            "inference_event_types",
            "intensity_evaluation_times",
            "kernel_functions",
            "base_intensity_functions",
        ]

        if not all(key in batch for key in required_keys):
            return {}

        # Validate inference path index
        P_inference = batch["inference_event_times"].shape[1]
        if self.inference_path_idx >= P_inference:
            self.inference_path_idx = 0  # Fall back to first path

        # Run model forward pass (this will automatically compute delta_times and normalization)
        with torch.amp.autocast(
            self.accel_type,
            enabled=self.use_mixeprecision and self.is_accelerator,
            dtype=self.auto_cast_type,
        ):
            model_output = self.model(batch)

        # Extract predicted and target intensities from model output
        if "predicted_intensity_values" not in model_output or "target_intensity_values" not in model_output:
            return {}  # Can't create plots without both predicted and target intensities

        predicted_intensities = model_output["predicted_intensity_values"]  # [B, M, P_inference, L_inference]
        target_intensities = model_output["target_intensity_values"]  # [B, M, P_inference, L_inference]

        # Get the actual number of marks from the batch
        num_marks = batch.get("num_marks", self.model.max_num_marks)
        if isinstance(num_marks, torch.Tensor):
            num_marks = num_marks.item()

        # Create plots
        figures = self.create_intensity_plots(
            target_intensities,
            predicted_intensities,
            batch["intensity_evaluation_times"],
            num_marks,
            self.inference_path_idx,
        )

        return {"figures": figures}
