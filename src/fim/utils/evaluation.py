import json
import os
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel
from tqdm.auto import tqdm
from transformers import AutoModel

from fim.trainers.utils import get_accel_type, move_batch_to_local_rank
from fim.utils.helper import load_yaml, yaml

from ..data.dataloaders import DataLoaderFactory
from ..models import AModel
from .helper import export_list_of_dicts_to_jsonl


# TODO: this is a temporary function to load a trained model from a given model ID. Once all the models are stored as a HuggingFace model, we can remove this function.
def load_trained_model(model_id: str) -> AModel:
    """Load a trained model from a given model ID.

    Args:
        model_id (str): The ID of the model to load.

    Returns:
        AModel: The loaded model.
    """
    try:
        model = AutoModel.from_pretrained(model_id)
    except Exception as e1:
        try:
            model = AModel.load_model(model_id)
        except Exception as e2:
            raise RuntimeError(f"Failed to load model using both HuggingFace and custom method: {e1}, {e2}")
    logger.info(f"Loaded model {model_id}")

    return model


class EvaluationConfig(BaseModel):
    evaluation_type: str
    evaluation_dir: Path
    accelerator: str = "auto"
    model_id: str | Path
    datasets: dict | list[dict]

    @classmethod
    def from_yaml(cls, path: Path | str) -> "EvaluationConfig":
        try:
            yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))
            config_dict = load_yaml(path)
            evaluation_type = config_dict.get("evaluation_type")
            if not evaluation_type:
                raise ValueError("evaluation_type is required in the config file")

            # Get the appropriate config class from the factory
            config_class = EvaluationConfigFactory.create(evaluation_type)
            config = config_class(**config_dict)
            logger.info(f"Loaded config from {path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise


class EvaluationConfigFactory:
    _config_classes = {}

    @classmethod
    def register(cls, evaluation_type: str, config_class: type[EvaluationConfig]):
        """Register a new evaluation config class."""
        cls._config_classes[evaluation_type] = config_class

    @classmethod
    def create(cls, evaluation_type: str) -> type[EvaluationConfig]:
        """Get the appropriate config class for the given evaluation type."""
        config_class = cls._config_classes.get(evaluation_type)
        if not config_class:
            raise ValueError(f"No config class registered for evaluation type: {evaluation_type}")
        return config_class

    @classmethod
    def get_available_types(cls) -> list[str]:
        """Get a list of all registered evaluation types."""
        return list(cls._config_classes.keys())


class Evaluation:
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        if self.config.accelerator == "auto":
            self.device = get_accel_type()
        else:
            self.device = self.config.accelerator
        self.predictions = []
        self.config.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.model = load_trained_model(self.config.model_id)
        self.model.to(self.device, dtype=torch.bfloat16)

    @abstractmethod
    def evaluate(self):
        """Run evaluation. The implementation of this method depends on the evaluation task.

        Raises:
            NotImplementedError: if the method is not implemented for the specific task.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self):
        """Save the evaluation results.

        Raises:
            NotImplementedError: if the method is not implemented for the specific task.
        """
        raise NotImplementedError()


class EvaluationFactory:
    evaluation_types = {}

    @classmethod
    def register(cls, evaluation_type, evaluation_class):
        cls.evaluation_types[evaluation_type] = evaluation_class

    @classmethod
    def create(cls, evaluation_type, **kwargs) -> Evaluation:
        evaluation_class = cls.evaluation_types.get(evaluation_type)
        if evaluation_class:
            return evaluation_class(**kwargs)
        else:
            raise ValueError("Invalid evaluation type")

    @classmethod
    def get_available_types(cls) -> list[str]:
        """Get a list of all registered evaluation types."""
        return list(cls.evaluation_types.keys())


class HawkesPPDatasetConfig(BaseModel):
    label: str
    dataloader: dict
    kernel_grids_sampler: dict


class HawkesPPEvaluationConfig(EvaluationConfig):
    evaluation_type: str = "hawkes_pp"
    datasets: HawkesPPDatasetConfig | list[HawkesPPDatasetConfig]


EvaluationConfigFactory.register("hawkes_pp", HawkesPPEvaluationConfig)


class HawkesPPEvaluation(Evaluation):
    def __init__(self, config: HawkesPPEvaluationConfig) -> None:
        super().__init__(config)
        self.kernels: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        self.kernel_grids: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

    @torch.inference_mode()
    def evaluate(self):
        logger.info("Starting evaluation...")
        self.model.eval()
        for dataset_eval_conf in self.config.datasets:
            label = dataset_eval_conf.label
            dataloader = self.__load_dataset(label, dataset_eval_conf.dataloader)
            self.__kernel_inference(dataset_eval_conf, label, dataloader)
            predictions = []
            for batch in tqdm(dataloader.test_it, desc=f"Evaluating {label}"):
                with torch.amp.autocast(str(self.device), enabled=True, dtype=torch.bfloat16):
                    logger.info("Evaluating batch for prediction ...")
                    batch = move_batch_to_local_rank(batch, self.device)
                    predictions = self.model.predict_one_step_at_every_event(batch)
                    predictions.extend(predictions)

    def __kernel_inference(self, dataset_eval_conf, label, dataloader):
        kernel_grid, kernel_evaluaiton = self.model.kernel_inference_from_datataset(
            dataloader.train, label, dataloader.train_it, dataset_eval_conf.kernel_grids_sampler
        )
        self.kernels[label] = kernel_evaluaiton
        self.kernel_grids[label] = kernel_grid

    def plot_kernel_grids_histograms(self):
        for label, kernel_grid in self.kernel_grids.items():
            plt.hist(kernel_grid[0, 0, :].cpu().numpy(), bins=30, edgecolor="black")
            plt.title(f"Kernel grid histogram for {label}")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.savefig(Path(self.config.evaluation_dir) / label / "kernel_grid_histogram.png")
            plt.close()

    def plot_kernels(self):
        for label, kernels in self.kernels.items():
            kernel_values = kernels["predicted_kernel_values"]  # Shape: [M, L]
            num_kernels = kernel_values.shape[0]

            # Calculate grid dimensions
            num_cols = min(3, num_kernels)  # Max 3 columns
            num_rows = (num_kernels + num_cols - 1) // num_cols

            # Create subplot grid
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
            if num_kernels == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            # Plot each kernel
            for i in range(num_kernels):
                axes[i].plot(self.kernel_grids[label][0, 0].cpu(), kernel_values[i].cpu())
                axes[i].set_title(f"Kernel {i + 1}")
                axes[i].set_xlabel("Time")
                axes[i].set_ylabel("Value")
                axes[i].spines[["top", "right"]].set_visible(False)

            # Remove empty subplots
            for i in range(num_kernels, len(axes)):
                fig.delaxes(axes[i])

            # Save plot
            label_dir = Path(self.config.evaluation_dir) / label
            label_dir.mkdir(parents=True, exist_ok=True)
            plot_path = label_dir / "kernel_values.png"
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

    def __load_dataset(self, label: str, dataset_conf: dict):
        logger.info(f"Loading dataset: {label}")
        dataloader = DataLoaderFactory.create(dataset_conf.pop("name"), **dataset_conf)
        logger.info(f"Loaded dataset: {dataloader}")
        return dataloader

    def save(self):
        logger.info("Saving evaluation results to JSON ...")
        for label, kernels in self.kernels.items():
            results = {key: value.cpu().numpy().tolist() for key, value in kernels.items()}
            label_dir = Path(self.config.evaluation_dir) / label
            label_dir.mkdir(parents=True, exist_ok=True)
            path = label_dir / "evaluation_results.json"
            with open(path, "w") as f:
                json.dump(results, f)
            logger.info(f"Evaluation results for {label} saved to {path}")
        logger.info("Plotting kernel grids histograms ...")
        self.plot_kernel_grids_histograms()
        logger.info("Plotting kernels ...")
        self.plot_kernels()


EvaluationFactory.register("hawkes_pp", HawkesPPEvaluation)


class PatchedTimeSeriesEvaluation(Evaluation):
    """Patched Time Series Evaluation."""

    def __init__(
        self,
        device_map: str,
        output_path: Union[Path, str],
        dataset_param: dict,
        model_param: dict,
        model_checkpoint_path: str,
        max_new_tokens: int = 1,
    ) -> None:
        super().__init__(device_map, output_path, dataset_param, model_param, model_checkpoint_path)

        self.max_new_tokens = max_new_tokens

        self.local_rank = 0 if torch.cuda.is_available() and self.device else "cpu"

    def evaluate(self, max_new_tokens: Optional[int] = None):
        """
        Run evaluation.

        Currently only implemented for synthetic data prediction (only 1 output token)

        creates list of dictionaries with keys: target, prediction, loss (all losses that are computed by model)
        """
        max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if max_new_tokens > 1:
            raise NotImplementedError("max_new_tokens > 1 is not yet supported.")

        dataset = getattr(self.dataloader, self.dataset_param["split"] + "_it")

        for x in tqdm(dataset, desc="Evaluating"):
            self._move_batch_to_local_rank(x)
            with torch.no_grad():
                prediction = self.model(x)

            self.predictions.extend(
                {
                    "target": t.cpu(),
                    "prediction": p.cpu(),
                    "input": i.cpu(),
                    "mask_point_level": m.cpu(),
                    "loss": self.model.loss(p, t),
                }
                for t, p, i, m in zip(x["output_values"], prediction["predictions"], x["input_values"], x["mask_point_level"])
            )

    def _move_batch_to_local_rank(self, batch):
        for key in batch.keys():
            batch[key] = batch[key].to(self.local_rank)

    def save(self):
        """Save prediction results"""
        self.output_path.mkdir(parents=True, exist_ok=True)
        # transform tensors to numpy arrays
        for data in self.predictions:
            data["target"] = data["target"].tolist()
            data["prediction"] = data["prediction"].tolist()
            data["input"] = data["input"].tolist()
            data["mask_point_level"] = data["mask_point_level"].tolist()
            data["loss"] = {k: v.item() for k, v in data["loss"].items()}
        export_list_of_dicts_to_jsonl(self.predictions, self.output_path / "predictions.jsonl")

    def visualize(self, indices: Optional[list[int]] = None):
        """
        Visualize the predictions: plot input sequence & target & prediction.

        Note: currently hard coded for synthetic data (grid size = 1/640)
        Args:
            indices (list[int]): indices of the predictions to visualize. If None, 16 random predictions will be visualized.

        Returns:
            fig, axes: matplotlib figure and axes
        """
        import matplotlib.pyplot as plt

        if indices is None:
            indices = np.random.choice(len(self.predictions), 16)

        grid_size = 1 / 640

        num_plots = int(np.ceil(len(indices) ** 0.5))
        fig, axes = plt.subplots(num_plots, num_plots, figsize=(10, 10))
        for i, ax in zip(indices, axes.flatten()):
            input = self.predictions[i]["input"]
            input = input[~self.predictions[i]["mask_point_level"].cpu()].flatten().cpu()

            ax.plot(np.linspace(0, input.shape[0] * grid_size, num=input.shape[0]), input, label="input")

            x_output = np.linspace(input.shape[0] * grid_size, (input.shape[0] + 128) * grid_size, num=128)
            ax.plot(x_output, self.predictions[i]["prediction"][-1], label="prediction")
            ax.plot(x_output, self.predictions[i]["target"], linestyle="--", label="target")

            loss = round(self.predictions[i]["loss"]["loss"].item(), 4)
            ax.set_title(f"Loss: {loss}")
            ax.legend()

            ax.spines[["top", "right"]].set_visible(False)

        # remove not used axes
        for i in range(len(indices), num_plots**2):
            fig.delaxes(axes.flatten()[i])

        fig.suptitle("Predictions for given input")
        fig.tight_layout()

        return fig, axes


EvaluationFactory.register("patched_ts", PatchedTimeSeriesEvaluation)


class TimeSeriesEvaluation(Evaluation):
    """Patched Time Series Evaluation."""

    def __init__(
        self,
        device_map: str,
        experiment_dir: Union[Path, str],
        dataset_param: dict,
        model_param: dict,
        model_checkpoint: str,
        sample_indices: Optional[list[int]] = None,
        plot_certainty: bool = True,
    ) -> None:
        output_path = Path(experiment_dir) / "evaluation"
        model_checkpoint_path = Path(experiment_dir) / "checkpoints" / model_checkpoint / "model-checkpoint.pth"
        super().__init__(
            device=device_map,
            output_path=output_path,
            dataset_param=dataset_param,
            model_param=model_param,
            model_checkpoint_path=model_checkpoint_path,
        )

        self.output_path = self.output_path / f"epoch-{self.last_epoch}"
        os.makedirs(self.output_path, exist_ok=True)

        self.plot_certainty = plot_certainty
        self.metrics = []
        self.avg_metrics = {}
        self.sample_indices = sample_indices
        self.init_condition: list[tuple] = []  # mean, std

        self.local_rank = 0 if torch.cuda.is_available() and self.device else "cpu"

    def evaluate(self):
        """
        Run evaluation of model on given data.

        Want per sample
            - metrics
            - prediction (for visualization) & target
        """
        dataset = getattr(self.dataloader, self.dataset_param["split"] + "_it")

        for x in tqdm(dataset, desc="Evaluating"):
            self._move_batch_to_local_rank(x)
            with torch.no_grad():
                model_output = self.model(x, training=False)

            for sample_id in range(len(model_output["metrics"]["mse"])):
                metrics_entry = {key: value[sample_id].item() for key, value in model_output["metrics"].items()}
                self.metrics.append(metrics_entry)

                predictions_entry = {
                    "fine_grid_grid": x["fine_grid_grid"][sample_id].cpu().flatten().tolist(),
                    "solution": {
                        key: value[sample_id].cpu().flatten().tolist() for key, value in model_output["visualizations"]["solution"].items()
                    },
                    "drift": {
                        key: value[sample_id].cpu().flatten().tolist() for key, value in model_output["visualizations"]["drift"].items()
                    },
                    "init_condition": {
                        key: value[sample_id].cpu().flatten().tolist()
                        for key, value in model_output["visualizations"]["init_condition"].items()
                    },
                }
                self.predictions.append(predictions_entry)

        # calculate average metrics
        for key in self.metrics[0].keys():
            self.avg_metrics[key] = np.mean([m[key] for m in self.metrics])

    def _move_batch_to_local_rank(self, batch):
        for key in batch.keys():
            batch[key] = batch[key].to(self.local_rank)

    def save(self, save_dir: Optional[str] = None):
        """Save prediction results"""
        if save_dir is None:
            save_dir = self.output_path

        save_dir.mkdir(parents=True, exist_ok=True)
        export_list_of_dicts_to_jsonl(self.predictions, save_dir / "predictions.jsonl")
        export_list_of_dicts_to_jsonl(self.metrics, save_dir / "metrics.jsonl")
        export_list_of_dicts_to_jsonl([self.avg_metrics], save_dir / "avg_metrics.jsonl")

    def report(self):
        """Print avg metrics."""
        print(json.dumps(self.avg_metrics, indent=4))

    def visualize_solutions(self, indices: Optional[list[int]] = None, save_dir: Optional[Path] = None):
        """
        Visualize the predicted solution: plot input sequence & target & prediction.

        Args:
            indices (list[int]): indices of the predictions to visualize. If None, 16 random predictions will be visualized.
            save_dir (Path): path to save the plots. If None, the plots will be shown.

        Returns:
            fig, axes: matplotlib figure and axes
        """
        # ensure to use the same indices as in drift plot if not specified differently
        if indices is not None:
            self.sample_indices = indices
        elif indices is None and self.sample_indices is None:
            self.sample_indices = np.sort(np.random.choice(len(self.predictions), 9))
        elif self.sample_indices is None:
            self.sample_indices = indices

        # prediction keys : dict_keys(['observation_values', 'observation_times', 'learnt_solution', 'target_path', 'fine_grid_times', 'target_drift', 'learnt_drift', 'learnt_std_drift'])
        num_plots = int(np.ceil(len(self.sample_indices) ** 0.5))
        fig, axes = plt.subplots(num_plots, num_plots, figsize=(10, 10))
        for i, ax in zip(self.sample_indices, axes.flatten()):
            sample_data = self.predictions[i]
            fine_grid_grid = sample_data["fine_grid_grid"]
            obs_mask = sample_data.get("solution").get("observation_mask")
            obs_values = [v for masked, v in zip(obs_mask, sample_data.get("solution").get("observation_values")) if not masked]
            obs_times = [v for masked, v in zip(obs_mask, sample_data.get("solution").get("observation_times")) if not masked]

            # ground truth
            ax.plot(
                fine_grid_grid,
                sample_data.get("solution").get("target"),
                label="Ground truth path",
                alpha=0.4,
                color="orange",
            )
            ax.scatter(
                obs_times,
                obs_values,
                marker="x",
                label="Observations",
                color="orange",
            )

            # prediction
            ax.plot(fine_grid_grid, sample_data.get("solution").get("learnt"), label="Inference path", color="blue")
            ax.spines[["top", "right"]].set_visible(False)

        # remove not used axes
        for i in range(len(self.sample_indices), num_plots**2):
            fig.delaxes(axes.flatten()[i])

        fig.suptitle("Solutions")
        axes[0, 0].legend()
        fig.tight_layout()

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / "solutions.png")

        return fig, axes

    def visualize_drift(self, indices: Optional[list[int]] = None, save_dir: Optional[Path] = None):
        """
        Visualize the predicted drift: plot target & prediction & certainty of prediction.

        Args:
            indices (list[int]): indices of the predictions to visualize. If None, 16 random predictions will be visualized.
            save_dir (Path): path to save the plots. If None, the plots will be shown.

        Returns:
            fig, axes: matplotlib figure and axes
        """
        # ensure to use the same indices as in solution plot if not specified differently
        if indices is not None:
            self.sample_indices = indices
        elif indices is None and self.sample_indices is None:
            self.sample_indices = np.random.choice(len(self.predictions), 9)
        elif self.sample_indices is None:
            self.sample_indices = indices

        num_plots = int(np.ceil(len(self.sample_indices) ** 0.5))
        fig, axes = plt.subplots(num_plots, num_plots, figsize=(10, 10))
        for i, ax in zip(self.sample_indices, axes.flatten()):
            sample_data = self.predictions[i]
            fine_grid_times = sample_data["fine_grid_grid"]

            # ground truth
            ax.plot(
                fine_grid_times,
                sample_data.get("drift", {}).get("target", None),
                label="Ground truth drift",
                alpha=0.4,
                color="orange",
            )

            # prediction
            ax.plot(fine_grid_times, sample_data.get("drift", {}).get("learnt", None), label="Inference drift", color="blue")
            if self.plot_certainty:
                ax.fill_between(
                    fine_grid_times,
                    np.array(sample_data.get("drift", {}).get("learnt", None))
                    - np.array(sample_data.get("drift", {}).get("certainty", None)),
                    np.array(sample_data.get("drift", {}).get("learnt", None))
                    + np.array(sample_data.get("drift", {}).get("certainty", None)),
                    alpha=0.3,
                    color="blue",
                    label="Certainty",
                )
            ax.spines[["top", "right"]].set_visible(False)

        # remove not used axes
        for i in range(len(self.sample_indices), num_plots**2):
            fig.delaxes(axes.flatten()[i])
        axes[0, 0].legend()
        fig.suptitle("Drift")
        fig.tight_layout()

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / "drift.png")

        return fig, axes

    def visualize_init_condition(self, indices: Optional[list[int]] = None, save_dir: Optional[Path] = None):
        n_predictions = len(self.predictions)

        if indices is None and self.sample_indices is None:
            indices = list(range(n_predictions))
        elif indices is None:
            indices = self.sample_indices

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.errorbar(
            range(len(indices)),
            [self.predictions[p_id].get("init_condition").get("learnt")[0] for p_id in range(n_predictions) if p_id in indices],
            yerr=[self.predictions[p_id].get("init_condition").get("certainty")[0] for p_id in range(n_predictions) if p_id in indices],
            fmt="o",
            color="blue",
            label="Inference init. condition",
        )
        ax.scatter(
            range(len(indices)),
            [self.predictions[p_id].get("init_condition").get("target")[0] for p_id in range(n_predictions) if p_id in indices],
            marker="x",
            color="orange",
            label="Ground truth init. condition",
        )
        ax.set_title("Initial Condition")
        ax.spines[["top", "right"]].set_visible(False)

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / "init_condition.png")
        fig.legend()
        return fig, ax

    def visualize(self, indices: Optional[list[int]] = None, save_dir: Optional[Path] = None):
        """
        Call visualization functions for drift, solution and initial condition.

        Args:
            indices (list[int]): indices of the predictions to visualize. If None, 16 random predictions will be visualized.
            save_dir (Path): path to save the plots. If None, the plots will be shown.

        Returns:
            fig, axes: matplotlib figure and axes
        """
        if indices is not None:
            self.sample_indices = indices
        elif indices is None and self.sample_indices is None:
            self.sample_indices = np.sort(np.random.choice(len(self.predictions), 9))
        elif self.sample_indices is None:
            self.sample_indices = indices

        plot_drift = self.visualize_drift(save_dir=save_dir)
        plot_sol = self.visualize_solutions(save_dir=save_dir)
        plot_init_cond = self.visualize_init_condition(save_dir=save_dir)
        plot_init_cond_distr = self.visualize_distribution_init_conditions(save_dir=save_dir)

        return plot_drift, plot_sol, plot_init_cond, plot_init_cond_distr

    def visualize_distribution_init_conditions(self, save_dir: Optional[str]):
        init_conds = [p.get("init_condition").get("learnt")[0] for p in self.predictions]
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.hist(init_conds)
        ax.set_title("Distribution Init. Condition")
        ax.spines[["top", "right"]].set_visible(False)
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / "init_condition_distribtion.png")
        return fig, ax


EvaluationFactory.register("ts", TimeSeriesEvaluation)


def create_evaluation_from_config(config_path: str | Path) -> Evaluation:
    """Create an evaluation instance from a YAML config file.

    This helper function combines loading the config and creating the evaluation instance
    into a single step. It will:
    1. Load the config from the YAML file
    2. Create the appropriate evaluation instance based on the config

    Args:
        config_path (str | Path): Path to the YAML config file

    Returns:
        Evaluation: An instance of the appropriate evaluation class

    Raises:
        ValueError: If the evaluation type is not registered
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is invalid
    """
    # Load the config
    config = EvaluationConfig.from_yaml(config_path)

    # Create the evaluation instance
    return EvaluationFactory.create(config.evaluation_type, config=config)
