import json
import logging
import os
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoModel

from fim.models.hawkes.hawkes import FIMHawkes
from fim.trainers.utils import get_accel_type, move_batch_to_local_rank
from fim.utils.helper import load_yaml, yaml
from fim.utils.interpolator import KernelInterpolator

from ..data.dataloaders import DataLoaderFactory
from ..models import AModel
from ..sampling.grid_samplers import sample_kernel_grid
from ..utils.logging import RankLoggerAdapter
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
    plot_kwargs: Optional[dict] = {}
    kernel_grids_sampler: dict
    load_if_exists: Optional[bool] = False


class HawkesPPEvaluationConfig(EvaluationConfig):
    evaluation_type: str = "hawkes_pp"
    datasets: HawkesPPDatasetConfig | list[HawkesPPDatasetConfig]

    def get_by_label(self, label: str) -> HawkesPPDatasetConfig:
        for dataset_eval_conf in self.datasets:
            if dataset_eval_conf.label == label:
                return dataset_eval_conf
        raise ValueError(f"Dataset with label {label} not found")


EvaluationConfigFactory.register("hawkes_pp", HawkesPPEvaluationConfig)


class HawkesPPEvaluation(Evaluation):
    def __init__(self, config: HawkesPPEvaluationConfig) -> None:
        super().__init__(config)
        self.res: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        self.logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))

    @torch.inference_mode()
    def evaluate(self):
        logger.info("Starting evaluation...")

        for dataset_eval_conf in self.config.datasets:
            label = dataset_eval_conf.label
            if (self.config.evaluation_dir / label / "evaluation_results.json").exists() and dataset_eval_conf.load_if_exists:
                self.logger.info(f"Evaluation results for {label} already exist. Skipping evaluation...")
                self.res[label] = json.load(open(self.config.evaluation_dir / label / "evaluation_results.json"))
                for key, value in self.res[label].items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            if isinstance(v, list):
                                if k not in ["target_intensities", "target_intensity_times"]:
                                    self.res[label][key][k] = np.array(v)
                                else:
                                    self.res[label][key][k] = [np.array(i) for i in v]
                continue
            dataloader = self.__load_dataset(label, dataset_eval_conf.dataloader)
            self.res[label]["kernel_inference"] = self.__kernel_inference(dataset_eval_conf, label, dataloader)
            self.res[label]["one_step_ahead_predictions"] = self.__one_step_ahead_pred(label, dataloader)

    def __one_step_ahead_pred(self, label, dataloader):
        kernel = KernelInterpolator(
            torch.from_numpy(self.res[label]["kernel_inference"]["sampled_kernel_grids"]).to(self.device),
            torch.from_numpy(self.res[label]["kernel_inference"]["predicted_kernel_values"]).to(self.device),
        )
        # kernel = KernelInterpolator(
        #     torch.from_numpy(self.res[label]["kernel_inference"]["target_kernel_grids"]).to(self.device),
        #     torch.from_numpy(self.res[label]["kernel_inference"]["target_kernel_values"]).to(self.device),
        # )
        intensity_fn = partial(
            FIMHawkes.intentsity,
            kernel=kernel,
            base_intensity=torch.from_numpy(self.res[label]["kernel_inference"]["target_base_intensity"]).to(self.device),
        )
        predictions = []
        for batch in tqdm(dataloader.test_it, desc=f"Evaluating {label}"):
            with torch.amp.autocast(str(self.device), enabled=True, dtype=torch.bfloat16):
                logger.info("Evaluating batch for prediction ...")
                batch = move_batch_to_local_rank(
                    batch, self.device, ignore_keys=["seq_idx", "target_intensities", "target_intensity_times"]
                )
                batch["delta_times"] = batch["delta_times"].squeeze(-1)
                t_values = torch.stack(
                    [torch.linspace(0, batch["event_times"][i, -1], 1000, device=self.device) for i in range(batch["delta_times"].shape[0])]
                )
                intensity = intensity_fn(t_values.unsqueeze(-1), batch["event_times"].to(self.device))
                prediction = self.model.predict_one_step_at_every_event(batch, intensity_fn)
                target_intensities = dataloader.test.data[batch["seq_idx"]]["target_intensities"]
                target_intensity_times = dataloader.test.data[batch["seq_idx"]]["target_intensity_times"]
                if isinstance(target_intensities, torch.Tensor):
                    target_intensities = [target_intensities[0]]
                if isinstance(target_intensity_times, torch.Tensor):
                    target_intensity_times = [target_intensity_times[0]]
                prediction["predicted_sampled_intensities"] = intensity
                prediction["predicted_sampled_intensity_times"] = t_values
                prediction["target_intensities"] = target_intensities
                prediction["target_intensity_times"] = target_intensity_times
                prediction["target_event_times"] = batch["event_times"]
                prediction["seq_idx"] = batch["seq_idx"]
                predictions.append(prediction)
        out = {
            key: torch.concat([prediction[key].cpu().float() for prediction in predictions]).numpy()
            for key in predictions[0].keys()
            if key not in ["target_intensities", "target_intensity_times"]
        }
        for key in ["target_intensities", "target_intensity_times"]:
            tmp = []
            for prediction in predictions:
                tmp.extend([p.numpy() for p in prediction[key]])
            out[key] = tmp
        return out

    def __kernel_inference(self, dataset_eval_conf, label, dataloader):
        max_intra_event_time = self.__get_max_intra_event_time(dataloader.train)
        sampled_kernel_grids = self.__sample_kernel_grids(dataset_eval_conf.kernel_grids_sampler, max_intra_event_time)
        infered_kernels = []
        for batch in tqdm(dataloader.train_it, desc=f"Evaluating {label}"):
            with torch.amp.autocast(str(self.device), enabled=True, dtype=torch.bfloat16):
                self.logger.info("Evaluating batch for prediction ...")
                batch["kernel_grids"] = sampled_kernel_grids
                batch = move_batch_to_local_rank(batch, self.device, ignore_keys=["seq_idx"])
                prediction = self.model(batch)
                infered_kernels.append(prediction)
        with torch.amp.autocast(str(self.device), enabled=True, dtype=torch.bfloat16):
            B = batch["kernel_grids"].shape[0]
            batch["kernel_grids"] = dataloader.train.data[0]["target_kernel_grids"].to(self.device).repeat(B, 1, 1)
            prediction = self.model(batch)

        out = self.__average_over_batches(infered_kernels)
        out["sampled_kernel_grids"] = sampled_kernel_grids.cpu().float()[0].numpy()
        out["target_kernel_grids"] = dataloader.train.data[0]["target_kernel_grids"].numpy()
        out["target_base_intensity"] = dataloader.train.data[0]["target_base_intensities"].numpy()
        out["target_kernel_values"] = dataloader.train.data[0]["target_kernel_evaluations"].numpy()
        out["predicted_kernel_values_on_target_grids"] = prediction["predicted_kernel_values"].cpu().float().numpy()
        return out

    def __average_over_batches(self, kernel_batches: list[dict]) -> dict[str, Tensor]:
        self.logger.info("Averaging over batches ...")
        averaged_kernels = defaultdict(list)

        for kernel_batch in kernel_batches:
            for key, value in kernel_batch.items():
                averaged_kernels[key].append(value)

        return {key: torch.concat(values).mean(dim=0).float().cpu().numpy() for key, values in averaged_kernels.items()}

    def __sample_kernel_grids(self, kernel_grids_sampler_conf: dict, max_intra_event_time: float) -> Tensor:
        self.logger.info("Sampling kernel grid ...")
        size = (1, kernel_grids_sampler_conf["num_marks"], kernel_grids_sampler_conf["sample_size"] - 1)
        factor = kernel_grids_sampler_conf.get("max_value_factor", 1)
        kernel_grid = sample_kernel_grid(**kernel_grids_sampler_conf, max_value=max_intra_event_time * factor, size=size)

        kernel_grid = kernel_grid.to(self.device)
        self.logger.info(f"Sampled kernel grid with max inter-event time: {torch.max(kernel_grid).item():,.2f}")
        return kernel_grid

    def __get_max_intra_event_time(self, dataset: Dataset):
        self.logger.info("Calculating max intra-event time ...")
        if isinstance(dataset.data[0]["delta_times"], torch.Tensor):
            max_intra_event_time = torch.max(dataset.data["delta_times"])
        else:
            max_intra_event_time = max([item["delta_times"][-1] for item in dataset.data])
        self.logger.info(f"Max intra-event time: {max_intra_event_time.item():,.2f}")
        return max_intra_event_time

    def _generate_padding_mask(self, sequence_lengths, L):
        B, P = sequence_lengths.shape
        mask = torch.arange(L).expand(B, P, L).to(self.device) >= sequence_lengths.unsqueeze(-1)
        return mask

    def plot_kernel_grids_histograms(self):
        for label, res in self.res.items():
            num_marks = res["kernel_inference"]["sampled_kernel_grids"].shape[0]
            fig, axes = plt.subplots(num_marks, 2, figsize=(12, 6 * num_marks))
            axes = axes.flatten()
            for mark in range(num_marks):
                # Plot sampled kernel grids for each mark
                axes[mark].hist(res["kernel_inference"]["sampled_kernel_grids"][mark, :], bins=100, edgecolor="black", color="tab:blue")
                axes[mark].set_title(f"Sampled Kernel Grids for {label} - Mark {mark + 1}")
                axes[mark].set_xlabel("Time")
                axes[mark].set_ylabel("Frequency")

                # Plot target kernel grids for each mark
                axes[mark + 1].hist(
                    res["kernel_inference"]["target_kernel_grids"][mark, :], bins=100, edgecolor="black", color="tab:orange"
                )
                axes[mark + 1].set_title(f"Target Kernel Grids for {label} - Mark {mark + 1}")
                axes[mark + 1].set_xlabel("Time")
                axes[mark + 1].set_ylabel("Frequency")

            plt.tight_layout()
            plt.savefig(Path(self.config.evaluation_dir) / label / "kernel_grid_histogram.png")
            plt.close()

    def plot_kernels(self):
        for label, res in self.res.items():
            kernel_values = res["kernel_inference"]["predicted_kernel_values"]  # Shape: [M, L]
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
                base_intensity = res["kernel_inference"]["predicted_base_intensity"][i]
                target_base_intensity = res["kernel_inference"]["target_base_intensity"][i]
                kernel_total = kernel_values[i] + base_intensity
                axes[i].plot(res["kernel_inference"]["sampled_kernel_grids"][i], kernel_total, color="tab:blue", label="Predicted")
                axes[i].plot(
                    res["kernel_inference"]["target_kernel_grids"][i],
                    res["kernel_inference"]["target_kernel_values"][i] + target_base_intensity,
                    color="tab:orange",
                    label="Target",
                )
                axes[i].axhline(base_intensity, color="tab:blue", label="Predicted Base Intensity", linestyle="--")
                axes[i].axhline(target_base_intensity, color="tab:orange", label="Target Base Intensity", linestyle="--")
                axes[i].set_title(f"Kernel {i + 1}")
                axes[i].set_xlim(0, res["kernel_inference"]["target_kernel_grids"][i].max() * 1.1)
                axes[i].set_xlabel("Time")
                axes[i].set_ylabel("Value")
                axes[i].spines[["top", "right"]].set_visible(False)
            plt.legend()

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

    def plot_intensities(self):
        for label, res in self.res.items():
            dataset_eval_conf = self.config.get_by_label(label)
            plot_kwargs = dataset_eval_conf.plot_kwargs
            if plot_kwargs is None:
                plot_kwargs = {}
            predictions = res["one_step_ahead_predictions"]
            predicted_dtimes = predictions["predicted_dtimes"]
            target_event_times = predictions["target_event_times"]
            intensities_at_times = predictions["predicted_intensities"]
            target_intensities = predictions["target_intensities"]
            target_intensity_times = predictions["target_intensity_times"]
            sampled_intensities = predictions["predicted_sampled_intensities"]
            sampled_intensity_times = predictions["predicted_sampled_intensity_times"]
            num_paths = plot_kwargs.get("number_of_paths", 50)
            seq_idx = predictions["seq_idx"][:num_paths]
            num_times = intensities_at_times.shape[-2]
            num_marks = intensities_at_times.shape[-1]
            fig, axes = plt.subplots(
                num_paths,
                num_marks,
                figsize=(10 * num_marks, 6 * num_paths),
            )
            axes = axes.flatten()
            for i in np.argsort(seq_idx):  # batch
                for j in range(num_marks):  # marks
                    for k in range(num_times):  # times
                        axes[i * num_marks + j].plot(
                            sampled_intensity_times[i],
                            sampled_intensities[i, :, k, j],
                            label="Sampled Intensity",
                            color="tab:blue",
                        )
                        axes[i * num_marks + j].plot(
                            target_intensity_times[i],
                            target_intensities[i][j],
                            linestyle="--",
                            color="tab:orange",
                            label="Target Intensity",
                        )
                        axes[i * num_marks + j].eventplot(
                            target_event_times[i],
                            linelengths=0.15,
                            colors="tab:orange",
                            lineoffsets=-0.1,
                            label="True events",
                        )
                        axes[i * num_marks + j].eventplot(
                            predicted_dtimes[i, :, k],
                            linelengths=0.15,
                            colors="tab:blue",
                            lineoffsets=-0.4 * (k + 1),
                            label="Sampled events",
                        )
                        axes[i * num_marks + j].set_xlim(0, sampled_intensity_times[i].max())
                        axes[i * num_marks + j].set_title(f"Intensity for {label} - Mark {j + 1} - Sample {seq_idx[i]}")
                        axes[i * num_marks + j].legend()
            plt.tight_layout()
            label_dir = Path(self.config.evaluation_dir) / label
            label_dir.mkdir(parents=True, exist_ok=True)
            plot_path = label_dir / "intensities.png"
            plt.savefig(plot_path)
            plt.close()

    def calculate_metrics(self):
        metrics = {}
        for label, res in self.res.items():
            predictions = res["one_step_ahead_predictions"]
            predicted_event_times = predictions["predicted_event_times"]
            target_event_times = predictions["target_event_times"]

            target_base_intensity = res["kernel_inference"]["target_base_intensity"]
            predicted_base_intensity = res["kernel_inference"]["predicted_base_intensity"]
            rmse_base_intensity = np.sqrt(np.mean((predicted_base_intensity - target_base_intensity) ** 2))
            rmse_event_times = np.sqrt(np.mean((predicted_event_times - target_event_times[:, 1:]) ** 2))
            rmse_kernel_values = np.sqrt(
                np.mean(
                    (res["kernel_inference"]["predicted_kernel_values_on_target_grids"] - res["kernel_inference"]["target_kernel_values"])
                    ** 2
                )
            )
            self.logger.info(f"RMSE event times for {label}: {rmse_event_times}")
            self.logger.info(f"RMSE base intensity for {label}: {rmse_base_intensity}")
            self.logger.info(f"RMSE kernel values for {label}: {rmse_kernel_values}")
            metrics[label] = {
                "rmse_event_times": rmse_event_times.item(),
                "rmse_base_intensity": rmse_base_intensity.item(),
                "rmse_kernel_values": rmse_kernel_values.item(),
            }
        return metrics

    def __load_dataset(self, label: str, dataset_conf: dict):
        logger.info(f"Loading dataset: {label}")
        dataloader = DataLoaderFactory.create(dataset_conf.pop("name"), **dataset_conf)
        logger.info(f"Loaded dataset: {dataloader}")
        return dataloader

    def save_inference_results(self, metrics: dict):
        for label, res in self.res.items():
            dataset_eval_conf = self.config.get_by_label(label)
            if (self.config.evaluation_dir / label / "evaluation_results.json").exists() and dataset_eval_conf.load_if_exists:
                logger.info(f"Evaluation results for {label} already exist. Skipping ...")
                continue
            results = {
                key: {k: v.tolist() if isinstance(v, np.ndarray) else [vv.tolist() for vv in v] for k, v in value.items()}
                for key, value in res.items()
            }

            label_dir = Path(self.config.evaluation_dir) / label
            label_dir.mkdir(parents=True, exist_ok=True)
            path = label_dir / "evaluation_results.json"
            with open(path, "w") as f:
                json.dump(results, f)
            with open(label_dir / "metrics.json", "w") as f:
                json.dump(metrics[label], f)
            logger.info(f"Evaluation results for {label} saved to {path}")

    def save(self):
        logger.info("Calculating metrics ...")
        metrics = self.calculate_metrics()
        logger.info("Saving evaluation results to JSON ...")
        self.save_inference_results(metrics)
        logger.info("Plotting kernel grids histograms ...")
        self.plot_kernel_grids_histograms()
        logger.info("Plotting kernels ...")
        self.plot_kernels()
        logger.info("Plotting intensities ...")
        self.plot_intensities()


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
