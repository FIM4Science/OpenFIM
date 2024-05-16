import re
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
from tqdm.auto import tqdm
import torch

from fim.data.dataloaders import DataLoaderFactory
from ..models import ModelFactory
from .helper import export_list_of_dicts_to_jsonl, export_list_of_lists_to_csv


class Evaluation(object):
    """Base class for model evaluation.

    Args:
        device (str): on which the evaluation will take place
        output_path (str | Path): path to a folder where the results will be stored
        tokenizer_param (dict): parameters for crating a tokenizer
        dataset_param (dict): parameters for creating a dataset
        model_param (dict): parameters for creating a model.
    """

    def __init__(
        self,
        device: str,
        output_path: Union[Path, str],
        dataset_param: dict,
        model_param: dict,
        model_checkpoint_path: str,
    ) -> None:
        self.device = device
        self.output_path = Path(output_path)
        self.dataset_param = dataset_param
        self.model_param = model_param
        self.model_checkpoint_path = Path(model_checkpoint_path)

        self.predictions = []

        self.dataloader = DataLoaderFactory.create(**self.dataset_param)

        self.model = ModelFactory.create(**self.model_param, device_map=self.device, resume=True)
        self.model.eval()
        # load model checkpoint
        self.model.load_state_dict(torch.load(self.model_checkpoint_path)["model_state"])
        print("successfully loaded model checkpoint")

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


class QAevaluation(Evaluation):
    """Question and answering evaluaiton."""

    def __init__(
        self,
        device: str,
        output_path: Union[Path, str],
        tokenizer_param: dict,
        dataset_param: dict,
        model_param: dict,
        answer_pattern: str,
        max_new_tokens: int = 20,
        do_sample: bool = False,
    ) -> None:
        super().__init__(device, output_path, tokenizer_param, dataset_param, model_param)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.answer_pattern = re.compile(answer_pattern)
        self.predictions = []

    def evaluate(self, max_new_tokens: Optional[int] = None):
        max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        dataset = getattr(self.dataloader, self.dataset_param["split"] + "_it")
        for x in tqdm(dataset):
            generate_txt = self.model.generate(
                x["input_ids"].to("cuda"),
                x["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=self.do_sample,
                tokenizer=self.tokenizer,
            )
            for _id, p in zip(x["id"], generate_txt):
                match = self.answer_pattern.search(p)
                if match:
                    extracted_value = match.group(1)
                else:
                    extracted_value = "X"
                self.predictions.append([_id, extracted_value.upper(), p])

    def save(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        export_list_of_lists_to_csv(self.predictions, self.output_path / "predictions.csv")


EvaluationFactory.register("qa", QAevaluation)


class QAevaluationSupervised(QAevaluation):
    """Question and answering evaluaiton."""

    def __init__(
        self,
        device: str,
        output_path: Union[Path, str],
        tokenizer_param: dict,
        dataset_param: dict,
        model_param: dict,
        answer_pattern: str,
        max_new_tokens: int = 20,
        do_sample: bool = False,
    ) -> None:
        super().__init__(
            device, output_path, tokenizer_param, dataset_param, model_param, answer_pattern, max_new_tokens, do_sample
        )
        self.targers = []

    def evaluate(self, max_new_tokens: Optional[int] = None):
        max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        dataset = getattr(self.dataloader, self.dataset_param["split"] + "_it")
        for x in tqdm(dataset):
            generate_txt = self.model.generate(
                x["input_ids"].to("cuda"),
                x["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=self.do_sample,
                tokenizer=self.tokenizer,
            )
            for _id, p, t in zip(x["id"], generate_txt, x["answerKey"]):
                match = self.answer_pattern.search(p)
                if match:
                    extracted_value = match.group(1)
                else:
                    extracted_value = "X"
                self.targers.append({"id": _id, "answerKey": t})
                self.predictions.append([_id, extracted_value.upper(), p])
            break

    def save(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        export_list_of_dicts_to_jsonl(self.targers, self.output_path / "targets.jsonl")
        export_list_of_lists_to_csv(self.predictions, self.output_path / "predictions.csv")


EvaluationFactory.register("qa_supervised", QAevaluationSupervised)


class MathQAevaluation(Evaluation):
    """Math Question and Answering Evaluation."""

    def __init__(
        self,
        device: str,
        output_path: Union[Path, str],
        tokenizer_param: dict,
        dataset_param: dict,
        model_param: dict,
        answer_pattern: str,
        max_new_tokens: int = 20,
        do_sample: bool = False,
    ) -> None:
        super().__init__(device, output_path, tokenizer_param, dataset_param, model_param)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.answer_pattern = re.compile(answer_pattern)
        self.predictions = []

    def evaluate(self, max_new_tokens: Optional[int] = None):
        max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        dataset = getattr(self.dataloader, self.dataset_param["split"] + "_it")
        for x in tqdm(dataset):
            generate_txt = self.model.generate(
                x["input_ids"].to("cuda"),
                x["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=self.do_sample,
                tokenizer=self.tokenizer,
            )
            for p, target in zip(generate_txt, x["answerKey"]):
                numbers = re.findall(p, self.answer_pattern)
                if numbers:
                    extracted_value = float(numbers[0].replace("$", "").replace(",", ""))
                else:
                    extracted_value = float(np.inf)
                target = target.item()
                self.predictions.append(
                    {
                        "target": target,
                        "prediction": extracted_value,
                        "is_correct": float(target) == extracted_value,
                        "answer": p,
                    }
                )
            break

    def save(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        export_list_of_dicts_to_jsonl(self.predictions, self.output_path / "predictions.jsonl")


EvaluationFactory.register("math_qa", MathQAevaluation)


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
                for t, p, i, m in zip(
                    x["output_values"], prediction["predictions"], x["input_values"], x["mask_point_level"]
                )
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
        fig, axes = plt.subplots(num_plots, num_plots, figsize=(20, 20))
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
