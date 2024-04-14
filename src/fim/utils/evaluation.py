import re
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
from tqdm.auto import tqdm

from ..data import DataLoaderFactory, load_tokenizer
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

    def __init__(self, device: str, output_path: Union[Path, str], tokenizer_param: dict, dataset_param: dict, model_param: dict) -> None:
        self.device = device
        self.output_path = Path(output_path)
        self.tokenizer_param = tokenizer_param
        self.dataset_param = dataset_param
        self.model_param = model_param

        self.tokenizer = load_tokenizer(**self.tokenizer_param)
        self.dataloader = DataLoaderFactory.create(**self.dataset_param, tokenizer=self.tokenizer)
        num_added_tokens = len(self.tokenizer.added_tokens_decoder)
        self.model = ModelFactory.create(
            **self.model_param, pad_token_id=self.tokenizer.pad_token_id, num_added_tokens=num_added_tokens, device_map=self.device
        )

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
        super().__init__(device, output_path, tokenizer_param, dataset_param, model_param, answer_pattern, max_new_tokens, do_sample)
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
                    {"target": target, "prediction": extracted_value, "is_correct": float(target) == extracted_value, "answer": p}
                )
            break

    def save(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        export_list_of_dicts_to_jsonl(self.predictions, self.output_path / "predictions.jsonl")


EvaluationFactory.register("math_qa", MathQAevaluation)
