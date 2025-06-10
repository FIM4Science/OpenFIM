import json
import pprint
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class MetricEvaluation:
    model_id: str
    model_json: str

    data_id: Any
    data_paths_json: str
    data_vector_fields_json: str

    metric_id: str
    metric_value: float

    def __eq__(self, other: object) -> bool:
        """
        Identify objects by their ids (model, data, metric). Paths to jsons and final metric value are ignored.
        """
        return (self.model_id == other.model_id) and (self.data_id == other.data_id) and (self.metric_id == other.metric_id)

    def __hash__(self):
        return hash((self.model_id, self.data_id, self.metric_id))

    def __str__(self) -> str:
        return pprint.pformat(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_json": str(self.model_json),
            "data_id": self.data_id,
            "data_paths_json": str(self.data_paths_json),
            "data_vector_fields_json": str(self.data_vector_fields_json),
            "metric_id": self.metric_id,
            "metric_value": self.metric_value,
        }

    @classmethod
    def from_dict(cls, src: dict):
        return cls(
            model_id=tuple(src.get("model_id")) if isinstance(src.get("model_id"), list) else src.get("model_id"),
            model_json=Path(src.get("model_json")),
            data_id=tuple(src.get("data_id")) if isinstance(src.get("data_id"), list) else src.get("data_id"),
            data_paths_json=Path(src.get("data_paths_json")),
            data_vector_fields_json=Path(src.get("data_vector_fields_json")),
            metric_id=src.get("metric_id"),
            metric_value=src.get("metric_value"),
        )

    def to_json(self, save_path: Path) -> None:
        evaluation_dict: dict = self.to_dict()

        with open(save_path, "w") as f:
            json.dump(evaluation_dict, f)

    @classmethod
    def from_json(cls, json_path: Path):
        with open(json_path, "r") as f:
            model_evaluation_data: dict = json.load(f)

        return cls.from_dict(model_evaluation_data)


def load_metric_evaluations_from_dirs(dir_paths: list[Path]) -> list[MetricEvaluation]:
    """
    Load MetricEvaluations from jsons contained in multiple (sub)directories.
    Raise ValueError, if multiple MetricEvaluations have same ids (model, data, metric).

    Args:
        dir_paths (list[Path]): Paths of directories containing ModelEvaluations stored as jsons.

    Returns:
        loaded_evaluation_list (list[MetricEvaluation]): Joint list of all found ModelEvaluations.
    """
    if not isinstance(dir_paths, list):
        dir_paths: list[Path] = [dir_paths]

    loaded_evaluation_list: list[list[MetricEvaluation]] = []

    for dir_path in dir_paths:
        metric_evaluations_jsons_paths = list(dir_path.glob("*.json"))
        loaded_evaluation_list = loaded_evaluation_list + [MetricEvaluation.from_json(path) for path in metric_evaluations_jsons_paths]

    # check for duplicates
    loaded_evaluation_set = set(loaded_evaluation_list)
    contains_duplicates: bool = len(loaded_evaluation_set) != len(loaded_evaluation_list)

    if contains_duplicates is True:
        # find duplicates to report
        duplicates = []

        for evaluation in loaded_evaluation_set:
            all_copies_of_evaluation = [
                metric_evaluation for metric_evaluation in loaded_evaluation_list if metric_evaluation == evaluation
            ]
            if len(all_copies_of_evaluation) > 1:
                duplicates = duplicates + all_copies_of_evaluation

        raise ValueError("Loaded same evaluations multiple times:\n" + ("\n".join(duplicates)))

    else:
        return loaded_evaluation_list


def nans_to_stars(series: pd.Series) -> str:
    """
    Pandas groupby aggregation function counting True values and depict them as stars.
    (This notation is used in the MMD tables).
    """
    nans_count = int(series.sum().item())

    if nans_count > 0:
        return "*" * nans_count

    else:
        return " "


def mean_plus_std_agg(series: pd.Series, precision: int) -> str:
    r"""
    Pandas groupby aggregation function computing mean and std and depict them like `mean $\pm$ std` up to some precision digit.
    """
    series = series.dropna()

    if len(series) > 1:
        mean = series.mean()
        std = series.std()
        return str(round(mean, precision)) + r" $\pm$ " + str(round(std, precision))

    elif len(series) == 1:
        mean = series.mean()

        return str(round(mean, precision)) + r" $\pm$ 0.0"

    else:
        return "-"


def mean_bracket_std_agg(series: pd.Series, precision: int) -> str:
    """
    Pandas groupby aggregation function computing mean and std and depict them with mean(std), where std is the first significant digit.

    precision: default precision if std is 0

    """
    series = series.dropna()

    if len(series) > 1:
        mean = series.mean()
        std = series.std()

        # round to one significant digit
        exponent = int(np.floor(np.log10(abs(std)))) if std != 0 else 0
        mean_rounded = round(mean, -exponent)

        # get the error digit
        std_rounded = round(std, -exponent)
        error_digit = int(std_rounded / (10**exponent))

        return f"${mean_rounded}({error_digit})$"

    elif len(series) == 1:
        mean = series.mean()
        mean_rounded = round(mean, precision)

        return f"${mean_rounded}(0)$"

    else:
        return "-"


def save_table(
    df: pd.DataFrame, dir: Path, filename: str, to_latex_kwargs: Optional[dict] = {}, to_markdown_kwargs: Optional[dict] = {}
) -> None:
    """
    Save dataframe as latex and markdown tables.

    Args:
        df (DataFrame): DataFrame to save as table.
        dir / filename (Path): Specifies path tables are saved at.
        ...kwargs (dict): kwargs passed to associated DataFrame method.
    """

    file_path: Path = dir / (filename + ".txt")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        latex_df: str = df.to_latex(escape=False, **to_latex_kwargs)
        latex_df = latex_df.replace("multirow[t]", "multirow[c]")  # should do it automatically, but somehow does not

        f.write(latex_df)
        f.write("\n\n")
        f.write(df.to_markdown(**to_markdown_kwargs))
