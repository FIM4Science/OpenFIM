import itertools
import json
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import optree
import pandas as pd
import torch
import yaml
from matplotlib.pyplot import Figure
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from fim import project_path, results_path
from fim.models.blocks import AModel, ModelFactory


ModelInitializer = Callable[[], AModel]  # returns instance of AModel, s.t. it does not have to be in memory
ModelMap = Mapping[str, ModelInitializer]  # returns ModelInitializer instance based on some ID

DataLoaderInitializer = Callable[[], DataLoader]  # returns instance of DataLoader, s.t. it does not have to be kept in memory
DataLoaderMap = Mapping[str, DataLoaderInitializer]  # returns DataLoaderInitializer instance based on some ID

DisplayIdMap = Mapping[str, str]  # remaps ids to something displayable in figures

EvaluationTask = Callable[[AModel, DataLoader], Any]  # evaluates model on data provided by dataloader and return results
StepFunc = Callable[[AModel, Any, str], Any]


@dataclass(eq=False)
class ModelEvaluation:
    """
    Stores results of model evaluated on dataloader.
    """

    model_id: str
    dataloader_id: str
    task_id: Optional[str] = "default"
    results: Optional[Any] = None

    def __eq__(self, other) -> bool:
        """
        String identifiers are identifiers for the evaluation."
        """
        return (self.model_id == other.model_id) and (self.dataloader_id == other.dataloader_id) and (self.task_id == other.task_id)

    def __repr__(self) -> str:
        return self.dataloader_id.__str__() + "_" + self.model_id.__str__()

    def save(self, save_dir: Path) -> None:
        """
        Save ModelEvaluation as `.pickle` in a `save_dir` directory.
        """
        save_path: Path = (
            save_dir
            / self.model_id.__str__()
            / self.dataloader_id.__str__()
            / (self.task_id.__str__() + (self.model_id.__str__() + "_" + self.dataloader_id.__str__() + ".pickle"))
        )
        save_path: Path = prepare_path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_path(cls, load_path: Path):
        """
        Load ModelEvaluation from a `.pickle`.
        """
        load_path = prepare_path(load_path)

        with open(load_path, "rb") as f:
            result = pickle.load(f)

        return result


@dataclass
class IdentityMap(Mapping):
    "Default Map that returns key as item."

    reference_map: Mapping

    def __len__(self):
        return self.reference_map.__len__()

    def __getitem__(self, key):
        return key

    def __iter__(self):
        return iter([])


@dataclass
class EvaluationConfig:
    """
    Stores maps that can instantiate their models and dataloders and a global directory to save evaluations
    in and maps which can rename ids of models and dataloaders to something pretty, to display e.g. in Figures.
    """

    model_map: ModelMap
    dataloader_map: DataLoaderMap
    save_dir: Path
    model_display_id_map: Optional[DisplayIdMap] = None
    dataloader_display_id_map: Optional[DisplayIdMap] = None

    def __post_init__(self):
        self.save_dir = prepare_path(self.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

        if self.model_display_id_map is None:
            self.model_display_id_map = IdentityMap(self.model_map)

        if self.dataloader_map is None:
            self.dataloader_display_id_map = IdentityMap(self.dataloader_map)

    def get_model(self, model_evaluation: ModelEvaluation) -> AModel:
        return self.model_map[model_evaluation.model_id]()

    def get_dataloader(self, model_evaluation: ModelEvaluation) -> DataLoader:
        return self.dataloader_map[model_evaluation.dataloader_id]()


def load_evaluations(paths: str | list[str]) -> list[ModelEvaluation]:
    """
    Load (multiple) ModelEvaluations from a list of paths to `pickles`.
    Non-absolute paths are considered relative to `fim.project_path/evaluations/`.
    If (an entry of) paths is a directory, traverse subdirectories recursively to find all files ending with `.pickle`

    Args:
        paths (str | list[str]): Paths to (subdirectories containing) ModelEvaluations in `.pickle`

    Return:
        model_evaluations (list[ModelEvaluation]): ModelEvaluations loaded from input paths.
    """
    # unify input to list[Path]
    if isinstance(paths, str):
        paths: list[str] = [paths]

    paths: list[Path] = [Path(path) for path in paths]

    # make paths absolute
    paths: list[Path] = [prepare_path(Path(project_path) / "evaluations" / path) for path in paths]

    # expands subdirectories into paths to pickles
    paths: list[list[Path]] = [[path] if path.is_file() else list(path.rglob("*.pickle")) for path in paths]
    paths: list[Path] = itertools.chain.from_iterable(paths)

    # Load model evaluations
    model_evaluations = []
    for path in (pbar := tqdm(list(paths), total=len(list(paths)), leave=False, desc="Loading saved evaluations")):
        model_evaluations.append(ModelEvaluation.from_path(path))
    pbar.close()

    return model_evaluations


def save_evaluations(results: ModelEvaluation | list[ModelEvaluation], save_dir: Path) -> None:
    """
    Save (multiple) ModelEvaluations as `pickles`.
    Non-absolute paths are considered relative to `fim.project_path/evaluations/`.
    If (an entry of) paths is a directory, traverse subdirectories recursively to find all files ending with `.pickle`

    Args:
        paths (str | list[str]): Paths to (subdirectories containing) ModelEvaluations in `.pickle`

    Return:
        model_evaluations (list[ModelEvaluation]): ModelEvaluations loaded from input paths.
    """
    save_dir = prepare_path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    if isinstance(results, ModelEvaluation):
        results: list[ModelEvaluation] = [results]

    for result in results:
        result.save(save_dir)


def _get_model_class(checkpoint_path: Path):
    """
    Returns the model class fromresult.task_id,  a checkpoint.
    Might improve this somehow.
    """
    with open(checkpoint_path.parent.parent / "train_parameters.yaml", "r") as f:
        config = yaml.load(f, yaml.BaseLoader)

    model_type = config["model"]["model_type"]
    model_class = ModelFactory.model_types.get(model_type)

    return model_class


def model_init_from_checkpoint(checkpoint_dir: Path | str, checkpoint_name: Optional[str] = "best-model") -> ModelInitializer:
    """
    Returns ModelInitializer that initializes model from a specified checkpoint.

    Args:
        checkpoint_dir (Path | str): Path to the `.../checkpoint` directory of a trained model.
        checkpoint_name (Optional[str]): Specifies checkpoint to load.

    Returns:
        model_init (ModelInitializer): When called, initializes model from checkpoint.
    """

    def model_init(checkpoint_dir, checkpoint_name):
        if isinstance(checkpoint_dir, str):
            checkpoint_dir: Path = Path(checkpoint_dir)

        if not checkpoint_dir.is_absolute():
            checkpoint_dir: Path = Path(results_path) / checkpoint_dir / "checkpoints"

        checkpoint_path: Path = checkpoint_dir / checkpoint_name

        model_class = _get_model_class(checkpoint_path)

        return model_class.load_model(model_path=checkpoint_path)

    return partial(model_init, checkpoint_dir=checkpoint_dir, checkpoint_name=checkpoint_name)


def model_init_from_instance(model: AModel) -> ModelInitializer:
    """
    Returns a ModelInitializer that just returns the given model. In case model should not be reloaded.
    """
    return lambda: model


def get_model_init(config: Any) -> ModelInitializer:
    """
    Common interface to get ModelInitializer by the type of config passed.

    Args:
        config (Any): type of config specifies inititalizer:
            if isinstance(config, dict): config = {"checkpoint_dir":..., "checkpoint_name"}, init from checkpoint
            if isinstance(config, str): config = checkpoint_dir, init from checkpoint
            if isinstance(config, AModel): config = AModel instance, init from instance

    Returns:
        model_init (ModelInitializer)
    """
    if isinstance(config, dict):
        return model_init_from_checkpoint(**config)

    elif isinstance(config, str) or isinstance(config, Path):
        config = {"checkpoint_dir": config}
        return model_init_from_checkpoint(**config)

    elif isinstance(config, AModel):
        return model_init_from_instance(config)

    else:
        raise ValueError(f"Model initialization configuration from {type(config)} not implemented.")


def model_map_from_dict(model_init_configs: dict[str, Any]) -> ModelMap:
    """
    Transform dict of ModelInitializer configurations into a dict with ModelInitializers as values.

    Args:
        model_init_configs (dict[str, Any]):
            {mode_a_id: model_a_init_config, model_b_id:...}

    Returns:
        dict_model_map (ModelMap, dict[str, ModelInitializer]):
            {model_a_id: model_a_init, model_b_id: ....}
    """
    return {key: get_model_init(value) for key, value in model_init_configs.items()}


def dataloader_init_from_instance(dataloader: DataLoader) -> DataLoaderInitializer:
    """
    Returns a DataLoaderInitializer that just returns the given dataloader. In case model should not be reloaded.
    """
    return lambda: dataloader


def get_dataloader_init(config: Any) -> DataLoaderInitializer:
    """
    Common interface to get ModelInitializer by the type of config passed.

    Args:
        config (Any): type of config specifies inititalizer:
            if isinstance(config, DataLoader): config = DataLoader instance, init from instance

    Returns:
        dataloader_init (ModelInitializer)
    """
    if isinstance(config, DataLoader):
        return model_init_from_instance(config)

    else:
        raise ValueError(f"Dataloader initialization configuration from {type(config)} not implemented.")


def dataloader_map_from_dict(dataloader_dict: dict[str, Any]) -> DataLoaderMap:
    """
    Transform dict of DataLoaderInitializer configurations into a dict with DataLoaderInitializer as values.

    Args:
        dataloader_init_configs (dict[str, Any]):
            {dataloder_a_id: mdataloder_a_init_config, mdataloder_b_id:...}

    Returns:
        dict_dataloder_map (DataLoaderMap, dict[str, DataLoaderInitializer]):
            {dataloader_a_id: dataloader_a_init, dataloader_b_id: ....}
    """
    return {key: get_dataloader_init(value) for key, value in dataloader_dict.items()}


def get_maps_from_dicts(
    models: dict, dataloaders: dict, models_display_id: Optional[dict] = {}, dataloaders_display_id: Optional[dict] = {}
) -> tuple[ModelMap, DataLoaderMap, DisplayIdMap, DisplayIdMap]:
    """
    Get ModelMap, DataLoaderMap and DisplayIdMaps for EvaluationConfig from dicts, by appyling
    `model_map_from_dict`, `dataloader_map_from_dict` and updating the passed model_display_ids with default ids.
    """
    model_map: ModelMap = model_map_from_dict(models)
    dataloader_map: DataLoaderMap = dataloader_map_from_dict(dataloaders)

    models_display_id_map: DisplayIdMap = {key: key for key in models.keys()}
    models_display_id_map.update(models_display_id)

    dataloaders_display_id_map: DisplayIdMap = {key: key for key in dataloaders.keys()}
    dataloaders_display_id_map.update(dataloaders_display_id)

    return model_map, dataloader_map, models_display_id_map, dataloaders_display_id_map


def run_model_evaluations(
    evaluations: list[ModelEvaluation],
    model_map: ModelMap,
    dataloader_map: DataLoaderMap,
    EvaluationTask: EvaluationTask,
    task_id: Optional[str] = "default",
) -> list[ModelEvaluation]:
    """
    Evaluate ModelEvaluations with a evaluation function.

    Args:
        evaluations (list[ModelEvaluation]): List of evaluations to run, specified by their model_id and dataloader_id
        model_map (ModelMap): To instantiate models from evaluations.
        dataloader_map (DataLoaderMap): To instantiate dataloaders from evaluations.
        EvaluationTask (EvaluationTask): Specifies how to evaluate model on dataloader.

    Return:
        evaluations (list[ModelEvaluation]): Input evaluations with results from EvaluationTask.
    """
    evaluations_with_results: list[ModelEvaluation] = []

    for evaluation in (pbar := tqdm(evaluations, total=len(evaluations), leave=False)):
        pbar.set_description(
            f"Task: {str(evaluation.task_id)}, Data: {str(evaluation.dataloader_id)}, Model: {str(evaluation.model_id)}. Overall progress"
        )

        model: AModel = model_map[evaluation.model_id]().to(torch.float)
        dataloader: DataLoader = dataloader_map[evaluation.dataloader_id]()

        evaluation.results = EvaluationTask(model, dataloader)
        evaluation.task_id = task_id
        evaluations_with_results.append(evaluation)

    return evaluations_with_results


@torch.no_grad
def evaluate_with_step_function(model: AModel, dataloader: DataLoader, step_func: StepFunc) -> Any:
    """
    Evaluate a model and dataloader with a step function. If step_func is partialed, this is an instance of `EvaluationTask`.

    Args:
        model (AModel): Model instance to evaluate.
        dataloader (DataLoader): DataLoader instance to evaluate model with.
        step_func (StepFunc): Specifies how model is evaluated on one batch from dataloader.

    Returns:
        results (Any): Tree of returns from step_func, concatenated at first dimension.
    """
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    results = []

    for batch in tqdm(dataloader, desc="Evaluation progress", leave=False):
        batch = optree.tree_map(lambda x: x.to(device), batch)
        result: Any = step_func(model, batch, device)
        results.append(result)

    results = optree.tree_map(lambda *x: torch.concatenate(x, dim=0).to("cpu"), *results, namespace="fimsde")
    model.to("cpu")

    return results


def prepare_path(path: Path) -> Path:
    """
    Make path relative to fim.project_path, if path is not absolute path.
    """
    if not path.is_absolute():
        path = Path(project_path) / path

    return path


def save_table(
    df: DataFrame, dir: Path, filename: str, to_latex_kwargs: Optional[dict] = {}, to_markdown_kwargs: Optional[dict] = {}
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


def save_fig(fig: Figure, dir: Path, filename: str) -> None:
    """
    Save figure as pdf.

    Args:
        fig (Figure): Figure to save.
        dir / filename (Path): Specifies path figure is saved at.
    """
    file_path: Path = dir / (filename + ".pdf")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(file_path, dpi=300, bbox_inches="tight")


def dataloader_get_all_elements(dataloader: DataLoader) -> Any:
    """
    Concatenate all batches from dataloader along first dimension.
    """
    all_batches = []

    for batch in iter(dataloader):
        all_batches.append(batch)

    return optree.tree_map(lambda *x: torch.concatenate(x, axis=0), *all_batches)


def get_data_from_model_evaluation(model_evaluation: ModelEvaluation, evaluation_config: EvaluationConfig):
    """
    Load all data from dataloader with model_evaluation.dataloader_id.
    """
    dataloader_id = model_evaluation.dataloader_id
    dataloader = evaluation_config.dataloader_map[dataloader_id]()

    return dataloader_get_all_elements(dataloader)


def find_indices_of_dim(dim: int, dim_mask: Tensor, indices_count: int, random_indices: bool):
    """
    From dimension mask of SDE data, find batch elements of specified dimension and return their indices.

    Args:
        dim (int): Dimension to find.
        dim_mask (Tensor): Mask indicating dimensionality. Shape: [B, G, D]
        indices_count (int): Maximum number of indices to return.
        random_indices (bool): If True, returns incides_count random indices of elements of dim.

    Returns:
        indices (Tensor[int]): Indices of elements of dimension dim.
    """
    assert dim_mask.ndim == 3

    dim_mask = dim_mask[:, 0, :]
    dim_per_element = dim_mask.bool().sum(axis=-1, dtype=torch.int)  # [B]

    is_of_dim = dim_per_element == dim

    all_indices = torch.nonzero(is_of_dim.float()).reshape(-1)

    if random_indices is True:
        perm = torch.randperm(all_indices.shape[0])
        all_indices = all_indices[perm]

    return all_indices[:indices_count]


# -------------  Metrics / Tables ------------------
def get_mse(estimation: Tensor, target: Tensor):
    """
    Return MSE between estimation and target per batch element.

    Args: estimation, target (Tensor): Shape: [B, G, D]
    Returns: mse (Tensor): Shape: [B]
    """
    assert estimation.ndim == target.ndim == 3

    se = (estimation - target) ** 2
    mse = torch.mean(se, dim=(-1, -2))

    return mse


def get_mae(estimation: Tensor, target: Tensor):
    """
    Return MAE between estimation and target per batch element.

    Args: estimation, target (Tensor): Shape: [B, G, D]
    Returns: mae (Tensor): Shape: [B]
    """
    assert estimation.ndim == target.ndim == 3

    ae = torch.abs(estimation - target)
    mae = torch.mean(ae, dim=(-1, -2))

    return mae


def get_norm_mse(estimation: Tensor, target: Tensor):
    """
    Return normalized MSE (normalized by target) between estimation and target per batch element.

    Args: estimation, target (Tensor): Shape: [B, G, D]
    Returns: norm_mse (Tensor): Shape: [B]
    """
    assert estimation.ndim == target.ndim == 3

    se = torch.mean((estimation - target) ** 2, dim=-1)  # [B, G]
    normalization = torch.mean(target**2, dim=-1)  # [B, G]

    norm_mse = torch.mean(se / (normalization + 1), dim=(-1))  # [B]

    return norm_mse


def get_norm_mae(estimation: Tensor, target: Tensor):
    """
    Return normalized MAE (normalized by target) between estimation and target per batch element.

    Args: estimation, target (Tensor): Shape: [B, G, D]
    Returns: norm_mae (Tensor): Shape: [B]
    """
    assert estimation.ndim == target.ndim == 3

    ae = torch.mean(torch.abs(estimation - target), dim=-1)  # [B, G]
    normalization = torch.mean(torch.abs(target), dim=-1)  # [B, G]

    norm_mae = torch.mean(ae / (normalization + 1), dim=(-1))  # [B]

    return norm_mae


def get_regression_metrics(estimation: Tensor, target: Tensor, metrics: list[str]) -> dict[str, Tensor]:
    """
    Compute specified metrics between estimation and target tensors.

    Args:
        estimation, target (Tensor): Shape: [B, G, D]
        metrics (list[str]): List of metrics to compute.

    Returns:
        metrics_per_element (dict[str, Tensor]): Each value of shape [B]
    """

    if isinstance(metrics, str):
        metrics: list[str] = [metrics]

    metrics_per_element: dict = {}

    if "mse" in metrics:
        metrics_per_element.update({"mse": get_mse(estimation, target)})

    if "mae" in metrics:
        metrics_per_element.update({"mae": get_mae(estimation, target)})

    if "norm_mse" in metrics:
        metrics_per_element.update({"norm_mse": get_norm_mse(estimation, target)})

    if "norm_mae" in metrics:
        metrics_per_element.update({"norm_mae": get_norm_mae(estimation, target)})

    for metric_key, metric_value in metrics_per_element.items():
        assert metric_value.ndim == 1, f"Metric `{metric_key}` got {metric_value.ndim} dimensions."

    return metrics_per_element


def save_df_per_metric(metrics_df: pd.DataFrame, save_dir: Path) -> None:
    """
    Save one df per metric. Each df has dataloaders as columns and models as rows.

    Args:
        metrics_df (pd.DataFrame): Columns: [dataloader_id, model_id, metrics...]
    """
    metrics = [col for col in metrics_df.columns if col not in ["dataloader_id", "model_id"]]

    for metric in (pbar := tqdm(metrics, total=len(metrics), leave=False)):
        pbar.set_description(f"Saving table for metric: {metric}")

        metric_df = metrics_df[["dataloader_id", "model_id", metric]]
        metric_df = metric_df.set_index(["model_id", "dataloader_id"])[metric].unstack()  # dataloaders as columns, models as rows
        save_table(metric_df, save_dir, metric)

    pbar.close()


def save_df_per_model_id(metrics_df: pd.DataFrame, save_dir: Path) -> None:
    """
    Save one df per model. Each df has dataloaders as rows and metrics as columns.

    Args:
        metrics_df (pd.DataFrame): Columns: [dataloader_id, model_id, metrics...]
    """
    for group in (pbar := tqdm(metrics_df.groupby("model_id"), leave=False)):
        model_id: str = group[0]
        pbar.set_description(f"Saving table for model: {model_id}")

        model_df = group[1].drop("model_id", axis=1)
        model_df = model_df.set_index("dataloader_id")

        model_id = model_id.replace(" ", "_").replace(":", "_").replace(".", "").replace(",", "")
        save_table(model_df, save_dir, model_id)

    pbar.close()


def save_df_per_dataloader_id(metrics_df: pd.DataFrame, save_dir: Path) -> None:
    """
    Save one df per dataloader. Each df has dataloaders as rows and metrics as columns.

    Args:
        metrics_df (pd.DataFrame): Columns: [dataloader_id, model_id, metrics...]
    """
    for group in (pbar := tqdm(metrics_df.groupby("dataloader_id"), leave=False)):
        dataloader_id: str = group[0]
        pbar.set_description(f"Saving table for dataloader: {dataloader_id}")

        dataloader_df = group[1].drop("dataloader_id", axis=1)
        dataloader_df = dataloader_df.set_index("model_id")

        dataloader_id = dataloader_id.replace(" ", "_").replace(":", "_").replace(".", "").replace(",", "")
        save_table(dataloader_df, save_dir, dataloader_id)

    pbar.close()


# Custom encoder to convert NumPy arrays to lists
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)


def preprocess_system_data(data: dict, apply_diffusion_sqrt: bool) -> dict:
    """
    Load observations, locations, real drift and diffusion from data of json.
    Reshape into single batch and single paths, processable by model.

    Args:
        data (dict): keys: name, observations, observations_time, locations_points, real_drift_at_locations, real_diffusion_at_locations
        apply_diffusion_sqrt (bool): If true, apply sqrt to loaded diffusion, to covert it to our convention of diffusion

    Returns:
        data (dict): keys: obs_times, obs_values, locations, drift_at_locations, diffusion_at_locations
    """
    # extract arrays
    data.pop("name")
    data = {k: np.array(v) for k, v in data.items()}

    obs_times = data.get("observations_time")[:, None]  # [T, 1]
    obs_values = data.get("observations")  # [T, D]
    locations = data.get("locations_points")  # [L, D]
    drift_at_locations = data.get("real_drift_at_locations")  # [L, D]

    diffusion_at_locations = data.get("real_diffusion_at_locations")  # [L, D]
    if apply_diffusion_sqrt is True:
        diffusion_at_locations = np.sqrt(diffusion_at_locations)

    # view single time series as one path
    obs_times = obs_times[None]  # [1, T, 1]
    obs_values = obs_values[None]  # [1, T, D]

    data = {
        "obs_times": obs_times,
        "obs_values": obs_values,
        "locations": locations,
        "drift_at_locations": drift_at_locations,
        "diffusion_at_locations": diffusion_at_locations,
    }

    # add batch dimension for convenience
    data = optree.tree_map(lambda x: x[None], data)

    return data


def preprocess_gp_results(data: dict) -> dict:
    """
    Process results from GP model (Opper)
    Reshape into single batch and single paths, for easier processing by evaluation script.

    Args:
        data (dict): keys: name, estimated_drift_at_locations, estimated_diffusion_at_locations, MSE drift, MSE diffusion

    Returns:
        data (dict): keys: estimated_drift_at_locations, estimated_diffusion at locations
    """
    # extract arrays
    data.pop("name")
    data = {k: np.array(v) for k, v in data.items()}

    estimated_drift_at_locations = data.get("estimated_drift_at_locations")  # [L, D]
    estimated_diffusion_at_locations = np.sqrt(np.clip(data.get("estimated_diffusion_at_locations"), a_min=0, a_max=np.inf))  # [L, D]

    results = {
        "estimated_drift_at_locations": estimated_drift_at_locations[None],  # [1, L, D]
        "estimated_diffusion_at_locations": estimated_diffusion_at_locations[None],  # [1, L, D]
    }
    # IMPORTANT, GP estimates diffusion value of function under the sqrt -> need to sqrt it to our convention

    return results
