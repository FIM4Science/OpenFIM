# coding: utf-8
from torch import Tensor
import copy
import csv
import itertools
import json
from collections import OrderedDict
from dataclasses import dataclass
from functools import reduce
from importlib import import_module
from logging import Logger
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
import yaml
from scipy import linalg as la
from torch.optim.lr_scheduler import LRScheduler

T = TypeVar("T", str, bytes)

def check_model_devices(x):
    return x.parameters().__next__().device

def nametuple_to_device(named_tuple, device):
    return named_tuple._replace(**{
        key: (value.to(device) if isinstance(value, torch.Tensor) else value)
        for key, value in named_tuple._asdict().items()
    })

@dataclass
class GenericConfig:
    def __init__(self, data_dict: dict):
        for key, value in data_dict.items():
            if isinstance(value, dict):
                setattr(self, key, GenericConfig(value))
            elif isinstance(value, tuple):
                values = []
                for v in value:
                    values.append(GenericConfig(v) if isinstance(v, dict) else v)
                setattr(self, key, values)
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict:
        data_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, GenericConfig):
                data_dict[key] = value.to_dict()
            elif isinstance(value, list):
                values = []
                for v in value:
                    if isinstance(v, GenericConfig):
                        values.append(v.to_dict())
                    else:
                        values.append(v)
                data_dict[key] = tuple(values)
            else:
                data_dict[key] = value
        return data_dict

    def __str__(self) -> str:
        return str(self.__dict__)


def create_class_instance(class_full_path: str, kwargs, *args):
    """Create an instance of a given class.

    :param module_name: where the class is located
    :param kwargs: arguments needed for the class constructor
    :returns: instance of 'class_name'

    """
    module_name, class_name = class_full_path.rsplit(".", 1)
    module = import_module(module_name)
    clazz = getattr(module, class_name)
    if kwargs is None:
        instance = clazz(*args)
    else:
        instance = clazz(*args, **kwargs)

    return instance


def create_instance(name, params, *args):
    """Creates an instance of class given configuration.

    :param name: of the module we want to create
    :param params: dictionary containing information how to instantiate the class
    :returns: instance of a class
    :rtype:

    """
    i_params = params[name]
    if isinstance(i_params, list):
        instance = [create_class_instance(p["module"], p["name"], p["args"], *args) for p in i_params]
    else:
        instance = create_class_instance(i_params["module"], i_params["name"], i_params["args"], *args)
    return instance


def create_nonlinearity(name):
    """
    Returns instance of non-linearity class (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)
    instance = clazz()

    return instance


def get_class_nonlinearity(name):
    """
    Returns non-linearity class (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)

    return clazz


def create_cost_function(name, *args):
    """
    Returns instance of cost functions (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)
    instance = clazz(*args)

    return instance


def save_dict_to_file(obj: dict, path: str):
    """Save dictionary to a file

    :param obj: dict that is stored,
    :param path: to the location where the dictionary is stored.

    """
    with open(path, "w") as f:
        json.dump(obj, f)


def to_one_hot(labels, num_classes):
    """
    Convert tensor of labels to one hot encoding of the labels.
    :param labels: to be encoded
    :param num_classes:
    :return:
    """
    shape = labels.size()
    shape = shape + (num_classes,)
    one_hot = torch.zeros(shape, dtype=torch.float, device=labels.device)
    dim = 1 if len(shape) == 2 else 2
    one_hot.scatter_(dim, labels.unsqueeze(-1), 1)
    return one_hot


def convert_tuples_2_list(arg):
    for key, value in arg.items():
        if isinstance(value, dict):
            convert_tuples_2_list(value)
        else:
            if isinstance(value, tuple):
                arg[key] = list(value)

    return arg


def unpack_cv_parameters(params, prefix=None):
    cv_params = []
    for key, value in params.items():
        if isinstance(value, dict):
            if prefix is None:
                prefix = key
            else:
                prefix = ".".join([prefix, key])
            param_pool = unpack_cv_parameters(value, prefix)
            if "." in prefix:
                prefix = prefix.rsplit(".", 1)[0]
            else:
                prefix = None

            if len(param_pool) > 0:
                cv_params.extend(param_pool)
        elif isinstance(value, tuple) and len(value) != 0 and isinstance(value[0], dict):
            for ix, v in enumerate(value):
                if isinstance(v, dict):
                    if prefix is None:
                        prefix = key
                    else:
                        prefix = ".".join([prefix, key + f"#{ix}"])
                    param_pool = unpack_cv_parameters(v, prefix)
                    if "." in prefix:
                        prefix = prefix.rsplit(".", 1)[0]
                    else:
                        prefix = None
                    if len(param_pool) > 0:
                        cv_params.extend(param_pool)
        elif isinstance(value, list):
            if prefix is None:
                prefix = key
            else:
                key = ".".join([prefix, key])
            cv_params.append([(key, v) for v in value])
    return cv_params


def dict_set_nested(d, keys, value):
    node = d
    key_count = len(keys)
    key_idx = 0

    for key in keys:
        key_idx += 1

        if key_idx == key_count:
            node[key] = value
            return d
        else:
            if "#" in key:
                key, _id = key.split("#")
                if key not in node:
                    node[key] = {}
                    node = node[key][int(_id)]
                else:
                    node = node[key][int(_id)]
            else:
                if key not in node:
                    node[key] = {}
                    node = node[key]
                else:
                    node = node[key]


def expand_params(params: dict):
    """
    Expand the hyperparamers for grid search

    :param params:
    :return:
    """
    cv_params = []
    param_pool = unpack_cv_parameters(params)

    for i in list(itertools.product(*param_pool)):
        d = copy.deepcopy(params)
        name = d["experiment"]["name"]
        for j in i:
            dict_set_nested(d, j[0].split("."), j[1])
            name += "-" + j[0] + "-" + str(j[1])
            d["experiment"]["name"] = name.replace(".", "-")
        cv_params.append(GenericConfig(d))

    return cv_params


def get_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def get_device(params: dict, rank: int = 0, logger: Logger = None) -> torch.device:
    """

    :param params:
    :param logger:
    :return: returns the device
    """
    gpus = params.get("gpus", [])
    if len(gpus) > 0:
        if not torch.cuda.is_available():
            if logger is not None:
                logger.warning("No GPU's available. Using CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(gpus[rank]))
    else:
        device = torch.device("cpu")
    return device


def gauss_legender_points(N=30):
    """Returns the quadratures_nodes anb weights of the Gaussian-Lenegendre quadrature"""
    beta = np.array([(n + 1.0) / np.sqrt((2.0 * n + 1.0) * (2.0 * n + 3.0)) for n in range(N - 1)], dtype=np.float32)
    M = np.diag(beta, -1) + np.diag(beta, 1)
    nodes, V = la.eigh(M, overwrite_a=True, overwrite_b=True)
    weight = 2 * V[0, :] ** 2
    return nodes, weight


def quadratures(f, a=-1, b=1, n=30):
    """
    Performing Legendre-Gauss quadrature integral approximation.

    :param f:
    :param a:
    :param b:
    :param n:
    :return:
    """
    nodes, weights = gauss_legender_points(n)
    w = torch.tensor(weights.reshape(1, 1, -1))
    nodes = torch.tensor(nodes.reshape(1, 1, -1))

    scale = (b - a) / 2.0

    x = scale * nodes + (b + a) / 2.0
    y = w * f(x)
    y = torch.sum(scale * y, dim=-1)
    return y.type(dtype=torch.float)


def greedy_sample_categorical(prob, one_hot=True):
    """
    Sample greedily from categorical distribution
    prob (Class probabilities): [B, n_classes]
    returns greedy samples, as one hot vectors if one_hot
    """
    dtype = prob.dtype
    n_classes = prob.shape[-1]
    z = torch.argmax(prob, dim=-1)
    if one_hot:
        z = torch.nn.functional.one_hot(z, num_classes=n_classes).type(dtype)

    return z


def gumbel_sample(shape, device, epsilon=1e-20):
    """
    Sample Gumbel(0,1)
    """
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + epsilon) + epsilon)


def gumbel_softmax_sample(pi, tau, device, epsilon=1e-12):
    """
    Sample Gumbel-softmax
    """
    dtype = pi.dtype
    y = torch.log(pi + epsilon) + gumbel_sample(pi.size(), device)
    return torch.nn.functional.softmax(y / tau, dim=-1).type(dtype)


def gumbel_softmax(pi, tau, device, hard=True):
    """
    Gumbel-Softmax distribution.
    Implementation from https://github.com/ericjang/gumbel-softmax.
    pi: [B, ..., n_classes] class probs of categorical z
    tau: temperature
    Returns [B, ..., n_classes] as a one-hot vector
    """
    y = gumbel_softmax_sample(pi, tau, device)
    if hard:
        shape = y.size()
        _, ind = y.max(dim=-1)  # [B, ...]
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y
    else:
        return y


def gumbel_softmax_argmax(pi, tau, device):
    """
    Gumbel-Softmax distribution.
    pi: [B, ..., n_classes] class probs of categorical z
    tau: temperature
    Returns [B, ..., n_classes] as a one-hot vector
            [B, ..., 1] (argmax over classes)
    """
    y = gumbel_softmax_sample(pi, tau, device)
    shape = y.size()
    _, ind = y.max(dim=-1)  # [B, ...]
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_ = (y_hard - y).detach() + y
    list_indices = torch.arange(shape[-1]).view(1, 1, shape[-1]).float()
    indices = torch.sum(y_ * list_indices, dim=-1)
    return y_, indices


def is_primitive(v):
    """
    Checks if v is of primitive type.
    """
    return isinstance(v, (int, float, bool, str, torch.Tensor))


def free_params(module):
    if not isinstance(module, list):
        module = [module]
    for m in module:
        for p in m.parameters():
            p.requires_grad = True


def frozen_params(module):
    if not isinstance(module, list):
        module = [module]
    for m in module:
        for p in m.parameters():
            p.requires_grad = False


def sum_dictionaries(dicts: List):
    """
    Sums the values of the common keys in dictionary.

    Parameters
    ----------
    dicts (list) dictionaries containing numeric values

    Returns
    -------
    dictionary with summed values
    """

    def reducer(accumulator, element):
        for key, value in element.items():
            accumulator[key] = accumulator.get(key, 0) + value
        return accumulator

    return reduce(reducer, dicts, {})


def get_file_line_number(file_path: str) -> int:
    with open(file_path, "r", encoding="utf-8") as f:
        buf_size = 1024**2
        read_f = f.read
        lines: int = 0
        buf = read_f(buf_size)
        while buf:
            lines += buf.count("\n")
            buf = read_f(buf_size)
        return lines


def get_independent_opts(params: dict) -> bool:
    independent_opts = params["trainer"]["args"]["independent_optimizers"]
    return independent_opts


def clip_grad_norm(parameters, optimizer: dict) -> None:
    if optimizer["grad_norm"] is not None:
        torch.nn.utils.clip_grad_norm_(parameters, optimizer["grad_norm"])


def iterable_to_str(iterable: Iterable) -> str:
    return "'" + "', '".join([str(item) for item in iterable]) + "'"


def verify_str_arg(value: T, arg: Optional[str] = None, valid_values: Optional[Iterable[T]] = None, custom_msg: Optional[str] = None) -> T:
    """Code taken from torchvision library."""

    if not isinstance(value, (str, bytes, type(None))):
        if arg is None:
            msg = "Expected type str, but got type {type}."
        else:
            msg = "Expected type str for argument {arg}, but got type {type}."
        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)

    if valid_values is None:
        return value

    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = "Unknown value '{value}' for argument {arg}. Valid values are {{{valid_values}}}."
            msg = msg.format(value=value, arg=arg, valid_values=iterable_to_str(valid_values))
        raise ValueError(msg)

    return value


def verify_int_arg(
    value: T,
    arg: Optional[str] = None,
    min_value: Optional[T] = None,
    max_value: Optional[T] = None,
    custom_msg: Optional[str] = None,
) -> T:
    """Code taken from torchvision library."""

    if not isinstance(value, int):
        if arg is None:
            msg = "Expected type int, but got type {type}."
        else:
            msg = "Expected type int for argument {arg}, but got type {type}."
        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)

    if min_value is None and max_value is None:
        return value

    if value < min_value or value > max_value:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = "Value '{value}' for argument {arg} not in range. Valid values are between {min_value} and {max_value}."
            msg = msg.format(value=value, arg=arg, min_value=min_value, max_value=max_value)
        raise ValueError(msg)

    return value


def load_yaml(file_path: Path, return_object: bool = False) -> Union[dict, GenericConfig]:
    """Load a YAML file.

    Args:
        file_path (Path): existing path to the YAML file
        return_object (bool, optional): If True custom object is returned, otherwise a dictionary. Defaults to False.

    Returns:
        Union[dict, YAMLObject]: with the information in the file
    """
    with open(file_path, "r", encoding="utf-8") as file:
        config = yaml.full_load(file)
    if return_object:
        return GenericConfig(config)
    return config


def export_list_of_dicts_to_jsonl(data_list, filename):
    with open(filename, "w", encoding="utf-8") as outfile:
        for data in data_list:
            json.dump(data, outfile)
            outfile.write("\n")


def export_list_of_lists_to_csv(data_list, filename):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data_list)


def create_optimizers(model, config: Dict[str, GenericConfig]):
    optimizers = {}
    config_c = copy.deepcopy(config)
    for opt in config_c:
        opt_name, opt_config = opt.__dict__.popitem()
        opt_config = opt_config.__dict__
        gradient_norm = opt_config.pop("gradient_norm_clipping", None)
        schedulers_config = opt_config.pop("schedulers", None)

        optimizer = create_class_instance(opt_config.pop("name"), opt_config, model.parameters())
        schedulers = None
        if schedulers_config is not None:
            schedulers = create_lr_schedulers(schedulers_config, optimizer)
        optimizers[opt_name] = {"opt": optimizer, "grad_norm": gradient_norm, "schedulers": schedulers}

    return optimizers


def create_lr_schedulers(schedulers_config: Dict[str, GenericConfig], optimizer, last_epoch=None) -> List[Tuple[str, LRScheduler]]:
    """
    Create learning rate schedulers based on the provided configuration.

    Args:
        schedulers_config (Dict[str, GenericConfig]): A dictionary containing the configuration for the schedulers.
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        last_epoch (int, optional): The index of the last epoch. Default is None.

    Returns:
        List[Tuple[str, torch.optim.lr_scheduler.LRScheduler]]: A list of tuples where each tuple contains the step type
        ('epoch' or 'iteration') and the corresponding learning rate scheduler.

    Special Case:
        If the scheduler name is "torch.optim.lr_scheduler.SequentialLR", the function will handle it by building a sequence
        of schedulers using the `_build_sequence_of_schedulers` helper function.
    """
    schedulers = []
    schedulers_config_ = copy.deepcopy(schedulers_config)
    if not isinstance(schedulers_config_, list):
        schedulers_config_ = [schedulers_config_]
    for scheduler_config in schedulers_config_:
        scheduler_config = scheduler_config.to_dict()
        if last_epoch is not None:
            scheduler_config["last_epoch"] = last_epoch
        step_type = scheduler_config.pop("step_type", "epoch")
        if scheduler_config["name"] == "torch.optim.lr_scheduler.SequentialLR":
            scheduler_config = _build_sequence_of_schedulers(optimizer, scheduler_config)
        scheduler = create_class_instance(scheduler_config.pop("name"), scheduler_config, optimizer)
        schedulers.append((step_type, scheduler))
    return schedulers


def _build_sequence_of_schedulers(optimizer, scheduler_config: dict) -> dict:
    seq_schedulers = []
    for scheduler in scheduler_config["schedulers"]:
        seq_schedulers.append(create_class_instance(scheduler.pop("name"), scheduler, optimizer))
        scheduler_config["schedulers"] = seq_schedulers
    return scheduler_config


def create_schedulers(schedulers_config: dict, max_steps: int, steps_in_epoch: int) -> dict:
    schedulers = {}
    schedulers_config_ = copy.deepcopy(schedulers_config)
    for scheduler_config in schedulers_config_:
        scheduler_config = scheduler_config.to_dict()
        label = scheduler_config.pop("label")
        scheduler_config["max_steps"] = max_steps if scheduler_config.pop("max_steps_type", "full") == "full" else steps_in_epoch
        schedulers[label] = create_class_instance(scheduler_config.pop("name"), scheduler_config)

    return schedulers


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print("Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def filter_keys_by_part(dictionary: OrderedDict, part: str):
    """
    Filter keys in an OrderedDict by a specific part of the key.

    Args:
        dictionary (OrderedDict): The input OrderedDict to filter.
        part (str): The part of the key to match when filtering.

    Returns:
        OrderedDict: A new OrderedDict containing key-value pairs
        from the input dictionary where the 'part' is found in the key.
    """
    filtered_dict = OrderedDict()
    for key, value in dictionary.items():
        if part not in key:
            filtered_dict[key] = value
    return filtered_dict

def select_dimension_for_plot_(dimension_to_choose:int,
                               dimension_mask:Tensor,
                               locations:Tensor,
                               drift_at_locations_estimation:Tensor,
                               diffusion_at_locations_estimation:Tensor,
                               drift_at_locations_real:Tensor,
                               diffusion_at_locations_real:Tensor,
                               index_to_select=0):
    """
    From the drift and diffusion estimators and real 
    """
    # process
    process_dimension = dimension_mask[:,0,:].sum(axis=-1).long()
    where_one_dimension = torch.where(process_dimension == dimension_to_choose)[0]

    # select for one dimension 
    locations = locations[where_one_dimension]
    drift_at_locations_real = drift_at_locations_real[where_one_dimension]
    diffusion_at_locations_real = diffusion_at_locations_real[where_one_dimension]
    drift_at_locations_estimation = drift_at_locations_estimation[where_one_dimension]
    diffusion_at_locations_estimation = diffusion_at_locations_estimation[where_one_dimension]

    # detach and 
    locations = locations.detach().cpu().numpy()
    drift_at_locations_real = drift_at_locations_real.detach().cpu().numpy()
    diffusion_at_locations_real = diffusion_at_locations_real.detach().cpu().numpy()
    drift_at_locations_estimation = drift_at_locations_estimation.detach().cpu().numpy()
    diffusion_at_locations_estimation = diffusion_at_locations_estimation.detach().cpu().numpy()

    # select dimension 
    locations = locations[index_to_select,:,:dimension_to_choose]
    drift_at_locations_real = drift_at_locations_real[index_to_select,:,:dimension_to_choose]
    diffusion_at_locations_real = diffusion_at_locations_real[index_to_select,:,:dimension_to_choose]
    drift_at_locations_estimation = drift_at_locations_estimation[index_to_select,:,:dimension_to_choose]
    diffusion_at_locations_estimation = diffusion_at_locations_estimation[index_to_select,:,:dimension_to_choose]
    
    return (locations,
            drift_at_locations_real,
            diffusion_at_locations_real,
            drift_at_locations_estimation,
            diffusion_at_locations_estimation)

def select_dimension_for_plot(dimension:int,
                              dimension_mask:Tensor,
                              locations:Tensor,
                              drift_at_locations_estimation:Tensor,
                              diffusion_at_locations_estimation:Tensor,
                              drift_at_locations_real:Tensor,
                              diffusion_at_locations_real:Tensor,
                              index_to_select=0):
    """
    From the drift and diffusion estimators and real 
    """
    # process
    process_dimension = dimension_mask[:,0,:].sum(axis=-1).long()
    where_one_dimension = torch.where(process_dimension == dimension)[0]

    # select for one dimension 
    locations = locations[where_one_dimension]
    drift_at_locations_real = drift_at_locations_real[where_one_dimension]
    diffusion_at_locations_real = diffusion_at_locations_real[where_one_dimension]
    drift_at_locations_estimation = drift_at_locations_estimation[where_one_dimension]
    diffusion_at_locations_estimation = diffusion_at_locations_estimation[where_one_dimension]

    # detach and 
    locations = locations.detach().cpu().numpy()
    drift_at_locations_real = drift_at_locations_real.detach().cpu().numpy()
    diffusion_at_locations_real = diffusion_at_locations_real.detach().cpu().numpy()
    drift_at_locations_estimation = drift_at_locations_estimation.detach().cpu().numpy()
    diffusion_at_locations_estimation = diffusion_at_locations_estimation.detach().cpu().numpy()

    # select dimension 
    locations = locations[index_to_select,:,:dimension]
    drift_at_locations_real = drift_at_locations_real[index_to_select,:,:dimension]
    diffusion_at_locations_real = diffusion_at_locations_real[index_to_select,:,:dimension]
    drift_at_locations_estimation = drift_at_locations_estimation[index_to_select,:,:dimension]
    diffusion_at_locations_estimation = diffusion_at_locations_estimation[index_to_select,:,:dimension]
    
    return (locations,
            drift_at_locations_real,
            diffusion_at_locations_real,
            drift_at_locations_estimation,
            diffusion_at_locations_estimation)
