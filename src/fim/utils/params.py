from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class Tokenizer(object):
    name: str
    cache_dir: Union[str, Path]


@dataclass
class Dataset(object):
    name: str
    root_dir: Union[str, Path]
    batch_size: int
    split: str
    num_workers: int
    supervised: bool
    prompt_path: Union[str, Path]
    tokenize: bool
    padding_side: str
    max_padding_length: int
    output_fields: list
    force_download: bool


@dataclass
class Model(object):
    module: str
    name: str
    path: str | Path


#   args:
#     backbone: meta-llama/Llama-2-7b-hf
#     backbone_path: null # if null it uses the default cache path set for huggingface


@dataclass
class Evaluation(object):
    device: str
    eval_type: str
    output_path: Union[str, Path]
    tokenizer: Tokenizer
    dataset: Dataset
    model: Model
