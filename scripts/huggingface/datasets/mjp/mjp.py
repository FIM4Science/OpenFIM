# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Collection of datasets for the MJP."""

import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import datasets
import torch

from fim.data.utils import load_file
from fim.typing import Path, Paths


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
# _URLS = {
#     "first_domain": "https://huggingface.co/great-new-dataset-first_domain.zip",
#     "second_domain": "https://huggingface.co/great-new-dataset-second_domain.zip",
# }

_ROOT_URL = "data/DFR"


@dataclass
class MJPDatasetsBuilderConfig(datasets.BuilderConfig):
    """MJPDatasets builder config.."""

    file_name: Optional[str] = None


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class MJP(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    BUILDER_CONFIG_CLASS = MJPDatasetsBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        MJPDatasetsBuilderConfig(
            name="DFR_V=0",
            file_name="6_st_DFR_V=0.zip",
            version=VERSION,
            description="This part of my dataset covers a first domain",
        ),
        MJPDatasetsBuilderConfig(
            name="DFR_V=1",
            file_name="6_st_DFR_V=1.zip",
            version=VERSION,
            description="This part of my dataset covers a first domain",
        ),
        MJPDatasetsBuilderConfig(
            name="DFR_V=2",
            file_name="6_st_DFR_V=2.zip",
            version=VERSION,
            description="This part of my dataset covers a first domain",
        ),
        MJPDatasetsBuilderConfig(
            name="DFR_V=3",
            file_name="6_st_DFR_V=3.zip",
            version=VERSION,
            description="This part of my dataset covers a first domain",
        ),
    ]

    DEFAULT_CONFIG_NAME = "DFR_V=0"

    files_to_load = {
        "observation_grid": "fine_grid_grid.pt",
        "observation_values": "fine_grid_noisy_sample_paths.pt",
        "mask_seq_lengths": "fine_grid_mask_seq_lengths.pt",
        "time_normalization_factors": "fine_grid_time_normalization_factors.pt",
        "intensity_matrices": "fine_grid_intensity_matrices.pt",
        "adjacency_matrices": "fine_grid_adjacency_matrices.pt",
        "initial_distributions": "fine_grid_initial_distributions.pt",
    }

    def _info(self):
        features = datasets.Features(
            {
                "observation_grid": datasets.Sequence(datasets.Sequence(datasets.Sequence(datasets.Value("float32")))),
                "observation_values": datasets.Sequence(datasets.Sequence(datasets.Sequence(datasets.Value("uint32")))),
                "time_normalization_factors": datasets.Value("float32"),
                "mask_seq_lengths": datasets.Sequence(datasets.Sequence(datasets.Value("int32"))),
                "intensity_matrices": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
                "adjacency_matrices": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
                "initial_distributions": datasets.Sequence(datasets.Value("uint64")),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = f"{_ROOT_URL}/{self.config.file_name}"
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"datadir": pathlib.Path(data_dir) / self.config.file_name.split(".")[0]},
            )
        ]

    def __get_files(self, path: Path) -> Paths:
        files_to_load = [(key, pathlib.Path(path) / file_name) for key, file_name in self.files_to_load.items()]
        return files_to_load

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, datadir):
        data = defaultdict(list)
        files_to_load = self.__get_files(datadir)
        for key, file_path in files_to_load:
            data[key].append(load_file(file_path))
        for k, v in data.items():
            data[k] = torch.cat(v)

        for id in range(len(data["observation_grid"])):
            yield id, {k: v[id].tolist() for k, v in data.items() if k in self.info.features}
