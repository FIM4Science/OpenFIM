from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from loguru import logger
from tqdm import tqdm


SOURCE_SPLIT = "train"
PATH = "/cephfs_projects/foundation_models/hawkes/data/10_1_st_hawkes_exp_smaller_scale_2000_paths_250_events/" + SOURCE_SPLIT

OUT_PATH = "/cephfs_projects/foundation_models/hawkes/data/"
OUT_PATH = Path(OUT_PATH)
OUT_PATH.mkdir(parents=True, exist_ok=True)
NUM_PROCESSES = 10
NUM_PATHS_FOR_INFERENCE = 1950
TOTAL_NUM_PATHS = 2000
if SOURCE_SPLIT == "train":
    TOTAL_NUM_PROCESSES = 10
else:
    TOTAL_NUM_PROCESSES = 1
idx = list(range(NUM_PROCESSES))
data = {}
for file in Path(PATH).rglob("*.pt"):
    with open(file, "rb") as f:
        logger.info(f"Loading file: {file}")
        if file.stem == "intensities":
            tmp = torch.load(f, weights_only=False)
            # for i in range(len(tmp)):  # 10
            #     for j in range(len(tmp[i])):  # 2000
            #         for k in range(len(tmp[i][j])):  # 3
            #             tmp[i][j][k] = torch.tensor(tmp[i][j][k])
            data[file.stem] = tmp
        elif file.stem == "intensity_times":
            tmp = torch.load(f, weights_only=False)
            # for i in range(len(tmp)):  # 10
            #     for j in range(len(tmp[i])):  # 2000
            #         tmp[i][j] = torch.tensor(tmp[i][j])
            data[file.stem] = tmp
        else:
            tmp = torch.load(f, weights_only=False)
            if not isinstance(tmp, torch.Tensor):
                tmp = torch.tensor(np.array(tmp))
            data[file.stem] = tmp

our_to_easytpp = {
    "event_times": "time_since_start",
    "event_types": "type_event",
    "delta_times": "time_since_last_event",
    "kernel_evaluations": "target_kernel_evaluations",
    "kernel_grids": "target_kernel_grids",
    "base_intensities": "target_base_intensities",
    "intensities": "target_intensities",
    "intensity_times": "target_intensity_times",
}

delta_times = torch.diff(data["event_times"], dim=-2)
delta_times = torch.cat((torch.zeros_like(delta_times[..., :1, :]), delta_times), dim=-2)
data["delta_times"] = delta_times

for key, value in data.items():
    if isinstance(value, torch.Tensor):
        logger.info(f"Shape of {key}: {value.shape}")


for i, process_idx in tqdm(enumerate(idx), total=len(idx), desc="Processing processes"):
    train_data = {
        k: v[i][:NUM_PATHS_FOR_INFERENCE] if k == "intensity_times" or k == "intensities" or v.dim() == 4 else v[i] for k, v in data.items()
    }
    test_data = {
        k: v[i][NUM_PATHS_FOR_INFERENCE:] if k == "intensity_times" or k == "intensities" or v.dim() == 4 else v[i] for k, v in data.items()
    }

    train_data["dim_process"] = torch.ones(train_data["event_times"].size(0), dtype=torch.int32) * train_data["event_times"].size(1)
    test_data["dim_process"] = torch.ones(test_data["event_times"].size(0), dtype=torch.int32) * test_data["event_times"].size(1)
    train_data["seq_len"] = torch.ones(train_data["event_times"].size(0), dtype=torch.int32) * train_data["event_times"].size(1)
    test_data["seq_len"] = torch.ones(test_data["event_times"].size(0), dtype=torch.int32) * test_data["event_times"].size(1)
    train_data["kernel_evaluations"] = train_data["kernel_evaluations"].unsqueeze(0).repeat(NUM_PATHS_FOR_INFERENCE, 1, 1)
    test_data["kernel_evaluations"] = test_data["kernel_evaluations"].unsqueeze(0).repeat(TOTAL_NUM_PATHS - NUM_PATHS_FOR_INFERENCE, 1, 1)
    train_data["kernel_grids"] = train_data["kernel_grids"].unsqueeze(0).repeat(NUM_PATHS_FOR_INFERENCE, 1, 1)
    test_data["kernel_grids"] = test_data["kernel_grids"].unsqueeze(0).repeat(TOTAL_NUM_PATHS - NUM_PATHS_FOR_INFERENCE, 1, 1)
    train_data["base_intensities"] = train_data["base_intensities"].unsqueeze(0).repeat(NUM_PATHS_FOR_INFERENCE, 1)
    test_data["base_intensities"] = test_data["base_intensities"].unsqueeze(0).repeat(TOTAL_NUM_PATHS - NUM_PATHS_FOR_INFERENCE, 1)
    train_data["event_times"] = train_data["event_times"].squeeze(-1)
    test_data["event_times"] = test_data["event_times"].squeeze(-1)
    train_data["event_types"] = train_data["event_types"].squeeze(-1)
    test_data["event_types"] = test_data["event_types"].squeeze(-1)
    train_data["seq_idx"] = torch.arange(NUM_PATHS_FOR_INFERENCE).int()
    test_data["seq_idx"] = torch.arange(TOTAL_NUM_PATHS - NUM_PATHS_FOR_INFERENCE).int()

    # Rename keys to match EasyTPP format
    train_data = {our_to_easytpp.get(k, k): v for k, v in train_data.items()}
    test_data = {our_to_easytpp.get(k, k): v for k, v in test_data.items()}

    logger.info(
        f"Exporting process {process_idx} with {train_data['time_since_start'].size(0)} train and {test_data['time_since_start'].size(0)} test paths"
    )
    train_dataset = Dataset.from_dict(train_data, split="train")
    test_dataset = Dataset.from_dict(test_data, split="test")

    DATASET_NAME = "FIM4Science/hawkes-synthetic-short-scale-single-process"
    CONFIG_NAME = f"{SOURCE_SPLIT}_process_{process_idx}"

    # Push to hub as a dataset
    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
    dataset_dict.push_to_hub(DATASET_NAME, private=False, config_name=CONFIG_NAME)
