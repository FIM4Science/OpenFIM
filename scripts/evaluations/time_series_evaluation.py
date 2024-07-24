"""Evaluate one or more model configurations on a dataset and save the results."""

import os

import matplotlib.pyplot as plt
import torch

from fim.utils.evaluation import EvaluationFactory
from fim.utils.helper import load_yaml


torch.manual_seed(4)

os.chdir("/home/koerner/FIM")

config_path = "/home/koerner/FIM/configs/inference/eval_fim_ode_sin_data.yaml"

config_inference = load_yaml(config_path)

config_train = load_yaml(config_inference["train_config"])

config_inference["evaluation"]["model_param"] = config_train["model"]
config_inference["evaluation"]["dataset_param"] = config_inference["dataset"]

evaluation = EvaluationFactory.create(**config_inference["evaluation"])

evaluation.evaluate()

# fig, axes = evaluation.visualize_solutions(save_dir=evaluation.output_path)
# plt.show()

# fig, axes = evaluation.visualize_drift(save_dir=evaluation.output_path)
# plt.show()

indices = [73, 3771, 3903, 4832]  #  5543, 6774, 6956, 7527, 9357]
fig, axes = evaluation.visualize(indices=indices, save_dir=evaluation.output_path)
plt.show()

evaluation.save()
