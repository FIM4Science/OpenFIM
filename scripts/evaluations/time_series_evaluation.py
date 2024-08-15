"""Evaluate one or more model configurations on a dataset and save the results."""

import os

import matplotlib.pyplot as plt
import torch

from fim.utils.evaluation import EvaluationFactory
from fim.utils.helper import load_yaml


torch.manual_seed(4)

os.chdir("/home/koerner/FIM")

config_path = "/home/koerner/FIM/configs/inference/eval_fim_ode.yaml"

config_inference = load_yaml(config_path)
train_config_dir = config_inference["evaluation"]["experiment_dir"] + "/train_parameters.yaml"
config_train = load_yaml(train_config_dir)

config_inference["evaluation"]["model_param"] = config_train.get("model")
# take dataset parameters from train config if not specified in inference config
for k, v in config_train.get("dataset").items():
    if k not in config_inference["evaluation"]["dataset_param"]:
        config_inference["evaluation"]["dataset_param"][k] = v

evaluation = EvaluationFactory.create(**config_inference["evaluation"])
evaluation.plot_certainty  = True
evaluation.evaluate()

indices = torch.randperm(len(evaluation.predictions))[:20]
# fig, axes = evaluation.visualize_solutions(indices=indices, save_dir=evaluation.output_path)
# plt.show()
# fig, axes = evaluation.visualize_drift(None, save_dir=evaluation.output_path)
# plt.show()
# fig, axes = evaluation.visualize_init_condition(indices=indices, save_dir=evaluation.output_path)
# plt.show()


return_values = evaluation.visualize(indices=indices, save_dir=evaluation.output_path / config_inference['evaluation']['dataset_param']['split'])
plt.show()

evaluation.save(save_dir = evaluation.output_path / config_inference['evaluation']['dataset_param']['split'])
