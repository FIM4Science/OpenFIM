import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import click

from fim.utils.helper import load_yaml
from fim.utils.evaluation import EvaluationFactory



@click.command()
@click.option(
    "--config", "-c", default="config.yaml", type=click.Path(exists=True, dir_okay=False), help="Path to config file."
)
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
def main(config: Path, log_level):
    non_click_main(config, log_level)


def non_click_main(config: str, log_level=logging.DEBUG):
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config_inference = load_yaml(config)
    config_train = load_yaml(config_inference["train_config"])

    config_inference["evaluation"]["model_param"] = config_train["model"]
    config_inference["evaluation"]["dataset_param"] = config_inference["dataset"]

    torch.cuda.empty_cache()
    torch.manual_seed(config_train["experiment"]["seed"])

    evaluation = EvaluationFactory.create(**config_inference["evaluation"])

    evaluation.evaluate()

    fig, axes = evaluation.visualize()
    plt.show()

    evaluation.save()

    logging.info(f"Evaluation finished. Results saved at {evaluation.output_path}/predictions.jsonl.")


if __name__ == "__main__":
    non_click_main("/home/koerner/FIM/configs/inference/synthetic_data_evaluation.yaml")
