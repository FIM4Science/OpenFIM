import logging
import sys
from pathlib import Path

import click

from reasoningschema.utils.evaluation import EvaluationFactory
from reasoningschema.utils.helper import load_yaml

# Define a list of valid model types
VALID_MODEL_TYPES = ["meta-llama/Llama-2-7b-chat-hf"]


@click.command()
@click.option("--config", "-c", default="config.yaml", type=click.Path(exists=True, dir_okay=False), help="Path to config file.")
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
def main(config: Path, log_level):
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config_dict = load_yaml(config)

    # Override parameters with command-line arguments, if provided
    # if model_type:
    #     config_dict["model"]["type"] = model_type
    # if model_location:
    #     config_dict["model"]["location"] = model_location
    # if dataset_name:
    #     config_dict["dataset"]["name"] = dataset_name
    # if dataset_location:
    # config_dict["dataset"]["location"] = dataset_location

    # Your inference code here using the provided config parameters and prompt
    print(f"Model Type: {config_dict['model']['name']}")
    # print(f"Model Location: {config_dict['model']['path']}")
    print(f"Dataset Name: {config_dict['dataset']['name']}")
    print(f"Dataset Location: {config_dict['dataset']['root_dir']}")
    print(f"Prompt Location: {config_dict['dataset']['prompt_path']}")
    print(f"Output Location: {config_dict['output_path']}")

    tokenizer_conf = config_dict["tokenizer"]
    dataset_conf = config_dict["dataset"]
    model_conf = config_dict["model"]
    generation_conf = config_dict["generation"]
    evaluation = EvaluationFactory.create(
        config_dict.get("evaluation_type"),
        device=config_dict["device_map"],
        output_path=config_dict["output_path"],
        tokenizer_param=tokenizer_conf,
        dataset_param=dataset_conf,
        model_param=model_conf,
        **generation_conf,
    )
    evaluation.evaluate()
    evaluation.save()


if __name__ == "__main__":
    main()
