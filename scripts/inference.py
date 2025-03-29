import logging
from pathlib import Path

import click
import torch

import fim.models  # noqa: F401
from fim.utils.evaluation import create_evaluation_from_config
from fim.utils.logging import RankLoggerAdapter, setup_logging


setup_logging()
logger = RankLoggerAdapter(logging.getLogger(__name__))


@click.command()
@click.option(
    "--config", "-c", "config_path", default="config.yaml", type=click.Path(exists=True, dir_okay=False), help="Path to config file."
)
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
def main(config_path: Path, log_level=logging.DEBUG):
    torch.cuda.empty_cache()

    evaluation = create_evaluation_from_config(config_path)
    evaluation.evaluate()
    evaluation.save()


if __name__ == "__main__":
    main()
