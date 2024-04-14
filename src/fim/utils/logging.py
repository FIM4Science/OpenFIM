import logging
import logging.config

import torch.distributed as dist
import yaml

from .. import logging_config_filename


def setup_logging():
    with open(logging_config_filename, "r", encoding="utf-8") as config_file:
        logging_config = yaml.safe_load(config_file)
        logging.config.dictConfig(logging_config)


class RankLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger):
        rank = 0
        if dist.is_initialized():
            try:
                rank = dist.get_rank()
            except RuntimeError:
                rank = 0
        super().__init__(logger, extra={"rank": rank})
