import logging
import logging.config

import torch.distributed as dist
import yaml

from .. import logging_config_filename


def setup_logging():
    with open(logging_config_filename, "r", encoding="utf-8") as config_file:
        logging_config = yaml.safe_load(config_file)
        logging.config.dictConfig(logging_config)

    # Ensure every LogRecord has a 'rank' attribute so formatters like
    # "RANK_%(rank)s - ..." don't fail for third-party loggers.
    class _EnsureRankFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            if not hasattr(record, "rank"):
                # Default to rank 0 for non-distributed contexts
                setattr(record, "rank", 0)
            return True

    rank_filter = _EnsureRankFilter()
    # Attach filter to all existing handlers (root and any configured loggers)
    all_logger_names = ["root"] + list(logging.root.manager.loggerDict.keys())
    for name in all_logger_names:
        logger = logging.getLogger(None if name == "root" else name)
        for handler in getattr(logger, "handlers", []):
            handler.addFilter(rank_filter)


class RankLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger):
        rank = 0
        if dist.is_initialized():
            try:
                rank = dist.get_rank()
            except RuntimeError:
                rank = 0
        super().__init__(logger, extra={"rank": rank})
