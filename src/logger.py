import sys

from loguru import logger

from src.config import Config


def setup_logger(config: Config):
    logger.remove()

    log_config = config.get("logging", {})
    level = log_config.get("level", "INFO")
    fmt = log_config.get("format", "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    logger.add(
        sys.stdout,
        level=level,
        format=fmt,
    )
    return logger