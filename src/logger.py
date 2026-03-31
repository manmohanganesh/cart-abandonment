from loguru import logger
import sys


def setup_logger(config: dict):
    logger.remove()

    log_config = config.get("logging", {})
    level = log_config.get("level", "INFO")
    fmt = log_config.get(
        "format",
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    logger.add(
        sys.stdout,
        level=level,
        format=fmt,
    )

    return logger