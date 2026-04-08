import sys
from loguru import logger

def setup_logger(config: dict):
    logger.remove() #Remove the default logger.

    log_config = config.get("logging", {}) #Go to the logging part of config or use {} as a fallback.
    level = log_config.get("level", "INFO") #Use the level from config or use INFO as fallback
    fmt = log_config.get("format","{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
 
    logger.add(
        sys.stdout,
        level=level,
        format=fmt,
    )
    return logger