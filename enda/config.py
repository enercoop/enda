"""Config of enda"""
import logging


# logger

LOGGER = logging.getLogger("enda")
# show timestamp on logs
logger_handler = logging.StreamHandler()

logger_handler.setFormatter(
    logging.Formatter(
        fmt="[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"
    )
)
LOGGER.addHandler(logger_handler)
LOGGER.propagate = False


# -------  config

logger = LOGGER


def set_logger(new_logger):
    """
    Define a new logger
    """
    global logger
    logger = new_logger


def get_logger():
    """
    Get the global logger
    """
    return logger
