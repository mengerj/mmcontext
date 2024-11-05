import importlib.resources
import json
import logging.config
import os
from importlib.metadata import version

from . import engine, pp

__all__ = ["pp", "engine", "eval"]

__version__ = version("mmcontext")


def setup_logging():
    # Load logging configuration from the specified file
    with importlib.resources.open_text("mmcontext.conf", "logging_config.json") as config_file:
        config_dict = json.load(config_file)

    # Configure logging
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(__name__)
    logger.info("Logging configured using the specified configuration file.")


# Call the function to set up logging when the package is imported
setup_logging()
