import json
import logging.config
import os
from importlib.metadata import version

from . import io, pl, pp, tl

__all__ = ["io", "pl", "pp", "tl"]

__version__ = version("mmcontext")


def setup_logging(logging_config_path=None):
    if logging_config_path is None:
        logging_config_path = os.path.join(os.path.dirname(__file__), "../../conf/logging_config.json")
        print(logging_config_path)

    # Load logging configuration from the specified file
    with open(logging_config_path) as config_file:
        config_dict = json.load(config_file)

    # Configure logging
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(__name__)
    logger.info("Logging configured using the specified configuration file.")


# Call the function to set up logging when the package is imported
setup_logging()
