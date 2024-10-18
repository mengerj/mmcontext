import importlib.resources
import json
import logging.config

import pytest


def setup_logging():
    # Load logging configuration from the specified file
    with importlib.resources.open_text("mmcontext.conf", "logging_config.json") as config_file:
        config_dict = json.load(config_file)

    # Configure logging
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(__name__)
    logger.info("Logging configured using the specified configuration file.")


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    setup_logging()
