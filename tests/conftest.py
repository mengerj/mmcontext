import json
import logging.config

import pytest


def setup_logging(logging_config_path="conf/logging_config.json"):
    # Load logging configuration from file
    with open(f"{logging_config_path}") as config_file:
        config_dict = json.load(config_file)

    # Configure logging
    logging.config.dictConfig(config_dict)
    # Additional configurations can be added here
    # Example: setting a specific logger differently
    # logger = logging.getLogger('some_specific_module')
    # logger.setLevel(logging.WARNING)


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    setup_logging()
