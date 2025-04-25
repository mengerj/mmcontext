'''
import importlib.resources
import json
import logging.config

import pytest
from dotenv import load_dotenv


def load_logging_config():
    """
    Load the logging configuration from `logging_config.json` in the `mmcontext.conf` module.

    Returns
    -------
    dict
        The parsed logging configuration as a dictionary.
    """
    try:
        # Get the path to the logging_config.json file
        resource_path = importlib.resources.files("mmcontext.conf") / "logging_config.json"

        # Open the resource file
        with resource_path.open("r", encoding="utf-8") as config_file:
            logging_config = json.load(config_file)

        return logging_config
    except FileNotFoundError as err:
        raise RuntimeError("The logging configuration file could not be found.") from err
    except json.JSONDecodeError as err:
        raise ValueError(f"Error decoding the JSON logging configuration: {err}") from err


def setup_logging():
    """Load the logging configuration from the logging_config.json file and configure the logging system."""
    config_dict = load_logging_config()
    # Configure logging
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(__name__)
    logger.info("mmcontext logging configured using the specified configuration file.")


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    setup_logging()


@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()
'''
