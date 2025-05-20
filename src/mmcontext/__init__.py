from importlib.metadata import version

from . import eval, models, pl

__all__ = ["models", "eval", "pl"]

__version__ = version("mmcontext")
