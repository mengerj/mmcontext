from importlib.metadata import version

from . import embed, eval, models, pl

__all__ = ["models", "eval", "pl", "embed"]

__version__ = version("mmcontext")
