from importlib.metadata import version

from . import engine, eval, models, pp

__all__ = ["models", "pp", "engine", "eval"]

__version__ = version("mmcontext")
