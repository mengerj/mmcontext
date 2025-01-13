from importlib.metadata import version

from . import engine, eval, pp

__all__ = ["pp", "engine", "eval"]

__version__ = version("mmcontext")
