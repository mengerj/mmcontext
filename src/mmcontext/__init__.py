from importlib.metadata import version

from . import embed, eval, io, modules, pl

__all__ = ["modules", "io", "eval", "pl", "embed"]

__version__ = version("mmcontext")
