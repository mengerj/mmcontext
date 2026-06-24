from importlib.metadata import version

from . import embed, eval, io, modules, pl

__all__ = ["embed", "eval", "io", "modules", "pl"]

__version__ = version("mmcontext")
