# embedding_benchmark/evaluators/registry.py
from mmcontext.eval.base import BaseEvaluator

_REGISTRY = {}


def register(cls):
    """Register a new evaluator. Called through get"""
    if not issubclass(cls, BaseEvaluator):
        raise TypeError("Can only register subclasses of BaseEvaluator.")
    _REGISTRY[cls.name] = cls
    return cls


def get(name):
    """Get an evaluator by name."""
    return _REGISTRY[name]
