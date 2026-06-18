"""mmcontext.eval — Evaluation tools for mmcontext models.

Evaluators follow a registry pattern: import a module to register its class,
then retrieve it by name via :func:`get`.  The evaluate-model-2.0 notebook
uses ``LabelSimilarity`` and ``OmicsQueryAnnotator`` as its primary tools.
"""

# Import evaluator modules so their @register decorators run.
from . import ari, label_similarity
from .base import BaseEvaluator, EvalResult
from .query_annotate import OmicsQueryAnnotator
from .registry import get, register

__all__ = [
    "BaseEvaluator",
    "EvalResult",
    "OmicsQueryAnnotator",
    "get",
    "register",
]
