# embedding_benchmark/evaluators/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


class EvalResult(dict):
    """Just a `dict[str, float]` with pretty `__repr__`."""

    def __repr__(self):
        lines = []
        for k, v in self.items():
            if isinstance(v, int | np.integer):
                lines.append(f"{k}: {v}")
            elif isinstance(v, float | np.floating):
                lines.append(f"{k}: {v:0.4f}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)


class BaseEvaluator(ABC):
    """Every metric or plotter derives from this class."""

    name: str  # folder & column prefix
    requires_pair: bool = False
    produces_plot: bool = False

    # ---- API -----------------------------------------------------------
    @abstractmethod
    def compute(
        self,
        emb1: np.ndarray,
        *,
        emb2: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        adata: ad.AnnData | None = None,
        **kw,
    ) -> EvalResult:
        """Return *deterministic* scalar metrics."""

    @abstractmethod
    def plot(  # optional
        self,
        emb1: np.ndarray,
        out_dir: Path,
        *,
        emb2: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        adata: ad.AnnData | None = None,
        **kw,
    ) -> None:
        """Plot the results."""
        pass
