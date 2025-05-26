# embedding_benchmark/evaluators/scib_bundle.py
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from mmcontext.eval.base import BaseEvaluator, EvalResult
from mmcontext.eval.evaluate_scib import scibEvaluator  # your class
from mmcontext.eval.registry import register


@register
class ScibBundle(BaseEvaluator):
    """Wraps the original `scibEvaluator` so it plugs into the new driver."""

    name = "scib"
    requires_pair = False  # uses one embedding at a time
    produces_plot = False  # scIB returns numbers only

    def compute(
        self,
        emb1: np.ndarray,
        *,
        labels: np.ndarray,
        adata: ad.AnnData,
        batch_key: str = "batch",
        label_key: str = "celltype",
        embed_name: str = "X_embed",
        **kw,
    ) -> EvalResult:
        """Compute scIB metrics."""
        # attach embedding to a *copy* so we don't pollute the shared object
        adata = adata.copy()
        adata.obsm[embed_name] = emb1

        evaluator = scibEvaluator(
            adata=adata,
            batch_key=batch_key,
            label_key=label_key,
            embedding_key=embed_name,
            reconstructed_keys=[],  # there are no reconstructed layers so far
            data_id=kw.get("data_id", ""),
            n_top_genes=kw.get("n_top_genes"),
            max_cells=kw.get("max_cells"),
            in_parallel=kw.get("in_parallel", True),
        )

        df = evaluator.evaluate()  # original return type
        return EvalResult(**df.iloc[0].to_dict())  # flatten 1-row DF to dict
