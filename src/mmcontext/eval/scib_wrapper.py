# embedding_benchmark/evaluators/scib_bundle.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from mmcontext.eval.base import BaseEvaluator, EvalResult
from mmcontext.eval.evaluate_scib import scibEvaluator  # your class
from mmcontext.eval.registry import register
from mmcontext.eval.utils import LabelKind


@register
class ScibBundle(BaseEvaluator):
    """Wraps the original `scibEvaluator` to handle all bio/batch combinations."""

    name = "scib"
    requires_pair = False  # uses one embedding at a time
    produces_plot = False  # scIB returns numbers only

    def compute_dataset_model(
        self,
        emb1: np.ndarray,
        adata: ad.AnnData,
        dataset_name: str,
        model_id: str,
        bio_labels: list[str],
        batch_labels: list[str],
        embed_name: str = "X_embed",
        **kw,
    ) -> list[dict[str, Any]]:
        """
        Compute scIB metrics for all bio/batch combinations.

        Returns a list of result dictionaries that can be directly added to the results DataFrame.
        """
        # attach embedding to a *copy* so we don't pollute the shared object
        adata = adata.copy()
        adata.obsm[embed_name] = emb1
        # a list of other embedding keys that are in the adata.obsm
        emb_keys = list(adata.obsm.keys())
        results = []

        # Check which labels actually exist in the data
        available_bio_labels = [label for label in bio_labels if label in adata.obs.columns]
        available_batch_labels = [label for label in batch_labels if label in adata.obs.columns]

        if not available_bio_labels:
            print(f"Warning: No biological labels found in data for {dataset_name}")
            return results

        if not available_batch_labels:
            print(f"Warning: No batch labels found in data for {dataset_name}")
            return results

        # Run scIB for each bio/batch combination
        for bio_label in available_bio_labels:
            for batch_label in available_batch_labels:
                print(f"Running scIB for {dataset_name}/{model_id}: {bio_label} (bio) vs {batch_label} (batch)")

                try:
                    evaluator = scibEvaluator(
                        adata=adata,
                        batch_key=batch_label,
                        label_key=bio_label,
                        embedding_key=emb_keys,
                        reconstructed_keys=[],  # there are no reconstructed layers so far
                        data_id=kw.get("data_id", f"{dataset_name}_{model_id}"),
                        n_top_genes=kw.get("n_top_genes"),
                        max_cells=kw.get("max_cells"),
                        in_parallel=kw.get("in_parallel", True),
                    )

                    df = evaluator.evaluate()  # original return type
                    result_dict = df.iloc[0].to_dict()

                    # Convert each metric to a separate row
                    for metric_name, value in result_dict.items():
                        if metric_name in ["data_id", "hvg", "type"]:
                            continue  # Skip metadata columns

                        results.append(
                            {
                                "dataset": dataset_name,
                                "model": model_id,
                                "bio_label": bio_label,
                                "batch_label": batch_label,
                                "metric": f"scib/{metric_name}",
                                "value": value,
                                "data_id": result_dict.get("data_id", ""),
                                "hvg": result_dict.get("hvg", ""),
                                "type": result_dict.get("type", ""),
                            }
                        )

                except Exception as e:
                    print(f"Error running scIB for {bio_label}/{batch_label}: {e}")
                    # Add error entry
                    results.append(
                        {
                            "dataset": dataset_name,
                            "model": model_id,
                            "bio_label": bio_label,
                            "batch_label": batch_label,
                            "metric": "scib/error",
                            "value": str(e),
                            "data_id": "",
                            "hvg": "",
                            "type": "",
                        }
                    )

        return results

    # Keep the old interface for backward compatibility, but it won't be used in the new workflow
    def compute(
        self,
        emb1: np.ndarray,
        *,
        labels: np.ndarray,
        adata: ad.AnnData,
        label_kind: LabelKind,
        label_key: str,
        **kw,
    ) -> EvalResult:
        """Legacy interface - not used in new workflow."""
        return EvalResult(error="ScibBundle should use compute_dataset_model method")

    def plot(self, adata: ad.AnnData, **kw) -> None:
        """Plot scIB metrics."""
        pass
