# embedding_benchmark/evaluators/ari.py
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from .base import BaseEvaluator, EvalResult
from .registry import register


@register
class ARI(BaseEvaluator):
    """
    Adjusted Rand Index

    between ground-truth labels and K-means clusters obtained from the embedding.

    Expects:
        • emb1   – (n × d) numpy array
        • labels – 1-D array‐like of length n
    """

    name = "ARI"
    requires_pair = False
    produces_plot = False

    # you can expose these as Hydra knobs if you like
    n_clusters: int = 30
    random_state: int = 0
    n_init: str | int = "auto"

    def compute(
        self,
        emb1: np.ndarray,
        *,
        labels: np.ndarray,
        **kw,
    ) -> EvalResult:
        """Return {'ari': <float>}."""
        k = kw.get("kmeans_k", self.n_clusters)

        km = KMeans(
            n_clusters=k,
            n_init=self.n_init,
            random_state=self.random_state,
        ).fit(emb1)

        ari = adjusted_rand_score(labels, km.labels_)
        return EvalResult(ari=ari)

    def plot(self, **kw):
        """Plot the ARI score."""
        pass
