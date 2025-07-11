# mmcontext/eval/label_retrieval.py
from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from scipy.stats import mannwhitneyu
from sklearn.metrics import auc, roc_curve

from .base import BaseEvaluator, EvalResult
from .registry import register
from .utils import LabelKind

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b)
    return (a_norm @ b_norm).astype(np.float32)


@register
class LabelSimilarity(BaseEvaluator):
    """
    Compute similarity scores and ROC metrics for each unique label.

    For each unique value v in labels:
        • Compute similarity scores between cell embeddings and label prototype
        • Generate ROC curves and calculate AUC scores
        • Create UMAP visualization colored by similarity scores
        • Plot similarity score distributions

    Returns
    -------
        - AUC score for each label
        - Mean AUC across all labels

    Produces:
        - ROC curve plots
        - UMAP visualizations
        - Similarity score histograms
    """

    name = "LabelSimilarity"
    requires_pair = True
    produces_plot = True

    similarity: str = "cosine"  # can be overridden via Hydra
    bins: int = 40
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_random_state: int = 42

    def _pair_sim(self, emb1: np.ndarray, vec: np.ndarray) -> np.ndarray:
        if self.similarity == "cosine":
            return _cosine(emb1, vec)
        if self.similarity == "dot":
            return emb1 @ vec
        raise ValueError(self.similarity)

    def _compute_umap(self, emb1: np.ndarray) -> np.ndarray:
        """Compute UMAP embedding of the data."""
        reducer = umap.UMAP(
            n_neighbors=self.umap_n_neighbors, min_dist=self.umap_min_dist, random_state=self.umap_random_state
        )
        return reducer.fit_transform(emb1)

    def _compute_roc(self, scores: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute ROC curve and AUC for a set of scores and true labels."""
        fpr, tpr, _ = roc_curve(mask, scores)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    def compute(
        self,
        emb1: np.ndarray,
        *,
        emb2: np.ndarray,
        labels: np.ndarray | list | pd.Series,
        label_key: str,
        label_kind: LabelKind,
        **kw,
    ) -> EvalResult:
        """Compute similarity scores and ROC metrics for each unique label."""
        labels = np.asarray(labels)
        uniq = np.unique(labels)
        out = {}
        auc_scores = []

        for v in uniq:
            mask = labels == v
            label_vec = emb2[mask][0]  # first row for that value
            sim = self._pair_sim(emb1, label_vec)

            # Compute ROC curve and AUC
            _, _, roc_auc = self._compute_roc(sim, mask)
            auc_scores.append(roc_auc)

            prefix = f"{v}"
            out[f"{prefix}/auc"] = float(roc_auc)

        # Add mean AUC and metadata
        out["mean_auc"] = float(np.mean(auc_scores))
        out["label_kind"] = label_kind.value
        out["n_labels"] = len(uniq)
        return EvalResult(**out)

    def plot(
        self,
        emb1: np.ndarray,
        out_dir: Path,
        *,
        emb2: np.ndarray,
        labels: np.ndarray | list | pd.Series,
        label_key: str,
        label_kind: LabelKind,
        **kw,
    ) -> None:
        """
        Generate plots for each unique label

        Plots:
        - ROC curves
        - UMAP visualizations
        - Similarity score histograms
        """
        labels = np.asarray(labels)
        uniq = np.unique(labels)

        # Compute UMAP once for all plots
        umap_emb = self._compute_umap(emb1)

        # Create subdirectories for different plot types
        roc_dir = out_dir / label_key / "roc_curves"
        umap_dir = out_dir / label_key / "umap"
        hist_dir = out_dir / label_key / "histograms"

        for d in [roc_dir, umap_dir, hist_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Plot mean ROC curve
        plt.figure(figsize=(6, 6))
        mean_tpr = np.zeros(100)
        mean_fpr = np.linspace(0, 1, 100)

        for v in uniq:
            mask = labels == v
            label_vec = emb2[mask][0]
            sim = self._pair_sim(emb1, label_vec)

            # Compute ROC
            fpr, tpr, roc_auc = self._compute_roc(sim, mask)
            mean_tpr += np.interp(mean_fpr, fpr, tpr)

            # Plot individual ROC
            safe_v = re.sub(r"[^\w\d\-\.]", "_", str(v))
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{v} ({label_kind.value})")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(roc_dir / f"{safe_v}.png", dpi=300)
            plt.close()

            # Plot UMAP
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(
                umap_emb[:, 0], umap_emb[:, 1], c=sim, cmap="RdBu_r", vmin=-1, vmax=1, s=10, alpha=0.6
            )
            plt.colorbar(scatter, label="Similarity Score")
            plt.title(f"{v} ({label_kind.value})")
            plt.tight_layout()
            plt.savefig(umap_dir / f"{safe_v}.png", dpi=300)
            plt.close()

            # Plot histogram with consistent scale
            plt.figure(figsize=(6, 4))
            plt.hist(sim[~mask], bins=self.bins, alpha=0.5, label="other", density=True, range=(-1, 1))
            plt.hist(sim[mask], bins=self.bins, alpha=0.7, label=str(v), density=True, range=(-1, 1))
            plt.xlabel(self.similarity)
            plt.ylabel("density")
            plt.xlim(-1, 1)
            plt.legend(frameon=False, fontsize=12)
            plt.title(f"{v} ({label_kind.value})")
            plt.tight_layout()
            plt.savefig(hist_dir / f"{safe_v}.png", dpi=300)
            plt.close()

        # Plot mean ROC curve
        mean_tpr /= len(uniq)
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(mean_fpr, mean_tpr, label=f"Mean AUC = {mean_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Mean ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_dir / "00_mean_roc.png", dpi=300)
        plt.close()

    @staticmethod
    def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
        nx, ny = len(x), len(y)
        vx, vy = x.var(ddof=1), y.var(ddof=1)
        pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
        return (x.mean() - y.mean()) / np.sqrt(pooled + 1e-8)
