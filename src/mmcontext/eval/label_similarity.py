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

    Additionally computes:
        • Accuracy by finding the label with highest similarity for each cell
        • Baseline random accuracy based on label distribution
        • Standard deviation of AUC scores

    Returns
    -------
        - AUC score for each label
        - Mean AUC across all labels
        - Standard deviation of AUC scores
        - Accuracy score (ratio of correct assignments)
        - Random baseline accuracy
        - Accuracy over random baseline ratio

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

    def _compute_accuracy(self, emb1: np.ndarray, emb2: np.ndarray, labels: np.ndarray, uniq: np.ndarray) -> float:
        """
        Compute accuracy by finding the label with highest similarity for each cell.

        Parameters
        ----------
        emb1 : np.ndarray
            Cell embeddings (N x D)
        emb2 : np.ndarray
            Label embeddings (M x D)
        labels : np.ndarray
            True labels for each cell (N,)
        uniq : np.ndarray
            Unique labels

        Returns
        -------
        float
            Accuracy score (ratio of correct assignments)
        """
        # Create a similarity matrix: cells x labels
        similarity_matrix = np.zeros((len(emb1), len(uniq)))

        # Get label prototype for each unique label
        label_prototypes = {}
        for i, v in enumerate(uniq):
            mask = labels == v
            label_prototypes[v] = emb2[mask][0]  # first row for that value
            similarity_matrix[:, i] = self._pair_sim(emb1, label_prototypes[v])

        # Find the label with highest similarity for each cell
        predicted_indices = np.argmax(similarity_matrix, axis=1)
        predicted_labels = uniq[predicted_indices]

        # Compute accuracy
        correct = np.sum(predicted_labels == labels)
        accuracy = correct / len(labels)

        return accuracy

    def _compute_random_baseline_accuracy(self, labels: np.ndarray) -> float:
        """
        Compute baseline random accuracy based on label distribution.

        For random assignment, accuracy equals the sum of squared proportions
        of each label (probability of correct assignment by chance).

        Parameters
        ----------
        labels : np.ndarray
            True labels

        Returns
        -------
        float
            Baseline random accuracy
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        proportions = counts / len(labels)
        # Random baseline accuracy is sum of squared proportions
        random_accuracy = np.sum(proportions**2)
        return random_accuracy

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

        # Compute accuracy metrics
        accuracy = self._compute_accuracy(emb1, emb2, labels, uniq)
        random_baseline = self._compute_random_baseline_accuracy(labels)

        # Add mean AUC, standard deviation, and accuracy metrics
        out["mean_auc"] = float(np.mean(auc_scores))
        out["std_auc"] = float(np.std(auc_scores))
        out["accuracy"] = float(accuracy)
        out["random_baseline_accuracy"] = float(random_baseline)
        out["accuracy_over_random"] = float(accuracy / random_baseline) if random_baseline > 0 else 0.0
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
        save_format: str = "png",
        figsize: tuple = (6, 6),
        dpi: int = 300,
        font_size: int = 12,
        font_style: str = "normal",
        font_weight: str = "normal",
        legend_fontsize: int = 10,
        axis_label_size: int = 12,
        axis_tick_size: int = 10,
        frameon: bool = False,
        **kw,
    ) -> None:
        """
        Generate plots for each unique label

        Plots:
        - ROC curves
        - UMAP visualizations
        - Similarity score histograms

        Parameters
        ----------
        emb1 : np.ndarray
            Cell embeddings
        out_dir : Path
            Output directory for plots
        emb2 : np.ndarray
            Label embeddings
        labels : np.ndarray | list | pd.Series
            Ground truth labels
        label_key : str
            Key for the labels
        label_kind : LabelKind
            Type of label (bio or batch)
        save_format : str, optional
            Format to save plots (default: "png")
        figsize : tuple, optional
            Figure size (default: (6, 6))
        dpi : int, optional
            DPI for saved plots (default: 300)
        font_size : int, optional
            General font size (default: 12)
        font_style : str, optional
            Font style (default: "normal")
        font_weight : str, optional
            Font weight (default: "normal")
        legend_fontsize : int, optional
            Font size for legends (default: 10)
        axis_label_size : int, optional
            Size of axis labels (default: 12)
        axis_tick_size : int, optional
            Size of axis tick labels (default: 10)
        frameon : bool, optional
            Whether to show frame around plots (default: False)
        **kw
            Additional keyword arguments
        """
        frameon = True  # hardcode for these plots for now
        labels = np.asarray(labels)
        uniq = np.unique(labels)

        # Configure matplotlib with the provided parameters
        plt.rcParams.update(
            {
                "font.size": font_size,
                "font.weight": font_weight,
                "font.style": font_style,
                "axes.labelsize": axis_label_size,
                "xtick.labelsize": axis_tick_size,
                "ytick.labelsize": axis_tick_size,
                "legend.fontsize": legend_fontsize,
                "axes.spines.left": frameon,
                "axes.spines.bottom": frameon,
                "axes.spines.top": frameon,
                "axes.spines.right": frameon,
            }
        )

        # Compute UMAP once for all plots
        umap_emb = self._compute_umap(emb1)

        # Create subdirectories for different plot types
        roc_dir = out_dir / label_key / "roc_curves"
        umap_dir = out_dir / label_key / "umap"
        hist_dir = out_dir / label_key / "histograms"

        for d in [roc_dir, umap_dir, hist_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Plot mean ROC curve
        plt.figure(figsize=figsize, dpi=dpi)
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
            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
            plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{v} ({label_kind.value})")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(roc_dir / f"{safe_v}.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()

            # Plot UMAP
            plt.figure(figsize=figsize, dpi=dpi)
            scatter = plt.scatter(
                umap_emb[:, 0], umap_emb[:, 1], c=sim, cmap="RdBu_r", vmin=-1, vmax=1, s=10, alpha=0.6
            )
            plt.colorbar(scatter, label="Similarity Score")
            plt.title(f"{v} ({label_kind.value})")
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.tight_layout()
            plt.savefig(umap_dir / f"{safe_v}.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()

            # Plot histogram with consistent scale
            plt.figure(figsize=figsize, dpi=dpi)
            plt.hist(sim[~mask], bins=self.bins, alpha=0.5, label="other", density=True, range=(-1, 1))
            plt.hist(sim[mask], bins=self.bins, alpha=0.7, label=str(v), density=True, range=(-1, 1))
            plt.xlabel(f"{self.similarity.title()} Similarity")
            plt.ylabel("Density")
            plt.xlim(-1, 1)
            plt.legend(frameon=frameon)
            plt.title(f"{v} ({label_kind.value})")
            plt.tight_layout()
            plt.savefig(hist_dir / f"{safe_v}.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()

        # Plot mean ROC curve
        mean_tpr /= len(uniq)
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(mean_fpr, mean_tpr, label=f"Mean AUC = {mean_auc:.3f}", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Mean ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_dir / f"00_mean_roc.{save_format}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # Reset matplotlib parameters to default
        plt.rcParams.update(plt.rcParamsDefault)

    @staticmethod
    def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
        nx, ny = len(x), len(y)
        vx, vy = x.var(ddof=1), y.var(ddof=1)
        pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
        return (x.mean() - y.mean()) / np.sqrt(pooled + 1e-8)
