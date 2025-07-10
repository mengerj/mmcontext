# mmcontext/eval/label_retrieval.py
from __future__ import annotations

import logging
import pickle
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


def _cosine_matrix(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between all pairs of embeddings."""
    emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
    emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
    return (emb1_norm @ emb2_norm.T).astype(np.float32)


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
    cache_results: bool = True  # Enable caching of similarity matrices

    def _compute_similarity_matrix(
        self, emb1: np.ndarray, emb2: np.ndarray, labels: np.ndarray, uniq: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity matrix between cell embeddings and label prototypes.

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
        np.ndarray
            Similarity matrix (N x len(uniq))
        """
        # Get label prototypes for each unique label
        label_prototypes = np.zeros((len(uniq), emb2.shape[1]))
        for i, v in enumerate(uniq):
            mask = labels == v
            label_prototypes[i] = emb2[mask][0]  # first row for that value

        # Compute similarity matrix using vectorized operations
        if self.similarity == "cosine":
            return _cosine_matrix(emb1, label_prototypes)
        elif self.similarity == "dot":
            return emb1 @ label_prototypes.T
        else:
            raise ValueError(f"Unknown similarity: {self.similarity}")

    def _pair_sim(self, emb1: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Legacy method for backward compatibility."""
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

    def _compute_accuracy_from_matrix(
        self, similarity_matrix: np.ndarray, labels: np.ndarray, uniq: np.ndarray
    ) -> float:
        """
        Compute accuracy using precomputed similarity matrix.

        Parameters
        ----------
        similarity_matrix : np.ndarray
            Precomputed similarity matrix (N x len(uniq))
        labels : np.ndarray
            True labels for each cell (N,)
        uniq : np.ndarray
            Unique labels

        Returns
        -------
        float
            Accuracy score (ratio of correct assignments)
        """
        # Find the label with highest similarity for each cell
        predicted_indices = np.argmax(similarity_matrix, axis=1)
        predicted_labels = uniq[predicted_indices]

        # Compute accuracy
        correct = np.sum(predicted_labels == labels)
        accuracy = correct / len(labels)

        return accuracy

    def _compute_accuracy(self, emb1: np.ndarray, emb2: np.ndarray, labels: np.ndarray, uniq: np.ndarray) -> float:
        """
        Compute accuracy by finding the label with highest similarity for each cell.

        Legacy method - kept for backward compatibility.
        """
        similarity_matrix = self._compute_similarity_matrix(emb1, emb2, labels, uniq)
        return self._compute_accuracy_from_matrix(similarity_matrix, labels, uniq)

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

    def _save_cache(
        self,
        cache_path: Path,
        similarity_matrix: np.ndarray,
        labels: np.ndarray,
        uniq: np.ndarray,
        umap_emb: np.ndarray,
        results: dict,
    ) -> None:
        """Save computed results to cache for faster plotting."""
        if not self.cache_results:
            return

        cache_data = {
            "similarity_matrix": similarity_matrix,
            "labels": labels,
            "uniq": uniq,
            "umap_emb": umap_emb,
            "results": results,
            "similarity_method": self.similarity,
            "bins": self.bins,
            "umap_params": {
                "n_neighbors": self.umap_n_neighbors,
                "min_dist": self.umap_min_dist,
                "random_state": self.umap_random_state,
            },
        }

        # Create cache directory
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)

        logger.info(f"Cached results saved to {cache_path}")

    def _load_cache(self, cache_path: Path) -> dict | None:
        """Load cached results if available."""
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)

            # Verify cache compatibility
            if (
                cache_data.get("similarity_method") != self.similarity
                or cache_data.get("bins") != self.bins
                or cache_data.get("umap_params", {}).get("n_neighbors") != self.umap_n_neighbors
                or cache_data.get("umap_params", {}).get("min_dist") != self.umap_min_dist
                or cache_data.get("umap_params", {}).get("random_state") != self.umap_random_state
            ):
                logger.warning("Cache parameters don't match current settings, ignoring cache")
                return None

            logger.info(f"Loaded cached results from {cache_path}")
            return cache_data

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def compute(
        self,
        emb1: np.ndarray,
        *,
        emb2: np.ndarray,
        labels: np.ndarray | list | pd.Series,
        label_key: str,
        label_kind: LabelKind,
        out_dir: Path = None,
        **kw,
    ) -> EvalResult:
        """Compute similarity scores and ROC metrics for each unique label."""
        labels = np.asarray(labels)
        uniq = np.unique(labels)

        # Try to load from cache first
        cache_path = None
        if self.cache_results and out_dir is not None:
            cache_path = out_dir / "label_similarity_cache.pkl"
            cache_data = self._load_cache(cache_path)
            if cache_data is not None:
                logger.info("Using cached similarity matrix")
                return EvalResult(**cache_data["results"])

        # Compute similarity matrix once using vectorized operations
        similarity_matrix = self._compute_similarity_matrix(emb1, emb2, labels, uniq)

        # Compute results using the precomputed matrix
        out = {}
        auc_scores = []

        for i, v in enumerate(uniq):
            mask = labels == v
            sim = similarity_matrix[:, i]  # Extract similarity scores for this label

            # Compute ROC curve and AUC
            _, _, roc_auc = self._compute_roc(sim, mask)
            auc_scores.append(roc_auc)

            prefix = f"{v}"
            out[f"{prefix}/auc"] = float(roc_auc)

        # Compute accuracy metrics using vectorized operations
        accuracy = self._compute_accuracy_from_matrix(similarity_matrix, labels, uniq)
        random_baseline = self._compute_random_baseline_accuracy(labels)

        # Add mean AUC, standard deviation, and accuracy metrics
        out["mean_auc"] = float(np.mean(auc_scores))
        out["std_auc"] = float(np.std(auc_scores))
        out["accuracy"] = float(accuracy)
        out["random_baseline_accuracy"] = float(random_baseline)
        out["accuracy_over_random"] = float(accuracy / random_baseline) if random_baseline > 0 else 0.0
        out["label_kind"] = label_kind.value
        out["n_labels"] = len(uniq)

        # Cache results if enabled
        if cache_path is not None:
            # Also compute UMAP for caching
            umap_emb = self._compute_umap(emb1)
            self._save_cache(cache_path, similarity_matrix, labels, uniq, umap_emb, out)

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
        """Generate plots for each unique label using cached similarity matrix if available."""
        frameon = True  # hardcode for these plots for now
        labels = np.asarray(labels)
        uniq = np.unique(labels)

        # Try to load from cache first
        cache_path = out_dir / "label_similarity_cache.pkl"
        cache_data = self._load_cache(cache_path)

        if cache_data is not None:
            # Use cached data
            similarity_matrix = cache_data["similarity_matrix"]
            umap_emb = cache_data["umap_emb"]
            logger.info("Using cached similarity matrix and UMAP for plotting")
        else:
            # Compute from scratch
            logger.info("Computing similarity matrix and UMAP for plotting")
            similarity_matrix = self._compute_similarity_matrix(emb1, emb2, labels, uniq)
            umap_emb = self._compute_umap(emb1)

            # Save to cache for future use
            if self.cache_results:
                # We need to compute the results again for caching
                out = {}
                auc_scores = []
                for i, v in enumerate(uniq):
                    mask = labels == v
                    sim = similarity_matrix[:, i]
                    _, _, roc_auc = self._compute_roc(sim, mask)
                    auc_scores.append(roc_auc)
                    out[f"{v}/auc"] = float(roc_auc)

                accuracy = self._compute_accuracy_from_matrix(similarity_matrix, labels, uniq)
                random_baseline = self._compute_random_baseline_accuracy(labels)
                out["mean_auc"] = float(np.mean(auc_scores))
                out["std_auc"] = float(np.std(auc_scores))
                out["accuracy"] = float(accuracy)
                out["random_baseline_accuracy"] = float(random_baseline)
                out["accuracy_over_random"] = float(accuracy / random_baseline) if random_baseline > 0 else 0.0
                out["label_kind"] = label_kind.value
                out["n_labels"] = len(uniq)

                self._save_cache(cache_path, similarity_matrix, labels, uniq, umap_emb, out)

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

        # Create subdirectories for different plot types
        roc_dir = out_dir / label_key / "roc_curves"
        umap_dir = out_dir / label_key / "umap"
        hist_dir = out_dir / label_key / "histograms"

        for d in [roc_dir, umap_dir, hist_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Compute predicted labels using the precomputed similarity matrix
        predicted_indices = np.argmax(similarity_matrix, axis=1)
        predicted_labels = uniq[predicted_indices]

        # Plot UMAP colored by true labels
        plt.figure(figsize=(8, 6))
        # Use seaborn/scanpy style colors instead of Set3
        unique_labels = np.unique(labels)
        colors = sns.color_palette("husl", len(unique_labels))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        point_colors = [color_map[label] for label in labels]

        # Create plot without frame
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)

        scatter = plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c=point_colors, s=5, alpha=0.7, edgecolors="none")
        plt.title(f"UMAP - True Labels ({label_kind.value})", fontweight="bold")
        plt.xticks([])
        plt.yticks([])

        # Create cleaner legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_map[label],
                markersize=6,
                label=str(label),
                markeredgecolor="none",
            )
            for label in unique_labels
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10, frameon=False)
        plt.tight_layout()
        plt.savefig(umap_dir / "true_labels.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Plot UMAP colored by predicted labels
        plt.figure(figsize=(8, 6))
        pred_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        pred_point_colors = [pred_color_map[label] for label in predicted_labels]

        # Create plot without frame
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)

        scatter = plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c=pred_point_colors, s=5, alpha=0.7, edgecolors="none")
        plt.title(f"UMAP - Predicted Labels ({label_kind.value})", fontweight="bold")
        plt.xticks([])
        plt.yticks([])

        # Create cleaner legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=pred_color_map[label],
                markersize=6,
                label=str(label),
                markeredgecolor="none",
            )
            for label in unique_labels
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10, frameon=False)
        plt.tight_layout()
        plt.savefig(umap_dir / "predicted_labels.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Plot mean ROC curve
        plt.figure(figsize=figsize, dpi=dpi)
        mean_tpr = np.zeros(100)
        mean_fpr = np.linspace(0, 1, 100)

        # Use precomputed similarity matrix for all plots
        for i, v in enumerate(uniq):
            mask = labels == v
            sim = similarity_matrix[:, i]  # Extract similarity scores for this label

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
            plt.figure(figsize=(8, 6))
            # Create plot without frame
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_visible(False)

            scatter = plt.scatter(
                umap_emb[:, 0], umap_emb[:, 1], c=sim, cmap="RdBu_r", vmin=-1, vmax=1, s=5, alpha=0.7, edgecolors="none"
            )
            plt.colorbar(scatter, label="Similarity Score")
            plt.title(f"{v} ({label_kind.value})", fontweight="bold")
            plt.xticks([])
            plt.yticks([])
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

    def plot_only(
        self,
        out_dir: Path,
        *,
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
        Generate plots using only cached data (no embeddings required).

        This method allows you to regenerate plots without recomputing embeddings or similarity matrices.
        Useful for adjusting plot parameters or formats without rerunning the expensive computation.
        """
        # Load cached data
        cache_path = out_dir / "label_similarity_cache.pkl"
        cache_data = self._load_cache(cache_path)

        if cache_data is None:
            raise ValueError(f"No cached data found at {cache_path}. Run compute() first.")

        # Extract cached data
        similarity_matrix = cache_data["similarity_matrix"]
        labels = cache_data["labels"]
        uniq = cache_data["uniq"]
        umap_emb = cache_data["umap_emb"]

        logger.info("Generating plots using cached data only")

        # Use the same plotting code but with cached data
        # This is essentially the same as the plot() method but without computing embeddings
        frameon = True  # hardcode for these plots for now

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

        # Create subdirectories for different plot types
        roc_dir = out_dir / label_key / "roc_curves"
        umap_dir = out_dir / label_key / "umap"
        hist_dir = out_dir / label_key / "histograms"

        for d in [roc_dir, umap_dir, hist_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Compute predicted labels using the cached similarity matrix
        predicted_indices = np.argmax(similarity_matrix, axis=1)
        predicted_labels = uniq[predicted_indices]

        # Plot UMAP colored by true labels
        plt.figure(figsize=(8, 6))
        # Use seaborn/scanpy style colors instead of Set3
        unique_labels = np.unique(labels)
        colors = sns.color_palette("husl", len(unique_labels))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        point_colors = [color_map[label] for label in labels]

        # Create plot without frame
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)

        scatter = plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c=point_colors, s=5, alpha=0.7, edgecolors="none")
        plt.title(f"UMAP - True Labels ({label_kind.value})", fontweight="bold")
        plt.xticks([])
        plt.yticks([])

        # Create cleaner legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_map[label],
                markersize=6,
                label=str(label),
                markeredgecolor="none",
            )
            for label in unique_labels
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10, frameon=False)
        plt.tight_layout()
        plt.savefig(umap_dir / "true_labels.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Plot UMAP colored by predicted labels
        plt.figure(figsize=(8, 6))
        pred_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        pred_point_colors = [pred_color_map[label] for label in predicted_labels]

        # Create plot without frame
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)

        scatter = plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c=pred_point_colors, s=5, alpha=0.7, edgecolors="none")
        plt.title(f"UMAP - Predicted Labels ({label_kind.value})", fontweight="bold")
        plt.xticks([])
        plt.yticks([])

        # Create cleaner legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=pred_color_map[label],
                markersize=6,
                label=str(label),
                markeredgecolor="none",
            )
            for label in unique_labels
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10, frameon=False)
        plt.tight_layout()
        plt.savefig(umap_dir / "predicted_labels.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Plot mean ROC curve
        plt.figure(figsize=figsize, dpi=dpi)
        mean_tpr = np.zeros(100)
        mean_fpr = np.linspace(0, 1, 100)

        # Use cached similarity matrix for all plots
        for i, v in enumerate(uniq):
            mask = labels == v
            sim = similarity_matrix[:, i]  # Extract similarity scores for this label

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
            plt.figure(figsize=(8, 6))
            # Create plot without frame
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_visible(False)

            scatter = plt.scatter(
                umap_emb[:, 0], umap_emb[:, 1], c=sim, cmap="RdBu_r", vmin=-1, vmax=1, s=5, alpha=0.7, edgecolors="none"
            )
            plt.colorbar(scatter, label="Similarity Score")
            plt.title(f"{v} ({label_kind.value})", fontweight="bold")
            plt.xticks([])
            plt.yticks([])
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
