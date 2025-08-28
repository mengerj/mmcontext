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


def _safe_tight_layout() -> None:
    """Safely apply tight_layout, suppressing warnings if it fails."""
    try:
        _safe_tight_layout()
    except Exception:
        # If tight_layout fails, just continue - bbox_inches="tight" in savefig will handle it
        pass


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
    skip_plotting: bool = False  # Skip all plotting for faster evaluation

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
            matching_indices = np.where(mask)[0]
            if len(matching_indices) == 0:
                logger.error(f"🚨 ERROR: No embeddings found for label '{v}'")
                raise ValueError(f"No embeddings found for label '{v}'")
            logger.debug(f"🔍 Label '{v}': found {len(matching_indices)} matching embeddings")
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

    def _add_non_overlapping_annotations(self, points: np.ndarray, labels: list, fontsize: int = 8) -> None:
        """
        Add text annotations with arrows to avoid overlap.

        Parameters
        ----------
        points : np.ndarray
            Array of (x, y) coordinates for label positions
        labels : list
            List of label strings
        fontsize : int
            Font size for the text
        """
        # Get plot bounds for relative offset calculation
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        # Much larger offset distance to avoid crowding
        base_offset = 0.25  # Increased from 0.15
        x_offset_base = base_offset * x_range
        y_offset_base = base_offset * y_range

        # Store chosen positions to avoid overlap
        chosen_positions = []

        # Define more angles for better distribution
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)  # 16 directions instead of 8

        for i, (point, label) in enumerate(zip(points, labels, strict=False)):
            # Add subtle dot at actual position - much smaller and less prominent
            plt.scatter(point[0], point[1], c="red", s=8, alpha=0.8, zorder=15, edgecolors="white", linewidths=0.5)

            best_pos = None
            max_min_distance = 0  # Find position with maximum minimum distance to others

            # Try different offset distances and angles
            for distance_multiplier in [1.0, 1.5, 2.0, 2.5]:  # Try multiple distances
                for angle in angles:
                    # Calculate offset position
                    x_offset = x_offset_base * distance_multiplier * np.cos(angle)
                    y_offset = y_offset_base * distance_multiplier * np.sin(angle)
                    test_pos = (point[0] + x_offset, point[1] + y_offset)

                    # Calculate minimum distance to all previously chosen positions
                    if chosen_positions:
                        min_dist_to_chosen = min(
                            np.sqrt((test_pos[0] - pos[0]) ** 2 + (test_pos[1] - pos[1]) ** 2)
                            for pos in chosen_positions
                        )
                    else:
                        min_dist_to_chosen = float("inf")

                    # Also check distance to original points
                    min_dist_to_points = (
                        min(
                            np.sqrt((test_pos[0] - other_point[0]) ** 2 + (test_pos[1] - other_point[1]) ** 2)
                            for j, other_point in enumerate(points)
                            if i != j
                        )
                        if len(points) > 1
                        else float("inf")
                    )

                    # Use the minimum of the two distances
                    min_dist = min(min_dist_to_chosen, min_dist_to_points)

                    # Keep the position with the largest minimum distance
                    if min_dist > max_min_distance:
                        max_min_distance = min_dist
                        best_pos = test_pos

            # Fallback if no good position found - use systematic distribution
            if best_pos is None:
                angle = (i / len(labels)) * 2 * np.pi
                x_offset = x_offset_base * 2.0 * np.cos(angle)  # Use larger multiplier
                y_offset = y_offset_base * 2.0 * np.sin(angle)
                best_pos = (point[0] + x_offset, point[1] + y_offset)

            chosen_positions.append(best_pos)

            # Add very subtle annotation - minimal visual impact
            plt.annotate(
                str(label),
                xy=(point[0], point[1]),  # Point to annotate
                xytext=best_pos,  # Text position
                fontsize=max(4, int(fontsize * 0.4)),  # Much smaller: 40% of legend font
                ha="center",
                va="center",
                weight="normal",
                bbox={
                    "boxstyle": "round,pad=0.05",
                    "facecolor": "white",
                    "alpha": 0.85,
                    "edgecolor": "gray",
                    "linewidth": 0.3,
                },  # Minimal box
                arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.6, "alpha": 0.7},  # Subtle line
                zorder=20,
            )

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
        label_embeddings: np.ndarray = None,
        combined_umap: np.ndarray = None,
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
            "label_embeddings": label_embeddings,
            "combined_umap": combined_umap,
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
        skip_plotting: bool = None,
        **kw,
    ) -> EvalResult:
        """Compute similarity scores and ROC metrics for each unique label."""
        # Override class attribute if parameter is provided
        if skip_plotting is not None:
            self.skip_plotting = skip_plotting

        labels = np.asarray(labels)
        uniq = np.unique(labels)

        # Try to load from cache first
        cache_path = None
        if self.cache_results and out_dir is not None:
            cache_path = out_dir / f"label_similarity_cache_{label_key}.pkl"
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

            # Also cache label embeddings and combined UMAP for text annotations
            unique_labels = np.unique(labels)
            label_embeddings = np.zeros((len(unique_labels), emb2.shape[1]))
            for i, label in enumerate(unique_labels):
                label_mask = labels == label
                label_embeddings[i] = emb2[label_mask][0]

            # Compute combined UMAP
            combined_embeddings = np.vstack([emb1, label_embeddings])
            combined_umap = self._compute_umap(combined_embeddings)

            self._save_cache(
                cache_path, similarity_matrix, labels, uniq, umap_emb, out, label_embeddings, combined_umap
            )

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
        skip_plotting: bool = None,
        **kw,
    ) -> None:
        """Generate plots for each unique label using cached similarity matrix if available."""
        # Override class attribute if parameter is provided
        if skip_plotting is not None:
            self.skip_plotting = skip_plotting

        if self.skip_plotting:
            logger.info("Skipping plotting due to skip_plotting flag.")
            return

        frameon = True  # hardcode for these plots for now
        labels = np.asarray(labels)
        uniq = np.unique(labels)

        # Try to load from cache first
        cache_path = out_dir.parent / f"label_similarity_cache_{label_key}.pkl"
        cache_data = self._load_cache(cache_path)

        if cache_data is not None:
            # Use cached data
            similarity_matrix = cache_data["similarity_matrix"]
            umap_emb = cache_data["umap_emb"]
        else:
            # Compute from scratch
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

                # Also cache label embeddings and combined UMAP for text annotations
                unique_labels = np.unique(labels)
                label_embeddings = np.zeros((len(unique_labels), emb2.shape[1]))
                for i, label in enumerate(unique_labels):
                    label_mask = labels == label
                    label_embeddings[i] = emb2[label_mask][0]

                # Compute combined UMAP
                combined_embeddings = np.vstack([emb1, label_embeddings])
                combined_umap = self._compute_umap(combined_embeddings)

                self._save_cache(
                    cache_path, similarity_matrix, labels, uniq, umap_emb, out, label_embeddings, combined_umap
                )

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
        roc_dir = out_dir / "roc_curves"
        umap_dir = out_dir / "umap"
        hist_dir = out_dir / "histograms"

        for d in [roc_dir, umap_dir, hist_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Compute predicted labels using the precomputed similarity matrix
        predicted_indices = np.argmax(similarity_matrix, axis=1)
        predicted_labels = uniq[predicted_indices]

        # Plot UMAP colored by true labels (with and without text annotations)
        unique_labels = np.unique(labels)

        # Use categorical color palettes for better distinction
        if len(unique_labels) <= 10:
            colors = sns.color_palette("tab10", len(unique_labels))
        elif len(unique_labels) <= 20:
            colors = sns.color_palette("tab20", len(unique_labels))
        elif len(unique_labels) <= 40:
            # For 21-40 classes, combine tab20 and tab20b
            colors = sns.color_palette("tab20", 20) + sns.color_palette("tab20b", len(unique_labels) - 20)
        else:
            # For >40 classes, use a continuous colormap to ensure we have enough colors
            # Use HSV colormap for maximum distinction between colors
            colors = sns.color_palette("hsv", len(unique_labels))

        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        point_colors = [color_map[label] for label in labels]

        # Get label embeddings for each unique label
        label_embeddings = np.zeros((len(unique_labels), emb2.shape[1]))
        for i, label in enumerate(unique_labels):
            label_mask = labels == label
            label_embeddings[i] = emb2[label_mask][0]  # first embedding for that label

        # Compute UMAP on combined cell + label embeddings
        combined_embeddings = np.vstack([emb1, label_embeddings])
        combined_umap = self._compute_umap(combined_embeddings)

        # Split back into cell and label coordinates
        cell_umap = combined_umap[: len(emb1)]
        label_umap = combined_umap[len(emb1) :]

        # Version 1: Without text annotations (using cell_umap which has label positions)
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)

        scatter = plt.scatter(cell_umap[:, 0], cell_umap[:, 1], c=point_colors, s=5, alpha=0.7, edgecolors="none")
        plt.xticks([])
        plt.yticks([])
        _safe_tight_layout()

        plt.savefig(umap_dir / f"00_true_labels.{save_format}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # Version 2: With text annotations
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)

        scatter = plt.scatter(cell_umap[:, 0], cell_umap[:, 1], c=point_colors, s=5, alpha=0.7, edgecolors="none")

        # Fix axis limits before adding annotations to prevent plot area from expanding
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Add text annotations with arrows to avoid overlap
        self._add_non_overlapping_annotations(label_umap, unique_labels, fontsize=max(6, legend_fontsize - 2))

        # Restore original axis limits to maintain consistent plot size
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        plt.xticks([])
        plt.yticks([])
        _safe_tight_layout()

        plt.savefig(umap_dir / f"00_true_labels_with_text.{save_format}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # Create and save separate legend
        fig_legend, ax_legend = plt.subplots(figsize=(min(len(unique_labels) * 1.2, 12), 1), dpi=dpi)
        ax_legend.axis("off")
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_map[label],
                markersize=8,
                label=str(label),
                markeredgecolor="none",
            )
            for label in unique_labels
        ]
        ax_legend.legend(
            handles=legend_elements,
            loc="center",
            fontsize=legend_fontsize,
            frameon=False,
            ncol=min(len(unique_labels), 6),
            bbox_to_anchor=(0.5, 0.5),
        )
        _safe_tight_layout()
        plt.savefig(umap_dir / f"00_true_labels_legend.{save_format}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # Plot UMAP colored by predicted labels (with and without text annotations)
        pred_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        pred_point_colors = [pred_color_map[label] for label in predicted_labels]

        # Version 1: Without text annotations
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)

        scatter = plt.scatter(cell_umap[:, 0], cell_umap[:, 1], c=pred_point_colors, s=5, alpha=0.7, edgecolors="none")
        plt.xticks([])
        plt.yticks([])
        _safe_tight_layout()

        plt.savefig(umap_dir / f"01_predicted_labels.{save_format}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # Version 2: With text annotations
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)

        scatter = plt.scatter(cell_umap[:, 0], cell_umap[:, 1], c=pred_point_colors, s=5, alpha=0.7, edgecolors="none")

        # Fix axis limits before adding annotations to prevent plot area from expanding
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Add text annotations with arrows to avoid overlap
        self._add_non_overlapping_annotations(label_umap, unique_labels, fontsize=max(6, legend_fontsize - 2))

        # Restore original axis limits to maintain consistent plot size
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        plt.xticks([])
        plt.yticks([])
        _safe_tight_layout()

        plt.savefig(umap_dir / f"01_predicted_labels_with_text.{save_format}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # Create and save separate legend (same as true labels since they use the same color scheme)
        fig_legend, ax_legend = plt.subplots(figsize=(min(len(unique_labels) * 1.2, 12), 1), dpi=dpi)
        ax_legend.axis("off")
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=pred_color_map[label],
                markersize=8,
                label=str(label),
                markeredgecolor="none",
            )
            for label in unique_labels
        ]
        ax_legend.legend(
            handles=legend_elements,
            loc="center",
            fontsize=legend_fontsize,
            frameon=False,
            ncol=min(len(unique_labels), 6),
            bbox_to_anchor=(0.5, 0.5),
        )
        _safe_tight_layout()
        plt.savefig(umap_dir / f"01_predicted_labels_legend.{save_format}", dpi=dpi, bbox_inches="tight")
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
            _safe_tight_layout()
            plt.savefig(roc_dir / f"{safe_v}.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()

            # Create and save separate legend for ROC curve
            fig_legend, ax_legend = plt.subplots(figsize=(3, 1), dpi=dpi)
            ax_legend.axis("off")
            legend_elements = [plt.Line2D([0], [0], color="tab:blue", linewidth=2, label=f"AUC = {roc_auc:.3f}")]
            ax_legend.legend(
                handles=legend_elements,
                loc="center",
                fontsize=legend_fontsize,
                frameon=False,
                bbox_to_anchor=(0.5, 0.5),
            )
            _safe_tight_layout()
            plt.savefig(roc_dir / f"{safe_v}_legend.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()

            # Plot UMAP
            plt.figure(figsize=figsize, dpi=dpi)
            # Create plot without frame
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Create linewidths to highlight true label cells with borders
            linewidths = np.where(mask, 0.3, 0.0)

            scatter = plt.scatter(
                umap_emb[:, 0],
                umap_emb[:, 1],
                c=sim,
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                s=5,
                alpha=0.7,
                edgecolors="black",
                linewidths=linewidths,
            )
            cbar = plt.colorbar(scatter, label="Similarity Score")
            cbar.set_ticks([-1, 0, 1])

            plt.xticks([])
            plt.yticks([])
            _safe_tight_layout()

            # Save plot without legend
            plt.savefig(umap_dir / f"{safe_v}.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()

            # Create and save separate legend for border meaning
            fig_legend, ax_legend = plt.subplots(figsize=(3, 1.5), dpi=dpi)
            ax_legend.axis("off")
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=(0.7, 0.7, 0.7, 0.6),
                    markeredgecolor="black",
                    markeredgewidth=1,
                    markersize=8,
                    label=f"True {v}",
                    linestyle="None",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=(0.7, 0.7, 0.7, 0.6),
                    markeredgecolor="none",
                    markersize=8,
                    label="Other cells",
                    linestyle="None",
                ),
            ]
            ax_legend.legend(
                handles=legend_elements,
                loc="center",
                frameon=True,
                fancybox=True,
                shadow=True,
                fontsize=legend_fontsize,
                ncol=1,
                bbox_to_anchor=(0.5, 0.5),
            )
            _safe_tight_layout()
            plt.savefig(umap_dir / f"{safe_v}_legend.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()

            # Plot histogram with consistent scale
            plt.figure(figsize=figsize, dpi=dpi)
            plt.hist(sim[~mask], bins=self.bins, alpha=0.5, label="other", density=True, range=(-1, 1))
            plt.hist(sim[mask], bins=self.bins, alpha=0.7, label=str(v), density=True, range=(-1, 1))
            plt.xlabel(f"{self.similarity.title()} Similarity")
            plt.ylabel("Density")
            plt.xlim(-1, 1)
            plt.title(f"{v} ({label_kind.value})")
            _safe_tight_layout()
            plt.savefig(hist_dir / f"{safe_v}.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()

            # Create and save separate legend for histogram
            fig_legend, ax_legend = plt.subplots(figsize=(3, 1), dpi=dpi)
            ax_legend.axis("off")
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor="tab:blue", alpha=0.5, label="other"),
                plt.Rectangle((0, 0), 1, 1, facecolor="tab:orange", alpha=0.7, label=str(v)),
            ]
            ax_legend.legend(
                handles=legend_elements,
                loc="center",
                fontsize=legend_fontsize,
                frameon=False,
                bbox_to_anchor=(0.5, 0.5),
            )
            _safe_tight_layout()
            plt.savefig(hist_dir / f"{safe_v}_legend.{save_format}", dpi=dpi, bbox_inches="tight")
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
        _safe_tight_layout()
        plt.savefig(roc_dir / f"00_mean_roc.{save_format}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # Create and save separate legend for mean ROC curve
        fig_legend, ax_legend = plt.subplots(figsize=(3, 1), dpi=dpi)
        ax_legend.axis("off")
        legend_elements = [plt.Line2D([0], [0], color="tab:blue", linewidth=2, label=f"Mean AUC = {mean_auc:.3f}")]
        ax_legend.legend(
            handles=legend_elements, loc="center", fontsize=legend_fontsize, frameon=False, bbox_to_anchor=(0.5, 0.5)
        )
        _safe_tight_layout()
        plt.savefig(roc_dir / f"00_mean_roc_legend.{save_format}", dpi=dpi, bbox_inches="tight")
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
        skip_plotting: bool = None,
        **kw,
    ) -> None:
        """
        Generate plots using only cached data (no embeddings required).

        This method allows you to regenerate plots without recomputing embeddings or similarity matrices.
        Useful for adjusting plot parameters or formats without rerunning the expensive computation.
        """
        # Override class attribute if parameter is provided
        if skip_plotting is not None:
            self.skip_plotting = skip_plotting

        if self.skip_plotting:
            logger.info("Skipping plotting due to skip_plotting flag.")
            return

        # Load cached data
        cache_path = out_dir / f"label_similarity_cache_{label_key}.pkl"
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
        roc_dir = out_dir / "roc_curves"
        umap_dir = out_dir / "umap"
        hist_dir = out_dir / "histograms"

        for d in [roc_dir, umap_dir, hist_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Compute predicted labels using the cached similarity matrix
        predicted_indices = np.argmax(similarity_matrix, axis=1)
        predicted_labels = uniq[predicted_indices]

        # Plot UMAP colored by true labels (with and without text annotations)
        unique_labels = np.unique(labels)

        # Use categorical color palettes for better distinction
        if len(unique_labels) <= 10:
            colors = sns.color_palette("tab10", len(unique_labels))
        elif len(unique_labels) <= 20:
            colors = sns.color_palette("tab20", len(unique_labels))
        elif len(unique_labels) <= 40:
            # For 21-40 classes, combine tab20 and tab20b
            colors = sns.color_palette("tab20", 20) + sns.color_palette("tab20b", len(unique_labels) - 20)
        else:
            # For >40 classes, use a continuous colormap to ensure we have enough colors
            # Use HSV colormap for maximum distinction between colors
            colors = sns.color_palette("hsv", len(unique_labels))

        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        point_colors = [color_map[label] for label in labels]

        # Get cached label embeddings and combined UMAP if available
        # cached_label_embeddings = cache_data.get("label_embeddings")
        cached_combined_umap = cache_data.get("combined_umap")

        if cached_combined_umap is not None:
            # Use cached combined UMAP coordinates
            cell_umap = cached_combined_umap[: len(labels)]
            label_umap = cached_combined_umap[len(labels) :]
        else:
            # Fallback to original UMAP (no text annotations possible)
            cell_umap = umap_emb
            label_umap = None

        # Version 1: Without text annotations
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)

        scatter = plt.scatter(cell_umap[:, 0], cell_umap[:, 1], c=point_colors, s=5, alpha=0.7, edgecolors="none")
        plt.xticks([])
        plt.yticks([])
        _safe_tight_layout()

        plt.savefig(umap_dir / f"00_true_labels.{save_format}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # Version 2: With text annotations (if cached data is available)
        if label_umap is not None:
            plt.figure(figsize=figsize, dpi=dpi)
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_visible(False)

            scatter = plt.scatter(cell_umap[:, 0], cell_umap[:, 1], c=point_colors, s=5, alpha=0.7, edgecolors="none")

            # Fix axis limits before adding annotations to prevent plot area from expanding
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Add text annotations with arrows to avoid overlap
            self._add_non_overlapping_annotations(label_umap, unique_labels, fontsize=max(6, legend_fontsize - 2))

            # Restore original axis limits to maintain consistent plot size
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            plt.xticks([])
            plt.yticks([])
            _safe_tight_layout()

            plt.savefig(umap_dir / f"00_true_labels_with_text.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()

        # Create and save separate legend
        fig_legend, ax_legend = plt.subplots(figsize=(min(len(unique_labels) * 1.2, 12), 1), dpi=dpi)
        ax_legend.axis("off")
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_map[label],
                markersize=8,
                label=str(label),
                markeredgecolor="none",
            )
            for label in unique_labels
        ]
        ax_legend.legend(
            handles=legend_elements,
            loc="center",
            fontsize=legend_fontsize,
            frameon=False,
            ncol=min(len(unique_labels), 6),
            bbox_to_anchor=(0.5, 0.5),
        )
        _safe_tight_layout()
        plt.savefig(umap_dir / f"00_true_labels_legend.{save_format}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # Plot UMAP colored by predicted labels (with and without text annotations)
        pred_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        pred_point_colors = [pred_color_map[label] for label in predicted_labels]

        # Version 1: Without text annotations
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)

        scatter = plt.scatter(cell_umap[:, 0], cell_umap[:, 1], c=pred_point_colors, s=5, alpha=0.7, edgecolors="none")
        plt.xticks([])
        plt.yticks([])
        _safe_tight_layout()

        plt.savefig(umap_dir / f"01_predicted_labels.{save_format}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # Version 2: With text annotations (if cached data is available)
        if label_umap is not None:
            plt.figure(figsize=figsize, dpi=dpi)
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_visible(False)

            scatter = plt.scatter(
                cell_umap[:, 0], cell_umap[:, 1], c=pred_point_colors, s=5, alpha=0.7, edgecolors="none"
            )

            # Fix axis limits before adding annotations to prevent plot area from expanding
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Add text annotations with arrows to avoid overlap
            self._add_non_overlapping_annotations(label_umap, unique_labels, fontsize=max(6, legend_fontsize - 2))

            # Restore original axis limits to maintain consistent plot size
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            plt.xticks([])
            plt.yticks([])
            _safe_tight_layout()

            plt.savefig(umap_dir / f"01_predicted_labels_with_text.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()

        # Create and save separate legend (same as true labels since they use the same color scheme)
        _fig_legend, ax_legend = plt.subplots(figsize=(min(len(unique_labels) * 1.2, 12), 1), dpi=dpi)
        ax_legend.axis("off")
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=pred_color_map[label],
                markersize=8,
                label=str(label),
                markeredgecolor="none",
            )
            for label in unique_labels
        ]
        ax_legend.legend(
            handles=legend_elements,
            loc="center",
            fontsize=legend_fontsize,
            frameon=False,
            ncol=min(len(unique_labels), 6),
            bbox_to_anchor=(0.5, 0.5),
        )
        _safe_tight_layout()
        plt.savefig(umap_dir / f"01_predicted_labels_legend.{save_format}", dpi=dpi, bbox_inches="tight")
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
            _safe_tight_layout()
            plt.savefig(roc_dir / f"{safe_v}.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()

            # Create and save separate legend for ROC curve
            fig_legend, ax_legend = plt.subplots(figsize=(3, 1), dpi=dpi)
            ax_legend.axis("off")
            legend_elements = [plt.Line2D([0], [0], color="tab:blue", linewidth=2, label=f"AUC = {roc_auc:.3f}")]
            ax_legend.legend(
                handles=legend_elements,
                loc="center",
                fontsize=legend_fontsize,
                frameon=False,
                bbox_to_anchor=(0.5, 0.5),
            )
            _safe_tight_layout()
            plt.savefig(roc_dir / f"{safe_v}_legend.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()

            # Plot UMAP
            plt.figure(figsize=figsize, dpi=dpi)
            # Create plot without frame
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Create linewidths to highlight true label cells with borders
            linewidths = np.where(mask, 0.3, 0.0)

            scatter = plt.scatter(
                umap_emb[:, 0],
                umap_emb[:, 1],
                c=sim,
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                s=5,
                alpha=0.7,
                edgecolors="black",
                linewidths=linewidths,
            )
            cbar = plt.colorbar(scatter, label="Similarity Score")
            cbar.set_ticks([-1, 0, 1])

            plt.xticks([])
            plt.yticks([])
            _safe_tight_layout()

            # Save plot without legend
            plt.savefig(umap_dir / f"{safe_v}.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()

            # Create and save separate legend for border meaning
            fig_legend, ax_legend = plt.subplots(figsize=(3, 1.5), dpi=dpi)
            ax_legend.axis("off")
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=(0.7, 0.7, 0.7, 0.6),
                    markeredgecolor="black",
                    markeredgewidth=1,
                    markersize=16,
                    label=f"True {v}",
                    linestyle="None",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=(0.7, 0.7, 0.7, 0.6),
                    markeredgecolor="none",
                    markersize=16,
                    label="Other cells",
                    linestyle="None",
                ),
            ]
            ax_legend.legend(
                handles=legend_elements,
                loc="center",
                frameon=True,
                fancybox=True,
                shadow=True,
                fontsize=legend_fontsize,
                ncol=1,
                bbox_to_anchor=(0.5, 0.5),
            )
            _safe_tight_layout()
            plt.savefig(umap_dir / f"{safe_v}_legend.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()

            # Plot histogram with consistent scale
            plt.figure(figsize=figsize, dpi=dpi)
            plt.hist(sim[~mask], bins=self.bins, alpha=0.5, label="other", density=True, range=(-1, 1))
            plt.hist(sim[mask], bins=self.bins, alpha=0.7, label=str(v), density=True, range=(-1, 1))
            plt.xlabel(f"{self.similarity.title()} Similarity")
            plt.ylabel("Density")
            plt.xlim(-1, 1)
            plt.title(f"{v} ({label_kind.value})")
            _safe_tight_layout()
            plt.savefig(hist_dir / f"{safe_v}.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()

            # Create and save separate legend for histogram
            fig_legend, ax_legend = plt.subplots(figsize=(3, 1), dpi=dpi)
            ax_legend.axis("off")
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor="tab:blue", alpha=0.5, label="other"),
                plt.Rectangle((0, 0), 1, 1, facecolor="tab:orange", alpha=0.7, label=str(v)),
            ]
            ax_legend.legend(
                handles=legend_elements,
                loc="center",
                fontsize=legend_fontsize,
                frameon=False,
                bbox_to_anchor=(0.5, 0.5),
            )
            _safe_tight_layout()
            plt.savefig(hist_dir / f"{safe_v}_legend.{save_format}", dpi=dpi, bbox_inches="tight")
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
        _safe_tight_layout()
        plt.savefig(roc_dir / f"00_mean_roc.{save_format}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # Create and save separate legend for mean ROC curve
        fig_legend, ax_legend = plt.subplots(figsize=(3, 1), dpi=dpi)
        ax_legend.axis("off")
        legend_elements = [plt.Line2D([0], [0], color="tab:blue", linewidth=2, label=f"Mean AUC = {mean_auc:.3f}")]
        ax_legend.legend(
            handles=legend_elements, loc="center", fontsize=legend_fontsize, frameon=False, bbox_to_anchor=(0.5, 0.5)
        )
        _safe_tight_layout()
        plt.savefig(roc_dir / f"00_mean_roc_legend.{save_format}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # Reset matplotlib parameters to default
        plt.rcParams.update(plt.rcParamsDefault)

    @staticmethod
    def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
        nx, ny = len(x), len(y)
        vx, vy = x.var(ddof=1), y.var(ddof=1)
        pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
        return (x.mean() - y.mean()) / np.sqrt(pooled + 1e-8)
