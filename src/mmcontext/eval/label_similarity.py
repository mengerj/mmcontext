# mmcontext/eval/label_similarity.py
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
from sklearn.metrics import auc, confusion_matrix, roc_curve

from .base import BaseEvaluator, EvalResult
from .registry import register

logger = logging.getLogger(__name__)

'''
class LabelSimilarityResult:
    """
    Simple result container for label similarity evaluation.

    This class stores evaluation metrics in a dictionary-like format
    with a clean string representation for display.
    """

    def __init__(self, **kwargs):
        """Initialize with evaluation metrics."""
        self._data = kwargs

    def __getitem__(self, key):
        """Allow dictionary-like access."""
        return self._data[key]

    def __setitem__(self, key, value):
        """Allow dictionary-like assignment."""
        self._data[key] = value

    def __contains__(self, key):
        """Check if key exists."""
        return key in self._data

    def keys(self):
        """Return keys."""
        return self._data.keys()

    def items(self):
        """Return items."""
        return self._data.items()

    def get(self, key, default=None):
        """Get value with default."""
        return self._data.get(key, default)

    def __repr__(self):
        """Pretty string representation."""
        lines = []
        for k, v in self._data.items():
            if isinstance(v, int | np.integer):
                lines.append(f"{k}: {v}")
            elif isinstance(v, float | np.floating):
                lines.append(f"{k}: {v:0.4f}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)
'''


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
        plt.tight_layout()
    except Exception:
        # If tight_layout fails, just continue - bbox_inches="tight" in savefig will handle it
        pass


@register
class LabelSimilarity(BaseEvaluator):
    """
    Label Similarity Evaluator for Cell-Type Classification.

    This class evaluates how well cell embeddings can be matched to their true labels
    by computing similarity scores between cells and label embeddings. It provides
    comprehensive evaluation metrics and visualizations for cell-type classification tasks.

    Key Features:
    - Computes similarity scores between cell embeddings and query label embeddings
    - Generates ROC curves and AUC scores for each label
    - Creates UMAP visualizations colored by true/predicted labels
    - Plots similarity score distributions as histograms
    - Computes accuracy metrics and confusion matrices
    - Provides automatic label filtering to remove noise labels (enabled by default)
    - Two-stage filtering: detects bimodal separation and rare meaningful labels
    - Allows inspection of which labels were used vs. filtered out

    Usage Example:
    ```python
    # Initialize evaluator with automatic label filtering (default)
    evaluator = LabelSimilarity(auto_filter_labels=True)

    # Compute evaluation metrics (only meaningful labels will be used)
    results = evaluator.compute(
        omics_embeddings=cell_embeddings,  # Shape: (N_cells, embedding_dim)
        label_embeddings=label_embeddings,  # Shape: (N_labels, embedding_dim)
        query_labels=["T_cell", "B_cell", "apple", "car"],  # Mix of meaningful and noise labels
        true_labels=cell_labels,  # True labels for each cell
        label_key="cell_types",  # Identifier for this evaluation
        out_dir=Path("./results"),  # Output directory for plots
    )

    # Check which labels were actually used
    used_labels = evaluator.get_used_labels()
    print(f"Used labels: {used_labels}")  # Will only include meaningful labels

    # Check filtering results
    filtering_results = evaluator.get_filtering_results()
    print(f"Bimodal labels: {filtering_results['bimodal_labels']}")
    print(f"Rare labels: {filtering_results['rare_labels']}")
    print(f"Remaining labels: {filtering_results['remaining_labels']}")

    # Generate plots (only for meaningful labels)
    evaluator.plot(
        omics_embeddings=cell_embeddings,
        label_embeddings=label_embeddings,
        query_labels=["T_cell", "B_cell", "apple", "car"],  # Same as before
        true_labels=cell_labels,
        label_key="cell_types",
        out_dir=Path("./results"),
    )

    # If you're sure all labels are meaningful, disable filtering
    evaluator_no_filter = LabelSimilarity(auto_filter_labels=False)
    ```

    Returns
    -------
    LabelSimilarityResult
        Dictionary-like object containing:
        - AUC scores for each label
        - Mean AUC across all labels
        - Accuracy metrics (overall, balanced, per-label)
        - Random baseline accuracy
        - Number of labels evaluated
    """

    name = "LabelSimilarity"
    requires_pair = True
    produces_plot = True

    def __init__(
        self,
        auto_filter_labels: bool = False,
        bimodal_threshold: float = 0.1,
        rare_threshold: float = 2.0,
        similarity: str = "cosine",
        logit_scale: float | None = None,
        score_norm_method: str | None = None,
        bins: int = 40,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_random_state: int = 42,
        umap_method: str = "combined",
        point_size: int = 5,
        text_plot_enlargement_factor: float = 1.5,
        legend_layout: str = "horizontal",
        legend_point_size: int = 8,
        text_dot_fill_color: str = "label",
        text_dot_edge_color: str = "black",
        text_dot_alpha: float = 0.8,
        skip_plotting: bool = False,
        save_labels: bool = True,
        output_sample_size: int = 100,
        **kw,
    ):
        """
        Initialize LabelSimilarity evaluator.

        Parameters
        ----------
        auto_filter_labels : bool, default True
            Whether to automatically filter query labels before computation.
            If True, only meaningful labels (detected by bimodality) will be used.
            If False, all provided labels will be used (use only if you're sure all labels are meaningful).
        bimodal_threshold : float, default 0.1
            Threshold for separation_score in label filtering (stage 1)
        rare_threshold : float, default 2.0
            Threshold for rare_label_score in label filtering (stage 2)
        similarity : str, default "cosine"
            Similarity metric ("cosine" or "dot")
        logit_scale : float, optional
            Scale factor to multiply similarity scores before normalization.
            If None, defaults to 1.0 (no scaling).
        score_norm_method : str, optional
            Normalization method to apply after similarity calculation and scaling.
            Options: "softmax" (normalizes across cells for each label, matching torch.softmax(..., dim=0)).
            If None, no normalization is applied.
        bins : int, default 40
            Number of bins for histograms
        umap_n_neighbors : int, default 15
            Number of neighbors for UMAP
        umap_min_dist : float, default 0.1
            Minimum distance for UMAP
        umap_random_state : int, default 42
            Random state for UMAP
        umap_method : str, default "combined"
            UMAP method ("combined" or "separate")
        point_size : int, default 5
            Size of points in UMAP plots
        text_plot_enlargement_factor : float, default 1.5
            Factor to enlarge figures with text annotations
        legend_layout : str, default "horizontal"
            Legend layout ("horizontal" or "vertical")
        legend_point_size : int, default 8
            Size of points in legends
        text_dot_fill_color : str, default "label"
            Fill color for text annotation dots
        text_dot_edge_color : str, default "black"
            Edge color for text annotation dots
        text_dot_alpha : float, default 0.8
            Transparency for text annotation dots
        skip_plotting : bool, default False
            Skip all plotting for faster evaluation
        save_labels : bool, default True
            Save true/predicted labels as parquet and CSV
        output_sample_size : int, default 100
            Number of cells to include in CSV sample
        """
        # Label filtering parameters
        self.auto_filter_labels = auto_filter_labels
        self.bimodal_threshold = bimodal_threshold
        self.rare_threshold = rare_threshold

        # Store filtering results for user inspection
        self._filtering_results = None
        self._filtered_query_labels = None
        self._filtered_label_embeddings = None

        # Other parameters
        self.similarity = similarity
        self.logit_scale = logit_scale if logit_scale is not None else 1.0
        self.score_norm_method = score_norm_method
        self.bins = bins
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_random_state = umap_random_state
        self.umap_method = umap_method
        self.point_size = point_size
        self.text_plot_enlargement_factor = text_plot_enlargement_factor
        self.legend_layout = legend_layout
        self.legend_point_size = legend_point_size
        self.text_dot_fill_color = text_dot_fill_color
        self.text_dot_edge_color = text_dot_edge_color
        self.text_dot_alpha = text_dot_alpha
        self.skip_plotting = skip_plotting
        self.save_labels = save_labels
        self.output_sample_size = output_sample_size

    def get_filtering_results(self) -> dict[str, list[str]]:
        """
        Get the results of the automatic label filtering.

        Returns
        -------
        Dict[str, List[str]] or None
            Dictionary containing filtering results if auto_filter_labels was True:
            - 'bimodal_labels': Labels detected in stage 1 (clear separation)
            - 'rare_labels': Labels detected in stage 2 (rare but meaningful)
            - 'remaining_labels': Labels not detected by either method
            - 'all_scores': Dictionary with all scores for inspection
            Returns None if auto_filter_labels was False or filtering hasn't been run yet.
        """
        return self._filtering_results

    def get_used_labels(self) -> list[str]:
        """
        Get the list of labels that were actually used in the last computation.

        Returns
        -------
        List[str] or None
            List of labels that were used for computation and plotting.
            Returns None if no computation has been run yet.
        """
        if self._filtered_query_labels is not None:
            return self._filtered_query_labels.tolist()
        return None

    def _apply_label_filtering(
        self,
        omics_embeddings: np.ndarray,
        label_embeddings: np.ndarray,
        query_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply automatic label filtering if enabled.

        Parameters
        ----------
        omics_embeddings : np.ndarray
            Cell embeddings
        label_embeddings : np.ndarray
            Label embeddings
        query_labels : np.ndarray
            Query labels

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Filtered label embeddings and query labels
        """
        if not self.auto_filter_labels:
            logger.info("Label filtering disabled - using all provided labels")
            self._filtering_results = None
            self._filtered_query_labels = query_labels
            self._filtered_label_embeddings = label_embeddings
            return label_embeddings, query_labels

        logger.info("Applying automatic label filtering...")

        # Run the filtering
        filtering_results = self.filter_labels_by_bimodality(
            omics_embeddings=omics_embeddings,
            label_embeddings=label_embeddings,
            query_labels=query_labels,
            bimodal_threshold=self.bimodal_threshold,
            rare_threshold=self.rare_threshold,
        )

        # Store results for user inspection
        self._filtering_results = filtering_results

        # Get the labels that passed filtering
        used_labels = filtering_results["bimodal_labels"] + filtering_results["rare_labels"]

        if not used_labels:
            logger.warning("No labels passed the filtering criteria! Using all labels.")
            used_labels = query_labels.tolist()

        # Filter the embeddings and labels
        used_indices = []
        for i, label in enumerate(query_labels):
            if str(label) in used_labels:
                used_indices.append(i)

        filtered_label_embeddings = label_embeddings[used_indices]
        filtered_query_labels = query_labels[used_indices]

        # Store for user inspection
        self._filtered_query_labels = filtered_query_labels
        self._filtered_label_embeddings = filtered_label_embeddings

        logger.info("Label filtering complete:")
        logger.info(f"  - Original labels: {len(query_labels)}")
        logger.info(f"  - Used labels: {len(filtered_query_labels)}")
        logger.info(f"  - Bimodal labels: {len(filtering_results['bimodal_labels'])}")
        logger.info(f"  - Rare labels: {len(filtering_results['rare_labels'])}")
        logger.info(f"  - Remaining labels: {len(filtering_results['remaining_labels'])}")

        return filtered_label_embeddings, filtered_query_labels

    def _compute_similarity_matrix(
        self,
        omics_embeddings: np.ndarray,
        label_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute similarity matrix between cell embeddings and query label embeddings.

        Parameters
        ----------
        omics_embeddings : np.ndarray
            Cell embeddings (N x D)
        label_embeddings : np.ndarray
            Query label embeddings (M x D) - one embedding per query label

        Returns
        -------
        np.ndarray
            Similarity matrix (N x M)
        """
        # Compute similarity matrix using vectorized operations
        if self.similarity == "cosine":
            sim_matrix = _cosine_matrix(omics_embeddings, label_embeddings)
        elif self.similarity == "dot":
            sim_matrix = omics_embeddings @ label_embeddings.T
        else:
            raise ValueError(f"Unknown similarity: {self.similarity}")

        # Apply logit scale
        sim_matrix = sim_matrix * self.logit_scale

        # Apply normalization if specified
        if self.score_norm_method == "softmax":
            # Apply softmax along axis=0 (normalize across cells for each label)
            # This matches the behavior of torch.softmax(..., dim=0) in the reference code
            # Shape: (N_cells, M_labels) -> softmax across cells for each label
            exp_scores = np.exp(sim_matrix - np.max(sim_matrix, axis=0, keepdims=True))  # Numerical stability
            sim_matrix = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
        elif self.score_norm_method is not None:
            raise ValueError(f"Unknown score_norm_method: {self.score_norm_method}")

        return sim_matrix

    def _compute_umap(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute UMAP embedding of the data."""
        reducer = umap.UMAP(
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            random_state=self.umap_random_state,
            metric="cosine",
        )
        return reducer.fit_transform(embeddings)

    def _compute_separate_umap_with_labels(
        self,
        omics_embeddings: np.ndarray,
        label_embeddings: np.ndarray,
        true_labels: np.ndarray,
        query_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute UMAP on cell embeddings only, then position labels based on highest similarity.

        Parameters
        ----------
        omics_embeddings : np.ndarray
            Cell embeddings (N x D)
        label_embeddings : np.ndarray
            Label embeddings (M x D)
        true_labels : np.ndarray
            True labels for each cell (N,)
        query_labels : np.ndarray
            Query labels corresponding to label_embeddings (M,)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Cell UMAP coordinates and label UMAP coordinates
        """
        # Compute UMAP on cell embeddings only
        cell_umap = self._compute_umap(omics_embeddings)

        # Compute similarity matrix between cells and labels
        similarity_matrix = self._compute_similarity_matrix(omics_embeddings, label_embeddings)

        # Position each label at the cell with highest similarity
        label_umap = np.zeros((len(query_labels), 2))

        for i, _label in enumerate(query_labels):
            # Find the cell with highest similarity to this label
            max_similarity_idx = np.argmax(similarity_matrix[:, i])
            # Position label at that cell's UMAP coordinates
            label_umap[i] = cell_umap[max_similarity_idx]

        return cell_umap, label_umap

    def _plot_confusion_matrix(
        self,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        output_path: Path,
        title: str = "Confusion Matrix",
        normalize: bool = True,
        figsize: tuple = (8, 8),
        font_size: int = 12,
        save_format: str = "png",
        dpi: int = 300,
        show_annotations: bool = None,
    ) -> None:
        """
        Plot a confusion matrix heatmap comparing true and predicted labels.

        Parameters
        ----------
        true_labels : np.ndarray
            Ground truth labels
        pred_labels : np.ndarray
            Predicted labels
        output_path : Path
            File path to save the confusion matrix figure
        title : str
            Title for the plot
        normalize : bool
            Whether to normalize the confusion matrix
        figsize : tuple
            Figure size
        font_size : int
            Font size for annotations
        save_format : str
            Format for saving the plot
        dpi : int
            DPI for the saved plot
        show_annotations : bool, optional
            Whether to show numerical annotations. If None, defaults to False for normalized matrices, True for count matrices
        """
        # Convert to pandas Series and ensure string type for consistent handling
        true_labels = pd.Series(true_labels).astype(str)
        pred_labels = pd.Series(pred_labels).astype(str)

        # Unified label set for consistent axis order
        labels = sorted(set(true_labels) | set(pred_labels))

        # Compute confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=labels)

        # Normalize if requested
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)  # handle divide-by-zero cases

        # Determine whether to show annotations
        if show_annotations is None:
            # Default: show annotations for count matrices, hide for normalized matrices
            show_annotations = not normalize

        # Adjust figure size based on number of labels
        width_per_label = 0.7
        height_per_label = 0.7
        fig_width = max(figsize[0], width_per_label * len(labels))
        fig_height = max(figsize[1], height_per_label * len(labels))

        plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

        # Create heatmap
        ax = sns.heatmap(
            cm,
            annot=show_annotations,
            square=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Proportion" if normalize else "Count", "shrink": 0.6, "aspect": 20},
            annot_kws={"size": max(6, font_size - 2)} if show_annotations else {},  # Only set if showing annotations
            vmax=1.0 if normalize else None,
            vmin=0,
        )

        # Customize colorbar
        cbar = ax.collections[0].colorbar
        if normalize:
            cbar.set_ticks([0, 0.5, 1])
        cbar.outline.set_edgecolor("black")

        # Set labels and title
        plt.xlabel("Predicted Label", fontsize=font_size)
        plt.ylabel("True Label", fontsize=font_size)
        plt.title(title, fontsize=font_size + 2, fontweight="bold")

        # Rotate x-axis labels if there are many classes
        if len(labels) > 10:
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)

        # Save the plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _safe_tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()

    def _add_non_overlapping_annotations(
        self,
        points: np.ndarray,
        labels: list,
        color_map: dict,
        fontsize: int = 8,
        fill_color: str = "red",
        edge_color: str = "white",
        alpha: float = 0.8,
    ) -> None:
        """
        Add text annotations with arrows to avoid overlap.

        Parameters
        ----------
        points : np.ndarray
            Array of (x, y) coordinates for label positions
        labels : list
            List of label strings
        color_map : dict
            Mapping from label to color
        fontsize : int
            Font size for the text
        fill_color : str
            Fill color of the annotation dots ("red", "blue", "none", "label", etc.)
            Use "label" to use the label's color from color_map
        edge_color : str
            Edge color of the annotation dots ("white", "black", "none", "label", etc.)
            Use "label" to use the label's color from color_map
        alpha : float
            Transparency of the annotation dots (0.0 = fully transparent, 1.0 = fully opaque)
        """
        # Get plot bounds for relative offset calculation
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        # Much larger offset distance to avoid crowding
        base_offset = 0.15  # Increased from 0.15
        x_offset_base = base_offset * x_range
        y_offset_base = base_offset * y_range

        # Store chosen positions to avoid overlap
        chosen_positions = []

        # Define more angles for better distribution
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)  # 16 directions instead of 8

        for i, (point, label) in enumerate(zip(points, labels, strict=False)):
            # Get label color from color_map
            label_color = color_map.get(label, "red")

            # Add subtle dot at actual position - much smaller and less prominent
            if fill_color.lower() != "none" or edge_color.lower() != "none":
                # Determine colors for fill and edge
                if fill_color.lower() == "label":
                    dot_fill = label_color
                elif fill_color.lower() == "none":
                    dot_fill = None
                else:
                    dot_fill = fill_color

                if edge_color.lower() == "label":
                    dot_edge = label_color
                elif edge_color.lower() == "none":
                    dot_edge = None
                else:
                    dot_edge = edge_color

                plt.scatter(
                    point[0], point[1], c=dot_fill, s=8, alpha=alpha, zorder=15, edgecolors=dot_edge, linewidths=0.5
                )

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

            # Add annotation with label color
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
                    "edgecolor": label_color,  # Use label color for box outline
                    "linewidth": 1.0,  # Slightly thicker for better visibility
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": label_color,
                    "lw": 0.8,
                    "alpha": 0.7,
                },  # Use label color for arrow
                zorder=20,
            )

    def _compute_roc(self, scores: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute ROC curve and AUC for a set of scores and true labels."""
        fpr, tpr, _ = roc_curve(mask, scores)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    def _compute_accuracy_from_matrix(
        self, similarity_matrix: np.ndarray, true_labels: np.ndarray, query_labels: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """
        Compute accuracy using precomputed similarity matrix.

        Parameters
        ----------
        similarity_matrix : np.ndarray
            Precomputed similarity matrix (N x M)
        true_labels : np.ndarray
            True labels for each cell (N,)
        query_labels : np.ndarray
            Query labels corresponding to similarity matrix columns (M,)

        Returns
        -------
        tuple[float, np.ndarray]
            Accuracy score and predicted labels
        """
        # Find the query label with highest similarity for each cell
        predicted_indices = np.argmax(similarity_matrix, axis=1)
        predicted_labels = query_labels[predicted_indices]

        # Compute accuracy
        correct = np.sum(predicted_labels == true_labels)
        accuracy = correct / len(true_labels)

        return accuracy, predicted_labels

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
        # proportions = counts / len(labels)
        # Random baseline respecting the distribution of the labels
        # random_accuracy = np.sum(proportions**2)
        # In cell whisperer, random accuracy is 1/number of classes. Makes more sense for zero shot classification.
        random_accuracy = 1 / len(unique_labels)
        return random_accuracy

    def _save_label_predictions(
        self,
        out_dir: Path,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        similarity_matrix: np.ndarray,
        query_labels: np.ndarray,
        label_key: str,
    ) -> None:
        """
        Save true and predicted labels with similarity scores as parquet and CSV.

        Parameters
        ----------
        out_dir : Path
            Output directory
        true_labels : np.ndarray
            True labels for each cell
        predicted_labels : np.ndarray
            Predicted labels for each cell
        similarity_matrix : np.ndarray
            Similarity matrix (N x M)
        query_labels : np.ndarray
            Query labels corresponding to similarity matrix columns
        label_key : str
            Label key for filename
        """
        if not self.save_labels:
            return

        # Create labels directory
        labels_dir = out_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Get max similarity scores for each cell (confidence scores)
        max_similarities = np.max(similarity_matrix, axis=1)

        # Create DataFrame with all data
        df_data = {
            "cell_id": np.arange(len(true_labels)),
            "true_label": true_labels,
            "predicted_label": predicted_labels,
            "max_similarity": max_similarities,
            "correct_prediction": true_labels == predicted_labels,
        }

        # Add individual similarity scores for each query label
        for i, label in enumerate(query_labels):
            df_data[f"similarity_{label}"] = similarity_matrix[:, i]

        df = pd.DataFrame(df_data)

        # Save full dataset as parquet
        parquet_path = labels_dir / f"{label_key}_predictions.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved full label predictions to {parquet_path}")

        # Save random sample as CSV for visual inspection
        if len(df) > self.output_sample_size:
            # Set random seed for reproducible sampling
            np.random.seed(42)
            sample_indices = np.random.choice(len(df), size=self.output_sample_size, replace=False)
            df_sample = df.iloc[sample_indices].copy()
            df_sample = df_sample.sort_values("cell_id").reset_index(drop=True)
        else:
            df_sample = df.copy()

        csv_path = labels_dir / f"{label_key}_predictions_sample.csv"
        df_sample.to_csv(csv_path, index=False)
        logger.info(f"Saved sample label predictions ({len(df_sample)} cells) to {csv_path}")

    def compute(
        self,
        omics_embeddings: np.ndarray,
        *,
        label_embeddings: np.ndarray,
        query_labels: np.ndarray | list[str] | pd.Series,
        true_labels: np.ndarray | list[str] | pd.Series,
        label_key: str,
        out_dir: Path | str | None = None,
        skip_plotting: bool = None,
        **kw,
    ) -> EvalResult:
        """
        Compute similarity scores and ROC metrics for each unique label.

        This method evaluates how well cell embeddings can be matched to their true labels
        by computing similarity scores between cells and query label embeddings.

        Parameters
        ----------
        omics_embeddings : np.ndarray
            Cell embeddings with shape (N_cells, embedding_dim)
        label_embeddings : np.ndarray
            Label embeddings with shape (N_labels, embedding_dim)
            One embedding per query label
        query_labels : Union[np.ndarray, List[str], pd.Series]
            Label strings corresponding to label_embeddings (length N_labels)
        true_labels : Union[np.ndarray, List[str], pd.Series]
            True labels for each cell in omics_embeddings (length N_cells)
        label_key : str
            Identifier for this evaluation (used in file naming)
        out_dir : Union[Path, str, None], optional
            Output directory for saving results and plots
        skip_plotting : bool, optional
            Whether to skip plotting (overrides class default)

        Returns
        -------
        EvalResult
            Dictionary-like object containing evaluation metrics:
            - Per-label AUC scores: {label_name}/auc
            - Per-label accuracy: {label_name}/accuracy
            - Mean AUC across all labels: mean_auc
            - Overall accuracy: accuracy
            - Balanced accuracy: balanced_accuracy
            - Random baseline: random_baseline_accuracy
            - Number of labels: n_labels
        """
        # Override class attribute if parameter is provided
        if skip_plotting is not None:
            self.skip_plotting = skip_plotting

        # Convert out_dir to Path if provided
        if out_dir is not None:
            out_dir = Path(out_dir)

        # Convert to numpy arrays
        query_labels = np.asarray(query_labels)
        true_labels = np.asarray(true_labels)

        # Validate input dimensions
        if len(query_labels) != label_embeddings.shape[0]:
            raise ValueError(
                f"query_labels length ({len(query_labels)}) must match label_embeddings.shape[0] ({label_embeddings.shape[0]})"
            )
        if len(true_labels) != omics_embeddings.shape[0]:
            raise ValueError(
                f"true_labels length ({len(true_labels)}) must match omics_embeddings.shape[0] ({omics_embeddings.shape[0]})"
            )

        # Apply automatic label filtering if enabled
        label_embeddings, query_labels = self._apply_label_filtering(omics_embeddings, label_embeddings, query_labels)

        # Get unique labels from true_labels (these are the labels we'll evaluate)
        unique_labels = np.unique(true_labels)

        # Compute similarity matrix once using vectorized operations
        similarity_matrix = self._compute_similarity_matrix(omics_embeddings, label_embeddings)

        # Compute results using the precomputed matrix
        out = {}
        auc_scores = []
        per_label_accuracies = []  # For balanced accuracy calculation

        # Compute accuracy metrics using vectorized operations
        accuracy, predicted_labels = self._compute_accuracy_from_matrix(similarity_matrix, true_labels, query_labels)
        random_baseline = self._compute_random_baseline_accuracy(true_labels)

        # Compute per-label metrics for ROC curves and balanced accuracy
        for _i, v in enumerate(unique_labels):
            mask = true_labels == v
            if np.sum(mask) > 0:  # Only compute if this label exists in the dataset
                # Find which query label corresponds to this true label
                query_label_idx = None
                for j, q_label in enumerate(query_labels):
                    if str(q_label) == str(v):
                        query_label_idx = j
                        break

                if query_label_idx is not None:
                    sim = similarity_matrix[:, query_label_idx]  # Extract similarity scores for this label
                    # Compute ROC curve and AUC
                    _, _, roc_auc = self._compute_roc(sim, mask)
                    auc_scores.append(roc_auc)

                    # Compute per-label accuracy (recall for this label)
                    correct_predictions_for_label = np.sum((true_labels == v) & (predicted_labels == v))
                    total_true_instances = np.sum(mask)
                    per_label_accuracy = correct_predictions_for_label / total_true_instances
                    per_label_accuracies.append(per_label_accuracy)

                    prefix = f"{v}"
                    out[f"{prefix}/auc"] = float(roc_auc)
                    out[f"{prefix}/accuracy"] = float(per_label_accuracy)

        # Compute balanced accuracy (mean of per-class recalls)
        balanced_accuracy = float(np.mean(per_label_accuracies)) if per_label_accuracies else 0.0

        # Add mean AUC, standard deviation, and accuracy metrics
        out["mean_auc"] = float(np.mean(auc_scores)) if auc_scores else 0.0
        out["std_auc"] = float(np.std(auc_scores)) if auc_scores else 0.0
        out["accuracy"] = float(accuracy)
        out["balanced_accuracy"] = balanced_accuracy
        out["random_baseline_accuracy"] = float(random_baseline)
        out["accuracy_over_random"] = float(accuracy / random_baseline) if random_baseline > 0 else 0.0
        out["n_labels"] = len(unique_labels)

        # Save label predictions if enabled and output directory is provided
        if out_dir is not None:
            # In compute(), out_dir is emb_dir, so we need to create the eval subdirectory structure
            eval_dir = out_dir / "eval" / "LabelSimilarity" / label_key
            self._save_label_predictions(
                eval_dir, true_labels, predicted_labels, similarity_matrix, query_labels, label_key
            )

        return EvalResult(**out)

    def plot(
        self,
        omics_embeddings: np.ndarray,
        out_dir: Path | str,
        *,
        label_embeddings: np.ndarray,
        query_labels: np.ndarray | list[str] | pd.Series,
        true_labels: np.ndarray | list[str] | pd.Series,
        label_key: str,
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
        point_size: int = None,
        text_plot_enlargement_factor: float = None,
        legend_layout: str = None,
        legend_point_size: int = None,
        text_dot_fill_color: str = None,
        text_dot_edge_color: str = None,
        text_dot_alpha: float = None,
        umap_method: str = None,
        **kw,
    ) -> None:
        """
        Generate comprehensive visualizations for label similarity evaluation.

        Creates multiple types of plots:
        - ROC curves for each label
        - UMAP visualizations colored by true/predicted labels
        - Similarity score histograms
        - Confusion matrices

        Parameters
        ----------
        omics_embeddings : np.ndarray
            Cell embeddings with shape (N_cells, embedding_dim)
        out_dir : Union[Path, str]
            Output directory for saving plots
        label_embeddings : np.ndarray
            Label embeddings with shape (N_labels, embedding_dim)
        query_labels : Union[np.ndarray, List[str], pd.Series]
            Label strings corresponding to label_embeddings
        true_labels : Union[np.ndarray, List[str], pd.Series]
            True labels for each cell
        label_key : str
            Identifier for this evaluation (used in file naming)
        save_format : str, default "png"
            Format for saving plots ("png", "svg", "pdf")
        figsize : tuple, default (6, 6)
            Figure size for plots
        dpi : int, default 300
            DPI for saved plots
        font_size : int, default 12
            Base font size
        font_style : str, default "normal"
            Font style ("normal", "italic")
        font_weight : str, default "normal"
            Font weight ("normal", "bold")
        legend_fontsize : int, default 10
            Font size for legends
        axis_label_size : int, default 12
            Font size for axis labels
        axis_tick_size : int, default 10
            Font size for axis ticks
        frameon : bool, default False
            Whether to show plot frames
        skip_plotting : bool, optional
            Whether to skip plotting (overrides class default)
        point_size : int, optional
            Size of points in UMAP plots
        text_plot_enlargement_factor : float, optional
            Factor to enlarge figures with text annotations
        legend_layout : str, optional
            Legend layout ("horizontal", "vertical")
        legend_point_size : int, optional
            Size of points in legends
        text_dot_fill_color : str, optional
            Fill color for text annotation dots
        text_dot_edge_color : str, optional
            Edge color for text annotation dots
        text_dot_alpha : float, optional
            Transparency for text annotation dots
        umap_method : str, optional
            UMAP computation method ("combined", "separate")
        **eval_cfg: Any
            Additional evaluation configuration parameters
        """
        # Override class attribute if parameter is provided
        if skip_plotting is not None:
            self.skip_plotting = skip_plotting

        # Convert out_dir to Path
        out_dir = Path(out_dir)

        # Convert to numpy arrays
        query_labels = np.asarray(query_labels)
        true_labels = np.asarray(true_labels)

        # Validate input dimensions
        if len(query_labels) != label_embeddings.shape[0]:
            raise ValueError(
                f"query_labels length ({len(query_labels)}) must match label_embeddings.shape[0] ({label_embeddings.shape[0]})"
            )
        if len(true_labels) != omics_embeddings.shape[0]:
            raise ValueError(
                f"true_labels length ({len(true_labels)}) must match omics_embeddings.shape[0] ({omics_embeddings.shape[0]})"
            )

        # Apply automatic label filtering if enabled
        label_embeddings, query_labels = self._apply_label_filtering(omics_embeddings, label_embeddings, query_labels)

        # Override parameters if provided
        plot_point_size = self.point_size if point_size is None else point_size
        enlargement_factor = (
            self.text_plot_enlargement_factor if text_plot_enlargement_factor is None else text_plot_enlargement_factor
        )
        legend_layout_mode = self.legend_layout if legend_layout is None else legend_layout
        legend_marker_size = self.legend_point_size if legend_point_size is None else legend_point_size
        annotation_fill_color = self.text_dot_fill_color if text_dot_fill_color is None else text_dot_fill_color
        annotation_edge_color = self.text_dot_edge_color if text_dot_edge_color is None else text_dot_edge_color
        annotation_alpha = self.text_dot_alpha if text_dot_alpha is None else text_dot_alpha
        umap_computation_method = self.umap_method if umap_method is None else umap_method

        # Validate umap_method parameter
        if umap_computation_method not in ["combined", "separate"]:
            raise ValueError(f"umap_method must be 'combined' or 'separate', got '{umap_computation_method}'")

        if self.skip_plotting:
            logger.info("Skipping plotting due to skip_plotting flag.")
            return

        frameon = True  # hardcode for these plots for now
        query_labels = np.asarray(query_labels)
        true_labels = np.asarray(true_labels)
        unique_labels = np.unique(true_labels)

        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(omics_embeddings, label_embeddings)

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
        true_label_boxplots_dir = out_dir / "true_label_boxplots"
        cm_dir = out_dir / "confusion_matrix"

        for d in [roc_dir, umap_dir, hist_dir, true_label_boxplots_dir, cm_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Compute predicted labels using the precomputed similarity matrix
        predicted_indices = np.argmax(similarity_matrix, axis=1)
        predicted_labels = query_labels[predicted_indices]

        # Save label predictions if enabled
        # In plot(), out_dir is already the eval subdirectory (plot_dir)
        self._save_label_predictions(out_dir, true_labels, predicted_labels, similarity_matrix, query_labels, label_key)

        # Plot confusion matrix
        cm_title = f"Confusion Matrix - {label_key}"
        self._plot_confusion_matrix(
            true_labels,
            predicted_labels,
            cm_dir / f"confusion_matrix.{save_format}",
            title=cm_title,
            normalize=True,
            figsize=figsize,
            font_size=font_size,
            save_format=save_format,
            dpi=dpi,
        )

        # Also create unnormalized version
        self._plot_confusion_matrix(
            true_labels,
            predicted_labels,
            cm_dir / f"confusion_matrix_counts.{save_format}",
            title=f"{cm_title} (Counts)",
            normalize=False,
            figsize=figsize,
            font_size=font_size,
            save_format=save_format,
            dpi=dpi,
        )

        # Remove normalized accuracy computation since we removed that method
        # Use regular predicted labels for all confusion matrices

        # Plot UMAP colored by true labels (with and without text annotations)
        unique_labels = np.unique(true_labels)

        # Create comprehensive color mapping for all query labels
        # This ensures all query labels have colors, even if they don't appear in true data
        all_labels = np.unique(np.concatenate([unique_labels, query_labels]))

        # Use categorical color palettes for better distinction
        if len(all_labels) <= 10:
            colors = sns.color_palette("tab10", len(all_labels))
        elif len(all_labels) <= 20:
            colors = sns.color_palette("tab20", len(all_labels))
        elif len(all_labels) <= 40:
            # For 21-40 classes, combine tab20 and tab20b
            colors = sns.color_palette("tab20", 20) + sns.color_palette("tab20b", len(all_labels) - 20)
        else:
            # For >40 classes, use a continuous colormap to ensure we have enough colors
            # Use HSV colormap for maximum distinction between colors
            colors = sns.color_palette("hsv", len(all_labels))

        # Create color map for all labels (query + unique)
        color_map = {label: colors[i] for i, label in enumerate(all_labels)}

        # For true labels plot, only use colors for labels that exist in true data
        point_colors = [color_map.get(label, "gray") for label in true_labels]

        # Use label_embeddings directly for UMAP visualization
        # Compute UMAP based on selected method
        if umap_computation_method == "separate":
            # Use separate UMAP computation with highest similarity positioning
            cell_umap, label_umap = self._compute_separate_umap_with_labels(
                omics_embeddings, label_embeddings, true_labels, query_labels
            )
        else:
            # Use combined UMAP computation (original method)
            combined_embeddings = np.vstack([omics_embeddings, label_embeddings])
            combined_umap = self._compute_umap(combined_embeddings)

            # Split back into cell and label coordinates
            cell_umap = combined_umap[: len(omics_embeddings)]
            label_umap = combined_umap[len(omics_embeddings) :]

        # Version 1: Without text annotations (using cell_umap which has label positions)
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.scatter(cell_umap[:, 0], cell_umap[:, 1], c=point_colors, s=plot_point_size, alpha=0.7, edgecolors="none")
        plt.xticks([])
        plt.yticks([])
        _safe_tight_layout()

        plt.savefig(umap_dir / f"00_true_labels.{save_format}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # Version 2: With text annotations (larger figure to accommodate text)
        text_figsize = (figsize[0] * enlargement_factor, figsize[1] * enlargement_factor)  # Enlarge to accommodate text
        plt.figure(figsize=text_figsize, dpi=dpi)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.scatter(cell_umap[:, 0], cell_umap[:, 1], c=point_colors, s=plot_point_size, alpha=0.7, edgecolors="none")

        # Fix axis limits before adding annotations to prevent plot area from expanding
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Add text annotations with arrows to avoid overlap
        self._add_non_overlapping_annotations(
            label_umap,
            query_labels,
            color_map=color_map,
            fontsize=max(6, legend_fontsize - 2),
            fill_color=annotation_fill_color,
            edge_color=annotation_edge_color,
            alpha=annotation_alpha,
        )

        # Restore original axis limits to maintain consistent plot size
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        plt.xticks([])
        plt.yticks([])
        _safe_tight_layout()

        plt.savefig(umap_dir / f"00_true_labels_with_text.{save_format}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # Create and save separate legend for all query labels
        if legend_layout_mode == "vertical":
            legend_figsize = (3, min(len(query_labels) * 0.4, 8))  # Narrow and tall for vertical layout
        else:
            legend_figsize = (min(len(query_labels) * 1.2, 12), 1)  # Wide and short for horizontal layout
        fig_legend, ax_legend = plt.subplots(figsize=legend_figsize, dpi=dpi)
        ax_legend.axis("off")
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_map[label],
                markersize=legend_marker_size,
                label=str(label),
                markeredgecolor="none",
            )
            for label in query_labels
        ]
        ax_legend.legend(
            handles=legend_elements,
            loc="center",
            fontsize=legend_fontsize,
            frameon=False,
            ncol=1 if legend_layout_mode == "vertical" else min(len(query_labels), 6),
            bbox_to_anchor=(0.5, 0.5),
        )
        _safe_tight_layout()
        plt.savefig(umap_dir / f"00_true_labels_legend.{save_format}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # Plot UMAP colored by predicted labels (with and without text annotations)
        # Use the same comprehensive color map for predicted labels
        pred_point_colors = [color_map.get(label, "gray") for label in predicted_labels]

        # Version 1: Without text annotations
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.scatter(
            cell_umap[:, 0], cell_umap[:, 1], c=pred_point_colors, s=plot_point_size, alpha=0.7, edgecolors="none"
        )
        plt.xticks([])
        plt.yticks([])
        _safe_tight_layout()

        plt.savefig(umap_dir / f"01_predicted_labels.{save_format}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # Version 2: With text annotations (larger figure to accommodate text)
        text_figsize = (figsize[0] * enlargement_factor, figsize[1] * enlargement_factor)  # Enlarge to accommodate text
        plt.figure(figsize=text_figsize, dpi=dpi)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.scatter(
            cell_umap[:, 0], cell_umap[:, 1], c=pred_point_colors, s=plot_point_size, alpha=0.7, edgecolors="none"
        )

        # Fix axis limits before adding annotations to prevent plot area from expanding
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Add text annotations with arrows to avoid overlap
        self._add_non_overlapping_annotations(
            label_umap,
            query_labels,
            color_map=color_map,
            fontsize=max(6, legend_fontsize - 2),
            fill_color=annotation_fill_color,
            edge_color=annotation_edge_color,
            alpha=annotation_alpha,
        )

        # Restore original axis limits to maintain consistent plot size
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        plt.xticks([])
        plt.yticks([])
        _safe_tight_layout()

        plt.savefig(umap_dir / f"01_predicted_labels_with_text.{save_format}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # Create and save separate legend for all query labels (same as true labels since they use the same color scheme)
        if legend_layout_mode == "vertical":
            legend_figsize = (3, min(len(query_labels) * 0.4, 8))  # Narrow and tall for vertical layout
        else:
            legend_figsize = (min(len(query_labels) * 1.2, 12), 1)  # Wide and short for horizontal layout
        fig_legend, ax_legend = plt.subplots(figsize=legend_figsize, dpi=dpi)
        ax_legend.axis("off")
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_map[label],
                markersize=legend_marker_size,
                label=str(label),
                markeredgecolor="none",
            )
            for label in query_labels
        ]
        ax_legend.legend(
            handles=legend_elements,
            loc="center",
            fontsize=legend_fontsize,
            frameon=False,
            ncol=1 if legend_layout_mode == "vertical" else min(len(query_labels), 6),
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
        # First, handle ROC curves only for labels that exist in the data
        for _i, v in enumerate(unique_labels):
            mask = true_labels == v
            if np.sum(mask) > 0:  # Only compute if this label exists in the dataset
                # Find which query label corresponds to this true label
                query_label_idx = None
                for j, q_label in enumerate(query_labels):
                    if str(q_label) == str(v):
                        query_label_idx = j
                        break

                if query_label_idx is not None:
                    sim = similarity_matrix[:, query_label_idx]  # Extract similarity scores for this label

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
                    plt.title(f"{v}")
                    _safe_tight_layout()
                    plt.savefig(roc_dir / f"{safe_v}.{save_format}", dpi=dpi, bbox_inches="tight")
                    plt.close()

                    # Create and save separate legend for ROC curve
                    fig_legend, ax_legend = plt.subplots(figsize=(3, 1), dpi=dpi)
                    ax_legend.axis("off")
                    legend_elements = [
                        plt.Line2D([0], [0], color="tab:blue", linewidth=2, label=f"AUC = {roc_auc:.3f}")
                    ]
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

        # Now create histograms for ALL query labels (including those not in the data)
        for i, query_label in enumerate(query_labels):
            sim = similarity_matrix[:, i]  # Extract similarity scores for this query label

            # Check if this query label exists in the true data
            mask = true_labels == query_label
            has_positive_samples = np.sum(mask) > 0

            # Plot histogram with consistent scale
            plt.figure(figsize=figsize, dpi=dpi)

            if has_positive_samples:
                # Label exists in data - show both positive and negative distributions
                # Use density=False and normalize manually to keep overall density scale
                plt.hist(sim[~mask], bins=self.bins, alpha=0.5, label="other", density=False, range=(-1, 1))
                plt.hist(sim[mask], bins=self.bins, alpha=0.7, label=str(query_label), density=False, range=(-1, 1))
                legend_elements = [
                    plt.Rectangle((0, 0), 1, 1, facecolor="tab:blue", alpha=0.5, label="other"),
                    plt.Rectangle((0, 0), 1, 1, facecolor="tab:orange", alpha=0.7, label=str(query_label)),
                ]
            else:
                # Label doesn't exist in data - show only the "other" distribution
                plt.hist(sim, bins=self.bins, alpha=0.5, label="other", density=False, range=(-1, 1))
                legend_elements = [
                    plt.Rectangle((0, 0), 1, 1, facecolor="tab:blue", alpha=0.5, label="other"),
                ]

            plt.xlabel(f"{self.similarity.title()} Similarity")
            plt.ylabel("Count")
            plt.xlim(-1, 1)
            plt.title(f"{query_label}")
            _safe_tight_layout()

            safe_query_label = re.sub(r"[^\w\d\-\.]", "_", str(query_label))
            plt.savefig(hist_dir / f"{safe_query_label}.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()

            # Create and save separate legend for histogram
            fig_legend, ax_legend = plt.subplots(figsize=(3, 1), dpi=dpi)
            ax_legend.axis("off")
            ax_legend.legend(
                handles=legend_elements,
                loc="center",
                fontsize=legend_fontsize,
                frameon=False,
                bbox_to_anchor=(0.5, 0.5),
            )
            _safe_tight_layout()
            plt.savefig(hist_dir / f"{safe_query_label}_legend.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()

        # Create second round of boxplots: for each true cell type, plot similarities to all query labels
        unique_true_labels = np.unique(true_labels)

        # Use the same color_map from UMAP plots to ensure consistency
        # The color_map is already created earlier for UMAP plots

        for true_label in unique_true_labels:
            # Get mask for cells with this true label
            mask = true_labels == true_label
            cells_with_label = np.sum(mask)

            if cells_with_label == 0:
                continue

            # For these cells, get their similarities to all query labels
            cells_similarities = similarity_matrix[mask, :]  # Shape: (N_cells_with_label, N_query_labels)

            # Plot boxplots for each query label
            plt.figure(figsize=figsize, dpi=dpi)

            # Prepare data for boxplots: list of similarity arrays for each query label
            data_to_plot = [cells_similarities[:, i] for i in range(len(query_labels))]
            query_label_names = [str(label) for label in query_labels]

            # Create boxplot with flipped axes (horizontal)
            bp = plt.boxplot(data_to_plot, labels=query_label_names, vert=False, widths=0.6, patch_artist=True)

            # Color the boxes with query colors and set outlier size
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(color_map[query_label_names[i]])
                patch.set_alpha(0.7)

            # Make outliers smaller
            for flier in bp["fliers"]:
                flier.set_markersize(3)
                flier.set_markeredgewidth(0.5)

            plt.xlabel(f"{self.similarity.title()} Similarity", fontsize=font_size)
            plt.ylabel("Query Labels", fontsize=font_size)
            plt.xlim(-0.6, 0.6)
            plt.title(f"True Label: {true_label} (n={cells_with_label})", fontsize=font_size)

            # Set tick label sizes for both axes
            ax = plt.gca()
            ax.tick_params(axis="x", labelsize=axis_tick_size)
            ax.tick_params(axis="y", labelsize=axis_tick_size)
            plt.yticks(rotation=0, ha="right")

            _safe_tight_layout()

            safe_true_label = re.sub(r"[^\w\d\-\.]", "_", str(true_label))
            plt.savefig(true_label_boxplots_dir / f"{safe_true_label}.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()

        # Plot mean ROC curve
        mean_tpr /= len(unique_labels)
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

    def _compute_bimodal_score(self, similarity_scores: np.ndarray, method: str = "separation_score") -> float:
        """
        Compute a bimodal score for similarity scores without needing true labels.

        Parameters
        ----------
        similarity_scores : np.ndarray
            Similarity scores for all cells against this query label (N,)
        method : str
            Method to compute bimodality: "separation_score" or "rare_label_score"

        Returns
        -------
        float
            Bimodal score (higher = more bimodal)
        """
        if len(similarity_scores) < 4:
            return 0.0

        if method == "rare_label_score":
            # Combined score specifically designed for rare labels
            # Combines multiple signals that indicate rare but meaningful labels

            # 1. Tail heaviness (rare high-similarity cells)
            from scipy import stats

            try:
                # Fit normal distribution to the data
                mu, sigma = stats.norm.fit(similarity_scores)

                # Calculate how many cells are in the upper tail (top 5%)
                threshold = np.percentile(similarity_scores, 95)
                tail_count = np.sum(similarity_scores > threshold)
                expected_tail = len(similarity_scores) * 0.05

                # Return ratio of actual vs expected tail
                tail_score = tail_count / (expected_tail + 1e-8)
            except ImportError:
                tail_score = 0.0

            # 2. Positive skewness (right tail)
            try:
                skewness = stats.skew(similarity_scores)
                # Only return positive skewness (right tail)
                skew_score = max(0, skewness)
            except ImportError:
                skew_score = 0.0

            # 3. Outlier score (rare high-similarity cells)
            q1, q3 = np.percentile(similarity_scores, [25, 75])
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr

            # Count outliers in upper tail
            outliers = np.sum(similarity_scores > upper_bound)
            outlier_score = outliers / len(similarity_scores)  # Proportion of outliers

            # 4. Z-score max (most extreme similarity)
            mean_score = np.mean(similarity_scores)
            std_score = np.std(similarity_scores)

            if std_score > 0:
                z_scores = (similarity_scores - mean_score) / std_score
                zscore_score = np.max(z_scores)
            else:
                zscore_score = 0.0

            # Combine scores (weighted average)
            combined_score = 0.3 * tail_score + 0.2 * skew_score + 0.2 * outlier_score + 0.3 * zscore_score

            return combined_score

        elif method == "separation_score":
            # Measure separation between background and positive populations
            # Made more inclusive to catch B_cell cases

            # Find the main peak (background)
            hist, bin_edges = np.histogram(similarity_scores, bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Find the highest peak (background)
            main_peak_idx = np.argmax(hist)
            main_peak_pos = bin_centers[main_peak_idx]

            # Find secondary peaks (positive population) - more inclusive
            from scipy.signal import find_peaks

            try:
                # Try multiple prominence thresholds to be more inclusive
                prominence_thresholds = [0.05, 0.03, 0.02]  # Start with 5%, then 3%, then 2%
                distance_thresholds = [3, 2, 1]  # Corresponding distance thresholds

                best_score = 0.0

                for prominence_thresh, distance_thresh in zip(prominence_thresholds, distance_thresholds, strict=False):
                    peaks, properties = find_peaks(
                        hist, prominence=np.max(hist) * prominence_thresh, distance=distance_thresh
                    )

                    if len(peaks) > 1:
                        # Find the peak furthest from the main peak
                        peak_positions = bin_centers[peaks]
                        distances = np.abs(peak_positions - main_peak_pos)
                        furthest_peak_idx = peaks[np.argmax(distances)]
                        furthest_peak_pos = bin_centers[furthest_peak_idx]

                        # Calculate separation score
                        separation = abs(furthest_peak_pos - main_peak_pos)

                        # Weight by the prominence of the secondary peak
                        secondary_prominence = properties["prominences"][np.argmax(distances)]

                        # Calculate score and keep the best one
                        score = separation * secondary_prominence
                        best_score = max(best_score, score)

                return best_score

            except ImportError:
                return 0.0

        else:
            raise ValueError(f"Unknown bimodal method: {method}")

    def compute_bimodal_scores(
        self,
        omics_embeddings: np.ndarray,
        label_embeddings: np.ndarray,
        query_labels: np.ndarray | list | pd.Series,
        method: str = "bimodal_coefficient",
    ) -> dict[str, float]:
        """
        Compute bimodal scores for all query labels without needing true labels.

        Parameters
        ----------
        omics_embeddings : np.ndarray
            Cell embeddings (N x D)
        label_embeddings : np.ndarray
            Label embeddings (M x D)
        query_labels : np.ndarray | list | pd.Series
            Query labels (M,)
        method : str
            Bimodal computation method

        Returns
        -------
        dict[str, float]
            Dictionary mapping query labels to bimodal scores
        """
        # Convert to numpy arrays
        query_labels = np.asarray(query_labels)

        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(omics_embeddings, label_embeddings)

        # Compute bimodal scores for each query label
        bimodal_scores = {}
        for i, query_label in enumerate(query_labels):
            similarity_scores = similarity_matrix[:, i]
            bimodal_score = self._compute_bimodal_score(similarity_scores, method)
            bimodal_scores[str(query_label)] = bimodal_score

        return bimodal_scores

    def filter_labels_by_bimodality(
        self,
        omics_embeddings: np.ndarray,
        label_embeddings: np.ndarray,
        query_labels: np.ndarray | list[str] | pd.Series,
        bimodal_threshold: float = 0.1,
        rare_threshold: float = 2.0,
    ) -> dict[str, list[str]]:
        """
        Two-stage label filtering based on bimodality detection.

        This method helps identify which query labels are meaningful by analyzing
        the similarity score distributions. It uses a two-stage approach:

        Stage 1: Detect labels with clear bimodal separation (e.g., T_cell, B_cell)
        Stage 2: Detect rare labels with meaningful signatures (e.g., rare cell types)

        Parameters
        ----------
        omics_embeddings : np.ndarray
            Cell embeddings with shape (N_cells, embedding_dim)
        label_embeddings : np.ndarray
            Label embeddings with shape (N_labels, embedding_dim)
        query_labels : Union[np.ndarray, List[str], pd.Series]
            Query label strings (length N_labels)
        bimodal_threshold : float, default 0.1
            Threshold for separation_score in stage 1
        rare_threshold : float, default 2.0
            Threshold for rare_label_score in stage 2

        Returns
        -------
        Dict[str, List[str]]
            Dictionary containing:
            - 'bimodal_labels': Labels detected in stage 1 (clear separation)
            - 'rare_labels': Labels detected in stage 2 (rare but meaningful)
            - 'remaining_labels': Labels not detected by either method
            - 'all_scores': Dictionary with all scores for inspection
        """
        # Convert to numpy arrays
        query_labels = np.asarray(query_labels)

        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(omics_embeddings, label_embeddings)

        # Stage 1: Compute separation_score for all labels
        bimodal_scores = {}
        for i, query_label in enumerate(query_labels):
            similarity_scores = similarity_matrix[:, i]
            bimodal_score = self._compute_bimodal_score(similarity_scores, "separation_score")
            bimodal_scores[str(query_label)] = bimodal_score

        # Select labels above bimodal threshold
        bimodal_labels = [label for label, score in bimodal_scores.items() if score >= bimodal_threshold]

        # Stage 2: Compute rare_label_score for remaining labels
        remaining_labels = [label for label in query_labels if str(label) not in bimodal_labels]
        rare_scores = {}

        for label in remaining_labels:
            # Find index of this label
            label_idx = np.where(query_labels == label)[0][0]
            similarity_scores = similarity_matrix[:, label_idx]
            rare_score = self._compute_bimodal_score(similarity_scores, "rare_label_score")
            rare_scores[str(label)] = rare_score

        # Select labels above rare threshold
        rare_labels = [label for label, score in rare_scores.items() if score >= rare_threshold]

        # Final remaining labels
        final_remaining = [label for label in remaining_labels if str(label) not in rare_labels]

        # Combine all scores for inspection
        all_scores = {}
        for label in query_labels:
            label_str = str(label)
            all_scores[label_str] = {
                "separation_score": bimodal_scores[label_str],
                "rare_label_score": rare_scores.get(label_str, 0.0),
                "stage": "bimodal"
                if label_str in bimodal_labels
                else "rare"
                if label_str in rare_labels
                else "remaining",
            }

        return {
            "bimodal_labels": bimodal_labels,
            "rare_labels": rare_labels,
            "remaining_labels": final_remaining,
            "all_scores": all_scores,
        }
