# mmcontext/eval/umap_plotter.py
from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np

from mmcontext.eval.base import BaseEvaluator, EvalResult
from mmcontext.eval.registry import register
from mmcontext.eval.utils import LabelKind
from mmcontext.pl.plotting import plot_umap
from mmcontext.utils import consolidate_low_frequency_categories

logger = logging.getLogger(__name__)


@register
class UmapPlotter(BaseEvaluator):
    """
    Evaluator that generates UMAP plots for all embeddings in adata.obsm.

    This evaluator doesn't compute any metrics but generates UMAP visualizations
    for all embeddings found in the AnnData object, colored by the specified labels.
    """

    name = "UmapPlotter"
    requires_pair = False  # Can work with just one embedding
    produces_plot = True  # This evaluator is all about plotting

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
        """Compute method - just returns empty results since this evaluator is for plotting only."""
        return EvalResult()

    def plot(
        self,
        emb1: np.ndarray,
        out_dir: Path,
        *,
        emb2: np.ndarray = None,
        labels: np.ndarray,
        adata: ad.AnnData,
        label_kind: LabelKind,
        label_key: str,
        save_format: str = "pdf",
        figsize: tuple = (8, 8),
        dpi: int = 300,
        point_size: int = None,
        **kw,
    ) -> None:
        """
        Generate UMAP plots for all embeddings in adata.obsm.

        Parameters
        ----------
        emb1 : np.ndarray
            Primary embedding (will be stored as 'data_rep_embedding' in adata.obsm)
        out_dir : Path
            Directory to save plots
        emb2 : np.ndarray, optional
            Secondary embedding (will be stored as 'caption_embedding' in adata.obsm)
        labels : np.ndarray
            Labels for coloring the UMAP
        adata : ad.AnnData
            AnnData object to store embeddings and use for plotting
        label_kind : LabelKind
            Type of label (bio or batch)
        label_key : str
            Key for the labels in adata.obs
        save_format : str, optional
            Format to save plots (default: "pdf")
        figsize : tuple, optional
            Figure size (default: (8, 8))
        dpi : int, optional
            DPI for saved plots (default: 300)
        point_size : int, optional
            Size of points in UMAP (default: None, uses scanpy default)
        **kw
            Additional keyword arguments
        """
        # Make a copy of adata to avoid modifying the original
        adata_full = adata.copy()
        adata_copy = consolidate_low_frequency_categories(adata_full, label_key, 10)
        # Store the embeddings in obsm
        adata_copy.obsm["data_rep_embedding"] = emb1
        if emb2 is not None:
            adata_copy.obsm["caption_embedding"] = emb2

        # Create output directory
        out_dir.mkdir(parents=True, exist_ok=True)

        # Get all embedding keys from obsm
        embedding_keys = list(adata_copy.obsm.keys())

        logger.info(f"Generating UMAP plots for {len(embedding_keys)} embeddings: {embedding_keys}")

        # Loop over all embeddings and create UMAP plots
        for emb_key in embedding_keys:
            try:
                logger.info(f"Generating UMAP for embedding: {emb_key}")

                # Generate a clean title
                title = f"UMAP - {emb_key} ({label_key})"

                # Create nametag for filename
                nametag = f"{emb_key}_{label_key}_{label_kind.value}"

                # Call the plot_umap function
                plot_umap(
                    adata=adata_copy,
                    embedding_key=emb_key,
                    color_key=label_key,
                    title=title,
                    save_plot=True,
                    save_dir=str(out_dir),
                    nametag=nametag,
                    figsize=figsize,
                    dpi=dpi,
                    point_size=point_size,
                    save_format=save_format,
                    frameon=False,
                    legend_fontsize=10,
                    font_weight="bold",
                    legend_loc="right margin",
                )

                logger.info(f"✓ UMAP plot saved for {emb_key}")

            except Exception as e:
                logger.error(f"✗ Failed to generate UMAP for {emb_key}: {e}")
                continue

        logger.info(f"✓ All UMAP plots completed for {label_key} ({label_kind.value})")
