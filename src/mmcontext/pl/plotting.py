import logging
import os

import anndata
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


def plot_umap(
    adata: anndata.AnnData,
    embedding_key: str,
    color_group: str,
    sample_size: int = None,
    save_plot: bool = False,
    save_dir: str = None,
    nametag: str = None,
    figsize: tuple = (8, 8),
    dpi: int = 300,
    point_size: int = None,
    frameon: bool = False,
    legend_fontsize: int = 10,
    font_weight: str = "bold",
    legend_loc: str = "right margin",
    save_format: str = "png",
    **kwargs,
):
    """
    Plots UMAP visualization for the given annotated data using Scanpy.

    Parameters
    ----------
    adata
        Annotated data matrix.
    embedding_key
        Key in `.obsm` to use for UMAP.
    color_group
        Key in `.obs` to color points.
    sample_size : int, optional
        Number of samples to randomly select for plotting. If None, use all data.
    save_plot
        Whether to save the plot.
    save_dir
        Directory where to save the plot.
    name_tag
        Tag to add to the saved plot name.
    figsize
        Size of the figure (width, height).
    dpi
        Dots per inch for the plot.
    point_size
        Size of the points in the plot.
    frameon
        Whether to draw a frame around the plot.
    legend_fontsize
        Font size for the legend.
    legend_loc
        Location of the legend.
    save_format
        Format to save the plot (png, pdf, etc.).
    **kwargs: Additional arguments to `sc.pp.neighbors`.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting UMAP computation and plotting.")
    if sample_size is not None and sample_size < len(adata):
        from mmcontext.pp.utils import consolidate_low_frequency_categories

        adata = adata[np.random.choice(adata.obs.index, sample_size, replace=False)]
        adata = consolidate_low_frequency_categories(adata, columns=[color_group], threshold=10, remove=True)

    try:
        # Compute UMAP
        sc.pp.neighbors(adata, use_rep=embedding_key, **kwargs)
        sc.tl.umap(adata, random_state=42)
        # Prepare figure for plotting
        plt.figure(figsize=figsize)
        sc.pl.umap(
            adata,
            color=color_group,
            title=f"UMAP of {embedding_key}",
            show=False,
            save=None,
            s=point_size,
            frameon=frameon,
            legend_fontsize=legend_fontsize,
            legend_loc=legend_loc,
        )

        # Apply font properties globally
        plt.rc("font", size=legend_fontsize, weight=font_weight)  # Adjust font size and weight globally

        # Save or show the plot
        if save_plot:
            if not save_dir:
                logger.error("Save directory is not specified.")
                raise ValueError("Save directory is not specified.")
            os.makedirs(save_dir, exist_ok=True)
            basename = "UMAP"
            if nametag:
                basename += f"_{nametag}"
            file_path = os.path.join(save_dir, f"{basename}_{embedding_key}_{color_group}.{save_format}")
            plt.savefig(file_path, bbox_inches="tight", dpi=dpi)
            plt.close()
            logger.info(f"UMAP plot saved successfully at {file_path}.")
        else:
            plt.show()
            logger.info("Displayed UMAP plot interactively.")

    except Exception as e:
        logger.error(f"An error occurred while generating UMAP plot: {e}", exc_info=True)
        raise
