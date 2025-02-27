import logging
import os
import random

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import trimap
import umap
from matplotlib.patches import Patch
from scipy.sparse import issparse
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def plot_umap(
    adata: anndata.AnnData,
    embedding_key: str | None = None,
    color_key: str | list[str] | None = None,
    title: str = None,
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
    """Plot a umap with custom options.

    Plots a UMAP visualization for the given annotated data using Scanpy. This
    version supports multiple color keys and an optional embedding key.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    embedding_key : str or None, optional
        Key in ``.obsm`` to use for UMAP. If None, the default representation
        (no ``use_rep``) is used, relying on the existing or newly computed
        neighbors graph. Defaults to None.
    color_key : str or list of str or None, optional
        One or more keys in ``adata.obs`` to color points. Defaults to None.
    sample_size : int, optional
        Number of samples to randomly select for plotting. If None, use all data. Defaults to None.
    save_plot : bool, optional
        Whether to save the plot. Defaults to False.
    save_dir : str, optional
        Directory where to save the plot (if ``save_plot=True``). Defaults to None.
    nametag : str, optional
        Tag to add to the saved plot name. Defaults to None.
    figsize : tuple of (float, float), optional
        Size of the figure (width, height). Defaults to (8, 8).
    dpi : int, optional
        Dots per inch for the plot. Defaults to 300.
    point_size : int, optional
        Size of the points in the plot. Defaults to None.
    frameon : bool, optional
        Whether to draw a frame around the plot. Defaults to False.
    legend_fontsize : int, optional
        Font size for the legend. Defaults to 10.
    font_weight : str, optional
        Weight for the font used in the plot. Defaults to "bold".
    legend_loc : str, optional
        Location of the legend. Defaults to "right margin". For no legend, set to None.
    save_format : str, optional
        Format to save the plot (e.g., "png", "pdf"). Defaults to "png".
    **kwargs : dict
        Additional keyword arguments passed to ``sc.pp.neighbors`` if new neighbors
        need to be computed (for example, specifying ``n_neighbors=20``).

    Returns
    -------
    None
        Displays or saves the UMAP plot.

    Notes
    -----
    - If ``sample_size`` is provided and is smaller than the total number of cells,
      a random subset is drawn for visualization.
    - If ``embedding_key`` is None, the default neighbors graph is computed or used
      (no ``use_rep``). Otherwise, it is passed via ``use_rep=embedding_key``.
    - If ``color_key`` is a list, each element in the list is used to generate a
      separate coloring on the same UMAP.

    Examples
    --------
    >>> plot_umap(
    ...     adata=my_adata,
    ...     embedding_key="X_pca",
    ...     color_key=["origin", "cell_type", "cluster"],
    ...     sample_size=1000,
    ...     save_plot=False,
    ... )
    """
    logger.info("Starting UMAP computation and plotting.")
    if title is None:
        title = f"UMAP of {embedding_key if embedding_key else 'default'}"

    # If subset sampling is desired
    if sample_size is not None and sample_size < adata.n_obs:
        # Optional data consolidation step from your codebase:
        from mmcontext.pp.utils import consolidate_low_frequency_categories

        # Randomly sample the cells
        adata = adata[np.random.choice(adata.obs_names, sample_size, replace=False)]
        # Example: removing low-frequency categories in color_key if needed
        if color_key is not None:
            # If color_key is a single string, wrap it in a list for convenience
            keys_to_check = [color_key] if isinstance(color_key, str) else color_key
            adata = consolidate_low_frequency_categories(adata, columns=keys_to_check, threshold=10, remove=True)

    try:
        # Compute neighbors
        if embedding_key is not None:
            logger.info(f"Computing neighbors with use_rep='{embedding_key}'.")
            sc.pp.neighbors(adata, use_rep=embedding_key, metric="cosine", **kwargs)
        else:
            logger.info("Embedding key is None; using default neighbors.")
            sc.pp.neighbors(adata, metric="cosine", **kwargs)

        # Compute UMAP if not already present
        # if "X_umap" not in adata.obsm:
        sc.tl.umap(adata, random_state=42)

        # Prepare figure for plotting
        plt.figure(figsize=figsize)
        # Convert color_key to list if it is a single string
        if isinstance(color_key, str):
            color_key = [color_key]
        if "cell_type_colors" in adata.uns:
            del adata.uns["cell_type_colors"]
        sc.pl.umap(
            adata,
            color=color_key,
            title=title,
            show=False,
            save=None,
            s=point_size,
            frameon=frameon,
            legend_fontsize=legend_fontsize,
            legend_loc=legend_loc,
        )

        # Apply font properties globally
        plt.rc("font", size=legend_fontsize, weight=font_weight)

        # Save or show the plot
        if save_plot:
            if not save_dir:
                logger.error("Save directory is not specified.")
                raise ValueError("Save directory is not specified.")
            os.makedirs(save_dir, exist_ok=True)
            basename = "UMAP"
            if nametag:
                basename += f"_{nametag}"
            # Include embedding key in filename if not None
            emb_id = embedding_key if embedding_key else "default"
            file_path = os.path.join(save_dir, f"{basename}_{emb_id}.{save_format}")
            plt.savefig(file_path, bbox_inches="tight", dpi=dpi)
            plt.close()
            logger.info(f"UMAP plot saved successfully at {file_path}.")
        else:
            plt.show()
            logger.info("Displayed UMAP plot interactively.")

    except Exception as e:
        logger.error(f"An error occurred while generating UMAP plot: {e}", exc_info=True)
        raise e


def plot_query_scores_umap(
    adata: anndata.AnnData,
    embedding_key: str | None = None,
    save_plot: bool = False,
    save_dir: str | None = None,
    nametag: str | None = None,
    **plot_kwargs,
):
    """
    Plot UMAPs colored by each query's similarity scores from `adata.obs["query_scores"]`.

    For each query in `queries`, this function:
      1) Copies the original AnnData object (so the original isn't modified),
      2) Extracts that query's similarity scores from `adata.obs["query_scores"]`,
      3) Stores them in a temporary column in the copy's .obs,
      4) Calls `plot_umap` using that temporary column as the color key,
      5) Optionally saves or shows the plot, inserting the query name in the nametag.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object that must contain `obs["query_scores"]`, where each
        element is a dictionary mapping {query_string: similarity_score}.
    queries : list of str or str
        One or more query names to plot. Must be keys in each entry of
        `adata.obs["query_scores"]`.
    embedding_key : str or None, optional
        Key in ``.obsm`` to use for the embedding. If None, uses default neighbors.
    save_plot : bool, optional
        Whether to save the plot instead of displaying it. Defaults to False.
    save_dir : str or None, optional
        Directory in which to save plots (used only if `save_plot=True`). Defaults to None.
    nametag : str or None, optional
        A base tag to include in the saved plot's file name. The query name is
        also appended automatically.
    **plot_kwargs : dict
        Additional keyword arguments passed to `plot_umap`. For example:
        - point_size : int
        - frameon : bool
        - legend_fontsize : int
        - device : str (if your `plot_umap` or neighbors step needs it)
        etc.

    Returns
    -------
    None
        Generates one UMAP per query, either shown interactively or saved to file.

    Notes
    -----
    - The function checks for `adata.obs["query_scores"]`. If it's absent or not
      structured as expected, raises a ValueError.
    - Each query produces one plot. The temporary column is named `_temp_query_score`
      (overwritten each iteration), so only one color is plotted at a time.
    - Because we copy `adata` internally, no permanent column is added to the original object.

    Examples
    --------
    >>> # Suppose adata.obs["query_scores"] exists, with keys: ["t-cell", "b-cell"]
    >>> plot_query_scores_umap(
    ...     adata,
    ...     queries=["t-cell", "b-cell"],
    ...     embedding_key="X_umap",
    ...     save_plot=True,
    ...     save_dir="plots",
    ...     point_size=30,
    ... )
    # This generates two UMAP plots: one colored by "t-cell" scores, the other by "b-cell".
    """
    # Validate presence of query_scores in adata.obs
    if "query_scores" not in adata.obs:
        raise ValueError("`adata.obs['query_scores']` not found. Cannot plot query-based scores.")
    # 1) Make a copy so we don't modify the original
    adata_copy = adata.copy()
    queries = adata.obs["query_scores"].iloc[0].keys()
    # We'll loop over each query and generate a separate UMAP
    for query in queries:
        # 2) Build a list of scores for the current query
        query_vals = []
        for score_dict in adata.obs["query_scores"]:
            if not isinstance(score_dict, dict):
                raise ValueError("Each entry in `adata.obs['query_scores']` must be a dict of {query: score}.")
            if query not in score_dict.keys():
                raise ValueError(f"Query '{query}' not found in one of the 'query_scores' dicts.")
            query_vals.append(score_dict[query])

        # 3) Store them in a temporary obs column
        adata_copy.obs["_temp_query_score"] = query_vals

        # 4) Build a combined nametag that includes the query
        combined_tag = f"{nametag}_{query}" if nametag else f"{query}"

        # 5) Call the existing plot_umap function, color by our temporary column
        plot_umap(
            adata_copy,
            embedding_key=embedding_key,
            color_key="_temp_query_score",
            save_plot=save_plot,
            save_dir=save_dir,
            title=query,
            nametag=combined_tag,
            **plot_kwargs,
        )

        # The function internally shows or saves the plot, then returns.
        # We discard `adata_copy` so no changes persist on the original.


def plot_clustered_heatmap(
    adata: sc.AnnData,
    num_hvgs: int = 20,
    clustering_method: str = "leiden",
    resolution: float = 1.0,
    figsize: tuple = (10, 12),
    cmap: str = "coolwarm",
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """Plot a heatmap

    Plot a heatmap with cells on the y-axis and genes on the x-axis.
    Only the cell dimension is clustered, not the gene dimension. Each cell
    is labeled (text) by its cell_type, and colored by its origin.

    Parameters
    ----------
    adata : scanpy.AnnData
        Annotated data matrix with observations (cells) in .obs and variables (genes) in .var.
        Must include 'origin' and 'cell_type' in adata.obs.
    num_hvgs : int, optional
        Number of highly variable genes to select (based on cells with origin == '0').
        Defaults to 20.
    clustering_method : str, optional
        Method to use for clustering. Defaults to "leiden".
    resolution : float, optional
        Resolution for clustering. Higher values produce more clusters.
        Defaults to 1.0.
    figsize : tuple, optional
        Width, height of the figure in inches. Defaults to (10, 12).
    cmap : str, optional
        Colormap to use for the heatmap. Defaults to "coolwarm".
    show : bool, optional
        Whether to display the plot interactively. Defaults to True.
    save_path : str, optional
        Path to save the resulting figure. If None, the plot is not saved.
        Defaults to None.

    Returns
    -------
    None
        Displays a clustermap with cells in rows, genes in columns,
        textual annotation of cell types, and color annotation of origins.

    Notes
    -----
    - This function will select HVGs based on cells whose `.obs["origin"] == "0"`,
      then extract those genes from the full dataset.
    - No scaling is performed, so the data are displayed as-is.
    - Clustering is only performed on the row dimension (cells). Genes are not clustered.
    - The row labels (yticks) show each cell's cell_type.
    - The row_colors argument in Seaborn is used to visualize each cell's origin.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting plot_cells_by_genes_heatmap.")

    # Validate that needed columns are present
    required_obs = ["origin", "cell_type"]
    missing_obs = [obs for obs in required_obs if obs not in adata.obs]
    if missing_obs:
        msg = f"Missing required annotations in adata.obs: {missing_obs}"
        logger.error(msg)
        raise ValueError(msg)

    # Subset to cells with origin == "0" for HVG selection
    original_cells = adata[adata.obs["origin"] == "0"].copy()
    if original_cells.n_obs == 0:
        msg = "No cells with origin == '0' found in adata."
        logger.error(msg)
        raise ValueError(msg)

    # Compute HVGs from just these original cells
    sc.pp.highly_variable_genes(original_cells, n_top_genes=num_hvgs, inplace=True)
    hvgs = original_cells.var_names[original_cells.var["highly_variable"]].tolist()

    if not hvgs:
        msg = "No HVGs found. Adjust num_hvgs or check your data."
        logger.error(msg)
        raise ValueError(msg)

    logger.info(f"Selected {len(hvgs)} HVGs based on cells with origin == '0'.")

    # Filter the full adata to these genes
    # (No scaling, per your request)
    genes_in_full = [g for g in hvgs if g in adata.var_names]
    adata_hvgs = adata[:, genes_in_full].copy()

    # Optionally do clustering on cells
    # (We do not cluster on genes, so col_cluster=False later)
    sc.pp.neighbors(adata_hvgs)
    if clustering_method.lower() == "leiden":
        sc.tl.leiden(adata_hvgs, resolution=resolution)
    else:
        sc.tl.louvain(adata_hvgs, resolution=resolution)

    # Convert expression to a dense dataframe if needed
    if issparse(adata_hvgs.X):
        data_matrix = adata_hvgs.X.toarray()
    else:
        data_matrix = adata_hvgs.X

    expression_df = pd.DataFrame(
        data_matrix,
        index=adata_hvgs.obs_names,  # cell IDs
        columns=adata_hvgs.var_names,  # gene names
    )

    # -- Prepare row labels (text) for cell_type
    # We will re-order these labels after clustering to match the row dendrogram
    cell_types = adata_hvgs.obs["cell_type"].copy()

    # -- Prepare row_colors for 'origin'
    origins = adata_hvgs.obs["origin"].unique().tolist()
    # Create a color palette for each distinct origin
    palette = sns.color_palette("hsv", len(origins))
    origin_color_map = dict(zip(origins, palette, strict=False))
    # set the origin as a string
    adata_hvgs.obs["origin"] = adata_hvgs.obs["origin"].astype(str)
    row_colors = adata_hvgs.obs["origin"].map(origin_color_map)
    # set the origin back to category
    adata_hvgs.obs["origin"] = adata_hvgs.obs["origin"].astype("category")

    # Build legend handles from your origin_color_map, which looks like:
    # {'0': (0.5176, 1.0, 0.0), '2': (0.0, 1.0, 0.9647), '1': (0.4470, 0.0, 1.0)}
    handles = [Patch(facecolor=color, label=f"origin={orig}") for orig, color in origin_color_map.items()]

    # Create the cluster map
    # Cluster only cells (rows), hence col_cluster=False.
    # This puts cells as rows and genes as columns by default.
    # standard_scale=None ensures we do not do any scaling inside clustermap.
    g = sns.clustermap(
        expression_df,
        cmap=cmap,
        row_cluster=True,
        col_cluster=False,
        row_colors=row_colors,
        figsize=figsize,
        xticklabels=True,  # Gene names on x-axis
        yticklabels=False,  # We'll manually set row labels
        dendrogram_ratio=(0.15, 0.02),
        cbar_pos=(0.02, 0.8, 0.02, 0.15),  # Move colorbar if desired
        standard_scale=None,
    )
    # Place the legend on the figure or a specific axis
    # Here we attach it to the heatmap axis:
    g.ax_heatmap.legend(
        handles=handles,
        title="Origin",
        bbox_to_anchor=(0, 1.2),  # Adjust placement if desired
        loc="upper left",
        borderaxespad=0,
    )

    g.figure.suptitle("Cells vs. HVGs: Clustered on Cells Only", y=1.02)

    # Re-order the cell_type labels to match the cluster order
    # The row indices after clustering are in g.dendrogram_row.reordered_ind
    reordered_row_indices = g.dendrogram_row.reordered_ind
    cell_types_ordered = cell_types.iloc[reordered_row_indices]

    # Set new y-tick labels (cell_type) with correct spacing
    ax = g.ax_heatmap
    ax.set_yticks(np.arange(len(expression_df)) + 0.5)  # center labels
    ax.set_yticklabels(cell_types_ordered, rotation=0)  # horizontal text

    # (Optional) if you want smaller or bigger text on the y-axis
    # for item in ax.get_yticklabels():
    #     item.set_size(8)

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Heatmap saved at {save_path}")

    if show:
        plt.show()
    else:
        logger.info("Plot generated but not displayed (show=False).")


def plot_grouped_bar_chart(
    data: pd.DataFrame,
    save_plot: bool = False,
    save_path: str | None = None,
    figsize: tuple = (12, 6),
    title: str = "Comparison of Metrics by Type",
    xlabel: str = "Metric",
    ylabel: str = "Value",
    rotation: int = 45,
    legend_title: str = "Type",
    bbox_to_anchor: tuple = (1.05, 1),
    loc: str = "upper left",
    edgecolor: str = "black",
) -> None:
    """Plot a grouped bar chart of various metrics.

    Plot a grouped bar chart of various metrics, with metrics on the x-axis and
    different 'type' values shown as different colors. Optionally save the plot to a file.

    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe containing columns for 'type' and the metrics you want to plot.
        It must include at least these columns:
        ['type', 'ARI', 'NMI', 'ASW', 'Isolated_Labels_ASW', 'Isolated_Labels_F1',
         'Bio_Conservation_Score', 'Silhouette_Batch', 'Graph_Connectivity',
         'Batch_Integration_Score', 'Overall_Score'].
    save_plot : bool, optional
        Whether to save the plot to a file. Defaults to False.
    save_path : str, optional
        The file path where the plot should be saved. Required if `save_plot` is True.
        Supports formats like 'png', 'pdf', etc. Defaults to None.
    figsize : tuple, optional
        Size of the figure (width, height). Defaults to (12, 6).
    title : str, optional
        Title of the plot. Defaults to "Comparison of Metrics by Type".
    xlabel : str, optional
        Label for the x-axis. Defaults to "Metric".
    ylabel : str, optional
        Label for the y-axis. Defaults to "Value".
    rotation : int, optional
        Rotation angle for x-axis labels. Defaults to 45.
    legend_title : str, optional
        Title for the legend. Defaults to "Type".
    bbox_to_anchor : tuple, optional
        Position of the legend. Defaults to (1.05, 1).
    loc : str, optional
        Location of the legend. Defaults to "upper left".
    edgecolor : str, optional
        Edge color for the bars. Defaults to "black".

    Returns
    -------
    None
        Displays the grouped bar chart and optionally saves it to a file.

    Examples
    --------
    >>> data = pd.DataFrame(
    ...     {
    ...         "type": [
    ...             "raw",
    ...             "reconstructed1",
    ...             "reconstructed2",
    ...             "embedding_scvi",
    ...             "embedding_data_encoder_mod_emb",
    ...             "embedding_context_encoder_mod_emb",
    ...         ],
    ...         "ARI": [0.319397, -0.001102, 0.000448, 0.388573, 0.403889, 0.403889],
    ...         "NMI": [0.650722, 0.177398, 0.198209, 0.701891, 0.690134, 0.690134],
    ...         "ASW": [0.412581, 0.435318, 0.416072, 0.507724, 0.505978, 0.505978],
    ...         "Isolated_Labels_ASW": [0.508366, 0.439518, 0.425415, 0.552753, 0.547546, 0.547546],
    ...         "Isolated_Labels_F1": [0.363206, 0.114639, 0.130031, 0.362576, 0.399302, 0.399302],
    ...         "Bio_Conservation_Score": [0.450854, 0.233154, 0.234035, 0.502703, 0.509370, 0.509370],
    ...         "Silhouette_Batch": [0.712724, 0.879405, 0.889953, 0.797206, 0.788198, 0.788198],
    ...         "Graph_Connectivity": [0.889386, 0.239613, 0.266225, 0.912457, 0.914519, 0.914519],
    ...         "Batch_Integration_Score": [0.801055, 0.559509, 0.578089, 0.854832, 0.851359, 0.851359],
    ...         "Overall_Score": [0.590935, 0.363696, 0.371656, 0.643555, 0.646165, 0.646165],
    ...     }
    ... )
    >>> plot_grouped_bar_chart(data, save_plot=True, save_path="metrics_comparison.png")
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating grouped bar chart with metrics on x-axis and type categories as color groups.")

    # List of metric columns (you can adjust this to your own subset if desired)
    metric_cols = [
        "ARI",
        "NMI",
        "ASW",
        "Isolated_Labels_ASW",
        "Isolated_Labels_F1",
        "Bio_Conservation_Score",
        "Silhouette_Batch",
        "Graph_Connectivity",
        "Batch_Integration_Score",
        "Overall_Score",
    ]

    # Validate required columns
    required_columns = ["type"] + metric_cols
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"The following required columns are missing from the data: {missing_columns}")
        raise ValueError(f"The following required columns are missing from the data: {missing_columns}")

    # Melt the DataFrame to long format:
    # - 'type' is our category
    # - 'metric' is the variable name
    # - 'value' is the metric value
    df_melted = data.melt(id_vars=["type"], value_vars=metric_cols, var_name="metric", value_name="value")

    # Create the grouped bar plot
    plt.figure(figsize=figsize)
    sns.barplot(data=df_melted, x="metric", y="value", hue="type", edgecolor=edgecolor)

    plt.title(title)
    plt.xticks(rotation=rotation, ha="right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title=legend_title, bbox_to_anchor=bbox_to_anchor, loc=loc)
    plt.tight_layout()

    if save_plot:
        if not save_path:
            logger.error("save_path must be provided if save_plot is True.")
            raise ValueError("save_path must be provided if save_plot is True.")
        try:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Grouped bar chart saved successfully at {save_path}.")
        except Exception as e:
            logger.error(f"Failed to save the plot at {save_path}: {e}")
            raise e
    else:
        plt.show()
        logger.info("Displayed grouped bar chart interactively.")


def visualize_embedding_clusters(
    df: pd.DataFrame,
    method: str = "umap",
    metric: str = "euclidean",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    n_samples: int = 50,
    # Plot appearance & saving
    nametag: str | None = None,
    figsize: tuple = (10, 8),
    dpi: int = 300,
    point_size: float = 50.0,
    frameon: bool = False,
    legend_fontsize: int = 10,
    font_weight: str = "bold",
    legend_loc: str = "best",
    save_format: str = "png",
    save_plot: bool = False,
    save_path: str | None = None,
):
    """
    Visualize embeddings using UMAP or TRIMAP, with custom plot appearance and saving options.

    With different shapes for embedding types ('omics' or 'text') and different colors for each sample ID.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing:
            - 'embedding': The actual embedding vectors (np.array or list of floats).
            - 'embedding_type': A categorical field (e.g., 'omics' or 'text').
            - 'sample_id': A unique identifier for each sample.
    method : str, optional
        Dimensionality reduction method ('umap' or 'trimap'). Default is 'umap'.
    metric : str, optional
        Distance metric for dimensionality reduction (e.g., 'euclidean' or 'cosine').
        For UMAP, it sets the 'metric' parameter. For TRIMAP, it sets 'distance'.
        Default is 'euclidean'.
    n_neighbors : int, optional
        Number of neighbors for UMAP or TRIMAP. Default is 15.
    min_dist : float, optional
        Minimum distance between points in UMAP. Ignored by TRIMAP. Default is 0.1.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    n_samples : int, optional
        Number of unique sample IDs to randomly select for plotting. Default is 50.
    nametag : str or None, optional
        A tag to add to the saved plot name if save_plot=True. Default is None.
    figsize : tuple, optional
        The size of the figure (width, height). Default is (10, 8).
    dpi : int, optional
        Dots per inch for the figure. Default is 300.
    point_size : float, optional
        Marker size for the scatter plot. Default is 50.0.
    frameon : bool, optional
        Whether to draw a frame around the plot. Default is False.
    legend_fontsize : int, optional
        Font size for the legend. Default is 10.
    font_weight : str, optional
        Weight for the font used in the plot. Default is "bold".
    legend_loc : str, optional
        Location of the legend (e.g., 'best', 'upper right'). Default is "best".
    save_format : str, optional
        File format if the plot is saved, e.g. "png" or "pdf". Default is "png".
    save_plot : bool, optional
        Whether to save the plot to disk instead of showing it interactively. Default is False.
    save_path : str or None, optional
        Path to save the figure. If save_plot=True, this must be provided.
        The final saved filename will be appended with nametag and `save_format` if needed.

    Returns
    -------
    None
        Displays or saves the plot.

    Notes
    -----
    - This function randomly samples 'n_samples' unique sample_ids from 'df'.
      The dimensionality reduction is applied only on this subset.
    - Marker shapes are currently mapped as {'omics': 'o', 'text': 's'}.
    - A minimal legend is created to illustrate the shape (embedding_type)
      and color (sample_id). Additional legend modifications can be done as needed.
    """
    # Check and prepare save parameters
    if save_plot and (not save_path):
        raise ValueError("`save_path` must be provided when `save_plot=True`.")

    # Randomly select unique sample IDs
    all_sample_ids = df["sample_id"].unique()
    if n_samples > len(all_sample_ids):
        logger.warning(
            f"Requested n_samples ({n_samples}) is greater than total unique sample_ids ({len(all_sample_ids)}). "
            "Using all sample_ids instead."
        )
        n_samples = len(all_sample_ids)
    selected_sample_ids = np.random.choice(all_sample_ids, size=n_samples, replace=False)
    df_sub = df[df["sample_id"].isin(selected_sample_ids)].reset_index(drop=True)

    # Extract embeddings
    embeddings = np.vstack(df_sub["embedding"].values)

    # Initialize reducer
    if method.lower() == "umap":
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)
    elif method.lower() == "trimap":
        reducer = trimap.TRIMAP(
            n_inliers=n_neighbors,  # approx. analogous to neighbors in UMAP
            n_random=n_neighbors,
            distance=metric,
            random_state=random_state,
        )
    else:
        raise ValueError("Method must be 'umap' or 'trimap'.")

    # Fit and transform
    logger.info(f"Applying {method.upper()} to subset of size={len(df_sub)} with metric={metric}...")
    embedding_2d = reducer.fit_transform(embeddings)

    # Prepare color palette for sample IDs
    unique_samples = df_sub["sample_id"].unique()
    color_palette = sns.color_palette("husl", len(unique_samples))
    sample_color_map = {sid: color_palette[i] for i, sid in enumerate(unique_samples)}

    # Prepare marker shapes
    marker_map = {"omics": "o", "text": "s"}

    # Create the plot
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    # Adjust frame, font, etc.
    plt.rc("font", size=legend_fontsize, weight=font_weight)
    if not frameon:
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Plot each combination of sample_id and embedding_type
    for s_id in unique_samples:
        for emb_type in df_sub["embedding_type"].unique():
            subset = df_sub[(df_sub["sample_id"] == s_id) & (df_sub["embedding_type"] == emb_type)]
            if subset.empty:
                continue
            indices = subset.index.values
            ax.scatter(
                embedding_2d[indices, 0],
                embedding_2d[indices, 1],
                c=[sample_color_map[s_id]],
                marker=marker_map.get(emb_type, "o"),
                s=point_size,
                alpha=0.7,
                edgecolors="k",
            )

    ax.set_title(f"Embedding Visualization using {method.upper()} ({metric} distance)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    # Minimal custom legend
    # The user might prefer a more advanced legend (one for color, one for shape).
    # As a quick approach, we'll show shape references for the embedding_type:
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    # Prepare shape legend entries
    shape_legend = []
    for etype in df_sub["embedding_type"].unique():
        shape_legend.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                marker=marker_map.get(etype, "o"),
                linestyle="None",
                markersize=10,
                label=f"{etype}",
                markeredgecolor="k",
            )
        )
    ax.legend(handles=shape_legend, loc=legend_loc, fontsize=legend_fontsize)

    # Handle saving
    if save_plot:
        # If user wants an extra nametag in the filename:
        base, ext = os.path.splitext(save_path)
        # If user didn't provide an extension, or wants to override with save_format:
        if not ext:
            ext = f".{save_format}"
        outname = base
        if nametag:
            outname += f"_{nametag}"
        outpath = f"{outname}{ext}"

        plt.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Plot saved to {outpath}")
    else:
        plt.show()
        logger.info("Displayed embedding clusters interactively.")


def plot_embedding_similarity(
    df: pd.DataFrame,
    emb1_type: str = "omics",
    emb2_type: str = "text",
    n_samples: int = 10,
    seed: int = 42,
    label_key: str | None = None,
    # Additional plotting & saving parameters
    save_plot: bool = False,
    save_dir: str | None = None,
    nametag: str | None = None,
    figsize: tuple = (10, 8),
    dpi: int = 300,
    font_scale: float = 1.0,
    save_format: str = "png",
):
    """
    Plot the pairwise cosine similarity matrix between two embedding types.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing:
          - 'embedding': The actual embedding vectors.
          - 'embedding_type': One of e.g. ['omics', 'text'] (str).
          - 'sample_id': A unique identifier for each sample.
          - Optionally, additional columns like label_key for annotated labels.
    emb1_type: str, optional
        The embedding type to appear on the y-axis (rows). Default is 'omics'.
    emb2_type: str, optional
        The embedding type to appear on the x-axis (columns). Default is 'text'.
    n_samples: int, optional
        Number of samples to randomly select for plotting. If 0, use all samples.
        If >10, axis labels and annotations are excluded from the heatmap (for clarity).
        Default is 10.
    seed: int, optional
        Random seed for reproducibility of sampling. Default is 42.
    label_key: str, optional
        Column name in `df` to use for axis labels if `n_samples <= 10`.
        If None, uses 'sample_id'. Default is None.

    save_plot: bool, optional
        Whether to save the plot instead of displaying it. Defaults to False.
    save_dir: str, optional
        Directory where the plot should be saved (if `save_plot=True`). Defaults to None.
    nametag: str, optional
        An extra tag to insert into the saved filename. Defaults to None.
    figsize: tuple, optional
        The size of the figure (width, height). Defaults to (10, 8).
    dpi: int, optional
        Dots per inch for the figure. Defaults to 300.
    font_scale: float, optional
        Seaborn font scale for controlling text size in the heatmap. Defaults to 1.0.
    save_format: str, optional
        Image format to save the plot (e.g. "png", "pdf"). Defaults to "png".

    Returns
    -------
    None
        Displays or saves a heatmap of pairwise cosine similarities.

    Notes
    -----
    1) Randomly samples 'n_samples' unique sample IDs from `df["sample_id"]` (if n_samples>0).
    2) Splits the data into two sets: one with `emb1_type` embeddings and one with `emb2_type`.
    3) Computes cosine similarity between those sets.
    4) If `n_samples > 10`, it omits text labels in the heatmap to keep things tidy.
    5) If `n_samples <= 10`, it uses `label_key` (or sample_id if label_key is None) for axis labels.
    """
    # Set random seed for reproducible sampling
    np.random.seed(seed)

    # Possibly sample a subset of data
    if n_samples > 0:
        unique_ids = df["sample_id"].unique()
        if n_samples > len(unique_ids):
            n_samples = len(unique_ids)  # avoid sampling error
        chosen_ids = np.random.choice(unique_ids, size=n_samples, replace=False)
        df = df[df["sample_id"].isin(chosen_ids)]

    # Use a Seaborn context manager to adjust font scale
    with sns.plotting_context("notebook", font_scale=font_scale):
        # Align data by sample_id
        df = df.set_index("sample_id", drop=False)  # keep sample_id as a column and index
        emb1_df = df[df["embedding_type"] == emb1_type]
        emb2_df = df[df["embedding_type"] == emb2_type]

        # Only consider intersection of sample IDs present in both emb1 and emb2
        common_ids = emb1_df.index.intersection(emb2_df.index)
        emb1_values = np.vstack(emb1_df.loc[common_ids, "embedding"].values)
        emb2_values = np.vstack(emb2_df.loc[common_ids, "embedding"].values)

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(emb1_values, emb2_values)

        # Create figure
        plt.figure(figsize=figsize, dpi=dpi)

        # If n_samples > 10, omit axis tick labels
        if n_samples > 10:
            sns.heatmap(
                similarity_matrix,
                annot=False,
                fmt=".2f",
                cmap="coolwarm",
                xticklabels=False,
                yticklabels=False,
            )
            plt.xlabel("")
            plt.ylabel("")
        else:
            # Use label_key for labels if provided, else sample_id
            if label_key is not None:
                labels = emb1_df.loc[common_ids, label_key]
            else:
                labels = common_ids

            sns.heatmap(
                similarity_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                xticklabels=labels,
                yticklabels=labels,
            )
            plt.xlabel(emb2_type)
            plt.ylabel(emb1_type)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)

        plt.title(f"Pairwise Cosine Similarity: {emb1_type} vs {emb2_type}")

        if save_plot:
            if save_dir is None:
                raise ValueError("Must provide `save_dir` when `save_plot=True`.")
            os.makedirs(save_dir, exist_ok=True)

            # Build filename
            base_name = f"embedding_similarity_{emb1_type}_vs_{emb2_type}"
            if nametag:
                base_name += f"_{nametag}"
            save_path = os.path.join(save_dir, f"{base_name}.{save_format}")

            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            print(f"Saved similarity heatmap to {save_path}")
        else:
            plt.show()
