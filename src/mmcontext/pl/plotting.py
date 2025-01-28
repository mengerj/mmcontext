import logging
import os

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.patches import Patch
from scipy.sparse import issparse

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
        Location of the legend. Defaults to "right margin".
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
            sc.pp.neighbors(adata, use_rep=embedding_key, **kwargs)
        else:
            logger.info("Embedding key is None; using default neighbors.")
            sc.pp.neighbors(adata, **kwargs)

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
