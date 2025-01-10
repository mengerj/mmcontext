import logging

import anndata
import numpy as np
import scanpy as sc


def combine_original_and_reconstructions(adata: anndata.AnnData, max_reconstructions: int = 3) -> anndata.AnnData:
    """Combine original and reconstructed data from anndata object

    Combine the original data in ``adata.X`` with up to ``max_reconstructions``
    from ``adata.layers[...]``, and mark each subset in ``.obs["origin"]``.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing the original data in ``adata.X`` and
        multiple reconstructed datasets in ``adata.layers[...]``.
    max_reconstructions : int, optional
        Maximum number of reconstructed layers to combine. Defaults to 3.

    Returns
    -------
    combined_adata : anndata.AnnData
        Combined AnnData containing rows from:
        - The original data in ``adata.X``
        - Up to ``max_reconstructions`` layers from ``adata.layers[...]``.
        The new combined object has an ``origin`` column in ``.obs`` indicating
        "original" or e.g. "reconstructed1", "reconstructed2", ...

    Notes
    -----
    Data reference: Single-cell RNA-seq data. Original data is in .X,
    reconstructed data is in .layers["reconstructedX"], etc.

    Examples
    --------
    >>> combined = combine_original_and_reconstructions(my_adata, max_reconstructions=2)
    >>> combined.obs["origin"].value_counts()
    """
    logger = logging.getLogger(__name__)
    logger.info("Combining original data with up to %d reconstruction layers.", max_reconstructions)

    # Identify the layers that match the pattern "reconstructed" (or any logic you prefer).
    # Here we simply look for any layer with 'reconstructed' in its name.
    # If you name them differently, adjust this filter.
    reconstruction_layers = [layer_name for layer_name in adata.layers.keys() if "reconstructed" in layer_name]

    # Sort them or pick them deterministically, e.g. "reconstructed1", "reconstructed2", etc.
    reconstruction_layers = sorted(reconstruction_layers)

    # Limit to max_reconstructions
    reconstruction_layers = reconstruction_layers[:max_reconstructions]

    if not reconstruction_layers:
        logger.warning("No reconstruction layers found matching 'reconstructed'. Returning original data only.")

    # We will create multiple AnnData objects (one for original, plus each reconstruction),
    # then concatenate them with `sc.concat`.
    adatas_to_concat = []

    # 1) Original
    original_adata = anndata.AnnData(X=adata.X.copy(), obs=adata.obs.copy(), var=adata.var.copy(), uns=adata.uns.copy())
    original_adata.obs["origin"] = "original"
    adatas_to_concat.append(original_adata)

    # 2) Each reconstruction
    for _idx, layer in enumerate(reconstruction_layers, start=1):
        logger.info("Processing reconstruction layer: %s", layer)
        recon_adata = anndata.AnnData(
            X=adata.layers[layer].copy(), obs=adata.obs.copy(), var=adata.var.copy(), uns=adata.uns.copy()
        )
        recon_adata.obs["origin"] = layer
        adatas_to_concat.append(recon_adata)

    logger.info("Concatenating %d AnnData objects.", len(adatas_to_concat))
    combined_adata = anndata.concat(
        adatas_to_concat,
        join="inner",  # or "outer" to include all genes, if you prefer
        label="origin",  # a new column in obs that identifies each chunk
        keys=None,  # rely on the existing "origin" column if you prefer
        merge="unique",
        index_unique=None,
    )

    logger.info("Combination complete. Returning combined AnnData with shape %s.", combined_adata.shape)
    return combined_adata


def cluster_combined_adata(
    combined_adata: anndata.AnnData,
    clustering_method: str = "leiden",
    resolution: float = 1.0,
    n_top_genes: int | None = None,
) -> anndata.AnnData:
    """Cluster anddata

    Take a combined AnnData (e.g., from ``combine_original_and_reconstructions``)
    and perform clustering.

    Parameters
    ----------
    combined_adata : anndata.AnnData
        AnnData object containing both original data (in some rows) and
        reconstructed data (in other rows), typically indicated by
        ``.obs["origin"]``.
    clustering_method : str, optional
        Clustering method to use. Supported: "leiden" or "louvain". Defaults to "leiden".
    resolution : float, optional
        Resolution parameter for clustering. Defaults to 1.0.

    Returns
    -------
    combined_adata : anndata.AnnData
        The same object with updated clustering labels in ``.obs[clustering_method]``.

    Notes
    -----
    Data reference: Single-cell RNA-seq data or reconstructed data from your pipeline.

    Examples
    --------
    >>> combined = combine_original_and_reconstructions(adata)
    >>> combined = cluster_combined_adata(combined, clustering_method="leiden", resolution=0.5)
    >>> combined.obs["leiden"].value_counts()
    """
    logger = logging.getLogger(__name__)
    logger.info("Clustering combined AnnData with %d cells.", combined_adata.n_obs)

    # Example preprocessing pipeline (adapt to your needs):
    if n_top_genes is not None:
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat_v3", n_top_genes=n_top_genes)
        combined_adata = combined_adata[:, combined_adata.var["highly_variable"]]

    sc.pp.normalize_total(combined_adata, target_sum=1e4)
    sc.pp.log1p(combined_adata)
    sc.pp.scale(combined_adata, max_value=10)
    sc.tl.pca(combined_adata, svd_solver="arpack")

    sc.pp.neighbors(combined_adata, n_neighbors=10, n_pcs=50)

    if clustering_method.lower() == "leiden":
        sc.tl.leiden(combined_adata, resolution=resolution, key_added="leiden")
        combined_adata.obs["cluster"] = combined_adata.obs["leiden"]
    elif clustering_method.lower() == "louvain":
        sc.tl.louvain(combined_adata, resolution=resolution, key_added="louvain")
        combined_adata.obs["cluster"] = combined_adata.obs["louvain"]
    else:
        msg = f"Unsupported clustering method: {clustering_method}"
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Clustering complete. Found %d clusters.", combined_adata.obs["cluster"].nunique())
    return combined_adata


def subsample_adata(adata: anndata.AnnData, subsample_size: int = 1000, random_state: int = 0) -> anndata.AnnData:
    """
    Subsample cells from an AnnData object without altering gene dimensions.

    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData object.
    subsample_size : int, optional
        Number of cells to retain in the subsampled dataset. Defaults to 1000.
    random_state : int, optional
        Seed for random subsampling. Defaults to 0.

    Returns
    -------
    subsampled_adata : anndata.AnnData
        AnnData object with a random subsample of cells from the original data.

    Notes
    -----
    Data reference: Single-cell RNA-seq data, originally from your pipeline or reconstructions.

    Examples
    --------
    >>> subsampled = subsample_adata(combined_adata, subsample_size=500)
    >>> subsampled
    """
    logger = logging.getLogger(__name__)
    logger.info("Subsampling AnnData to %d cells.", subsample_size)
    if subsample_size >= adata.n_obs:
        logger.warning(
            "Requested subsample_size (%d) is >= number of cells (%d). Returning original data.",
            subsample_size,
            adata.n_obs,
        )
        return adata

    np.random.seed(random_state)
    keep_indices = np.random.choice(adata.n_obs, size=subsample_size, replace=False)
    subsampled_adata = adata[keep_indices].copy()
    logger.info("Subsampling complete. Returning new AnnData object with shape %s.", subsampled_adata.shape)
    return subsampled_adata
