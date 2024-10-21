# tests/utils.py
import anndata
import numpy as np
import pandas as pd


def create_test_anndata(n_samples=20, n_features=100, cell_types=None, tissues=None):
    """
    Create a test AnnData object with synthetic data.

    Parameters
    ----------
    n_samples (int): Number of cells (observations). Default is 20.
    n_features (int): Number of genes (variables). Default is 100.
    cell_types (list, optional): List of cell types. Defaults to ["B cell", "T cell", "NK cell"].
    tissues (list, optional): List of tissues. Defaults to ["blood", "lymph"].

    Returns
    -------
    anndata.AnnData: Generated AnnData object.
    """
    # Set default values for mutable arguments if they are None
    if cell_types is None:
        cell_types = ["B cell", "T cell", "NK cell"]
    if tissues is None:
        tissues = ["blood", "lymph"]

    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate a random data matrix (e.g., gene expression values)
    X = np.random.rand(n_samples, n_features)

    # Create observation (cell) metadata
    obs = pd.DataFrame(
        {
            "cell_type": np.random.choice(cell_types, n_samples),
            "tissue": np.random.choice(tissues, n_samples),
            "sample_id": np.arange(n_samples),
        }
    )
    obs.index = [f"Cell_{i}" for i in range(n_samples)]

    # Create variable (gene) metadata
    var = pd.DataFrame({"gene_symbols": [f"Gene_{i}" for i in range(n_features)]})
    var.index = [f"Gene_{i}" for i in range(n_features)]

    # Create the AnnData object
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    return adata


def create_test_emb_anndata(n_samples, emb_dim, data_key="d_emb_aligned", context_key="c_emb_aligned", sample_ids=None):
    """
    Helper function to create a test AnnData object with specified embeddings and sample IDs.

    Args:
        n_samples (int): Number of samples (cells).
        emb_dim (int): Embedding dimension.
        data_key (str): Key for data embeddings in adata.obsm.
        context_key (str): Key for context embeddings in adata.obsm.
        sample_ids (list): List of sample IDs. If None, default IDs are assigned.

    Returns
    -------
        AnnData: The constructed AnnData object.
    """
    adata = create_test_anndata(n_samples=n_samples)
    adata.obsm[data_key] = np.random.rand(n_samples, emb_dim)
    adata.obsm[context_key] = np.random.rand(n_samples, emb_dim)
    if sample_ids is not None:
        adata.obs["sample_id"] = sample_ids
    return adata
