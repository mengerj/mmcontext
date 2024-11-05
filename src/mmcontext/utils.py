# tests/utils.py
import anndata
import numpy as np
import pandas as pd


def create_test_anndata(n_samples=20, n_features=100, cell_types=None, tissues=None, batch_categories=None):
    """
    Create a test AnnData object with synthetic data, including batch information.

    Parameters
    ----------
    n_samples : int
        Number of cells (observations). Default is 20.
    n_features : int
        Number of genes (variables). Default is 100.
    cell_types : list, optional
        List of cell types. Defaults to ["B cell", "T cell", "NK cell"].
    tissues : list, optional
        List of tissues. Defaults to ["blood", "lymph"].
    batch_categories : list, optional
        List of batch categories. Defaults to ["Batch1", "Batch2"].

    Returns
    -------
    anndata.AnnData
        Generated AnnData object.
    """
    import numpy as np

    # Set default values for mutable arguments if they are None
    if cell_types is None:
        cell_types = ["B cell", "T cell", "NK cell"]
    if tissues is None:
        tissues = ["blood", "lymph"]
    if batch_categories is None:
        batch_categories = ["Batch1", "Batch2"]

    # Set a random seed for reproducibility
    np.random.seed(42)

    # Determine the number of batches and allocate samples to batches
    n_batches = len(batch_categories)
    samples_per_batch = n_samples // n_batches
    remainder = n_samples % n_batches

    batch_labels = []
    for i, batch in enumerate(batch_categories):
        n = samples_per_batch + (1 if i < remainder else 0)
        batch_labels.extend([batch] * n)

    # Shuffle batch labels
    np.random.shuffle(batch_labels)

    # Generate observation (cell) metadata
    obs = pd.DataFrame(
        {
            "cell_type": np.random.choice(cell_types, n_samples),
            "tissue": np.random.choice(tissues, n_samples),
            "batch": batch_labels,
            "sample_id": np.arange(n_samples),
        }
    )
    obs.index = [f"Cell_{i}" for i in range(n_samples)]

    # Generate a random data matrix (e.g., gene expression values)
    X = np.zeros((n_samples, n_features))
    for i, batch in enumerate(batch_categories):
        # Get indices of cells in this batch
        idx = obs[obs["batch"] == batch].index
        idx = [obs.index.get_loc(i) for i in idx]

        # Generate data for this batch
        # For simplicity, let's make a mean shift between batches
        mean = np.random.rand(n_features) * (i + 1)  # Different mean for each batch
        X[idx, :] = np.random.normal(loc=mean, scale=1.0, size=(len(idx), n_features))

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
