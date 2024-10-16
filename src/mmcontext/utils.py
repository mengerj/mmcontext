# tests/utils.py

import anndata
import numpy as np
import pandas as pd


def create_test_anndata(n_cells=20, n_genes=100, cell_types=None, tissues=None):
    """
    Create a test AnnData object with synthetic data.

    Parameters
    ----------
    n_cells (int): Number of cells (observations). Default is 20.
    n_genes (int): Number of genes (variables). Default is 100.
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
    X = np.random.rand(n_cells, n_genes)

    # Create observation (cell) metadata
    obs = pd.DataFrame(
        {
            "cell_type": np.random.choice(cell_types, n_cells),
            "tissue": np.random.choice(tissues, n_cells),
            "sample_id": [f"Sample_{i}" for i in range(n_cells)],
        }
    )

    # Create variable (gene) metadata
    var = pd.DataFrame({"gene_symbols": [f"Gene_{i}" for i in range(n_genes)]})

    # Create the AnnData object
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    return adata
