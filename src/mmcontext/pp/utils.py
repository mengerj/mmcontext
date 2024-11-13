import logging

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp


def split_anndata(adata: anndata.AnnData, train_size: float = 0.8):
    """
    Splits an AnnData object into training and validation sets.

    Parameters
    ----------
    adata
        The complete AnnData object to be split.
    train_size
        The proportion of the dataset to include in the train split. Should be between 0 and 1.

    Returns
    -------
    anndata.AnnData: The training AnnData object.
    anndata.AnnData: The validation AnnData object.
    """
    # Ensure train_size is a valid proportion
    if not 0 < train_size < 1:
        raise ValueError("train_size must be a float between 0 and 1.")

    # Generate random indices
    indices = np.arange(adata.n_obs)
    np.random.shuffle(indices)

    # Calculate the number of observations for the train set
    train_indices_count = int(train_size * adata.n_obs)

    # Split indices for train and validation sets
    train_indices = indices[:train_indices_count]
    val_indices = indices[train_indices_count:]

    # Subset the AnnData object
    train_adata = adata[train_indices]
    val_adata = adata[val_indices]

    return train_adata, val_adata


def consolidate_low_frequency_categories(adata: anndata.AnnData, columns: list, threshold: int, remove=False):
    """Consolidates low frequency categories in specified columns of an AnnData object.

    Modifies the AnnData object's .obs by setting entries in specified columns
    to 'remaining {column_name}' or removing them if their frequency is below a specified threshold.
    Converts columns to non-categorical if necessary to adjust the categories dynamically.

    Parameters
    ----------
    adata
        The AnnData object to be processed.
    columns
        List of column names in adata.obs to check for low frequency.
    threshold
        Frequency threshold below which categories are considered low.
    remove
        If True, categories below the threshold are removed entirely.

    Returns
    -------
    anndata.AnnData: The modified AnnData object.
    """
    adata = adata.copy()

    for col in columns:
        if col in adata.obs.columns:
            # Convert column to string if it's categorical
            if isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
                adata.obs[col] = adata.obs[col].astype(str)

            # Calculate the frequency of each category
            freq = adata.obs[col].value_counts()

            # Identify low frequency categories
            low_freq_categories = freq[freq < threshold].index

            if remove:
                # Remove entries with low frequency categories entirely
                adata = adata[~adata.obs[col].isin(low_freq_categories)].copy()
                # Convert column back to categorical with new categories
                adata.obs[col] = pd.Categorical(adata.obs[col])
            else:
                # Update entries with low frequency categories to 'remaining {col}'
                adata.obs.loc[adata.obs[col].isin(low_freq_categories), col] = f"remaining {col}"

                # Convert column back to categorical with new categories
                adata.obs[col] = pd.Categorical(adata.obs[col])

        else:
            print(f"Column {col} not found in adata.obs")

    return adata


def remove_zero_rows_and_columns(adata: anndata.AnnData, inplace: bool = True):
    """
    Removes rows (cells) and columns (genes) from adata.X that are all zeros, or that have zero variance.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to filter.
    inplace : bool, optional (default: False)
        If True, modifies the adata object in place.
        If False, returns a new filtered AnnData object.

    Returns
    -------
    AnnData or None
        If inplace is False, returns the filtered AnnData object.
        If inplace is True, modifies adata in place and returns None.
    """
    logger = logging.getLogger(__name__)
    # Check if adata.X is sparse or dense
    if sp.issparse(adata.X):
        # Sparse matrix
        non_zero_row_indices = adata.X.getnnz(axis=1) > 0
        non_zero_col_indices = adata.X.getnnz(axis=0) > 0
    else:
        # Dense matrix
        non_zero_row_indices = np.any(adata.X != 0, axis=1)
        non_zero_col_indices = np.any(adata.X != 0, axis=0)
    logger.info(f"Number of zero rows: {adata.X.shape[0] - np.sum(non_zero_row_indices)}")
    logger.info(f"Number of zero columns: {adata.X.shape[1] - np.sum(non_zero_col_indices)}")
    logger.info("Removing zero rows and columns...")
    if inplace:
        # Filter the AnnData object in place
        adata._inplace_subset_obs(non_zero_row_indices)
        adata._inplace_subset_var(non_zero_col_indices)
    else:
        # Return a filtered copy
        adata = adata[:, non_zero_col_indices]  # Subset variables (genes)
        adata = adata[non_zero_row_indices, :]  # Subset observations (cells)

        return adata.copy()


def remove_zero_variance_genes(adata):
    """Remove genes with zero variance from an AnnData object."""
    logger = logging.getLogger(__name__)
    if sp.issparse(adata.X):
        # For sparse matrices
        gene_variances = np.array(adata.X.power(2).mean(axis=0) - np.square(adata.X.mean(axis=0))).flatten()
    else:
        # For dense matrices
        gene_variances = np.var(adata.X, axis=0)
    zero_variance_genes = gene_variances == 0
    num_zero_variance_genes = np.sum(zero_variance_genes)

    if np.any(zero_variance_genes):
        adata = adata[:, ~zero_variance_genes]
        logger.info(f"Removed {num_zero_variance_genes} genes with zero variance.")
    else:
        logger.info("No genes with zero variance found.")


def check_zero_variance_cells(adata):
    """Check for cells with zero variance in an AnnData object."""
    logger = logging.getLogger(__name__)
    if sp.issparse(adata.X):
        cell_variances = np.array(adata.X.power(2).mean(axis=1) - np.square(adata.X.mean(axis=1))).flatten()
    else:
        cell_variances = np.var(adata.X, axis=1)
    zero_variance_cells = cell_variances == 0
    num_zero_variance_cells = np.sum(zero_variance_cells)
    if np.any(zero_variance_cells):
        adata = adata[~zero_variance_cells, :]
        logger.info(f"Removed {num_zero_variance_cells} cells with zero variance.")
    else:
        logger.info("No cells with zero variance found.")


def remove_duplicate_cells(adata: anndata.AnnData, inplace: bool = True):
    """
    Removes duplicate cells (rows) from adata.X.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to filter.
    inplace : bool, optional (default: False)
        If True, modifies the adata object in place.
        If False, returns a new filtered AnnData object.

    Returns
    -------
    AnnData or None
        If inplace is False, returns the filtered AnnData object.
        If inplace is True, modifies adata in place and returns None.
    """
    logger = logging.getLogger(__name__)
    # Convert adata.X to dense array if it's sparse
    if sp.issparse(adata.X):
        X_dense = adata.X.toarray()
    else:
        X_dense = adata.X

    # Convert to DataFrame for easier comparison
    df = pd.DataFrame(X_dense)

    # Find duplicate rows
    duplicate_cell_mask = df.duplicated(keep="first")
    num_duplicates = duplicate_cell_mask.sum()
    logger.info(f"Number of duplicate cells: {num_duplicates}")

    if num_duplicates == 0:
        logger.info("No duplicate cells to remove.")
        if inplace:
            return None
        else:
            return adata.copy()

    # Indices of unique cells
    unique_cell_mask = ~duplicate_cell_mask

    if inplace:
        adata._inplace_subset_obs(unique_cell_mask)
        logger.info(f"Removed {num_duplicates} duplicate cells.")
        return None
    else:
        filtered_adata = adata[unique_cell_mask].copy()
        logger.info(f"Returning new AnnData object with {filtered_adata.n_obs} cells.")
        return filtered_adata


def remove_entries(adata):
    """Remove zero variance genes and cells, and zero rows and columns from an AnnData object."""
    # Remove zero variance genes
    remove_zero_variance_genes(adata)
    # Remove zero variance cells
    check_zero_variance_cells(adata)
    # Remove zero rows and columns
    remove_zero_rows_and_columns(adata)
    # remove duplicate cells
    remove_duplicate_cells(adata)
