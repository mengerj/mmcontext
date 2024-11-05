import anndata
import numpy as np
import pandas as pd


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
            else:
                # Update entries with low frequency categories to 'remaining {col}'
                adata.obs.loc[adata.obs[col].isin(low_freq_categories), col] = f"remaining {col}"

                # Convert column back to categorical with new categories
                adata.obs[col] = pd.Categorical(adata.obs[col])

        else:
            print(f"Column {col} not found in adata.obs")

    return adata
