import logging
import os
from datetime import datetime

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from omegaconf import DictConfig
from sentence_transformers import evaluation, losses
from tqdm import tqdm

logger = logging.getLogger(__name__)


def consolidate_low_frequency_categories(
    adata: anndata.AnnData, columns: str | list[str], threshold: int, remove=False
):
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
    # Ensure the object is loaded into memory if it's in backed mode
    if adata.isbacked:
        adata = adata.to_memory()

    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        if col in adata.obs.columns:
            # Convert column to string if it's categorical
            if isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
                as_string = adata.obs[col].astype(str)
                adata.obs[col] = as_string

            # Calculate the frequency of each category
            freq = adata.obs[col].value_counts()

            # Identify low frequency categories
            low_freq_categories = freq[freq < threshold].index

            if remove:
                # Remove entries with low frequency categories entirely
                mask = ~adata.obs[col].isin(low_freq_categories)
                adata._inplace_subset_obs(mask)
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
        return adata
    else:
        logger.info("No genes with zero variance found.")
        return adata


def remove_zero_variance_cells(adata):
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
        return adata
    else:
        logger.info("No cells with zero variance found.")
        return adata


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


def setup_logging(logging_dir="logs"):
    """Set up logging configuration for the module.

    This function configures the root logger to display messages in the console and to write them to a file
    named by the day. The log level is set to INFO.
    """
    # Create the logs directory
    os.makedirs(logging_dir, exist_ok=True)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers (important if function is called multiple times)
    if not logger.hasHandlers():
        # Create a file handler
        log_file = logging.FileHandler(f"logs/{datetime.now().strftime('%Y-%m-%d')}.log")
        log_file.setLevel(logging.INFO)

        # Create a console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # Create a formatter and set it for the handlers
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        log_file.setFormatter(formatter)
        console.setFormatter(formatter)

        # Add the handlers to the root logger
        logger.addHandler(log_file)
        logger.addHandler(console)

    return logger


def compute_cosine_similarity(sample_embeddings, query_embeddings, device="cpu"):
    """
    Compute pairwise cosine similarity between samples and queries using PyTorch.

    Parameters
    ----------
    sample_embeddings : np.ndarray
        2D array of shape (num_samples, embedding_dim) containing
        the sample (omics) embeddings. Source: adata.obsm["omics_emb"].
    query_embeddings : np.ndarray
        2D array of shape (num_queries, embedding_dim) containing
        the query embeddings. Typically from model.encode(...).
    device : str, optional
        The device on which to run the computation. One of ["cpu", "cuda","mps"].

    Returns
    -------
    np.ndarray
        A matrix of shape (num_queries, num_samples), containing
        the cosine similarity scores for each query against each sample.
    """
    logger.info("Converting numpy arrays to torch Tensors.")
    # Convert to torch Tensors
    sample_t = torch.from_numpy(sample_embeddings).float().to(device)
    query_t = torch.from_numpy(query_embeddings).float().to(device)

    logger.info("L2-normalizing embeddings for cosine similarity.")
    # L2 normalize if we want to treat dot product as cosine
    sample_t = sample_t / (sample_t.norm(dim=1, keepdim=True) + 1e-9)
    query_t = query_t / (query_t.norm(dim=1, keepdim=True) + 1e-9)

    logger.info("Performing matrix multiplication on device=%s", device)
    # matrix shape: (num_queries, embedding_dim) x (embedding_dim, num_samples)
    # result -> (num_queries, num_samples)
    sim_t = query_t.mm(sample_t.transpose(0, 1))

    # Move back to CPU, convert to numpy
    sim = sim_t.cpu().numpy()
    return sim


def get_loss(dataset_type: str, loss_name: str = None):
    """
    Return a suitable loss object based on the provided loss name and dataset type.

    Parameters
    ----------
    loss_name : str | None
        A string identifying the loss function to be used.
    model : MMContextEncoder
        The multimodal context encoder model.
    dataset_type : str
        The type of the dataset (e.g., 'pairs', 'triplets').

    Returns
    -------
    losses.Loss
        An instance of a SentenceTransformer loss object.

    Notes
    -----
    This is a stub. You will add compatibility checks later (e.g., if dataset_type
    is 'pairs', maybe only certain losses are supported).
    """
    pairs_losses = ["ContrastiveLoss", "OnlineContrastiveLoss"]
    pairs_losses_default = "ContrastiveLoss"
    multiplets_losses = ["MultipleNegativesRankingLoss", "CachedMultipleNegativesRankingLoss", "CachedGISTEmbedLoss"]
    multiplets_losses_default = "MultipleNegativesRankingLoss"
    if dataset_type == "pairs" and loss_name is None:
        loss_name = pairs_losses_default
    elif dataset_type == "pairs" and loss_name not in pairs_losses:
        raise ValueError(f"Loss '{loss_name}' is not supported for pairs dataset. Choose from {pairs_losses}")
    if dataset_type == "multiplets":
        if loss_name is None:
            loss_name = multiplets_losses_default
        elif loss_name not in multiplets_losses:
            raise ValueError(
                f"Loss '{loss_name}' is not supported for multiplets dataset. Choose from {multiplets_losses}"
            )

    # Dynamically fetch the loss class from the sentence_transformers.losses module
    try:
        LossClass = getattr(losses, loss_name)
    except AttributeError as e:
        raise f"Loss class '{loss_name}' not found in sentence_transformers.losses" from e
    # Instantiate the loss class with the given model
    return LossClass


def get_evaluator(
    dataset_type: str, dataset, evaluator_name: str | None = None, batch_size: int = 32, current_eval_name: str = None
):
    """
    Return a suitable evaluator object

    It is obtained based on the provided evaluator name, dataset type, and dataset. Has to be one of the evalutors supported
    from the sentence_transformers.evaluation module. Also has to be explicitly defined for the dataset type in this function.

    Parameters
    ----------
    dataset_type : str
        The type of the dataset (e.g., 'pairs', 'multiplets').
    dataset: dict
        The dataset dictionary containing the necessary data for the evaluator.
    evaluator_name : str
        A string identifying the evaluator to be used. If none is given a default will be used depending on the dataset type.

    Returns
    -------
    Evaluator
        An instance of a SentenceTransformer evaluator object.

    Raises
    ------
    ValueError
        If the provided evaluator name is not supported for the given dataset type
        or if the required data keys are missing in the dataset.
    AttributeError
        If the evaluator class is not found in the sentence_transformers.evaluation module.
    """
    pairs_evaluators = ["BinaryClassificationEvaluator"]
    pairs_evaluator_default = "BinaryClassificationEvaluator"
    multiplets_evaluators = ["TripletEvaluator"]
    multiplets_evaluator_default = "TripletEvaluator"

    if dataset_type == "pairs":
        if evaluator_name is None:
            evaluator_name = pairs_evaluator_default
        if evaluator_name not in pairs_evaluators:
            raise ValueError(
                f"Evaluator '{evaluator_name}' is not supported for pairs dataset. Choose from {pairs_evaluators}"
            )

        required_keys = {"sentence_1", "sentence_2", "label"}
        if not required_keys.issubset(dataset.column_names):
            raise ValueError(f"Dataset for 'pairs' evaluator must contain keys: {required_keys}")
        try:
            EvaluatorClass = getattr(evaluation, evaluator_name)
            evaluator_obj = EvaluatorClass(
                sentences1=dataset["sentence_1"],
                sentences2=dataset["sentence_2"],
                labels=dataset["label"],
                batch_size=batch_size,
                name=current_eval_name,
            )
        except AttributeError as e:
            raise f"Evaluator class '{evaluator_name}' not found in sentence_transformers.evaluation" from e
        except TypeError as e:  # Catch potential errors due to missing or incorrect arguments
            raise ValueError(f"Error instantiating {evaluator_name}: {e}") from e

    elif dataset_type == "multiplets":
        if evaluator_name is None:
            evaluator_name = multiplets_evaluator_default
        if evaluator_name not in multiplets_evaluators:
            raise ValueError(
                f"Evaluator '{evaluator_name}' is not supported for multiplets dataset. "
                f"Choose from {multiplets_evaluators}"
            )

        required_keys = {"anchor", "positive", "negative_1"}  # Updated to match TripletEvaluator
        if not required_keys.issubset(dataset.column_names):
            raise ValueError(f"Dataset for 'multiplets' evaluator must contain keys: {required_keys}")
        try:
            EvaluatorClass = getattr(evaluation, evaluator_name)

            evaluator_obj = EvaluatorClass(
                anchors=dataset["anchor"],
                positives=dataset["positive"],
                negatives=dataset["negative_1"],
                name=current_eval_name,
            )
        except AttributeError as e:
            raise f"Evaluator class '{evaluator_name}' not found in sentence_transformers.evaluation" from e
        except TypeError as e:  # Catch potential errors due to missing or incorrect arguments
            raise ValueError(f"Error instantiating {evaluator_name}: {e}") from e

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return evaluator_obj


def get_device():
    """Helper function to get the available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def resolve_negative_indices_and_rename(
    dataset,
    primary_cell_sentence_col: str,
    positive_col: str = "positive",
    negative_prefix: str = "negative",
    index_col: str = "sample_idx",
    remove_index_col: bool = False,
):
    """
    Resolve negative indices in multiplet datasets and rename columns appropriately.

    This function processes multiplet datasets by:
    1. Resolving sample indices in negative columns to actual values
    2. Renaming columns by removing "_idx" suffix
    3. Renaming primary column to "anchor"
    4. Optionally removing the index column

    Parameters
    ----------
    dataset : Dataset or DatasetDict
        The dataset containing negative indices to resolve
    primary_cell_sentence_col : str
        Name of the primary cell sentence column (will be renamed to "anchor")
    positive_col : str, optional
        Name of the positive column (default: "positive")
    negative_prefix : str, optional
        Prefix for negative columns (default: "negative")
    index_col : str, optional
        Name of the index column (default: "sample_idx")
    remove_index_col : bool, optional
        Whether to remove the index column after processing (default: True)

    Returns
    -------
    Dataset or DatasetDict
        Processed dataset with resolved indices and renamed columns
    """
    import re

    from datasets import Dataset, DatasetDict

    def _resolve_negative_indices_split(split: Dataset) -> Dataset:
        """Resolve sample indices in negative columns to actual values for a single split."""
        negative_cols = [c for c in split.column_names if c.startswith(negative_prefix)]

        if not index_col or not negative_cols:
            return split

        # Create mappings from index to both positive and anchor values
        index_to_positive = {}
        index_to_anchor = {}

        for _, row in enumerate(split):
            if index_col in row:
                idx_val = str(row[index_col])
                if positive_col in row:
                    index_to_positive[idx_val] = row[positive_col]
                if primary_cell_sentence_col in row:
                    index_to_anchor[idx_val] = row[primary_cell_sentence_col]

        def _resolve_negatives(batch):
            for neg_col in negative_cols:
                if neg_col in batch:
                    # Extract the number from column name (e.g., "negative_1_idx" -> 1)
                    match = re.search(r"negative_(\d+)_idx", neg_col)
                    if match:
                        neg_num = int(match.group(1))
                        is_odd = neg_num % 2 == 1

                        resolved_values = []
                        for idx_value in batch[neg_col]:
                            if is_odd and idx_value in index_to_positive:
                                resolved_values.append(index_to_positive[idx_value])
                            elif not is_odd and idx_value in index_to_anchor:
                                resolved_values.append(index_to_anchor[idx_value])
                            else:
                                # Fallback: keep original value if mapping not found
                                resolved_values.append(idx_value)

                        batch[neg_col] = resolved_values
            return batch

        # Apply the resolution
        proc = split.map(_resolve_negatives, batched=True, desc="Resolving negative indices")

        # Rename columns to remove "_idx" suffix
        for neg_col in negative_cols:
            if neg_col in proc.column_names and neg_col.endswith("_idx"):
                new_col_name = neg_col.replace("_idx", "")
                proc = proc.rename_column(neg_col, new_col_name)

        # Rename primary column to "anchor"
        if primary_cell_sentence_col in proc.column_names and primary_cell_sentence_col != "anchor":
            proc = proc.rename_column(primary_cell_sentence_col, "anchor")

        # Remove index column if requested
        if remove_index_col and index_col in proc.column_names:
            proc = proc.remove_columns([index_col])

        return proc

    # Process the dataset
    if isinstance(dataset, Dataset):
        processed_dataset = _resolve_negative_indices_split(dataset)
    elif isinstance(dataset, DatasetDict):
        processed_dataset = DatasetDict(
            {name: _resolve_negative_indices_split(split) for name, split in dataset.items()}
        )
    else:
        raise TypeError("resolve_negative_indices expects a Dataset or DatasetDict")

    return processed_dataset


'''
def prepare_omics_resources(hf_ds, *, prefix: str = "sample_idx:"):
    """
    Parameters
    ----------
    hf_ds : datasets.Dataset
        Must expose ``sample_idx`` and ``data_representation``.
    prefix : str, optional
        Tag added in front of each key in the lookup (default ``"sample_idx:"``).

    Returns
    -------
    embedding_matrix : np.ndarray  (num_ids, dim)
    lookup           : dict[str, int]
    """
    lookup = {}
    vectors = []
    # loop over row with progress bar
    for row in tqdm(hf_ds, desc="Preparing omics resources", unit="row"):
        sid = row["sample_idx"]
        vec = row["data_representation"]
        row_id = len(lookup)  # 0‥N–1 in encounter order
        lookup[f"{prefix}{sid}"] = row_id
        vectors.append(vec)

    embedding_matrix = np.stack(vectors).astype("float32")
    return embedding_matrix, lookup
'''
