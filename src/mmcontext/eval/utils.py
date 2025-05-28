import logging
import os
import random
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from anndata import AnnData

logger = logging.getLogger(__name__)


class LabelKind(str, Enum):
    """Name for the kind of label."""

    BIO = "bio"
    BATCH = "batch"


@dataclass
class LabelSpec:
    """Specification for a label column.

    To be used with the BaseEvaluator derived class.
    """

    name: str  # column in adata.obs
    kind: LabelKind  # 'bio' or 'batch'


def load_evaluation_results(results_path: str) -> pd.DataFrame:
    """
    Load evaluation results from CSV file.

    Parameters
    ----------
    results_path : str
        Path to the evaluation results CSV file

    Returns
    -------
    pd.DataFrame
        Loaded evaluation results
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    results_df = pd.read_csv(results_path)
    logger.info(f"Loaded {len(results_df)} evaluation results from {results_path}")
    return results_df


def subset_adata_by_query_score(adata: AnnData, query_key: str, percentile: float) -> tuple[AnnData, AnnData]:
    """
    Create two subsets of the given AnnData by filtering on a particular query score.

    Parameters
    ----------
    adata : AnnData
        The AnnData containing a column adata.obs["query_scores"].
        Each row of this column should be a dictionary with a float score
        for the given `query_key`.
    query_key : str
        The key within each dictionary in adata.obs["query_scores"]
        whose float value we want to filter on (e.g. "leukemia").
    percentile : float
        The percentile (from 0 to 100) used to create the top and bottom filters.
        For example, 10.0 means top 10% vs bottom 10%.

    Returns
    -------
    top_adata : AnnData
        Subset of `adata` where the row's score is in the top `percentile`.
    bottom_adata : AnnData
        Subset of `adata` where the row's score is in the bottom `percentile`.

    References
    ----------
    This function uses the `adata.obs["query_scores"]` column (dict-like per row).
    Make sure your data is loaded and structured such that each
    entry in `adata.obs["query_scores"]` is a dictionary containing `query_key`.
    """
    # Extract the numeric values for the requested query_key
    if "query_scores" not in adata.obs:
        raise KeyError("adata.obs does not contain 'query_scores'.")

    # Convert each dict in adata.obs["query_scores"] to the float value for query_key
    try:
        scores = adata.obs["query_scores"].apply(lambda d: d[query_key])
    except Exception as e:
        raise KeyError(f"Could not extract '{query_key}' from adata.obs['query_scores'].") from e

    # Compute percentile thresholds
    lower_thresh = np.percentile(scores, percentile)  # e.g., 10th percentile
    upper_thresh = np.percentile(scores, 100.0 - percentile)  # e.g., 90th percentile

    # Create boolean masks
    bottom_mask = scores <= lower_thresh
    top_mask = scores >= upper_thresh

    # Log how many observations are in each subset
    logger.info(
        f"Splitting data by '{query_key}' scores at the {percentile}th/({100 - percentile}th) percentile. "
        f"Found {bottom_mask.sum()} cells in bottom subset, {top_mask.sum()} cells in top subset."
    )

    # Subset the AnnData
    bottom_adata = adata[bottom_mask].copy()
    top_adata = adata[top_mask].copy()

    return top_adata, bottom_adata


def create_emb_pair_dataframe(
    adata,
    embedding_dict: dict[str, str],
    label_keys: str | list = None,
    subset_size=100,
    seed=42,
    obs_filter_key=None,
    obs_filter_value=None,
):
    """
    Create a DataFrame for multi-modal embeddings from an anndata object.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing embeddings in .obsm
    embedding_dict : dict[str, str]
        Dictionary mapping modality names to their corresponding obsm keys
        e.g., {"omics": "X_pca", "text": "text_emb", "protein": "protein_emb"}
    label_keys : str or list, optional
        Column name(s) in adata.obs to add as labels
    subset_size : int, optional
        Number of samples to select. If the dataset has fewer than `subset_size` samples,
        all available samples are used. Default is 100.
    seed : int, optional
        Random seed for reproducibility, default 42
    obs_filter_key : str, optional
        Column name in `adata.obs` to filter samples.
    obs_filter_value : Any, optional
        Value that `obs_filter_key` should match to include samples.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:
        - 'sample_id': The index of the sample in `adata.obs`
        - 'embedding': The embedding vector as a list of floats
        - 'modality': The type of embedding (either `modality_1` or `modality_2`)
        - Additional columns for each label specified in `label_keys`

    Notes
    -----
    1. If `obs_filter_key` and `obs_filter_value` are provided, only samples where
       `adata.obs[obs_filter_key] == obs_filter_value` are included.
    2. This function assumes that each sample has one paired embedding in `.obsm[emb1_key]`
       and `.obsm[emb2_key]`.
    3. The function randomly selects `subset_size` samples from `adata.obs` while maintaining
       the pairing between `emb1_key` and `emb2_key`.
    4. The returned DataFrame contains two rows per sample (one for each embedding type).
    """
    # Apply filtering if specified
    if obs_filter_key is not None and obs_filter_value is not None:
        logger.info("Filtering dataset for %s == %s.", obs_filter_key, obs_filter_value)
        adata = adata[adata.obs[obs_filter_key] == obs_filter_value]

    # Ensure subset_size is not larger than available samples
    available_samples = adata.n_obs
    subset_size = min(subset_size, available_samples)

    # Sample a subset of indices
    random.seed(seed)
    sampled_indices = random.sample(range(available_samples), k=subset_size)

    # Extract selected sample IDs
    sample_ids = adata.obs.index[sampled_indices]

    data_rows = []

    # Convert label_keys to a list if it's a string
    if isinstance(label_keys, str):
        label_keys = [label_keys]

    # Validate label keys
    if label_keys and bool(set(label_keys) & {"sample_index", "embedding", "embedding_type"}):
        raise ValueError(
            "label_keys cannot contain reserved column names: 'sample_index', 'embedding', 'embedding_type'."
        )

    data_rows = []

    # Create rows for each modality
    for modality, obsm_key in embedding_dict.items():
        if obsm_key not in adata.obsm:
            raise KeyError(f"Embedding key '{obsm_key}' not found in adata.obsm")

        embeddings = adata.obsm[obsm_key][sampled_indices]

        for idx, sample_id in enumerate(sample_ids):
            row = {"sample_index": sample_id}

            # Add labels
            if label_keys:
                for label_key in label_keys:
                    row[label_key] = adata.obs[label_key][sampled_indices[idx]]

            # Add embedding and type
            row.update({"embedding": embeddings[idx].tolist(), "embedding_type": modality})
            data_rows.append(row)

    df = pd.DataFrame(data_rows)
    logger.info(
        "Created DataFrame with %d rows (%d samples × %d modalities)", len(df), subset_size, len(embedding_dict)
    )
    return df
