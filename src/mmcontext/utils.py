from __future__ import annotations

import inspect
import json
import logging
import os
import re
import shutil
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from omegaconf import DictConfig
from sentence_transformers import evaluation, losses
from torch import nn
from tqdm import tqdm

try:
    import wandb  # optional; wrapper no-ops if not available or not initialized
except Exception:  # pragma: no cover
    wandb = None

try:
    from torch.utils.tensorboard import SummaryWriter  # optional
except Exception:  # pragma: no cover
    SummaryWriter = None

logger = logging.getLogger(__name__)


def truncate_cell_sentences(
    dataset,
    column_name: str,
    max_length: int,
    num_proc: int = None,
    filter_strings: list[str] = None,
    gene_special_token: str = None,
    delimiter: str = " ",
):
    """
    Truncate cell sentences to the first max_length tokens efficiently, with optional gene filtering.

    Parameters
    ----------
    dataset : Dataset
        HuggingFace dataset containing cell sentences
    column_name : str
        Name of the column containing cell sentences to truncate
    max_length : int
        Maximum number of tokens/words to keep (first n elements)
    num_proc : int, optional
        Number of processes to use. If None, uses single process for small datasets
        and multiple processes for large datasets (>50k samples).
    filter_strings : list[str], optional
        List of strings to filter out from genes. Any gene containing any of these
        strings will be removed before truncation. Useful for filtering ribosomal
        genes (e.g., ['RPS', 'RPL']) or other unwanted gene sets.
    gene_special_token : str, optional
        Special token to add in front of each gene name (e.g., '[GENE]'). If None,
        no special token is added.
    delimiter : str, optional
        Delimiter to use between tokens (default: " "). Can be customized for
        different training experiments.

    Returns
    -------
    Dataset
        Dataset with filtered and truncated cell sentences
    """
    import re

    # Pre-compile regex pattern for better performance
    # This pattern matches sequences of non-whitespace characters (tokens)
    token_pattern = re.compile(r"\S+")

    def _truncate_batch(batch):
        truncated = []
        for sentence in batch[column_name]:
            # Find all tokens using regex (faster than split for large texts)
            tokens = token_pattern.findall(sentence)

            # Filter out genes containing any of the filter strings
            if filter_strings:
                original_count = len(tokens)
                filtered_tokens = []
                removed_counts = {filter_str: 0 for filter_str in filter_strings}

                for token in tokens:
                    should_keep = True
                    for filter_str in filter_strings:
                        if filter_str in token:
                            removed_counts[filter_str] += 1
                            should_keep = False
                            break  # No need to check other filter strings for this token

                    if should_keep:
                        filtered_tokens.append(token)

                tokens = filtered_tokens

                # Log filtering results for this sentence (only if genes were removed)
                total_removed = original_count - len(tokens)
                if total_removed > 0:
                    filter_details = ", ".join(
                        [f"{count} with '{filter_str}'" for filter_str, count in removed_counts.items() if count > 0]
                    )
                    logger.debug(f"Filtered {total_removed} genes from sentence: {filter_details}")

            # Take first max_length tokens and apply gene special token if specified
            final_tokens = tokens[:max_length] if len(tokens) > max_length else tokens

            # Add gene special token in front of each token if specified
            if gene_special_token:
                final_tokens = [f"{gene_special_token}{delimiter}{token}" for token in final_tokens]

            # Join with delimiter
            truncated.append(delimiter.join(final_tokens))

        batch[column_name] = truncated
        return batch

    # Auto-determine number of processes based on dataset size
    if num_proc is None:
        dataset_size = len(dataset)
        if dataset_size > 50000:
            import os

            num_proc = min(os.cpu_count() or 1, 4)  # Use up to 4 processes for large datasets
        else:
            num_proc = 1  # Single process for smaller datasets to avoid overhead

    # Log filtering information if filter strings are provided
    if filter_strings:
        logger.info(f"Filtering genes containing: {filter_strings}")

    # Create description for progress bar
    desc_parts = []
    if filter_strings:
        desc_parts.append(f"Filtering genes with {filter_strings}")
    desc_parts.append(f"Truncating {column_name} to {max_length} genes")
    desc = " and ".join(desc_parts)

    # Use larger batch size for better performance and disable caching for memory efficiency
    processed_dataset = dataset.map(
        _truncate_batch,
        batched=True,
        batch_size=10000,  # Larger batch size for better performance
        desc=desc,
        load_from_cache_file=False,  # Disable caching to avoid disk I/O overhead
        num_proc=num_proc,
    )

    # Log summary statistics if filtering was applied
    if filter_strings:
        logger.info(f"Gene filtering and truncation completed for {len(processed_dataset)} samples")

    return processed_dataset


_GENE_TOKEN_RE = re.compile(r"[A-Z0-9-]+")  # fast & permissive for symbols like MT-CO1, RPS27A


def _is_gene_like(raw: str) -> bool:
    """Return True if `raw` looks like a gene symbol (uppercase, digits, hyphens)."""
    if not raw:
        return False
    tok = raw.strip()
    # trim common trailing punctuation/quotes
    for char in ".''\"":
        tok = tok.strip(char)
    if tok in {"...", "…"}:
        return False
    # Some datasets inject 'and' in lists: make it not gene-like.
    if tok.upper() in {"AND"}:
        return False
    # Require uppercase-ish & pattern match
    return tok == tok.upper() and bool(_GENE_TOKEN_RE.fullmatch(tok)) and len(tok) >= 2


def _is_actual_gene_symbol(token: str) -> bool:
    """
    Return True if the token is likely a gene symbol, filtering out common English words.

    This is a more restrictive check than _is_gene_like to avoid false positives
    with common English words that happen to be uppercase.
    """
    if not token:
        return False

    # Common English words that shouldn't be considered genes
    common_words = {
        "GENES",
        "LIKE",
        "ARE",
        "IS",
        "THE",
        "AND",
        "OR",
        "OF",
        "IN",
        "TO",
        "FOR",
        "WITH",
        "HALLMARKS",
        "MARKERS",
        "EXPRESSION",
        "LEVELS",
        "HIGH",
        "LOW",
        "POSITIVE",
        "NEGATIVE",
        "CELL",
        "CELLS",
        "TYPE",
        "TYPES",
        "TISSUE",
        "TISSUES",
        "SAMPLE",
        "SAMPLES",
        "EFFECTOR",
        "MEMORY",
        "NAIVE",
        "ACTIVATED",
        "RESTING",
        "PROLIFERATING",
        "ALPHA",
        "BETA",
        "GAMMA",
        "DELTA",
        "EPSILON",
        "ZETA",
        "ETA",
        "THETA",
        "ALPHA-BETA",
        "CD8-POSITIVE",
        "CD4-POSITIVE",
        "CD8-NEGATIVE",
        "CD4-NEGATIVE",
        "TERMINALLY",
        "DIFFERENTIATED",
        "UNDIFFERENTIATED",
        "MATURE",
        "IMMATURE",
        "IMPORTANT",
        "KEY",
        "MAIN",
        "MAJOR",
        "MINOR",
        "SIGNIFICANT",
        "CRITICAL",
        "SHOWS",
        "SHOW",
        "DEMONSTRATES",
        "INDICATE",
        "INDICATES",
        "SUGGEST",
        "SUGGESTS",
        "ACTIVITY",
        "FUNCTION",
        "ROLE",
        "PATHWAY",
        "PATHWAYS",
        "NETWORK",
        "NETWORKS",
    }

    if token.upper() in common_words:
        return False

    # Require at least one digit OR a hyphen (many gene symbols have these)
    # This helps distinguish gene symbols from common English words
    has_digit = any(c.isdigit() for c in token)
    has_hyphen = "-" in token

    # If it's a short token (2-3 chars) without digits or hyphens, be more restrictive
    if len(token) <= 3 and not (has_digit or has_hyphen):
        # Allow some known short gene patterns
        if token in {"TP53", "MYC", "RAS", "AKT", "MDM", "BCL", "BAX", "BAD", "BID"}:
            return True
        # Reject purely alphabetic short tokens that are likely English words
        if token.isalpha():
            return False

    return True


def _clean_gene_token(raw: str) -> str:
    """Normalize a raw token into a clean gene symbol (strip punctuation/whitespace)."""
    return raw.strip().strip(".,;:!'’\"").upper()


def _find_gene_span(parts: list[str]) -> tuple[int, int]:
    """
    Find a gene sppan

    From a list of comma-split parts, find the start/end indices [start, end)
    of the longest contiguous run where most items are gene-like.

    Returns
    -------
    start : int
        Inclusive start index of the best span.
    end : int
        Exclusive end index of the best span.
    """
    best_start = best_end = 0
    start = 0
    while start < len(parts):
        # Skip non-gene-like until we hit a gene-like token
        while start < len(parts) and not _is_gene_like(parts[start].strip()):
            start += 1
        if start >= len(parts):
            break
        end = start
        while end < len(parts) and _is_gene_like(parts[end].strip()):
            end += 1
        if (end - start) > (best_end - best_start):
            best_start, best_end = start, end
        start = end + 1
    return best_start, best_end


def truncate_semantic_cell_sentence(
    sentence: str,
    max_genes: int,
    filter_strings: Iterable[str] | None = None,
    add_ellipsis_when_truncated: bool = True,
    gene_special_token: str = None,
    delimiter: str = " ",
) -> str:
    """
    Truncate the gene list inside a semantic cell sentence

    Truncate the comma-separated gene list inside a semantic cell sentence, preserving
    any prose before/after the list.

    Parameters
    ----------
    sentence : str
        A sentence containing arbitrary prose with a long, comma-separated list of gene symbols.
        Example: "Genes like MALAT1, EEF1A1, MT-CO1, ... are hallmarks of ..."
    max_genes : int
        Maximum number of genes to keep from the detected list (from the start of the list).
    filter_strings : Iterable[str], optional
        Substrings; any gene containing any of these will be removed prior to truncation
        (e.g., ['RPS', 'RPL'] to drop ribosomal genes).
    add_ellipsis_when_truncated : bool, default True
        If True, append an ellipsis right after the truncated gene list when there were
        more genes originally.
    gene_special_token : str, optional
        Special token to add in front of each gene name (e.g., '[GENE]'). If None,
        no special token is added.
    delimiter : str, optional
        Delimiter to use between tokens (default: " "). Can be customized for
        different training experiments.

    Returns
    -------
    str
        The sentence with its gene list filtered and truncated, with all non-gene text preserved.

    Notes
    -----
    - Gene span detection uses the longest contiguous run of "gene-like" tokens between commas.
    - "Gene-like" ≈ uppercase alphanumerics with optional hyphens, length ≥ 2.
    - Gracefully ignores tokens like "...", "and", and trailing punctuation.
    """
    if max_genes <= 0:
        # Degenerate request: drop the gene list entirely
        parts = list(sentence.split(","))
        s, e = _find_gene_span(parts)
        if e <= s:
            return sentence  # no detected list
        prefix = ", ".join(parts[:s]).strip()
        suffix = ", ".join(parts[e:]).strip()
        glue_left = (prefix + ", ") if prefix else ""
        glue_right = (", " + suffix) if suffix else ""
        return (glue_left + glue_right).strip().strip(", ")

    # Optimized gene extraction: find first and last gene positions, assume genes in between
    parts = sentence.split(",")

    # Find first part containing a gene (search from start)
    first_gene_part = -1
    first_gene = None
    for i, part in enumerate(parts):
        tokens = part.strip().split()
        for token in tokens:
            clean_token = _clean_gene_token(token)
            if _is_gene_like(clean_token) and _is_actual_gene_symbol(clean_token):
                first_gene_part = i
                first_gene = clean_token
                break
        if first_gene_part != -1:
            break

    # Find last part containing a gene (search from end)
    last_gene_part = -1
    last_gene = None
    for i in range(len(parts) - 1, -1, -1):
        part = parts[i]
        tokens = part.strip().split()
        for token in tokens:
            clean_token = _clean_gene_token(token)
            if _is_gene_like(clean_token) and _is_actual_gene_symbol(clean_token):
                last_gene_part = i
                last_gene = clean_token
                break
        if last_gene_part != -1:
            break

    # If no genes found, return original
    if first_gene_part == -1 or last_gene_part == -1:
        return sentence

    # Extract genes: first gene + all parts between + last gene (if different)
    original_genes = [first_gene]

    # Add genes from intermediate parts (assume they are all genes)
    for i in range(first_gene_part + 1, last_gene_part):
        part_clean = _clean_gene_token(parts[i])
        if part_clean:  # Only add non-empty cleaned tokens
            original_genes.append(part_clean)

    # Add last gene if it's different from first
    if last_gene_part != first_gene_part and last_gene:
        original_genes.append(last_gene)

    # Optional filtering
    filtered_genes = original_genes
    if filter_strings:
        fs = list(filter_strings)
        filtered_genes = []
        for g in original_genes:
            if not any(f in g for f in fs):
                filtered_genes.append(g)

    # Truncate to max_genes (no ellipsis)
    truncated_genes = filtered_genes[:max_genes]

    # Apply gene special token if specified
    if gene_special_token:
        processed_truncated_genes = [f"{gene_special_token}{delimiter}{gene}" for gene in truncated_genes]
    else:
        processed_truncated_genes = truncated_genes

    # Simple replacement approach: replace the original gene sequence with truncated list
    original_sequence = ", ".join(original_genes)
    truncated_sequence = delimiter.join(processed_truncated_genes)

    # Try to find and replace the gene sequence
    if original_sequence in sentence:
        result = sentence.replace(original_sequence, truncated_sequence, 1)
    else:
        # Fallback: more complex replacement by removing genes one by one
        result = sentence

        # Remove genes that are not in the truncated list
        genes_to_remove = []
        if filter_strings:
            # If filtering, remove filtered genes
            for gene in original_genes:
                if any(f in gene for f in filter_strings):
                    genes_to_remove.append(gene)

        # Also remove genes beyond max_genes from the filtered list
        if len(filtered_genes) > max_genes:
            genes_to_remove.extend(filtered_genes[max_genes:])

        # Remove the unwanted genes
        for gene in genes_to_remove:
            # Try different patterns to remove the gene
            patterns_to_try = [
                f", {gene},",  # middle gene
                f", {gene} ",  # gene followed by space
                f" {gene},",  # gene preceded by space
                f" {gene} ",  # gene with spaces
                gene,  # just the gene
            ]

            for pattern in patterns_to_try:
                if pattern in result:
                    if pattern == f", {gene},":
                        result = result.replace(pattern, ",", 1)
                    elif pattern == f", {gene} ":
                        result = result.replace(pattern, " ", 1)
                    elif pattern == f" {gene},":
                        result = result.replace(pattern, ",", 1)
                    elif pattern == f" {gene} ":
                        result = result.replace(pattern, " ", 1)
                    else:
                        result = result.replace(pattern, "", 1)
                    break

    # Clean up extra spaces and commas
    result = result.replace("  ", " ").replace(", ,", ",").replace(",,", ",")
    result = result.strip()

    # Ensure sentence-ending punctuation is preserved
    if sentence.rstrip()[-1:] in {".", "!", "?"} and result.rstrip()[-1:] not in {".", "!", "?"}:
        result += "."

    return result


def truncate_semantic_cell_sentences_dataset(
    dataset,
    column_name: str,
    max_genes: int,
    num_proc: int | None = None,
    filter_strings: list[str] | None = None,
    add_ellipsis_when_truncated: bool = False,
    gene_special_token: str = None,
    delimiter: str = " ",
):
    """
    Vectorized version for HuggingFace Datasets.

    Parameters
    ----------
    dataset : Dataset
        HuggingFace dataset containing semantic cell sentences.
    column_name : str
        Name of the column with the semantic cell sentences.
    max_genes : int
        Maximum number of genes to keep from the detected gene list.
    num_proc : int, optional
        Parallel workers; auto-chosen based on dataset size if None.
    filter_strings : list of str, optional
        Substrings to filter out from the gene symbols.
    add_ellipsis_when_truncated : bool, default True
        Append ellipsis if truncation occurred.
    gene_special_token : str, optional
        Special token to add in front of each gene name (e.g., '[GENE]'). If None,
        no special token is added.
    delimiter : str, optional
        Delimiter to use between tokens (default: " "). Can be customized for
        different training experiments.

    Returns
    -------
    Dataset
        Dataset with truncated semantic cell sentences in `column_name`.
    """

    def _process_batch(batch):
        out = []
        for s in batch[column_name]:
            out.append(
                truncate_semantic_cell_sentence(
                    s,
                    max_genes=max_genes,
                    filter_strings=filter_strings,
                    add_ellipsis_when_truncated=add_ellipsis_when_truncated,
                    gene_special_token=gene_special_token,
                    delimiter=delimiter,
                )
            )
        batch[column_name] = out
        return batch

    # Auto determine num_proc
    if num_proc is None:
        sz = len(dataset)
        if sz > 50000:
            import os

            num_proc = min(os.cpu_count() or 1, 4)
        else:
            num_proc = 1

    desc = f"Truncating gene lists in '{column_name}' to {max_genes}"
    if filter_strings:
        desc += f" (filter: {filter_strings})"

    return dataset.map(
        _process_batch,
        batched=True,
        batch_size=10000,
        desc=desc,
        load_from_cache_file=False,
        num_proc=num_proc,
    )


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


class PerDatasetLossLogger(nn.Module):
    """
    Wrap a Sentence-Transformers loss to log per-dataset training loss.

    Parameters
    ----------
    inner_loss : nn.Module
        The actual loss instance (e.g., MultipleNegativesRankingLoss).
    dataset_name : str
        Dataset identifier used in log keys, e.g. 'cellxgene' or 'geo'.
    log_backend : {'wandb', 'tensorboard', 'both', 'none'}, optional
        Where to send logs. Defaults to 'wandb' if Weights & Biases is available
        and initialized; otherwise 'none'.
    tb_writer : SummaryWriter, optional
        TensorBoard writer if using 'tensorboard' or 'both'.
    log_prefix : str, optional
        Prefix for the scalar key, default is 'train/loss'.
    """

    def __init__(
        self,
        inner_loss: nn.Module,
        dataset_name: str,
        log_backend: str = "auto",
        tb_writer: SummaryWriter | None = None,
        log_prefix: str = "train/loss",
    ) -> None:
        super().__init__()
        self.inner = inner_loss
        self.dataset_name = dataset_name
        self.tb_writer = tb_writer
        self.log_prefix = log_prefix
        self._step = 0

        if log_backend == "auto":
            if wandb is not None and getattr(wandb, "run", None) is not None:
                self.log_backend = "wandb"
            else:
                self.log_backend = "none"
        else:
            self.log_backend = log_backend

    def forward(self, features: dict[str, Any], labels: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass for the PerDatasetLossLogger."""
        loss: torch.Tensor = self.inner(features, labels)
        self._step += 1

        key = f"{self.log_prefix}/{self.dataset_name}"
        val = float(loss.detach().item())

        # Lightweight, optional logging
        logging.info("%s: %.6f", key, val)
        if self.log_backend in ("tensorboard", "both") and self.tb_writer is not None:
            self.tb_writer.add_scalar(key, val, self._step)
        if self.log_backend in ("wandb", "both") and (wandb is not None and getattr(wandb, "run", None) is not None):
            # Let W&B manage the step; keeping commit=False makes multiple logs per step play nice
            wandb.log({key: val}, commit=False)

        return loss


def _instantiate_loss(loss_cls: type, model: nn.Module, **kwargs: Any) -> nn.Module:
    """
    Instantiate a Sentence-Transformers loss class with best-effort argument matching.

    Many ST loss constructors accept `model` (e.g., MultipleNegativesRankingLoss),
    while some custom ones might not. This helper inspects the signature and only
    passes supported args.
    """
    sig = inspect.signature(loss_cls.__init__)
    accepted = set(sig.parameters.keys())
    ctor_kwargs = {}
    if "model" in accepted:
        ctor_kwargs["model"] = model
    for k, v in kwargs.items():
        if k in accepted:
            ctor_kwargs[k] = v
    return loss_cls(**ctor_kwargs)  # type: ignore[misc]


def get_loss(
    *,
    model: nn.Module,
    dataset_type: str,
    loss_name: str | None = None,
    dataset_name: str | None = None,
    log_backend: str = "auto",
    tb_writer: SummaryWriter | None = None,
    log_prefix: str = "train/loss",
    **loss_kwargs: Any,
) -> nn.Module:
    """
    Return a wrapped Sentence-Transformers loss with per-dataset logging.

    Parameters
    ----------
    model : nn.Module
        The Sentence-Transformers model (or compatible) used by the loss.
    dataset_type : {'pairs', 'multiplets'}
        The dataset structure; used to validate/choose defaults.
    loss_name : str, optional
        Name of the loss in `sentence_transformers.losses` (e.g., 'MultipleNegativesRankingLoss').
        If None, a sensible default is chosen based on `dataset_type`.
    dataset_name : str, optional
        Name used to tag logs (recommended: match the key in your `train_dataset` dict).
    log_backend : {'auto', 'wandb', 'tensorboard', 'both', 'none'}, default 'auto'
        Where to emit per-dataset training loss. 'auto' uses W&B if available, else none.
    tb_writer : SummaryWriter, optional
        TensorBoard writer when using 'tensorboard' or 'both'.
    log_prefix : str, default 'train/loss'
        Scalar key prefix. Final key will be f"{log_prefix}/{dataset_name}".
    **loss_kwargs
        Extra keyword arguments forwarded to the underlying loss constructor
        (e.g., `scale=20.0`, `similarity_fct=...`).

    Returns
    -------
    nn.Module
        A loss module ready for `SentenceTransformerTrainer(..., loss={dataset_name: loss})`
        or the classic `fit(train_objectives=[(dl, loss), ...])` API.

    Notes
    -----
    - This function validates `loss_name` against simple presets by `dataset_type`
      and wraps the instantiated loss in `PerDatasetLossLogger` so that each forward
      call emits a scalar to your chosen backend.
    - Works seamlessly with W&B's Transformers callback: your custom scalars appear
      in the same run without interfering with trainer-reported metrics.
    """
    pairs_losses = {"ContrastiveLoss", "OnlineContrastiveLoss"}
    multiplets_losses = {
        "MultipleNegativesRankingLoss",
        "CachedMultipleNegativesRankingLoss",
        "CachedGISTEmbedLoss",
    }

    if dataset_type not in {"pairs", "multiplets"}:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'. Expected 'pairs' or 'multiplets'.")

    if dataset_type == "pairs":
        default = "ContrastiveLoss"
        allowed = pairs_losses
    else:
        default = "MultipleNegativesRankingLoss"
        allowed = multiplets_losses

    if loss_name is None:
        loss_name = default
    elif loss_name not in allowed:
        raise ValueError(
            f"Loss '{loss_name}' is not supported for {dataset_type} dataset. Choose from {sorted(allowed)}."
        )

    try:
        LossClass = getattr(losses, loss_name)
    except AttributeError as e:
        raise ValueError(f"Loss class '{loss_name}' not found in sentence_transformers.losses") from e

    # Instantiate the underlying loss, passing `model` only if the constructor expects it
    inner = _instantiate_loss(LossClass, model=model, **loss_kwargs)

    # If no dataset_name provided, fall back to the loss name
    ds_name = dataset_name or loss_name

    return PerDatasetLossLogger(
        inner_loss=inner,
        dataset_name=ds_name,
        log_backend=log_backend,
        tb_writer=tb_writer,
        log_prefix=log_prefix,
    )


'''
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
'''


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

        # rename positive column to "positive"
        if positive_col in proc.column_names and positive_col != "positive":
            proc = proc.rename_column(positive_col, "positive")

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


def copy_resolved_config(cfg, hydra_output_dir: Path, named_output_dir: Path) -> None:
    """
    Copy the resolved config to the output directory.

    Serialize the *instantiated* Hydra config twice: once in the Hydra run
    directory and once in the named output directory so downstream users can
    locate it without knowing the run-dir.

    The config is enriched with a few runtime fields first (git SHA, SLURM
    job-ID, command line, etc.).
    """
    import subprocess

    from hydra.utils import to_absolute_path
    from omegaconf import OmegaConf

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # --- enrich with runtime metadata -------------------------------------
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    cfg_dict["_meta"] = {
        "git_sha": git_sha,
        "cmd": " ".join([to_absolute_path("main.py"), *os.sys.argv[1:]]),
        "slurm_job_id": os.getenv("SLURM_JOB_ID") if cfg.slurm.store_id else None,
    }

    # -----------------------------------------------------------------------
    for target_dir in (hydra_output_dir, named_output_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        out = target_dir / "resolved_config.json"
        with out.open("w") as fp:
            json.dump(cfg_dict, fp, indent=2)
        logger.info("Wrote resolved Hydra config → %s", out)

    # Also copy *raw* yaml files for debugging
    orig_conf_dir = Path("conf").absolute()
    shutil.copytree(orig_conf_dir, named_output_dir / "conf_raw", dirs_exist_ok=True)
