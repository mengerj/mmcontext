import logging

import numpy as np
import pandas as pd
from numpy.linalg import norm

logger = logging.getLogger(__name__)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

    Parameters
    ----------
    x : np.ndarray
        First embedding vector.
    y : np.ndarray
        Second embedding vector.

    Returns
    -------
    float
        Cosine similarity between x and y.
    """
    denom = norm(x) * norm(y)
    return float(np.dot(x, y) / denom) if denom else 0.0


def evaluate_modality_alignment(df: pd.DataFrame) -> (float, float):
    """
    Evaluate two cross-modal alignment scores for omics/text embeddings.

    How close are pairs of cross-modal embeddings in the embedding space, compared to false pairs?
    The input DataFrame is assumed to have exactly one 'omics' and one 'text'
    embedding for each unique ID (e.g., cell ID) in order. Each row contains:
    - A vector of float embeddings in one column (e.g. 'embedding'),
    - A modality column with values in {'omics', 'text'}.

    The function calculates two metrics:

    1) Modality-gap-irrelevant score: For each (omics, text) pair i,
       compare the similarity of that true pair only to other cross-modal
       similarities (ignoring any intra-modal comparisons).

       The score is the average fraction of cross-modal pairs whose similarity
       is less than the true pair's similarity.

    2) Full-comparison score: For each (omics, text) pair i,
       compare the similarity of that true pair to *both* other cross-modal
       pairs and all intra-modal pairs.

       The score is again the average fraction of *all* other pairs (cross- or
       intra-modal) whose similarity is less than the true pair's similarity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns:
            - 'embedding_type': (e.g. 'omics' or 'text')
            - 'embedding': vector of floats (list or np.ndarray)
            - some ID columns to match pairs (e.g. 'sample_index').
        It is assumed each ID has exactly two rows: one 'omics' and one 'text'.

    Returns
    -------
    (float, float)
        A tuple (score_modality_gap_irrelevant, score_full_comparison).
    """
    logger.info("Starting modality alignment evaluation.")

    # Convert embedding column to numpy arrays (if stored as lists).
    df["embedding"] = df["embedding"].apply(np.array)

    # Separate data for each modality, indexing by some ID.
    # Here we assume there's a column 'sample_index' that identifies the pairs.
    # Adapt if your ID column is different.
    omics_df = df[df["embedding_type"] == "omics"].copy()
    text_df = df[df["embedding_type"] == "text"].copy()

    # Merge omics and text on the ID to get direct pairs
    # (Assuming 'sample_index' is the matching column).
    merged = pd.merge(
        omics_df[["sample_index", "embedding"]],
        text_df[["sample_index", "embedding"]],
        on="sample_index",
        suffixes=("_omics", "_text"),
    )

    # Build lists for all cross-modal pairs
    omics_embeddings = omics_df["embedding"].values
    text_embeddings = text_df["embedding"].values

    # For easy iteration over cross-modal combos
    all_cross_pairs = []
    for _i, emb_o in enumerate(omics_embeddings):
        for _j, emb_t in enumerate(text_embeddings):
            all_cross_pairs.append(cosine_similarity(emb_o, emb_t))
    all_cross_pairs = np.array(all_cross_pairs)

    # Build lists for all intra-modal pairs
    # We do omics-omics and text-text separately.
    all_omics_pairs = []
    omics_embeddings_list = omics_embeddings.tolist()
    for i in range(len(omics_embeddings_list)):
        for j in range(i + 1, len(omics_embeddings_list)):
            all_omics_pairs.append(cosine_similarity(omics_embeddings_list[i], omics_embeddings_list[j]))
    all_omics_pairs = np.array(all_omics_pairs)

    all_text_pairs = []
    text_embeddings_list = text_embeddings.tolist()
    for i in range(len(text_embeddings_list)):
        for j in range(i + 1, len(text_embeddings_list)):
            all_text_pairs.append(cosine_similarity(text_embeddings_list[i], text_embeddings_list[j]))
    all_text_pairs = np.array(all_text_pairs)

    # Prepare accumulators for each metric
    ranks_modality_gap_irrelevant = []
    ranks_full_comparison = []

    # For each 'true' pair (row in merged), get its sim and rank it vs others
    for _idx, row in merged.iterrows():
        emb_o = row["embedding_omics"]
        emb_t = row["embedding_text"]
        true_sim = cosine_similarity(emb_o, emb_t)

        # 1) Compare to cross-modal only
        # Count how many cross-modal similarities are below 'true_sim'
        # Subtract 1 if needed to exclude the pair i,i from the distribution
        # but that’s typically negligible if each ID is unique.
        cross_count_below = np.sum(all_cross_pairs < true_sim)
        # Turn into a fraction
        rank_cross = cross_count_below / len(all_cross_pairs)
        ranks_modality_gap_irrelevant.append(rank_cross)

        # 2) Compare to cross-modal + intra-modal
        # Combine cross-modal array with omics-omics & text-text
        # For a single “true” ID i, you might also exclude the sim of
        # (emb_o, emb_o) or (emb_t, emb_t) if it appears, but typically
        # we’re ignoring self-sim because it’s not in the data.
        all_pairs_combined = np.concatenate([all_cross_pairs, all_omics_pairs, all_text_pairs])
        count_below = np.sum(all_pairs_combined < true_sim)
        rank_full = count_below / len(all_pairs_combined)
        ranks_full_comparison.append(rank_full)

    # Average rank across all pairs
    modality_gap_irrelevant_score = float(np.mean(ranks_modality_gap_irrelevant))
    full_comparison_score = float(np.mean(ranks_full_comparison))

    logger.info("Modality-gap-irrelevant score: %f", modality_gap_irrelevant_score)
    logger.info("Full-comparison score: %f", full_comparison_score)

    return modality_gap_irrelevant_score, full_comparison_score
