import logging

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, roc_curve

from mmcontext.file_utils import compute_cosine_similarity

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def zero_shot_classification_roc(
    adata,
    model,
    label_key: str = "cell_type",
    emb_key: str = "mmcontext_emb",
    text_template: str = "A sample of {} from a healthy individual",
    device: str = "cpu",
) -> tuple[float, dict[str, float]]:
    """
    Compute a zero-shot cell property (cell type) ROC-AUC evaluation.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing embeddings in adata.obsm[emb_key],
        and true labels in adata.obs[label_key].
        Data source: user-provided single-cell dataset with embeddings.
    model : sentence_transformers.SentenceTransformer or similar
        The text encoder model used to embed property queries.
    label_key : str, optional
        The column in adata.obs that holds the ground-truth labels, by default 'cell_type'.
    emb_key : str, optional
        The key in adata.obsm to retrieve sample embeddings, by default 'omics_emb'.
    text_template : str, optional
        A text template that has a placeholder (e.g. {} for the label name).
        By default "A sample of {} from a healthy individual".
    device : str, optional
        Device for similarity computations ('cpu', 'cuda', or 'mps'), by default 'cpu'.

    Returns
    -------
    tuple[float, Dict[str, float]]
        - macro_auc: The average ROC-AUC across all distinct labels
        - auc_dict: A dictionary mapping label -> that label's ROC-AUC

    Notes
    -----
    The process:
    1) Extract unique labels from adata.obs[label_key].
    2) Build text queries with text_template.format(label).
    3) Embed them with `model.encode`.
    4) Compute pairwise cosine similarity between (num_samples, emb_dim) and (num_labels, emb_dim).
    5) Softmax across labels for each sample => predicted probability.
    6) For each label, compute one-vs-rest ROC-AUC via scikit-learn.
    7) Average to get a macro-level AUC.
    """
    # 1) Extract unique labels
    labels_series = adata.obs[label_key].astype(str)
    unique_labels = sorted(labels_series.unique())

    # 2) Build text queries
    logger.info("Building text queries for each label...")
    queries = [text_template.format(lbl) for lbl in unique_labels]

    # 3) Embed queries
    logger.info("Embedding %d label queries...", len(queries))
    query_embeddings = model.encode(queries)
    if isinstance(query_embeddings, list):
        query_embeddings = np.array(query_embeddings)

    # 4) Retrieve sample embeddings
    logger.info("Retrieving sample embeddings from adata.obsm[%s]...", emb_key)
    sample_embeddings = adata.obsm[emb_key]
    if not isinstance(sample_embeddings, np.ndarray):
        sample_embeddings = np.array(sample_embeddings)

    # 5) Compute similarity
    sim = compute_cosine_similarity(sample_embeddings, query_embeddings, device=device)
    # shape => (num_labels, num_samples)

    # 6) Softmax across labels for each sample => predicted probabilities
    # shape remains (num_labels, num_samples)
    probs = softmax(sim, axis=0)

    # 7) For each label, compute one-vs-rest ROC-AUC
    # y_true_mat = []
    # y_score_mat = []

    # n_samples = adata.n_obs
    label_aucs = {}
    for label_idx, label in enumerate(unique_labels):
        # Construct the ground-truth vector for this label
        # (1 for matching label, 0 otherwise)
        y_true = (labels_series.values == label).astype(int)

        # Predicted probability for this label is probs[label_idx, :]
        y_scores = probs[label_idx, :]

        # If all y_true are 0 or all 1, roc_auc_score can't be computed.
        # We'll skip or assign AUC=nan in that case.
        if len(np.unique(y_true)) < 2:
            logger.warning(f"Label '{label}' has only one class present in data. Skipping.")
            label_aucs[label] = np.nan
            continue

        auc_val = roc_auc_score(y_true, y_scores)
        label_aucs[label] = auc_val

    # 8) Average AUC across labels
    valid_aucs = [v for v in label_aucs.values() if not np.isnan(v)]
    if len(valid_aucs) == 0:
        macro_auc = float("nan")
    else:
        macro_auc = float(np.mean(valid_aucs))

    logger.info("Zero-shot cell-type prediction macro-AUC: %.4f", macro_auc)
    return macro_auc, label_aucs
