import logging

logger = logging.getLogger(__name__)


def evaluate_annotation_accuracy(adata, true_key: str, inferred_key: str) -> float:
    """Get accurcary of sample annotations.

    Evaluate the accuracy of cell annotations by comparing original (true) labels
    to inferred labels stored in the same AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the cell annotations in `adata.obs`.
        This object serves as the data source for both original and inferred labels.
    true_key : str
        The column name in `adata.obs` that stores the original (ground-truth) cell annotations.
    inferred_key : str
        The column name in `adata.obs` that stores the inferred cell annotations.

    Returns
    -------
    float
        The fraction (between 0 and 1) of cells whose `inferred_key` matches the `true_key`.
        If the columns or the data are unavailable, raises an error.

    Raises
    ------
    ValueError
        If the specified keys are not present in `adata.obs` or if the
        annotation columns have different lengths.

    References
    ----------
    Data are sourced from `adata.obs[true_key]` and `adata.obs[inferred_key]`.

    Examples
    --------
    >>> # Suppose we have an AnnData with obs["original_label"] and obs["predicted_label"]
    >>> accuracy = evaluate_annotation_accuracy(adata, "original_label", "predicted_label")
    >>> print("Annotation accuracy:", accuracy)
    """
    # Check if both keys exist in adata.obs
    if true_key not in adata.obs:
        raise ValueError(f"The true_key '{true_key}' is not present in adata.obs.")
    if inferred_key not in adata.obs:
        raise ValueError(f"The inferred_key '{inferred_key}' is not present in adata.obs.")

    true_labels = adata.obs[true_key]
    inferred_labels = adata.obs[inferred_key]

    # Ensure the two columns have the same length (they should by definition, but just in case)
    if len(true_labels) != len(inferred_labels):
        raise ValueError("The length of true annotations differs from inferred annotations.")

    # Compare element-wise and calculate fraction of matches
    matches = true_labels == inferred_labels
    accuracy = matches.mean()

    # Optionally, log the result
    logger.info(
        "Evaluating annotation accuracy for keys '%s' and '%s'. Accuracy = %.2f%%",
        true_key,
        inferred_key,
        accuracy * 100,
    )

    return accuracy
