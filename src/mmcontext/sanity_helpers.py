from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score  # optional


def plot_pca(
    embeddings,
    labels=None,
    title=None,
    ax=None,
    *,
    cmap="tab10",
    point_size=18,
    legend_out=False,
):
    """
    2-D PCA scatter.

    Parameters
    ----------
    embeddings : (N, D) array-like
    labels     : sequence[str|int] or None
    title      : str | None
    ax         : matplotlib Axes | None
    cmap       : Matplotlib colormap name
    point_size : int – marker size passed to *scatter*
    legend_out : bool – True puts legend to the plot’s right
    """
    emb = np.asarray(embeddings)
    xy = PCA(n_components=2).fit_transform(emb)

    # normalise label input ----------------------------------------------
    if labels is not None:
        if "pandas" in str(type(labels)):
            labels = labels.values
        labels = np.asarray(labels)

    # categorical → map to ints
    if labels is not None and labels.dtype.kind in ("U", "S", "O"):
        uniq = np.unique(labels)
        lab2id = {lab: i for i, lab in enumerate(uniq)}
        colours = np.vectorize(lab2id.get)(labels)
    else:
        colours = labels
        uniq = None

    # plotting ------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    sc = ax.scatter(xy[:, 0], xy[:, 1], c=colours, cmap=cmap, s=point_size)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if title:
        ax.set_title(title)

    if uniq is not None:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=sc.cmap(lab2id[l] / max(lab2id.values())),
                label=l,
                markersize=6,
            )
            for l in uniq
        ]
        if legend_out:
            ax.legend(
                handles=handles,
                title="label",
                fontsize="small",
                bbox_to_anchor=(1.02, 0.5),
                loc="center left",
                borderaxespad=0.0,
            )
        else:
            ax.legend(handles=handles, title="label", fontsize="small")
    return ax


def cluster_variances(embeddings, labels, *, ddof: int = 0, return_silhouette: bool = False):
    """
    Compute the within- and between-cluster variances of a set of embeddings.

    Parameters
    ----------
    embeddings : array-like, shape (N, D)
    labels     : sequence of length N
    ddof       : delta-dof for *var*  (0 → population var, 1 → sample var)
    return_silhouette : bool, add sklearn silhouette score

    Returns
    -------
    within_var  : float
    between_var : float
    (silhouette : float)   # only if requested
    """
    emb = np.asarray(embeddings)
    labs = np.asarray(labels)

    df = pd.DataFrame(emb)
    df["label"] = labs

    within = df.groupby("label").var(ddof=ddof).mean().mean()
    between = df.groupby("label").mean().var(ddof=ddof).mean()

    if return_silhouette:
        sil = silhouette_score(emb, labs, metric="cosine")
        return within, between, sil
    return within, between


def stack_embeddings(df: pd.DataFrame, column: str = "embedding") -> np.ndarray:
    """
    Stack a DataFrame column of list-like vectors into a 2-D ``np.ndarray``.

    Parameters
    ----------
    df : pandas.DataFrame
        Table that contains an *embedding* column.
        **Source** – your dataframe shown in the chat (five rows, each list
        holding the numeric embedding of a token).
    column : str, default "embedding"
        Name of the column that stores the vectors.

    Returns
    -------
    ndarray, shape (n_samples, emb_dim)
        Two-dimensional array where each row is the embedding that was stored
        in ``df[column]`` for the corresponding sample.

    Raises
    ------
    ValueError
        If the column does not exist or the vectors have inconsistent length.

    Examples
    --------
    >>> arr = stack_embeddings(df)
    >>> arr.shape  # (5, 768) for typical BERT-sized embeddings
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    # Convert the Series of lists to a single ndarray
    try:
        embedded: Sequence[Sequence[float]] = df[column].tolist()
        arr = np.asarray(embedded, dtype=np.float32)
    except Exception:
        raise

    # Sanity-check: all rows must have equal length
    if arr.ndim != 2:
        raise ValueError("Embeddings have inconsistent dimensions; make sure every row stores a list of equal length.")

    return arr


def class_scatter(emb: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """
    Compute simple clustering diagnostics for an embedding.

    Parameters
    ----------
    emb : np.ndarray of shape (n_samples, d)
        The embedding vectors.
    labels : array-like of length n_samples
        Class labels.

    Returns
    -------
    dict
        * ``within_var``   – average squared distance to the class mean
        * ``between_var``  – average squared distance between class means
        * ``silhouette``   – sklearn’s silhouette coefficient
    """
    # means per class
    uniq = np.unique(labels)
    means = np.vstack([emb[labels == c].mean(0) for c in uniq])

    # assign class mean to each sample
    mean_per_sample = np.vstack([means[np.where(uniq == c)[0][0]] for c in labels])

    within = np.mean(np.sum((emb - mean_per_sample) ** 2, axis=1))
    # pair-wise mean distances
    diffs = means[:, None, :] - means[None, :, :]
    between = np.mean(np.sum(diffs**2, axis=-1)[np.triu_indices(len(uniq), 1)])

    sil = silhouette_score(emb, labels, metric="euclidean")

    print(
        "scatter: within %.4f – between %.4f – silhouette %.3f",
        within,
        between,
        sil,
    )
    return {"within_var": within, "between_var": between, "silhouette": sil}
