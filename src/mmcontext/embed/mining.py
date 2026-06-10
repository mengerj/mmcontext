"""Hard-negative mining helpers for the MMContext pipeline.

These wrap :func:`sentence_transformers.util.mine_hard_negatives` so the same
core can back both an offline exploration script
(``scripts/explore_hard_negatives.py``) and a future in-training re-mining
callback for ``scripts/train_config.py``.

Mining only makes sense against a model that already has a meaningful joint
embedding space — mining with a freshly initialised bimodal model produces
noise. The input is always an ``anchor`` / ``positive`` dataset as produced by
:func:`~mmcontext.embed.prepare_dataset` (with ``use_hard_negatives=False`` for
the omics path, so the dataset's own pre-baked negatives are dropped first):

* For ``modality="bimodal"`` datasets the ``anchor`` column holds omics ids
  (``"omics:..."``) that the model resolves through its attached VectorStore;
  the ``positive`` texts form the corpus, so mined negatives are confusable
  cell descriptions in the shared space.
* For text datasets both columns are plain text.

Because the sentence-transformers trainer samples each batch from a single
dataset (in-batch negatives stay within-dataset), mining should likewise be run
**per dataset** rather than over a merged corpus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pandas as pd
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class MiningConfig:
    """Parameters forwarded to :func:`sentence_transformers.util.mine_hard_negatives`.

    The defaults mirror ``scripts/finetune_tiny.py`` and are a sensible starting
    point for an already-aligned model. ``max_score`` is intentionally on by
    default: as the model sharpens, the closest candidates increasingly become
    *false* negatives (descriptions that are actually correct), and the ceiling
    drops them.

    Parameters
    ----------
    num_negatives : int, default 3
        Negatives to mine per anchor.
    range_min : int, default 1
        Skip the N closest candidates (rank 0 is usually the true positive).
    range_max : int or None, default None
        Only consider candidates up to this rank (``None`` = no upper bound).
    max_score : float or None, default 0.95
        Drop candidates scoring above this ceiling (false-negative guard).
    min_score : float or None, default None
        Drop candidates scoring below this floor (too-easy guard).
    absolute_margin : float or None, default None
        Negative score must be at least this far below the positive score.
    relative_margin : float or None, default None
        Negative score must be below ``relative_margin * positive_score``
        (e.g. ``0.95``).
    sampling_strategy : {"top", "random"}, default "top"
        Which qualifying candidates to keep.
    use_faiss : bool, default False
        Use FAISS for the similarity search (recommended for large corpora).
    batch_size : int, default 32
        Encoding batch size for the mining search.
    """

    num_negatives: int = 3
    range_min: int = 1
    range_max: int | None = None
    max_score: float | None = 0.95
    min_score: float | None = None
    absolute_margin: float | None = None
    relative_margin: float | None = None
    sampling_strategy: Literal["top", "random"] = "top"
    use_faiss: bool = False
    batch_size: int = 32

    def as_kwargs(self) -> dict[str, Any]:
        """Return the config as ``mine_hard_negatives`` keyword arguments."""
        return {
            "num_negatives": self.num_negatives,
            "range_min": self.range_min,
            "range_max": self.range_max,
            "max_score": self.max_score,
            "min_score": self.min_score,
            "absolute_margin": self.absolute_margin,
            "relative_margin": self.relative_margin,
            "sampling_strategy": self.sampling_strategy,
            "use_faiss": self.use_faiss,
            "batch_size": self.batch_size,
        }


def mine_negatives(
    pairs: Dataset,
    model: SentenceTransformer,
    cfg: MiningConfig,
    *,
    anchor_column_name: str = "anchor",
    positive_column_name: str = "positive",
    output_format: Literal["triplet", "n-tuple", "labeled-pair", "labeled-list"] = "n-tuple",
    output_scores: bool = False,
    verbose: bool = True,
) -> Dataset:
    """Mine model-specific hard negatives for an ``anchor`` / ``positive`` dataset.

    Thin wrapper around :func:`sentence_transformers.util.mine_hard_negatives`
    that applies a :class:`MiningConfig`. The model is switched to ``eval`` mode
    for the duration of mining and restored to its prior mode afterwards.

    Parameters
    ----------
    pairs : datasets.Dataset
        Dataset with at least *anchor_column_name* and *positive_column_name*.
        For the bimodal path the anchors must be resolvable by the model's
        attached VectorStore.
    model : sentence_transformers.SentenceTransformer
        An *already-trained* model (mining a fresh model yields noise).
    cfg : MiningConfig
        Mining parameters.
    anchor_column_name, positive_column_name : str
        Column names to use as query and corpus.
    output_format : {"triplet", "n-tuple", "labeled-pair", "labeled-list"}
        Forwarded to ``mine_hard_negatives``. ``"n-tuple"`` gives
        ``anchor, positive, negative_1, ...`` which is what
        :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss`
        expects.
    output_scores : bool, default False
        If True, ``mine_hard_negatives`` appends similarity-score columns.
    verbose : bool, default True
        Print mining statistics.

    Returns
    -------
    datasets.Dataset
        The mined dataset in the requested *output_format*.
    """
    from sentence_transformers.util import mine_hard_negatives

    was_training = model.training
    model.eval()
    try:
        mined = mine_hard_negatives(
            pairs,
            model,
            anchor_column_name=anchor_column_name,
            positive_column_name=positive_column_name,
            output_format=output_format,
            output_scores=output_scores,
            verbose=verbose,
            **cfg.as_kwargs(),
        )
    finally:
        if was_training:
            model.train()

    logger.info(
        "Mined %d rows from %d pairs (columns=%s)",
        len(mined),
        len(pairs),
        mined.column_names,
    )
    return mined


def mining_report(
    mined: Dataset,
    model: SentenceTransformer,
    *,
    anchor_column_name: str = "anchor",
    positive_column_name: str = "positive",
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Build a human-readable table of what was mined, with cosine sims.

    Re-encodes the anchor, positive and each ``negative_*`` column (cosine,
    via normalised embeddings) so each row shows the positive similarity, each
    negative similarity, and the positive-minus-negative margin. This is
    version-independent (it does not rely on ``output_scores``) and gives the
    positive similarity, which the mining call itself does not return.

    Parameters
    ----------
    mined : datasets.Dataset
        An ``n-tuple`` mined dataset (``anchor``, ``positive``, ``negative_*``).
    model : sentence_transformers.SentenceTransformer
        The model used for mining (used to recompute similarities).
    anchor_column_name, positive_column_name : str
        Column names in *mined*.
    max_rows : int or None
        Limit the report to the first N rows (``None`` = all).

    Returns
    -------
    pandas.DataFrame
        One row per anchor: anchor, positive, ``pos_sim``, and for each mined
        negative its text, ``neg_<i>_sim`` and ``margin_<i>`` (pos − neg).
    """
    import numpy as np
    import pandas as pd

    ds = mined.select(range(min(max_rows, len(mined)))) if max_rows else mined
    neg_cols = sorted(
        (c for c in ds.column_names if c.startswith("negative") and not c.endswith("_idx")),
        key=lambda c: (len(c), c),
    )

    def _enc(texts: list[str]) -> np.ndarray:
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    anchor_emb = _enc(list(ds[anchor_column_name]))
    pos_emb = _enc(list(ds[positive_column_name]))
    pos_sim = np.sum(anchor_emb * pos_emb, axis=1)

    rows: list[dict[str, Any]] = []
    for i in range(len(ds)):
        rows.append(
            {
                "anchor": ds[anchor_column_name][i],
                "positive": ds[positive_column_name][i],
                "pos_sim": round(float(pos_sim[i]), 4),
            }
        )

    for j, col in enumerate(neg_cols, start=1):
        neg_emb = _enc(list(ds[col]))
        neg_sim = np.sum(anchor_emb * neg_emb, axis=1)
        for i in range(len(ds)):
            rows[i][f"negative_{j}"] = ds[col][i]
            rows[i][f"neg_{j}_sim"] = round(float(neg_sim[i]), 4)
            rows[i][f"margin_{j}"] = round(float(pos_sim[i] - neg_sim[i]), 4)

    return pd.DataFrame(rows)
