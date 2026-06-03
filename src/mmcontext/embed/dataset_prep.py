"""Unified dataset preparation for training and inference.

mmcontext aligns text and omics embeddings with a sentence-transformers
pipeline whose first module is an :class:`~mmcontext.modules.MMContextModule`.
That module resolves omics samples at runtime from an attached
:class:`~mmcontext.io.VectorStore`: any input string starting with the omics
prefix (``"omics:"`` by default) is looked up in the store, while plain strings
are tokenised as text.

This module provides a single, reusable preparation API that turns a raw
HuggingFace dataset into the column shape the trainer / encoder expects:

* :func:`prepare_dataset` — the pure ``dataset -> dataset`` core. It builds the
  ``anchor`` column (omics id for ``modality="bimodal"`` or cell-sentence text
  for ``modality="text"``) and, for ``purpose="train"``, resolves the
  ``positive`` / hard-negative columns.
* :func:`prepare_inference` — an orchestrator for the evaluation/inference path.
  It loads the referenced AnnData chunk, subsets the dataset to it, builds an
  anchor-ready dataset, and (for bimodal) builds and attaches a VectorStore to
  the model.

Training datasets carry ``positive`` and ``negative_*_idx`` columns; test
datasets carry only ``sample_idx``, ``cell_sentence_*`` and ``adata_link``.
The same :func:`prepare_dataset` call handles both via the ``purpose`` switch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from datasets import Dataset, DatasetDict

from mmcontext.utils import resolve_negative_indices_and_rename, truncate_cell_sentences

if TYPE_CHECKING:
    import anndata as ad
    from sentence_transformers import SentenceTransformer

    from mmcontext.io import VectorStore

logger = logging.getLogger(__name__)

Purpose = Literal["train", "inference"]
Modality = Literal["bimodal", "text"]


def _has_negative_idx_columns(ds: Dataset) -> bool:
    """Return True if *ds* exposes ``negative_*_idx`` columns."""
    return any(c.startswith("negative_") and c.endswith("_idx") for c in ds.column_names)


def _prepare_split(
    ds: Dataset,
    *,
    purpose: Purpose,
    modality: Modality,
    primary_cell_sentence: str,
    sample_id_col: str,
    positive_col: str,
    omics_prefix: str,
    use_hard_negatives: bool,
    truncate: bool,
    truncate_kwargs: dict | None,
) -> Dataset:
    """Prepare a single :class:`~datasets.Dataset` split.

    See :func:`prepare_dataset` for the meaning of the parameters.
    """
    if modality not in ("bimodal", "text"):
        raise ValueError(f"modality must be 'bimodal' or 'text', got {modality!r}")
    if purpose not in ("train", "inference"):
        raise ValueError(f"purpose must be 'train' or 'inference', got {purpose!r}")

    if primary_cell_sentence not in ds.column_names:
        raise KeyError(
            f"Primary cell-sentence column {primary_cell_sentence!r} not found. "
            f"Available columns: {ds.column_names}"
        )
    if modality == "bimodal" and sample_id_col not in ds.column_names:
        raise KeyError(
            f"modality='bimodal' requires the sample-id column {sample_id_col!r}. "
            f"Available columns: {ds.column_names}"
        )

    # ------------------------------------------------------------------
    # text modality: optionally truncate the cell sentence before it
    # becomes the anchor (gene filtering, max length, …).
    # ------------------------------------------------------------------
    if modality == "text" and truncate:
        tkwargs = dict(truncate_kwargs or {})
        max_length = tkwargs.pop("max_length", 64)
        ds = truncate_cell_sentences(ds, primary_cell_sentence, max_length=max_length, **tkwargs)

    if purpose == "train":
        return _prepare_train_split(
            ds,
            modality=modality,
            primary_cell_sentence=primary_cell_sentence,
            sample_id_col=sample_id_col,
            positive_col=positive_col,
            omics_prefix=omics_prefix,
            use_hard_negatives=use_hard_negatives,
        )
    return _prepare_inference_split(
        ds,
        modality=modality,
        primary_cell_sentence=primary_cell_sentence,
        sample_id_col=sample_id_col,
        omics_prefix=omics_prefix,
    )


def _prepare_train_split(
    ds: Dataset,
    *,
    modality: Modality,
    primary_cell_sentence: str,
    sample_id_col: str,
    positive_col: str,
    omics_prefix: str,
    use_hard_negatives: bool,
) -> Dataset:
    """Build a training-ready split: ``anchor`` + ``positive`` (+ ``negative_*``)."""
    if positive_col not in ds.column_names:
        raise KeyError(
            f"purpose='train' requires a positive column {positive_col!r}. "
            f"Available columns: {ds.column_names}"
        )

    resolve = use_hard_negatives and _has_negative_idx_columns(ds)
    if resolve:
        # Resolves negative_*_idx -> text, renames primary_cell_sentence -> 'anchor'
        # and positive_col -> 'positive'. Keep sample_idx so we can build the
        # omics anchor for the bimodal case afterwards.
        ds = resolve_negative_indices_and_rename(
            ds,
            primary_cell_sentence_col=primary_cell_sentence,
            positive_col=positive_col,
            negative_prefix="negative",
            index_col=sample_id_col,
            remove_index_col=False,
        )
    else:
        if primary_cell_sentence != "anchor":
            ds = ds.rename_column(primary_cell_sentence, "anchor")
        if positive_col != "positive":
            ds = ds.rename_column(positive_col, "positive")

    if modality == "bimodal":
        ds = ds.map(
            lambda row: {"anchor": f"{omics_prefix}{row[sample_id_col]}"},
            desc="Building omics anchors",
        )

    # Keep only resolved negatives (named "negative_N"); never raw "_idx" columns.
    neg_cols = sorted(c for c in ds.column_names if c.startswith("negative") and not c.endswith("_idx"))
    keep = ["anchor", "positive", *neg_cols]
    ds = ds.select_columns(keep)
    logger.info("Prepared train split: %d rows, columns=%s", len(ds), ds.column_names)
    return ds


def _prepare_inference_split(
    ds: Dataset,
    *,
    modality: Modality,
    primary_cell_sentence: str,
    sample_id_col: str,
    omics_prefix: str,
) -> Dataset:
    """Build an inference-ready split: ``anchor`` (+ ``sample_idx``/``adata_link``).

    No ``positive`` / ``negative_*`` columns are required or produced.
    """
    if modality == "bimodal":
        ds = ds.map(
            lambda row: {"anchor": f"{omics_prefix}{row[sample_id_col]}"},
            desc="Building omics anchors",
        )
    else:  # text
        if primary_cell_sentence != "anchor":
            ds = ds.map(
                lambda row: {"anchor": row[primary_cell_sentence]},
                desc="Building text anchors",
            )

    # Retain identifiers needed downstream to align embeddings with AnnData.
    keep = ["anchor"]
    for col in (sample_id_col, "adata_link", "share_link"):
        if col in ds.column_names and col not in keep:
            keep.append(col)
    ds = ds.select_columns(keep)
    logger.info("Prepared inference split: %d rows, columns=%s", len(ds), ds.column_names)
    return ds


def prepare_dataset(
    ds: Dataset | DatasetDict,
    *,
    purpose: Purpose = "train",
    modality: Modality = "bimodal",
    primary_cell_sentence: str = "cell_sentence_1",
    sample_id_col: str = "sample_idx",
    positive_col: str = "positive",
    omics_prefix: str = "omics:",
    use_hard_negatives: bool = True,
    truncate: bool = False,
    truncate_kwargs: dict | None = None,
) -> Dataset | DatasetDict:
    """Prepare a raw HuggingFace dataset for training or inference.

    The first module of an mmcontext pipeline
    (:class:`~mmcontext.modules.MMContextModule`) routes inputs by modality:
    strings prefixed with *omics_prefix* are resolved through an attached
    :class:`~mmcontext.io.VectorStore`, everything else is treated as text.
    This function builds the ``anchor`` column accordingly and, for training,
    the ``positive`` and hard-negative columns.

    Parameters
    ----------
    ds : datasets.Dataset or datasets.DatasetDict
        Raw dataset. A training dataset is expected to carry *positive_col* and
        (optionally) ``negative_*_idx`` columns; an inference/test dataset only
        needs *sample_id_col* / *primary_cell_sentence* (and an adata link for
        the bimodal path).
    purpose : {"train", "inference"}, default "train"
        ``"train"`` produces ``anchor`` + ``positive`` (+ ``negative_*``).
        ``"inference"`` produces only ``anchor`` plus retained identifier
        columns; ``positive`` / ``negative_*`` are neither required nor created.
    modality : {"bimodal", "text"}, default "bimodal"
        ``"bimodal"`` builds ``anchor = f"{omics_prefix}{sample_idx}"`` so the
        model resolves it via the VectorStore. ``"text"`` uses the
        *primary_cell_sentence* text as the anchor.
    primary_cell_sentence : str, default "cell_sentence_1"
        Column holding the cell sentence (used as the text anchor and as the
        primary column for negative resolution during training).
    sample_id_col : str, default "sample_idx"
        Column with sample identifiers (used to build omics anchors and to
        resolve negatives). Required for ``modality="bimodal"``.
    positive_col : str, default "positive"
        Column with the positive text. Required for ``purpose="train"``.
    omics_prefix : str, default "omics:"
        Prefix marking an omics sample id. Must match the prefix configured on
        the model's :class:`~mmcontext.modules.MMContextModule`.
    use_hard_negatives : bool, default True
        For ``purpose="train"``: resolve ``negative_*_idx`` columns into text
        negatives when present. Ignored when no such columns exist.
    truncate : bool, default False
        For ``modality="text"``: truncate the cell sentence via
        :func:`~mmcontext.utils.truncate_cell_sentences` before it becomes the
        anchor.
    truncate_kwargs : dict, optional
        Extra keyword arguments forwarded to
        :func:`~mmcontext.utils.truncate_cell_sentences` (e.g. ``max_length``,
        ``filter_strings``). Defaults to ``max_length=64``.

    Returns
    -------
    datasets.Dataset or datasets.DatasetDict
        Same container type as *ds*, with prepared columns.

    Examples
    --------
    >>> train_ds = prepare_dataset(raw, purpose="train", modality="bimodal")
    >>> train_ds.column_names
    ['anchor', 'positive', 'negative_1']
    >>> test_ds = prepare_dataset(raw_test, purpose="inference", modality="bimodal")
    >>> test_ds.column_names
    ['anchor', 'sample_idx', 'adata_link']
    """
    kwargs = dict(
        purpose=purpose,
        modality=modality,
        primary_cell_sentence=primary_cell_sentence,
        sample_id_col=sample_id_col,
        positive_col=positive_col,
        omics_prefix=omics_prefix,
        use_hard_negatives=use_hard_negatives,
        truncate=truncate,
        truncate_kwargs=truncate_kwargs,
    )

    if isinstance(ds, DatasetDict):
        return DatasetDict({name: _prepare_split(split, **kwargs) for name, split in ds.items()})
    if isinstance(ds, Dataset):
        return _prepare_split(ds, **kwargs)
    raise TypeError(f"prepare_dataset expects a Dataset or DatasetDict, got {type(ds).__name__}")


@dataclass
class InferenceData:
    """Bundle of everything needed to encode and evaluate one test chunk.

    Attributes
    ----------
    dataset : datasets.Dataset
        Anchor-ready dataset aligned to *adata* (encode ``dataset["anchor"]``).
    adata : anndata.AnnData
        AnnData chunk referenced by the dataset, subset to the dataset rows.
    vector_store : VectorStore or None
        Store attached to the model for the bimodal path; ``None`` for text.
    local_path : pathlib.Path
        Path to the downloaded/cached AnnData store.
    """

    dataset: Dataset
    adata: "ad.AnnData"
    vector_store: "VectorStore | None"
    local_path: Path


def prepare_inference(
    model: "SentenceTransformer",
    ds: Dataset,
    *,
    modality: Modality = "bimodal",
    obsm_key: str | None = None,
    cache_dir: str | Path,
    store_path: str | Path | None = None,
    sample_id_col: str = "sample_idx",
    adata_link_col: str = "adata_link",
    omics_prefix: str = "omics:",
    zenodo_token: str | None = None,
    truncate: bool = False,
    truncate_kwargs: dict | None = None,
    overwrite_store: bool = False,
) -> InferenceData:
    """Wire up a model + dataset + AnnData for inference on one test chunk.

    This replaces the manual sequence of loading the AnnData chunk, subsetting
    the dataset, building a VectorStore, and attaching it to the model. For the
    bimodal path it builds the store from *obsm_key* and calls
    ``model[0].set_vector_store(store)`` so ``model.encode(bundle.dataset["anchor"])``
    works directly.

    Parameters
    ----------
    model : sentence_transformers.SentenceTransformer
        Model whose first module is an
        :class:`~mmcontext.modules.MMContextModule` (only required for the
        bimodal path, where the VectorStore is attached to it).
    ds : datasets.Dataset
        Test split referencing a single AnnData chunk via *adata_link_col*.
    modality : {"bimodal", "text"}, default "bimodal"
        ``"bimodal"`` builds and attaches a VectorStore; ``"text"`` skips it.
    obsm_key : str, optional
        ``adata.obsm`` key to extract for the VectorStore (e.g. ``"X_scvi_fm"``).
        Required when ``modality="bimodal"``.
    cache_dir : str or Path
        Directory for caching the downloaded AnnData store (shared by the chunk
        loader and the VectorStore builder, which use the same cache key).
    store_path : str or Path, optional
        Output ``.mmap`` path for the VectorStore. Defaults to
        ``<cache_dir>/vector_store_inference.mmap``. Unused for text modality.
    sample_id_col : str, default "sample_idx"
        Column with sample identifiers.
    adata_link_col : str, default "adata_link"
        Column with the link to the AnnData chunk.
    omics_prefix : str, default "omics:"
        Prefix for omics anchors (must match the model's module).
    zenodo_token : str, optional
        Token for authenticating Zenodo draft downloads.
    truncate : bool, default False
        Forwarded to :func:`prepare_dataset` (text modality).
    truncate_kwargs : dict, optional
        Forwarded to :func:`prepare_dataset` (text modality).
    overwrite_store : bool, default False
        Rebuild the VectorStore even if *store_path* already exists.

    Returns
    -------
    InferenceData
        Bundle with the prepared dataset, the subset AnnData, the VectorStore
        (or ``None``), and the local store path.

    Examples
    --------
    >>> bundle = prepare_inference(model, test_ds, obsm_key="X_scvi_fm", cache_dir="./cache")
    >>> emb = model.encode(bundle.dataset["anchor"])
    >>> bundle.adata.obsm["mmcontext_emb"] = emb
    """
    # Imported here to keep module import light and avoid heavy/circular deps.
    from mmcontext.file_utils import load_test_adata_from_hf_dataset, subset_dataset_by_chunk

    if modality == "bimodal" and obsm_key is None:
        raise ValueError("obsm_key is required when modality='bimodal'.")

    # The text anchor source follows from the modality: bimodal anchors are
    # omics ids (cell_sentence unused), text anchors use the text description.
    primary_cell_sentence = "cell_sentence_1" if modality == "bimodal" else "cell_sentence_2"

    cache_dir = Path(cache_dir)

    # 1) Download + load the single AnnData chunk referenced by this split.
    adata, local_path = load_test_adata_from_hf_dataset(
        ds,
        save_dir=cache_dir,
        layer_key=obsm_key if modality == "bimodal" else None,
        link_column=adata_link_col,
        zenodo_token=zenodo_token,
    )

    # 2) Restrict the dataset to the rows present in this chunk.
    adata, ds_sub = subset_dataset_by_chunk(adata, ds, sample_idx_col=sample_id_col)

    # 3) Build the anchor-ready dataset (no positive/negative needed).
    dataset = prepare_dataset(
        ds_sub,
        purpose="inference",
        modality=modality,
        primary_cell_sentence=primary_cell_sentence,
        sample_id_col=sample_id_col,
        omics_prefix=omics_prefix,
        truncate=truncate,
        truncate_kwargs=truncate_kwargs,
    )

    # 4) For bimodal, build + attach the VectorStore so omics anchors resolve.
    vector_store: "VectorStore | None" = None
    if modality == "bimodal":
        from mmcontext.io import prepare_vector_store

        if store_path is None:
            store_path = cache_dir / "vector_store_inference.mmap"
        vector_store = prepare_vector_store(
            ds_sub,
            obsm_key=obsm_key,
            output_path=store_path,
            cache_dir=cache_dir,
            sample_id_column=sample_id_col,
            adata_link_column=adata_link_col,
            overwrite=overwrite_store,
        )
        model[0].set_vector_store(vector_store)

    return InferenceData(dataset=dataset, adata=adata, vector_store=vector_store, local_path=local_path)
