"""Dataset loading helpers for the embed/inference path.

Holds :func:`load_generic_dataset` (HF hub / on-disk / CSV loader) and
re-exports :func:`~mmcontext.io.collect_adata_subset` (which now lives in
:mod:`mmcontext.io`) so downstream consumers can keep importing both from this
module, as ``mmcontext-benchmark`` does.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk

# Re-export: collect_adata_subset moved to mmcontext.io during the v2.0 refactor;
# keep the historical import path working for downstream consumers.
from mmcontext.io import collect_adata_subset

logger = logging.getLogger(__name__)

__all__ = ["collect_adata_subset", "load_generic_dataset"]


def load_generic_dataset(
    *,
    source: str | Path,
    fmt: Literal["hub", "hf_disk", "csv"],
    split: str | None,
    max_rows: int | None,
    seed: int,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
) -> Dataset:
    """
    Load a dataset in Arrow (hub / on-disk) **or** CSV form.

    For hub datasets, uses streaming to efficiently load only the requested
    number of rows without downloading the entire dataset.

    Parameters
    ----------
    source : str | Path
        Dataset source (HF hub name, local path, etc.)
    fmt : Literal["hub", "hf_disk", "csv"]
        Dataset format
    split : str | None
        Dataset split to load
    max_rows : int | None
        Maximum number of rows to load
    seed : int
        Random seed for shuffling
    cache_dir : str | Path | None
        Cache directory for HF datasets
    revision : str | None
        Specific revision/branch to load (only for hub datasets)

    Returns
    -------
    datasets.Dataset
        Always returns a datasets.Dataset (CSV is converted from pandas)
    """
    from itertools import islice

    source = Path(source).expanduser()

    if fmt == "hub":
        if max_rows is not None:
            # Use streaming to load only the requested number of rows
            ds_iter = load_dataset(
                str(source),
                split=split or "test",
                streaming=True,
                cache_dir=cache_dir,
                revision=revision,
                download_mode="force_redownload",
            )
            # Convert iterator to list with the requested number of items
            subset_list = list(islice(ds_iter, max_rows))
            ds = Dataset.from_list(subset_list)
            revision_info = f" (revision: {revision})" if revision else ""
            logger.info("Loaded %s via streaming (%d rows)%s", fmt, len(ds), revision_info)
        else:
            # Load the full dataset if no max_rows specified
            ds = load_dataset(
                str(source),
                split=split or "test",
                download_mode="force_redownload",
                cache_dir=cache_dir,
                revision=revision,
            )
            revision_info = f" (revision: {revision})" if revision else ""
            logger.info("Loaded %s (%d rows)%s. Data stored in %s", fmt, len(ds), revision_info, cache_dir)
    elif fmt == "hf_disk":
        disk = load_from_disk(str(source))
        ds = disk[split] if split else disk

        # Apply sub-sampling if requested
        if max_rows and len(ds) > max_rows:
            ds = ds.shuffle(seed=seed).select(range(max_rows))
        logger.info("Loaded %s (%d rows)", fmt, len(ds))
    elif fmt == "csv":
        if not source.is_file() or source.suffix != ".csv":
            raise FileNotFoundError(f"CSV file not found: {source}")

        if max_rows is not None:
            # For CSV, read only the requested number of rows + header
            ds_df = pd.read_csv(source, nrows=max_rows)
        else:
            ds_df = pd.read_csv(source)

        # Convert the csv to HFDataset
        ds = Dataset.from_pandas(ds_df, split=split or "test")
        logger.info("Loaded %s (%d rows)", fmt, len(ds))
    else:
        raise ValueError(f"Unknown format '{fmt}'")

    return ds
