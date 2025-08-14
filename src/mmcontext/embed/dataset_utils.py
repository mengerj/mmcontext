# embedding_benchmark/model_utils.py
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anndata as ad
import pandas as pd
import torch
from datasets import Dataset, load_dataset, load_from_disk

logger = logging.getLogger(__name__)


@dataclass
class SentenceDataset(Dataset):
    """
    A minimal Dataset that returns (index, text) tuples needed for batching.

    Parameters
    ----------
    indices : List[int] | List[str]
        Unique sample identifiers.
    texts : List[str]
        Sentences or tokenised strings.
    """

    indices: list[int]
    texts: list[str]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx], self.texts[idx]


def load_generic_dataset(
    *,
    source: str | Path,
    fmt: Literal["hub", "hf_disk", "csv"],
    split: str | None,
    max_rows: int | None,
    seed: int,
) -> Dataset:
    """
    Load a dataset in Arrow (hub / on-disk) **or** CSV form.

    For hub datasets, uses streaming to efficiently load only the requested
    number of rows without downloading the entire dataset.

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
            ds_iter = load_dataset(str(source), split=split or "test", streaming=True)
            # Convert iterator to list with the requested number of items
            subset_list = list(islice(ds_iter, max_rows))
            ds = Dataset.from_list(subset_list)
            logger.info("Loaded %s via streaming (%d rows)", fmt, len(ds))
        else:
            # Load the full dataset if no max_rows specified
            ds = load_dataset(str(source), split=split or "test", download_mode="force_redownload")
            logger.info("Loaded %s (%d rows)", fmt, len(ds))
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


def collect_adata_subset(
    download_dir: str | Path | None = None,
    sample_ids: Sequence[str] | None = None,
    *,
    file_paths: list[Path] | None = None,
    join_vars: str = "outer",  # 'inner' if you need exact var overlap
) -> ad.AnnData:
    """
    Load a subset of the data from the download directory or from specific file paths.

    Gather a subset of observations that may live in **multiple** Zarr/H5AD
    chunks into a single in-memory `AnnData`.

    Parameters
    ----------
    download_dir : str | Path | None
        Directory that contains files written by `get_initial_embeddings`
        (mix of `<dataset>.zarr/` folders and/or `<dataset>.h5ad` files).
        If None, file_paths must be provided.
    sample_ids : Sequence[str] | None
        The observation indices to collect. Order will be preserved.
        If None, all observations from the files will be collected.
    file_paths : list[Path] | None
        List of specific file paths to read from. If provided, download_dir is ignored.
        Used when files are in their original locations (e.g., local datasets).
    join_vars : {'outer', 'inner'}, default 'outer'
        How to merge variables (genes) when concatenating chunks.

    Returns
    -------
    AnnData
        Combined object with `obs_names` matching the order of `sample_ids`.

    Notes
    -----
    * For now, all chunks are loaded into memory.
    * Either download_dir or file_paths must be provided.
    """
    if download_dir is None and file_paths is None:
        raise ValueError("Either download_dir or file_paths must be provided")

    if sample_ids is None:
        raise ValueError("sample_ids must be provided")

    if file_paths is not None:
        # Use specific file paths
        paths_to_scan = file_paths
    else:
        # Use directory scanning (existing behavior)
        download_dir = Path(download_dir).expanduser()
        if not download_dir.exists():
            raise FileNotFoundError(download_dir)
        paths_to_scan = list(download_dir.glob("*.zarr")) + list(download_dir.glob("*.h5ad"))

    # keep a *mutable* set for O(1) look-up
    remaining = set(sample_ids)
    collected: list[ad.AnnData] = []

    # scan zarr files/directories first (usually many small shards) -------------
    for path in paths_to_scan:
        if not remaining:
            break

        path = Path(path)
        if path.suffix == ".zarr" or (path.is_dir() and path.name.endswith(".zarr")):
            logger.debug("Scanning Zarr store %s", path)
            view = ad.read_zarr(path)
            wanted = remaining.intersection(view.obs_names)
            if wanted:
                logger.info("  ↳ %d obs found in %s", len(wanted), path.name)
                collected.append(view[list(wanted)].copy())  # loads only slice
                remaining.difference_update(wanted)
            if hasattr(view, "file") and view.file:
                view.file.close()  # be nice to the OS

    # scan *.h5ad files -----------------------------------------------
    for path in paths_to_scan:
        if not remaining:
            break

        path = Path(path)
        if path.suffix == ".h5ad":
            logger.debug("Scanning H5AD file %s", path.name)
            view = ad.read_h5ad(path, backed="r")
            wanted = remaining.intersection(view.obs_names)
            if hasattr(view, "file") and view.file:
                view.file.close()

            if wanted:
                logger.info("  ↳ %d obs found in %s", len(wanted), path.name)
                full = ad.read_h5ad(path)  # loads entire file once
                collected.append(full[list(wanted)].copy())
                remaining.difference_update(wanted)

    if remaining:
        logger.warning(
            "collect_adata_subset | %d sample_ids were *not* found in available files: %s",
            len(remaining),
            sorted(remaining)[:10],
        )

    if not collected:
        raise ValueError("None of the requested sample_ids were found in any chunk.")

    logger.info("Concatenating %d partial AnnData objects …", len(collected))
    adata = ad.concat(
        collected,
        axis=0,
        join=join_vars,
        merge="same",
        label=None,
        index_unique=None,
    )

    # restore original order  -----------------------------------------
    adata = adata[sample_ids].copy()
    return adata
