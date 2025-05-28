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

    Returns
    -------
    • datasets.Dataset   (for 'hub' or 'hf_disk')
    • pandas.DataFrame   (for 'csv')
    """
    source = Path(source).expanduser()
    if fmt == "hub":
        ds = load_dataset(str(source), split=split or "test", download_mode="force_redownload")
    elif fmt == "hf_disk":
        disk = load_from_disk(str(source))
        ds = disk[split] if split else disk
    elif fmt == "csv":
        if not source.is_file() or source.suffix != ".csv":
            raise FileNotFoundError(f"CSV file not found: {source}")
        ds_df = pd.read_csv(source)
        # convert the csv to HFDataset
        ds = Dataset.from_pandas(ds_df, split=split or "test")
    else:
        raise ValueError(f"Unknown format '{fmt}'")

    # ── optional sub-sampling ───────────────────────────────────────────
    if max_rows and len(ds) > max_rows:
        if isinstance(ds, pd.DataFrame):
            ds = ds.sample(n=max_rows, random_state=seed)
        else:  # Arrow
            ds = ds.shuffle(seed=seed).select(range(max_rows))
    logger.info("Loaded %s (%d rows)", fmt, len(ds))
    return ds


def collect_adata_subset(
    download_dir: str | Path,
    sample_ids: Sequence[str],
    *,
    join_vars: str = "outer",  # 'inner' if you need exact var overlap
) -> ad.AnnData:
    """
    Load a subset of the data from the download directory.

    Gather a subset of observations that may live in **multiple** Zarr/H5AD
    chunks into a single in-memory `AnnData`.

    Parameters
    ----------
    download_dir : str | Path
        Directory that contains files written by `get_initial_embeddings`
        (mix of `<dataset>.zarr/` folders and/or `<dataset>.h5ad` files).
    sample_ids : Sequence[str]
        The observation indices to collect.  Order will be preserved.
    join_vars : {'outer', 'inner'}, default 'outer'
        How to merge variables (genes) when concatenating chunks.

    Returns
    -------
    AnnData
        Combined object with `obs_names` matching the order of `sample_ids`.

    Notes
    -----
    * For now, all chunks are loaded into memory.
    """
    download_dir = Path(download_dir).expanduser()
    if not download_dir.exists():
        raise FileNotFoundError(download_dir)

    # keep a *mutable* set for O(1) look-up
    remaining = set(sample_ids)
    collected: list[ad.AnnData] = []

    # scan *.zarr folders first (usually many small shards) -------------
    for zarr_dir in sorted(download_dir.glob("*.zarr")):
        logger.debug("Scanning Zarr store %s", zarr_dir)
        view = ad.read_zarr(zarr_dir)
        wanted = remaining.intersection(view.obs_names)
        if wanted:
            logger.info("  ↳ %d obs found in %s", len(wanted), zarr_dir.name)
            collected.append(view[list(wanted)].copy())  # loads only slice
            remaining.difference_update(wanted)
        view.file.close()  # be nice to the OS

    # scan *.h5ad files -----------------------------------------------
    for h5_path in sorted(download_dir.glob("*.h5ad")):
        if not remaining:
            break
        logger.debug("Scanning H5AD file %s", h5_path.name)
        view = ad.read_h5ad(h5_path, backed="r")
        wanted = remaining.intersection(view.obs_names)
        view.file.close()

        if wanted:
            logger.info("  ↳ %d obs found in %s", len(wanted), h5_path.name)
            full = ad.read_h5ad(h5_path)  # loads entire file once
            collected.append(full[list(wanted)].copy())
            remaining.difference_update(wanted)

    if remaining:
        logger.warning(
            "collect_adata_subset | %d sample_ids were *not* found in %s: %s",
            len(remaining),
            download_dir,
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
