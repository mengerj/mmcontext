"""Collect a subset of AnnData observations from multiple Zarr/H5AD chunks.

Moved from ``embed.dataset_utils`` during v2.0 cleanup — it belongs with the
I/O utilities in ``mmcontext.io``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import anndata as ad
import numpy as np

from mmcontext.file_utils import remove_corrupted_null_arrays
from mmcontext.io._zarr_read import _open_zarr, _read_obs_names_zarr, read_obs_and_obsm

logger = logging.getLogger(__name__)


def collect_adata_subset(
    download_dir: str | Path | None = None,
    sample_ids: Sequence[str] | None = None,
    *,
    file_paths: list[Path] | None = None,
    join_vars: str = "outer",
    obsm_key: str | None = None,
    obs_columns: list[str] | None = None,
) -> ad.AnnData:
    """Load a subset of observations from multiple Zarr/H5AD chunks.

    Gather observations that may live in **multiple** Zarr/H5AD chunks into a
    single in-memory :class:`~anndata.AnnData`.

    Parameters
    ----------
    download_dir : str, Path, or None
        Directory containing ``*.zarr/`` folders and/or ``*.h5ad`` files.
        Ignored when *file_paths* is given.
    sample_ids : Sequence[str] or None
        Observation indices to collect. Order is preserved in the output.
    file_paths : list[Path] or None
        Explicit file paths to read from. Takes precedence over *download_dir*.
    join_vars : {"outer", "inner"}, default "outer"
        How to merge variables (genes) when concatenating chunks.
    obsm_key : str or None
        If given, use the memory-lean path: only the requested ``obsm`` layer
        and the ``obs`` table are read per chunk. The returned AnnData has no
        ``X`` (``n_vars == 0``). If ``None``, the full-read path is used.
    obs_columns : list[str] or None
        With *obsm_key*, restrict collected ``obs`` to these columns (plus
        the index). Ignored when *obsm_key* is ``None``.

    Returns
    -------
    AnnData
        Combined object with ``obs_names`` matching the order of *sample_ids*.
    """
    if download_dir is None and file_paths is None:
        raise ValueError("Either download_dir or file_paths must be provided")

    if sample_ids is None:
        raise ValueError("sample_ids must be provided")

    if file_paths is not None:
        paths_to_scan = file_paths
    else:
        download_dir = Path(download_dir).expanduser()
        if not download_dir.exists():
            raise FileNotFoundError(download_dir)
        paths_to_scan = list(download_dir.glob("*.zarr")) + list(download_dir.glob("*.h5ad"))

    remaining = set(sample_ids)
    collected: list[ad.AnnData] = []

    # Scan zarr stores first
    for path in paths_to_scan:
        if not remaining:
            break
        path = Path(path)
        if path.suffix == ".zarr" or (path.is_dir() and path.name.endswith(".zarr")):
            logger.debug("Scanning Zarr store %s", path)
            remove_corrupted_null_arrays(path)

            if obsm_key is not None:
                obs_names = _read_obs_names_zarr(_open_zarr(path))
                name_to_idx = {n: i for i, n in enumerate(obs_names)}
                wanted = remaining.intersection(obs_names)
                if wanted:
                    logger.info("  ↳ %d obs found in %s", len(wanted), path.name)
                    wanted_list = [n for n in obs_names if n in wanted]
                    rows = np.array([name_to_idx[n] for n in wanted_list])
                    obs_df, obsm = read_obs_and_obsm(path, obsm_key=obsm_key, rows=rows, obs_columns=obs_columns)
                    sub = ad.AnnData(obs=obs_df)
                    sub.obsm[obsm_key] = obsm
                    collected.append(sub)
                    remaining.difference_update(wanted)
            else:
                view = ad.read_zarr(path)
                wanted = remaining.intersection(view.obs_names)
                if wanted:
                    logger.info("  ↳ %d obs found in %s", len(wanted), path.name)
                    collected.append(view[list(wanted)].copy())
                    remaining.difference_update(wanted)
                if hasattr(view, "file") and view.file:
                    view.file.close()

    # Scan *.h5ad files
    for path in paths_to_scan:
        if not remaining:
            break
        path = Path(path)
        if path.suffix == ".h5ad":
            logger.debug("Scanning H5AD file %s", path.name)
            view = ad.read_h5ad(path, backed="r")
            obs_names = list(view.obs_names)
            wanted = remaining.intersection(obs_names)
            if hasattr(view, "file") and view.file:
                view.file.close()

            if wanted:
                logger.info("  ↳ %d obs found in %s", len(wanted), path.name)
                if obsm_key is not None:
                    name_to_idx = {n: i for i, n in enumerate(obs_names)}
                    wanted_list = [n for n in obs_names if n in wanted]
                    rows = np.array([name_to_idx[n] for n in wanted_list])
                    obs_df, obsm = read_obs_and_obsm(path, obsm_key=obsm_key, rows=rows, obs_columns=obs_columns)
                    sub = ad.AnnData(obs=obs_df)
                    sub.obsm[obsm_key] = obsm
                    collected.append(sub)
                else:
                    full = ad.read_h5ad(path)
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

    # Restore original order
    adata = adata[sample_ids].copy()
    return adata
