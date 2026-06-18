"""Low-level zarr/h5ad readers that load only what a caller needs.

The functions here open an AnnData-formatted store *without* materialising the
full :class:`~anndata.AnnData` object. For zarr stores this means reading only
the ``obs`` group (cheap: ``N`` rows × a few columns), the requested ``obsm``
layer, and — optionally — only a subset of rows of that layer. The large ``X``
matrix and any unrequested ``obsm`` layers are never touched.

This module deliberately keeps its top-level imports light (``numpy`` and
``zarr`` only). ``anndata`` is imported lazily inside the functions that need it
so that ``import mmcontext`` stays cheap.

The zarr-handle helpers (:func:`_open_zarr`, :func:`_read_obs_names_zarr`,
:func:`_get_obsm_zarr_array`) are the single source of truth reused by
:mod:`mmcontext.io.prepare_store`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import zarr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Zarr handle helpers — open a store and read obs_names / obsm lazily
# ---------------------------------------------------------------------------


def _open_zarr(path: Path) -> zarr.Group:
    """Open a zarr store from a directory or zip file (read-only)."""
    path = Path(path)
    if path.suffix == ".zip" and path.is_file():
        return zarr.open(str(path), mode="r")
    elif path.is_dir():
        # Could be a zarr directory, or a directory containing a single zarr.
        # v3 stores carry ``zarr.json``; v2 stores carry ``.zgroup``/``.zattrs``.
        if (path / "zarr.json").exists() or (path / ".zgroup").exists() or (path / ".zattrs").exists():
            return zarr.open(str(path), mode="r")
        # Check for a single subdirectory that is the actual zarr
        subdirs = [p for p in path.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            return zarr.open(str(subdirs[0]), mode="r")
        raise ValueError(f"Cannot determine zarr root in {path}")
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")


def _read_obs_names_zarr(root: zarr.Group) -> list[str]:
    """Read observation names from a zarr-backed AnnData store.

    AnnData zarr layout stores the obs index under ``obs/_index`` (recent
    anndata) or ``obs/index`` (older convention). In both cases the values
    may be stored directly or via ``__categories``.
    """
    obs = root["obs"]

    # Determine the index key name
    if "_index" in obs.attrs:
        idx_key = obs.attrs["_index"]
    elif "__index_level_0__" in obs:
        idx_key = "__index_level_0__"
    else:
        idx_key = "_index"

    idx_elem = obs[idx_key]

    # anndata ≥ 0.11 with zarr 3.x stores array elements as zarr Groups
    # (encoded format with encoding-type metadata). Use read_elem which handles
    # both old (plain zarr Array) and new (encoded zarr Group) layouts.
    if isinstance(idx_elem, zarr.Group):
        index_data = _read_elem(idx_elem)
        return [str(v) for v in index_data]

    # Legacy path: plain zarr Array
    # Handle categorical encoding (older anndata versions)
    if hasattr(idx_elem, "attrs") and "categories" in idx_elem.attrs:
        cat_path = idx_elem.attrs["categories"]
        codes = idx_elem[:]
        categories = root[cat_path][:]
        return [categories[c].decode() if isinstance(categories[c], bytes) else str(categories[c]) for c in codes]

    values = idx_elem[:]
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def _get_obsm_zarr_array(root: zarr.Group, obsm_key: str) -> zarr.Array:
    """Return a zarr Array handle for an obsm layer (no data read yet).

    Callers can use ``.get_orthogonal_selection()`` or ``[rows]`` to read
    only the rows they need, so the full matrix is never in memory.
    """
    if "obsm" not in root:
        raise KeyError(f"No 'obsm' group in zarr store. Available: {list(root.keys())}")

    obsm = root["obsm"]
    if obsm_key not in obsm:
        raise KeyError(f"Key '{obsm_key}' not in obsm. Available: {list(obsm.keys())}")

    return obsm[obsm_key]


# ---------------------------------------------------------------------------
# Row-selection helper
# ---------------------------------------------------------------------------


def _select_rows_float32(arr: zarr.Array, rows: np.ndarray | None) -> np.ndarray:
    """Read ``rows`` of a zarr array as ``float32``, or all rows if None.

    Row indices are sorted before the read for sequential disk access and the
    result is unsorted back into the caller's order, mirroring the strategy in
    :func:`mmcontext.io.prepare_store.prepare_vector_store`.
    """
    if rows is None:
        return arr[:].astype(np.float32, copy=False)

    rows = np.asarray(rows)
    if rows.size == 0:
        return np.empty((0, arr.shape[1]), dtype=np.float32)

    sort_order = np.argsort(rows)
    sorted_rows = rows[sort_order]
    # Use oindex for orthogonal row selection — compatible with zarr 2.x and 3.x
    # (zarr 3.x deprecated get_orthogonal_selection in favour of arr.oindex).
    selected = arr.oindex[sorted_rows, :].astype(np.float32, copy=False)
    # Unsort back to the caller's original row order
    unsort = np.argsort(sort_order)
    return selected[unsort]


# ---------------------------------------------------------------------------
# High-level lazy reader
# ---------------------------------------------------------------------------


def read_obs_and_obsm(
    path: str | Path,
    *,
    obsm_key: str | None = None,
    rows: np.ndarray | list[int] | None = None,
    obs_columns: list[str] | None = None,
):
    """Read only the obs table and (optionally) one obsm layer from a store.

    Parameters
    ----------
    path
        Path to a zarr store (directory or ``.zip``) or an ``.h5ad`` file.
    obsm_key
        Name of the single ``obsm`` layer to read. If ``None`` no obsm data
        is read and the returned array is ``None``.
    rows
        Integer row indices to read (in the order they should be returned). If
        ``None`` all rows are read. The returned obs table is sliced to the
        same rows/order.
    obs_columns
        Subset of ``obs`` columns to keep. Columns not present in the store are
        silently ignored. If ``None`` all obs columns are kept. The obs index
        is always retained.

    Returns
    -------
    obs : pandas.DataFrame
        The (optionally row- and column-subset) obs table, index preserved.
    obsm : numpy.ndarray | None
        ``float32`` array of shape ``(len(rows), D)`` for the requested layer,
        or ``None`` if *obsm_key* was ``None``.

    Notes
    -----
    For zarr stores the full ``X`` matrix and any unrequested ``obsm`` layers
    are never read. ``obs`` is read in full (it is small) and then subset. For
    ``.h5ad`` files the store is opened with ``backed="r"`` so only the needed
    elements are pulled into memory.
    """
    path = Path(path)
    is_h5ad = path.suffix == ".h5ad"
    if is_h5ad:
        return _read_obs_and_obsm_h5ad(path, obsm_key=obsm_key, rows=rows, obs_columns=obs_columns)
    return _read_obs_and_obsm_zarr(path, obsm_key=obsm_key, rows=rows, obs_columns=obs_columns)


def _read_elem(elem):
    """Import-light wrapper around anndata's ``read_elem``."""
    try:
        from anndata.io import read_elem
    except ImportError:  # pragma: no cover - older anndata layout
        from anndata.experimental import read_elem
    return read_elem(elem)


def _subset_obs(obs, rows, obs_columns):
    """Apply column then row subsetting to an obs DataFrame."""
    if obs_columns is not None:
        keep = [c for c in obs_columns if c in obs.columns]
        obs = obs[keep]
    if rows is not None:
        obs = obs.iloc[np.asarray(rows)]
    return obs


def _read_obs_and_obsm_zarr(path, *, obsm_key, rows, obs_columns):
    root = _open_zarr(path)
    obs = _read_elem(root["obs"])  # full obs table — cheap (N × few cols)
    obs = _subset_obs(obs, rows, obs_columns)

    obsm = None
    if obsm_key is not None:
        arr = _get_obsm_zarr_array(root, obsm_key)
        obsm = _select_rows_float32(arr, None if rows is None else np.asarray(rows))
    return obs, obsm


def _read_obs_and_obsm_h5ad(path, *, obsm_key, rows, obs_columns):
    import anndata as ad

    # backed="r" keeps X on disk; obs/obsm are read into memory (small relative
    # to X). We then slice to the requested rows/columns.
    adata = ad.read_h5ad(path, backed="r")
    try:
        obs = adata.obs
        obs = _subset_obs(obs, rows, obs_columns)
        obsm = None
        if obsm_key is not None:
            if obsm_key not in adata.obsm:
                raise KeyError(f"Key '{obsm_key}' not in obsm. Available: {list(adata.obsm.keys())}")
            layer = np.asarray(adata.obsm[obsm_key])
            if rows is not None:
                layer = layer[np.asarray(rows)]
            obsm = layer.astype(np.float32, copy=False)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
    return obs, obsm
