"""Build a VectorStore from adata_link URLs in a HuggingFace dataset.

The key design constraint is **memory efficiency**: zarr files are opened
directly (without loading the full AnnData) so that only the requested
``obsm`` layer and the obs index are read.  This makes it feasible to
prepare stores from large datasets on a 16 GB laptop.

Typical usage::

    from datasets import load_dataset
    from mmcontext.io import prepare_vector_store

    ds = load_dataset("jo-mengr/cxg_schaefer_tiny", split="train")
    store = prepare_vector_store(
        ds,
        obsm_key="X_scvi_fm",
        output_path="data/store.mmap",
        cache_dir="data/zarr_cache",
    )
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile, is_zipfile

import numpy as np
import requests
import zarr
from tqdm.auto import tqdm

from .vector_store import VectorStore

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Zarr helpers — read obs_names and obsm without loading full AnnData
# ---------------------------------------------------------------------------


def _read_obs_names_zarr(root: zarr.Group) -> list[str]:
    """Read observation names from a zarr-backed AnnData store.

    AnnData zarr layout stores the obs index under ``obs/_index`` (recent
    anndata) or ``obs/index`` (older convention).  In both cases the values
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

    idx_array = obs[idx_key]

    # Handle categorical encoding (older anndata versions)
    if hasattr(idx_array, "attrs") and "categories" in idx_array.attrs:
        cat_path = idx_array.attrs["categories"]
        codes = idx_array[:]
        categories = root[cat_path][:]
        return [categories[c].decode() if isinstance(categories[c], bytes) else str(categories[c]) for c in codes]

    values = idx_array[:]
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
# Download helper
# ---------------------------------------------------------------------------


def _url_to_cache_name(url: str) -> str:
    """Deterministic short name for a URL, used as cache directory name."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _download_zarr(url: str, cache_dir: Path) -> Path:
    """Download a zarr archive (zip or directory) with caching.

    Returns the local path to the zarr store (directory or zip).
    """
    cache_name = _url_to_cache_name(url)

    # Check for already-extracted zarr directory
    zarr_dir = cache_dir / f"{cache_name}.zarr"
    if zarr_dir.is_dir():
        logger.debug("Cache hit (zarr dir): %s", zarr_dir)
        return zarr_dir

    # Check for already-downloaded zip
    zip_path = cache_dir / f"{cache_name}.zip"
    if zip_path.is_file():
        logger.debug("Cache hit (zip): %s", zip_path)
        return _maybe_extract_zip(zip_path, zarr_dir)

    # Download
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Zenodo draft→published URL fixup: published records use the non-draft
    # API path, but datasets often store the draft URL from upload time.
    download_url = url
    if "zenodo.org" in download_url and "/draft/" in download_url:
        download_url = download_url.replace("/draft/", "/")
        logger.info("Converted Zenodo draft URL to published: %s", download_url)

    logger.info("Downloading %s → %s", download_url, zip_path)

    headers = {}
    if "zenodo.org" in download_url:
        headers["User-Agent"] = "Mozilla/5.0 (compatible; mmcontext/1.0)"

    with requests.get(download_url, stream=True, timeout=(30, 600), headers=headers) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(zip_path, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc="Downloading", leave=False) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))

    return _maybe_extract_zip(zip_path, zarr_dir)


def _maybe_extract_zip(zip_path: Path, zarr_dir: Path) -> Path:
    """If *zip_path* is a zip, extract to *zarr_dir* and return it.

    If it's already a plain zarr directory (mis-named), just rename.
    """
    if is_zipfile(zip_path):
        logger.info("Extracting %s → %s", zip_path, zarr_dir)
        with ZipFile(zip_path) as zf:
            zf.extractall(zarr_dir)
        # Optionally delete zip to save space
        # zip_path.unlink()
        return zarr_dir

    # Not a zip — the "download" was already a zarr directory or similar
    zip_path.rename(zarr_dir)
    return zarr_dir


def _open_zarr(path: Path) -> zarr.Group:
    """Open a zarr store from a directory or zip file."""
    if path.suffix == ".zip" and path.is_file():
        return zarr.open(str(path), mode="r")
    elif path.is_dir():
        # Could be a zarr directory, or a directory containing a single zarr
        if (path / ".zgroup").exists() or (path / ".zattrs").exists():
            return zarr.open(str(path), mode="r")
        # Check for a single subdirectory that is the actual zarr
        subdirs = [p for p in path.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            return zarr.open(str(subdirs[0]), mode="r")
        raise ValueError(f"Cannot determine zarr root in {path}")
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_vector_store(
    dataset: Dataset,
    *,
    obsm_key: str,
    output_path: str | Path,
    cache_dir: str | Path | None = None,
    sample_id_column: str = "sample_idx",
    adata_link_column: str = "adata_link",
    overwrite: bool = False,
) -> VectorStore:
    """Build a VectorStore from adata_link URLs in a HuggingFace dataset.

    For each unique ``adata_link`` in the dataset, the zarr file is downloaded
    (with local caching), and only the ``obsm[obsm_key]`` layer plus the obs
    index are read — the full AnnData is **never** loaded into memory.

    The resulting VectorStore maps ``sample_id_column`` values to their
    embedding vectors and is written as a memory-mapped file at *output_path*.

    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace dataset with at least *sample_id_column* and
        *adata_link_column* columns.
    obsm_key : str
        Key in ``adata.obsm`` to extract (e.g. ``"X_scvi_fm"``,
        ``"X_pca"``).
    output_path : str or Path
        Where to write the ``.mmap`` file (a sidecar ``.index.json``
        is created alongside it).
    cache_dir : str, Path, or None
        Directory for caching downloaded zarr files. Defaults to a
        ``mmcontext_zarr_cache`` folder inside the system temp directory.
    sample_id_column : str
        Column in *dataset* containing sample IDs (must match
        ``adata.obs_names``).
    adata_link_column : str
        Column in *dataset* containing URLs to zarr-backed AnnData files.
    overwrite : bool
        If False and *output_path* already exists, load and return the
        existing store.

    Returns
    -------
    VectorStore
        Memory-mapped store ready to attach to an MMContextModule.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        logger.info("VectorStore already exists at %s, loading.", output_path)
        return VectorStore.load(output_path)

    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "mmcontext_zarr_cache"
    cache_dir = Path(cache_dir)

    # Collect unique adata links and which sample IDs come from each
    link_to_sample_ids: dict[str, list[str]] = {}
    for row in dataset:
        link = row[adata_link_column]
        sid = str(row[sample_id_column])
        link_to_sample_ids.setdefault(link, []).append(sid)

    logger.info(
        "Preparing VectorStore: %d samples across %d adata files, obsm_key=%r",
        sum(len(v) for v in link_to_sample_ids.values()),
        len(link_to_sample_ids),
        obsm_key,
    )

    # Process each adata file and collect (id, vector) pairs
    all_ids: list[str] = []
    all_vectors: list[np.ndarray] = []

    for link, sample_ids in tqdm(link_to_sample_ids.items(), desc="Processing adata files"):
        # Download / cache
        if link.startswith(("http://", "https://")):
            local_path = _download_zarr(link, cache_dir)
        else:
            local_path = Path(link)

        # Open zarr and read only what we need
        root = _open_zarr(local_path)
        obs_names = _read_obs_names_zarr(root)
        obs_name_to_idx = {name: i for i, name in enumerate(obs_names)}

        # Resolve which rows we need
        needed_rows = []
        needed_ids = []
        for sid in sample_ids:
            if sid not in obs_name_to_idx:
                raise KeyError(
                    f"Sample ID '{sid}' not found in obs_names of {link}. First 5 obs_names: {obs_names[:5]}"
                )
            needed_rows.append(obs_name_to_idx[sid])
            needed_ids.append(sid)

        # Read only the needed rows from obsm — zarr chunks are paged on
        # demand so we never materialise the full (N_total, D) matrix.
        # Sort row indices for sequential disk access, then unsort.
        sort_order = np.argsort(needed_rows)
        sorted_rows = np.array(needed_rows)[sort_order]

        obsm_array = _get_obsm_zarr_array(root, obsm_key)
        selected = obsm_array.get_orthogonal_selection((sorted_rows, slice(None))).astype(
            np.float32
        )  # (len(needed_rows), D)

        # Unsort back to original order
        unsort = np.argsort(sort_order)
        selected = selected[unsort]

        all_ids.extend(needed_ids)
        all_vectors.append(selected)

    # Combine and build VectorStore
    matrix = np.concatenate(all_vectors, axis=0)  # (N_total, D)
    logger.info("Building VectorStore: %d vectors, dim=%d", matrix.shape[0], matrix.shape[1])

    store = VectorStore.from_numpy(matrix, all_ids, path=output_path)
    logger.info("VectorStore saved to %s", output_path)
    return store


def build_namespaced_vector_store(
    per_dataset: dict[str, tuple[VectorStore, Sequence[str]]],
    *,
    output_path: str | Path,
    namespace_sep: str = ":",
) -> VectorStore:
    """Merge several per-dataset stores into one with namespaced ids.

    Multiple omics datasets typically number their samples independently
    (``sample_idx`` "0", "1", … restarts in each dataset), so concatenating
    their stores directly would collide. A single mmcontext pipeline holds
    exactly one :class:`VectorStore`, so the per-dataset stores must be merged.
    This helper re-keys every vector to ``{dataset_name}{sep}{sample_idx}`` and
    writes one combined store.

    The same namespacing must be applied to the dataset's ``sample_idx`` column
    *before* building anchors, so the resulting ``omics:{name}{sep}{idx}``
    anchors resolve against these keys. Build each per-dataset store on the
    **original** ids first (so the lookup against ``adata.obs_names`` in
    :func:`prepare_vector_store` succeeds), then pass it here.

    Parameters
    ----------
    per_dataset : dict[str, tuple[VectorStore, Sequence[str]]]
        Maps a dataset name to ``(store, original_sample_ids)``. The ids are
        looked up in *store* via :meth:`VectorStore.batch_lookup`; duplicates
        are de-duplicated while preserving order.
    output_path : str or Path
        Where to write the merged ``.mmap`` file (a sidecar ``.index.json`` is
        written alongside it).
    namespace_sep : str, default ":"
        Separator between the dataset name and the original id.

    Returns
    -------
    VectorStore
        The merged store, keyed by ``{name}{sep}{orig_id}``.

    Raises
    ------
    ValueError
        If *per_dataset* is empty or the stores disagree on dimensionality.
    """
    if not per_dataset:
        raise ValueError("per_dataset is empty: at least one store is required.")

    all_ids: list[str] = []
    all_vectors: list[np.ndarray] = []
    dim: int | None = None

    for name, (store, orig_ids) in per_dataset.items():
        # Unique, order-preserving so rows line up with the namespaced ids.
        unique_ids = list(dict.fromkeys(str(i) for i in orig_ids))
        if not unique_ids:
            continue
        vectors = store.batch_lookup(unique_ids)  # (N, D)
        if dim is None:
            dim = vectors.shape[1]
        elif vectors.shape[1] != dim:
            raise ValueError(
                f"Store '{name}' has dim {vectors.shape[1]} but a previous store has dim {dim}. "
                "All omics datasets in one run must share the same obsm_key / dimensionality."
            )
        all_vectors.append(vectors)
        all_ids.extend(f"{name}{namespace_sep}{sid}" for sid in unique_ids)

    matrix = np.concatenate(all_vectors, axis=0)
    logger.info(
        "Merged %d namespaced VectorStores: %d vectors, dim=%d",
        len(per_dataset),
        matrix.shape[0],
        matrix.shape[1],
    )
    return VectorStore.from_numpy(matrix, all_ids, path=output_path)
