# tests/utils.py
import json
import logging
import os
import random
import shutil
import tempfile
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Literal
from uuid import uuid4

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
import torch
import torch.nn as nn
import zarr
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def load_test_adata_from_hf_dataset(
    test_split,
    save_dir: str | Path,
    *,
    layer_key: str | None = None,
    axis: str = "obs",
) -> tuple[ad.AnnData, Path]:
    """
    Download the unique AnnData chunk referenced in *test_split* and load it.

    Parameters
    ----------
    test_split
        A *split* of a HF DatasetDict (`ds_dict["test"]` or similar)
        containing a ``"share_link"`` column that points to **one** file.
    layer_key
        Optional key to verify exists in ``.obsm`` or ``.varm``.
        If provided and missing, raises ``KeyError``.
    axis
        ``"obs"`` or ``"var"`` – used only for the optional *layer_key* check.

    Returns
    -------
    adata : AnnData
        The loaded AnnData (in memory, not backed).
    local_path : pathlib.Path
        Path to the downloaded store (`*.zarr`, `*.h5ad`, or `*.zip`).

    Raises
    ------
    ValueError
        If the split references multiple different files.
    """
    # 1) ensure there is exactly ONE unique link
    links, _ = collect_unique_links({"test": test_split}, split="test")
    # pick the first one
    link = links[0]
    logger.info("Picked share-link %s (out of %d)", link, len(links))

    # 2) download (extract_zip=False – keep archive if it's a ZIP)
    local_map = download_and_extract_links(
        links=[link],
        target_dir=save_dir,
        extract=True,
        overwrite=False,
    )
    local_path = next(iter(local_map.values()))

    # 3) open store
    if local_path.suffix == ".h5ad":
        adata = ad.read_h5ad(local_path)
    elif local_path.suffix == ".zip":
        import zarr

        zroot = zarr.open(local_path)
        adata = ad.read_zarr(zroot)
    else:  # .zarr dir
        adata = ad.read_zarr(str(local_path))

    # 4) optional sanity-check layer_key
    if layer_key is not None:
        has_layer = layer_key in adata.obsm if axis == "obs" else layer_key in adata.varm
        if not has_layer:
            raise KeyError(f"Layer '{layer_key}' not in adata.{axis + 'm'} for file {local_path}")

    logger.info("Loaded test AnnData with %d cells × %d genes", adata.n_obs, adata.n_vars)
    return adata, local_path


def subset_dataset_by_chunk(
    adata: ad.AnnData,
    full_dataset: Dataset,
    *,
    index_axis: str = "obs",
    sample_idx_col: str = "sample_idx",
) -> tuple[ad.AnnData, Dataset]:
    """
    Subset *full_dataset* to the rows that belong to *adata*.

    Parameters
    ----------
    adata
        AnnData previously loaded from one chunk.
    full_dataset
        Hugging Face Dataset (one split) that contains a *sample_idx* column.
    index_axis
        ``"obs"`` → compare against ``adata.obs.index``
        ``"var"`` → compare against ``adata.var.index``.
    sample_idx_col
        Column in the HF dataset that stores the IDs.

    Returns
    -------
    adata
        Same AnnData (unchanged, returned for convenience).
    subset_ds
        A *new* Hugging Face ``Dataset`` containing only the matching rows.
    """
    ids_in_chunk = adata.obs.index.to_numpy() if index_axis == "obs" else adata.var.index.to_numpy()
    id_set = set(ids_in_chunk)

    # boolean mask over HF dataset
    mask = [sid in id_set for sid in full_dataset[sample_idx_col]]
    subset_ds = full_dataset.filter(lambda _, idx: mask[idx], with_indices=True)

    logger.info(
        "Subset HF dataset: kept %d / %d rows",
        subset_ds.num_rows,
        full_dataset.num_rows,
    )
    return adata, subset_ds


def download_file_from_share_link(share_link, save_path, chunk_size=8192):
    """
    Downloads a file from a Nextcloud share link and validates it based on its suffix.

    Parameters
    ----------
    share_link : str
        The full share link URL to the file.
    save_path : str
        The local path where the file should be saved.
    chunk_size : int, optional
        Size of each chunk in bytes during streaming; defaults to 8192.

    Returns
    -------
    bool
        True if the download was successful and the file is valid based on its suffix;
        False otherwise.

    References
    ----------
    Data is expected to come from a Nextcloud share link and is validated in memory.
    """
    # Step 1: Stream download the file
    try:
        with requests.get(share_link, stream=True) as response:
            response.raise_for_status()

            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    file.write(chunk)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download the file from '{share_link}': {e}")
        return False

    # Step 2: Validate based on suffix
    file_suffix = os.path.splitext(save_path)[1].lower()

    try:
        if file_suffix == ".h5ad":
            # Validate as an anndata-compatible HDF5 file
            with h5py.File(save_path, "r") as h5_file:
                required_keys = ["X", "obs", "var"]  # Common in .h5ad
                if all(key in h5_file for key in required_keys):
                    logger.info("File is a valid .h5ad file.")
                    return True
                else:
                    logger.warning("File is an HDF5 file but missing required .h5ad keys.")
                    return False

        elif file_suffix == ".npz":
            # Validate as a .npz file (we can at least confirm we can load it)
            try:
                np.load(save_path, allow_pickle=True)
                logger.info("File is a valid .npz file.")
                return True
            except Exception as e:
                logger.error(f"Error while validating the downloaded file: {e}")
                return False

        elif file_suffix == ".npy":
            # Validate as a .npy file
            try:
                np.load(save_path, allow_pickle=True)
                logger.info("File is a valid .npy file.")
                return True
            except Exception as e:
                logger.error(f"Error while validating the downloaded file: {e}")
                return False
        else:
            # If your use-case requires more file types, add them here
            logger.warning(f"No specific validation logic for files of type '{file_suffix}'. Skipping validation.")
            return True

    except Exception as e:
        logger.error(f"Error while validating the downloaded file: {e}")
        return False


def create_test_anndata(n_samples=20, n_features=100, cell_types=None, tissues=None, batch_categories=None):
    """
    Create a test AnnData object with synthetic data, including batch information.

    Parameters
    ----------
    n_samples : int
        Number of cells (observations). Default is 20.
    n_features : int
        Number of genes (variables). Default is 100.
    cell_types : list, optional
        List of cell types. Defaults to ["B cell", "T cell", "NK cell"].
    tissues : list, optional
        List of tissues. Defaults to ["blood", "lymph"].
    batch_categories : list, optional
        List of batch categories. Defaults to ["Batch1", "Batch2"].

    Returns
    -------
    anndata.AnnData
        Generated AnnData object.
    """
    import numpy as np

    # Set default values for mutable arguments if they are None
    if cell_types is None:
        cell_types = ["B cell", "T cell", "NK cell"]
    if tissues is None:
        tissues = ["blood", "lymph"]
    if batch_categories is None:
        batch_categories = ["Batch1", "Batch2"]

    # Set a random seed for reproducibility
    np.random.seed(42)

    # Determine the number of batches and allocate samples to batches
    n_batches = len(batch_categories)
    samples_per_batch = n_samples // n_batches
    remainder = n_samples % n_batches

    batch_labels = []
    for i, batch in enumerate(batch_categories):
        n = samples_per_batch + (1 if i < remainder else 0)
        batch_labels.extend([batch] * n)

    # Shuffle batch labels
    np.random.shuffle(batch_labels)

    # Generate observation (cell) metadata

    obs = pd.DataFrame(
        {
            "cell_type": np.random.choice(cell_types, n_samples),
            "tissue": np.random.choice(tissues, n_samples),
            "batch": batch_labels,
        }
    )
    obs.index = [f"Cell_{i}" for i in range(n_samples)]
    # transform obs to categorical
    obs = obs.astype("category")
    obs["sample_id"] = np.arange(n_samples)
    # Generate a random data matrix (e.g., gene expression values)
    X = np.zeros((n_samples, n_features))
    for i, batch in enumerate(batch_categories):
        # Get indices of cells in this batch
        idx = obs[obs["batch"] == batch].index
        idx = [obs.index.get_loc(i) for i in idx]

        # Generate data for this batch
        # For simplicity, let's make a mean shift between batches
        mean = np.random.rand(n_features) * (i + 1)  # Different mean for each batch
        X[idx, :] = np.random.normal(loc=mean, scale=1.0, size=(len(idx), n_features))

    # Create variable (gene) metadata
    var = pd.DataFrame({"gene_symbols": [f"Gene_{i}" for i in range(n_features)]})
    var.index = [f"Gene_{i}" for i in range(n_features)]

    # Create the ad object
    adata = ad.AnnData(X=X, obs=obs, var=var)

    return adata


def create_test_emb_anndata(n_samples, emb_dim, data_key="d_emb_aligned", context_key="c_emb_aligned", sample_ids=None):
    """
    Helper function to create a test AnnData object with specified embeddings and sample IDs.

    Args:
        n_samples (int): Number of samples (cells).
        emb_dim (int): Embedding dimension.
        data_key (str): Key for data embeddings in adata.obsm.
        context_key (str): Key for context embeddings in adata.obsm.
        sample_ids (list): List of sample IDs. If None, default IDs are assigned.

    Returns
    -------
        AnnData: The constructed AnnData object.
    """
    adata = create_test_anndata(n_samples=n_samples)
    adata.obsm[data_key] = np.random.rand(n_samples, emb_dim)
    adata.obsm[context_key] = np.random.rand(n_samples, emb_dim)
    if sample_ids is not None:
        adata.obs["sample_id"] = sample_ids
    return adata


PK_MAGIC = b"PK\x03\x04"  # zip / npz
H5_MAGIC = b"\x89HDF\r\n\x1a\n"  # .h5/.h5ad
ZARR_MAGIC = b"{"  # first byte of .zmetadata  (fallback)


# ---------------------------------------------------------------------
# 1) gather the links once
# ---------------------------------------------------------------------
def collect_unique_links(
    ds_dict: Dataset | DatasetDict,
    split: str | None = None,
    link_column: str = "share_link",
) -> tuple[list[str], "pd.DataFrame"]:
    """
    Collect all unique share links from a DatasetDict.

    Parameters
    ----------
    ds_dict
        Hugging Face DatasetDict that contains one or more splits.
    split
        Specific split to inspect (e.g. "train").
        If *None*, walk every split in *ds_dict*.
    link_column
        Column name that stores the share link.

    Returns
    -------
    links : list[str]
        Unique links in deterministic order.
    df_map : pandas.DataFrame
        ``chunk_id ↔ share_link`` mapping (useful for merges).

    Notes
    -----
    Requires **pandas** only for the tiny mapping dataframe
    (import is deferred inside the function).
    """
    import pandas as pd

    if split is not None:
        splits = [split]
    elif isinstance(ds_dict, Dataset):
        splits = [ds_dict.split]
        ds_dict = {splits[0]: ds_dict}
    else:
        splits = list(ds_dict.keys())

    all_links: OrderedDict[str, None] = OrderedDict()
    for split in splits:
        ds: Dataset = ds_dict[split]
        for link in ds[link_column]:
            if link not in all_links:
                all_links[link] = None

    links = list(all_links)
    logger.info("Found %d unique share links across %s splits", len(links), splits)
    df_map = pd.DataFrame({"chunk_id": range(len(links)), "share_link": links})
    return links, df_map


def download_and_extract_links(
    links: list[str],
    target_dir: str | Path,
    *,
    temp_dir: str | Path | None = None,
    overwrite: bool = False,
    extract: bool = True,  # NEW – choose A (False) or B (True)
) -> dict[str, Path]:
    """
    Download every share-link or handle local paths.  If it is a ZIP (Nextcloud folder-download) either

    • keep the archive as ``chunk_<n>.zip``   (extract=False, default)  **OR**
    • extract *all* members to ``chunk_<n>.zarr/`` (extract=True).

    For local paths, they are used directly without downloading or copying.

    Returns
    -------
    dict[link, Path] – local path of .zip or .zarr/.h5ad
    """
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(temp_dir or tempfile.gettempdir())
    out_map: dict[str, Path] = {}

    def _is_local_path(link: str) -> bool:
        """Check if a link is a local file path rather than a URL."""
        # Check if it's a valid local path that exists
        try:
            path = Path(link)
            return path.exists() and (path.is_file() or path.is_dir())
        except (OSError, ValueError):
            return False

    for idx, link in enumerate(tqdm(links, desc="Processing", unit="file")):
        # Handle local paths directly
        if _is_local_path(link):
            local_path = Path(link).resolve()
            logger.info(f"Using local path: {local_path}")
            out_map[link] = local_path
            continue

        # For URLs, proceed with download logic
        # ------------------------- check if present----
        already = target_dir / f"chunk_{idx}.zip"
        if already.exists() and not overwrite:
            out_map[link] = already
            continue  # ← skip download altogether

        already = target_dir / f"chunk_{idx}.zarr"
        if already.exists() and not overwrite:
            out_map[link] = already
            continue
        # ---- stream into tmp file -----------------------------------------
        tmp = tmp_root / f"{uuid4().hex}.tmp"
        d_link = link  # use normal link to store later, to be consistent with how link is stored in dataset
        if not d_link.endswith("/download"):
            d_link += "/download"  # add NC download suffix
        with requests.get(d_link, stream=True, timeout=(10, 900)) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)

        m4 = tmp.read_bytes()[:4]
        if m4 == PK_MAGIC:  # ⇒ ZIP archive
            if extract:
                out_path = target_dir / f"chunk_{idx}.zarr"
                if overwrite and out_path.exists():
                    shutil.rmtree(out_path)
                with zipfile.ZipFile(tmp) as zf:
                    zf.extractall(out_path)
                tmp.unlink(missing_ok=True)
            else:  # keep the zip
                out_path = target_dir / f"chunk_{idx}.zip"
                if overwrite and out_path.exists():
                    out_path.unlink()
                shutil.move(tmp, out_path)
        else:  # raw .h5ad etc.
            suffix = Path(link).suffix or ".bin"
            out_path = target_dir / f"chunk_{idx}{suffix}"
            if overwrite and out_path.exists():
                out_path.unlink()
            shutil.move(tmp, out_path)

        out_map[link] = out_path
    return out_map


def build_embedding_df(
    link2path: dict[str, Path],
    *,
    layer_key: str,
    axis: Literal["obs", "var"] = "obs",
    chunk_rows: int = 1024,
) -> pd.DataFrame:
    """
    Get embeddings, either for *obs* or *var*.

    Collect **one** embedding vector per *obs* / *var* entity from a set of
    locally unpacked Zarr or H5AD stores – without reading the full object.

    Parameters
    ----------
    link2path
        Mapping ``share_link → local_path`` (output of `download_and_extract_links`).
        Each ``local_path`` can be:
        * a directory ending in ``.zarr``
        * a ZIP file with a `.zip` extension (will be opened in-place)
        * a single ``.h5ad`` file.
    layer_key
        Key in ``.obsm`` (for *axis="obs"*) **or** ``.varm`` (*axis="var"*)
        that holds the numeric embedding to extract.
    axis
        ``"obs"`` → use ``adata.obs.index`` and ``adata.obsm[layer_key]``
        ``"var"`` → use ``adata.var.index`` and ``adata.varm[layer_key]``.
    chunk_rows
        Number of rows to pull at once from the embedding matrix.

    Returns
    -------
    pandas.DataFrame
        Two columns: ``token`` (index label) + ``embedding`` (numpy array).

    Notes
    -----
    *Works with AnnData ≥ 0.10 and Zarr ≥ 2.16.*
    """
    rows: list[dict] = []

    for p in link2path.values():
        p = Path(p)
        logger.info("Reading %s", p)

        # -------------------------------------------------- open store ----
        if p.suffix == ".h5ad":  # plain HDF5
            adata = ad.read_h5ad(p, backed="r")
        elif p.suffix == ".zip":  # zipped zarr
            zroot = zarr.open(p, mode="r")
            adata = ad.read_zarr(zroot)
        elif p.suffix == ".zarr" or p.is_dir():  # dir zarr
            adata = ad.read_zarr(str(p))
        else:
            logger.warning("Skip unsupported file %s", p)
            continue

        emb_matrix = adata.obsm[layer_key] if axis == "obs" else adata.varm[layer_key]
        tokens = adata.obs.index.to_numpy() if axis == "obs" else adata.var.index.to_numpy()

        # -------------------------------------------------- stream rows ----
        n_rows = emb_matrix.shape[0]
        for start in range(0, n_rows, chunk_rows):
            end = min(start + chunk_rows, n_rows)
            chunk = emb_matrix[start:end]
            for i, vec in enumerate(chunk):
                # Convert to numpy array, handling both sparse matrices and regular lists
                if isinstance(vec, sp.spmatrix):
                    vec = np.asarray(vec.todense()).flatten()
                else:
                    vec = np.asarray(vec)
                rows.append({"token": tokens[start + i], "embedding": vec})
        adata.file.close() if hasattr(adata, "file") else None  # close backing

    df = pd.DataFrame(rows, columns=["token", "embedding"])
    logger.info("Built DataFrame with %d rows × %d-dim embeddings", len(df), len(df["embedding"].iloc[0]))
    return df


def save_table(
    df: pd.DataFrame, out_path: Path, fmt: Literal["csv", "parquet"] = "csv", *, index: bool = False
) -> None:
    """
    Persist a DataFrame to disk.

    Parameters
    ----------
    df : DataFrame
        Table to write.
    out_path : Path
        Full path, including the desired file name *without* extension.
    fmt : {'csv', 'parquet'}, default 'csv'
        File type.
    index : bool, default False
        Keep row index.

    Notes
    -----
    For Parquet we use pyarrow with ZSTD compression if available.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        file = out_path.with_suffix(".csv")
        df.to_csv(file, index=index)
    elif fmt == "parquet":
        file = out_path.with_suffix(".parquet")
        df.to_parquet(file, index=index, compression="zstd")
    else:
        raise ValueError(f"Unsupported format '{fmt}'")
    logger.info("Saved %s (%d rows) → %s", fmt.upper(), len(df), file)


def copy_resolved_config(cfg, hydra_output_dir: Path, named_output_dir: Path) -> None:
    """
    Copy the resolved config to the output directory.

    Serialize the *instantiated* Hydra config twice: once in the Hydra run
    directory and once in the named output directory so downstream users can
    locate it without knowing the run-dir.

    The config is enriched with a few runtime fields first (git SHA, SLURM
    job-ID, command line, etc.).
    """
    import subprocess

    from hydra.utils import to_absolute_path
    from omegaconf import OmegaConf

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # --- enrich with runtime metadata -------------------------------------
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    cfg_dict["_meta"] = {
        "git_sha": git_sha,
        "cmd": " ".join([to_absolute_path("main.py"), *os.sys.argv[1:]]),
        "slurm_job_id": os.getenv("SLURM_JOB_ID") if cfg.slurm.store_id else None,
    }

    # -----------------------------------------------------------------------
    for target_dir in (hydra_output_dir, named_output_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        out = target_dir / "resolved_config.json"
        with out.open("w") as fp:
            json.dump(cfg_dict, fp, indent=2)
        logger.info("Wrote resolved Hydra config → %s", out)

    # Also copy *raw* yaml files for debugging
    orig_conf_dir = Path("conf").absolute()
    shutil.copytree(orig_conf_dir, named_output_dir / "conf_raw", dirs_exist_ok=True)
