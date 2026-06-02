"""VectorStore — disk-backed vector lookup using numpy memory-mapped files.

Provides efficient storage and retrieval of high-dimensional embedding vectors
(e.g., scVI latent spaces, Geneformer embeddings, gene count vectors) without
loading the full matrix into RAM. Vectors are stored in a flat numpy memmap
file on disk; the OS pages in only the rows that are actually accessed.

Typical usage
-------------
**Create from AnnData:**

>>> store = VectorStore.from_adata(adata, layer_key="X_scvi", axis="obs", path="cache/vectors.mmap")

**Create from numpy:**

>>> store = VectorStore.from_numpy(matrix, ids, path="cache/vectors.mmap")

**Lookup:**

>>> vec = store["cell_42"]  # single lookup → (D,)
>>> batch = store.batch_lookup(ids)  # batch lookup  → (N, D)

**Persistence:**

>>> store2 = VectorStore.load("cache/vectors.mmap")  # reopen later
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """Disk-backed vector lookup using numpy memory-mapped files.

    Stores embedding vectors in a flat ``(N, D)`` numpy memmap file on disk,
    with a JSON sidecar file mapping string IDs to row indices. Only the rows
    actually accessed are paged into RAM by the OS, making it feasible to work
    with large embedding matrices (e.g., 300k × 10k) on machines with limited
    memory.

    Parameters
    ----------
    mmap : np.memmap
        Memory-mapped array of shape ``(N, D)``.
    index : dict[str, int]
        Mapping from string IDs to row indices in the memmap.
    path : Path
        Path to the memmap file on disk.

    Notes
    -----
    Do not instantiate directly — use the class methods
    :meth:`from_numpy`, :meth:`from_dataframe`, :meth:`from_adata`,
    :meth:`from_dict`, or :meth:`load`.
    """

    def __init__(
        self,
        mmap: np.memmap,
        index: dict[str, int],
        path: Path,
    ) -> None:
        self._mmap = mmap
        self._index = index
        self._path = path

    # ------------------------------------------------------------------
    # Construction class methods
    # ------------------------------------------------------------------
    @classmethod
    def from_numpy(
        cls,
        matrix: np.ndarray,
        ids: Sequence[str],
        *,
        path: str | Path,
    ) -> VectorStore:
        """Create a VectorStore from a numpy array and a list of IDs.

        Parameters
        ----------
        matrix : np.ndarray, shape (N, D)
            Embedding matrix. Row *i* is the vector for ``ids[i]``.
        ids : sequence of str
            Token/sample IDs. Must have the same length as ``matrix.shape[0]``.
        path : str or Path
            Where to write the memmap file. A sidecar ``<path>.index.json``
            is written alongside it.

        Returns
        -------
        VectorStore

        Raises
        ------
        ValueError
            If ``ids`` is empty, contains duplicates, or its length doesn't
            match ``matrix.shape[0]``.
        """
        ids = list(ids)
        cls._validate_ids_and_matrix(ids, matrix)

        path = Path(path)
        index = {sid: i for i, sid in enumerate(ids)}

        # Write memmap
        path.parent.mkdir(parents=True, exist_ok=True)
        mmap = np.memmap(path, dtype=matrix.dtype, mode="w+", shape=matrix.shape)
        mmap[:] = matrix
        mmap.flush()

        # Write sidecar index
        cls._write_index(path, index, matrix.shape, matrix.dtype)

        # Re-open as read-only
        mmap = np.memmap(path, dtype=matrix.dtype, mode="r", shape=matrix.shape)

        logger.info(
            "Created VectorStore: %d vectors × %d dims (%s) at %s",
            matrix.shape[0],
            matrix.shape[1],
            matrix.dtype,
            path,
        )
        return cls(mmap=mmap, index=index, path=path)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        path: str | Path,
        id_col: str = "token",
        embedding_col: str = "embedding",
    ) -> VectorStore:
        """Create a VectorStore from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns ``id_col`` (str IDs) and ``embedding_col``
            (array-like vectors).
        path : str or Path
            Where to write the memmap file.
        id_col : str
            Column name for token/sample IDs. Default: ``"token"``.
        embedding_col : str
            Column name for embedding vectors. Default: ``"embedding"``.

        Returns
        -------
        VectorStore
        """
        ids = df[id_col].tolist()
        matrix = np.vstack(df[embedding_col].to_numpy())
        return cls.from_numpy(matrix, ids, path=path)

    @classmethod
    def from_dict(
        cls,
        mapping: dict[str, np.ndarray],
        *,
        path: str | Path,
    ) -> VectorStore:
        """Create a VectorStore from a ``{id: vector}`` mapping.

        Parameters
        ----------
        mapping : dict[str, np.ndarray]
            Keys are string IDs, values are 1-D numpy arrays.
        path : str or Path
            Where to write the memmap file.

        Returns
        -------
        VectorStore
        """
        ids = list(mapping.keys())
        matrix = np.vstack(list(mapping.values()))
        return cls.from_numpy(matrix, ids, path=path)

    @classmethod
    def from_adata(
        cls,
        adata: ad.AnnData,
        *,
        layer_key: str,
        axis: Literal["obs", "var"] = "obs",
        path: str | Path,
    ) -> VectorStore:
        """Create a VectorStore from an AnnData object.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object containing embeddings.
        layer_key : str
            Key in ``.obsm`` (if *axis="obs"*) or ``.varm`` (if *axis="var"*).
        axis : {"obs", "var"}
            Which axis to extract embeddings from.
        path : str or Path
            Where to write the memmap file.

        Returns
        -------
        VectorStore

        Raises
        ------
        KeyError
            If ``layer_key`` is not found in the specified axis.
        """
        if axis == "obs":
            if layer_key not in adata.obsm:
                raise KeyError(f"Key '{layer_key}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}")
            matrix = np.asarray(adata.obsm[layer_key])
            ids = adata.obs.index.tolist()
        elif axis == "var":
            if layer_key not in adata.varm:
                raise KeyError(f"Key '{layer_key}' not found in adata.varm. Available keys: {list(adata.varm.keys())}")
            matrix = np.asarray(adata.varm[layer_key])
            ids = adata.var.index.tolist()
        else:
            raise ValueError(f"axis must be 'obs' or 'var', got '{axis}'")

        # Ensure float32 if not already a float type
        if not np.issubdtype(matrix.dtype, np.floating):
            matrix = matrix.astype(np.float32)

        return cls.from_numpy(matrix, ids, path=path)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> VectorStore:
        """Re-open a previously saved VectorStore from disk.

        Parameters
        ----------
        path : str or Path
            Path to the memmap file (the sidecar ``.index.json`` must exist
            alongside it).

        Returns
        -------
        VectorStore

        Raises
        ------
        FileNotFoundError
            If the memmap file or the index file does not exist.
        """
        path = Path(path)
        index_path = Path(str(path) + ".index.json")

        if not path.is_file():
            raise FileNotFoundError(f"Memmap file not found: {path}")
        if not index_path.is_file():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        with open(index_path) as f:
            meta = json.load(f)

        index = meta["index"]
        shape = tuple(meta["shape"])
        dtype = np.dtype(meta["dtype"])

        mmap = np.memmap(path, dtype=dtype, mode="r", shape=shape)

        logger.info(
            "Loaded VectorStore: %d vectors × %d dims (%s) from %s",
            shape[0],
            shape[1],
            dtype,
            path,
        )
        return cls(mmap=mmap, index=index, path=path)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> np.ndarray:
        """Look up a single vector by ID.

        Parameters
        ----------
        key : str
            Token/sample ID.

        Returns
        -------
        np.ndarray, shape (D,)
            The embedding vector (read from memmap, not a copy).

        Raises
        ------
        KeyError
            If the ID is not in the store.
        """
        try:
            idx = self._index[key]
        except KeyError:
            raise KeyError(f"ID '{key}' not found in VectorStore. Store contains {len(self._index)} entries.") from None
        return np.array(self._mmap[idx])

    def batch_lookup(self, ids: Sequence[str]) -> np.ndarray:
        """Look up multiple vectors by ID.

        Parameters
        ----------
        ids : sequence of str
            Token/sample IDs.

        Returns
        -------
        np.ndarray, shape (len(ids), D)
            Stacked embedding vectors.

        Raises
        ------
        KeyError
            If any ID is not in the store.
        """
        indices = []
        for sid in ids:
            try:
                indices.append(self._index[sid])
            except KeyError:
                raise KeyError(
                    f"ID '{sid}' not found in VectorStore. Store contains {len(self._index)} entries."
                ) from None
        return np.array(self._mmap[indices])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        """Embedding dimensionality (D)."""
        return self._mmap.shape[1]

    @property
    def dtype(self) -> np.dtype:
        """Data type of stored vectors."""
        return self._mmap.dtype

    def __len__(self) -> int:
        """Number of stored vectors."""
        return len(self._index)

    def __contains__(self, key: str) -> bool:
        """Check if an ID is in the store."""
        return key in self._index

    def __repr__(self) -> str:
        return f"VectorStore(n={len(self)}, dim={self.dim}, dtype={self.dtype}, path='{self._path}')"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_ids_and_matrix(ids: list[str], matrix: np.ndarray) -> None:
        """Validate that ids and matrix are consistent."""
        if len(ids) == 0:
            raise ValueError("Empty ID list: at least one vector is required.")
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2-D matrix, got {matrix.ndim}-D array with shape {matrix.shape}.")
        if len(ids) != matrix.shape[0]:
            raise ValueError(f"Length mismatch: {len(ids)} IDs but matrix has {matrix.shape[0]} rows.")
        if len(set(ids)) != len(ids):
            duplicates = [x for x in ids if ids.count(x) > 1]
            raise ValueError(f"Duplicate IDs found: {sorted(set(duplicates))[:10]}. All IDs must be unique.")

    @staticmethod
    def _write_index(
        path: Path,
        index: dict[str, int],
        shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> None:
        """Write the sidecar JSON index file."""
        index_path = Path(str(path) + ".index.json")
        meta = {
            "shape": list(shape),
            "dtype": str(dtype),
            "index": index,
        }
        with open(index_path, "w") as f:
            json.dump(meta, f)
