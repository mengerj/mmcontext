"""Tests for VectorStore — memory-mapped vector lookup.

These tests define the contract that VectorStore must satisfy:
construction from multiple data sources, lookup, persistence, and edge cases.
"""

from __future__ import annotations

import os
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mmcontext.io import VectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_matrix():
    """A small (5, 8) float32 matrix with known values."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((5, 8)).astype(np.float32)


@pytest.fixture
def sample_ids():
    """Token IDs matching sample_matrix rows."""
    return ["cell_A", "cell_B", "cell_C", "cell_D", "cell_E"]


@pytest.fixture
def sample_df(sample_matrix, sample_ids):
    """DataFrame with 'token' and 'embedding' columns."""
    return pd.DataFrame(
        {
            "token": sample_ids,
            "embedding": [sample_matrix[i] for i in range(len(sample_ids))],
        }
    )


@pytest.fixture
def sample_adata_obs(sample_matrix, sample_ids):
    """AnnData with obs-level embeddings in .obsm['X_scvi']."""
    adata = ad.AnnData(
        X=np.zeros((5, 3)),  # dummy expression matrix
        obs=pd.DataFrame(index=sample_ids),
    )
    adata.obsm["X_scvi"] = sample_matrix
    return adata


@pytest.fixture
def sample_adata_var():
    """AnnData with var-level embeddings in .varm['gene_emb']."""
    gene_ids = ["EGFR", "KRAS", "TP53", "BRCA1"]
    rng = np.random.default_rng(99)
    gene_matrix = rng.standard_normal((4, 16)).astype(np.float32)

    adata = ad.AnnData(
        X=np.zeros((2, 4)),
        var=pd.DataFrame(index=gene_ids),
    )
    adata.varm["gene_emb"] = gene_matrix
    return adata


@pytest.fixture
def tmp_dir():
    """Temporary directory for memmap files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------
class TestVectorStoreConstruction:
    """Tests for creating VectorStore from different data sources."""

    def test_from_numpy(self, sample_matrix, sample_ids, tmp_dir):
        """Round-trip: create from numpy, values match on lookup."""
        path = os.path.join(tmp_dir, "test.mmap")
        store = VectorStore.from_numpy(sample_matrix, sample_ids, path=path)

        for i, sid in enumerate(sample_ids):
            result = store[sid]
            np.testing.assert_array_almost_equal(result, sample_matrix[i])

    def test_from_dataframe(self, sample_df, tmp_dir):
        """Create from DataFrame with token/embedding columns."""
        path = os.path.join(tmp_dir, "test.mmap")
        store = VectorStore.from_dataframe(sample_df, path=path)

        for _, row in sample_df.iterrows():
            result = store[row["token"]]
            np.testing.assert_array_almost_equal(result, row["embedding"])

    def test_from_dataframe_custom_columns(self, sample_matrix, sample_ids, tmp_dir):
        """Create from DataFrame with custom column names."""
        df = pd.DataFrame(
            {
                "my_id": sample_ids,
                "my_vec": [sample_matrix[i] for i in range(len(sample_ids))],
            }
        )
        path = os.path.join(tmp_dir, "test.mmap")
        store = VectorStore.from_dataframe(
            df, path=path, id_col="my_id", embedding_col="my_vec"
        )
        result = store["cell_A"]
        np.testing.assert_array_almost_equal(result, sample_matrix[0])

    def test_from_adata_obs(self, sample_adata_obs, tmp_dir):
        """Create from adata.obsm, lookup by obs index."""
        path = os.path.join(tmp_dir, "test.mmap")
        store = VectorStore.from_adata(
            sample_adata_obs, layer_key="X_scvi", axis="obs", path=path
        )
        expected = sample_adata_obs.obsm["X_scvi"]

        for i, sid in enumerate(sample_adata_obs.obs.index):
            result = store[sid]
            np.testing.assert_array_almost_equal(result, expected[i])

    def test_from_adata_var(self, sample_adata_var, tmp_dir):
        """Create from adata.varm, lookup by var index."""
        path = os.path.join(tmp_dir, "test.mmap")
        store = VectorStore.from_adata(
            sample_adata_var, layer_key="gene_emb", axis="var", path=path
        )
        expected = sample_adata_var.varm["gene_emb"]

        for i, gid in enumerate(sample_adata_var.var.index):
            result = store[gid]
            np.testing.assert_array_almost_equal(result, expected[i])

    def test_from_dict(self, sample_matrix, sample_ids, tmp_dir):
        """Create from {id: vector} mapping."""
        mapping = {sid: sample_matrix[i] for i, sid in enumerate(sample_ids)}
        path = os.path.join(tmp_dir, "test.mmap")
        store = VectorStore.from_dict(mapping, path=path)

        for sid, vec in mapping.items():
            np.testing.assert_array_almost_equal(store[sid], vec)


# ---------------------------------------------------------------------------
# Lookup tests
# ---------------------------------------------------------------------------
class TestVectorStoreLookup:
    """Tests for looking up vectors by ID."""

    def test_single_lookup(self, sample_matrix, sample_ids, tmp_dir):
        """Single ID lookup returns 1-D array."""
        path = os.path.join(tmp_dir, "test.mmap")
        store = VectorStore.from_numpy(sample_matrix, sample_ids, path=path)

        result = store["cell_C"]
        assert result.ndim == 1
        assert result.shape == (8,)
        np.testing.assert_array_almost_equal(result, sample_matrix[2])

    def test_batch_lookup(self, sample_matrix, sample_ids, tmp_dir):
        """Batch of IDs returns correct (N, D) array."""
        path = os.path.join(tmp_dir, "test.mmap")
        store = VectorStore.from_numpy(sample_matrix, sample_ids, path=path)

        ids = ["cell_E", "cell_A", "cell_C"]
        result = store.batch_lookup(ids)
        assert result.shape == (3, 8)
        np.testing.assert_array_almost_equal(result[0], sample_matrix[4])
        np.testing.assert_array_almost_equal(result[1], sample_matrix[0])
        np.testing.assert_array_almost_equal(result[2], sample_matrix[2])

    def test_unknown_id_raises(self, sample_matrix, sample_ids, tmp_dir):
        """KeyError for missing IDs."""
        path = os.path.join(tmp_dir, "test.mmap")
        store = VectorStore.from_numpy(sample_matrix, sample_ids, path=path)

        with pytest.raises(KeyError, match="unknown_cell"):
            store["unknown_cell"]

    def test_batch_lookup_unknown_raises(self, sample_matrix, sample_ids, tmp_dir):
        """KeyError in batch lookup for missing IDs."""
        path = os.path.join(tmp_dir, "test.mmap")
        store = VectorStore.from_numpy(sample_matrix, sample_ids, path=path)

        with pytest.raises(KeyError):
            store.batch_lookup(["cell_A", "nonexistent"])

    def test_duplicate_ids_rejected(self, sample_matrix, tmp_dir):
        """Duplicate token IDs are rejected at construction time."""
        ids_with_dup = ["cell_A", "cell_B", "cell_A", "cell_D", "cell_E"]
        path = os.path.join(tmp_dir, "test.mmap")

        with pytest.raises(ValueError, match="[Dd]uplicate"):
            VectorStore.from_numpy(sample_matrix, ids_with_dup, path=path)


# ---------------------------------------------------------------------------
# Properties tests
# ---------------------------------------------------------------------------
class TestVectorStoreProperties:
    """Tests for VectorStore metadata properties."""

    def test_dim(self, sample_matrix, sample_ids, tmp_dir):
        """Reports correct embedding dimension."""
        path = os.path.join(tmp_dir, "test.mmap")
        store = VectorStore.from_numpy(sample_matrix, sample_ids, path=path)
        assert store.dim == 8

    def test_dtype(self, sample_matrix, sample_ids, tmp_dir):
        """Reports correct dtype."""
        path = os.path.join(tmp_dir, "test.mmap")
        store = VectorStore.from_numpy(sample_matrix, sample_ids, path=path)
        assert store.dtype == np.float32

    def test_len(self, sample_matrix, sample_ids, tmp_dir):
        """Reports correct number of stored vectors."""
        path = os.path.join(tmp_dir, "test.mmap")
        store = VectorStore.from_numpy(sample_matrix, sample_ids, path=path)
        assert len(store) == 5

    def test_contains(self, sample_matrix, sample_ids, tmp_dir):
        """Supports 'in' operator for checking IDs."""
        path = os.path.join(tmp_dir, "test.mmap")
        store = VectorStore.from_numpy(sample_matrix, sample_ids, path=path)
        assert "cell_A" in store
        assert "nonexistent" not in store


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------
class TestVectorStorePersistence:
    """Tests for save/load and memmap persistence."""

    def test_persistence_across_reopen(self, sample_matrix, sample_ids, tmp_dir):
        """Store survives close/reopen cycle."""
        path = os.path.join(tmp_dir, "test.mmap")
        store = VectorStore.from_numpy(sample_matrix, sample_ids, path=path)

        # Delete the Python object
        del store

        # Reopen from the same path
        store2 = VectorStore.load(path)
        assert len(store2) == 5
        assert store2.dim == 8
        np.testing.assert_array_almost_equal(store2["cell_A"], sample_matrix[0])
        np.testing.assert_array_almost_equal(store2["cell_E"], sample_matrix[4])

    def test_load_nonexistent_raises(self):
        """Loading from non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            VectorStore.load("/nonexistent/path/store.mmap")

    def test_files_created_on_disk(self, sample_matrix, sample_ids, tmp_dir):
        """Construction creates memmap file and index file on disk."""
        path = os.path.join(tmp_dir, "test.mmap")
        VectorStore.from_numpy(sample_matrix, sample_ids, path=path)

        # Memmap data file should exist
        assert os.path.isfile(path)
        # Index file should exist alongside
        index_path = path + ".index.json"
        assert os.path.isfile(index_path)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestVectorStoreEdgeCases:
    """Tests for edge cases and special inputs."""

    def test_single_vector(self, tmp_dir):
        """Store with a single vector works correctly."""
        matrix = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        path = os.path.join(tmp_dir, "test.mmap")
        store = VectorStore.from_numpy(matrix, ["only_one"], path=path)

        assert len(store) == 1
        assert store.dim == 3
        np.testing.assert_array_almost_equal(store["only_one"], [1.0, 2.0, 3.0])

    def test_large_dimension(self, tmp_dir):
        """Handles high-dimensional vectors (gs10k-like)."""
        rng = np.random.default_rng(7)
        matrix = rng.standard_normal((10, 10_000)).astype(np.float32)
        ids = [f"cell_{i}" for i in range(10)]
        path = os.path.join(tmp_dir, "test.mmap")
        store = VectorStore.from_numpy(matrix, ids, path=path)

        assert store.dim == 10_000
        np.testing.assert_array_almost_equal(store["cell_5"], matrix[5])

    def test_float16_dtype(self, tmp_dir):
        """Supports float16 for memory efficiency."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
        path = os.path.join(tmp_dir, "test.mmap")
        store = VectorStore.from_numpy(matrix, ["a", "b"], path=path)

        assert store.dtype == np.float16
        np.testing.assert_array_almost_equal(store["a"], [1.0, 2.0], decimal=3)

    def test_empty_ids_raises(self, tmp_dir):
        """Empty ID list is rejected."""
        matrix = np.empty((0, 8), dtype=np.float32)
        path = os.path.join(tmp_dir, "test.mmap")

        with pytest.raises(ValueError, match="[Ee]mpty"):
            VectorStore.from_numpy(matrix, [], path=path)

    def test_mismatched_ids_matrix_raises(self, sample_matrix, tmp_dir):
        """Mismatched number of IDs and matrix rows raises ValueError."""
        path = os.path.join(tmp_dir, "test.mmap")
        with pytest.raises(ValueError, match="[Mm]ismatch|[Ll]ength"):
            VectorStore.from_numpy(sample_matrix, ["a", "b"], path=path)  # 5 rows, 2 IDs
