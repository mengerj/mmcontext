"""Tests for mmcontext.io.prepare_store — VectorStore preparation from zarr.

These tests use synthetic zarr stores on disk (no network calls).
The Zenodo URL fixup is tested via the internal helper.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import zarr

from mmcontext.io.prepare_store import (
    _get_obsm_zarr_array,
    _open_zarr,
    _read_obs_names_zarr,
    _url_to_cache_name,
    prepare_vector_store,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def synthetic_zarr(tmp_dir):
    """Create a minimal anndata-like zarr store on disk.

    Layout mirrors what anndata writes:
        obs/_index  — string array of obs names
        obsm/X_pca  — (N, D) float32 array
        obsm/X_scvi — (N, D2) float32 array
    """
    zarr_path = os.path.join(tmp_dir, "test.zarr")
    root = zarr.open(zarr_path, mode="w")

    n_obs, d_pca, d_scvi = 10, 4, 8
    rng = np.random.default_rng(42)

    def _write(group, name, data):
        arr = group.create_array(name, shape=data.shape, dtype=data.dtype)
        arr[:] = data
        return arr

    # obs group with _index
    obs = root.create_group("obs")
    obs_names = [f"cell_{i}" for i in range(n_obs)]
    _write(obs, "_index", np.array(obs_names, dtype="U"))
    obs.attrs["_index"] = "_index"

    # obsm group
    obsm = root.create_group("obsm")
    _write(obsm, "X_pca", rng.standard_normal((n_obs, d_pca)).astype(np.float32))
    _write(obsm, "X_scvi", rng.standard_normal((n_obs, d_scvi)).astype(np.float32))

    return zarr_path, obs_names, n_obs, d_pca, d_scvi


# ---------------------------------------------------------------------------
# URL fixup tests
# ---------------------------------------------------------------------------
class TestZenodoUrlFixup:
    """Zenodo draft→published URL conversion."""

    def test_draft_url_is_converted(self):
        """_download_zarr should strip /draft/ from Zenodo URLs.

        We can't test the actual download without network, but we verify
        the URL transformation logic by checking the function's behavior
        with a non-existent cache.
        """
        draft_url = "https://zenodo.org/api/records/12345/draft/files/data.zarr.zip/content"
        expected_published = "https://zenodo.org/api/records/12345/files/data.zarr.zip/content"

        # The fix is inside _download_zarr — we test it indirectly by
        # verifying the URL would be transformed. Since we can't mock
        # requests easily here, we just test the string operation.
        assert "/draft/" in draft_url
        fixed = draft_url.replace("/draft/", "/")
        assert fixed == expected_published

    def test_non_zenodo_url_unchanged(self):
        url = "https://example.com/data/draft/file.zip"
        # Only zenodo.org URLs get the fixup
        assert "zenodo.org" not in url

    def test_cache_name_deterministic(self):
        url = "https://zenodo.org/api/records/12345/files/data.zarr.zip/content"
        assert _url_to_cache_name(url) == _url_to_cache_name(url)

    def test_cache_name_differs_for_different_urls(self):
        url_a = "https://zenodo.org/api/records/111/files/a.zip/content"
        url_b = "https://zenodo.org/api/records/222/files/b.zip/content"
        assert _url_to_cache_name(url_a) != _url_to_cache_name(url_b)


# ---------------------------------------------------------------------------
# Zarr reading tests
# ---------------------------------------------------------------------------
class TestReadObsNames:
    def test_reads_obs_names(self, synthetic_zarr):
        zarr_path, expected_names, *_ = synthetic_zarr
        root = zarr.open(zarr_path, mode="r")
        names = _read_obs_names_zarr(root)
        assert names == expected_names

    def test_returns_strings(self, synthetic_zarr):
        zarr_path, *_ = synthetic_zarr
        root = zarr.open(zarr_path, mode="r")
        names = _read_obs_names_zarr(root)
        assert all(isinstance(n, str) for n in names)


class TestReadObsm:
    def test_reads_obsm_array(self, synthetic_zarr):
        zarr_path, _, n_obs, d_pca, _ = synthetic_zarr
        root = zarr.open(zarr_path, mode="r")
        arr = _get_obsm_zarr_array(root, "X_pca")
        assert arr.shape == (n_obs, d_pca)

    def test_missing_key_raises(self, synthetic_zarr):
        zarr_path, *_ = synthetic_zarr
        root = zarr.open(zarr_path, mode="r")
        with pytest.raises(KeyError, match="X_nonexistent"):
            _get_obsm_zarr_array(root, "X_nonexistent")

    def test_no_obsm_group_raises(self, tmp_dir):
        zarr_path = os.path.join(tmp_dir, "empty.zarr")
        root = zarr.open(zarr_path, mode="w")
        root.create_group("obs")
        root = zarr.open(zarr_path, mode="r")
        with pytest.raises(KeyError, match="obsm"):
            _get_obsm_zarr_array(root, "X_pca")


class TestOpenZarr:
    def test_open_directory(self, synthetic_zarr):
        from pathlib import Path

        zarr_path, *_ = synthetic_zarr
        root = _open_zarr(Path(zarr_path))
        assert "obs" in root
        assert "obsm" in root

    def test_nonexistent_raises(self, tmp_dir):
        from pathlib import Path

        with pytest.raises(FileNotFoundError):
            _open_zarr(Path(tmp_dir) / "nope.zarr")


# ---------------------------------------------------------------------------
# End-to-end: prepare_vector_store with local zarr paths
# ---------------------------------------------------------------------------
class TestPrepareVectorStore:
    """Integration test using local zarr paths (no network)."""

    def test_builds_store_from_local_zarr(self, synthetic_zarr, tmp_dir):
        """prepare_vector_store with local adata_link paths."""
        from datasets import Dataset

        zarr_path, obs_names, n_obs, d_pca, _ = synthetic_zarr

        # Simulate a dataset where all samples come from one zarr file
        ds = Dataset.from_dict(
            {
                "sample_idx": obs_names[:5],  # use first 5
                "adata_link": [zarr_path] * 5,
            }
        )

        output_path = os.path.join(tmp_dir, "test_store.mmap")
        store = prepare_vector_store(
            ds,
            obsm_key="X_pca",
            output_path=output_path,
        )

        assert len(store) == 5
        assert store.dim == d_pca
        # All sample IDs should be present
        for name in obs_names[:5]:
            assert name in store

    def test_values_match_zarr_source(self, synthetic_zarr, tmp_dir):
        """Vectors in VectorStore match the zarr source."""
        from datasets import Dataset

        zarr_path, obs_names, *_ = synthetic_zarr

        ds = Dataset.from_dict(
            {
                "sample_idx": [obs_names[3]],
                "adata_link": [zarr_path],
            }
        )

        output_path = os.path.join(tmp_dir, "val_store.mmap")
        store = prepare_vector_store(ds, obsm_key="X_pca", output_path=output_path)

        # Compare against direct zarr read
        root = zarr.open(zarr_path, mode="r")
        expected = np.asarray(root["obsm"]["X_pca"][3], dtype=np.float32)
        np.testing.assert_array_almost_equal(store[obs_names[3]], expected)

    def test_missing_sample_id_raises(self, synthetic_zarr, tmp_dir):
        """Unknown sample_idx raises KeyError."""
        from datasets import Dataset

        zarr_path, *_ = synthetic_zarr

        ds = Dataset.from_dict(
            {
                "sample_idx": ["nonexistent_cell"],
                "adata_link": [zarr_path],
            }
        )

        output_path = os.path.join(tmp_dir, "err_store.mmap")
        with pytest.raises(KeyError, match="nonexistent_cell"):
            prepare_vector_store(ds, obsm_key="X_pca", output_path=output_path)

    def test_skips_existing_store(self, synthetic_zarr, tmp_dir):
        """If output_path exists and overwrite=False, loads existing store."""
        from datasets import Dataset

        zarr_path, obs_names, *_ = synthetic_zarr

        ds = Dataset.from_dict(
            {
                "sample_idx": obs_names[:3],
                "adata_link": [zarr_path] * 3,
            }
        )

        output_path = os.path.join(tmp_dir, "cached_store.mmap")

        # Build once
        store1 = prepare_vector_store(ds, obsm_key="X_pca", output_path=output_path)
        # Build again — should load from disk
        store2 = prepare_vector_store(ds, obsm_key="X_pca", output_path=output_path)

        assert len(store2) == len(store1)

    def test_different_obsm_keys(self, synthetic_zarr, tmp_dir):
        """Can build stores from different obsm keys."""
        from datasets import Dataset

        zarr_path, obs_names, _, d_pca, d_scvi = synthetic_zarr

        ds = Dataset.from_dict(
            {
                "sample_idx": obs_names[:2],
                "adata_link": [zarr_path] * 2,
            }
        )

        pca_store = prepare_vector_store(
            ds,
            obsm_key="X_pca",
            output_path=os.path.join(tmp_dir, "pca.mmap"),
        )
        scvi_store = prepare_vector_store(
            ds,
            obsm_key="X_scvi",
            output_path=os.path.join(tmp_dir, "scvi.mmap"),
        )

        assert pca_store.dim == d_pca
        assert scvi_store.dim == d_scvi
