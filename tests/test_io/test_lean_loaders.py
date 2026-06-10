"""Tests for the opt-in memory-lean paths in the AnnData loaders.

Covers:
* :func:`mmcontext.embed.dataset_utils.collect_adata_subset` with ``obsm_key``
* :func:`mmcontext.file_utils.load_test_adata_from_hf_dataset` with ``lean=True``

Both should read only the obs table and the requested ``obsm`` layer, return a
minimal AnnData (no ``X``), and never call ``anndata.read_zarr``.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mmcontext import file_utils
from mmcontext.embed import dataset_utils
from mmcontext.embed.dataset_utils import collect_adata_subset
from mmcontext.file_utils import load_test_adata_from_hf_dataset

D = 8


def _make_adata(ids: list[str], seed: int) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    n = len(ids)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(rng.choice(["T", "B", "NK"], n)),
            "score": rng.random(n).astype(np.float32),
        },
        index=ids,
    )
    adata = ad.AnnData(X=rng.random((n, 5)).astype(np.float32), obs=obs)
    adata.obsm["X_scvi"] = rng.random((n, D)).astype(np.float32)
    adata.obsm["X_pca"] = rng.random((n, 3)).astype(np.float32)
    return adata


@pytest.fixture
def two_chunks(tmp_path):
    """Two disjoint chunks (zarr + h5ad) plus an id→vector reference map."""
    a1 = _make_adata([f"cell_{i}" for i in range(30)], seed=1)
    a2 = _make_adata([f"cell_{i}" for i in range(30, 60)], seed=2)
    p1 = tmp_path / "chunk1.zarr"
    p2 = tmp_path / "chunk2.h5ad"
    a1.write_zarr(p1)
    a2.write_h5ad(p2)
    ref = {}
    for a in (a1, a2):
        for i, name in enumerate(a.obs_names):
            ref[name] = a.obsm["X_scvi"][i]
    return [p1, p2], ref


def test_collect_subset_lean_order_and_values(two_chunks):
    paths, ref = two_chunks
    ids = ["cell_45", "cell_2", "cell_58", "cell_10", "cell_31"]
    adata = collect_adata_subset(file_paths=paths, sample_ids=ids, obsm_key="X_scvi", obs_columns=["cell_type"])

    assert list(adata.obs_names) == ids
    assert adata.n_vars == 0  # no X materialised
    assert list(adata.obs.columns) == ["cell_type"]
    expected = np.vstack([ref[i] for i in ids])
    np.testing.assert_allclose(adata.obsm["X_scvi"], expected)


def test_collect_subset_lean_matches_legacy(two_chunks):
    paths, _ = two_chunks
    ids = ["cell_5", "cell_50", "cell_29", "cell_30"]
    lean = collect_adata_subset(file_paths=paths, sample_ids=ids, obsm_key="X_scvi")
    legacy = collect_adata_subset(file_paths=paths, sample_ids=ids)
    np.testing.assert_allclose(lean.obsm["X_scvi"], legacy.obsm["X_scvi"])
    assert list(lean.obs_names) == list(legacy.obs_names) == ids


def test_collect_subset_lean_does_not_read_full_zarr(two_chunks, monkeypatch):
    paths, _ = two_chunks

    def _boom(*args, **kwargs):
        raise AssertionError("lean collect_adata_subset must not call ad.read_zarr")

    monkeypatch.setattr(dataset_utils.ad, "read_zarr", _boom)
    ids = ["cell_1", "cell_31"]
    adata = collect_adata_subset(file_paths=paths, sample_ids=ids, obsm_key="X_scvi")
    assert list(adata.obs_names) == ids


@pytest.fixture
def hf_split(tmp_path, monkeypatch):
    """A single-chunk HF split whose download step is stubbed to a local zarr."""
    from datasets import Dataset

    adata = _make_adata([f"cell_{i}" for i in range(40)], seed=7)
    zpath = tmp_path / "test_chunk.zarr"
    adata.write_zarr(zpath)

    url = "https://example.org/fake.zarr"
    split = Dataset.from_dict({"adata_link": [url] * adata.n_obs, "sample_idx": list(adata.obs_names)})

    def _fake_download(links, target_dir, **kwargs):
        return {links[0]: zpath}

    monkeypatch.setattr(file_utils, "download_and_extract_links", _fake_download)
    return split, adata, tmp_path


def test_load_test_adata_lean(hf_split):
    split, adata, save_dir = hf_split
    lean, _ = load_test_adata_from_hf_dataset(split, save_dir, layer_key="X_scvi", lean=True, obs_columns=["cell_type"])
    assert lean.n_vars == 0
    assert list(lean.obs.columns) == ["cell_type"]
    assert list(lean.obs_names) == list(adata.obs_names)
    np.testing.assert_allclose(lean.obsm["X_scvi"], adata.obsm["X_scvi"])


def test_load_test_adata_lean_matches_full(hf_split):
    split, adata, save_dir = hf_split
    lean, _ = load_test_adata_from_hf_dataset(split, save_dir, layer_key="X_scvi", lean=True)
    full, _ = load_test_adata_from_hf_dataset(split, save_dir, layer_key="X_scvi", lean=False)
    np.testing.assert_allclose(lean.obsm["X_scvi"], full.obsm["X_scvi"])
    assert list(lean.obs_names) == list(full.obs_names)
