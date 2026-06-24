"""Tests for the memory-lean zarr/h5ad readers in ``mmcontext.io._zarr_read``.

These verify that :func:`read_obs_and_obsm` reads only the obs table and the
requested ``obsm`` layer (optionally row-subset), preserves categoricals and
ordering, and never materialises the full store via ``anndata.read_zarr``.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mmcontext.io._zarr_read import read_obs_and_obsm

N, G, D = 60, 15, 8


@pytest.fixture
def synthetic_adata() -> ad.AnnData:
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(rng.choice(["T", "B", "NK"], N)),
            "tissue": rng.choice(["lung", "liver"], N),
            "score": rng.random(N).astype(np.float32),
        },
        index=[f"cell_{i}" for i in range(N)],
    )
    adata = ad.AnnData(X=rng.random((N, G)).astype(np.float32), obs=obs)
    adata.obsm["X_scvi"] = rng.random((N, D)).astype(np.float32)
    adata.obsm["X_pca"] = rng.random((N, 4)).astype(np.float32)
    return adata


@pytest.fixture(params=["zarr", "h5ad"])
def store(request, synthetic_adata, tmp_path) -> tuple[str, ad.AnnData]:
    """Write the synthetic AnnData as zarr or h5ad and return its path."""
    if request.param == "zarr":
        path = tmp_path / "chunk.zarr"
        synthetic_adata.write_zarr(path)
    else:
        path = tmp_path / "chunk.h5ad"
        synthetic_adata.write_h5ad(path)
    return str(path), synthetic_adata


def test_full_read_matches_source(store):
    path, adata = store
    obs, obsm = read_obs_and_obsm(path, obsm_key="X_scvi")
    assert list(obs.index) == list(adata.obs.index)
    assert obsm.dtype == np.float32
    np.testing.assert_allclose(obsm, adata.obsm["X_scvi"])


def test_row_subset_preserves_order(store):
    path, adata = store
    rows = np.array([7, 3, 40, 0, 25])
    obs, obsm = read_obs_and_obsm(path, obsm_key="X_scvi", rows=rows)
    assert list(obs.index) == [f"cell_{i}" for i in rows]
    np.testing.assert_allclose(obsm, adata.obsm["X_scvi"][rows])


def test_obs_columns_subset_and_missing_ignored(store):
    path, _ = store
    obs, _ = read_obs_and_obsm(path, obsm_key="X_scvi", obs_columns=["cell_type", "score", "absent"])
    assert list(obs.columns) == ["cell_type", "score"]


def test_categorical_preserved(store):
    path, adata = store
    rows = np.array([5, 1, 30])
    obs, _ = read_obs_and_obsm(path, obsm_key="X_scvi", rows=rows, obs_columns=["cell_type"])
    assert isinstance(obs["cell_type"].dtype, pd.CategoricalDtype)
    assert list(obs["cell_type"]) == list(adata.obs["cell_type"].iloc[rows])


def test_no_obsm_key_returns_none(store):
    path, _ = store
    obs, obsm = read_obs_and_obsm(path, obsm_key=None)
    assert obsm is None
    assert len(obs) == N


def test_empty_rows(store):
    path, _ = store
    obs, obsm = read_obs_and_obsm(path, obsm_key="X_scvi", rows=np.array([], dtype=int))
    assert obsm.shape == (0, D)
    assert len(obs) == 0


def test_missing_obsm_key_raises(store):
    path, _ = store
    with pytest.raises(KeyError):
        read_obs_and_obsm(path, obsm_key="does_not_exist")


def test_zarr_path_is_lazy(synthetic_adata, tmp_path, monkeypatch):
    """The zarr lean read must never call ``anndata.read_zarr`` (which would
    materialise the whole store)."""
    path = tmp_path / "chunk.zarr"
    synthetic_adata.write_zarr(path)

    def _boom(*args, **kwargs):
        raise AssertionError("read_obs_and_obsm must not call anndata.read_zarr")

    monkeypatch.setattr(ad, "read_zarr", _boom)
    obs, obsm = read_obs_and_obsm(str(path), obsm_key="X_scvi", rows=np.array([1, 2, 3]))
    assert obsm.shape == (3, D)
