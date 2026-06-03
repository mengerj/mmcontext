"""Tests for mmcontext.embed.dataset_prep — unified train/inference prep.

The pure ``prepare_dataset`` tests run entirely in-memory. The
``prepare_inference`` integration test uses a local AnnData zarr store on disk
(no network) and a lightweight fake model that only needs ``set_vector_store``.
"""

from __future__ import annotations

import os
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from mmcontext.embed import InferenceData, prepare_dataset, prepare_inference


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _train_ds(n: int = 6) -> Dataset:
    """Training-shaped dataset with positive + negative_*_idx columns."""
    return Dataset.from_dict(
        {
            "sample_idx": [f"cell_{i}" for i in range(n)],
            "cell_sentence_1": [f"GENE{i} CALM1 GNAS" for i in range(n)],
            "cell_sentence_2": [f"gene desc {i}" for i in range(n)],
            "positive": [f"a description of cell {i}" for i in range(n)],
            "negative_1_idx": [f"cell_{(i + 1) % n}" for i in range(n)],
            "negative_2_idx": [f"cell_{(i + 2) % n}" for i in range(n)],
            "adata_link": ["local://chunk"] * n,
        }
    )


def _test_ds(n: int = 6) -> Dataset:
    """Inference-shaped dataset: no positive/negative columns."""
    return Dataset.from_dict(
        {
            "sample_idx": [f"cell_{i}" for i in range(n)],
            "cell_sentence_1": [f"GENE{i} CALM1 GNAS" for i in range(n)],
            "cell_sentence_2": [f"gene desc {i}" for i in range(n)],
            "adata_link": ["local://chunk"] * n,
        }
    )


@pytest.fixture
def adata_zarr(tmp_dir):
    """A real AnnData written to a .zarr directory (readable by anndata)."""
    n_obs, d = 6, 8
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    adata = ad.AnnData(X=rng.standard_normal((n_obs, 4)).astype(np.float32), obs=obs)
    adata.obsm["X_test"] = rng.standard_normal((n_obs, d)).astype(np.float32)
    zarr_path = os.path.join(tmp_dir, "chunk.zarr")
    adata.write_zarr(zarr_path)
    return zarr_path, n_obs, d


class _FakeModule:
    def __init__(self):
        self.store = None

    def set_vector_store(self, store):
        self.store = store


class _FakeModel:
    """Stand-in for a SentenceTransformer: only ``model[0]`` is exercised."""

    def __init__(self):
        self._module = _FakeModule()

    def __getitem__(self, idx):
        return self._module


# ---------------------------------------------------------------------------
# prepare_dataset — training
# ---------------------------------------------------------------------------
class TestPrepareTrain:
    def test_bimodal_columns_and_omics_anchor(self):
        out = prepare_dataset(_train_ds(), purpose="train", modality="bimodal")
        assert out.column_names == ["anchor", "positive", "negative_1", "negative_2"]
        assert all(a.startswith("omics:") for a in out["anchor"])
        # omics ids carry the original sample_idx
        assert out["anchor"][0] == "omics:cell_0"

    def test_text_anchor_is_cell_sentence(self):
        raw = _train_ds()
        out = prepare_dataset(raw, purpose="train", modality="text")
        assert out.column_names == ["anchor", "positive", "negative_1", "negative_2"]
        # anchor equals the (unprefixed) cell sentence text
        assert out["anchor"] == raw["cell_sentence_1"]

    def test_without_hard_negatives(self):
        out = prepare_dataset(_train_ds(), purpose="train", modality="bimodal", use_hard_negatives=False)
        assert out.column_names == ["anchor", "positive"]
        assert all(a.startswith("omics:") for a in out["anchor"])

    def test_missing_positive_raises(self):
        ds = _test_ds()  # no positive column
        with pytest.raises(KeyError, match="positive"):
            prepare_dataset(ds, purpose="train", modality="bimodal")


# ---------------------------------------------------------------------------
# prepare_dataset — inference
# ---------------------------------------------------------------------------
class TestPrepareInference:
    def test_bimodal_keeps_identifiers(self):
        out = prepare_dataset(_test_ds(), purpose="inference", modality="bimodal")
        assert "anchor" in out.column_names
        assert "sample_idx" in out.column_names
        assert "adata_link" in out.column_names
        assert "positive" not in out.column_names
        assert all(a.startswith("omics:") for a in out["anchor"])

    def test_text_anchor(self):
        raw = _test_ds()
        out = prepare_dataset(raw, purpose="inference", modality="text")
        assert out["anchor"] == raw["cell_sentence_1"]

    def test_no_positive_or_negative_required(self):
        # The whole point: a test dataset without positive/negative must not error.
        out = prepare_dataset(_test_ds(), purpose="inference", modality="bimodal")
        assert len(out) == 6

    def test_datasetdict(self):
        dd = DatasetDict({"test": _test_ds()})
        out = prepare_dataset(dd, purpose="inference", modality="bimodal")
        assert isinstance(out, DatasetDict)
        assert all(a.startswith("omics:") for a in out["test"]["anchor"])


# ---------------------------------------------------------------------------
# prepare_dataset — validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_bad_modality(self):
        with pytest.raises(ValueError, match="modality"):
            prepare_dataset(_test_ds(), purpose="inference", modality="nope")

    def test_bad_purpose(self):
        with pytest.raises(ValueError, match="purpose"):
            prepare_dataset(_test_ds(), purpose="nope", modality="text")

    def test_bimodal_requires_sample_id(self):
        ds = Dataset.from_dict({"cell_sentence_1": ["a", "b"]})
        with pytest.raises(KeyError, match="sample"):
            prepare_dataset(ds, purpose="inference", modality="bimodal")


# ---------------------------------------------------------------------------
# prepare_inference — end-to-end with local zarr
# ---------------------------------------------------------------------------
class TestPrepareInferenceOrchestrator:
    def test_bimodal_builds_store_and_subsets(self, adata_zarr, tmp_dir):
        zarr_path, n_obs, d = adata_zarr
        ds = _test_ds(n_obs)
        ds = ds.remove_columns("adata_link")
        ds = ds.add_column("adata_link", [zarr_path] * n_obs)

        model = _FakeModel()
        bundle = prepare_inference(
            model,
            ds,
            modality="bimodal",
            obsm_key="X_test",
            cache_dir=tmp_dir,
            store_path=os.path.join(tmp_dir, "store.mmap"),
        )

        assert isinstance(bundle, InferenceData)
        assert bundle.vector_store is not None
        assert bundle.vector_store.dim == d
        # store was attached to the model's first module
        assert model[0].store is bundle.vector_store
        # dataset aligns with the loaded chunk
        assert len(bundle.dataset) == bundle.adata.n_obs == n_obs
        # every omics anchor resolves in the store
        for anchor in bundle.dataset["anchor"]:
            sid = anchor[len("omics:") :]
            assert bundle.vector_store[sid].shape == (d,)

    def test_text_skips_store(self, adata_zarr, tmp_dir):
        zarr_path, n_obs, _ = adata_zarr
        ds = _test_ds(n_obs)
        ds = ds.remove_columns("adata_link")
        ds = ds.add_column("adata_link", [zarr_path] * n_obs)

        model = _FakeModel()
        bundle = prepare_inference(
            model,
            ds,
            modality="text",
            cache_dir=tmp_dir,
        )
        assert bundle.vector_store is None
        assert model[0].store is None
        # text modality picks cell_sentence_2 internally
        assert bundle.dataset["anchor"] == ds["cell_sentence_2"]

    def test_bimodal_requires_obsm_key(self, tmp_dir):
        with pytest.raises(ValueError, match="obsm_key"):
            prepare_inference(_FakeModel(), _test_ds(), modality="bimodal", cache_dir=tmp_dir)
