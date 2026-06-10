"""Test that ``prepare_model_and_embed`` builds embeddings via a preallocated
float32 array (instead of a list[list[float]]) without changing the returned
DataFrame schema or values.
"""

from __future__ import annotations

import numpy as np
import pytest
from datasets import Dataset

from mmcontext.embed.model_utils import prepare_model_and_embed

D = 6


class _FakeModule:
    """Stand-in for ``st_model[0]`` with none of the optional hooks."""


class _FakeSentenceTransformer:
    """Minimal object exposing the interface prepare_model_and_embed needs."""

    def __init__(self, table: dict[str, np.ndarray]):
        self._table = table
        self._module = _FakeModule()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self._module

    def eval(self):
        return self

    def encode(self, texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False):
        return np.vstack([self._table[t] for t in texts]).astype(np.float64)  # non-float32 on purpose


@pytest.fixture
def fake_data():
    rng = np.random.default_rng(0)
    n = 25
    texts = [f"sentence {i}" for i in range(n)]
    idx = [f"s{i}" for i in range(n)]
    table = {t: rng.random(D) for t in texts}
    ds = Dataset.from_dict({"text": texts, "idx": idx})
    expected = {i: table[t] for i, t in zip(idx, texts, strict=True)}
    return ds, table, expected


def test_preallocated_embeddings_roundtrip(fake_data):
    ds, table, expected = fake_data
    model = _FakeSentenceTransformer(table)

    emb_df, path_map = prepare_model_and_embed(
        model,
        ds,
        main_col="text",
        index_col="idx",
        batch_size=7,  # uneven final batch
        text_only=True,
    )

    assert path_map is None
    assert list(emb_df.columns) == ["sample_idx", "embedding"]
    assert len(emb_df) == len(ds)

    # Each cell is a 1-D float32 array (not a Python list).
    first = emb_df["embedding"].iloc[0]
    assert isinstance(first, np.ndarray)
    assert first.dtype == np.float32

    # Values round-trip correctly, keyed by sample id.
    for sid, vec in zip(emb_df["sample_idx"], emb_df["embedding"], strict=True):
        np.testing.assert_allclose(vec, expected[sid], rtol=1e-6, atol=1e-6)


def test_empty_dataset_returns_empty_frame(fake_data):
    _, table, _ = fake_data
    model = _FakeSentenceTransformer(table)
    ds = Dataset.from_dict({"text": [], "idx": []})
    emb_df, _ = prepare_model_and_embed(model, ds, main_col="text", index_col="idx", text_only=True)
    assert len(emb_df) == 0
    assert list(emb_df.columns) == ["sample_idx", "embedding"]
