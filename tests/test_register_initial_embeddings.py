# test_register_initial_embeddings.py
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mmcontext.mmcontextencoder import MMContextEncoder


# --- helper that builds a text-only encoder cheaply (no real BERT) --------
@patch("transformers.AutoModel.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def make_encoder(mock_tok, mock_model):
    enc = MMContextEncoder(
        text_encoder_name="bert-base-uncased",
        adapter_hidden_dim=32,
        adapter_output_dim=64,
    )
    return enc


def df_from(tokens, vectors):
    return pd.DataFrame({"token": tokens, "embedding": list(vectors)})


def test_register_initial_embeddings_with_overlap():
    enc = make_encoder()

    # first dataset: A,B,C
    # pad_vec = np.zeros(3, dtype=np.float32)
    vec1 = np.arange(9).reshape(3, 3).astype(np.float32)
    # vec1 = np.vstack([pad_vec, vec1])  # add padding vector
    ds1 = df_from(["A", "B", "C"], vec1)
    added1 = enc.register_initial_embeddings(ds1, data_origin="pca", return_added=True)
    assert added1 == {
        "A": 1,
        "B": 2,
        "C": 3,
    }  # First added element should have index 1, as index 0 is reserved for padding

    # second dataset: B,C,D,E  (B,C duplicate)
    vec2 = (np.arange(12).reshape(4, 3) + 100).astype(np.float32)
    ds2 = df_from(["B", "C", "D", "E"], vec2)
    added2 = enc.register_initial_embeddings(ds2, data_origin="pca", return_added=True)
    assert added2 == {"D": 4, "E": 5}  # only new ones

    # ------- verify embedding matrix contents --------------------------
    matrix = enc.omics_encoder.embeddings.weight.detach().cpu().numpy()
    # expected rows: A,B,C from vec1, D,E from vec2 (indices 2 and 3 of vec2)
    padding_row = np.zeros_like(vec1[0])  # first row (inx 0) is a padding vector
    expected = np.vstack([padding_row, vec1, vec2[2, :], vec2[3, :]])
    assert np.allclose(matrix, expected)

    # ------- verify lookup works --------------------------------------
    lookup = enc._omics_lookup
    expected_map = {"A": vec1[0], "B": vec1[1], "C": vec1[2], "D": vec2[2], "E": vec2[3]}
    for tok, row in expected_map.items():
        idx = lookup[tok]
        assert np.allclose(matrix[idx], row), f"row for {tok} incorrect"


def test_register_initial_embeddings_inserts_pad_row():
    """First call to `register_initial_embeddings` must create PAD row 0 (all-zero)."""
    enc = make_encoder()

    # numeric dataset WITHOUT a padding vector
    token_vecs = {
        "F1": np.ones(8, dtype=np.float32),
        "F2": np.full(8, 2.0, dtype=np.float32),
    }

    added = enc.register_initial_embeddings(token_vecs, data_origin="pca", return_added=True)

    # --- lookup indices --------------------------------------------------
    assert added == {"F1": 1, "F2": 2}  # first real token → index 1
    assert enc._omics_lookup["F1"] == 1
    assert enc._omics_lookup["F2"] == 2
    assert enc.processor.lookup is enc._omics_lookup  # processor updated

    # --- embedding matrix ------------------------------------------------
    weight = enc.omics_encoder.embeddings.weight.detach().cpu().numpy()
    assert weight.shape == (3, 8)  # 1 PAD + 2 real = 3 rows

    # row-0 must be exactly zeros
    assert np.allclose(weight[0], 0.0), "PAD row is not all zeros!"

    # rows 1 and 2 must equal the supplied vectors
    assert np.allclose(weight[1], token_vecs["F1"])
    assert np.allclose(weight[2], token_vecs["F2"])


def test_adding_incompatible_data(numeric_df_other_dim):
    """Test that adding data with incompatible dimensions raises an error."""
    enc = make_encoder()

    # first dataset: A,B,C
    vec1 = np.arange(9).reshape(3, 3).astype(np.float32)
    ds1 = df_from(["A", "B", "C"], vec1)
    enc.register_initial_embeddings(ds1, data_origin="pca")

    # Adding data with different dimensions should fail
    with pytest.raises(ValueError, match="dimension mismatch"):
        enc.register_initial_embeddings(numeric_df_other_dim, data_origin="pca")

    # Adding data with other origin should fail
    with pytest.raises(ValueError, match="cannot register"):
        enc.register_initial_embeddings(ds1, data_origin="hvg")
