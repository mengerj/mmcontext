# test_miniomics_embedding.py
import numpy as np
import pytest
import torch
from torch import nn

from mmcontext.models.omicsencoder import MiniOmicsModel


# ---------------------------------------------------------------------- #
# helper: build an embedding matrix with or without a zero pad-row
# ---------------------------------------------------------------------- #
def make_matrix(vocab_size=10, hidden=4, zero_pad=True, seed=42):
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((vocab_size, hidden)).astype(np.float32)
    if zero_pad:
        mat[0] = 0.0  # force PAD row to zeros
    return mat


# ---------------------------------------------------------------------- #
# 1) the model must *reject* a matrix whose pad-row is non-zero
# ---------------------------------------------------------------------- #
def test_from_numpy_raises_on_nonzero_pad():
    bad_matrix = make_matrix(zero_pad=False)  # row-0 contains random numbers

    with pytest.raises(ValueError) as exc:
        MiniOmicsModel.from_numpy(bad_matrix, padding_idx=0)

    # check the message helps the user
    assert "row of zeros" in str(exc.value).lower()


# ---------------------------------------------------------------------- #
# 2) with a correct pad-row the model returns zeros where id==0 and
#    the pooler output is the mean over *non-pad* tokens only
# ---------------------------------------------------------------------- #
def test_pooled_output_correct_mean_and_pad_handling():
    good_matrix = make_matrix(zero_pad=True)
    model = MiniOmicsModel.from_numpy(good_matrix, padding_idx=0)

    # two samples, three tokens each – last token in 2nd sample is PAD (id 0)
    ids = torch.tensor([[2, 3, 4], [9, 1, 0]])

    out = model(ids)

    # ------------- a) pad position should yield an all-zero vector --------
    assert torch.allclose(out.last_hidden_state[1, 2], torch.zeros_like(out.last_hidden_state[1, 2]))

    # ------------- b) pooler_output must be masked mean ------------------
    mask = ids != 0  # (B, L)
    n_tokens = mask.float().sum(dim=1, keepdim=True)  # (B, 1)
    expected_pooled = (out.last_hidden_state * mask.unsqueeze(-1)).sum(dim=1) / n_tokens

    assert torch.allclose(out.pooler_output, expected_pooled, atol=0, rtol=0)


def test_embedding_lookup_exact():
    vocab_size, hidden = 10, 4
    matrix = make_matrix(vocab_size, hidden, zero_pad=True)

    model = MiniOmicsModel.from_numpy(embedding_matrix=matrix, padding_idx=0)

    # ---------------------------------------------------------------------
    # simulated batch of IDs: two sequences of length 3
    ids = torch.tensor([[0, 3, 4], [9, 1, 0]])

    out = model(ids)  # (2, 3, 4)

    # verify every returned vector equals the row in the lookup
    for b in range(ids.size(0)):
        for t in range(ids.size(1)):
            idx = ids[b, t].item()
            expected = torch.from_numpy(matrix[idx])
            assert torch.allclose(out.last_hidden_state[b, t], expected, atol=0, rtol=0), (
                f"ID {idx}: {out[b, t]} != {expected}"
            )


def make_model_and_data(vocab_size=6, hidden=3, pad_id=0):
    """Utility: deterministic lookup + tiny input batch."""
    rng = np.random.default_rng(0)
    lookup = rng.standard_normal((vocab_size, hidden)).astype(np.float32)

    model = MiniOmicsModel.from_numpy(embedding_matrix=lookup, padding_idx=pad_id)

    # two samples, three tokens each
    ids = torch.tensor([[0, 2, 1], [4, 5, 0]])  # shape (B=2, L=3)

    # expected vectors straight from the numpy table
    expected = torch.from_numpy(lookup)[ids]

    return model, ids, expected


# -------------------------------------------------------------------- #
# Helper: wrapper that adds a trainable head on top of the omics model
# -------------------------------------------------------------------- #
class OmicsWithHead(torch.nn.Module):
    def __init__(self, encoder: MiniOmicsModel, hidden_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(hidden_dim, 1)  # ← stays trainable

    def forward(self, ids, mask=None):
        pooled = self.encoder(ids, attention_mask=mask).pooler_output  # (B, H)
        return self.head(pooled).squeeze(-1)  # (B,)


# -------------------------------------------------------------------- #
# Test
# -------------------------------------------------------------------- #
@pytest.mark.parametrize("freeze_embeddings", [True, False])
def test_embedding_freeze_effect(freeze_embeddings):
    vocab_size, hidden, pad_id = 8, 4, 0

    # deterministic lookup matrix
    rng = np.random.default_rng(0)
    lookup = rng.standard_normal((vocab_size, hidden)).astype(np.float32)
    # make first row vector of zeros
    lookup[pad_id] = 0.0

    # base encoder + wrapper
    encoder = MiniOmicsModel.from_numpy(embedding_matrix=lookup, padding_idx=pad_id)
    model = OmicsWithHead(encoder, hidden_dim=hidden)

    # freeze or not
    encoder.embeddings.weight.requires_grad = not freeze_embeddings

    # simple optimiser over *all* params (frozen ones get skipped automatically)
    optim = torch.optim.SGD(model.parameters(), lr=1.0)

    # tiny batch (B=2, L=3) + dummy mask (all real tokens here)
    ids = torch.tensor([[0, 2, 1], [6, 3, 0]])
    mask = torch.ones_like(ids, dtype=torch.bool)

    # ------------------------------------------------------------------ #
    # Record initial weights
    emb_before = encoder.embeddings.weight.detach().clone()
    head_before = model.head.weight.detach().clone()

    # ------------------------------------------------------------------ #
    # One training step
    model.train()
    pred = model(ids, mask)  # (B,)
    loss = pred.pow(2).mean()  # dummy MSE to zero
    loss.backward()
    optim.step()

    emb_after = encoder.embeddings.weight.detach()
    head_after = model.head.weight.detach()

    # ------------------------------------------------------------------ #
    # Assertions
    if freeze_embeddings:
        # embedding table must remain unchanged
        assert torch.allclose(emb_before, emb_after, atol=0, rtol=0), "Embeddings changed although they were frozen"
    else:
        # embedding table must have updated
        assert not torch.allclose(emb_before, emb_after, atol=0, rtol=0), (
            "Embeddings did not update when they were trainable"
        )

    # head must always update
    assert not torch.allclose(head_before, head_after, atol=0, rtol=0), (
        "Head did not update — optimiser step seems to have failed"
    )


def test_single_id_batch_keeps_shape_and_values():
    # 1) deterministic lookup table
    vocab, hid = 5, 4
    table = make_matrix(vocab, hid, zero_pad=True)

    model = MiniOmicsModel.from_numpy(table, padding_idx=0)

    # 2) input: batch_size = 1, seq_len = 1
    ids = torch.tensor([1])  # <- 1-D on purpose

    out = model(ids)

    # 3) shape assertions
    assert out.last_hidden_state.shape == (1, 1, hid)
    assert out.pooler_output.shape == (1, hid)

    # 4) value assertions – row 0 of the table
    expected_vec = torch.from_numpy(table[1])  # (hid,)
    assert torch.allclose(out.last_hidden_state[0, 0], expected_vec, atol=0, rtol=0)
    assert torch.allclose(out.pooler_output[0], expected_vec, atol=0, rtol=0)
