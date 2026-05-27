"""Tests for AdapterModule — modality-aware projection for the ST pipeline.

The AdapterModule sits after MMContextModule in the sentence-transformers
pipeline. It reads ``modality_ids`` from the features dict and applies
separate learned projections for text (modality_id=0) and omics
(modality_id=1) tokens, mapping them into a shared embedding space.

These tests define the contract and are written FIRST (TDD).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from mmcontext.modules.adapter_module import AdapterModule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def real_safetensors():
    """Undo the global safetensors.torch.load_model patch for persistence tests.

    The root conftest.py patches ``safetensors.torch.load_model`` to return None,
    which prevents real weight loading. This fixture restores the original
    function for tests that need actual save/load roundtrips.
    """
    import importlib
    import safetensors.torch
    importlib.reload(safetensors.torch)
    yield
    # The session-scoped patch in conftest will reassert on the next test that needs it


@pytest.fixture
def adapter():
    """Default adapter: text_dim=32, omics_dim=8, shared_dim=16, hidden_dim=64."""
    return AdapterModule(
        text_input_dim=32,
        omics_input_dim=8,
        shared_dim=16,
        hidden_dim=64,
    )


@pytest.fixture
def identity_adapter():
    """Adapter in identity mode (no projection)."""
    return AdapterModule(
        text_input_dim=16,
        omics_input_dim=16,
        shared_dim=16,
        hidden_dim=None,
        force_identity=True,
    )


def _make_features(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    modality_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Helper to build a features dict matching the pipeline contract."""
    return {
        "token_embeddings": token_embeddings,
        "attention_mask": attention_mask,
        "modality_ids": modality_ids,
    }


# ---------------------------------------------------------------------------
# Forward — single modality
# ---------------------------------------------------------------------------
class TestForwardSingleModality:
    """Tests for forward() with all-text or all-omics batches."""

    def test_forward_text_only(self, adapter):
        """All-text batch produces output with shared_dim."""
        B, L, D_text = 2, 5, 32
        features = _make_features(
            token_embeddings=torch.randn(B, L, D_text),
            attention_mask=torch.ones(B, L, dtype=torch.long),
            modality_ids=torch.zeros(B, L, dtype=torch.long),  # all text
        )
        result = adapter(features)

        assert result["token_embeddings"].shape == (B, L, 16)

    def test_forward_omics_only(self, adapter):
        """All-omics batch produces output with shared_dim."""
        B, L, D_omics = 3, 1, 8
        features = _make_features(
            token_embeddings=torch.randn(B, L, D_omics),
            attention_mask=torch.ones(B, L, dtype=torch.long),
            modality_ids=torch.ones(B, L, dtype=torch.long),  # all omics
        )
        result = adapter(features)

        assert result["token_embeddings"].shape == (B, L, 16)

    def test_forward_text_output_dimension(self, adapter):
        """Text projection maps D_text → D_shared exactly."""
        features = _make_features(
            token_embeddings=torch.randn(1, 3, 32),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            modality_ids=torch.zeros(1, 3, dtype=torch.long),
        )
        result = adapter(features)
        assert result["token_embeddings"].shape[-1] == adapter.get_sentence_embedding_dimension()

    def test_forward_omics_output_dimension(self, adapter):
        """Omics projection maps D_omics → D_shared exactly."""
        features = _make_features(
            token_embeddings=torch.randn(1, 1, 8),
            attention_mask=torch.ones(1, 1, dtype=torch.long),
            modality_ids=torch.ones(1, 1, dtype=torch.long),
        )
        result = adapter(features)
        assert result["token_embeddings"].shape[-1] == adapter.get_sentence_embedding_dimension()


# ---------------------------------------------------------------------------
# Forward — mixed batch
# ---------------------------------------------------------------------------
class TestForwardMixedBatch:
    """Tests for forward() with mixed text + omics tokens."""

    def test_forward_mixed_batch(self):
        """Mixed batch: text and omics tokens get different projections."""
        # Both modalities have same input dim for simplicity
        adapter = AdapterModule(
            text_input_dim=16, omics_input_dim=16, shared_dim=8, hidden_dim=32
        )
        B, L = 1, 4
        features = _make_features(
            token_embeddings=torch.randn(B, L, 16),
            attention_mask=torch.ones(B, L, dtype=torch.long),
            # First 2 tokens text, last 2 omics
            modality_ids=torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
        )
        result = adapter(features)

        assert result["token_embeddings"].shape == (B, L, 8)

    def test_pad_tokens_stay_zero(self, adapter):
        """Pad tokens (modality_id=2) remain zero after projection."""
        B, L = 1, 4
        embeddings = torch.randn(B, L, 32)
        embeddings[0, 3, :] = 0.0  # pad token is zero

        features = _make_features(
            token_embeddings=embeddings,
            attention_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
            modality_ids=torch.tensor([[0, 0, 0, 2]], dtype=torch.long),  # last is pad
        )
        result = adapter(features)

        # Pad token should remain zero
        assert torch.all(result["token_embeddings"][0, 3] == 0)


# ---------------------------------------------------------------------------
# Separate weights
# ---------------------------------------------------------------------------
class TestSeparateWeights:
    """Tests verifying text and omics projections are independent."""

    def test_separate_weights(self, adapter):
        """text_proj and omics_proj have independent parameter sets."""
        text_params = set(id(p) for p in adapter.text_proj.parameters())
        omics_params = set(id(p) for p in adapter.omics_proj.parameters())

        # No overlap
        assert text_params.isdisjoint(omics_params)
        # Both have parameters
        assert len(text_params) > 0
        assert len(omics_params) > 0

    def test_text_omics_produce_different_outputs(self):
        """Same input through text vs omics projection gives different results."""
        adapter = AdapterModule(
            text_input_dim=16, omics_input_dim=16, shared_dim=8, hidden_dim=32
        )
        x = torch.randn(1, 3, 16)

        text_features = _make_features(
            token_embeddings=x.clone(),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            modality_ids=torch.zeros(1, 3, dtype=torch.long),
        )
        omics_features = _make_features(
            token_embeddings=x.clone(),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            modality_ids=torch.ones(1, 3, dtype=torch.long),
        )

        text_result = adapter(text_features)
        omics_result = adapter(omics_features)

        # Different projections should produce different outputs (with overwhelming probability)
        assert not torch.allclose(
            text_result["token_embeddings"], omics_result["token_embeddings"]
        )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------
class TestGradientFlow:
    """Tests for gradient computation through adapter projections."""

    def test_gradient_flow_text(self, adapter):
        """Gradients flow through text projection."""
        features = _make_features(
            token_embeddings=torch.randn(1, 3, 32, requires_grad=True),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            modality_ids=torch.zeros(1, 3, dtype=torch.long),
        )
        result = adapter(features)
        # Use squared loss: plain sum(LayerNorm(x)) has zero gradient
        # because LayerNorm centers each token and summing centered
        # values cancels out. Squaring breaks this cancellation.
        loss = (result["token_embeddings"] ** 2).sum()
        loss.backward()

        for param in adapter.text_proj.parameters():
            assert param.grad is not None
            assert param.grad.abs().sum() > 0

    def test_gradient_flow_omics(self, adapter):
        """Gradients flow through omics projection."""
        features = _make_features(
            token_embeddings=torch.randn(2, 1, 8, requires_grad=True),
            attention_mask=torch.ones(2, 1, dtype=torch.long),
            modality_ids=torch.ones(2, 1, dtype=torch.long),
        )
        result = adapter(features)
        loss = (result["token_embeddings"] ** 2).sum()
        loss.backward()

        for param in adapter.omics_proj.parameters():
            assert param.grad is not None
            assert param.grad.abs().sum() > 0

    def test_weights_update(self, adapter):
        """Optimizer step changes both projection weights."""
        optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)

        # Snapshot weights before
        text_before = {n: p.clone() for n, p in adapter.text_proj.named_parameters()}
        omics_before = {n: p.clone() for n, p in adapter.omics_proj.named_parameters()}

        # Forward + backward through text
        features_t = _make_features(
            token_embeddings=torch.randn(2, 3, 32),
            attention_mask=torch.ones(2, 3, dtype=torch.long),
            modality_ids=torch.zeros(2, 3, dtype=torch.long),
        )
        result_t = adapter(features_t)
        loss_t = result_t["token_embeddings"].sum()

        # Forward + backward through omics
        features_o = _make_features(
            token_embeddings=torch.randn(2, 1, 8),
            attention_mask=torch.ones(2, 1, dtype=torch.long),
            modality_ids=torch.ones(2, 1, dtype=torch.long),
        )
        result_o = adapter(features_o)
        loss_o = result_o["token_embeddings"].sum()

        total_loss = loss_t + loss_o
        total_loss.backward()
        optimizer.step()

        # Both projections should have changed (check total param delta,
        # not per-parameter allclose, since some biases may get tiny gradients)
        text_delta = sum(
            (p - text_before[n]).abs().sum().item()
            for n, p in adapter.text_proj.named_parameters()
        )
        omics_delta = sum(
            (p - omics_before[n]).abs().sum().item()
            for n, p in adapter.omics_proj.named_parameters()
        )
        assert text_delta > 0, "text_proj parameters did not change"
        assert omics_delta > 0, "omics_proj parameters did not change"


# ---------------------------------------------------------------------------
# Identity mode
# ---------------------------------------------------------------------------
class TestIdentityMode:
    """Tests for identity/passthrough mode."""

    def test_identity_passthrough(self, identity_adapter):
        """When force_identity=True and dims match, input passes through."""
        x = torch.randn(1, 3, 16)
        features = _make_features(
            token_embeddings=x.clone(),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            modality_ids=torch.zeros(1, 3, dtype=torch.long),
        )
        result = identity_adapter(features)
        torch.testing.assert_close(result["token_embeddings"], x)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------
class TestProperties:
    """Tests for module properties and metadata."""

    def test_get_sentence_embedding_dimension(self, adapter):
        """Returns D_shared."""
        assert adapter.get_sentence_embedding_dimension() == 16

    def test_preserves_attention_mask(self, adapter):
        """Attention mask passes through unchanged."""
        mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
        features = _make_features(
            token_embeddings=torch.randn(1, 3, 32),
            attention_mask=mask,
            modality_ids=torch.zeros(1, 3, dtype=torch.long),
        )
        result = adapter(features)
        torch.testing.assert_close(result["attention_mask"], mask)

    def test_preserves_modality_ids(self, adapter):
        """modality_ids pass through unchanged."""
        mod_ids = torch.tensor([[0, 0, 1]], dtype=torch.long)
        features = _make_features(
            token_embeddings=torch.randn(1, 3, 32),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            modality_ids=mod_ids,
        )
        # Need same input dim for this test
        adapter2 = AdapterModule(
            text_input_dim=32, omics_input_dim=32, shared_dim=16, hidden_dim=64
        )
        result = adapter2(features)
        torch.testing.assert_close(result["modality_ids"], mod_ids)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
class TestPersistence:
    """Tests for save/load roundtrip."""

    def test_save_creates_files(self, adapter, tmp_dir, real_safetensors):
        """save() creates config and weight files on disk."""
        adapter.save(tmp_dir)

        config_path = os.path.join(tmp_dir, adapter.config_file_name)
        assert os.path.isfile(config_path)
        assert os.path.isfile(os.path.join(tmp_dir, "model.safetensors"))

    def test_save_load_roundtrip(self, adapter, tmp_dir, real_safetensors):
        """Config and weights survive save/load cycle."""
        adapter.save(tmp_dir)
        loaded = AdapterModule.load(tmp_dir)

        assert loaded.get_sentence_embedding_dimension() == adapter.get_sentence_embedding_dimension()
        assert loaded.text_input_dim == adapter.text_input_dim
        assert loaded.omics_input_dim == adapter.omics_input_dim

    def test_save_load_produces_same_output(self, adapter, tmp_dir, real_safetensors):
        """Loaded adapter produces identical output to original."""
        adapter.eval()
        x = torch.randn(1, 3, 32)
        features = _make_features(
            token_embeddings=x,
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            modality_ids=torch.zeros(1, 3, dtype=torch.long),
        )
        original_out = adapter(features)["token_embeddings"].detach()

        adapter.save(tmp_dir)
        loaded = AdapterModule.load(tmp_dir)
        loaded.eval()

        features2 = _make_features(
            token_embeddings=x,
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            modality_ids=torch.zeros(1, 3, dtype=torch.long),
        )
        loaded_out = loaded(features2)["token_embeddings"].detach()

        torch.testing.assert_close(original_out, loaded_out)
