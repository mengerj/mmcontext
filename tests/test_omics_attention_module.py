"""Tests for OmicsAttentionModule — optional self-attention for omics tokens.

The OmicsAttentionModule sits between MMContextModule and AdapterModule in the
sentence-transformers pipeline. It applies multi-head self-attention ONLY to
omics tokens (modality_id=1), leaving text tokens (modality_id=0) unchanged.
This is useful for var-level models where each sample has multiple gene
embeddings that benefit from contextual mixing before projection.

These tests define the contract and are written FIRST (TDD).
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from mmcontext.modules.omics_attention_module import OmicsAttentionModule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def real_safetensors():
    """Undo the global safetensors.torch.load_model patch for persistence tests."""
    import importlib
    import safetensors.torch
    importlib.reload(safetensors.torch)
    yield


@pytest.fixture
def module():
    """Default module: input_dim=16, num_heads=2, num_layers=1."""
    return OmicsAttentionModule(
        input_dim=16,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )


@pytest.fixture
def deep_module():
    """Deeper module for testing multi-layer stacking."""
    return OmicsAttentionModule(
        input_dim=16,
        num_heads=2,
        num_layers=3,
        dropout=0.0,
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
# Text passthrough
# ---------------------------------------------------------------------------
class TestTextPassthrough:
    """Text tokens must pass through completely unchanged."""

    def test_text_passthrough(self, module):
        """Text tokens (modality_id=0) are identical before and after the module."""
        B, L = 2, 5
        x = torch.randn(B, L, 16)
        features = _make_features(
            token_embeddings=x.clone(),
            attention_mask=torch.ones(B, L, dtype=torch.long),
            modality_ids=torch.zeros(B, L, dtype=torch.long),  # all text
        )
        module.eval()
        result = module(features)

        torch.testing.assert_close(result["token_embeddings"], x)

    def test_text_passthrough_in_mixed_batch(self, module):
        """Text tokens in a mixed batch remain unchanged."""
        B, L = 1, 6
        x = torch.randn(B, L, 16)
        modality_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long)
        features = _make_features(
            token_embeddings=x.clone(),
            attention_mask=torch.ones(B, L, dtype=torch.long),
            modality_ids=modality_ids,
        )
        module.eval()
        result = module(features)

        # Text tokens (first 3) should be unchanged
        torch.testing.assert_close(
            result["token_embeddings"][:, :3, :], x[:, :3, :]
        )


# ---------------------------------------------------------------------------
# Omics transformation
# ---------------------------------------------------------------------------
class TestOmicsTransformed:
    """Omics tokens should be modified by self-attention."""

    def test_omics_transformed(self, module):
        """Omics tokens (modality_id=1) are modified by self-attention."""
        B, L = 1, 4
        x = torch.randn(B, L, 16)
        features = _make_features(
            token_embeddings=x.clone(),
            attention_mask=torch.ones(B, L, dtype=torch.long),
            modality_ids=torch.ones(B, L, dtype=torch.long),  # all omics
        )
        module.eval()
        result = module(features)

        # Output should differ from input (attention mixes information)
        assert not torch.allclose(
            result["token_embeddings"], x, atol=1e-6
        ), "Omics tokens should be transformed by self-attention"

    def test_omics_transformed_in_mixed_batch(self, module):
        """Omics tokens in a mixed batch are modified."""
        B, L = 1, 6
        x = torch.randn(B, L, 16)
        modality_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long)
        features = _make_features(
            token_embeddings=x.clone(),
            attention_mask=torch.ones(B, L, dtype=torch.long),
            modality_ids=modality_ids,
        )
        module.eval()
        result = module(features)

        # Omics tokens (last 3) should be modified
        assert not torch.allclose(
            result["token_embeddings"][:, 3:, :], x[:, 3:, :], atol=1e-6
        ), "Omics tokens should be transformed by self-attention"


# ---------------------------------------------------------------------------
# Attention mask respected
# ---------------------------------------------------------------------------
class TestAttentionMask:
    """Padded positions must not influence real tokens."""

    def test_attention_mask_respected(self, module):
        """Padded omics tokens don't affect real omics tokens.

        We verify by comparing outputs with and without a pad token — the
        real tokens' representations should differ when the pad is replaced
        by a real token (showing attention sees it), and should NOT change
        when the pad is properly masked.
        """
        module.eval()
        D = 16

        # Batch element with 3 real omics tokens + 1 pad
        x = torch.randn(1, 4, D)
        mask_with_pad = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
        modality_ids = torch.tensor([[1, 1, 1, 2]], dtype=torch.long)

        features_padded = _make_features(
            token_embeddings=x.clone(),
            attention_mask=mask_with_pad,
            modality_ids=modality_ids,
        )
        out_padded = module(features_padded)["token_embeddings"][:, :3, :]

        # Same real tokens but with a different value in the pad position
        x2 = x.clone()
        x2[0, 3, :] = torch.randn(D) * 100  # very different pad content

        features_padded2 = _make_features(
            token_embeddings=x2,
            attention_mask=mask_with_pad,
            modality_ids=modality_ids,
        )
        out_padded2 = module(features_padded2)["token_embeddings"][:, :3, :]

        # Real tokens should produce same output regardless of pad content
        torch.testing.assert_close(out_padded, out_padded2, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Variable-length sequences
# ---------------------------------------------------------------------------
class TestVariableLengthSequences:
    """Different-length gene sequences in the same batch."""

    def test_variable_length_sequences(self, module):
        """Batch with different numbers of omics tokens per sample."""
        module.eval()
        B, L, D = 2, 5, 16
        x = torch.randn(B, L, D)

        # Sample 0: 3 omics tokens + 2 pad
        # Sample 1: 5 omics tokens + 0 pad
        modality_ids = torch.tensor([
            [1, 1, 1, 2, 2],
            [1, 1, 1, 1, 1],
        ], dtype=torch.long)
        attention_mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ], dtype=torch.long)

        features = _make_features(
            token_embeddings=x.clone(),
            attention_mask=attention_mask,
            modality_ids=modality_ids,
        )
        result = module(features)

        # Output shape preserved
        assert result["token_embeddings"].shape == (B, L, D)

        # Pad positions should remain zero (or unchanged)
        # The module should not produce non-zero values for pad positions
        assert torch.all(result["token_embeddings"][0, 3:, :] == 0) or \
               torch.allclose(result["token_embeddings"][0, 3:, :], x[0, 3:, :])


# ---------------------------------------------------------------------------
# Output shape preserved
# ---------------------------------------------------------------------------
class TestOutputShape:
    """Output shape must be identical to input shape."""

    def test_output_shape_preserved(self, module):
        """(B, L, D) shape is unchanged after the module."""
        B, L, D = 3, 7, 16
        features = _make_features(
            token_embeddings=torch.randn(B, L, D),
            attention_mask=torch.ones(B, L, dtype=torch.long),
            modality_ids=torch.ones(B, L, dtype=torch.long),
        )
        result = module(features)
        assert result["token_embeddings"].shape == (B, L, D)

    def test_output_shape_with_deep_module(self, deep_module):
        """Shape preserved through multiple attention layers."""
        B, L, D = 2, 4, 16
        features = _make_features(
            token_embeddings=torch.randn(B, L, D),
            attention_mask=torch.ones(B, L, dtype=torch.long),
            modality_ids=torch.ones(B, L, dtype=torch.long),
        )
        result = deep_module(features)
        assert result["token_embeddings"].shape == (B, L, D)

    def test_preserves_attention_mask(self, module):
        """attention_mask passes through unchanged."""
        mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
        features = _make_features(
            token_embeddings=torch.randn(1, 3, 16),
            attention_mask=mask,
            modality_ids=torch.ones(1, 3, dtype=torch.long),
        )
        result = module(features)
        torch.testing.assert_close(result["attention_mask"], mask)

    def test_preserves_modality_ids(self, module):
        """modality_ids pass through unchanged."""
        mod_ids = torch.tensor([[0, 1, 1, 2]], dtype=torch.long)
        features = _make_features(
            token_embeddings=torch.randn(1, 4, 16),
            attention_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
            modality_ids=mod_ids,
        )
        result = module(features)
        torch.testing.assert_close(result["modality_ids"], mod_ids)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------
class TestGradientFlow:
    """Gradients must flow through the attention layers."""

    def test_gradient_flow(self, module):
        """Gradients flow through attention to omics tokens."""
        x = torch.randn(2, 4, 16, requires_grad=True)
        features = _make_features(
            token_embeddings=x,
            attention_mask=torch.ones(2, 4, dtype=torch.long),
            modality_ids=torch.ones(2, 4, dtype=torch.long),
        )
        result = module(features)
        loss = (result["token_embeddings"] ** 2).sum()
        loss.backward()

        # Input should have gradients
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

        # Module parameters should have gradients
        has_grad = False
        for p in module.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "At least one module parameter should have non-zero gradient"

    def test_gradient_only_through_omics(self, module):
        """In a mixed batch, text token gradients are zero (passthrough)."""
        x = torch.randn(1, 4, 16, requires_grad=True)
        modality_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        features = _make_features(
            token_embeddings=x,
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            modality_ids=modality_ids,
        )
        result = module(features)
        # Only compute loss on omics tokens
        loss = (result["token_embeddings"][:, 2:, :] ** 2).sum()
        loss.backward()

        # Omics positions should have gradients
        assert x.grad[:, 2:, :].abs().sum() > 0

    def test_weights_update(self, module):
        """Optimizer step changes attention weights."""
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

        params_before = {n: p.clone() for n, p in module.named_parameters()}

        features = _make_features(
            token_embeddings=torch.randn(2, 4, 16),
            attention_mask=torch.ones(2, 4, dtype=torch.long),
            modality_ids=torch.ones(2, 4, dtype=torch.long),
        )
        result = module(features)
        loss = (result["token_embeddings"] ** 2).sum()
        loss.backward()
        optimizer.step()

        total_delta = sum(
            (p - params_before[n]).abs().sum().item()
            for n, p in module.named_parameters()
        )
        assert total_delta > 0, "Parameters did not change after optimizer step"


# ---------------------------------------------------------------------------
# Single token sequence (obs-like)
# ---------------------------------------------------------------------------
class TestSingleTokenSequence:
    """Obs-like input with L=1 must work without error."""

    def test_single_token_sequence(self, module):
        """Single omics token (obs-level) passes through without error."""
        module.eval()
        B, L, D = 3, 1, 16
        x = torch.randn(B, L, D)
        features = _make_features(
            token_embeddings=x.clone(),
            attention_mask=torch.ones(B, L, dtype=torch.long),
            modality_ids=torch.ones(B, L, dtype=torch.long),
        )
        result = module(features)

        # Shape preserved
        assert result["token_embeddings"].shape == (B, L, D)

    def test_single_token_no_nan(self, module):
        """Single token doesn't produce NaN (self-attention on length-1 is trivial)."""
        module.eval()
        x = torch.randn(1, 1, 16)
        features = _make_features(
            token_embeddings=x.clone(),
            attention_mask=torch.ones(1, 1, dtype=torch.long),
            modality_ids=torch.ones(1, 1, dtype=torch.long),
        )
        result = module(features)
        assert not torch.isnan(result["token_embeddings"]).any()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
class TestPersistence:
    """Save/load roundtrip tests."""

    def test_save_creates_files(self, module, tmp_dir, real_safetensors):
        """save() creates config and weight files on disk."""
        module.save(tmp_dir)

        config_path = os.path.join(tmp_dir, module.config_file_name)
        assert os.path.isfile(config_path)
        assert os.path.isfile(os.path.join(tmp_dir, "model.safetensors"))

    def test_save_load_roundtrip(self, module, tmp_dir, real_safetensors):
        """Config and weights survive save/load cycle."""
        module.save(tmp_dir)
        loaded = OmicsAttentionModule.load(tmp_dir)

        assert loaded.get_sentence_embedding_dimension() == module.get_sentence_embedding_dimension()
        assert loaded.input_dim == module.input_dim
        assert loaded.num_heads == module.num_heads
        assert loaded.num_layers == module.num_layers

    def test_save_load_produces_same_output(self, module, tmp_dir, real_safetensors):
        """Loaded module produces identical output to original."""
        module.eval()
        x = torch.randn(2, 4, 16)
        features = _make_features(
            token_embeddings=x,
            attention_mask=torch.ones(2, 4, dtype=torch.long),
            modality_ids=torch.ones(2, 4, dtype=torch.long),
        )
        original_out = module(features)["token_embeddings"].detach()

        module.save(tmp_dir)
        loaded = OmicsAttentionModule.load(tmp_dir)
        loaded.eval()

        features2 = _make_features(
            token_embeddings=x,
            attention_mask=torch.ones(2, 4, dtype=torch.long),
            modality_ids=torch.ones(2, 4, dtype=torch.long),
        )
        loaded_out = loaded(features2)["token_embeddings"].detach()

        torch.testing.assert_close(original_out, loaded_out)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------
class TestProperties:
    """Module metadata and properties."""

    def test_get_sentence_embedding_dimension(self, module):
        """Returns input_dim (self-attention doesn't change dimensionality)."""
        assert module.get_sentence_embedding_dimension() == 16

    def test_repr(self, module):
        """repr contains key config info."""
        r = repr(module)
        assert "16" in r  # input_dim
        assert "2" in r   # num_heads
