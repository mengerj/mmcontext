"""Tests for MMContextModule — the core InputModule for multimodal encoding.

These tests define the contract that MMContextModule must satisfy.
They are written FIRST (TDD) and drive the implementation.

MMContextModule extends sentence-transformers InputModule (v5.4+) and serves
as the first module in the ST pipeline. It handles:
  - Text preprocessing via AutoTokenizer
  - Omics vector pass-through (direct or via VectorStore lookup)
  - Producing a unified features dict with token_embeddings, attention_mask,
    and modality_ids for downstream modules (AdapterModule, Pooling).

The v5.4 API uses ``preprocess()`` instead of the deprecated ``tokenize()``.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from mmcontext.io import VectorStore
from mmcontext.modules import MMContextModule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_dir():
    """Temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def omics_store(tmp_dir):
    """A small VectorStore with 5 cells, dim=8."""
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((5, 8)).astype(np.float32)
    ids = ["cell_A", "cell_B", "cell_C", "cell_D", "cell_E"]
    path = os.path.join(tmp_dir, "test_store.mmap")
    return VectorStore.from_numpy(matrix, ids, path=path)


@pytest.fixture
def module():
    """MMContextModule backed by stub text encoder (patched globally in conftest).

    The session-scoped ``patch_model_loading`` fixture in conftest.py patches
    ``AutoModel.from_pretrained`` and ``AutoTokenizer.from_pretrained`` to
    return lightweight stubs, so no real HF download occurs.
    """
    return MMContextModule(model_name_or_path="bert-base-uncased")


@pytest.fixture
def module_with_store(module, omics_store):
    """Module with a VectorStore attached."""
    module.set_vector_store(omics_store)
    return module


# ---------------------------------------------------------------------------
# Preprocess tests — text
# ---------------------------------------------------------------------------
class TestPreprocessText:
    """Tests for preprocess() with text-only inputs."""

    def test_preprocess_text_returns_required_keys(self, module):
        """Text preprocessing produces input_ids, attention_mask, and modality marker."""
        features = module.preprocess(["This is a test sentence."])

        assert "input_ids" in features
        assert "attention_mask" in features
        assert features["modality"] == "text"

    def test_preprocess_text_batch(self, module):
        """Batch of text strings produces correct batch dimension."""
        texts = ["First sentence.", "Second sentence.", "Third one."]
        features = module.preprocess(texts)

        assert features["input_ids"].shape[0] == 3
        assert features["attention_mask"].shape[0] == 3

    def test_preprocess_text_tensor_types(self, module):
        """Preprocessed text features are torch tensors."""
        features = module.preprocess(["Hello world."])

        assert isinstance(features["input_ids"], torch.Tensor)
        assert isinstance(features["attention_mask"], torch.Tensor)

    def test_preprocess_text_with_prompt(self, module):
        """Optional prompt is prepended to text inputs."""
        module.preprocess(["Test."])
        features_with_prompt = module.preprocess(["Test."], prompt="Query: ")

        # Both should produce valid features; the prompt version processes
        # "Query: Test." instead of "Test."
        assert "input_ids" in features_with_prompt
        assert features_with_prompt["modality"] == "text"


# ---------------------------------------------------------------------------
# Preprocess tests — omics via VectorStore
# ---------------------------------------------------------------------------
class TestPreprocessOmicsViaStore:
    """Tests for preprocess() with omics IDs resolved through VectorStore."""

    def test_preprocess_omics_ids_returns_embeddings(self, module_with_store):
        """Prefixed omics IDs are resolved to vectors via VectorStore."""
        inputs = ["omics:cell_A", "omics:cell_B"]
        features = module_with_store.preprocess(inputs)

        assert "input_values" in features
        assert features["modality"] == "omics"
        # Each cell maps to a single vector → (B, 1, D)
        assert features["input_values"].shape == (2, 1, 8)

    def test_preprocess_omics_ids_values_match_store(self, module_with_store, omics_store):
        """Resolved vectors match the VectorStore content."""
        inputs = ["omics:cell_C"]
        features = module_with_store.preprocess(inputs)

        expected = omics_store["cell_C"]
        actual = features["input_values"][0, 0].numpy()
        np.testing.assert_array_almost_equal(actual, expected)

    def test_preprocess_omics_ids_attention_mask(self, module_with_store):
        """Omics tokens get attention_mask=1."""
        inputs = ["omics:cell_A", "omics:cell_B"]
        features = module_with_store.preprocess(inputs)

        assert "attention_mask" in features
        assert features["attention_mask"].shape == (2, 1)
        assert (features["attention_mask"] == 1).all()

    def test_preprocess_omics_unknown_id_raises(self, module_with_store):
        """Unknown omics ID raises KeyError."""
        with pytest.raises(KeyError, match="unknown_cell"):
            module_with_store.preprocess(["omics:unknown_cell"])


# ---------------------------------------------------------------------------
# Preprocess tests — omics direct vectors
# ---------------------------------------------------------------------------
class TestPreprocessOmicsDirect:
    """Tests for preprocess() with direct omics vectors (via dict input)."""

    def test_preprocess_direct_single_vector(self, module):
        """Dict with omics_values (1-D array) → single omics token per sample."""
        inputs = [
            {"omics_values": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
            {"omics_values": np.array([4.0, 5.0, 6.0], dtype=np.float32)},
        ]
        features = module.preprocess(inputs)

        assert features["modality"] == "omics"
        assert features["input_values"].shape == (2, 1, 3)
        assert features["attention_mask"].shape == (2, 1)

    def test_preprocess_direct_values_preserved(self, module):
        """Direct vectors appear unchanged in input_values."""
        vec = np.array([1.5, -2.5, 3.5], dtype=np.float32)
        features = module.preprocess([{"omics_values": vec}])

        actual = features["input_values"][0, 0].numpy()
        np.testing.assert_array_almost_equal(actual, vec)

    def test_preprocess_direct_var_multiple_vectors(self, module):
        """List of gene vectors → variable-length omics sequence (var case)."""
        gene_vecs_1 = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([5.0, 6.0], dtype=np.float32),
        ]
        gene_vecs_2 = [
            np.array([7.0, 8.0], dtype=np.float32),
            np.array([9.0, 10.0], dtype=np.float32),
        ]
        inputs = [
            {"omics_values": gene_vecs_1},  # 3 genes
            {"omics_values": gene_vecs_2},  # 2 genes
        ]
        features = module.preprocess(inputs)

        # Padded to max length in batch → (2, 3, 2)
        assert features["input_values"].shape == (2, 3, 2)
        # Attention mask reflects real vs padded tokens
        assert features["attention_mask"][0].tolist() == [1, 1, 1]
        assert features["attention_mask"][1].tolist() == [1, 1, 0]


# ---------------------------------------------------------------------------
# Preprocess tests — error cases
# ---------------------------------------------------------------------------
class TestPreprocessErrors:
    """Tests for preprocess() error handling."""

    def test_no_store_omics_id_raises(self, module):
        """Prefixed omics IDs without a VectorStore → clear error."""
        with pytest.raises(ValueError, match="[Vv]ector[Ss]tore|[Nn]o.*store"):
            module.preprocess(["omics:cell_A"])

    def test_empty_input_raises(self, module):
        """Empty input list raises ValueError."""
        with pytest.raises((ValueError, RuntimeError)):
            module.preprocess([])


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------
class TestForward:
    """Tests for forward() producing the features dict contract."""

    def test_forward_text_produces_token_embeddings(self, module):
        """Text forward: input_ids → token_embeddings via text encoder."""
        features = module.preprocess(["A test sentence."])
        result = module.forward(features)

        assert "token_embeddings" in result
        assert "attention_mask" in result
        assert "modality_ids" in result

    def test_forward_text_shapes(self, module):
        """Text forward produces correct shapes: (B, L, D)."""
        features = module.preprocess(["Hello.", "World."])
        result = module.forward(features)

        B = 2
        L = features["input_ids"].shape[1]  # sequence length from tokenizer
        D = module.get_word_embedding_dimension()

        assert result["token_embeddings"].shape == (B, L, D)
        assert result["attention_mask"].shape == (B, L)
        assert result["modality_ids"].shape == (B, L)

    def test_forward_text_modality_ids_zero(self, module):
        """Text tokens get modality_id=0."""
        features = module.preprocess(["Test."])
        result = module.forward(features)

        assert (result["modality_ids"] == 0).all()

    def test_forward_omics_passthrough(self, module_with_store):
        """Omics forward: input_values pass through to token_embeddings unchanged."""
        features = module_with_store.preprocess(["omics:cell_A"])
        input_embeddings = features["input_values"].clone()
        result = module_with_store.forward(features)

        assert "token_embeddings" in result
        torch.testing.assert_close(result["token_embeddings"], input_embeddings)

    def test_forward_omics_modality_ids_one(self, module_with_store):
        """Omics tokens get modality_id=1."""
        features = module_with_store.preprocess(["omics:cell_A"])
        result = module_with_store.forward(features)

        assert (result["modality_ids"] == 1).all()

    def test_forward_omics_shapes(self, module):
        """Omics forward produces correct shapes."""
        vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        features = module.preprocess([{"omics_values": vec}])
        result = module.forward(features)

        assert result["token_embeddings"].shape == (1, 1, 4)
        assert result["attention_mask"].shape == (1, 1)
        assert result["modality_ids"].shape == (1, 1)

    def test_forward_returns_dict(self, module):
        """Forward returns a dict (pipeline contract)."""
        features = module.preprocess(["Test sentence."])
        result = module.forward(features)

        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Properties tests
# ---------------------------------------------------------------------------
class TestProperties:
    """Tests for module properties and metadata."""

    def test_modalities_property(self, module):
        """Module declares text and omics modalities."""
        assert hasattr(module, "modalities")
        assert "text" in module.modalities
        assert "omics" in module.modalities

    def test_max_seq_length(self, module):
        """max_seq_length is accessible and returns a positive int."""
        assert hasattr(module, "max_seq_length")
        assert isinstance(module.max_seq_length, int)
        assert module.max_seq_length > 0

    def test_max_seq_length_settable(self, module):
        """max_seq_length can be set to a new value."""
        module.max_seq_length = 256
        assert module.max_seq_length == 256

    def test_get_word_embedding_dimension(self, module):
        """Reports the text encoder's hidden size."""
        dim = module.get_word_embedding_dimension()
        assert isinstance(dim, int)
        assert dim > 0

    def test_vector_store_management(self, module, omics_store):
        """set_vector_store and remove_vector_store work correctly."""
        assert module._vector_store is None

        module.set_vector_store(omics_store)
        assert module._vector_store is omics_store

        module.remove_vector_store()
        assert module._vector_store is None


# ---------------------------------------------------------------------------
# Text encoder freezing tests
# ---------------------------------------------------------------------------
class TestFreezing:
    """Tests for freezing/unfreezing the text encoder."""

    def test_unfreeze_text_encoder(self, module):
        """Unfreezing restores gradient computation after a full freeze."""
        module.freeze_all_but_top_layers(0)
        module.unfreeze_text_encoder()
        for param in module.auto_model.parameters():
            assert param.requires_grad

    def test_freeze_all_but_top_layers_keeps_only_top(self, module):
        """freeze_all_but_top_layers(N) trains only the top N layers."""
        layers = module._get_encoder_layers()
        assert layers is not None and len(layers) >= 2, "stub encoder needs >=2 layers"

        module.freeze_all_but_top_layers(1)

        # The last layer is trainable; all earlier layers are frozen.
        for param in layers[-1].parameters():
            assert param.requires_grad
        for layer in layers[:-1]:
            for param in layer.parameters():
                assert not param.requires_grad

    def test_freeze_all_but_top_zero_freezes_everything(self, module):
        """num_trainable_layers=0 freezes the whole encoder."""
        module.freeze_all_but_top_layers(0)
        for param in module.auto_model.parameters():
            assert not param.requires_grad

    def test_freeze_all_but_top_n_exceeds_total_keeps_all_layers(self, module):
        """Asking for more layers than exist keeps every encoder layer trainable."""
        layers = module._get_encoder_layers()
        module.freeze_all_but_top_layers(len(layers) + 5)
        for layer in layers:
            for param in layer.parameters():
                assert param.requires_grad


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------
class TestPersistence:
    """Tests for save/load roundtrip."""

    def test_save_creates_files(self, module, tmp_dir):
        """save() creates config and weight files on disk."""
        module.save(tmp_dir)

        # Config file should exist
        config_path = os.path.join(tmp_dir, module.config_file_name)
        assert os.path.isfile(config_path)

    def test_save_load_roundtrip(self, module, tmp_dir):
        """Config values survive save/load cycle."""
        original_dim = module.get_word_embedding_dimension()
        original_seq_len = module.max_seq_length

        module.save(tmp_dir)
        loaded = MMContextModule.load(tmp_dir)

        assert loaded.get_word_embedding_dimension() == original_dim
        assert loaded.max_seq_length == original_seq_len

    def test_save_load_forward_text(self, module, tmp_dir):
        """Loaded module can run forward on text."""
        module.save(tmp_dir)
        loaded = MMContextModule.load(tmp_dir)

        features = loaded.preprocess(["Test after reload."])
        result = loaded.forward(features)

        assert "token_embeddings" in result
        assert result["token_embeddings"].shape[0] == 1

    def test_save_excludes_vector_store(self, module_with_store):
        """VectorStore data is NOT saved with the model weights."""
        # Use a separate directory for saving (not the one containing the store)
        with tempfile.TemporaryDirectory() as save_dir:
            module_with_store.save(save_dir)

            # No memmap files should appear in the save directory
            saved_files = os.listdir(save_dir)
            mmap_files = [f for f in saved_files if f.endswith(".mmap")]
            assert len(mmap_files) == 0

            # Loaded module should not have a vector store attached
            loaded = MMContextModule.load(save_dir)
            assert loaded._vector_store is None

    def test_load_preserves_modalities(self, module, tmp_dir):
        """Loaded module still reports correct modalities."""
        module.save(tmp_dir)
        loaded = MMContextModule.load(tmp_dir)

        assert "text" in loaded.modalities
        assert "omics" in loaded.modalities
