"""Integration tests for the full SentenceTransformer pipeline.

These tests verify that the mmcontext modules compose correctly into a
sentence-transformers pipeline and that encode(), save/load, and training
work end-to-end.

Two pipeline configurations are tested:

1. **Obs pipeline** (samples have a single omics vector each):
   ``[MMContextModule, AdapterModule, Pooling, Normalize]``

2. **Var pipeline** (samples have multiple gene vectors each):
   ``[MMContextModule, OmicsAttentionModule, AdapterModule, Pooling, Normalize]``
"""

from __future__ import annotations

import importlib
import json
import os
import tempfile

import numpy as np
import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Normalize, Pooling

from mmcontext.io import VectorStore
from mmcontext.modules import AdapterModule, MMContextModule, OmicsAttentionModule


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
    import safetensors.torch
    importlib.reload(safetensors.torch)
    yield


# -- Shared dims used across all fixtures -----------------------------------
TEXT_DIM = 32     # matches _TextEncStub hidden_size
OMICS_DIM = 8
SHARED_DIM = 16


@pytest.fixture
def obs_store(tmp_dir):
    """VectorStore with 5 obs-level samples, dim=OMICS_DIM."""
    ids = [f"cell_{i}" for i in range(5)]
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, OMICS_DIM)).astype(np.float32)
    store = VectorStore.from_numpy(data, ids, path=os.path.join(tmp_dir, "obs_store.mmap"))
    return store


@pytest.fixture
def mmcontext_module():
    """MMContextModule backed by the conftest stubs (patched AutoModel/Tokenizer)."""
    return MMContextModule(
        model_name_or_path="bert-base-uncased",  # intercepted by conftest patch
        max_seq_length=8,
    )


@pytest.fixture
def adapter_module():
    """AdapterModule: TEXT_DIM → SHARED_DIM, OMICS_DIM → SHARED_DIM."""
    return AdapterModule(
        text_input_dim=TEXT_DIM,
        omics_input_dim=OMICS_DIM,
        shared_dim=SHARED_DIM,
        hidden_dim=32,
    )


@pytest.fixture
def attention_module():
    """OmicsAttentionModule matching OMICS_DIM."""
    return OmicsAttentionModule(
        input_dim=OMICS_DIM,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )


@pytest.fixture
def obs_pipeline(mmcontext_module, adapter_module):
    """Obs pipeline: MMContext → Adapter → Pooling → Normalize."""
    return SentenceTransformer(modules=[
        mmcontext_module,
        adapter_module,
        Pooling(embedding_dimension=SHARED_DIM, pooling_mode="mean"),
        Normalize(),
    ])


@pytest.fixture
def var_pipeline(mmcontext_module, attention_module, adapter_module):
    """Var pipeline: MMContext → OmicsAttention → Adapter → Pooling → Normalize."""
    return SentenceTransformer(modules=[
        mmcontext_module,
        attention_module,
        adapter_module,
        Pooling(embedding_dimension=SHARED_DIM, pooling_mode="mean"),
        Normalize(),
    ])


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------
class TestPipelineConstruction:
    """Modules compose into a valid SentenceTransformer."""

    def test_obs_pipeline_construction(self, obs_pipeline):
        """Obs pipeline constructs without error."""
        assert isinstance(obs_pipeline, SentenceTransformer)
        # Should have 4 modules: MMContext, Adapter, Pooling, Normalize
        modules = list(obs_pipeline.children())
        assert len(modules) == 4

    def test_var_pipeline_construction(self, var_pipeline):
        """Var pipeline constructs without error."""
        assert isinstance(var_pipeline, SentenceTransformer)
        # Should have 5 modules: MMContext, OmicsAttention, Adapter, Pooling, Normalize
        modules = list(var_pipeline.children())
        assert len(modules) == 5


# ---------------------------------------------------------------------------
# Encode — text
# ---------------------------------------------------------------------------
class TestEncodeText:
    """model.encode() with text inputs."""

    def test_encode_text_shape(self, obs_pipeline):
        """Encoding text produces embeddings of shape (N, SHARED_DIM)."""
        embeddings = obs_pipeline.encode(["Hello world", "Another sentence"])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, SHARED_DIM)

    def test_encode_text_single(self, obs_pipeline):
        """Single text input returns (SHARED_DIM,) array."""
        embedding = obs_pipeline.encode("Hello world")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (SHARED_DIM,)


# ---------------------------------------------------------------------------
# Encode — omics direct
# ---------------------------------------------------------------------------
class TestEncodeOmicsDirect:
    """model.encode() with direct omics vectors."""

    def test_encode_omics_direct_obs(self, obs_pipeline):
        """Encoding direct omics vector produces correct shape."""
        vec = np.random.randn(OMICS_DIM).astype(np.float32)
        embedding = obs_pipeline.encode([{"omics_values": vec}])
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, SHARED_DIM)

    def test_encode_omics_direct_var(self, var_pipeline):
        """Encoding var-level (multiple gene vectors) produces correct shape."""
        genes = [
            np.random.randn(OMICS_DIM).astype(np.float32)
            for _ in range(5)
        ]
        embedding = var_pipeline.encode([{"omics_values": genes}])
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, SHARED_DIM)


# ---------------------------------------------------------------------------
# Encode — omics via VectorStore
# ---------------------------------------------------------------------------
class TestEncodeOmicsViaStore:
    """model.encode() with prefixed IDs resolved through VectorStore."""

    def test_encode_omics_via_store(self, obs_pipeline, obs_store):
        """Prefixed IDs resolved through VectorStore produce correct shape."""
        # Attach store to the MMContextModule (first module)
        first_module = list(obs_pipeline.children())[0]
        first_module.set_vector_store(obs_store)

        embeddings = obs_pipeline.encode(["omics:cell_0", "omics:cell_1"])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, SHARED_DIM)

        first_module.remove_vector_store()


# ---------------------------------------------------------------------------
# Encode — normalization
# ---------------------------------------------------------------------------
class TestEncodeNormalize:
    """Output vectors should be L2-normalized (Normalize module is last)."""

    def test_encode_normalize(self, obs_pipeline):
        """Encoded vectors have unit L2 norm."""
        embeddings = obs_pipeline.encode(["A sample text", "Another text"])
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_encode_omics_normalize(self, obs_pipeline):
        """Omics embeddings also have unit L2 norm."""
        vec = np.random.randn(OMICS_DIM).astype(np.float32)
        embedding = obs_pipeline.encode([{"omics_values": vec}])
        norm = np.linalg.norm(embedding)
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# max_seq_length propagation
# ---------------------------------------------------------------------------
class TestMaxSeqLength:
    """max_seq_length accessible from SentenceTransformer level."""

    def test_max_seq_length_propagation(self, obs_pipeline):
        """max_seq_length from MMContextModule is accessible on the ST model."""
        assert obs_pipeline.max_seq_length == 8


# ---------------------------------------------------------------------------
# Obs pipeline end-to-end
# ---------------------------------------------------------------------------
class TestObsPipeline:
    """Full obs pipeline: [MMContext, Adapter, Pooling, Normalize]."""

    def test_obs_text_and_omics_same_output_dim(self, obs_pipeline):
        """Text and omics inputs produce same dimensionality."""
        text_emb = obs_pipeline.encode(["Some text"])
        omics_emb = obs_pipeline.encode([{
            "omics_values": np.random.randn(OMICS_DIM).astype(np.float32)
        }])
        assert text_emb.shape[-1] == omics_emb.shape[-1] == SHARED_DIM

    def test_obs_deterministic(self, obs_pipeline):
        """Same input produces same output (model in eval mode)."""
        emb1 = obs_pipeline.encode(["Determinism test"])
        emb2 = obs_pipeline.encode(["Determinism test"])
        np.testing.assert_allclose(emb1, emb2)


# ---------------------------------------------------------------------------
# Var pipeline end-to-end
# ---------------------------------------------------------------------------
class TestVarPipeline:
    """Full var pipeline: [MMContext, OmicsAttn, Adapter, Pooling, Normalize]."""

    def test_var_single_gene(self, var_pipeline):
        """Single gene vector (like obs) works in var pipeline too."""
        genes = [np.random.randn(OMICS_DIM).astype(np.float32)]
        embedding = var_pipeline.encode([{"omics_values": genes}])
        assert embedding.shape == (1, SHARED_DIM)

    def test_var_multiple_genes(self, var_pipeline):
        """Multiple gene vectors are attended and pooled."""
        genes = [
            np.random.randn(OMICS_DIM).astype(np.float32)
            for _ in range(10)
        ]
        embedding = var_pipeline.encode([{"omics_values": genes}])
        assert embedding.shape == (1, SHARED_DIM)

    def test_var_text_passthrough(self, var_pipeline):
        """Text inputs work through var pipeline (attention is no-op on text)."""
        embedding = var_pipeline.encode("Some text through var pipeline")
        assert embedding.shape == (SHARED_DIM,)


# ---------------------------------------------------------------------------
# Save / Load full pipeline
# ---------------------------------------------------------------------------
class TestSaveLoad:
    """Save and load full pipeline roundtrip."""

    def test_save_load_full_pipeline(self, obs_pipeline, tmp_dir, real_safetensors):
        """Saved pipeline can be loaded and produces same output."""
        obs_pipeline.eval()

        # Encode before save
        text_input = ["Test save load roundtrip"]
        original_emb = obs_pipeline.encode(text_input)

        # Save
        save_path = os.path.join(tmp_dir, "test_model")
        obs_pipeline.save(save_path)

        # Verify modules.json exists
        modules_json_path = os.path.join(save_path, "modules.json")
        assert os.path.isfile(modules_json_path)

        # Load and re-encode
        loaded = SentenceTransformer(save_path, trust_remote_code=True)
        loaded.eval()
        loaded_emb = loaded.encode(text_input)

        np.testing.assert_allclose(original_emb, loaded_emb, atol=1e-5)

    def test_modules_json_structure(self, obs_pipeline, tmp_dir, real_safetensors):
        """modules.json contains correct module chain."""
        save_path = os.path.join(tmp_dir, "test_model")
        obs_pipeline.save(save_path)

        with open(os.path.join(save_path, "modules.json")) as f:
            modules_config = json.load(f)

        assert len(modules_config) == 4

        # Check types contain our custom module classes
        types = [m["type"] for m in modules_config]
        assert any("MMContextModule" in t for t in types)
        assert any("AdapterModule" in t for t in types)
        assert any("Pooling" in t for t in types)
        assert any("Normalize" in t for t in types)

    def test_save_load_var_pipeline(self, var_pipeline, tmp_dir, real_safetensors):
        """Var pipeline save/load roundtrip."""
        var_pipeline.eval()

        save_path = os.path.join(tmp_dir, "test_var_model")
        var_pipeline.save(save_path)

        with open(os.path.join(save_path, "modules.json")) as f:
            modules_config = json.load(f)

        assert len(modules_config) == 5
        types = [m["type"] for m in modules_config]
        assert any("OmicsAttentionModule" in t for t in types)


# ---------------------------------------------------------------------------
# Precision conversion
# ---------------------------------------------------------------------------
class TestPrecisionConversion:
    """fp32 → fp16 works across all modules."""

    def test_precision_conversion_parameters(self, mmcontext_module, adapter_module):
        """Pipeline modules can be converted to fp16."""
        pipeline = SentenceTransformer(modules=[
            mmcontext_module,
            adapter_module,
            Pooling(embedding_dimension=SHARED_DIM, pooling_mode="mean"),
            Normalize(),
        ])
        pipeline.half()

        # All learnable parameters should now be fp16
        for name, param in pipeline.named_parameters():
            assert param.dtype == torch.float16, (
                f"Parameter {name} is {param.dtype}, expected float16"
            )

        # Restore to fp32 — the session-scoped _TextEncStub (from conftest)
        # is shared across all tests; leaving it in fp16 would pollute
        # subsequent tests that expect fp32 weights.
        pipeline.float()

    def test_precision_conversion_omics_encode(self):
        """fp16 pipeline produces valid omics embeddings.

        Skipped: the stub text encoder always returns fp32 tensors
        regardless of module dtype, causing a dtype mismatch when the
        fp32 input hits fp16 Linear weights. Real encoders handle
        autocasting properly, but stubs don't.
        """
        pytest.skip("Stub encoder produces fp32 regardless of model dtype — not testable without real encoder")


# ---------------------------------------------------------------------------
# Training with SentenceTransformerTrainer + MNR loss
# ---------------------------------------------------------------------------
class TestTraining:
    """Training integration with SentenceTransformerTrainer.

    Uses MultipleNegativesRankingLoss with (anchor, positive) pairs.
    Three modes mirror real usage:

    1. Text-only — both columns are plain text.
    2. Bimodal — anchors are omics (via VectorStore), positives are text.
    3. Gene-list — anchors are gene-name strings (treated as text), positives are text.
    """

    def test_training_text_only(self, obs_pipeline, tmp_dir):
        """One training step with text-only (anchor, positive) pairs."""
        from datasets import Dataset
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

        ds = Dataset.from_dict({
            "anchor": [
                "A neuron from the thalamus.",
                "An epithelial cell from the lung.",
                "A B cell from peripheral blood.",
                "A fibroblast from skin tissue.",
            ],
            "positive": [
                "Thalamic neuron expressing SYT1 and GNAS.",
                "Lung epithelial cell with high EPCAM.",
                "CD19-positive B lymphocyte.",
                "Dermal fibroblast with COL1A1 expression.",
            ],
        })

        loss = MultipleNegativesRankingLoss(obs_pipeline)
        args = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(tmp_dir, "train_text"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            learning_rate=1e-3,
            no_cuda=True,
            report_to="none",
        )
        trainer = SentenceTransformerTrainer(
            model=obs_pipeline,
            args=args,
            train_dataset=ds,
            loss=loss,
        )

        # Snapshot parameters before training
        params_before = {n: p.clone() for n, p in obs_pipeline.named_parameters() if p.requires_grad}

        trainer.train()

        # At least some parameters should have changed
        total_delta = sum(
            (p - params_before[n]).abs().sum().item()
            for n, p in obs_pipeline.named_parameters()
            if n in params_before
        )
        assert total_delta > 0, "No parameters changed during text-only training"

    def test_training_bimodal(self, obs_pipeline, obs_store, tmp_dir):
        """One training step with omics anchors + text positives.

        Mimics the real dataset structure: sample_idx column is prefixed
        with the omics prefix, positive column is plain text.
        """
        from datasets import Dataset
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

        # Attach VectorStore
        first_module = list(obs_pipeline.children())[0]
        first_module.set_vector_store(obs_store)

        ds = Dataset.from_dict({
            "anchor": [
                "omics:cell_0",
                "omics:cell_1",
                "omics:cell_2",
                "omics:cell_3",
            ],
            "positive": [
                "Thalamic neuron expressing SYT1.",
                "Lung epithelial cell with EPCAM.",
                "CD19-positive B lymphocyte.",
                "Dermal fibroblast with COL1A1.",
            ],
        })

        loss = MultipleNegativesRankingLoss(obs_pipeline)
        args = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(tmp_dir, "train_bimodal"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            learning_rate=1e-3,
            no_cuda=True,
            report_to="none",
        )
        trainer = SentenceTransformerTrainer(
            model=obs_pipeline,
            args=args,
            train_dataset=ds,
            loss=loss,
        )

        params_before = {n: p.clone() for n, p in obs_pipeline.named_parameters() if p.requires_grad}

        trainer.train()

        total_delta = sum(
            (p - params_before[n]).abs().sum().item()
            for n, p in obs_pipeline.named_parameters()
            if n in params_before
        )
        assert total_delta > 0, "No parameters changed during bimodal training"

        first_module.remove_vector_store()

    def test_training_gene_list(self, obs_pipeline, tmp_dir):
        """One training step with gene-name strings as anchors.

        Gene-name lists (like cell_sentence_1 in the real dataset) are
        plain text — they go through the text tokenization path.
        """
        from datasets import Dataset
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

        ds = Dataset.from_dict({
            "anchor": [
                "MALAT1 MT-CO3 GNAS SYT1 CALM1",
                "EPCAM KRT8 KRT18 MUC1",
                "CD19 MS4A1 CD79A PAX5",
                "COL1A1 COL3A1 FN1 VIM",
            ],
            "positive": [
                "Thalamic neuron expressing SYT1.",
                "Lung epithelial cell with EPCAM.",
                "CD19-positive B lymphocyte.",
                "Dermal fibroblast with COL1A1.",
            ],
        })

        loss = MultipleNegativesRankingLoss(obs_pipeline)
        args = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(tmp_dir, "train_genelist"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            learning_rate=1e-3,
            no_cuda=True,
            report_to="none",
        )
        trainer = SentenceTransformerTrainer(
            model=obs_pipeline,
            args=args,
            train_dataset=ds,
            loss=loss,
        )

        params_before = {n: p.clone() for n, p in obs_pipeline.named_parameters() if p.requires_grad}

        trainer.train()

        total_delta = sum(
            (p - params_before[n]).abs().sum().item()
            for n, p in obs_pipeline.named_parameters()
            if n in params_before
        )
        assert total_delta > 0, "No parameters changed during gene-list training"

    def test_training_save_load_roundtrip(self, obs_pipeline, tmp_dir, real_safetensors):
        """Trained model can be saved, reloaded, and produces identical encodings."""
        from datasets import Dataset
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

        ds = Dataset.from_dict({
            "anchor": [
                "MALAT1 MT-CO3 GNAS SYT1",
                "EPCAM KRT8 KRT18 MUC1",
                "CD19 MS4A1 CD79A PAX5",
                "COL1A1 COL3A1 FN1 VIM",
            ],
            "positive": [
                "Thalamic neuron expressing SYT1.",
                "Lung epithelial cell with EPCAM.",
                "CD19-positive B lymphocyte.",
                "Dermal fibroblast with COL1A1.",
            ],
        })

        loss = MultipleNegativesRankingLoss(obs_pipeline)
        save_path = os.path.join(tmp_dir, "trained_model")
        args = SentenceTransformerTrainingArguments(
            output_dir=save_path,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            learning_rate=1e-3,
            no_cuda=True,
            report_to="none",
        )
        trainer = SentenceTransformerTrainer(
            model=obs_pipeline,
            args=args,
            train_dataset=ds,
            loss=loss,
        )
        trainer.train()

        # Encode with trained model
        obs_pipeline.eval()
        test_inputs = ["Test sentence after training."]
        trained_emb = obs_pipeline.encode(test_inputs)

        # Save and reload
        obs_pipeline.save(save_path)
        loaded = SentenceTransformer(save_path)
        loaded.eval()
        loaded_emb = loaded.encode(test_inputs)

        np.testing.assert_allclose(trained_emb, loaded_emb, atol=1e-5)
