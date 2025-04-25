# test_mm_context_encoder.py

import json
import logging
import os
from pathlib import Path
from unittest.mock import Mock, patch

import datasets
import numpy as np
import pytest
import torch
from torch import Tensor

# Make sure these imports match your actual module paths
from mmcontext.models.MMContextEncoder import MMContextEncoder, MMContextProcessor

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# fast dummy stubs (shared by all tests)
# --------------------------------------------------------------------- #
class _TokStub:
    def __call__(self, texts, padding=True, **kw):
        b = len(texts)
        maxlen = 8
        ids = torch.zeros(b, maxlen, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": ids}


class _EncStub(torch.nn.Module):
    def __init__(self, hidden=32, model_type="bert"):
        super().__init__()
        self.config = type("cfg", (), {"hidden_size": hidden})()

        # This dummy_param should be registered properly but isn't important for tests
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        # Create structure based on model type
        if model_type == "bert":
            # BERT-like structure with real nn.Module to ensure proper parameter registration
            self.encoder = torch.nn.Module()
            self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(hidden, hidden) for _ in range(3)])
        elif model_type == "roberta":
            # RoBERTa-like structure with real nn.Module to ensure proper parameter registration
            self.roberta = torch.nn.Module()
            self.roberta.encoder = torch.nn.Module()
            self.roberta.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(hidden, hidden) for _ in range(3)])
        else:
            # Unsupported structure for testing warnings
            pass

    def forward(self, input_ids=None, **kw):
        b = input_ids.size(0)
        return type("o", (), {"pooler_output": torch.zeros(b, self.config.hidden_size)})


# MiniOmicsStub that mimics the real MiniOmicsModel with embeddings attribute
class _MiniOmicsStub(torch.nn.Module):
    def __init__(self, matrix):
        super().__init__()
        if isinstance(matrix, np.ndarray):
            weight = torch.from_numpy(matrix)
        else:
            weight = matrix
        self.embeddings = torch.nn.Embedding(matrix.shape[0], matrix.shape[1], padding_idx=0)
        with torch.no_grad():
            self.embeddings.weight.copy_(weight)

    def forward(self, input_ids=None, **kw):
        hidden = self.embeddings(input_ids)  # (B, L, H)
        pooled = hidden[:, 0]  # (B, H)
        return type("o", (), {"pooler_output": pooled})

    @classmethod
    def from_numpy(cls, matrix):
        return cls(matrix)


# --------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------- #


@pytest.fixture(scope="function")
def dummy_dataset():
    """Create a small dummy dataset for testing."""
    return datasets.Dataset.from_dict(
        {
            "sample_idx": ["A1", "A2"],
            "data_representation": [
                np.ones(4, dtype="float32"),
                np.full(4, 2, dtype="float32"),
            ],
            "caption": [
                "One vector of ones.",
                "Two vector of twos.",
            ],
        }
    )


@pytest.fixture(scope="function")
def dummy_dataset_different_dim():
    """Create a dataset with a different embedding dimension."""
    return datasets.Dataset.from_dict(
        {
            "sample_idx": ["B1", "B2"],
            "data_representation": [
                np.ones(6, dtype="float32"),  # Different dimension (6 instead of 4)
                np.full(6, 2, dtype="float32"),
            ],
        }
    )


@pytest.fixture(scope="function")
def text_only_encoder():
    """Create a text-only encoder for testing."""
    # No need for patching, it's handled by the global fixture
    encoder = MMContextEncoder(
        text_encoder_name="bert-base-uncased",
        adapter_hidden_dim=8,
        adapter_output_dim=4,
    ).eval()
    return encoder


@pytest.fixture(scope="function")
def bimodal_encoder(text_only_encoder, dummy_dataset):
    """Create a bimodal encoder by registering numeric data."""
    # Register numeric data with the text-only encoder
    prefixed_dataset = text_only_encoder.register_numeric_ds(dummy_dataset, data_origin="pca")
    text_only_encoder.eval()
    # Check that data was added successfully
    assert text_only_encoder._has_omics
    assert text_only_encoder.omics_encoder is not None

    # Return both the encoder and the prefixed dataset for use in tests
    return text_only_encoder, prefixed_dataset


@pytest.fixture(scope="function")
def no_adapter_encoder():
    """Create an encoder without adapter layers."""
    # No need for patching, it's handled by the global fixture
    encoder = MMContextEncoder(
        text_encoder_name="bert-base-uncased",
        adapter_hidden_dim=None,  # No adapter
    ).eval()
    return encoder


# --------------------------------------------------------------------- #
# Global patches for test efficiency
# --------------------------------------------------------------------- #
@pytest.fixture(scope="session", autouse=True)
def patch_model_loading():
    """Patch model loading functions for the entire test session."""
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=_EncStub(model_type="bert")),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_TokStub()),
        patch("mmcontext.models.MMContextEncoder.MiniOmicsModel", _MiniOmicsStub),
        patch("safetensors.torch.load_model", return_value=None),
    ):
        # This patch will be active for the entire test session
        yield


# --------------------------------------------------------------------- #
# Core functionality tests
# --------------------------------------------------------------------- #
def test_text_only_encoder(text_only_encoder):
    """Test that a text-only encoder works correctly."""
    encoder = text_only_encoder

    # Test tokenization
    texts = ["This is a test", "Another test"]
    features = encoder.tokenize(texts)

    # Check that the features have the right keys
    assert "input_ids" in features
    assert "attention_mask" in features
    assert "omics_text_info" in features
    assert torch.all(features["omics_text_info"] == 1)  # All text

    # Test forward pass
    out = encoder(features)

    # Check that we get the correct output shape
    assert isinstance(out, dict)
    assert "sentence_embedding" in out
    assert out["sentence_embedding"].shape == (2, 4)  # batch_size=2, output_dim=4


def test_bimodal_encoder(bimodal_encoder):
    """Test that a bimodal encoder works correctly with mixed input."""
    encoder, prefixed_dataset = bimodal_encoder

    # Get a prefixed ID from the dataset for testing
    prefixed_id = prefixed_dataset["prefixed_id"][0]

    # Mixed batch: one omics, one text
    features = encoder.tokenize([prefixed_id, "This is a test"])

    # Check that the features have the right keys
    assert "input_ids" in features
    assert "attention_mask" in features
    assert "omics_ids" in features
    assert "omics_text_info" in features
    assert features["omics_text_info"][0] == 0  # Omics
    assert features["omics_text_info"][1] == 1  # Text

    # Test forward pass
    out = encoder(features)

    # Check that we get the correct output
    assert isinstance(out, dict)
    assert "sentence_embedding" in out
    assert out["sentence_embedding"].shape == (2, 4)  # batch_size=2, output_dim=4

    # Make sure the embeddings are different (omics vs text should be different)
    assert not torch.allclose(out["sentence_embedding"][0], out["sentence_embedding"][1]), (
        "Text and omics embeddings should be different"
    )


def test_encoder_return_tensor(bimodal_encoder):
    """Test that the encoder can return tensors directly."""
    encoder, prefixed_dataset = bimodal_encoder

    # Get a prefixed ID from the dataset for testing
    prefixed_id = prefixed_dataset["prefixed_id"][0]

    # Mixed batch with return_tensor=True
    features = encoder.tokenize([prefixed_id, "This is a test"])
    features["return_tensor"] = True

    # Test forward pass
    out = encoder.forward(features)

    # Check that we get a tensor directly
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 4)  # batch_size=2, output_dim=4


def test_no_adapter_encoder(no_adapter_encoder):
    """Test that an encoder without adapters works correctly."""
    encoder = no_adapter_encoder

    # Test with text
    features = encoder.tokenize(["This is a test"])
    out = encoder.forward(features)

    # Check output dimension matches text encoder hidden size
    assert out["sentence_embedding"].shape[1] == 32  # Hidden size from _EncStub


def test_adding_numeric_data(text_only_encoder, dummy_dataset):
    """Test that we can add numeric data to a text-only encoder."""
    encoder = text_only_encoder

    # Initially it's text-only
    assert not hasattr(encoder, "_has_omics") or not encoder._has_omics

    # Register numeric data
    prefixed_dataset = encoder.register_numeric_ds(dummy_dataset, data_origin="pca")

    # Check that data was added correctly
    assert encoder._has_omics
    assert encoder.omics_encoder is not None
    assert encoder.omics_adapter is not None

    # Verify that the omics encoder has the embedding attribute
    assert hasattr(encoder.omics_encoder, "embeddings")
    assert hasattr(encoder.omics_encoder.embeddings, "weight")

    # Check that the returned dataset has prefixed IDs
    assert "prefixed_id" in prefixed_dataset.column_names
    assert prefixed_dataset["prefixed_id"][0].startswith("sample_idx:")

    # Now test with mixed input using the prefixed ID
    prefixed_id = prefixed_dataset["prefixed_id"][0]
    features = encoder.tokenize([prefixed_id, "This is a test"])

    # Check that the features have the right keys
    assert "omics_ids" in features
    assert features["omics_text_info"][0] == 0  # Omics

    encoder.eval()
    # Test forward pass
    out = encoder(features)
    assert out["sentence_embedding"].shape == (2, 4)


def test_adding_incompatible_data(bimodal_encoder, dummy_dataset_different_dim):
    """Test that adding data with incompatible dimensions raises an error."""
    encoder, _ = bimodal_encoder

    # Adding data with different dimensions should fail
    with pytest.raises(ValueError, match="Dimension mismatch"):
        encoder.register_numeric_ds(dummy_dataset_different_dim, data_origin="pca")


def test_registered_data_origin_validation():
    """Test that data type validation works correctly."""
    # No need for patching, it's handled by the global fixture

    # Invalid data type should raise error
    with pytest.raises(ValueError, match="registered_data_origin must be one of"):
        MMContextEncoder(text_encoder_name="bert-base-uncased", registered_data_origin="invalid_type")

    # Valid data type should work
    encoder = MMContextEncoder(text_encoder_name="bert-base-uncased", registered_data_origin="pca")
    assert encoder.registered_data_origin == "pca"


def test_registered_data_origin_consistency(text_only_encoder, dummy_dataset):
    """Test that data type consistency is enforced when registering data."""
    encoder = text_only_encoder

    # First registration requires registered_data_origin
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        encoder.register_numeric_ds(dummy_dataset)

    # Register with valid data type
    _prefixed_dataset = encoder.register_numeric_ds(dummy_dataset, data_origin="pca")
    assert encoder.registered_data_origin == "pca"

    # Try to register different type
    with pytest.raises(ValueError, match="Cannot register data of type"):
        encoder.register_numeric_ds(dummy_dataset, data_origin="hvg")


def test_save_load_without_omics_matrix(text_only_encoder, dummy_dataset, tmp_path):
    """Test that omics matrix is not saved but data type is preserved."""
    encoder = text_only_encoder

    # Register data
    _prefixed_dataset = encoder.register_numeric_ds(dummy_dataset, data_origin="scvi_fm")

    # Save the model
    save_dir = tmp_path / "encoder_without_matrix"
    encoder.save(str(save_dir))

    # Check config file contains data type but not matrix
    config_path = save_dir / "config.json"
    assert config_path.exists()
    with open(config_path) as f:
        config = json.load(f)
    assert config["registered_data_origin"] == "scvi_fm"
    assert "omics_embedding" not in config

    # Load the model
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=_EncStub()),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_TokStub()),
    ):
        loaded_encoder = MMContextEncoder.load(str(save_dir))

    # Check data type is preserved but omics not initialized
    assert loaded_encoder.registered_data_origin == "scvi_fm"
    assert not loaded_encoder._has_omics

    # Register same type of data
    _prefixed_dataset = loaded_encoder.register_numeric_ds(
        dummy_dataset,
        data_origin="scvi_fm",  # Same type works
    )

    # Try to register different type
    with pytest.raises(ValueError, match="Cannot register data of type"):
        loaded_encoder.register_numeric_ds(
            dummy_dataset,
            data_origin="geneformer",  # Different type fails
        )


def test_registered_input_dim():
    """Test that registered_input_dim is handled correctly."""
    # No need for patching, it's handled by the global fixture

    # Create model with registered input dim
    encoder = MMContextEncoder(
        text_encoder_name="bert-base-uncased",
        registered_data_origin="pca",
        registered_input_dim=100,
        adapter_hidden_dim=64,
        adapter_output_dim=32,
    )

    # Check that adapter was initialized correctly
    assert encoder.registered_input_dim == 100
    assert encoder.omics_adapter is not None
    assert next(encoder.omics_adapter.parameters()).shape[1] == 100  # Input dim


def test_registered_input_dim_consistency(text_only_encoder, tmp_path):
    """Test that registered_input_dim is preserved and enforced."""
    encoder = text_only_encoder

    # Create test data with specific dimension
    test_data = datasets.Dataset.from_dict(
        {
            "sample_idx": ["A1"],
            "data_representation": [np.ones(50, dtype="float32")],
        }
    )

    # Register data
    _prefixed_dataset = encoder.register_numeric_ds(test_data, data_origin="pca")

    assert encoder.registered_input_dim == 50

    # Save the model
    save_dir = tmp_path / "encoder_with_dim"
    encoder.save(str(save_dir))

    # Check config contains input dim
    with open(save_dir / "config.json") as f:
        config = json.load(f)
    assert config["registered_input_dim"] == 50

    # Load the model
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=_EncStub()),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_TokStub()),
    ):
        loaded_encoder = MMContextEncoder.load(str(save_dir))

    assert loaded_encoder.registered_input_dim == 50

    # Try to register data with wrong dimension
    wrong_dim_data = datasets.Dataset.from_dict(
        {
            "sample_idx": ["B1"],
            "data_representation": [np.ones(60, dtype="float32")],  # Wrong dimension
        }
    )

    with pytest.raises(ValueError, match="Input dimension mismatch"):
        loaded_encoder.register_numeric_ds(wrong_dim_data, data_origin="pca")


def test_safetensors_exclude_embeddings(text_only_encoder, dummy_dataset, tmp_path):
    """Test that omics embeddings are excluded when using safetensors."""
    encoder = text_only_encoder

    # Register data - no need for additional patching
    _prefixed_dataset = encoder.register_numeric_ds(dummy_dataset, data_origin="pca")

    # Save with safetensors - no need for additional patching
    save_dir = tmp_path / "encoder_safetensors"
    encoder.save(str(save_dir), safe_serialization=True)

    # Verify safetensors file exists
    safetensors_path = save_dir / "model.safetensors"
    assert safetensors_path.exists()

    # Load - no need for additional patching
    loaded_encoder = MMContextEncoder.load(str(save_dir))

    # Check that omics encoder is not initialized but adapter is
    assert loaded_encoder.omics_encoder is None
    assert loaded_encoder.omics_adapter is not None
    assert loaded_encoder.registered_input_dim is not None

    # Register same data type and verify adapter works
    prefixed_dataset = loaded_encoder.register_numeric_ds(dummy_dataset, data_origin="pca")

    # Verify we can use the model
    prefixed_id = prefixed_dataset["prefixed_id"][0]
    features = loaded_encoder.tokenize([prefixed_id, "This is a test"])
    loaded_encoder.eval()
    out = loaded_encoder(features)
    assert "sentence_embedding" in out


def test_model_dtype_conversion():
    """Test converting model dtype after registration using PyTorch's to() method."""
    # Create dataset
    ds = datasets.Dataset.from_dict(
        {
            "sample_idx": ["X1", "X2"],
            "data_representation": [
                np.ones(4, dtype="float32"),
                np.ones(4, dtype="float32") * 2,
            ],
        }
    )

    # Create model and register data
    encoder = MMContextEncoder(
        text_encoder_name="bert-base-uncased",
        adapter_hidden_dim=16,
        adapter_output_dim=8,
    )

    prefixed_ds = encoder.register_numeric_ds(ds, data_origin="scvi_fm")
    encoder.eval()
    # Check original dtype
    assert encoder.omics_encoder.embeddings.weight.dtype == torch.float32

    # Test in float32 first to capture input and outputs
    prefixed_id = prefixed_ds["prefixed_id"][0]
    features = encoder.tokenize([prefixed_id, "This is a test"])

    # Use return_tensor=False to get dict with intermediate outputs
    features["return_tensor"] = False
    out_dict = encoder(features)
    assert out_dict["sentence_embedding"].dtype == torch.float32

    # Convert to half precision
    encoder.half()  # Equivalent to encoder.to(torch.float16)

    # Verify all components converted
    assert encoder.omics_encoder.embeddings.weight.dtype == torch.float16
    assert next(encoder.omics_adapter.parameters()).dtype == torch.float16
    assert next(encoder.text_adapter.parameters()).dtype == torch.float16

    # Test forward pass with converted model
    features = encoder.tokenize([prefixed_id, "This is a test"])

    # Test with return_tensor=True
    features["return_tensor"] = True
    out_tensor = encoder(features)
    assert out_tensor.dtype == torch.float16

    # Test with return_tensor=False
    features["return_tensor"] = False
    out_dict = encoder(features)
    assert out_dict["sentence_embedding"].dtype == torch.float16

    # Test with both text and omics inputs
    features = encoder.tokenize([prefixed_id, "This is a test"])
    out = encoder(features)
    # Should have consistent dtype regardless of input type
    assert out["sentence_embedding"].dtype == torch.float16

    # Register more data to ensure it works after conversion
    more_ds = datasets.Dataset.from_dict(
        {
            "sample_idx": ["X3", "X4"],
            "data_representation": [
                np.ones(4, dtype="float32") * 3,
                np.ones(4, dtype="float32") * 4,
            ],
        }
    )

    more_prefixed = encoder.register_numeric_ds(more_ds, data_origin="scvi_fm")

    # Verify newly registered data works and maintains float16
    new_id = more_prefixed["prefixed_id"][0]
    features = encoder.tokenize([new_id, "Another test"])
    out = encoder(features)
    assert out["sentence_embedding"].dtype == torch.float16
    encoder.to(torch.float32)  # Reset to float32 for other tests


# --------------------------------------------------------------------- #
# Freezing tests
# --------------------------------------------------------------------- #
def test_full_freezing():
    """Test that freezing the text encoder works correctly."""
    # Create a fresh stub and verify its parameters are initially trainable
    enc_stub = _EncStub(model_type="bert")
    for p in enc_stub.parameters():
        assert p.requires_grad, "Stub parameters should start with requires_grad=True"

    # Test with BERT-like model
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=enc_stub),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_TokStub()),
    ):
        # Create model with frozen text encoder
        model_frozen = MMContextEncoder(
            text_encoder_name="bert-base-uncased",
            freeze_text_encoder=True,
            unfreeze_last_n_layers=0,
        )

        # Verify we have parameters in the layers
        layer_params = list(model_frozen.text_encoder.encoder.layer.parameters())
        assert len(layer_params) > 0, "No parameters found in encoder layers"

        # All text encoder params should be frozen
        assert all(not p.requires_grad for p in model_frozen.text_encoder.parameters())


def test_partial_unfreezing():
    """Test that partial unfreezing works correctly."""
    # Create a fresh stub and verify its parameters are initially trainable
    enc_stub = _EncStub(model_type="bert")
    for p in enc_stub.parameters():
        assert p.requires_grad, "Stub parameters should start with requires_grad=True"

    # Test with BERT-like model
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=enc_stub),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_TokStub()),
    ):
        # Create model with unfrozen last layer
        model_partial = MMContextEncoder(
            text_encoder_name="bert-base-uncased",
            freeze_text_encoder=True,
            unfreeze_last_n_layers=1,
        )

        # The last layer should be trainable, others frozen
        last_layer_params = list(model_partial.text_encoder.encoder.layer[-1].parameters())
        assert len(last_layer_params) > 0, "No parameters found in last layer"
        assert all(p.requires_grad for p in last_layer_params), "Last layer should be unfrozen"

        other_layers_params = []
        for i in range(len(model_partial.text_encoder.encoder.layer) - 1):
            other_layers_params.extend(list(model_partial.text_encoder.encoder.layer[i].parameters()))
        assert len(other_layers_params) > 0, "No parameters found in other layers"
        assert all(not p.requires_grad for p in other_layers_params), "Other layers should be frozen"


def test_no_freezing():
    """Test that no freezing works correctly."""
    # Create a fresh stub and verify its parameters are initially trainable
    enc_stub = _EncStub(model_type="bert")
    for p in enc_stub.parameters():
        assert p.requires_grad, "Stub parameters should start with requires_grad=True"

    # Test with BERT-like model
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=enc_stub),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_TokStub()),
    ):
        # Create model with unfrozen text encoder
        model_unfrozen = MMContextEncoder(
            text_encoder_name="bert-base-uncased",
            freeze_text_encoder=False,
        )

        # Print parameters for debugging
        print("\nParameters in unfrozen model:")
        for i, p in enumerate(model_unfrozen.text_encoder.parameters()):
            print(f"  Param {i}: requires_grad={p.requires_grad}")

        # All text encoder params should be trainable
        all_params = list(model_unfrozen.text_encoder.parameters())
        assert len(all_params) > 0, "No parameters found in text encoder"
        assert all(p.requires_grad for p in all_params), (
            "All parameters should be trainable when freeze_text_encoder=False"
        )


def test_freezing_unfreezing_roberta():
    """Test that freezing and unfreezing work correctly for RoBERTa-like models."""
    # Create a fresh stub and verify its parameters are initially trainable
    rob_stub = _EncStub(model_type="roberta")
    for p in rob_stub.parameters():
        assert p.requires_grad, "Stub parameters should start with requires_grad=True"

    with (
        patch("transformers.AutoModel.from_pretrained", return_value=rob_stub),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_TokStub()),
    ):
        # Create model with unfrozen last layer
        model_partial = MMContextEncoder(
            text_encoder_name="roberta-base",
            freeze_text_encoder=True,
            unfreeze_last_n_layers=1,
        )

        # Verify we have parameters in the layers
        layer_params = list(model_partial.text_encoder.roberta.encoder.layer.parameters())
        assert len(layer_params) > 0, "No parameters found in roberta encoder layers"

        # The last layer should be trainable, others frozen
        last_layer_params = list(model_partial.text_encoder.roberta.encoder.layer[-1].parameters())
        assert len(last_layer_params) > 0, "No parameters found in last layer"
        assert all(p.requires_grad for p in last_layer_params), "Last layer should be unfrozen"

        other_layers_params = []
        for i in range(len(model_partial.text_encoder.roberta.encoder.layer) - 1):
            other_layers_params.extend(list(model_partial.text_encoder.roberta.encoder.layer[i].parameters()))
        assert len(other_layers_params) > 0, "No parameters found in other layers"
        assert all(not p.requires_grad for p in other_layers_params), "Other layers should be frozen"


def test_freezing_unsupported_architecture(caplog):
    """Test that a warning is logged for unsupported architecture."""
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=_EncStub(model_type="unsupported")),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_TokStub()),
    ):
        # Create model with unfrozen last layer for unsupported architecture
        _model_partial = MMContextEncoder(
            text_encoder_name="unsupported-model",
            freeze_text_encoder=True,
            unfreeze_last_n_layers=1,
        )

        # Check that a warning was logged
        assert any("Unsupported architecture" in record.message for record in caplog.records)


# --------------------------------------------------------------------- #
# Adapter weight tests
# --------------------------------------------------------------------- #
def test_adapter_weights_preserved(text_only_encoder, tmp_path):
    """Test that adapter weights are preserved after save and load."""
    encoder = text_only_encoder

    # Initial forward pass to initialize weights
    features = encoder.tokenize(["This is a test"])
    encoder(features)

    # Get initial weights
    orig_weight = next(encoder.text_adapter.parameters()).clone()

    # Modify a weight
    with torch.no_grad():
        for p in encoder.text_adapter.parameters():
            p.add_(torch.ones_like(p) * 0.1)

    # Verify weight changed
    modified_weight = next(encoder.text_adapter.parameters()).clone()
    assert not torch.allclose(orig_weight, modified_weight)

    # Save the model
    save_dir = tmp_path / "encoder_with_adapters"
    encoder.save(str(save_dir))

    # Load the model
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=_EncStub()),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_TokStub()),
    ):
        loaded_encoder = MMContextEncoder.load(str(save_dir))

    # Check that loaded weights match modified weights
    loaded_weight = next(loaded_encoder.text_adapter.parameters()).clone()
    assert torch.allclose(modified_weight, loaded_weight)


# --------------------------------------------------------------------- #
# Save/Load tests
# --------------------------------------------------------------------- #
def test_save_load_text_only(text_only_encoder, tmp_path):
    """Test saving and loading a text-only encoder."""
    encoder = text_only_encoder

    # Do a forward pass to initialize weights
    features = encoder.tokenize(["This is a test"])
    orig_out = encoder(features)

    # Save the model
    save_dir = tmp_path / "text_only_encoder"
    encoder.save(str(save_dir))

    # Check config file
    config_path = save_dir / "config.json"
    assert config_path.exists()

    # Load the model - no need for patching
    loaded_encoder = MMContextEncoder.load(str(save_dir))

    # Check that the outputs match
    loaded_features = loaded_encoder.tokenize(["This is a test"])
    loaded_encoder.eval()
    loaded_out = loaded_encoder(loaded_features)

    assert torch.allclose(orig_out["sentence_embedding"], loaded_out["sentence_embedding"])


def test_save_load_with_numeric_data(text_only_encoder, dummy_dataset, tmp_path):
    """Test saving and loading an encoder after adding numeric data."""
    encoder = text_only_encoder

    # Add numeric data - no need for patching
    _prefixed_dataset = encoder.register_numeric_ds(dummy_dataset, data_origin="pca")

    # Do a forward pass with mixed input
    features = encoder.tokenize(["sample_idx:A1", "This is a test"])
    encoder.eval()
    orig_out = encoder(features)

    # Capture parameter states before saving
    param_states = {}
    # Check text adapter weights
    text_adapter_params = list(encoder.text_adapter.parameters())
    param_states["text_adapter"] = [p.clone().detach() for p in text_adapter_params]

    # Check omics adapter weights
    omics_adapter_params = list(encoder.omics_adapter.parameters())
    param_states["omics_adapter"] = [p.clone().detach() for p in omics_adapter_params]

    # Check omics embeddings
    if hasattr(encoder.omics_encoder, "embeddings"):
        param_states["omics_embeddings"] = encoder.omics_encoder.embeddings.weight.clone().detach()

    # Save the model
    save_dir = tmp_path / "encoder_with_numeric"
    encoder.save(str(save_dir))

    # Check config file
    config_path = save_dir / "config.json"
    assert config_path.exists()
    with open(config_path) as f:
        _config = json.load(f)

    # Note key attributes before loading
    print("Original model attributes:")
    print(f"- registered_data_origin: {encoder._registered_data_origin}")
    print(f"- registered_input_dim: {encoder._registered_input_dim}")
    print(f"- has_omics: {encoder._has_omics}")

    # Load the model - no need for patching
    loaded_encoder = MMContextEncoder.load(str(save_dir))

    # Note key attributes after loading
    print("Loaded model attributes:")
    print(f"- registered_data_origin: {loaded_encoder._registered_data_origin}")
    print(f"- registered_input_dim: {loaded_encoder._registered_input_dim}")
    print(f"- has_omics: {loaded_encoder._has_omics}")

    # Re-register the same data to the loaded model
    loaded_encoder.register_numeric_ds(dummy_dataset, data_origin="pca")

    # Compare parameter counts
    print("Parameter comparison:")
    orig_params = sum(p.numel() for p in encoder.parameters())
    loaded_params = sum(p.numel() for p in loaded_encoder.parameters())
    print(f"- Original parameter count: {orig_params}")
    print(f"- Loaded parameter count: {loaded_params}")

    # Check text adapter parameters
    loaded_text_adapter_params = list(loaded_encoder.text_adapter.parameters())
    assert len(loaded_text_adapter_params) == len(param_states["text_adapter"]), "Text adapter parameter count mismatch"
    for i, (orig, loaded) in enumerate(zip(param_states["text_adapter"], loaded_text_adapter_params, strict=False)):
        assert torch.allclose(orig, loaded), f"Text adapter parameter {i} mismatch"

    # Check omics adapter parameters
    loaded_omics_adapter_params = list(loaded_encoder.omics_adapter.parameters())
    assert len(loaded_omics_adapter_params) == len(param_states["omics_adapter"]), (
        "Omics adapter parameter count mismatch"
    )
    for i, (orig, loaded) in enumerate(zip(param_states["omics_adapter"], loaded_omics_adapter_params, strict=False)):
        assert torch.allclose(orig, loaded), f"Omics adapter parameter {i} mismatch"

    # Check the loaded model outputs match for mixed input
    loaded_features = loaded_encoder.tokenize(["sample_idx:A1", "This is a test"])
    loaded_encoder.eval()
    loaded_out = loaded_encoder(loaded_features)

    assert torch.allclose(orig_out["sentence_embedding"], loaded_out["sentence_embedding"]), (
        "Output embeddings don't match after loading"
    )


def test_save_load_no_adapter(no_adapter_encoder, tmp_path):
    """Test saving and loading an encoder without adapters."""
    encoder = no_adapter_encoder

    # Do a forward pass
    features = encoder.tokenize(["This is a test"])
    orig_out = encoder(features)

    # Save the model
    save_dir = tmp_path / "no_adapter_encoder"
    encoder.save(str(save_dir))

    # Check config file
    config_path = save_dir / "config.json"
    assert config_path.exists()
    with open(config_path) as f:
        config = json.load(f)
    assert config["adapter_hidden_dim"] is None

    # Load the model
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=_EncStub()),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_TokStub()),
    ):
        loaded_encoder = MMContextEncoder.load(str(save_dir))

    # Check that the outputs match
    loaded_features = loaded_encoder.tokenize(["This is a test"])
    loaded_out = loaded_encoder(loaded_features)

    assert torch.allclose(orig_out["sentence_embedding"], loaded_out["sentence_embedding"])


# --------------------------------------------------------------------- #
# Error handling tests
# --------------------------------------------------------------------- #
def test_missing_columns(text_only_encoder):
    """Test error handling for missing columns in dataset."""
    # Create dataset with missing columns
    bad_dataset = datasets.Dataset.from_dict(
        {
            "wrong_id_col": ["A1", "A2"],
            "data_representation": [
                np.ones(4, dtype="float32"),
                np.full(4, 2, dtype="float32"),
            ],
        }
    )

    # Should raise KeyError due to missing id_col
    with pytest.raises(KeyError):
        text_only_encoder.register_numeric_ds(
            bad_dataset,
            id_col="sample_idx",  # This doesn't exist
            rep_col="data_representation",
            data_origin="pca",
        )

    # Dataset with missing representation column
    bad_dataset2 = datasets.Dataset.from_dict(
        {
            "sample_idx": ["A1", "A2"],
            "wrong_rep_col": [
                np.ones(4, dtype="float32"),
                np.full(4, 2, dtype="float32"),
            ],
        }
    )

    # Should raise KeyError due to missing rep_col
    with pytest.raises(KeyError):
        text_only_encoder.register_numeric_ds(
            bad_dataset2,
            id_col="sample_idx",
            rep_col="data_representation",  # This doesn't exist
            data_origin="pca",
        )


def test_omics_query_without_initialization(text_only_encoder):
    """Test error handling when trying to use omics without initialization."""
    # Should raise runtime error when trying to encode omics
    with pytest.raises(RuntimeError, match="not initialized"):
        features = text_only_encoder.tokenize(["This is ok", "This is ok too"])
        # Manually change omics_text_info to trick the model
        features["omics_text_info"][0] = 0  # Pretend this is omics
        text_only_encoder(features)


def test_inconsistent_numeric_types(text_only_encoder):
    """Test handling of inconsistent numeric data types."""
    # Create dataset with inconsistent dimensions
    bad_dataset = datasets.Dataset.from_dict(
        {
            "sample_idx": ["A1", "A2"],
            "data_representation": [
                np.ones(4, dtype="float32"),  # Good
                np.ones(5, dtype="float32"),  # Wrong dimension
            ],
        }
    )

    # Should raise ValueError due to inconsistent dimensions
    with pytest.raises(ValueError, match="Dimension mismatch"):
        text_only_encoder.register_numeric_ds(
            bad_dataset, id_col="sample_idx", rep_col="data_representation", data_origin="pca"
        )


# --------------------------------------------------------------------- #
# Legacy compatibility tests
# --------------------------------------------------------------------- #
def test_legacy_roundtrip_compatibility(bimodal_encoder):
    """Test compatibility with the old roundtrip test."""
    encoder, prefixed_dataset = bimodal_encoder

    # Get a prefixed ID from the dataset for testing
    prefixed_id = prefixed_dataset["prefixed_id"][0]

    # Mixed batch: one omics, one caption
    features = encoder.tokenize([prefixed_id, "This is a caption"])
    features["return_tensor"] = True  # Get tensor output directly
    out = encoder(features)

    assert isinstance(out, torch.Tensor) and out.shape == (2, 4)
    assert not torch.allclose(out[0], out[1]), "text & omics embeddings should differ"


# Add a test to check that prefixed_id column works correctly
def test_prefixed_id_column(text_only_encoder, dummy_dataset):
    """Test that register_numeric_ds returns dataset with prefixed_id column."""
    # Register numeric data
    enc = text_only_encoder
    prefixed_dataset = enc.register_numeric_ds(
        dummy_dataset,
        data_origin="pca",
        prefix="custom_prefix:",  # Custom prefix
    )

    # Check that the prefixed_id column was added
    assert "prefixed_id" in prefixed_dataset.column_names

    # Check that all IDs have the correct prefix
    for i in range(len(prefixed_dataset)):
        assert prefixed_dataset["prefixed_id"][i].startswith("custom_prefix:")
        assert prefixed_dataset["prefixed_id"][i].endswith(prefixed_dataset["sample_idx"][i])

    # Verify we can tokenize using the prefixed IDs
    prefixed_id = prefixed_dataset["prefixed_id"][0]
    features = enc.tokenize([prefixed_id])

    # Check that the features have the right keys for omics
    assert "omics_ids" in features
    assert features["omics_text_info"][0] == 0  # Omics

    # Test forward pass
    enc.eval()
    out = enc(features)
    assert "sentence_embedding" in out
    assert out["sentence_embedding"].shape == (1, 4)


def test_register_with_different_types(text_only_encoder):
    """Test registration with different data types."""
    # Dataset with list inputs
    list_dataset = datasets.Dataset.from_dict(
        {
            "sample_idx": ["L1", "L2"],
            "data_representation": [
                [1.0, 2.0, 3.0, 4.0],  # List input
                [5.0, 6.0, 7.0, 8.0],  # List input
            ],
        }
    )

    # Dataset with tensor inputs
    tensor_dataset = datasets.Dataset.from_dict(
        {
            "sample_idx": ["T1", "T2"],
            "data_representation": [
                torch.tensor([1.0, 2.0, 3.0, 4.0]),  # Tensor input
                torch.tensor([5.0, 6.0, 7.0, 8.0]),  # Tensor input
            ],
        }
    )

    # Register with list inputs
    encoder = text_only_encoder
    prefixed_list_ds = encoder.register_numeric_ds(list_dataset, data_origin="pca")

    # Verify registration worked
    assert encoder._has_omics
    assert encoder.omics_encoder is not None
    assert encoder.omics_encoder.embeddings.weight.dtype == torch.float32  # Default is float32

    # Test with mixed input
    prefixed_id = prefixed_list_ds["prefixed_id"][0]
    features = encoder.tokenize([prefixed_id, "This is a test"])
    encoder.eval()
    out = encoder(features)
    assert out["sentence_embedding"].shape == (2, 4)

    # Convert model to float16 using PyTorch's native method
    encoder2 = MMContextEncoder(
        text_encoder_name="bert-base-uncased",
        adapter_hidden_dim=8,
        adapter_output_dim=4,
    )

    # Register data first (in default float32)
    prefixed_tensor_ds = encoder2.register_numeric_ds(tensor_dataset, data_origin="pca")
    encoder2.eval()
    # Then convert the entire model to float16
    encoder2.to(torch.float16)

    # Verify model was converted to float16
    assert encoder2.omics_encoder.embeddings.weight.dtype == torch.float16
    assert next(encoder2.text_encoder.parameters()).dtype == torch.float16
    assert next(encoder2.omics_adapter.parameters()).dtype == torch.float16
    assert next(encoder2.text_adapter.parameters()).dtype == torch.float16

    # Test with mixed input - should work with consistent dtype
    prefixed_id = prefixed_tensor_ds["prefixed_id"][0]
    features = encoder2.tokenize([prefixed_id, "This is a test"])
    out = encoder2(features)
    assert out["sentence_embedding"].shape == (2, 4)
    assert out["sentence_embedding"].dtype == torch.float16  # Output is float16

    # Reset dtype back to float32 to avoid affecting other tests
    encoder2.to(torch.float32)


def test_preserve_registration_info_after_save_load(text_only_encoder, dummy_dataset, tmp_path):
    """Test that registered_data_origin and registered_input_dim are preserved after save/load."""
    encoder = text_only_encoder

    # Get the vector dimension from the first dataset entry
    input_dim = len(dummy_dataset["data_representation"][0])

    # Register data with a specific data type
    data_type = "geneformer"
    _prefixed_dataset = encoder.register_numeric_ds(
        dummy_dataset,
        data_origin=data_type,
    )

    # Verify registration attributes
    assert encoder.registered_data_origin == data_type
    assert encoder.registered_input_dim == input_dim
    assert encoder._has_omics

    # Get the adapter weights before saving
    adapter_weights = [p.clone().detach() for p in encoder.omics_adapter.parameters()]

    # Save the model
    save_dir = tmp_path / "registered_model"
    encoder.save(str(save_dir))

    # Check config file contains correct values
    config_path = save_dir / "config.json"
    assert config_path.exists()
    with open(config_path) as f:
        config = json.load(f)
    assert config["registered_data_origin"] == data_type
    assert config["registered_input_dim"] == input_dim

    # Load the model
    loaded_encoder = MMContextEncoder.load(str(save_dir))

    # Verify registration attributes are preserved in loaded model
    assert loaded_encoder.registered_data_origin == data_type
    assert loaded_encoder.registered_input_dim == input_dim
    assert not loaded_encoder._has_omics  # Should be False initially since embeddings aren't saved

    # Re-register the same data
    loaded_encoder.register_numeric_ds(dummy_dataset, data_origin=data_type)

    # Verify omics capabilities are restored with the same dimension
    assert loaded_encoder._has_omics
    assert loaded_encoder.registered_input_dim == input_dim

    # Verify adapter dimensions match the registered dimension
    for param in loaded_encoder.omics_adapter.parameters():
        if len(param.shape) > 1:  # Only check matrices, not bias vectors
            if param.shape[1] == input_dim:  # Input dimension
                assert param.shape[1] == input_dim, "Adapter input dimension doesn't match registered dimension"

    # Check that the adapter weights are preserved
    loaded_adapter_weights = [p.clone().detach() for p in loaded_encoder.omics_adapter.parameters()]
    assert len(loaded_adapter_weights) == len(adapter_weights), "Number of adapter parameters doesn't match"

    for i, (orig, loaded) in enumerate(zip(adapter_weights, loaded_adapter_weights, strict=False)):
        assert torch.allclose(orig, loaded), f"Adapter parameter {i} not preserved after save/load"

    # Try registering data with different type (should fail)
    with pytest.raises(ValueError, match="Cannot register data of type"):
        loaded_encoder.register_numeric_ds(
            dummy_dataset,
            data_origin="hvg",  # Different from original "geneformer"
        )

    # Try registering data with different dimension (should fail)
    different_dim_data = datasets.Dataset.from_dict(
        {
            "sample_idx": ["B1"],
            "data_representation": [np.ones(input_dim + 2, dtype="float32")],  # Different dimension
        }
    )

    with pytest.raises(ValueError, match="Dimension mismatch|Input dimension mismatch"):
        loaded_encoder.register_numeric_ds(different_dim_data, data_origin=data_type)
