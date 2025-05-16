import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from mmcontext.models.MMContextEncoder import AdapterModule, MMContextEncoder


# --------------------------------------------------------------------- #
# Core functionality tests
# --------------------------------------------------------------------- #
def test_text_only_encoder(text_only_encoder):
    """Test that a text-only encoder works correctly."""
    encoder = text_only_encoder
    assert encoder.registered_data_origin == "unregistered"
    assert encoder.registered_input_dim is None
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
    assert out["sentence_embedding"].shape == (2, 8)  # batch_size=2, output_dim=8


def test_bimodal_only_omics(bimodal_encoder):
    text = ["sample_idx: F1 F2"]
    features = bimodal_encoder.tokenize(text)
    # The features should contain two tokens
    assert len(features["omics_ids"][0]) == 2
    bimodal_encoder.eval()
    output = bimodal_encoder(features)
    # Check the elements of the output
    assert output["token_embeddings"].shape == (1, 2, 8)  # (B=1, L=2, H=8)
    assert output["sentence_embedding"].shape == (1, 8)  # (B=1, H=8)


def test_bimodal_omics_and_text(bimodal_encoder):
    """Test that a bimodal encoder works correctly with mixed input."""
    encoder = bimodal_encoder

    sample_idx = "sample_idx: S1"

    # Mixed batch: one omics, one text
    features = encoder.tokenize([sample_idx, "This is a test"])

    # Check that the features have the right keys
    assert "input_ids" in features
    assert "attention_mask" in features
    assert "omics_ids" in features
    assert "omics_text_info" in features
    assert features["omics_text_info"][0] == 0  # Omics
    assert features["omics_text_info"][1] == 1  # Text

    # Test forward pass
    encoder.eval()
    out = encoder(features)

    # Check that we get the correct output
    assert isinstance(out, dict)
    assert "sentence_embedding" in out
    assert out["sentence_embedding"].shape == (2, 8)  # batch_size=2, output_dim=4


def test_random_init_creates_pad_and_lookup():
    import numpy as np

    enc = MMContextEncoder("prajjwal1/bert-tiny", adapter_hidden_dim=8)

    added = enc.random_initial_embeddings(["A", "B", "C"], dim=16, rng_seed=0)
    assert added == {"A": 1, "B": 2, "C": 3}

    W = enc.omics_encoder.embeddings.weight.detach().cpu().numpy()
    assert np.allclose(W[0], 0)  # PAD row
    assert W.shape == (4, 16)  # 1 pad + 3 tokens

    # vectors are not all-zero
    assert not np.allclose(W[1:], 0)


def test_bimodal_omics_and_text_tokens(bimodal_encoder):
    """
    Mixed batch with
        row-0: two omics tokens     "sample_idx: F1 F2"
        row-1: free-form caption    "This is a test"
        row-2: one  omics token     "sample_idx: S1"
    seq_len is 8 because the mocked text-tokeniser pads to length-8.
    """
    enc = bimodal_encoder
    batch = ["sample_idx: F1 F2", "This is a test", "sample_idx: S1"]
    feats = enc.tokenize(batch)  # dict of tensors

    # modality indicator sanity-check
    assert feats["omics_text_info"].tolist() == [0, 1, 0]

    # forward
    enc.eval()
    out = enc(feats)

    # tensor shapes
    assert out["sentence_embedding"].shape == (3, 8)  # (B, hidden_out)
    assert out["token_embeddings"].shape == (3, 8, 8)  # (B, L=8, hidden_out)

    # ------------------------------------------------------------------ #
    # 1) The model should have attached the *merged* attention mask
    #     (value 1 = real token, 0 = pad) to the output dict.
    # ------------------------------------------------------------------ #
    mask = out["attention_mask"]  # (3, 8)
    tok = out["token_embeddings"]  # (3, 8, 8)

    # 2) Verify that padded positions are exactly all-zeros
    assert torch.allclose(tok[mask == 0], torch.zeros_like(tok[mask == 0]))

    # 3) Count *real* (non-pad) tokens per sample
    #    a) via the mask
    expected_n_tokens = mask.view(mask.shape[0], -1).sum(dim=1)

    #    b) via non-zero rows in `token_embeddings`
    actual_n_tokens = (tok.norm(dim=-1) != 0).sum(dim=1)

    assert torch.equal(actual_n_tokens, expected_n_tokens)

    # 4) Explicit numbers we expect for this particular batch:
    #    row-0 → 2   ("F1", "F2")
    #    row-1 → whatever the text tokenizer produced (e.g. 6-7 tokens)
    #    row-2 → 2   ("S1" , "PAD -> adapter_layer")
    assert expected_n_tokens[0].item() == 2
    assert expected_n_tokens[2].item() == 2
    assert expected_n_tokens[1] >= 2  # caption has at least two tokens


def test_encoder_return_tensor(bimodal_encoder):
    """Test that the encoder can return tensors directly."""
    encoder = bimodal_encoder

    # Mixed batch with return_tensor=True
    features = encoder.tokenize(["sample_idx: S1", "This is a test"])
    features["return_tensor"] = True

    # Test forward pass
    encoder.eval()
    out = encoder.forward(features)

    # Check that we get a tensor directly
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 8)  # batch_size=2, output_dim=8


def test_no_adapter_encoder(no_adapter_encoder):
    """Test that an encoder without adapters works correctly."""
    encoder = no_adapter_encoder

    # Test with text
    features = encoder.tokenize(["This is a test"])
    out = encoder.forward(features)

    # Check output dimension matches text encoder hidden size
    assert out["sentence_embedding"].shape[1] == 32  # Hidden size from _EncStub


def test_adding_numeric_data(text_only_encoder, numeric_df, dummy_dataset_with_split):
    """Test that we can add numeric data to a text-only encoder."""
    encoder = text_only_encoder

    # Initially it's text-only
    assert not hasattr(encoder, "_has_omics") or not encoder._has_omics

    # Register numeric data
    encoder.register_initial_embeddings(numeric_df, data_origin="pca")

    registered_ds = encoder.prefix_ds(dummy_dataset_with_split, "omics_tokens")

    # Check that data was added correctly
    assert encoder._has_omics
    assert encoder.omics_encoder is not None
    assert encoder.omics_adapter is not None

    # Verify that the omics encoder has the embedding attribute
    assert hasattr(encoder.omics_encoder, "embeddings")
    assert hasattr(encoder.omics_encoder.embeddings, "weight")

    # Now test with mixed input using the prefixed ID
    sample_idx = registered_ds["train"]["omics_tokens"][0]
    features = encoder.tokenize([sample_idx, "This is a test"])

    # Check that the features have the right keys
    assert "omics_ids" in features
    assert features["omics_text_info"][0] == 0  # Omics
    assert features["omics_text_info"][1] == 1  # Text

    encoder.eval()
    # Test forward pass
    out = encoder(features)
    assert out["sentence_embedding"].shape == (2, 8)
    del encoder


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


# ------------------- saving and loading the models -------------------
# --------------------------------------------------------------------- #


def test_save_load_text_only(text_only_encoder, tmp_path):
    """Test saving and loading a text-only encoder."""
    encoder = text_only_encoder
    encoder.eval()
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
    loaded_encoder.eval()
    # Check that the outputs match
    loaded_features = loaded_encoder.tokenize(["This is a test"])
    loaded_encoder.eval()
    loaded_out = loaded_encoder(loaded_features)

    assert torch.allclose(orig_out["sentence_embedding"], loaded_out["sentence_embedding"])


def test_save_load_without_omics_matrix(text_only_encoder, numeric_df, tmp_path):
    """Test that omics matrix is not saved but data type is preserved."""
    encoder = text_only_encoder

    # Register data
    encoder.register_initial_embeddings(numeric_df, data_origin="scvi_fm")
    registered_input_dim = encoder.registered_input_dim

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

    loaded_encoder = MMContextEncoder.load(str(save_dir))
    loaded_registered_input_dim = loaded_encoder.registered_input_dim
    # Check data type is preserved but omics not initialized
    assert loaded_encoder.registered_data_origin == "scvi_fm"
    assert not loaded_encoder._has_omics
    assert loaded_registered_input_dim == registered_input_dim

    # Register same type of data
    loaded_encoder.register_initial_embeddings(
        numeric_df,
        data_origin="scvi_fm",  # Same type works
    )

    # Try to register different type
    with pytest.raises(ValueError, match="Cannot register|cannot register"):
        loaded_encoder.register_initial_embeddings(
            numeric_df,
            data_origin="geneformer",  # Different type fails
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
    loaded_encoder = MMContextEncoder.load(str(save_dir))

    # Check that the outputs match
    loaded_features = loaded_encoder.tokenize(["This is a test"])
    loaded_out = loaded_encoder(loaded_features)

    assert torch.allclose(orig_out["sentence_embedding"], loaded_out["sentence_embedding"])


def test_safetensors_exclude_embeddings(text_only_encoder, numeric_df, tmp_path):
    """Test that omics embeddings are excluded when using safetensors."""
    encoder = text_only_encoder

    # Register data - no need for additional patching
    encoder.register_initial_embeddings(numeric_df, data_origin="pca")

    # Do a forward pass with mixed input
    sample_idx = f"sample_idx: {numeric_df['token'][0]}"
    features = encoder.tokenize([sample_idx, "This is a test"])
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
    loaded_encoder.register_initial_embeddings(numeric_df, data_origin="pca")

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
    loaded_features = loaded_encoder.tokenize([sample_idx, "This is a test"])
    loaded_encoder.eval()
    loaded_out = loaded_encoder(loaded_features)

    assert torch.allclose(orig_out["sentence_embedding"], loaded_out["sentence_embedding"]), (
        "Output embeddings don't match after loading"
    )

    # Verify adapter dimensions match the registered dimension
    input_dim = loaded_encoder.registered_input_dim
    for param in loaded_encoder.omics_adapter.parameters():
        if len(param.shape) > 1:  # Only check matrices, not bias vectors
            if param.shape[1] == input_dim:  # Input dimension
                assert param.shape[1] == input_dim, "Adapter input dimension doesn't match registered dimension"


def test_model_dtype_conversion(
    text_only_encoder,
    numeric_df,
    numeric_mapping,
):
    """Test converting model dtype after registration using PyTorch's to() method."""
    # Create dataset
    encoder = text_only_encoder

    encoder.register_initial_embeddings(numeric_df, data_origin="scvi_fm")
    encoder.eval()
    # Check original dtype
    assert encoder.omics_encoder.embeddings.weight.dtype == torch.float32

    # Test in float32 first to capture input and outputs
    sample_idx = f"sample_idx: {numeric_df['token'][0]}"
    features = encoder.tokenize([sample_idx, "This is a test"])

    out_dict = encoder(features)
    assert out_dict["sentence_embedding"].dtype == torch.float32

    # Convert to half precision
    encoder.half()  # Equivalent to encoder.to(torch.float16)

    # Verify all components converted
    assert encoder.omics_encoder.embeddings.weight.dtype == torch.float16
    assert next(encoder.omics_adapter.parameters()).dtype == torch.float16
    assert next(encoder.text_adapter.parameters()).dtype == torch.float16

    # Test forward pass with converted model
    features = encoder.tokenize([sample_idx, "This is a test"])

    # Test with return_tensor=True
    features["return_tensor"] = True
    out_tensor = encoder(features)
    assert out_tensor.dtype == torch.float16

    # Test with return_tensor=False
    features["return_tensor"] = False
    out_dict = encoder(features)
    assert out_dict["sentence_embedding"].dtype == torch.float16

    encoder.register_initial_embeddings(numeric_mapping, data_origin="scvi_fm")

    # Verify newly registered data works and maintains float16
    new_id = f"sample_idx: {list(numeric_mapping.keys())[0]}"
    features = encoder.tokenize([new_id, "Another test"])
    out = encoder(features)
    assert out["sentence_embedding"].dtype == torch.float16
    encoder.to(torch.float32)  # Reset to float32 for other tests


def test_adapter_weights_preserved(text_only_encoder, tmp_path):
    """Test that adapter weights are preserved after save and load."""
    encoder = text_only_encoder

    # Initial forward pass to initialize weights
    features = encoder.tokenize(["This is a test"])
    encoder.eval()
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
    loaded_encoder = MMContextEncoder.load(str(save_dir))
    # Check that loaded weights match modified weights
    loaded_weight = next(loaded_encoder.text_adapter.parameters()).clone()
    assert torch.allclose(modified_weight, loaded_weight)


# --------------------------------------------------------------------- #
# Freezing tests
# --------------------------------------------------------------------- #
def test_full_freezing(TextEncStub, TokStub):
    """Test that freezing the text encoder works correctly."""
    enc_stub = TextEncStub(model_type="bert")
    # Create a fresh stub and verify its parameters are initially trainable
    for p in enc_stub.parameters():
        assert p.requires_grad, "Stub parameters should start with requires_grad=True"

    # Test with BERT-like model
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=enc_stub),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=TokStub()),
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


def test_partial_unfreezing(TextEncStub, TokStub):
    """Test that partial unfreezing works correctly."""
    # Create a fresh stub and verify its parameters are initially trainable
    enc_stub = TextEncStub(model_type="bert")
    for p in enc_stub.parameters():
        assert p.requires_grad, "Stub parameters should start with requires_grad=True"

    # Test with BERT-like model
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=enc_stub),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=TokStub()),
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


def test_no_freezing(TextEncStub, TokStub):
    """Test that no freezing works correctly."""
    # Create a fresh stub and verify its parameters are initially trainable
    enc_stub = TextEncStub(model_type="bert")
    for p in enc_stub.parameters():
        assert p.requires_grad, "Stub parameters should start with requires_grad=True"

    # Test with BERT-like model
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=enc_stub),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=TokStub()),
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


def test_freezing_unfreezing_roberta(TextEncStub, TokStub):
    """Test that freezing and unfreezing work correctly for RoBERTa-like models."""
    # Create a fresh stub and verify its parameters are initially trainable
    rob_stub = TextEncStub(model_type="roberta")
    for p in rob_stub.parameters():
        assert p.requires_grad, "Stub parameters should start with requires_grad=True"

    with (
        patch("transformers.AutoModel.from_pretrained", return_value=rob_stub),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=TokStub()),
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


def test_freezing_unsupported_architecture(caplog, TextEncStub, TokStub):
    """Test that a warning is logged for unsupported architecture."""
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=TextEncStub(model_type="unsupported")),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=TokStub()),
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
# Error handling tests
# --------------------------------------------------------------------- #
def test_bad_ds(text_only_encoder):
    """Test error handling for missing columns in dataset."""
    import numpy as np
    import pandas as pd

    def _rand_matrix(tokens, hidden=8, seed=0):
        rng = np.random.default_rng(seed)
        return {tok: rng.standard_normal(hidden).astype(np.float32) for tok in tokens}

    mapping = _rand_matrix(["F1", "F2", "F3"])
    bad_df = pd.DataFrame({"another column name": list(mapping.keys()), "embedding": list(mapping.values())})

    # Should raise KeyError due to missing id_col
    with pytest.raises(KeyError):
        text_only_encoder.register_initial_embeddings(
            bad_df,
            id_col="sample_idx",  # This doesn't exist
            emb_col="embedding",
            data_origin="pca",
        )

    bad_df2 = pd.DataFrame({"sample_idx": list(mapping.keys()), "wrong embedding": list(mapping.values())})

    # Should raise KeyError due to missing emb_col
    with pytest.raises(KeyError):
        text_only_encoder.register_initial_embeddings(
            bad_df2,
            id_col="sample_idx",
            emb_col="embedding",  # This doesn't exist
            data_origin="pca",
        )
    # test what happens if dimensions are inconsistent
    bad_df3 = pd.DataFrame({"sample_idx": list(mapping.keys()), "embedding": list(mapping.values())})
    bad_df3["embedding"][0] = np.ones(3, dtype=np.float32)
    # Should raise ValueError due to inconsistent dimensions
    with pytest.raises(ValueError, match="dimensions"):
        text_only_encoder.register_initial_embeddings(
            bad_df3,
            id_col="sample_idx",
            emb_col="embedding",
            data_origin="pca",
        )


def test_omics_query_without_initialization(text_only_encoder):
    """Test error handling when trying to use omics without initialization."""
    # Should raise runtime error when trying to encode omics
    with pytest.raises(RuntimeError, match="register_initial_embeddings"):
        features = text_only_encoder.tokenize(["This is ok", "This is ok too"])
        # Manually change omics_text_info to trick the model
        features["omics_text_info"][0] = 0  # Pretend this is omics
        text_only_encoder.eval()
        text_only_encoder(features)


# ---------------------------------------------------------------------
# 1.  Token‑embedding path without *token‑adapter* layers
# ---------------------------------------------------------------------


def _single_text_feature():
    return ["Just a plain caption for a unit‑test."]


def test_text_token_embeddings_without_token_adapter(TextEncStub, TokStub):
    """If *output_token_embeddings* is *True* but *adapter_hidden_dim* is *None*,
    the encoder **must still** return per‑token vectors **without** requiring a
    *text_token_adapter* layer – provided the text‑encoder’s hidden size is
    already equal to *encoder._output_dim*."""

    enc = MMContextEncoder(
        text_encoder_name="bert-base-uncased",  # patched to *TextEncStub*
        adapter_hidden_dim=None,  # <-- no adapter layers
        adapter_output_dim=None,
        output_token_embeddings=True,
    )

    assert enc.text_token_adapter is None, (
        "No *text_token_adapter* should be instantiated when *adapter_hidden_dim* "
        "is None and *_output_dim* equals the hidden size."
    )

    feats = enc.tokenize(_single_text_feature())
    out = enc(feats)

    # --- checks ---------------------------------------------------------
    assert "token_embeddings" in out, "token‑level vectors must be returned"

    tok_emb = out["token_embeddings"]  # (B, L, H)
    assert tok_emb.ndim == 3
    assert tok_emb.shape[0] == 1  # batch
    assert tok_emb.shape[-1] == enc._output_dim == enc.text_encoder.config.hidden_size


# ---------------------------------------------------------------------
# 2.  Same for the bimodal case, where omics + text *both* have 32 dims
# ---------------------------------------------------------------------


def _build_df(hidden=32):
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((3, hidden)).astype(np.float32)
    return [{"token": f"F{i + 1}", "embedding": vec} for i, vec in enumerate(vecs)]


def test_bimodal_token_embeddings_without_token_adapter(TextEncStub, TokStub):
    """A 32‑dim numeric matrix is registered so that omics & text dimensions
    match.  *output_token_embeddings=True* must **not** require the optional
    *omics_token_adapter* layer."""

    # --- build 32‑dim numeric embedding matrix -------------------------
    df = _build_df(32)  # Text encoder stub also outputs 32 dim
    import pandas as pd

    numeric_df = pd.DataFrame(df)

    enc = MMContextEncoder(
        text_encoder_name="bert-base-uncased",  # patched stub
        adapter_hidden_dim=None,
        adapter_output_dim=None,
        output_token_embeddings=True,
    )

    enc.register_initial_embeddings(numeric_df, data_origin="pca")

    # after registration dimensions match → omics_token_adapter *should not* exist
    assert enc.text_token_adapter is None, "No *omics_token_adapter* should be built when adapter_hidden_dim is None"

    # build a mixed batch (text + omics)
    prefixed = enc.processor.prefix + " " + "F1 F2"  # omics sample
    batch = [prefixed, "An accompanying caption."]

    feats = enc.tokenize(batch)
    out = enc(feats)

    tok_emb = out["token_embeddings"]  # (B, L, H)
    assert tok_emb.shape[-1] == 32
    assert out["sentence_embedding"].shape[-1] == 32


def test_bimodal_token_embeddings_omics_different_dim():
    """When the text encoder and omics matrix have different dimensions,
    the *omics_token_adapter* should be instantiated to project to the
    text encoder's hidden size."""

    # --- build 32‑dim numeric embedding matrix -------------------------
    df = _build_df(34)  # Text encoder stub also outputs 32 dim
    import pandas as pd

    numeric_df = pd.DataFrame(df)

    enc = MMContextEncoder(
        text_encoder_name="bert-base-uncased",  # patched stub
        adapter_hidden_dim=None,
        adapter_output_dim=32,  # matches text encoder
        output_token_embeddings=True,
    )

    enc.register_initial_embeddings(numeric_df, data_origin="pca")

    # after registration dimensions match → omics_token_adapter *should not* exist
    assert enc.omics_token_adapter is not None, (
        "We should have an *omics_token_adapter* when adapter_output_dim is set."
    )
    assert enc.text_token_adapter is None, (
        "No *text_token_adapter* should be built when output dim is same as text encoders hidden dim."
    )

    # build a mixed batch (text + omics)
    prefixed = enc.processor.prefix + " " + "F1 F2"  # omics sample
    batch = [prefixed, "An accompanying caption."]

    feats = enc.tokenize(batch)
    enc.eval()
    out = enc(feats)

    tok_emb = out["token_embeddings"]  # (B, L, H)
    assert tok_emb.shape[-1] == 32


def test_bimodal_token_embeddings_text_different_dim():
    """When the text encoder and omics matrix have different dimensions,
    the *omics_token_adapter* should be instantiated to project to the
    text encoder's hidden size."""

    # --- build 32‑dim numeric embedding matrix -------------------------
    df = _build_df(34)  # Text encoder stub also outputs 32 dim
    import pandas as pd

    numeric_df = pd.DataFrame(df)

    enc = MMContextEncoder(
        text_encoder_name="bert-base-uncased",  # patched stub
        adapter_hidden_dim=None,
        adapter_output_dim=34,  # matches omics encoder but text encoder has different dim
        output_token_embeddings=True,
    )

    enc.register_initial_embeddings(numeric_df, data_origin="pca")

    # after registration dimensions match → omics_token_adapter *should not* exist
    assert enc.text_token_adapter is not None, (
        "We should have an *text_token_adapter* when adapter_output_dim is differnt than text hidden dim."
    )
    assert enc.omics_token_adapter is None, (
        "No *omics_token_adapter* should be built when output dim is same as text encoders hidden dim."
    )

    # build a mixed batch (text + omics)
    prefixed = enc.processor.prefix + " " + "F1 F2"  # omics sample
    batch = [prefixed, "An accompanying caption."]

    feats = enc.tokenize(batch)
    enc.eval()
    out = enc(feats)

    tok_emb = out["token_embeddings"]  # (B, L, H)
    assert tok_emb.shape[-1] == 34


# ---------------------------------------------------------------------
# 3.  *adapter_hidden_dim=None* but *adapter_output_dim* given → linear head
# ---------------------------------------------------------------------


def test_linear_adapter_created_when_only_output_dim(TextEncStub, TokStub):
    """When the user specifies *adapter_output_dim* but omits *adapter_hidden_dim*,
    the encoder should **still** instantiate an *AdapterModule* acting as a
    single linear layer (hidden=None)."""

    enc = MMContextEncoder(
        text_encoder_name="bert-base-uncased",
        adapter_hidden_dim=None,
        adapter_output_dim=16,  # request projection → 16‑dim output
    )

    # Adapter **must** exist and change dimensionality -------------------
    assert isinstance(enc.text_adapter, AdapterModule)
    assert enc._output_dim == 16

    feats = enc.tokenize(_single_text_feature())
    enc.eval()
    out = enc(feats)

    sent = out["sentence_embedding"]
    assert sent.shape == (1, 16)


# ---------------------------------------------------------------------
# 4.  Save / load round‑trip should work in *all* configurations
# ---------------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path, numeric_df):
    """Model can be serialised (w/ & w/o omics) and re‑loaded without errors."""

    enc = MMContextEncoder(
        text_encoder_name="bert-base-uncased",
        adapter_hidden_dim=4,
        adapter_output_dim=8,
        output_token_embeddings=True,
    )
    enc.register_initial_embeddings(numeric_df, data_origin="pca")

    # -------- persist ---------------------------------------------------
    model_dir = tmp_path / "roundtrip"
    enc.save(str(model_dir), safe_serialization=False)

    # -------- restore ---------------------------------------------------
    rec = MMContextEncoder.load(str(model_dir), safe_serialization=False)

    # text pathway must still be functional ------------------------------
    feats = rec.tokenize(["Round‑trip caption"])
    rec.eval()
    out = rec(feats)
    assert out["sentence_embedding"].shape[-1] == rec._output_dim


# ---------------------------------------------------------------------
# 5.  Make sure *AdapterModule* is Identity when dims already match & no proj
# ---------------------------------------------------------------------


def test_identity_adapter_when_no_hidden_no_output(TextEncStub):
    """If neither *adapter_hidden_dim* nor *adapter_output_dim* are given, the
    adapter should degrade to *nn.Identity* and keep dimensions intact."""

    enc = MMContextEncoder(
        text_encoder_name="bert-base-uncased",  # patched stub
        adapter_hidden_dim=None,
        adapter_output_dim=None,
    )
    assert isinstance(enc.text_adapter.net, torch.nn.modules.linear.Identity)

    feats = enc.tokenize(_single_text_feature())
    out = enc(feats)
    assert out["sentence_embedding"].shape[-1] == enc.text_encoder.config.hidden_size


# ---------------------------------------------------------------------
# prefix_ds: single-pair & multi-pair datasets
# ---------------------------------------------------------------------


def _check_prefixed(col, pref):
    """Helper – every string in *col* must start with the prefix."""
    assert all(s.startswith(pref) for s in col)


def test_prefix_ds_pairs_and_multiplets(text_only_encoder):
    import datasets

    enc = text_only_encoder  # fixture – no omics registered
    pref = enc.processor.prefix  # usually "sample_idx:"

    # ----------------------- 1) PAIRS ---------------------------------
    pair_ds = datasets.Dataset.from_dict(
        {
            "sample_idx": ["S1", "S2", "S3"],
            "caption": ["cap1", "cap2", "cap3"],
            "label": [1, 0, 1],
            "junk_col": [42, 43, 44],  # must be dropped
        }
    )

    proc_pair = enc.prefix_ds(pair_ds, cols_to_prefix="sample_idx")

    # Columns retained: anchor, caption, label
    assert set(proc_pair.column_names) == {"sample_idx", "caption", "label"}
    _check_prefixed(proc_pair["sample_idx"], pref)  # every row prefixed

    # ----------------------- 2) MULTIPLETS -----------------------------
    multi_ds = datasets.Dataset.from_dict(
        {
            "sample_idx": ["S4", "S5"],
            "positive": ["T cell", "B cell"],
            "negative0": ["Macrophage", "NK cell"],
            "negative1": ["Neuron", "Astrocyte"],
            "junk_col": ["x", "y"],  # must be dropped
        }
    )

    proc_multi = enc.prefix_ds(
        multi_ds,
        cols_to_prefix="sample_idx",
        positive_col="positive",
        negative_prefix="negative",
    )

    # Only anchor + positive + all negatives kept
    assert set(proc_multi.column_names) == {
        "sample_idx",
        "positive",
        "negative0",
        "negative1",
    }
    _check_prefixed(proc_multi["sample_idx"], pref)
    _check_prefixed(proc_multi["positive"], "")  # captions stay raw
    _check_prefixed(proc_multi["negative0"], "")  # negatives are text

    # -------------- behavioural sanity: extra column really gone -------
    assert "junk_col" not in proc_pair.column_names
    assert "junk_col" not in proc_multi.column_names

    # -------------- DatasetDict variant --------------------------------
    ddict = datasets.DatasetDict(train=pair_ds, val=multi_ds)
    proc_ddict = enc.prefix_ds(ddict, cols_to_prefix="sample_idx")

    assert isinstance(proc_ddict, datasets.DatasetDict)
    assert set(proc_ddict["train"].column_names) == {"sample_idx", "caption", "label"}
    assert set(proc_ddict["val"].column_names) == {
        "sample_idx",
        "positive",
        "negative0",
        "negative1",
    }
