# test_mm_context_encoder.py

import json
import logging
import os
from pathlib import Path

import pytest
import torch

# Make sure these imports match your actual module paths
from mmcontext.models import MMContextEncoder

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def mock_model_path(tmp_path_factory):
    """
    Creates a temporary directory path for saving and loading models.

    Returns
    -------
    Path
        A temporary directory path where model artifacts can be saved.
    """
    return tmp_path_factory.mktemp("model_tests")


@pytest.fixture(scope="module")
def encoder_model():
    """
    Creates an instance of the MMContextEncoder using a tiny HuggingFace model
    for quick initialization.

    Returns
    -------
    MMContextEncoder
        An instantiated MMContextEncoder object with a small BERT model and a mock omics input dimension of 128.
    """
    text_encoder_name = "prajjwal1/bert-tiny"  # A small model for fast tests
    omics_input_dim = 128
    model = MMContextEncoder(
        text_encoder_name=text_encoder_name,
        omics_input_dim=omics_input_dim,
        processor_obsm_key="X_test",
        freeze_text_encoder=False,
        unfreeze_last_n_layers=0,
    )
    return model


def test_encoder_instantiation(encoder_model):
    """
    Tests that the MMContextEncoder is instantiated correctly.

    Parameters
    ----------
    encoder_model : MMContextEncoder
        Fixture that provides a ready-to-use model instance.
    """
    assert isinstance(encoder_model, MMContextEncoder)
    assert encoder_model._get_sentence_embedding_dimension() == 2048
    logger.info("Instantiation test passed. Model dimension is 2048 as expected.")


def test_forward_text_only(encoder_model):
    """
    Tests that forward pass with text-only inputs works correctly.

    The text data references the public Hugging Face model "prajjwal1/bert-tiny"
    token space. We create a random input_ids and attention_mask for demonstration.

    Parameters
    ----------
    encoder_model : MMContextEncoder
        Fixture that provides a ready-to-use model instance.
    """
    batch_size = 4
    seq_length = 8
    # Fake inputs for text
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.int64)

    features = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        # No omics data
        "omics_text_info": [1] * batch_size,  # All are text
    }

    output = encoder_model(features)
    emb = output["sentence_embedding"]
    assert emb.shape == (batch_size, 2048)
    logger.info("Text-only forward pass test passed. Output shape is correct.")


def test_forward_omics_only(encoder_model):
    """
    Tests that forward pass works correctly with omics-only inputs.

    The omics data here is random. We artificially create a shape (batch_size, omics_input_dim).

    Parameters
    ----------
    encoder_model : MMContextEncoder
        Fixture that provides a ready-to-use model instance.
    """
    batch_size = 4
    omics_dim = encoder_model.omics_input_dim
    omics_input = torch.randn(batch_size, omics_dim)

    features = {
        "omics_representation": omics_input,
        "omics_text_info": [0] * batch_size,  # All are omics
    }

    output = encoder_model(features)
    emb = output["sentence_embedding"]
    assert emb.shape == (batch_size, 2048)
    logger.info("Omics-only forward pass test passed. Output shape is correct.")


def test_forward_mixed_inputs(encoder_model):
    """
    Tests that forward pass works correctly when mixing both text and omics.

    We create a small batch where half the items are text, half are omics.

    Parameters
    ----------
    encoder_model : MMContextEncoder
        Fixture that provides a ready-to-use model instance.
    """
    batch_size = 6
    seq_length = 5
    omics_dim = encoder_model.omics_input_dim

    # Let the first 3 be omics, last 3 be text
    omics_input = torch.randn(3, omics_dim)
    text_input_ids = torch.randint(0, 1000, (3, seq_length))
    text_mask = torch.ones((3, seq_length), dtype=torch.int64)

    # Combine them in a features dict
    # But we need them in the right order according to omics_text_info
    # We'll do [0, 0, 0, 1, 1, 1]
    features = {
        "omics_representation": omics_input,  # shape (3, omics_dim)
        "input_ids": text_input_ids,  # shape (3, seq_len)
        "attention_mask": text_mask,
        "omics_text_info": [0, 0, 0, 1, 1, 1],
    }

    output = encoder_model(features)
    emb = output["sentence_embedding"]
    assert emb.shape == (batch_size, 2048)
    logger.info("Mixed (omics + text) forward pass test passed. Output shape is correct.")


def test_freezing_unfreezing():
    """
    Tests that freezing and unfreezing the last n layers works as expected
    by verifying some parameters become trainable while others remain frozen.

    References
    ----------
    The data to check here is purely the named parameters of the text encoder.
    """

    # Freeze, but unfreeze last 1 layer
    model = MMContextEncoder(
        text_encoder_name="prajjwal1/bert-tiny",
        omics_input_dim=100,
        freeze_text_encoder=True,
        unfreeze_last_n_layers=1,
    )

    # Count how many layers are free
    free_params = [p for p in model.text_encoder.parameters() if p.requires_grad]
    all_params = list(model.text_encoder.parameters())

    assert len(free_params) > 0, "At least some parameters (the last layer) should be trainable."
    assert len(free_params) < len(all_params), "Not all parameters should be free, as we froze them."

    logger.info(
        f"Freezing/unfreezing test passed. {len(free_params)} trainable params, "
        f"{len(all_params)} total params in text encoder."
    )


def test_model_saving_loading(encoder_model, mock_model_path):
    """
    Tests that saving and loading the model state preserves configuration and parameters.

    Parameters
    ----------
    encoder_model : MMContextEncoder
        Fixture that provides a ready-to-use model instance.
    mock_model_path : Path
        A temporary directory path provided by pytest for saving/loading.
    """
    # 1) Save the model
    save_path = mock_model_path / "test_saved_model"
    encoder_model.save(str(save_path), safe_serialization=True)
    original_model = encoder_model

    # Verify config file
    config_path = os.path.join(save_path, "config.json")
    assert os.path.exists(config_path), "Config file was not saved."

    with open(config_path) as f:
        config_data = json.load(f)
    assert "text_encoder_name" in config_data, "Config missing text_encoder_name key."

    # 2) Load a fresh model from that path
    loaded_model = MMContextEncoder.load(str(save_path), safe_serialization=True)

    # 3) Compare some essential config fields
    assert loaded_model.text_encoder_name == encoder_model.text_encoder_name
    assert loaded_model.omics_input_dim == encoder_model.omics_input_dim

    # 4) Confirm weights are the same
    for p1, p2 in zip(encoder_model.parameters(), loaded_model.parameters(), strict=False):
        assert torch.equal(p1, p2), "Parameters differ after loading."

    # 5) Confirm passing input through original and loaded gives the same results
    seq_length = 5
    omics_dim = encoder_model.omics_input_dim
    # Let the first 3 be omics, last 3 be text
    omics_input = torch.randn(3, omics_dim)
    text_input_ids = torch.randint(0, 1000, (3, seq_length))
    text_mask = torch.ones((3, seq_length), dtype=torch.int64)

    # Combine them in a features dict
    # But we need them in the right order according to omics_text_info
    # We'll do [0, 0, 0, 1, 1, 1]
    features = {
        "omics_representation": omics_input,  # shape (3, omics_dim)
        "input_ids": text_input_ids,  # shape (3, seq_len)
        "attention_mask": text_mask,
        "omics_text_info": [0, 0, 0, 1, 1, 1],
    }
    original_emb = original_model(features)["sentence_embedding"]
    loaded_emb = loaded_model(features)["sentence_embedding"]
    assert torch.equal(original_emb, loaded_emb), "Output embeddings differ after loading."
    logger.info("Saving/loading test passed. Model configuration and weights match.")
