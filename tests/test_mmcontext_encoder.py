# tests/test_mmcontext_encoder.py

import logging

import pytest
import torch

from mmcontext.engine import MMContextEncoder


@pytest.fixture
def common_setup():
    # Common dimensions for all tests
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 3
    num_heads = 4
    batch_size = 20
    seq_length = 10  # Number of samples per sequence

    # Sample data embeddings
    data_embeddings = torch.randn(batch_size, seq_length, embedding_dim)

    # Sample context embeddings
    context_embeddings = torch.randn(batch_size, seq_length, embedding_dim)

    return {
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "data_embeddings": data_embeddings,
        "context_embeddings": context_embeddings,
    }


def test_mlp_only_model(common_setup):
    """Test MMContextEncoder in MLP-only configuration."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_mlp_only_model")
    model = MMContextEncoder(
        embedding_dim=common_setup["embedding_dim"],
        hidden_dim=common_setup["hidden_dim"],
        num_layers=common_setup["num_layers"],
        use_self_attention=False,
        use_cross_attention=False,
    )

    output = model(in_main=common_setup["data_embeddings"])

    # Check output shape
    expected_shape = (
        common_setup["batch_size"],
        common_setup["seq_length"],
        common_setup["embedding_dim"],
    )
    assert output.shape == expected_shape


def test_self_attention_model(common_setup):
    """Test MMContextEncoder with self-attention."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_self_attention_model")
    model = MMContextEncoder(
        embedding_dim=common_setup["embedding_dim"],
        hidden_dim=common_setup["hidden_dim"],
        num_layers=common_setup["num_layers"],
        num_heads=common_setup["num_heads"],
        use_self_attention=True,
        use_cross_attention=False,
    )

    output = model(in_main=common_setup["data_embeddings"])

    # Check output shape
    expected_shape = (
        common_setup["batch_size"],
        common_setup["seq_length"],
        common_setup["embedding_dim"],
    )
    assert output.shape == expected_shape


def test_cross_attention_model(common_setup):
    """Test MMContextEncoder with cross-attention."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_cross_attention_model")
    model = MMContextEncoder(
        embedding_dim=common_setup["embedding_dim"],
        hidden_dim=common_setup["hidden_dim"],
        num_layers=common_setup["num_layers"],
        num_heads=common_setup["num_heads"],
        use_self_attention=False,
        use_cross_attention=True,
    )

    # Forward pass with context embeddings
    output = model(in_main=common_setup["data_embeddings"], in_cross=common_setup["context_embeddings"])

    # Check output shape
    expected_shape = (
        common_setup["batch_size"],
        common_setup["seq_length"],
        common_setup["embedding_dim"],
    )
    assert output.shape == expected_shape


def test_combined_attention_model(common_setup):
    """Test MMContextEncoder with both self-attention and cross-attention."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_combined_attention_model")

    model = MMContextEncoder(
        embedding_dim=common_setup["embedding_dim"],
        hidden_dim=common_setup["hidden_dim"],
        num_layers=common_setup["num_layers"],
        num_heads=common_setup["num_heads"],
        use_self_attention=True,
        use_cross_attention=True,
    )

    # Forward pass with context embeddings
    output = model(in_main=common_setup["data_embeddings"], in_cross=common_setup["context_embeddings"])

    # Check output shape
    expected_shape = (
        common_setup["batch_size"],
        common_setup["seq_length"],
        common_setup["embedding_dim"],
    )
    assert output.shape == expected_shape


def test_missing_context_error(common_setup):
    """Test that ValueError is raised when in_cross embeddings are missing."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_missing_context_error")

    model = MMContextEncoder(
        embedding_dim=common_setup["embedding_dim"],
        hidden_dim=common_setup["hidden_dim"],
        num_layers=common_setup["num_layers"],
        use_self_attention=False,
        use_cross_attention=True,
    )

    with pytest.raises(ValueError, match="in_cross embeddings are required when using cross-attention."):
        model(in_main=common_setup["data_embeddings"])


def test_save_and_load_model(tmp_path, common_setup):
    """Test saving and loading of model weights."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_save_and_load_model")
    model = MMContextEncoder(
        embedding_dim=common_setup["embedding_dim"],
        hidden_dim=common_setup["hidden_dim"],
        num_layers=common_setup["num_layers"],
        num_heads=common_setup["num_heads"],
        use_self_attention=True,
        use_cross_attention=True,
    )
    model.eval()
    # Forward pass to initialize model parameters
    output_before_save = model(in_main=common_setup["data_embeddings"], in_cross=common_setup["context_embeddings"])

    # Save the model weights to a temporary file
    save_path = tmp_path / "model.pth"
    model.save(save_path)

    # Create a new model instance and load the weights
    new_model = MMContextEncoder(
        embedding_dim=common_setup["embedding_dim"],
        hidden_dim=common_setup["hidden_dim"],
        num_layers=common_setup["num_layers"],
        num_heads=common_setup["num_heads"],
        use_self_attention=True,
        use_cross_attention=True,
    )
    new_model.load(save_path)
    new_model.eval()

    state_dict_before = model.state_dict()
    state_dict_after = new_model.state_dict()

    for key in state_dict_before.keys():
        if not torch.equal(state_dict_before[key], state_dict_after[key]):
            ValueError(f"Mismatch in parameter {key}")

    # Verify that the new model produces the same output
    output_after_load = new_model(in_main=common_setup["data_embeddings"], in_cross=common_setup["context_embeddings"])

    # Check that the outputs are the same
    assert torch.allclose(output_before_save, output_after_load, atol=1e-6)


def test_incorrect_input_dimensions(common_setup):
    """Test that an error is raised when input dimensions are incorrect."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_incorrect_input_dimensions")
    model = MMContextEncoder(
        embedding_dim=common_setup["embedding_dim"],
        hidden_dim=common_setup["hidden_dim"],
        num_layers=common_setup["num_layers"],
        use_self_attention=False,
        use_cross_attention=False,
    )

    # Incorrect data embeddings shape (missing sequence dimension)
    incorrect_data_embeddings = torch.randn(common_setup["batch_size"], common_setup["embedding_dim"])

    with pytest.raises(ValueError, match="Expected in_main to have 3 dimensions"):
        model(in_main=incorrect_data_embeddings)


def test_different_context_dimensions(common_setup):
    """Test handling when context embeddings have different dimensions."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_different_context_dimensions")
    use_cross_attention = True  # Otherwise context embeddings are not used
    model = MMContextEncoder(
        embedding_dim=common_setup["embedding_dim"],
        hidden_dim=common_setup["hidden_dim"],
        num_layers=common_setup["num_layers"],
        use_self_attention=False,
        use_cross_attention=use_cross_attention,
    )

    # Context embeddings with different embedding dimension
    incorrect_context_embeddings = torch.randn(
        common_setup["batch_size"], common_setup["seq_length"], common_setup["embedding_dim"] + 10
    )

    with pytest.raises(RuntimeError):
        model(in_main=common_setup["data_embeddings"], in_cross=incorrect_context_embeddings)
