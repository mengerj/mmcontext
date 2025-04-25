'''
# test_sentence_transformer_integration.py
import json
import logging
import os

import numpy as np
import pytest
import torch
from sentence_transformers import SentenceTransformer

from mmcontext.models import MMContextEncoder, MMContextProcessor

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_processor(monkeypatch, tmp_path):
    """
    Mock processor for NumPy retrieval mode.
    """
    # Create test data
    np.random.seed(42)
    test_data = np.random.randn(4, 8)  # 4 samples, 8 dimensions
    sample_ids = np.array(["SAMPLE_0", "SAMPLE_1", "SAMPLE_2", "SAMPLE_3"])

    # Save test data
    np.savez(tmp_path / "test_embeddings.npz", data=test_data, sample_ids=sample_ids)

    # Create processor first
    processor = MMContextProcessor(
        processor_name="precomputed", text_encoder_name="prajjwal1/bert-tiny", obsm_key="X_emb"
    )

    # Then patch the specific instance's method
    def mock__resolve_file_path(self, file_path, suffix=".npz"):
        """Mock file path resolution to return our test NPZ file."""
        return str(tmp_path / "test_embeddings.npz")

    # Directly set the method on the instance
    processor.omics_processor._resolve_file_path = lambda file_path, suffix=".npz": mock__resolve_file_path(
        None, file_path, suffix
    )

    return processor


@pytest.fixture
def bimodal_encoder(mock_processor):
    """Creates an MMContextEncoder instance."""
    text_encoder_name = "prajjwal1/bert-tiny"
    omics_input_dim = 8

    model = MMContextEncoder(
        text_encoder_name=text_encoder_name,
        omics_input_dim=omics_input_dim,
        processor_name="precomputed",
        processor_obsm_key="X_emb",
    )
    model.processor = mock_processor
    model.processor.omics_processor.clear_cache()
    return model


def test_omics_retrieval(bimodal_encoder):
    """Test omics data retrieval."""
    omics_data = [
        {
            "file_record": {"dataset_path": "fake_path_1", "embeddings": {"X_emb": "fake_path_1"}},
            "sample_id": "SAMPLE_1",
        },
        {
            "file_record": {"dataset_path": "fake_path_2", "embeddings": {"X_emb": "fake_path_2"}},
            "sample_id": "SAMPLE_2",
        },
    ]
    bimodal_encoder.processor.omics_processor.clear_cache()
    rep = bimodal_encoder.processor.omics_processor.get_rep(omics_data)
    assert rep is not None, "get_rep returned None"
    assert isinstance(rep, torch.Tensor), "Expected torch.Tensor output"
    assert rep.shape == (2, 8), f"Expected shape (2,8), got {rep.shape}"
    logger.info("Omics retrieval: shape is correct, tensor type verified")


def test_sentence_transformer_integration(bimodal_encoder, tmp_path):
    """Test full SentenceTransformer integration."""
    st_model = SentenceTransformer(modules=[bimodal_encoder])

    # Test both omics and text inputs
    inputs = [
        {
            "file_record": {"dataset_path": "fake_path_1", "embeddings": {"X_emb": "fake_path_1"}},
            "sample_id": "SAMPLE_1",
        },
        "This is a test text input",
        {
            "file_record": {"dataset_path": "fake_path_2", "embeddings": {"X_emb": "fake_path_2"}},
            "sample_id": "SAMPLE_2",
        },
    ]

    # First test tokenization
    features = bimodal_encoder.tokenize(inputs)
    assert "omics_representation" in features, "Missing omics_representation in features"
    assert features["omics_representation"] is not None, "omics_representation is None"
    assert isinstance(features["omics_representation"], torch.Tensor), "omics_representation should be a tensor"
    assert "omics_text_info" in features, "Missing omics_text_info in features"
    assert features["omics_text_info"] == [0, 1, 0], "Incorrect omics_text_info values"

    # Test encoding
    emb = st_model.encode(inputs, convert_to_tensor=True)
    assert emb is not None, "encode returned None"
    assert isinstance(emb, torch.Tensor), "encode should return a tensor"
    assert emb.shape == (3, 2048), f"Expected shape (3, 2048), got {emb.shape}"

    # Test model saving and loading
    save_dir = tmp_path / "st_model"
    st_model.save(str(save_dir))
    loaded_model = SentenceTransformer(str(save_dir))

    # Apply the same mock to the loaded model
    def mock__resolve_file_path(self, file_path, suffix=".npz"):
        """Mock file path resolution to return our test NPZ file."""
        return str(tmp_path / "test_embeddings.npz")

    # Get the MMContextEncoder module from the loaded model
    loaded_encoder = loaded_model._first_module()
    loaded_encoder.processor.omics_processor._resolve_file_path = (
        lambda file_path, suffix=".npz": mock__resolve_file_path(None, file_path, suffix)
    )
    loaded_encoder.processor.omics_processor.clear_cache()

    # Compare embeddings
    emb_loaded = loaded_model.encode(inputs, convert_to_tensor=True)
    assert torch.allclose(emb, emb_loaded, atol=1e-5)
    logger.info("SentenceTransformer integration successful with save/load verification")


def test_processor_debug(mock_processor):
    """Debug test to verify processor behavior."""
    # Test basic processor setup
    assert mock_processor.omics_processor is not None
    assert mock_processor.omics_processor.obsm_key == "X_emb"

    # Test single item retrieval
    single_item = {
        "file_record": {"dataset_path": "fake_path_1", "embeddings": {"X_emb": "fake_path_1"}},
        "sample_id": "SAMPLE_1",
    }

    result = mock_processor.omics_processor.get_rep([single_item])
    assert result is not None, "get_rep returned None"
    assert isinstance(result, torch.Tensor), "Expected torch.Tensor output"
    assert result.shape == (1, 8), f"Expected shape (1,8), got {result.shape}"

    # Test cache behavior
    mock_processor.omics_processor.clear_cache()
    assert len(mock_processor.omics_processor._data_cache) == 0, "Cache not cleared"
    assert len(mock_processor.omics_processor._path_cache) == 0, "Path cache not cleared"
'''
