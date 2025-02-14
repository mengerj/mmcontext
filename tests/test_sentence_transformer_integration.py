# test_mmcontext_processor.py
import json
import logging

import anndata
import numpy as np
import pytest
import torch
from sentence_transformers import SentenceTransformer

from mmcontext.models import MMContextEncoder

# Adjust these imports to match your real module structure
from mmcontext.pp import AnnDataRetrievalProcessor, MMContextProcessor

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_processor(monkeypatch):
    """
    This fixture patches AnnDataRetrievalProcessor so it can handle fake paths
    and read a mocked AnnData object that has obsm["X_test"] and obs.index
    containing expected sample IDs.
    """

    def mock__resolve_file_path(self, file_path: str) -> str:
        """
        Replace any incoming file path with something that ends in '.h5ad'.
        That way, we pass the 'endswith(".h5ad")' check in get_rep().
        """
        return "mocked_file.h5ad"

    def mock_read_h5ad(file, *args, **kwargs):
        """
        Returns a small AnnData with obsm["X_test"].
        Also sets obs_names so that 'sample_id' lookups succeed.
        """
        # Let's say we have 4 samples
        # set a seed
        np.random.seed(42)
        X = np.random.randn(4, 5)  # shape (n_samples, n_features)
        adata = anndata.AnnData(X)

        # The obsm key that your code uses, e.g. "X_test"
        # We'll store a (4 x 8) matrix to match an omics dim=8 scenario
        adata.obsm["X_test"] = np.random.randn(4, 8)

        # Create sample IDs that we might expect in your tests
        adata.obs_names = ["SAMPLE_0", "SAMPLE_1", "SAMPLE_2", "SAMPLE_3"]
        return adata

    # Patch the resolve_file_path at the CLASS level
    monkeypatch.setattr(AnnDataRetrievalProcessor, "_resolve_file_path", mock__resolve_file_path)

    # Patch anndata.read_h5ad
    monkeypatch.setattr(anndata, "read_h5ad", mock_read_h5ad)

    # If you ever need read_zarr, do similarly:
    # monkeypatch.setattr(anndata, "read_zarr", mock_read_zarr)

    # Return a processor with obsm_key="X_test" for convenience
    processor = MMContextProcessor(obsm_key="X_test")
    return processor


@pytest.fixture
def bimodal_encoder(mock_processor):
    """
    Creates an MMContextEncoder instance that uses the patched processor.
    We do so by passing 'processor_obsm_key="X_test"' and a matching dimension.
    """
    # The text model can be anything small
    text_encoder_name = "prajjwal1/bert-tiny"
    omics_input_dim = 8  # Must match the shape of adata.obsm["X_test"] in the mock

    # Instead of letting MMContextProcessor auto-create the processor,
    # we can directly assign it after the model is made, or we can monkeypatch
    # the entire class initialization. For simplicity, let's do this:
    model = MMContextEncoder(
        text_encoder_name=text_encoder_name,
        omics_input_dim=omics_input_dim,
        processor_obsm_key="X_test",
        freeze_text_encoder=False,
        unfreeze_last_n_layers=0,
    )

    # The line below overrides the automatically created processor with our mocked one:
    model.processor = mock_processor
    # (As an alternative, you could patch the entire
    #  'MMContextProcessor.__init__' to return 'mock_processor'.)

    return model


def test_omics_with_sample_ids(bimodal_encoder):
    """
    Demonstrates calling get_rep() with fake file_paths and real sample_ids,
    verifying that it returns a valid tensor.
    """
    # We'll feed 2 items that both refer to "mocked_file.h5ad" behind the scenes
    # and each has a sample_id that we know is in adata.obs_names:
    omics_data = [
        {"file_path": "fake_path_1", "sample_id": "SAMPLE_1"},
        {"file_path": "fake_path_2", "sample_id": "SAMPLE_2"},
    ]

    # Now we call get_rep() directly:
    rep = bimodal_encoder.processor.omics_processor.get_rep(omics_data)
    # We expect shape (2, 8), because we inserted an (4,8) matrix in the mock,
    # and we're requesting 2 of the samples
    assert rep.shape == (2, 8), f"Expected shape (2,8), got {rep.shape}"
    logger.info("Omics retrieval via mock: shape is correct, no file format errors.")


def test_sentence_transformer_integration_with_mocked_obsm(bimodal_encoder, tmp_path):
    """
    Integrates everything with SentenceTransformer. We pass
    JSON strings containing file_path + sample_id, the model
    calls get_rep() with the patched logic, and no error occurs.
    """
    st_model = SentenceTransformer(modules=[bimodal_encoder])

    # We'll encode a small batch of 2 items:
    # 1) Omics data in JSON
    # 2) Omics data in JSON
    # If you want to mix text + omics, just add text strings to the list.
    inputs = [
        json.dumps({"file_path": "fake_path_1", "sample_id": "SAMPLE_1"}),
        json.dumps({"file_path": "fake_path_2", "sample_id": "SAMPLE_3"}),
    ]

    # Because these are recognized as omics (due to "file_path"), your tokenize()
    # sets omics_text_info=0 for each, so the final embedding dimension is 2048.
    emb = st_model.encode(inputs, convert_to_tensor=True)
    assert emb.shape == (2, 2048)
    logger.info("SentenceTransformer integration test with mocked obsm succeeded.")

    # Optionally test saving/loading
    save_dir = tmp_path / "st_model"
    st_model.save(str(save_dir))
    loaded_st_model = SentenceTransformer(str(save_dir))
    # Compare the parameters between the original and loaded models
    original_model = st_model
    loaded_model = loaded_st_model
    original_state = original_model.state_dict()
    loaded_state = loaded_model.state_dict()

    # They should have the same keys
    assert original_state.keys() == loaded_state.keys(), "State dict keys differ!"

    # Compare each param/buffer
    for key in original_state.keys():
        param_orig = original_state[key]
        param_loaded = loaded_state[key]
        # Use allclose with a tight tolerance
        if not torch.allclose(param_orig, param_loaded, atol=1e-8, rtol=1e-5):
            raise ValueError(f"Mismatch in param/buffer {key}")
    print("All parameters & buffers match perfectly between original and loaded models.")
    # Check embeddings match after load

    # test that passing text strings gives the same embeddings
    text_inputs = ["This is a test.", "This is another test."]
    emb_orig = st_model.encode(text_inputs)
    emb_loaded = loaded_st_model.encode(text_inputs)
    assert torch.allclose(torch.tensor(emb_orig), torch.tensor(emb_loaded), atol=1e-5)
    # test that passing omics inputs gives the same embeddings
    emb_orig = st_model.encode(inputs)
    emb_loaded = loaded_st_model.encode(inputs)
    assert torch.allclose(torch.tensor(emb_orig), torch.tensor(emb_loaded), atol=1e-5)

    logger.info("Embeddings match after saving/loading the SentenceTransformer.")
