"""
Test suite for the EncoderPreTrained class.

This module demonstrates how to test:
1) Loading pre-trained weights into `MMContextEncoder` models.
2) Encoding data and context in two stages:
   - (A) The initial embedding step with a user-defined embedder.
   - (B) The final embedding step with the loaded `MMContextEncoder`.
3) Ensuring that final embeddings end up in the correct `AnnData.obsm` keys.
4) Proper handling of missing encoders or missing initial embeddings.
"""

import importlib
import logging
import shutil

import anndata
import numpy as np
import pytest

from mmcontext.engine import MMContextEncoder

# Always use the predefined logger per your instructions
logger = logging.getLogger(__name__)

# --- Example imports; adapt to your local paths ---
# from mmcontext.models import MMContextEncoder
# from mmcontext.pp import CategoryEmbedder, AnnDataStoredEmbedder
# from path.to.your.module import EncoderPreTrained  # Adjust this import


@pytest.fixture
def toy_anndata() -> anndata.AnnData:
    """
    Create a small synthetic AnnData object for testing.

    This function simulates a dataset of 10 samples with 5 features.
    We also add a categorical column for context embedding.

    Returns
    -------
    anndata.AnnData
        Synthetic test data stored in an AnnData object.
    """
    X = np.random.rand(10, 5)  # 10 cells, 5 genes/features
    adata = anndata.AnnData(X)
    adata.obs["cell_type"] = ["typeA"] * 5 + ["typeB"] * 5
    return adata


@pytest.fixture
def data_context_encoders() -> dict:
    """
    Create two MMContextEncoder instances for data and context,
    pointing to user-provided weight paths.

    Returns
    -------
    dict
        A dictionary containing two `MMContextEncoder` instances:
        {"data_encoder": data_enc, "context_encoder": ctx_enc}
    """
    latent_dim = 64
    hidden_dim = 64
    num_layers = 1
    num_heads = 1

    # Instantiate the encoders
    data_enc = MMContextEncoder(
        embedding_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        use_self_attention=False,
        use_cross_attention=False,
        activation="relu",
        dropout=0.1,
    )
    ctx_enc = MMContextEncoder(
        embedding_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        use_self_attention=False,
        use_cross_attention=False,
        activation="relu",
        dropout=0.1,
    )

    return {
        "data_encoder": data_enc,
        "context_encoder": ctx_enc,
    }


@pytest.fixture
def weights_paths(tmp_path) -> dict:
    """
    Return the dictionary of weights paths for data and context encoders.

    Notes
    -----
    Adjust these paths to match your actual file system structure.
    """
    with importlib.resources.path("mmcontext.data", "best_encoder_weights") as actual_weight_paths:
        temp_file_paths = {}
        weights_paths = {
            "data_encoder": "best_data_encoder_weights.pth",
            "context_encoder": "best_context_encoder_weights.pth",
        }
        for key in weights_paths:
            temp_file_paths[key] = tmp_path / weights_paths[key]
            shutil.copy(actual_weight_paths / weights_paths[key], temp_file_paths[key])
    return temp_file_paths


@pytest.fixture
def encoder_pretrained(data_context_encoders, weights_paths, tmp_path):
    """
    Create an instance of EncoderPreTrained with two encoders and their weights.

    Returns
    -------
    EncoderPreTrained
        An instance of the class under test, loaded with the specified models and weights.
    """
    # Import inside the fixture to avoid import-time side effects
    import torch

    from mmcontext.engine import EncoderPreTrained

    encoder_pretrained_obj = EncoderPreTrained(
        encoders=data_context_encoders,
        weights_paths=weights_paths,
        data_obsm_key_final="X_data_final",
        context_obsm_key_final="X_context_final",
        device=torch.device("cpu"),
    )
    return encoder_pretrained_obj


class DataEmbedder:
    """
    Create a simple AnnDataStoredEmbedder or a custom function that places
    initial data embeddings into adata.obsm["X_data_init"].

    For demonstration, we place random embeddings. In practice, you could
    instantiate something like `AnnDataStoredEmbedder(...)`.
    """

    # from mmcontext.pp import AnnDataStoredEmbedder
    # embedder_obj = AnnDataStoredEmbedder(obsm_key="scvi")

    # For testing, just embed random vectors:
    def embed(adata: anndata.AnnData) -> None:
        """
        Embeds data by generating random vectors.

        Parameters
        ----------
        adata : anndata.AnnData
            The AnnData object into which we store random vectors.
        """
        embeddings = np.random.rand(adata.n_obs, 64)

        return embeddings


class ContextEmbedder:
    """
    Create a simple CategoryEmbedder or a custom function that places
    initial context embeddings into adata.obsm["X_context_init"].

    For demonstration, we place random embeddings. In practice, you could
    instantiate `CategoryEmbedder(...)`.
    """

    # from mmcontext.pp import CategoryEmbedder
    # category_embedder = CategoryEmbedder(
    #     metadata_categories=["cell_type"],
    #     embeddings_file_path="context_embeddings.pkl",
    #     combination_method="concatenate",
    #     one_hot=False,
    #     unknown_threshold=20,
    # )

    # For testing, just embed random vectors:
    def embed(adata: anndata.AnnData) -> None:
        """
        Embeds context data by generating random vectors.

        Parameters
        ----------
        adata : anndata.AnnData
            The AnnData object into which we store random vectors for context.
        """
        embeddings = np.random.rand(adata.n_obs, 64)

        return embeddings


def test_encoder_pretrained_data_flow(
    encoder_pretrained,
    toy_anndata,
    data_embedder=DataEmbedder,
):
    """
    Tests that the EncoderPreTrained class can embed data in two stages
    and store the final embeddings in adata.obsm["X_data_final"].

    The data here is synthetic random data. The initial embeddings
    are produced by a simple function that writes random vectors
    into `adata.obsm["X_data_init"]`.

    Parameters
    ----------
    encoder_pretrained : EncoderPreTrained
        The instance of EncoderPreTrained with loaded data/context encoders.
    toy_anndata : anndata.AnnData
        Synthetic AnnData used for testing.
    data_embedder : Callable[[anndata.AnnData], None]
        A function or object that places initial data embeddings into
        adata.obsm["X_data_init"].

    Returns
    -------
    None
        Asserts that final data embeddings exist in `adata.obsm["X_data_final"]`.
    """
    # Run the two-stage data embedding
    encoder_pretrained.encode_data(
        adata=toy_anndata, data_embedder=data_embedder, data_source_info="Synthetic random data for testing"
    )
    # Verify that final embeddings exist and have correct shape
    assert "X_data_final" in toy_anndata.obsm, "Final data embeddings not found in adata.obsm."
    final_shape = toy_anndata.obsm["X_data_final"].shape
    logger.info(f"Data final embedding shape: {final_shape}")
    assert final_shape[0] == toy_anndata.n_obs, "Row count of final embeddings should match n_obs."


def test_encoder_pretrained_context_flow(
    encoder_pretrained,
    toy_anndata,
    context_embedder=ContextEmbedder,
):
    """
    Tests that the EncoderPreTrained class can embed context in two stages
    and store the final embeddings in adata.obsm["X_context_final"].

    The data here is synthetic random data. The initial embeddings
    are produced by a simple function that writes random vectors
    into `adata.obsm["X_context_init"]`.

    Parameters
    ----------
    encoder_pretrained : EncoderPreTrained
        The instance of EncoderPreTrained with loaded data/context encoders.
    toy_anndata : anndata.AnnData
        Synthetic AnnData used for testing.
    context_embedder : Callable[[anndata.AnnData], None]
        A function or object that places initial context embeddings into
        adata.obsm["X_context_init"].

    Returns
    -------
    None
        Asserts that final context embeddings exist in `adata.obsm["X_context_final"]`.
    """
    # Run the two-stage context embedding
    encoder_pretrained.encode_context(
        adata=toy_anndata, context_embedder=context_embedder, context_source_info="Synthetic random context data"
    )
    # Verify that final embeddings exist and have correct shape
    assert "X_context_final" in toy_anndata.obsm, "Final context embeddings not found in adata.obsm."
    final_shape = toy_anndata.obsm["X_context_final"].shape
    logger.info(f"Context final embedding shape: {final_shape}")
    assert final_shape[0] == toy_anndata.n_obs, "Row count of final embeddings should match n_obs."


def test_encoder_pretrained_missing_data_encoder(
    data_context_encoders, weights_paths, toy_anndata, data_embedder=DataEmbedder
):
    """
    Tests that missing the 'data_encoder' key in the encoders dictionary
    raises an appropriate ValueError.

    Parameters
    ----------
    data_context_encoders : dict
        Dictionary containing "data_encoder" and "context_encoder" MMContextEncoders.
    weights_paths : dict
        Dictionary containing paths to data and context encoder weights.
    toy_anndata : anndata.AnnData
        Synthetic AnnData for testing.
    data_embedder : Callable[[anndata.AnnData], None]
        A function that produces initial data embeddings.
    """
    from mmcontext.engine import EncoderPreTrained  # <--- Adjust to your actual import path

    # Remove "data_encoder" to simulate the error
    encoders_modified = {"context_encoder": data_context_encoders["context_encoder"]}

    with pytest.raises(ValueError, match="No 'data_encoder' found"):
        encoder_pretrained = EncoderPreTrained(encoders_modified, weights_paths)
        encoder_pretrained.encode_data(toy_anndata, data_embedder)
