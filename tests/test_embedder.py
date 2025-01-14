# tests/test_embedder.py

import importlib.resources
import logging
import shutil

import numpy as np
import pytest

from mmcontext.pp.context_embedder import CategoryEmbedder, PlaceholderContextEmbedder
from mmcontext.pp.data_embedder import PlaceholderDataEmbedder

# Import the Embedder and embedders from your package
from mmcontext.pp.embedder import Embedder

# Import the utility function to create test AnnData
from mmcontext.utils import create_test_anndata


def test_embedder_with_external_embeddings():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_embedder_with_external_embeddings")
    adata = create_test_anndata()
    n_samples = adata.n_obs
    data_embedding_dim = 64
    context_embedding_dim = 128

    # Create external embeddings
    external_data_embeddings = np.random.rand(n_samples, data_embedding_dim)
    external_context_embeddings = np.random.rand(n_samples, context_embedding_dim)

    # Initialize the Embedder without embedders
    embedder = Embedder()

    # Create embeddings using external embeddings
    embedder.create_embeddings(
        adata, data_embeddings=external_data_embeddings, context_embeddings=external_context_embeddings
    )

    # Assert that embeddings are stored correctly
    assert "d_emb" in adata.obsm
    assert "c_emb" in adata.obsm
    assert adata.obsm["d_emb"].shape == (n_samples, data_embedding_dim)
    assert adata.obsm["c_emb"].shape == (n_samples, context_embedding_dim)
    np.testing.assert_array_equal(adata.obsm["d_emb"], external_data_embeddings)
    np.testing.assert_array_equal(adata.obsm["c_emb"], external_context_embeddings)


def test_embedder_with_embedders():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_embedder_with_embedders")
    adata = create_test_anndata()
    n_samples = adata.n_obs

    # Initialize placeholder embedders
    data_embedder = PlaceholderDataEmbedder()
    context_embedder = PlaceholderContextEmbedder()

    # Initialize the Embedder with embedders
    embedder = Embedder(data_embedder=data_embedder, context_embedder=context_embedder)

    # Create embeddings
    embedder.create_embeddings(adata)

    # Assert that embeddings are stored correctly
    assert "d_emb" in adata.obsm
    assert "c_emb" in adata.obsm
    assert adata.obsm["d_emb"].shape == (n_samples, 64)
    assert adata.obsm["c_emb"].shape == (n_samples, 128)


def test_embedder_with_existing_embeddings():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_embedder_with_existing_embeddings")
    adata = create_test_anndata()
    n_samples = adata.n_obs

    # Create existing embeddings
    existing_data_embeddings = np.random.rand(n_samples, 64)
    existing_context_embeddings = np.random.rand(n_samples, 128)
    adata.obsm["d_emb"] = existing_data_embeddings.copy()
    adata.obsm["c_emb"] = existing_context_embeddings.copy()

    # Initialize embedders
    data_embedder = PlaceholderDataEmbedder()
    context_embedder = PlaceholderContextEmbedder()

    # Initialize the Embedder
    embedder = Embedder(data_embedder=data_embedder, context_embedder=context_embedder)

    # Create embeddings
    embedder.create_embeddings(
        adata, data_embeddings=existing_data_embeddings, context_embeddings=existing_context_embeddings
    )

    # Assert that existing embeddings are not overwritten
    np.testing.assert_array_equal(adata.obsm["d_emb"], existing_data_embeddings)
    np.testing.assert_array_equal(adata.obsm["c_emb"], existing_context_embeddings)


def test_embedder_with_external_context_embeddings_only():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_embedder_with_external_context_embeddings_only")
    adata = create_test_anndata()
    n_samples = adata.n_obs
    context_embedding_dim = 128

    # Create external context embeddings
    external_context_embeddings = np.random.rand(n_samples, context_embedding_dim)

    # Initialize data embedder
    data_embedder = PlaceholderDataEmbedder()

    # Initialize the Embedder
    embedder = Embedder(data_embedder=data_embedder)

    # Create embeddings
    embedder.create_embeddings(adata, context_embeddings=external_context_embeddings)

    # Assert context embeddings are stored correctly
    assert "c_emb" in adata.obsm
    np.testing.assert_array_equal(adata.obsm["c_emb"], external_context_embeddings)

    # Assert data embeddings are generated
    assert "d_emb" in adata.obsm
    assert adata.obsm["d_emb"].shape == (n_samples, 64)


def test_embedder_with_mismatched_external_embeddings():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_embedder_with_mismatched_external_embeddings")
    adata = create_test_anndata()
    n_samples = adata.n_obs
    data_embedding_dim = 64

    # Create external data embeddings with mismatched number of samples
    external_data_embeddings = np.random.rand(n_samples + 1, data_embedding_dim)

    # Initialize the Embedder
    embedder = Embedder()

    # Attempt to create embeddings with mismatched external embeddings
    with pytest.raises(ValueError) as excinfo:
        embedder.create_embeddings(adata, data_embeddings=external_data_embeddings)

    # Check the error message
    assert (
        "The number of samples in the provided d_emb embeddings does not match the number of observations in adata."
        in str(excinfo.value)
    )


def test_embedder_with_non_numpy_embeddings():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_embedder_with_non_numpy_embeddings")
    adata = create_test_anndata()
    n_samples = adata.n_obs
    data_embedding_dim = 64

    # Create external data embeddings as a list (not a NumPy array)
    external_data_embeddings = [[0.1] * data_embedding_dim] * n_samples  # This is a list of lists

    # Initialize embedders
    data_embedder = PlaceholderDataEmbedder()
    context_embedder = PlaceholderContextEmbedder()

    # Initialize the Embedder
    embedder = Embedder(data_embedder=data_embedder, context_embedder=context_embedder)

    # Attempt to create embeddings with non-NumPy array embeddings
    with pytest.raises(TypeError) as excinfo:
        embedder.create_embeddings(adata, data_embeddings=external_data_embeddings)

    # Check the error message
    assert "The provided d_emb embeddings must be a numpy.ndarray, but got list." in str(excinfo.value)


def test_embedder_with_onehot_context(tmp_path):
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_embedder_with_onehot_context")
    adata = create_test_anndata()
    n_samples = adata.n_obs
    data_embedding_dim = 64

    # Create external context embeddings
    external_data_embeddings = np.random.randint(0, 2, (n_samples, data_embedding_dim))

    metadata_categories = ["cell_type", "tissue"]

    # Access the test dictionary file from package resources
    with importlib.resources.path("mmcontext.data", "test_dict.pkl.gz") as resource_path:
        # Create a temporary directory
        temp_file_path = tmp_path / "test_dict.pkl.gz"

        # Copy the resource file to the temporary location
        shutil.copy(resource_path, temp_file_path)

        context_embedder = CategoryEmbedder(
            metadata_categories=metadata_categories,
            embeddings_file_path=temp_file_path,
            one_hot=True,
            model="text-embedding-3-small",
            combination_method="concatenate",
        )
        embedder = Embedder(context_embedder=context_embedder)

        embedder.create_embeddings(adata, data_embeddings=external_data_embeddings)
        # Confirm the shape of the context embeddings
        assert set(np.unique(adata.obsm["cell_type_emb"])) == {0, 1}
        assert set(np.unique(adata.obsm["tissue_emb"])) == {0, 1}
