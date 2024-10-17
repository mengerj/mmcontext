import logging

import numpy as np
import pytest

from mmcontext.pp import PCAReducer
from mmcontext.utils import create_test_anndata


def test_pca_reducer_reduce_embeddings():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_pca_reducer_reduce_embeddings")
    # Embeddings larger than latent_dim
    n_samples = 100
    d_emb_dim = 128  # Larger than latent_dim
    c_emb_dim = 100  # Larger than latent_dim
    latent_dim = 64

    # Generate random embeddings
    np.random.seed(0)
    d_emb = np.random.rand(n_samples, d_emb_dim)
    c_emb = np.random.rand(n_samples, c_emb_dim)

    # Create AnnData object
    adata = create_test_anndata(n_samples=n_samples)
    adata.obsm["d_emb_norm"] = d_emb
    adata.obsm["c_emb_norm"] = c_emb

    # Initialize PCAReducer
    aligner = PCAReducer(latent_dim=latent_dim)

    # Align embeddings
    aligner.align(adata)

    # Assert dimensions are reduced to latent_dim
    assert adata.obsm["d_emb_aligned"].shape == (n_samples, latent_dim)
    assert adata.obsm["c_emb_aligned"].shape == (n_samples, latent_dim)


def test_pca_reducer_extend_embeddings():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_pca_reducer_extend_embeddings")
    # Embeddings smaller than latent_dim
    n_samples = 100
    d_emb_dim = 32  # Smaller than latent_dim
    c_emb_dim = 50  # Smaller than latent_dim
    latent_dim = 64

    # Generate random embeddings
    np.random.seed(1)
    d_emb = np.random.rand(n_samples, d_emb_dim)
    c_emb = np.random.rand(n_samples, c_emb_dim)

    # Create AnnData object
    adata = create_test_anndata(n_samples=n_samples)
    adata.obsm["d_emb_norm"] = d_emb
    adata.obsm["c_emb_norm"] = c_emb

    # Initialize PCAReducer
    aligner = PCAReducer(latent_dim=latent_dim)

    # Align embeddings
    aligner.align(adata)

    # Assert dimensions are extended to latent_dim
    assert adata.obsm["d_emb_aligned"].shape == (n_samples, latent_dim)
    assert adata.obsm["c_emb_aligned"].shape == (n_samples, latent_dim)

    # Check that original embeddings are correctly placed
    np.testing.assert_array_equal(adata.obsm["d_emb_aligned"][:, :d_emb_dim], d_emb)
    np.testing.assert_array_equal(adata.obsm["c_emb_aligned"][:, :c_emb_dim], c_emb)

    # Check that padding is zero
    assert np.all(adata.obsm["d_emb_aligned"][:, d_emb_dim:] == 0)
    assert np.all(adata.obsm["c_emb_aligned"][:, c_emb_dim:] == 0)


def test_pca_reducer_embeddings_equal_to_latent_dim():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_pca_reducer_embeddings_equal_to_latent_dim")
    # Embeddings equal to latent_dim
    n_samples = 100
    d_emb_dim = 64  # Equal to latent_dim
    c_emb_dim = 64  # Equal to latent_dim
    latent_dim = 64

    # Generate random embeddings
    np.random.seed(2)
    d_emb = np.random.rand(n_samples, d_emb_dim)
    c_emb = np.random.rand(n_samples, c_emb_dim)

    # Create AnnData object
    adata = create_test_anndata(n_samples=n_samples)
    adata.obsm["d_emb_norm"] = d_emb
    adata.obsm["c_emb_norm"] = c_emb

    # Initialize PCAReducer
    aligner = PCAReducer(latent_dim=latent_dim)

    # Align embeddings
    aligner.align(adata)

    # Assert embeddings remain unchanged
    np.testing.assert_array_equal(adata.obsm["d_emb_aligned"], d_emb)
    np.testing.assert_array_equal(adata.obsm["c_emb_aligned"], c_emb)


def test_pca_reducer_custom_embedding_keys():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_pca_reducer_custom_embedding_keys")
    # Embeddings with custom keys
    n_samples = 100
    d_emb_dim = 128
    c_emb_dim = 128
    latent_dim = 64

    # Generate random embeddings
    np.random.seed(3)
    d_emb = np.random.rand(n_samples, d_emb_dim)
    c_emb = np.random.rand(n_samples, c_emb_dim)

    # Create AnnData object with custom keys
    adata = create_test_anndata(n_samples=n_samples)
    adata.obsm["my_d_emb"] = d_emb
    adata.obsm["my_c_emb"] = c_emb

    # Initialize PCAReducer with custom keys
    aligner = PCAReducer(latent_dim=latent_dim, data_key="my_d_emb", context_key="my_c_emb")

    # Align embeddings
    aligner.align(adata)

    # Assert dimensions are reduced to latent_dim
    assert adata.obsm["d_emb_aligned"].shape == (n_samples, latent_dim)
    assert adata.obsm["c_emb_aligned"].shape == (n_samples, latent_dim)


def test_pca_reducer_missing_embeddings():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_pca_reducer_missing_embeddings")
    # No embeddings in adata.obsm
    n_samples = 100
    adata = create_test_anndata(n_samples=n_samples)
    latent_dim = 64

    # Initialize PCAReducer
    aligner = PCAReducer(latent_dim=latent_dim)

    # Expect ValueError when calling align
    with pytest.raises(ValueError) as exc_info:
        aligner.align(adata)

    expected_message = f"Embeddings {aligner.data_key} and {aligner.context_key} must be present in adata.obsm."
    assert expected_message in str(exc_info.value)


def test_pca_reducer_large_dataset():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_pca_reducer_large_dataset")
    # Large dataset
    n_samples = 20000  # Larger than max_samples
    d_emb_dim = 128
    c_emb_dim = 128
    latent_dim = 64
    max_samples = 10000

    # Generate random embeddings
    np.random.seed(4)
    d_emb = np.random.rand(n_samples, d_emb_dim)
    c_emb = np.random.rand(n_samples, c_emb_dim)

    # Create AnnData object
    adata = create_test_anndata(n_samples=n_samples)
    adata.obsm["d_emb_norm"] = d_emb
    adata.obsm["c_emb_norm"] = c_emb

    # Initialize PCAReducer
    aligner = PCAReducer(latent_dim=latent_dim, max_samples=max_samples)

    # Align embeddings
    aligner.align(adata)

    # Assert dimensions are reduced to latent_dim
    assert adata.obsm["d_emb_aligned"].shape == (n_samples, latent_dim)
    assert adata.obsm["c_emb_aligned"].shape == (n_samples, latent_dim)


def test_pca_reducer_random_state_consistency():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_pca_reducer_random_state_consistency")
    # Same embeddings, same random_state
    n_samples = 100
    d_emb_dim = 128
    c_emb_dim = 128
    latent_dim = 64
    random_state = 42

    # Generate random embeddings
    np.random.seed(5)
    d_emb = np.random.rand(n_samples, d_emb_dim)
    c_emb = np.random.rand(n_samples, c_emb_dim)

    # Create two AnnData objects with the same embeddings
    adata1 = create_test_anndata(n_samples=n_samples)
    adata1.obsm["d_emb"] = d_emb.copy()
    adata1.obsm["c_emb"] = c_emb.copy()

    adata2 = create_test_anndata(n_samples=n_samples)
    adata2.obsm["d_emb"] = d_emb.copy()
    adata2.obsm["c_emb"] = c_emb.copy()

    # Initialize PCAReducer with same random_state
    aligner1 = PCAReducer(latent_dim=latent_dim, random_state=random_state, data_key="d_emb", context_key="c_emb")
    aligner2 = PCAReducer(latent_dim=latent_dim, random_state=random_state, data_key="d_emb", context_key="c_emb")

    # Align embeddings
    aligner1.align(adata1)
    aligner2.align(adata2)

    # Assert that the reduced embeddings are the same
    np.testing.assert_array_almost_equal(adata1.obsm["d_emb_aligned"], adata2.obsm["d_emb_aligned"])


def test_pca_reducer_zero_variance_dimensions():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_pca_reducer_zero_variance_dimensions")
    # Embeddings with zero variance in some dimensions
    n_samples = 100
    d_emb_dim = 128
    c_emb_dim = 128
    latent_dim = 64

    # Generate embeddings with zero variance in first 5 dimensions
    np.random.seed(6)
    d_emb = np.random.rand(n_samples, d_emb_dim)
    d_emb[:, :5] = 0.5  # Zero variance dimensions
    c_emb = np.random.rand(n_samples, c_emb_dim)
    # Create AnnData object
    adata = create_test_anndata(n_samples=n_samples)
    adata.obsm["d_emb_norm"] = d_emb  # _norm is the default key
    adata.obsm["c_emb_norm"] = c_emb

    # Initialize PCAReducer
    aligner = PCAReducer(latent_dim=latent_dim)

    # Align embeddings
    aligner.align(adata)

    # Assert dimensions are reduced to latent_dim
    assert adata.obsm["d_emb_aligned"].shape == (n_samples, latent_dim)
