import logging

import numpy as np
import pytest

from mmcontext.pp import MinMaxNormalizer, ZScoreNormalizer
from mmcontext.utils import create_test_anndata


def test_zscore_normalization():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_zscore_normalization")

    adata = create_test_anndata()
    adata.obsm["d_emb"] = np.random.rand(adata.n_obs, 64)
    adata.obsm["c_emb"] = np.random.rand(adata.n_obs, 128)

    normalizer = ZScoreNormalizer()
    normalizer.normalize(adata)

    # Test data embeddings
    d_emb_norm = adata.obsm["d_emb_norm"]
    assert np.allclose(np.mean(d_emb_norm, axis=0), 0, atol=1e-7)
    assert np.allclose(np.std(d_emb_norm, axis=0), 1, atol=1e-7)

    # Test context embeddings
    c_emb_norm = adata.obsm["c_emb_norm"]
    assert np.allclose(np.mean(c_emb_norm, axis=0), 0, atol=1e-7)
    assert np.allclose(np.std(c_emb_norm, axis=0), 1, atol=1e-7)


def test_minmax_normalization():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_minmax_normalization")

    adata = create_test_anndata()
    adata.obsm["d_emb"] = np.random.rand(adata.n_obs, 64)
    adata.obsm["c_emb"] = np.random.rand(adata.n_obs, 128)

    normalizer = MinMaxNormalizer()

    normalizer.normalize(adata)
    d_emb_norm = adata.obsm["d_emb_norm"]
    c_emb_norm = adata.obsm["c_emb_norm"]
    # Verify that all values are within [0, 1]
    assert np.all(d_emb_norm >= 0) and np.all(d_emb_norm <= 1), "d_emb_norm values are not within [0, 1]"
    assert np.all(c_emb_norm >= 0) and np.all(c_emb_norm <= 1), "c_emb_norm values are not within [0, 1]"

    # Check that the min and max are 0 and 1 for each feature
    # For data embeddings
    d_min = np.min(d_emb_norm, axis=0)
    d_max = np.max(d_emb_norm, axis=0)
    assert np.allclose(d_min, 0), "Minimum of d_emb_norm is not 0"
    assert np.allclose(d_max, 1), "Maximum of d_emb_norm is not 1"

    # For context embeddings
    c_min = np.min(c_emb_norm, axis=0)
    c_max = np.max(c_emb_norm, axis=0)
    assert np.allclose(c_min, 0), "Minimum of c_emb_norm is not 0"
    assert np.allclose(c_max, 1), "Maximum of c_emb_norm is not 1"


def test_missing_embeddings():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_missing_embeddings")

    adata = create_test_anndata()  # Create an AnnData object without d_emb or c_emb
    normalizer = ZScoreNormalizer()

    # Test missing d_emb
    with pytest.raises(KeyError):
        normalizer.normalize(adata)

    # Add d_emb and test missing c_emb
    adata.obsm["d_emb"] = np.random.rand(adata.n_obs, 64)
    with pytest.raises(KeyError):
        normalizer.normalize(adata)


# Test for Constant Embedding Values
def test_constant_embeddings_zscore():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_constant_embeddings_zscore")
    # Create an AnnData object with constant embeddings
    adata = create_test_anndata()
    n_samples, emb_dim = adata.n_obs, 64
    adata.obsm["d_emb"] = np.ones((n_samples, emb_dim))
    adata.obsm["c_emb"] = np.ones((n_samples, emb_dim))

    normalizer = ZScoreNormalizer()

    # Expect no error, but normalized embeddings should be NaNs due to division by zero
    with pytest.warns(RuntimeWarning):
        normalizer.normalize(adata)

    assert np.isnan(
        adata.obsm["d_emb_norm"]
    ).all(), "Expected NaN values in d_emb_norm for constant embeddings in ZScoreNormalizer"
    assert np.isnan(
        adata.obsm["c_emb_norm"]
    ).all(), "Expected NaN values in c_emb_norm for constant embeddings in ZScoreNormalizer"


def test_constant_embeddings_minmax():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_constant_embeddings_minmax")
    # Create an AnnData object with constant embeddings
    adata = create_test_anndata()
    n_samples, emb_dim = adata.n_obs, 64
    adata.obsm["d_emb"] = np.ones((n_samples, emb_dim))
    adata.obsm["c_emb"] = np.ones((n_samples, emb_dim))

    normalizer = MinMaxNormalizer()

    # Expect no error, but normalized embeddings should be zeros due to min == max
    normalizer.normalize(adata)
    assert np.isnan(
        adata.obsm["d_emb_norm"]
    ).all(), "Expected NaN values in d_emb_norm for constant embeddings in MinMaxNormalizer"
    assert np.isnan(
        adata.obsm["c_emb_norm"]
    ).all(), "Expected NaN values in c_emb_norm for constant embeddings in MinMaxNormalizer"


def test_extreme_values_zscore():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_extreme_values_zscore")
    # Create an AnnData object with extreme values
    adata = create_test_anndata()
    adata.obsm["d_emb"] = np.random.rand(adata.n_obs, 64)
    adata.obsm["c_emb"] = np.random.rand(adata.n_obs, 128)
    # change one of the values to an extreme values
    adata.obsm["d_emb"][0, 1] = 1e12
    adata.obsm["c_emb"][0, 2] = 2e12

    normalizer = ZScoreNormalizer()

    # Apply normalization, and it should handle extreme values without NaN or Inf
    normalizer.normalize(adata)
    assert np.isfinite(
        adata.obsm["d_emb_norm"]
    ).all(), "ZScoreNormalizer should handle extreme values without NaNs or Infs"
    assert np.isfinite(
        adata.obsm["c_emb_norm"]
    ).all(), "ZScoreNormalizer should handle extreme values without NaNs or Infs"


def test_extreme_values_minmax():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_extreme_values_minmax")

    # Create an AnnData object with extreme values
    adata = create_test_anndata()
    adata.obsm["d_emb"] = np.random.rand(adata.n_obs, 64)
    adata.obsm["c_emb"] = np.random.rand(adata.n_obs, 128)
    # change one of the values to an extreme values
    adata.obsm["d_emb"][0, 1] = 1e12
    adata.obsm["c_emb"][0, 2] = 2e12

    normalizer = MinMaxNormalizer()

    # Apply normalization and check if extreme values are normalized correctly
    normalizer.normalize(adata)

    assert np.isfinite(
        adata.obsm["d_emb_norm"]
    ).all(), "MinMaxNormalizer should handle extreme values without NaNs or Infs"
    assert np.isfinite(
        adata.obsm["c_emb_norm"]
    ).all(), "MinMaxNormalizer should handle extreme values without NaNs or Infs"


def test_mixed_datatypes_zscore():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_mixed_datatypes_zscore")

    # Create an AnnData object with mixed datatype embeddings
    adata = create_test_anndata()
    n_samples, emb_dim = adata.n_obs, 64
    adata.obsm["d_emb"] = np.random.randint(0, 100, (n_samples, emb_dim))
    adata.obsm["c_emb"] = np.random.randint(0, 100, (n_samples, emb_dim))

    normalizer = ZScoreNormalizer()

    # Apply normalization and check if mixed data types are handled properly
    normalizer.normalize(adata)

    assert adata.obsm["d_emb_norm"].dtype == float, "ZScoreNormalizer should convert d_emb to float"
    assert adata.obsm["c_emb_norm"].dtype == float, "ZScoreNormalizer should convert c_emb to float"


def test_mixed_datatypes_minmax():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_mixed_datatypes_minmax")

    # Create an AnnData object with mixed datatype embeddings
    adata = create_test_anndata()
    n_samples, emb_dim = adata.n_obs, 64
    adata.obsm["d_emb"] = np.random.randint(0, 100, (n_samples, emb_dim))
    adata.obsm["c_emb"] = np.random.randint(0, 100, (n_samples, emb_dim))

    normalizer = MinMaxNormalizer()

    # Apply normalization and check if mixed data types are handled properly
    normalizer.normalize(adata)

    assert adata.obsm["d_emb_norm"].dtype == float, "MinMaxNormalizer should convert d_emb to float"
    assert adata.obsm["c_emb_norm"].dtype == float, "MinMaxNormalizer should convert c_emb to float"
