import logging

import numpy as np
import pytest

from mmcontext.pp import DataSetConstructor

# , EmbeddingDataset
from mmcontext.utils import create_test_anndata, create_test_emb_anndata


def test_inconsistent_dimensions_within_anndata():
    """
    Test that adding an AnnData object with inconsistent dimensions between data and context embeddings
    raises a ValueError.
    """
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_inconsistent_dimensions_within_anndata")
    n_samples = 100
    data_emb_dim = 64
    context_emb_dim = 32  # Different dimension
    adata = create_test_anndata(n_samples=n_samples)
    adata.obsm["d_emb_aligned"] = np.random.rand(n_samples, data_emb_dim)
    adata.obsm["c_emb_aligned"] = np.random.rand(n_samples, context_emb_dim)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    dataset_constructor = DataSetConstructor()
    with pytest.raises(ValueError) as exc_info:
        dataset_constructor.add_anndata(adata)
    assert "different dimensions" in str(exc_info.value)


def test_dimension_consistency_across_anndatas():
    """
    Test that the DataSetConstructor raises an error when adding embeddings with inconsistent dimensions across AnnData objects.
    """
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_dimension_consistency_across_anndatas")
    # First AnnData object with embeddings of dimension 64
    adata1 = create_test_emb_anndata(n_samples=100, emb_dim=64)
    # Second AnnData object with embeddings of dimension 64
    # sample ids in range 100-149
    adata2 = create_test_emb_anndata(n_samples=50, emb_dim=64, sample_ids=np.arange(100, 150))
    # Third AnnData object with embeddings of dimension 32 (inconsistent)
    adata3 = create_test_emb_anndata(n_samples=30, emb_dim=32, sample_ids=np.arange(150, 180))

    dataset_constructor = DataSetConstructor()
    dataset_constructor.add_anndata(adata1)
    dataset_constructor.add_anndata(adata2)

    # Adding adata3 should raise a ValueError
    with pytest.raises(ValueError) as exc_info:
        dataset_constructor.add_anndata(adata3)
    assert "Inconsistent embedding dimensions" in str(exc_info.value)


def test_sample_ids_are_integers():
    """Test that the DataSetConstructor raises an error when sample IDs are not integers."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_sample_ids_are_integers")
    n_samples = 100
    emb_dim = 64
    # Create sample IDs that are not integers
    sample_ids = [f"sample_{i}" for i in range(n_samples)]
    adata = create_test_emb_anndata(n_samples=n_samples, emb_dim=emb_dim, sample_ids=sample_ids)

    dataset_constructor = DataSetConstructor()

    with pytest.raises(ValueError) as exc_info:
        dataset_constructor.add_anndata(adata)
    assert "Sample ID" in str(exc_info.value) and "integer" in str(exc_info.value)


def test_redundant_sample_ids():
    """Test that adding AnnData objects with overlapping sample IDs raises a ValueError."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_redundant_sample_ids")
    # First AnnData object
    adata1 = create_test_emb_anndata(n_samples=100, emb_dim=64)
    # Second AnnData object with overlapping sample IDs
    adata2 = create_test_emb_anndata(n_samples=50, emb_dim=64, sample_ids=np.arange(50, 100))

    dataset_constructor = DataSetConstructor()
    dataset_constructor.add_anndata(adata1)

    with pytest.raises(ValueError) as exc_info:
        dataset_constructor.add_anndata(adata2)
    assert "Duplicate sample IDs found" in str(exc_info.value)


def test_embeddings_correspond_to_sample_ids():
    """Test that retrieving a sample from the EmbeddingDataset corresponds to the embeddings in the original AnnData object."""

    logger = logging.getLogger(__name__)
    logger.info("TEST: test_embeddings_correspond_to_sample_ids")
    n_samples = 100
    emb_dim = 64

    adata = create_test_emb_anndata(n_samples=n_samples, emb_dim=emb_dim)

    # Initialize DataSetConstructor
    dataset_constructor = DataSetConstructor()
    dataset_constructor.add_anndata(adata)
    dataset = dataset_constructor.construct_dataset()

    # Verify that the sample IDs in the dataset match those in adata.obs
    dataset_sample_ids = dataset.sample_ids
    adata_sample_ids = adata.obs["sample_id"].values

    # Ensure that the sample IDs are the same
    np.testing.assert_array_equal(dataset_sample_ids.astype(int), adata_sample_ids)

    # Test a few random samples
    indices_to_test = [0, 25, 50, 75, 99]  # Or choose random indices

    for idx in indices_to_test:
        sample = dataset[idx]
        sample_id = sample["sample_id"]
        data_embedding = sample["data_embedding"].numpy()
        context_embedding = sample["context_embedding"].numpy()

        # Since sample IDs are integers and correspond to their indices, we can use sample_id directly
        adata_idx = sample_id  # Assuming sample_id equals the index in adata

        # Retrieve embeddings from adata
        adata_data_embedding = adata.obsm["d_emb_aligned"][adata_idx]
        adata_context_embedding = adata.obsm["c_emb_aligned"][adata_idx]

        # Verify that embeddings match
        np.testing.assert_array_almost_equal(data_embedding, adata_data_embedding)
        np.testing.assert_array_almost_equal(context_embedding, adata_context_embedding)


def test_successful_dataset_construction():
    """Test that the DataSetConstructor successfully constructs a dataset when adding multiple AnnData objects
    with consistent dimensions and non-overlapping sample IDs."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_successful_dataset_construction")
    # First AnnData object
    adata1 = create_test_emb_anndata(n_samples=100, emb_dim=64)
    # Second AnnData object
    adata2 = create_test_emb_anndata(n_samples=50, emb_dim=64, sample_ids=np.arange(100, 150))

    dataset_constructor = DataSetConstructor()
    dataset_constructor.add_anndata(adata1)
    dataset_constructor.add_anndata(adata2)

    dataset = dataset_constructor.construct_dataset()

    # Verify the dataset length
    assert len(dataset) == 150

    # Verify embeddings dimensions
    sample = dataset[0]
    assert sample["data_embedding"].shape[0] == 64
    assert sample["context_embedding"].shape[0] == 64


def test_embedding_dataset_length_and_getitem():
    """Test that the EmbeddingDataset correctly implements __len__ and __getitem__ methods."""
    n_samples = 100
    emb_dim = 64
    adata = create_test_emb_anndata(n_samples=n_samples, emb_dim=emb_dim)

    dataset_constructor = DataSetConstructor()
    dataset_constructor.add_anndata(adata)
    dataset = dataset_constructor.construct_dataset()

    # Verify dataset length
    assert len(dataset) == n_samples

    # Verify that items can be retrieved without errors
    for idx in range(0, n_samples, 10):  # Test every 10th sample
        sample = dataset[idx]
        assert "data_embedding" in sample
        assert "context_embedding" in sample
        assert "sample_id" in sample
        assert sample["data_embedding"].shape[0] == emb_dim
        assert sample["context_embedding"].shape[0] == emb_dim