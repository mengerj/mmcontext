import logging

import numpy as np
import pytest
from torch.utils.data import DataLoader

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
    dataset_constructor = DataSetConstructor(
        batch_size=16, chunk_size=16, out_emb_keys={"data_embedding": "d_emb", "context_embedding": "c_emb"}
    )  # batch size arbiraty here
    with pytest.raises(ValueError) as exc_info:
        dataset_constructor.add_anndata(
            adata,
            emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
            sample_id_key="sample_id",
        )
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

    dataset_constructor = DataSetConstructor(
        batch_size=16, chunk_size=16, out_emb_keys={"data_embedding": "d_emb", "context_embedding": "c_emb"}
    )
    dataset_constructor.add_anndata(
        adata1,
        emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
        sample_id_key="sample_id",
    )
    dataset_constructor.add_anndata(
        adata2,
        emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
        sample_id_key="sample_id",
    )

    # Adding adata3 should raise a ValueError
    with pytest.raises(ValueError) as exc_info:
        dataset_constructor.add_anndata(
            adata3,
            emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
            sample_id_key="sample_id",
        )
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

    dataset_constructor = DataSetConstructor(
        batch_size=16, chunk_size=16, out_emb_keys={"data_embedding": "d_emb", "context_embedding": "c_emb"}
    )

    with pytest.raises(ValueError) as exc_info:
        dataset_constructor.add_anndata(
            adata,
            emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
            sample_id_key="sample_id",
        )
    assert "Sample ID" in str(exc_info.value) and "integer" in str(exc_info.value)


def test_redundant_sample_ids():
    """Test that adding AnnData objects with overlapping sample IDs raises a ValueError."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_redundant_sample_ids")
    # First AnnData object
    adata1 = create_test_emb_anndata(n_samples=100, emb_dim=64)
    # Second AnnData object with overlapping sample IDs
    adata2 = create_test_emb_anndata(n_samples=50, emb_dim=64, sample_ids=np.arange(50, 100))

    dataset_constructor = DataSetConstructor(
        batch_size=16, chunk_size=16, out_emb_keys={"data_embedding": "d_emb", "context_embedding": "c_emb"}
    )
    dataset_constructor.add_anndata(
        adata1,
        emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
        sample_id_key="sample_id",
    )

    with pytest.raises(ValueError) as exc_info:
        dataset_constructor.add_anndata(
            adata2,
            emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
            sample_id_key="sample_id",
        )
    assert "Duplicate sample IDs found" in str(exc_info.value)


def test_embeddings_correspond_to_sample_ids():
    """Test that retrieving a sample from the EmbeddingDataset corresponds to the embeddings in the original AnnData object."""

    logger = logging.getLogger(__name__)
    logger.info("TEST: test_embeddings_correspond_to_sample_ids")
    n_samples = 100
    emb_dim = 64

    adata = create_test_emb_anndata(n_samples=n_samples, emb_dim=emb_dim)

    # Initialize DataSetConstructor
    dataset_constructor = DataSetConstructor(
        batch_size=16, chunk_size=16, out_emb_keys={"data_embedding": "d_emb", "context_embedding": "c_emb"}
    )
    dataset_constructor.add_anndata(
        adata,
        emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
        sample_id_key="sample_id",
    )
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
        data_embedding = sample["d_emb"].numpy()
        context_embedding = sample["c_emb"].numpy()

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

    dataset_constructor = DataSetConstructor(
        batch_size=16, chunk_size=16, out_emb_keys={"data_embedding": "d_emb", "context_embedding": "c_emb"}
    )
    dataset_constructor.add_anndata(
        adata1,
        emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
        sample_id_key="sample_id",
    )
    dataset_constructor.add_anndata(
        adata2,
        emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
        sample_id_key="sample_id",
    )

    dataset = dataset_constructor.construct_dataset()

    # Verify the dataset length
    assert len(dataset) == 150

    # Verify embeddings dimensions
    sample = dataset[0]
    assert sample["d_emb"].shape[0] == 64
    assert sample["c_emb"].shape[0] == 64


def test_embedding_dataset_length_and_getitem():
    """Test that the EmbeddingDataset correctly implements __len__ and __getitem__ methods."""
    n_samples = 100
    emb_dim = 64
    adata = create_test_emb_anndata(n_samples=n_samples, emb_dim=emb_dim)

    dataset_constructor = DataSetConstructor(
        batch_size=16, chunk_size=16, out_emb_keys={"data_embedding": "d_emb", "context_embedding": "c_emb"}
    )
    dataset_constructor.add_anndata(
        adata,
        emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
        sample_id_key="sample_id",
    )
    dataset = dataset_constructor.construct_dataset()

    # Verify dataset length
    assert len(dataset) == n_samples

    # Verify that items can be retrieved without errors
    for idx in range(0, n_samples, 10):  # Test every 10th sample
        sample = dataset[idx]
        assert "d_emb" in sample
        assert "c_emb" in sample
        assert "sample_id" in sample
        assert sample["d_emb"].shape[0] == emb_dim
        assert sample["c_emb"].shape[0] == emb_dim


def test_dataset_construction_with_sequences():
    """Test that the DataSetConstructor successfully constructs a dataset with sequences."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_dataset_construction_with_sequences")

    # First AnnData object
    adata1 = create_test_emb_anndata(n_samples=128, emb_dim=64)
    # Second AnnData object
    adata2 = create_test_emb_anndata(n_samples=96, emb_dim=64, sample_ids=np.arange(128, 224))

    seq_length = 8  # Sequence length
    dataset_constructor = DataSetConstructor(
        batch_size=4, chunk_size=4 * seq_length, out_emb_keys={"data_embedding": "d_emb", "context_embedding": "c_emb"}
    )
    dataset_constructor.add_anndata(
        adata1,
        emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
        sample_id_key="sample_id",
    )
    dataset_constructor.add_anndata(
        adata2,
        emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
        sample_id_key="sample_id",
    )

    dataset = dataset_constructor.construct_dataset(seq_length=seq_length)

    # Verify the dataset length (number of sequences)
    total_samples = 128 + 96  # 224 samples
    expected_num_sequences = total_samples // seq_length  # Should be 7 sequences (truncated to fit)
    assert len(dataset) == expected_num_sequences

    # Verify embeddings dimensions
    sample = dataset[0]
    assert sample["d_emb"].shape == (seq_length, 64)
    assert sample["c_emb"].shape == (seq_length, 64)
    assert sample["sample_id"].shape == (seq_length,)


def test_dataloader_with_individual_samples():
    """Test that a DataLoader can be constructed and used with individual samples."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_dataloader_with_individual_samples")

    # Create test AnnData objects
    adata1 = create_test_emb_anndata(n_samples=100, emb_dim=64)
    adata2 = create_test_emb_anndata(n_samples=50, emb_dim=64, sample_ids=np.arange(100, 150))

    batch_size = 16
    dataset_constructor = DataSetConstructor(
        batch_size=batch_size, chunk_size=16, out_emb_keys={"data_embedding": "d_emb", "context_embedding": "c_emb"}
    )
    dataset_constructor.add_anndata(
        adata1,
        emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
        sample_id_key="sample_id",
    )
    dataset_constructor.add_anndata(
        adata2,
        emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
        sample_id_key="sample_id",
    )

    # Construct dataset without sequences
    dataset = dataset_constructor.construct_dataset(seq_length=None)

    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Iterate over DataLoader and verify batch shapes
    for batch in data_loader:
        data_embeddings = batch["d_emb"]  # Shape: (batch_size, embedding_dim)
        context_embeddings = batch["c_emb"]  # Shape: (batch_size, embedding_dim)
        sample_ids = batch["sample_id"]  # Shape: (batch_size,)

        assert data_embeddings.shape == (batch_size, 64)
        assert context_embeddings.shape == (batch_size, 64)
        assert sample_ids.shape == (batch_size,)
        break  # Only need to check the first batch


def test_dataloader_with_sequences():
    """Test that a DataLoader can be constructed and used with sequences."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_dataloader_with_sequences")

    # Create test AnnData objects
    adata1 = create_test_emb_anndata(n_samples=128, emb_dim=64)
    adata2 = create_test_emb_anndata(n_samples=96, emb_dim=64, sample_ids=np.arange(128, 224))

    batch_size = 4  # Number of sequences per batch
    seq_length = 32
    dataset_constructor = DataSetConstructor(
        batch_size=batch_size,
        chunk_size=batch_size * seq_length,
        out_emb_keys={"data_embedding": "d_emb", "context_embedding": "c_emb"},
    )
    dataset_constructor.add_anndata(
        adata1,
        emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
        sample_id_key="sample_id",
    )
    dataset_constructor.add_anndata(
        adata2,
        emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
        sample_id_key="sample_id",
    )

    dataset = dataset_constructor.construct_dataset(seq_length=seq_length)

    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Iterate over DataLoader and verify batch shapes
    for batch in data_loader:
        data_embeddings = batch["d_emb"]  # Shape: (batch_size, seq_length, embedding_dim)
        context_embeddings = batch["c_emb"]  # Shape: (batch_size, seq_length, embedding_dim)
        sample_ids = batch["sample_id"]  # Shape: (batch_size, seq_length)

        assert data_embeddings.shape == (batch_size, seq_length, 64)
        assert context_embeddings.shape == (batch_size, seq_length, 64)
        assert sample_ids.shape == (batch_size, seq_length)
        break  # Only need to check the first batch
