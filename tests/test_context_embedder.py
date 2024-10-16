import logging

import anndata
import numpy as np
import pytest

from mmcontext.pp import CategoryEmbedder, Embedder, PlaceholderDataEmbedder
from mmcontext.utils import create_test_anndata


def test_context_embedder_with_missing_obs_columns():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_context_embedder_with_missing_obs_columns")

    adata = create_test_anndata()
    del adata.obs["cell_type"]  # Remove required metadata column

    # Initialize CategoryEmbedder
    metadata_categories = ["cell_type", "tissue"]
    embeddings_file_path = "./data/emb_dicts/test_dict.pkl"
    model = "text-embedding-3-small"
    combination_method = "concatenate"

    context_embedder = CategoryEmbedder(
        metadata_categories=metadata_categories,
        embeddings_file_path=embeddings_file_path,
        model=model,
        combination_method=combination_method,
        one_hot=False,
    )
    data_embedder = PlaceholderDataEmbedder()

    # Initialize the Embedder
    embedder = Embedder(context_embedder=context_embedder, data_embedder=data_embedder)

    # Attempt to create embeddings
    with pytest.raises(ValueError) as excinfo:
        embedder.create_embeddings(adata)

    # Check the error message
    assert "Metadata category 'cell_type' not found in adata.obs." in str(excinfo.value)


def test_embeddings_dictionary_loading(monkeypatch):
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_embeddings_dictionary_loading")
    # Load the original test dataset
    adata = anndata.read_h5ad("data/test_data/test_adata.h5ad")
    # Set the embeddings file path
    embeddings_file_path = "./data/emb_dicts/test_dict.pkl.gz"

    # Initialize the CategoryEmbedder without an API key
    metadata_categories = ["cell_type", "tissue"]
    context_embedder = CategoryEmbedder(
        metadata_categories=metadata_categories,
        embeddings_file_path=embeddings_file_path,
        model="text-embedding-3-small",
        combination_method="concatenate",
        one_hot=False,
        unknown_threshold=20,
    )

    # Remove the API key from the environment (simulate missing API key)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Run the embedder
    context_embedder.embed(adata)

    # Check that embeddings are loaded from the dictionary
    assert "cell_type_emb" in adata.obsm
    assert "tissue_emb" in adata.obsm

    # Verify that no new embeddings were attempted to be generated
    assert len(context_embedder.metadata_embeddings["cell_type"]) == 3  # Should be the original 3 cell types
    assert len(context_embedder.metadata_embeddings["tissue"]) == 2  # Should be the original 2 tissues


def test_unknown_elements_less_than_threshold(monkeypatch):
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_unknown_elements_less_than_threshold")

    # Load the new test dataset with extra categories
    adata = anndata.read_h5ad("data/test_data/new_test_adata.h5ad")

    # Set the embeddings file path
    embeddings_file_path = "./data/emb_dicts/test_dict.pkl"

    # Initialize the CategoryEmbedder without an API key
    metadata_categories = ["cell_type", "tissue"]
    context_embedder = CategoryEmbedder(
        metadata_categories=metadata_categories,
        embeddings_file_path=embeddings_file_path,
        model="text-embedding-3-small",
        combination_method="concatenate",
        one_hot=False,
        unknown_threshold=10,  # Set threshold higher than the number of unknown elements
    )

    # Remove the API key from the environment (simulate missing API key)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Run the embedder
    context_embedder.embed(adata)

    # Check that embeddings are present
    assert "cell_type_emb" in adata.obsm
    assert "tissue_emb" in adata.obsm

    # Verify that the embeddings for unknown elements are zero vectors
    cell_type_embeddings = adata.obsm["cell_type_emb"]
    tissue_embeddings = adata.obsm["tissue_emb"]

    # Find indices of unknown cell types
    unknown_cell_types = ["Dendritic cell", "Monocyte"]
    unknown_indices = adata.obs["cell_type"].isin(unknown_cell_types)

    # Check that the embeddings for unknown cell types are zeros
    assert (cell_type_embeddings[unknown_indices] == 0).all()

    # Similarly for tissues
    unknown_tissues = ["bone marrow"]
    unknown_indices = adata.obs["tissue"].isin(unknown_tissues)
    assert (tissue_embeddings[unknown_indices] == 0).all()


def test_unknown_elements_exceed_threshold(monkeypatch):
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_unknown_elements_exceed_threshold")
    # Create a dataset with many unknown categories
    adata = create_test_anndata(
        n_cells=30,
        n_genes=100,
        cell_types=["CellType" + str(i) for i in range(10)],  # 10 new cell types
        tissues=["Tissue" + str(i) for i in range(5)],  # 5 new tissues
    )

    # Set the embeddings file path
    embeddings_file_path = "./data/emb_dicts/test_dict.pkl"

    # Initialize the CategoryEmbedder without an API key
    metadata_categories = ["cell_type", "tissue"]
    context_embedder = CategoryEmbedder(
        metadata_categories=metadata_categories,
        embeddings_file_path=embeddings_file_path,
        model="text-embedding-3-small",
        combination_method="concatenate",
        one_hot=False,
        unknown_threshold=10,  # Set threshold lower than the number of unknown elements
    )

    # Remove the API key from the environment (simulate missing API key)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Attempt to run the embedder and expect an error
    with pytest.raises(ValueError) as excinfo:
        context_embedder.embed(adata)

    # Check that the error message is as expected
    assert "Unknown elements exceed the threshold" in str(excinfo.value)


def test_one_hot_encoding_metadata():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_one_hot_encoding_metadata")
    adata = create_test_anndata()

    # Initialize the CategoryEmbedder with one-hot encoding enabled
    metadata_categories = ["cell_type", "tissue"]
    embeddings_file_path = "./data/emb_dicts/test_dict.pkl"
    context_embedder = CategoryEmbedder(
        metadata_categories=metadata_categories, embeddings_file_path=embeddings_file_path, one_hot=True
    )

    # Run the embedder
    context_embedder.embed(adata)

    # Check that one-hot encoded embeddings are stored in adata.obsm
    assert "cell_type_emb" in adata.obsm
    assert "tissue_emb" in adata.obsm

    # Verify that the embeddings are indeed one-hot encoded
    assert set(np.unique(adata.obsm["cell_type_emb"])) == {0, 1}
    assert set(np.unique(adata.obsm["tissue_emb"])) == {0, 1}


def test_combination_methods():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_combination_methods")

    adata = create_test_anndata()

    # Initialize the CategoryEmbedder with both combination methods
    metadata_categories = ["cell_type", "tissue"]
    embeddings_file_path = "./data/emb_dicts/test_dict.pkl"

    # Test 'concatenate' method
    context_embedder_concat = CategoryEmbedder(
        metadata_categories=metadata_categories,
        embeddings_file_path=embeddings_file_path,
        model="text-embedding-3-small",
        combination_method="concatenate",
    )
    embeddings_concat = context_embedder_concat.embed(adata)
    assert embeddings_concat.shape[1] == adata.obsm["cell_type_emb"].shape[1] + adata.obsm["tissue_emb"].shape[1]

    # Test 'average' method
    context_embedder_avg = CategoryEmbedder(
        metadata_categories=metadata_categories,
        embeddings_file_path=embeddings_file_path,
        model="text-embedding-3-small",
        combination_method="average",
    )
    embeddings_avg = context_embedder_avg.embed(adata)
    assert embeddings_avg.shape[1] == adata.obsm["cell_type_emb"].shape[1]
