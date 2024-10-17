# mmcontext/pp/data_set_constructor.py


import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from torch.utils.data import Dataset


class DataSetConstructor:
    """
    Constructs datasets from aligned embeddings stored in AnnData objects.

    Allows combining multiple AnnData objects into a single dataset suitable for use with PyTorch DataLoaders.
    Ensures sample ID consistency and that data and context embeddings have the same dimensions.
    """

    def __init__(
        self, data_key: str = "d_emb_aligned", context_key: str = "c_emb_aligned", sample_id_key: str = "sample_id"
    ):
        """
        Initializes an empty dataset and sets the keys for data and context embeddings.

        Args:
            data_key (str): Key for data embeddings in `adata.obsm`. Default is `'d_emb_aligned'`.
            context_key (str): Key for context embeddings in `adata.obsm`. Default is `'c_emb_aligned'`.
        """
        self.data_key = data_key
        self.context_key = context_key
        self.sample_id_key = sample_id_key
        self.data_embeddings: list[np.ndarray] = []
        self.context_embeddings: list[np.ndarray] = []
        self.sample_ids: list[str] = []

    def add_anndata(self, adata: AnnData):
        """
        Adds data from an AnnData object into the dataset.

        Args:
            adata (AnnData): The AnnData object containing data and context embeddings.

        Raises
        ------
            ValueError: If embeddings are missing, dimensions do not match, or sample IDs are inconsistent.
        """
        # Check that the embeddings exist
        if self.data_key not in adata.obsm:
            raise ValueError(f"Data embeddings '{self.data_key}' not found in adata.obsm.")
        if self.context_key not in adata.obsm:
            raise ValueError(f"Context embeddings '{self.context_key}' not found in adata.obsm.")

        # Retrieve embeddings
        d_emb = adata.obsm[self.data_key]
        c_emb = adata.obsm[self.context_key]

        # Check that the embeddings have the same number of samples
        if d_emb.shape[0] != c_emb.shape[0]:
            raise ValueError("Data and context embeddings have different numbers of samples.")

        # Check that the embeddings have the same dimensions
        if d_emb.shape[1] != c_emb.shape[1]:
            raise ValueError("Data and context embeddings have different dimensions.")

        # If embeddings have been previously added, check that dimensions match
        if self.data_embeddings:
            expected_dim = self.data_embeddings[0].shape[1]
            if d_emb.shape[1] != expected_dim:
                raise ValueError(
                    f"Inconsistent embedding dimensions: expected {expected_dim}, but got {d_emb.shape[1]}."
                )

        # Verify sample IDs are integers
        sample_ids = adata.obs[self.sample_id_key].tolist()
        try:
            pd.to_numeric(adata.obs[self.sample_id_key], downcast="integer")  # Convert to integer
        except ValueError as err:
            raise ValueError(
                f"Sample IDs must be integers. Non-integer values found in '{self.sample_id_key}' column."
            ) from err

        # Verify sample IDs consistency
        if len(self.sample_ids) > 0:
            # Verify that there are no duplicate sample IDs
            duplicate_ids = set(self.sample_ids).intersection(set(sample_ids))
            if duplicate_ids:
                raise ValueError(f"Duplicate sample IDs found: {duplicate_ids}")

        # Append embeddings and sample IDs
        self.data_embeddings.append(d_emb)
        self.context_embeddings.append(c_emb)
        self.sample_ids.extend(sample_ids)

    def construct_dataset(self) -> Dataset:
        """
        Constructs and returns a PyTorch Dataset combining all added embeddings.

        Returns
        -------
            Dataset: A PyTorch Dataset containing data embeddings, context embeddings, and sample IDs.

        Raises
        ------
            ValueError: If no data has been added.
        """
        if not self.data_embeddings or not self.context_embeddings:
            raise ValueError("No data has been added to the dataset.")

        # Concatenate all embeddings
        data_embeddings = np.vstack(self.data_embeddings)
        context_embeddings = np.vstack(self.context_embeddings)
        sample_ids = np.array(self.sample_ids)

        # Create and return a PyTorch Dataset
        dataset = EmbeddingDataset(data_embeddings, context_embeddings, sample_ids)
        return dataset

    def initialize_dataset(self):
        """Resets the dataset to be empty."""
        self.data_embeddings = []
        self.context_embeddings = []
        self.sample_ids = []


class EmbeddingDataset(Dataset):
    """
    A PyTorch Dataset class for embeddings.

    This class provides an interface to access data and context embeddings along with their sample IDs.
    It can be used to create a DataLoader for training PyTorch models.
    """

    def __init__(self, data_embeddings: np.ndarray, context_embeddings: np.ndarray, sample_ids: np.ndarray):
        """
        Initializes the Dataset with data and context embeddings.

        Args:
            data_embeddings (np.ndarray): Data embeddings.
            context_embeddings (np.ndarray): Context embeddings.
            sample_ids (np.ndarray): Sample IDs corresponding to the embeddings.
        """
        self.data_embeddings = torch.tensor(data_embeddings, dtype=torch.float32)
        self.context_embeddings = torch.tensor(context_embeddings, dtype=torch.float32)
        self.sample_ids = sample_ids

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns
        -------
            int: Number of samples.
        """
        return len(self.sample_ids)

    def __getitem__(self, idx):
        """
        Retrieves the data at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns
        -------
            dict: A dictionary containing:
                - 'data_embedding': The data embedding tensor at index `idx`.
                - 'context_embedding': The context embedding tensor at index `idx`.
                - 'sample_id': The sample ID corresponding to index `idx`.
        """
        return {
            "data_embedding": self.data_embeddings[idx],
            "context_embedding": self.context_embeddings[idx],
            "sample_id": self.sample_ids[idx],
        }
