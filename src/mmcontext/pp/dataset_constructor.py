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

    def construct_dataset(
        self,
        seq_length: int = None,
        data_emb_key="data_embedding",
        context_emb_key="context_embedding",
        sample_id_key="sample_id",
    ) -> Dataset:
        """
        Constructs and returns a PyTorch Dataset combining all added embeddings.

        Args:
            seq_length (int, optional): Length of sequences. If provided, data will be divided into sequences of this length.
            data_emb_key (str): Key for data embeddings in the Dataset. Default is `'data_embedding'`.
            context_emb_key (str): Key for context embeddings in the Dataset. Default is `'context_embedding'`.
            sample_id_key (str): Key for sample IDs in the Dataset. Default is `'sample_id'`.

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

        if seq_length is not None:
            # Use the create_sequences method to split data into sequences
            (
                data_embeddings_seq,
                context_embeddings_seq,
                sample_ids_seq,
            ) = self.create_sequences(data_embeddings, context_embeddings, sample_ids, seq_length)
            # Create and return a PyTorch Dataset with sequences
            dataset = EmbeddingDataset(
                data_embeddings_seq,
                context_embeddings_seq,
                sample_ids_seq,
                seq_length=seq_length,
                data_emb_key=data_emb_key,
                context_emb_key=context_emb_key,
                sample_id_key=sample_id_key,
            )
        else:
            # Create and return a PyTorch Dataset with individual samples
            dataset = EmbeddingDataset(
                data_embeddings,
                context_embeddings,
                sample_ids,
                seq_length=None,
                data_emb_key=data_emb_key,
                context_emb_key=context_emb_key,
                sample_id_key=sample_id_key,
            )

        return dataset

    def initialize_dataset(self):
        """Resets the dataset to be empty."""
        self.data_embeddings = []
        self.context_embeddings = []
        self.sample_ids = []

    def create_sequences(self, data_embeddings, context_embeddings, sample_ids, seq_length=64):
        """
        Splits embeddings and sample IDs into sequences of a specified length.

        Args:
        data_embeddings (np.ndarray): Data embeddings of shape (num_samples, embedding_dim).
        context_embeddings (np.ndarray): Context embeddings of shape (num_samples, embedding_dim).
        sample_ids (np.ndarray): Sample IDs of shape (num_samples,).
        seq_length (int): The desired sequence length.

        Returns
        -------
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
                - data_embeddings_seq: Data embeddings reshaped into sequences.
                - context_embeddings_seq: Context embeddings reshaped into sequences.
                - sample_ids_seq: Sample IDs reshaped into sequences.
        """
        num_samples = data_embeddings.shape[0]
        num_sequences = num_samples // seq_length

        # Calculate the total number of samples that can be evenly divided into sequences
        total_samples = num_sequences * seq_length

        # Truncate the arrays to have total_samples
        data_embeddings = data_embeddings[:total_samples]
        context_embeddings = context_embeddings[:total_samples]
        sample_ids = sample_ids[:total_samples]

        # Reshape into sequences
        data_embeddings_seq = data_embeddings.reshape(num_sequences, seq_length, -1)
        context_embeddings_seq = context_embeddings.reshape(num_sequences, seq_length, -1)
        sample_ids_seq = sample_ids.reshape(num_sequences, seq_length)

        return data_embeddings_seq, context_embeddings_seq, sample_ids_seq


class EmbeddingDataset(Dataset):
    """A PyTorch Dataset class for embeddings, supporting both individual samples and sequences."""

    def __init__(
        self,
        data_embeddings: np.ndarray,
        context_embeddings: np.ndarray,
        sample_ids: np.ndarray,
        seq_length: int = None,
        data_emb_key: str = "data_embedding",
        context_emb_key: str = "context_embedding",
        sample_id_key: str = "sample_id",
    ):
        """
        Initializes the Dataset with data and context embeddings.

        Args:
            data_embeddings (np.ndarray): Data embeddings of shape (num_samples, embedding_dim) or (num_sequences, seq_length, embedding_dim).
            context_embeddings (np.ndarray): Context embeddings of shape matching data_embeddings.
            sample_ids (np.ndarray): Sample IDs corresponding to the embeddings.
            seq_length (int, optional): Length of sequences. If provided, data will be divided into sequences of this length.
        """
        self.seq_length = seq_length

        self.data_emb_key = data_emb_key
        self.context_emb_key = context_emb_key
        self.sample_id_key = sample_id_key
        # Convert numpy arrays to torch tensors
        self.data_embeddings = torch.tensor(data_embeddings, dtype=torch.float32)
        self.context_embeddings = torch.tensor(context_embeddings, dtype=torch.float32)
        self.sample_ids = sample_ids  # Keep as numpy array for IDs

        # Determine data structure (individual samples or sequences)
        if seq_length is not None:
            # If seq_length is provided, ensure data is in sequences
            if self.data_embeddings.ndim != 3:
                raise ValueError("Data embeddings must be 3-dimensional when seq_length is specified.")
            self.num_sequences = self.data_embeddings.shape[0]
        else:
            # If seq_length is not provided, data can be individual samples
            if self.data_embeddings.ndim != 2:
                raise ValueError("Data embeddings must be 2-dimensional when seq_length is not specified.")

    def __len__(self):
        if self.seq_length is not None:
            return self.num_sequences
        else:
            return len(self.sample_ids)

    def __getitem__(self, idx):
        if self.seq_length is not None:
            # Return sequence
            data_seq = self.data_embeddings[idx]
            context_seq = self.context_embeddings[idx]
            sample_ids_seq = self.sample_ids[idx]

            return {
                self.data_emb_key: data_seq,  # Shape: (seq_length, embedding_dim)
                self.context_emb_key: context_seq,  # Shape: (seq_length, embedding_dim)
                self.sample_id_key: sample_ids_seq,  # Shape: (seq_length,)
            }
        else:
            # Return individual sample
            data_sample = self.data_embeddings[idx]
            context_sample = self.context_embeddings[idx]
            sample_id = self.sample_ids[idx]

            return {
                self.data_emb_key: data_sample,  # Shape: (embedding_dim,)
                self.context_emb_key: context_sample,  # Shape: (embedding_dim,)
                self.sample_id_key: sample_id,  # Scalar
            }
