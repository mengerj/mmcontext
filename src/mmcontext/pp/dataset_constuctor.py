import dask
import dask.array as da
import scipy.sparse as sp
import torch
from anndata import AnnData
from torch.utils.data import Dataset


class DataSetConstructor:
    """
    Constructs datasets from aligned embeddings stored in AnnData objects.

    Allows combining multiple AnnData objects into a single dataset suitable for use with PyTorch DataLoaders.
    Ensures sample ID consistency and that data and context embeddings have the same dimensions.

    If non default 'out_keys' are used, make sure to also pass them to the loss function and the trainer.

    Parameters
    ----------
    in_data_key
        Key for data embeddings in the AnnData object.
    in_context_key
        Key for context embeddings in the AnnData object.
    in_sample_id_key
        Key for sample IDs in the AnnData object.
    out_data_key
        Key to be used for data embeddings in the Dataset.
    out_context_key
        Key to be used for context embeddings in the Dataset.
    out_sample_id_key
        Key to be used for sample IDs in the Dataset.
    out_raw_data_key
        Key to be used for raw data in the Dataset.
    """

    def __init__(
        self,
        in_data_key: str = "d_emb_aligned",
        in_context_key: str = "c_emb_aligned",
        in_sample_id_key: str = "sample_id",
        out_data_key: str = "data_embedding",
        out_context_key: str = "context_embedding",
        out_sample_id_key: str = "sample_id",
        out_raw_data_key: str = "raw_data",
        chunk_size: int | None = None,
        batch_size=None,
    ):
        self.in_data_key = in_data_key
        self.in_context_key = in_context_key
        self.in_sample_id_key = in_sample_id_key
        self.out_data_key = out_data_key
        self.out_context_key = out_context_key
        self.out_sample_id_key = out_sample_id_key
        self.out_raw_data_key = out_raw_data_key
        self.data_embeddings = None
        self.context_embeddings = None
        self.sample_ids = None
        self.raw_data = None
        self.chunk_size = chunk_size
        self.total_samples = 0  # Add for each anndata object added to the dataset
        if batch_size is None:
            raise ValueError(
                "Batch size must be specified. Not used here, but important to ensure it is valid for the data size."
            )
        else:
            self.batch_size = batch_size

        if chunk_size is None:
            raise ValueError(
                "Chunk size must be specified. Ideally it should be a multiple of the batch size and the sequence length."
            )

    def add_anndata(self, adata: AnnData):
        """
        Adds data from an AnnData object into the dataset.

        Parameters
        ----------
        adata
            The AnnData object containing data and context embeddings.

        Raises
        ------
        ValueError
            If embeddings are missing, dimensions do not match, or sample IDs are inconsistent.
        """
        # Check that the embeddings exist
        if self.in_data_key not in adata.obsm:
            raise ValueError(f"Data embeddings '{self.in_data_key}' not found in adata.obsm.")
        if self.in_context_key not in adata.obsm:
            raise ValueError(f"Context embeddings '{self.in_context_key}' not found in adata.obsm.")

        # Retrieve embeddings
        d_emb = da.from_array(
            adata.obsm[self.in_data_key], chunks=(adata.obsm[self.in_data_key].shape[0], self.chunk_size)
        )
        c_emb = da.from_array(
            adata.obsm[self.in_context_key], chunks=(adata.obsm[self.in_context_key].shape[0], self.chunk_size)
        )
        # Retrieve raw data
        X = adata.X.toarray() if isinstance(adata.X, sp.spmatrix) else adata.X
        sample_ids = da.from_array(adata.obs[self.in_sample_id_key].values, chunks=self.chunk_size)
        raw_data = da.from_array(X, chunks=(X.shape[0], self.chunk_size))
        # Check that the embeddings have the same number of samples
        if d_emb.shape[0] != c_emb.shape[0]:
            raise ValueError("Data and context embeddings have different numbers of samples.")

        # Check that the embeddings have the same dimensions
        if d_emb.shape[1] != c_emb.shape[1]:
            raise ValueError("Data and context embeddings have different dimensions.")

        if self.data_embeddings is not None:
            expected_dim = self.data_embeddings.shape[1]
            if d_emb.shape[1] != expected_dim:
                raise ValueError(
                    f"Inconsistent embedding dimensions: expected {expected_dim}, but got {d_emb.shape[1]}."
                )
        else:
            # Store the expected dimension for future checks
            expected_dim = d_emb.shape[1]
        # Concatenate with existing arrays
        self.data_embeddings = (
            d_emb if self.data_embeddings is None else da.concatenate([self.data_embeddings, d_emb], axis=0)
        )
        self.context_embeddings = (
            c_emb if self.context_embeddings is None else da.concatenate([self.context_embeddings, c_emb], axis=0)
        )
        self.sample_ids = (
            sample_ids if self.sample_ids is None else da.concatenate([self.sample_ids, sample_ids], axis=0)
        )
        self.raw_data = raw_data if self.raw_data is None else da.concatenate([self.raw_data, raw_data], axis=0)
        self.total_samples += adata.shape[0]
        """
        # Verify sample IDs are integers
        sample_ids = adata.obs[self.in_sample_id_key].tolist()
        try:
            pd.to_numeric(adata.obs[self.in_sample_id_key], downcast="integer")  # Convert to integer
        except ValueError as err:
            raise ValueError(
                f"Sample IDs must be integers. Non-integer values found in '{self.in_sample_id_key}' column."
            ) from err

        # Verify sample IDs consistency
        if len(self.sample_ids) > 0:
            # Verify that there are no duplicate sample IDs
            duplicate_ids = set(self.sample_ids).intersection(set(sample_ids))
            if duplicate_ids:
                raise ValueError(f"Duplicate sample IDs found: {duplicate_ids}")
        """

    def construct_dataset(self, seq_length: int = None, return_indices=False) -> Dataset:
        """
        Constructs and returns a PyTorch Dataset combining all added embeddings.

        Parameters
        ----------
        seq_length
            Length of sequences.
        return_indices
            If True, the dataset will return the index of the sample. Only makes sense if just one anndata object is used, eg. during inference.

        Returns
        -------
        A PyTorch Dataset containing data embeddings, context embeddings, and sample IDs.
        """
        if self.data_embeddings is None or self.context_embeddings is None:
            raise ValueError("No data has been added to the dataset.")
        if seq_length is not None and self.batch_size * seq_length > self.total_samples:
            raise ValueError("Batch size and sequence length are too large for the dataset.")

        # No need to reshape or rechunk the arrays
        dataset = DaskEmbeddingDataset(
            data_embeddings=self.data_embeddings,
            context_embeddings=self.context_embeddings,
            sample_ids=self.sample_ids,
            raw_data=self.raw_data,
            return_indices=return_indices,
            seq_length=seq_length,
            out_data_key=self.out_data_key,
            out_context_key=self.out_context_key,
            out_sample_id_key=self.out_sample_id_key,
            out_raw_data_key=self.out_raw_data_key,
        )

        return dataset


class DaskEmbeddingDataset(Dataset):
    """Custom PyTorch Dataset class is used to load embeddings from Dask arrays.

    Due to the usage of dask arrays, which store computational graphs rather than the actual data, the data is loaded into memory when accessed.

    Parameters
    ----------
    data_embeddings
        Dask array containing data embeddings.
    context_embeddings
        Dask array containing context embeddings.
    sample_ids
        Dask array containing sample IDs.
    raw_data
        Dask array containing raw data.
    return_indices
        If True, the dataset will return the index of the sample. Only makes sense if just one anndata object is used, eg. during inference.
    seq_length
        Length of sequences. If None, the dataset will return individual samples.

    Returns
    -------
    A PyTorch Dataset object.
    """

    def __init__(
        self,
        data_embeddings,
        context_embeddings,
        sample_ids,
        raw_data,
        return_indices=False,
        seq_length=None,
        out_data_key: str = "data_embedding",
        out_context_key: str = "context_embedding",
        out_sample_id_key: str = "sample_id",
        out_raw_data_key: str = "raw_data",
    ):
        self.data_embeddings = data_embeddings
        self.context_embeddings = context_embeddings
        self.sample_ids = sample_ids
        self.raw_data = raw_data
        self.return_indices = return_indices
        self.seq_length = seq_length
        self.out_data_key = out_data_key
        self.out_context_key = out_context_key
        self.out_sample_id_key = out_sample_id_key
        self.out_raw_data_key = out_raw_data_key

        # Compute total number of samples
        self.total_samples = self.data_embeddings.shape[0]

    def __len__(self):
        if self.seq_length is not None:
            return self.total_samples // self.seq_length
        else:
            return self.total_samples

    def __getitem__(self, idx):
        if self.seq_length is not None:
            start_idx = idx * self.seq_length
            end_idx = start_idx + self.seq_length
            return self._get_sequence(start_idx, end_idx)
        else:
            return self._get_sample(idx)

    def _get_sample(self, idx):
        data_emb, context_emb, sample_ids, raw_data = dask.compute(
            self.data_embeddings[idx], self.context_embeddings[idx], self.sample_ids[idx], self.raw_data[idx]
        )
        sample = {
            self.out_data_key: torch.tensor(data_emb, dtype=torch.float32),
            self.out_context_key: torch.tensor(context_emb, dtype=torch.float32),
            self.out_sample_id_key: sample_ids,
            self.out_raw_data_key: torch.tensor(raw_data, dtype=torch.float32),
        }
        if self.return_indices:
            sample["indices"] = idx
        return sample

    def _get_sequence(self, start_idx, end_idx):
        slices = slice(start_idx, end_idx)
        data_emb, context_emb, sample_ids, raw_data = dask.compute(
            self.data_embeddings[slices],
            self.context_embeddings[slices],
            self.sample_ids[slices],
            self.raw_data[slices],
        )

        sample = {
            self.out_data_key: torch.tensor(data_emb, dtype=torch.float32),
            self.out_context_key: torch.tensor(context_emb, dtype=torch.float32),
            self.out_sample_id_key: sample_ids,
            self.out_raw_data_key: torch.tensor(raw_data, dtype=torch.float32),
        }
        if self.return_indices:
            sample["indices"] = torch.arange(start_idx, end_idx)
        return sample
