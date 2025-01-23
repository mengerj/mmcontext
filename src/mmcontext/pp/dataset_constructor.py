import dask.array as da
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from anndata import AnnData
from torch.utils.data import Dataset


class DataSetConstructor:
    """
    Constructs datasets from embeddings stored in AnnData objects.

    Parameters
    ----------
    out_emb_keys : Dict[str, str]
        Dictionary mapping internal names to output names in the Dataset.
        Example: {'data_embeddings': 'data_emb', 'context_embeddings': 'context_emb'}
    out_sample_id_key : str
        Key to be used for sample IDs in the Dataset.
    out_raw_data_key : str
        Key to be used for raw data in the Dataset.
    use_raw : bool
        Whether to include raw data (adata.X) in the Dataset.
    chunk_size : int
        Chunk size to use for Dask arrays.
    batch_size : int
        Batch size for the DataLoader.
    use_dask : bool
    """

    def __init__(
        self,
        out_emb_keys: dict,
        out_sample_id_key: str = "sample_id",
        out_raw_data_key: str = "raw_data",
        use_raw: bool = True,
        chunk_size: int | None = None,
        batch_size: int | None = None,
        use_dask: bool = False,
    ):
        self.out_emb_keys = out_emb_keys  # Dict[str, str]
        self.out_sample_id_key = out_sample_id_key
        self.out_raw_data_key = out_raw_data_key
        self.use_raw = use_raw
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.use_dask = use_dask

        # Initialize empty dicts for embeddings
        self.embeddings = {key: None for key in self.out_emb_keys.keys()}
        self.sample_ids_list = []  # For checking duplicate sample ids
        self.sample_ids = None
        self.raw_data = None
        self.total_samples = 0

        if batch_size is None:
            raise ValueError(
                "Batch size must be specified. Not used here, but important to ensure it is valid for the data size."
            )
        if chunk_size is None:
            raise ValueError(
                "Chunk size must be specified. Ideally it should be a multiple of the batch size and the sequence length."
            )

    def add_anndata(self, adata: AnnData, emb_keys: dict, sample_id_key: str):
        """
        Adds data from an AnnData object into the dataset.

        Parameters
        ----------
        adata : AnnData
            The AnnData object containing embeddings.
        emb_keys : Dict[str, str]
            Dictionary mapping internal names (same as in out_emb_keys) to keys in adata.obsm where the embeddings are stored.
        sample_id_key : str
            Key in adata.obs where sample IDs are stored.
        """
        # Check that the sample_id_key exists in adata.obs
        if sample_id_key not in adata.obs:
            raise ValueError(f"Sample IDs '{sample_id_key}' not found in adata.obs.")

        # Verify sample IDs are integers
        try:
            pd.to_numeric(adata.obs[sample_id_key], downcast="integer")
        except ValueError as err:
            raise ValueError(
                f"Sample IDs must be integers. Non-integer values found in '{sample_id_key}' column."
            ) from err

        sample_ids_list = adata.obs[sample_id_key].tolist()
        if len(self.sample_ids_list) > 0:
            # Verify that there are no duplicate sample IDs
            duplicate_ids = set(self.sample_ids_list).intersection(set(sample_ids_list))
            if duplicate_ids:
                raise ValueError(f"Duplicate sample IDs found: {duplicate_ids}")
        # Collect all sample ids to check for duplicates
        self.sample_ids_list.extend(sample_ids_list)

        emb_dims = []
        # Retrieve embeddings and store them in self.embeddings
        for key in self.out_emb_keys.keys():
            in_key = emb_keys[key]
            if in_key not in adata.obsm:
                raise ValueError(f"Embedding '{in_key}' not found in adata.obsm.")
            emb_array = adata.obsm[in_key]
            emb_dims.append(emb_array.shape[1])
            if len(set(emb_dims)) > 1:
                raise ValueError("Input embeddings have different dimensions. (within adata)")

            # Convert to Dask array or use directly based on the flag
            if self.use_dask:
                emb_processed = da.from_array(emb_array, chunks=(adata.shape[0], self.chunk_size))
            else:
                emb_processed = emb_array  # Use NumPy array directly

            if self.embeddings[key] is not None:
                # Check dimensions of new embeddings
                if emb_processed.shape[1] != self.embeddings[key].shape[1]:
                    raise ValueError(
                        f"Inconsistent embedding dimensions between adata objects for '{key}': expected {self.embeddings[key].shape[1]}, but got {emb_processed.shape[1]}."
                    )
                # Concatenate
                if self.use_dask:
                    self.embeddings[key] = da.concatenate([self.embeddings[key], emb_processed], axis=0)
                else:
                    self.embeddings[key] = np.concatenate([self.embeddings[key], emb_processed], axis=0)
            else:
                self.embeddings[key] = emb_processed

        # Retrieve sample IDs
        sample_ids = (
            da.from_array(adata.obs[sample_id_key].values, chunks=self.chunk_size)
            if self.use_dask
            else adata.obs[sample_id_key].values
        )
        if self.sample_ids is not None:
            if self.use_dask:
                self.sample_ids = da.concatenate([self.sample_ids, sample_ids], axis=0)
            else:
                self.sample_ids = np.concatenate([self.sample_ids, sample_ids], axis=0)
        else:
            self.sample_ids = sample_ids

        # Optionally retrieve raw data
        if self.use_raw:
            X = adata.X
            if isinstance(X, sp.spmatrix):
                X = X.toarray()
            raw_data = da.from_array(X, chunks=(adata.shape[0], self.chunk_size)) if self.use_dask else X
            if self.raw_data is not None:
                if self.use_dask:
                    self.raw_data = da.concatenate([self.raw_data, raw_data], axis=0)
                else:
                    self.raw_data = np.concatenate([self.raw_data, raw_data], axis=0)
            else:
                self.raw_data = raw_data

        # Update total_samples
        self.total_samples += adata.shape[0]

        # Check that all embeddings have the same number of samples
        for key in self.embeddings:
            if self.embeddings[key].shape[0] != self.sample_ids.shape[0]:
                raise ValueError(f"Number of samples mismatch between embeddings and sample IDs for '{key}'.")

    def construct_dataset(self, seq_length: int = None, return_indices=False) -> Dataset:
        """
        Constructs and returns a PyTorch Dataset combining all added embeddings.

        Parameters
        ----------
        seq_length : int, optional
            Length of sequences.
        return_indices : bool, optional
            If True, the dataset will return the index of the sample.

        Returns
        -------
        Dataset
            A PyTorch Dataset containing embeddings and sample IDs.
        """
        # Remove self.sample_ids_list to save memory
        self.sample_ids_list = None
        if not self.embeddings:
            raise ValueError("No data has been added to the dataset.")
        if seq_length is not None and self.batch_size * seq_length > self.total_samples:
            raise ValueError("Batch size and sequence length are too large for the dataset.")

        dataset = EmbeddingDataset(
            embeddings=self.embeddings,
            sample_ids=self.sample_ids,
            raw_data=self.raw_data,
            return_indices=return_indices,
            seq_length=seq_length,
            out_emb_keys=self.out_emb_keys,
            out_sample_id_key=self.out_sample_id_key,
            out_raw_data_key=self.out_raw_data_key,
            use_dask=self.use_dask,
        )
        return dataset


class EmbeddingDataset(Dataset):
    """
    Custom PyTorch Dataset class to load embeddings from Dask or in-memory arrays.

    Parameters
    ----------
    embeddings : Dict[str, Union[da.Array, np.ndarray]]
        Dictionary of embeddings. Can be Dask arrays or NumPy arrays.
    sample_ids : Union[da.Array, np.ndarray]
        Dask array or NumPy array containing sample IDs.
    raw_data : Union[da.Array, np.ndarray], optional
        Dask array or NumPy array containing raw data, if use_raw is True.
    return_indices : bool, optional
        If True, the dataset will return the index of the sample.
    seq_length : int, optional
        Length of sequences. If None, the dataset will return individual samples.
    out_emb_keys : Dict[str, str]
        Dictionary mapping internal names to output names in the Dataset.
    out_sample_id_key : str
        Key to be used for sample IDs in the Dataset.
    out_raw_data_key : str
        Key to be used for raw data in the Dataset.
    use_dask : bool, optional
        Whether to use Dask for processing. Defaults to False.
    """

    def __init__(
        self,
        embeddings: dict,
        sample_ids,
        raw_data=None,
        return_indices: bool = False,
        seq_length: int = None,
        out_emb_keys: dict = None,
        out_sample_id_key: str = "sample_id",
        out_raw_data_key: str = "raw_data",
        use_dask: bool = False,
    ):
        self.embeddings = embeddings
        self.sample_ids = sample_ids
        self.raw_data = raw_data
        self.return_indices = return_indices
        self.seq_length = seq_length
        self.out_emb_keys = out_emb_keys
        self.out_sample_id_key = out_sample_id_key
        self.out_raw_data_key = out_raw_data_key
        self.use_dask = use_dask

        # Compute total number of samples
        self.total_samples = self.sample_ids.shape[0]

    def __len__(self):
        if self.seq_length is not None:
            return self.total_samples // self.seq_length
        return self.total_samples

    def __getitem__(self, idx):
        if self.seq_length is not None:
            start_idx = idx * self.seq_length
            end_idx = start_idx + self.seq_length
            return self._get_sequence(start_idx, end_idx)
        return self._get_sample(idx)

    def _get_sample(self, idx):
        arrays_to_compute = []
        for key in self.embeddings:
            arrays_to_compute.append(self.embeddings[key][idx])
        arrays_to_compute.append(self.sample_ids[idx])
        if self.raw_data is not None:
            arrays_to_compute.append(self.raw_data[idx])

        if self.use_dask:
            # Compute arrays with Dask
            computed_arrays = da.compute(*arrays_to_compute)
        else:
            # Directly use NumPy arrays
            computed_arrays = arrays_to_compute

        sample = {}
        i = 0
        for key in self.embeddings:
            out_key = self.out_emb_keys[key]
            sample[out_key] = torch.tensor(computed_arrays[i], dtype=torch.float32)
            i += 1

        sample[self.out_sample_id_key] = computed_arrays[i]
        i += 1

        if self.raw_data is not None:
            sample[self.out_raw_data_key] = torch.tensor(computed_arrays[i], dtype=torch.float32)
            i += 1

        if self.return_indices:
            sample["indices"] = idx

        return sample

    def _get_sequence(self, start_idx, end_idx):
        slices = slice(start_idx, end_idx)
        arrays_to_compute = []
        for key in self.embeddings:
            arrays_to_compute.append(self.embeddings[key][slices])
        arrays_to_compute.append(self.sample_ids[slices])
        if self.raw_data is not None:
            arrays_to_compute.append(self.raw_data[slices])

        if self.use_dask:
            # Compute arrays with Dask
            computed_arrays = da.compute(*arrays_to_compute)
        else:
            # Directly use NumPy arrays
            computed_arrays = arrays_to_compute

        sample = {}
        i = 0
        for key in self.embeddings:
            out_key = self.out_emb_keys[key]
            sample[out_key] = torch.tensor(computed_arrays[i], dtype=torch.float32)
            i += 1

        sample[self.out_sample_id_key] = computed_arrays[i]
        i += 1

        if self.raw_data is not None:
            sample[self.out_raw_data_key] = torch.tensor(computed_arrays[i], dtype=torch.float32)
            i += 1

        if self.return_indices:
            sample["indices"] = torch.arange(start_idx, end_idx)

        return sample
