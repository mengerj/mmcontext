import hashlib
import logging
import os
import tempfile

import anndata
import numpy as np
import scipy.sparse as sp
import torch
import transformers

from mmcontext.utils import download_file_from_share_link

# from mmcontext.pp.efficient_ds import OptimizedProcessor
logger = logging.getLogger(__name__)


class MMContextProcessor:
    """A Processor to create initial embeddings for text and omics data input.

    Uses a tokenizer for text data and a custom processor for omics data.
    The latter can be chosen from several apporaches
    """

    def __init__(
        self,
        processor_name="precomputed",
        text_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        **processor_kwargs,
    ):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(text_encoder_name)
        self.omics_processor = self._load_omics_processor(processor_name=processor_name, **processor_kwargs)

    def _load_omics_processor(self, processor_name, **processor_kwargs):
        if processor_name == "precomputed":
            # obsm_key and retrieval mode should be passed as kwargs
            return PrecomputedProcessor(**processor_kwargs)
        # elif processor_name == "optimized":
        #    return OptimizedProcessor(**processor_kwargs)
        else:
            raise ValueError(f"Invalid omics processor class: {processor_name}. Only 'precomputed' are supported.")


class PrecomputedProcessor:
    """
    Processor for retrieving precomputed embeddings.

    Takes remote or local paths from the provided dictionary and retrieves the embeddings. Use "numpy" mode to only
    load the embedding matrices. Use "adata" mode to load the entire AnnData object and extract the embeddings from it.

    Parameters
    ----------
    obsm_key: str
        The key in `adata.obsm` to use for retrieving metadata. Either to get
        from `adata.obsm` or from a NumPy file with a share_link associated
        with the key.
    retrieval_mode: str, optional
        The retrieval mode, either "adata" or "numpy". Defaults to "numpy".
    obsm_key: str, optional
        The key of the embedding within the 'embeddings' dict of `file_record`
        in "numpy" mode.

    Raises
    ------
    ValueError
        If `retrieval_mode` is "numpy" and `obsm_key` is not provided.
    """

    def __init__(self, obsm_key, retrieval_mode="numpy", **kwargs):
        """Initialize the PrecomputedProcessor."""
        self.retrieval_mode = retrieval_mode
        self.obsm_key = obsm_key
        self._data_cache = {}
        self._path_cache = {}  # Cache for resolved file paths
        # Initialize device in constructor
        logger.info(
            f"Initialized PrecomputedProcessor. "
            f"Retrieval mode: {retrieval_mode}. "
            f"Metadata from AnnData obsm_key: {obsm_key}. "
        )
        if retrieval_mode == "numpy" and obsm_key is None:
            raise ValueError("In numpy mode, 'obsm_key' must be provided.")

    def _convert_to_tensor(self, data):
        """
        Convert data to PyTorch tensor efficiently.

        Parameters
        ----------
        data: torch.Tensor, np.ndarray, or sp.spmatrix
            The data to be converted.

        Returns
        -------
        torch.Tensor
            The converted tensor.

        Raises
        ------
        ValueError
            If the data type is unsupported.
        """
        if isinstance(data, torch.Tensor):
            return data.float()
        if isinstance(data, sp.spmatrix):
            return torch.from_numpy(data.toarray()).float()
        if isinstance(data, np.ndarray):
            if data.dtype == object:
                # Handle array of sparse matrices
                dense_arrays = [mat[0].toarray() if isinstance(mat[0], sp.spmatrix) else mat[0] for mat in data]
                return torch.from_numpy(np.vstack(dense_arrays)).float()
            return torch.from_numpy(data).float()
        raise ValueError(f"Unsupported data type: {type(data)}")

    @staticmethod
    def _resolve_file_path(file_path, suffix=".npz"):
        """
        Resolve the file path.

        If the `file_path` is a share link (starts with 'https'), download
        the file locally and return the local file path.

        Parameters
        ----------
        file_path: str
            The original file path or share link.

        Returns
        -------
        str
            The resolved local file path.
        """
        if file_path.startswith("https"):
            # Generate a unique local filename based on the MD5 hash of
            # the share link.
            hash_object = hashlib.md5(file_path.encode())
            file_hash = hash_object.hexdigest()
            local_file = os.path.join(tempfile.gettempdir(), f"{file_hash}{suffix}")
            if not os.path.exists(local_file):
                logger.info(f"Downloading file from share link: {file_path} to {local_file}")
                success = download_file_from_share_link(file_path, local_file)
                if not success:
                    raise ValueError(f"Failed to download file from share link: {file_path}")
            return local_file
        return file_path

    def get_rep(self, data):
        """
        Retrieve the omics representation.

        Parameters
        ----------
        data: list of dict
            A list of dictionaries, where each dictionary contains:
                - "file_record": A dictionary with paths to the AnnData file
                  ("dataset_path") and precomputed embeddings ("embeddings").
                - "sample_id": The ID of the sample within the AnnData file.

        Returns
        -------
        torch.Tensor
            A PyTorch tensor of shape (batch_size, feature_dim) containing the
            retrieved embeddings.

        Raises
        ------
        ValueError
            If `data` is not a list or if an error occurs during retrieval.
        """
        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries")

        batch_size = len(data)
        features = None

        # Group data items by embedding path for efficient processing
        grouped_data = {}
        for data_item in data:
            file_record = data_item["file_record"]
            embedding_path = file_record["embeddings"][self.obsm_key]
            if embedding_path not in grouped_data:
                grouped_data[embedding_path] = []
            grouped_data[embedding_path].append(data_item)

        # Pre-allocate the features tensor once we know the dimensions
        for embedding_path, data_items in grouped_data.items():
            if self.retrieval_mode == "numpy":
                # Resolve the file path only once per unique embedding_path
                if embedding_path not in self._data_cache:
                    if embedding_path not in self._path_cache:
                        self._path_cache[embedding_path] = self._resolve_file_path(embedding_path, suffix=".npz")
                    resolved_path = self._path_cache[embedding_path]

                    # Load the data with mmap_mode='r' for memory efficiency
                    npzfile = np.load(resolved_path, allow_pickle=True, mmap_mode="r")
                    emb_matrix = npzfile["data"]
                    sample_ids = npzfile.get("sample_ids", None)

                    # Convert to tensor once and store in cache
                    if isinstance(emb_matrix, torch.Tensor):
                        tensor_matrix = emb_matrix.to(self.device)
                    else:
                        # Move conversion to CUDA if available
                        if isinstance(emb_matrix, sp.spmatrix):
                            tensor_matrix = torch.from_numpy(emb_matrix.toarray()).float()
                        else:
                            tensor_matrix = torch.from_numpy(emb_matrix).float()

                    self._data_cache[embedding_path] = {"matrix": tensor_matrix, "sample_ids": sample_ids}

                # Get cached data
                cached_data = self._data_cache[embedding_path]
                emb_matrix = cached_data["matrix"]
                sample_ids = cached_data["sample_ids"]

                # Initialize features tensor if not already done
                if features is None:
                    feature_dim = emb_matrix.shape[1]
                    features = torch.zeros((batch_size, feature_dim), dtype=torch.float32)  # , device=self.device)

                # Collect all sample IDs for this batch
                batch_sample_ids = [item["sample_id"] for item in data_items]

                # Use vectorized operations for indexing
                sorted_idx = np.argsort(sample_ids)
                sorted_sample_ids = sample_ids[sorted_idx]
                pos_in_sorted = np.searchsorted(sorted_sample_ids, batch_sample_ids)

                # Get indices for this batch
                for _i, (item_idx, data_item) in enumerate(zip(pos_in_sorted, data_items, strict=False)):
                    item_pos = list(data).index(data_item)
                    features[item_pos] = emb_matrix[sorted_idx[item_idx]]

        return features

    def clear_cache(self):
        """Clear the caches to free memory."""
        self._data_cache.clear()
        self._path_cache.clear()


'''
class PrecomputedProcessor:
    """Unified processor for retrieving precomputed embeddings, supporting AnnData and NumPy formats."""

    def __init__(self, obsm_key, retrieval_mode="numpy", **kwargs):
        """
        Initialize the PrecomputedProcessor.

        Parameters
        ----------
        obsm_key : str
            The key in adata.obsm to use for retrieving metadata. Either to get from adata.obsm or from a numpy file with a share_link accociated with the key.
        retrieval_mode : str, optional
            The retrieval mode, either "adata" or "numpy". Defaults to "adata".
        obsm_key : str, optional
            The key of the embedding within the 'embeddings' dict of file_record in numpy mode.
        """
        self.retrieval_mode = retrieval_mode
        self.obsm_key = obsm_key
        logger.info(
            f"Initialized PrecomputedProcessor. Retrieval mode: {retrieval_mode}. Metadata from AnnData obsm_key: {obsm_key}."
        )
        if retrieval_mode == "numpy" and obsm_key is None:
            raise ValueError("In numpy mode, 'obsm_key' must be provided.")
        self._data_cache = {}

    def _convert_to_tensor(self, data):
        """
        Convert data to torch tensor efficiently.

        Parameters
        ----------
        data : torch.Tensor, np.ndarray, or sp.spmatrix
            The data to be converted.

        Returns
        -------
        torch.Tensor
            The converted tensor.

        Raises
        ------
        ValueError
            If the data type is unsupported.
        """
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, sp.spmatrix):
            return torch.from_numpy(data.toarray())
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        raise ValueError(f"Unsupported data type: {type(data)}")

    @staticmethod
    def _resolve_file_path(file_path):
        """
        Resolve the file path.

        If the file_path is a share link (starts with 'https'), download the file locally and return the local file path.

        Parameters
        ----------
        file_path : str
            The original file path or share link.

        Returns
        -------
        str
            The resolved local file path.
        """
        if file_path.startswith("https"):
            # Generate a unique local filename based on the MD5 hash of the share link.
            hash_object = hashlib.md5(file_path.encode())
            file_hash = hash_object.hexdigest()
            local_file = os.path.join(tempfile.gettempdir(), f"{file_hash}.h5ad")
            if not os.path.exists(local_file):
                logger.info(f"Downloading file from share link: {file_path} to {local_file}")
                success = download_file_from_share_link(file_path, local_file)
                if not success:
                    raise ValueError(f"Failed to download file from share link: {file_path}")
            return local_file
        return file_path

    def get_rep(self, data):
        """Retrieve the omics representation."""
        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries")

        batch_size = len(data)
        features = None  # Initialize features outside the loops

        for i, data_item in enumerate(data):
            if self.retrieval_mode == "adata":
                # ... (adata retrieval logic remains unchanged)
                file_path = self._resolve_file_path(data_item["file_record"]["dataset_path"])
                sample_id = data_item["sample_id"]

                if file_path not in self._data_cache:
                    adata = (
                        anndata.read_h5ad(file_path) if file_path.endswith(".h5ad") else anndata.read_zarr(file_path)
                    )
                    self._data_cache[file_path] = adata
                adata = self._data_cache[file_path]
                if features is None:
                    feature_dim = adata.obsm[self.obsm_key].shape[1]
                    features = torch.zeros((batch_size, feature_dim), dtype=torch.float32)

                sample_idx = adata.obs.index == sample_id
                features[i] = self._convert_to_tensor(adata.obsm[self.obsm_key][sample_idx][0])

            elif self.retrieval_mode == "numpy":
                file_record = data_item["file_record"]
                sample_id = data_item["sample_id"]
                #adata_path = self._resolve_file_path(file_record["dataset_path"])
                embedding_path = self._resolve_file_path(file_record["embeddings"][self.obsm_key])

                if embedding_path not in self._data_cache:
                    npzfile = np.load(embedding_path, allow_pickle=True)
                    emb_matrix = npzfile["data"]
                    self._data_cache[embedding_path] = emb_matrix

                emb_matrix = self._data_cache[embedding_path]

                #if adata_path not in self._data_cache:
                #    adata = (
                #        anndata.read_h5ad(adata_path) if adata_path.endswith(".h5ad") else anndata.read_zarr(adata_path)
                #    )
                #    self._data_cache[adata_path] = adata

                #adata = self._data_cache[adata_path]

                if features is None:
                    feature_dim = emb_matrix.shape[1]
                    features = torch.zeros((batch_size, feature_dim), dtype=torch.float32)

                sample_ids = npzfile.get("sample_ids", adata.obs.index.values)
                sample_idx = np.isin(sample_ids, sample_id)
                sample_embedding = emb_matrix[sample_idx]
                features[i] = self._convert_to_tensor(sample_embedding[0])

        return features

    def clear_cache(self):
        """Clear the AnnData cache to free memory."""
        self._data_cache.clear()
'''
