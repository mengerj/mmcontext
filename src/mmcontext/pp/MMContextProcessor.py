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
            return AnnDataRetrievalProcessor(**processor_kwargs)
        else:
            raise ValueError(f"Invalid omics processor class: {processor_name}. Only 'precomputed' is supported.")


class AnnDataRetrievalProcessor:
    """
    Processor that retrieves the raw data without any processing.

    Attributes
    ----------
    obsm_key : str
        The key to access the desired representation in adata.obsm.
    _adata_cache : dict
        Cache for loaded AnnData objects to avoid reloading from disk.
    """

    def __init__(self, obsm_key, **kwargs):
        """
        Initialize the AnnDataRetrievalProcessor.

        Parameters
        ----------
        obsm_key : str
            The key in adata.obsm to use for initial embeddings.
        """
        self.obsm_key = obsm_key
        logger.info(
            f"Initialized AnnDataRetrievalProcessor. Will use embeddings from obsm_key: {obsm_key} as initial embeddings."
        )
        self._adata_cache = {}

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
        """
        Retrieve the omics representation of the data based on the provided file paths and sample IDs.

        Parameters
        ----------
        data : list of dict
            A list of dictionaries, each containing 'file_path' and 'sample_id' keys.
            The 'file_path' can be a local file (h5ad or zarr) or a share link (starts with https).

        Returns
        -------
        torch.Tensor
            The omics representation tensor with shape (batch_size, feature_dim).

        Raises
        ------
        ValueError
            If data is not a list or if an unsupported file format is encountered.
        """
        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries")

        batch_size = len(data)
        # Process the first file to determine the feature dimension.
        first_file = self._resolve_file_path(data[0]["file_path"])
        if first_file not in self._adata_cache:
            if first_file.endswith(".h5ad"):
                adata = anndata.read_h5ad(first_file)
            elif first_file.endswith(".zarr"):
                adata = anndata.read_zarr(first_file)
            else:
                raise ValueError(f"Unsupported file format for file: {first_file}")
            self._adata_cache[first_file] = adata

        first_adata = self._adata_cache[first_file]
        feature_dim = first_adata.obsm[self.obsm_key].shape[1]

        # Pre-allocate the output tensor.
        features = torch.zeros((batch_size, feature_dim), dtype=torch.float32)

        for i, sample_dict in enumerate(data):
            file_path = self._resolve_file_path(sample_dict["file_path"])
            sample_id = sample_dict["sample_id"]

            if file_path not in self._adata_cache:
                if file_path.endswith(".h5ad"):
                    adata = anndata.read_h5ad(file_path)
                elif file_path.endswith(".zarr"):
                    adata = anndata.read_zarr(file_path)
                else:
                    raise ValueError(f"Unsupported file format for file: {file_path}")
                self._adata_cache[file_path] = adata

            adata = self._adata_cache[file_path]
            # Retrieve the sample's representation using the provided sample_id.
            sample_idx = adata.obs.index == sample_id
            features[i] = self._convert_to_tensor(adata.obsm[self.obsm_key][sample_idx][0])
        return features

    def clear_cache(self):
        """Clear the AnnData cache to free memory."""
        self._adata_cache.clear()
