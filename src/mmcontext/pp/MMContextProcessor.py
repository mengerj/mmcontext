import logging

# from mmcontext.models.geneformer_model import GeneformerModel
import os

import anndata
import numpy as np
import torch
import transformers
from scipy import sparse as sp
from sklearn.decomposition import PCA
from torch import Tensor

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
    """Processor that retrieves the raw data without any processing."""

    def __init__(self, obsm_key, **kwargs):
        self.obsm_key = obsm_key
        logger.info(
            f"Initialized AnnDataRetrievalProcessor. Will get use embeddings from obsm_key: {obsm_key} as initial embeddings."
        )
        # Add a cache for loaded AnnData files
        self._adata_cache = {}

    def _convert_to_tensor(self, data):
        """Convert data to torch tensor efficiently."""
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, sp.spmatrix):
            return torch.from_numpy(data.toarray())
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        raise ValueError(f"Unsupported data type: {type(data)}")

    def get_rep(self, data):
        """Use the given file_path and sample ID as well as the obsm_key to retrieve the correct representation of the data.

        Parameters
        ----------
        data : list
            A list of dictionaries, each containing 'file_path' and 'sample_id' keys.
        obsm_key : str
            The key to access the desired representation in adata.obsm

        Returns
        -------
        features : torch.Tensor
            The omics representation tensor with shape
            (batch_size, feature_dim).
        """
        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries")

        batch_size = len(data)
        # self.clear_cache()
        # Load first file to get feature dimension
        first_file = data[0]["file_path"]
        if first_file not in self._adata_cache:
            if first_file.endswith(".h5ad"):
                adata = anndata.read_h5ad(first_file)
            elif first_file.endswith(".zarr"):
                adata = anndata.read_zarr(first_file)
            # Convert to tensor once during caching
            # adata.obsm[self.obsm_key] = self._convert_to_tensor(adata.obsm[self.obsm_key])
            self._adata_cache[first_file] = adata

        first_adata = self._adata_cache[first_file]
        feature_dim = first_adata.obsm[self.obsm_key].shape[1]

        # Pre-allocate the output tensor
        features = torch.zeros((batch_size, feature_dim), dtype=torch.float32)

        for i, sample_dict in enumerate(data):
            file_path = sample_dict["file_path"]
            sample_id = sample_dict["sample_id"]

            # Use cached AnnData if available, otherwise load and convert
            if file_path not in self._adata_cache:
                if file_path.endswith(".h5ad"):
                    adata = anndata.read_h5ad(file_path)
                if file_path.endswith(".zarr"):
                    adata = anndata.read_zarr(file_path)
                # adata.obsm[self.obsm_key] = self._convert_to_tensor(adata.obsm[self.obsm_key])
                self._adata_cache[file_path] = adata

            adata = self._adata_cache[file_path]

            # Get the specific sample's representation
            sample_idx = adata.obs.index == sample_id
            features[i] = self._convert_to_tensor(adata.obsm[self.obsm_key][sample_idx][0])
        return features

    def clear_cache(self):
        """Clear the AnnData cache to free memory"""
        self._adata_cache.clear()


class PCAOmicsProcessor:
    """Processor to encode omics data"""

    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca = None
        self.mean_ = None

    def fit(self, array: np.ndarray | Tensor | sp.spmatrix):
        """Fits PCA on the input array."""
        count_matrix = array
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(count_matrix)
        self.mean_ = np.mean(count_matrix, axis=0)

    def encode(self, count_vector):
        """Encodes a single count vector using the fitted PCA."""
        if self.pca is None or self.mean_ is None:
            raise ValueError("PCA encoder must be fitted before encoding.")
        standardized_vector = count_vector - self.mean_
        return self.pca.transform(standardized_vector.reshape(1, -1))[0]

    def save(self, filepath):
        """Saves the PCA components and mean."""
        np.savez(filepath, components=self.pca.components_, mean=self.mean_)

    def load(self, filepath):
        """Loads the PCA components and mean."""
        data = np.load(filepath)
        self.pca = PCA(n_components=self.n_components)
        self.pca.components_ = data["components"]
        self.mean_ = data["mean"]
