# pp/data_embedder.py

from abc import ABC, abstractmethod

import numpy as np
from sklearn.decomposition import PCA


class DataEmbedder(ABC):
    """Abstract base class for generating data embeddings for AnnData objects."""

    @abstractmethod
    def embed(self, adata):
        """
        Abstract method for generating embeddings for the given AnnData object.

        Args:
            adata (anndata.AnnData): The AnnData object containing the data.

        Returns
        -------
            np.ndarray: The embeddings generated from the data.
        """
        pass


class PCADataEmbedder(DataEmbedder):
    """
    Data embedder that uses Principal Component Analysis (PCA) to generate embeddings.

    Args:
        n_components (int): Number of principal components to retain in the embeddings.
    """

    def __init__(self, n_components=128):
        """
        Initializes the PCADataEmbedder with the specified number of components.

        Args:
            n_components (int, optional): The number of PCA components to generate. Defaults to 128.
        """
        self.n_components = n_components
        self.pca = None

    def embed(self, adata):
        """
        Generates PCA embeddings for the given AnnData object.

        Args:
            adata (anndata.AnnData): The AnnData object containing the data to be transformed.

        Returns
        -------
            np.ndarray: PCA-transformed embeddings.
        """
        # Assuming adata.X contains the raw data matrix
        self.pca = PCA(n_components=self.n_components)
        embeddings = self.pca.fit_transform(adata.X)
        return embeddings


class PlaceholderDataEmbedder(DataEmbedder):
    """Generates random embeddings as a placeholder for real data embeddings."""

    def embed(self, adata):
        """
        Generates random embeddings for the given AnnData object.

        Args:
            adata (anndata.AnnData): The AnnData object to generate embeddings for.

        Returns
        -------
            np.ndarray: Randomly generated embeddings.
        """
        n_samples = adata.n_obs
        embedding_dim = 64  # Example embedding dimension
        embeddings = np.random.rand(n_samples, embedding_dim)
        return embeddings
