# mmcontext/pp/dim_aligner.py

import logging
from abc import ABC, abstractmethod

import numpy as np
from anndata import AnnData
from sklearn.decomposition import PCA


class DimAligner(ABC):
    """
    Abstract base class for aligning dimensions of embeddings.

    This class defines the interface for dimension alignment, including methods for
    reducing and extending embeddings to achieve a consistent target dimension.

    Parameters
    ----------
    latent_dim
        Target dimension for alignment (default: 64).
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        latent_dim: int = 64,
        context_key: str = "c_emb_norm",
        data_key: str = "d_emb_norm",
    ):
        """
        Initializes the DimAligner with the target latent dimension.

        Parameters
        ----------
        latent_dim
            The target dimension to align embeddings to.
        context_key
            The key for context embeddings in adata.obsm. Defaults are normalized embeddings
        data_key
            The key for data embeddings in adata.obsm. Defaults are normalized embeddings
        """
        self.latent_dim = latent_dim
        self.context_key = context_key
        self.data_key = data_key
        self.logger = logger or logging.getLogger(__name__)

    def align(self, adata: AnnData):
        """Aligns the dimensions of data and context embeddings

        Aligns the dimensions of data and context embeddings in an AnnData object to the target dimension.

        Aligned embeddings are stored in 'd_emb_aligned' and 'c_emb_aligned' in adata.obsm.

        If an embedding's dimension is larger than the target dimension, it is reduced.
        If it is smaller, it is extended (padded).

        Parameters
        ----------
        adata
            The AnnData object containing embeddings in 'd_emb' and 'c_emb'.

        Raises
        ------
        ValueError
            If embeddings are missing from adata.obsm.
        """
        if self.data_key not in adata.obsm or self.context_key not in adata.obsm:
            raise ValueError(f"Embeddings {self.data_key} and {self.context_key} must be present in adata.obsm.")

        d_emb = adata.obsm[self.data_key]
        c_emb = adata.obsm[self.context_key]

        # Align data embeddings
        if d_emb.shape[1] > self.latent_dim:
            d_emb_aligned = self.reduce(d_emb)
        elif d_emb.shape[1] < self.latent_dim:
            d_emb_aligned = self.extend(d_emb)
        else:
            d_emb_aligned = d_emb  # Already at target dimension

        # Align context embeddings
        if c_emb.shape[1] > self.latent_dim:
            c_emb_aligned = self.reduce(c_emb)
        elif c_emb.shape[1] < self.latent_dim:
            c_emb_aligned = self.extend(c_emb)
        else:
            c_emb_aligned = c_emb  # Already at target dimension

        # Store aligned embeddings
        adata.obsm["d_emb_aligned"] = d_emb_aligned
        adata.obsm["c_emb_aligned"] = c_emb_aligned

    @abstractmethod
    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduces the dimensions of the embeddings to the target dimension.

        Parameters
        ----------
        embeddings
            The embeddings to reduce.

        Returns
        -------
        The dimensionally reduced embeddings.
        """
        pass

    def extend(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Extends the dimensions of the embeddings to the target dimension by zero-padding.

        Parameters
        ----------
        embeddings
            The embeddings to extend.

        Returns
        -------
        The dimensionally extended embeddings.
        """
        current_dim = embeddings.shape[1]
        if current_dim >= self.latent_dim:
            return embeddings

        padding = np.zeros((embeddings.shape[0], self.latent_dim - current_dim))
        extended_embeddings = np.hstack((embeddings, padding))
        return extended_embeddings


class PCAReducer(DimAligner):
    """
    Aligns dimensions of embeddings using PCA for dimensionality reduction.

    This class reduces the dimensions of embeddings to the target latent dimension using PCA.

    Parameters
    ----------
    latent_dim
        Target dimension for alignment.
    context_key
        Key for context embeddings in adata.obsm.
    data_key
        Key for data embeddings in adata.obsm.
    max_samples
        Maximum number of samples to use for fitting PCA.
    random_state
        Random seed for reproducibility.
    """

    def __init__(
        self, logger: logging.Logger | None = None, *args, max_samples: int = 10000, random_state: int = None, **kwargs
    ):
        """
        Initializes the PCAReducer with the target latent dimension and parameters.

        Parameters
        ----------
        max_samples
            Maximum number of samples to use for fitting PCA.
        random_state
            Random seed for reproducibility.
        logger
            An optional logger object to use for logging.
        *args
            Positional arguments for the base class.
        **kwargs
            Keyword arguments for the base class (e.g., latent_dim, context_key, data_key).
        """
        super().__init__(logger, *args, **kwargs)
        self.max_samples = max_samples
        self.random_state = random_state

    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduces the dimensions of the embeddings to the target dimension using PCA.

        Fits the PCA model on a subset of the data if necessary and transforms the embeddings.

        Parameters
        ----------
        embeddings
            The embeddings to reduce.

        Returns
        -------
        The dimensionally reduced embeddings.
        """
        n_samples = embeddings.shape[0]
        if n_samples > self.max_samples:
            # Randomly sample max_samples indices
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(n_samples, size=self.max_samples, replace=False)
            sample_embeddings = embeddings[indices]
        else:
            sample_embeddings = embeddings

        # Fit PCA model
        pca = PCA(n_components=self.latent_dim, random_state=self.random_state)
        pca.fit(sample_embeddings)

        # Transform the entire embeddings
        reduced_embeddings = pca.transform(embeddings)

        return reduced_embeddings
