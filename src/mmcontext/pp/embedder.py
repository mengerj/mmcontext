# pp/embedder.py
import logging

import anndata
import numpy as np

from mmcontext.pp import ContextEmbedder, DataEmbedder


class Embedder:
    """
    A class that handles both data and context embeddings for AnnData objects.

    This class can take data and context embedders as input to create embeddings.
    Instead of providing emvedders, external embeddings can be provided to the create embeddings method.

    Parameters
    ----------
    data_embedder
        An instance of a data embedder class. Defaults to None.
    context_embedder
        An instance of a context embedder class. Defaults to None.
    """

    def __init__(self, data_embedder: DataEmbedder | None = None, context_embedder: ContextEmbedder | None = None):
        """Initializes the Embedder with optional data and context embedders."""
        self.data_embedder = data_embedder
        self.context_embedder = context_embedder
        self.logger = logging.getLogger(__name__)

    def create_embeddings(
        self,
        adata: anndata.AnnData,
        data_embeddings: np.ndarray | None = None,
        context_embeddings: np.ndarray | None = None,
    ):
        """
        Creates or stores embeddings and adds them to adata.obsm.

        Parameters
        ----------
        adata
            The AnnData object containing the dataset.
        data_embeddings
            External data embeddings to store.
            Shape should be (n_samples, data_embedding_dim).
        context_embeddings
            External context embeddings to store.
            Shape should be (n_samples, context_embedding_dim).

        Raises
        ------
        ValueError
            If embeddings are missing and no embedder is provided to create them.
        """
        # Store external data embeddings if provided
        if data_embeddings is not None:
            self.logger.info("Using external data embeddings provided.")
            self.store_embeddings(adata, data_embeddings, key="d_emb")
        elif self.data_embedder is not None:
            self.logger.info("Creating data embeddings...")
            data_embeddings = self.data_embedder.embed(adata)
            self.store_embeddings(adata, data_embeddings, key="d_emb")
        else:
            self.logger.info("Data embeddings are missing, and no data embedder is provided.")

        # Store external context embeddings if provided
        if context_embeddings is not None:
            self.logger.info("Using external context embeddings provided.")
            self.store_embeddings(adata, context_embeddings, key="c_emb")
        elif self.context_embedder is not None:
            self.logger.info("Creating context embeddings...")
            context_embeddings = self.context_embedder.embed(adata)
            self.store_embeddings(adata, context_embeddings, key="c_emb")
        else:
            self.logger.info("Context embeddings are missing, and no context embedder is provided.")
            
    def store_embeddings(self, adata: anndata.AnnData, embeddings: np.ndarray, key: str):
        """
        Stores embeddings in adata.obsm with the given key, after validating the shape.

        Parameters
        ----------
        adata
            The AnnData object into which to store the embeddings.
        embeddings
            Embeddings to store.
        key
            Key under which to store the embeddings in adata.obsm.

        Raises
        ------
        TypeError
            If the embeddings are not a numpy.ndarray.
        ValueError
            If the number of samples in embeddings does not match adata.n_obs.
        """
        # Check if embeddings are a NumPy array
        if not isinstance(embeddings, np.ndarray):
            self.logger.error(
                f"The provided {key} embeddings must be a numpy.ndarray, but got {type(embeddings).__name__}."
            )
            raise TypeError(
                f"The provided {key} embeddings must be a numpy.ndarray, but got {type(embeddings).__name__}."
            )

        if embeddings.shape[0] != adata.n_obs:
            self.logger.error(
                f"The number of samples in the provided {key} embeddings does not match the number of observations in adata."
            )
            raise ValueError(
                f"The number of samples in the provided {key} embeddings does not match the number of observations in adata."
            )
        adata.obsm[key] = embeddings
