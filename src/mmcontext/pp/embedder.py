# pp/embedder.py
import logging

import numpy as np


class Embedder:
    """
    A class that handles both data and context embeddings for AnnData objects.

    Args:
        data_embedder (DataEmbedder, optional): An instance of a data embedder class. Defaults to None.
        context_embedder (ContextEmbedder, optional): An instance of a context embedder class. Defaults to None.
    """

    def __init__(self, data_embedder=None, context_embedder=None):
        """
        Initializes the Embedder with optional data and context embedders.

        Args:
            data_embedder (DataEmbedder, optional): An instance of a data embedder class. Defaults to None.
            context_embedder (ContextEmbedder, optional): An instance of a context embedder class. Defaults to None.
        """
        self.data_embedder = data_embedder
        self.context_embedder = context_embedder
        self.logger = logging.getLogger(__name__)

    def check_embeddings(self, adata):
        """
        Checks if data and context embeddings are present in the AnnData object.

        Args:
            adata (anndata.AnnData): The AnnData object to check for embeddings.

        Returns
        -------
            tuple: A tuple of two boolean values indicating if data embeddings ('d_emb')
                   and context embeddings ('c_emb') are present in `adata.obsm`.
        """
        has_data_emb = "d_emb" in adata.obsm
        has_context_emb = "c_emb" in adata.obsm
        return has_data_emb, has_context_emb

    def create_embeddings(self, adata, data_embeddings=None, context_embeddings=None):
        """
        Creates or stores embeddings and adds them to adata.obsm.

        Parameters
        ----------
            adata (AnnData): The AnnData object containing the dataset.
            data_embeddings (np.ndarray, optional): External data embeddings to store.
                Shape should be (n_samples, data_embedding_dim).
            context_embeddings (np.ndarray, optional): External context embeddings to store.
                Shape should be (n_samples, context_embedding_dim).

        Raises
        ------
            ValueError: If embeddings are missing and no embedder is provided to create them.
        """
        # Store external data embeddings if provided
        if data_embeddings is not None:
            self.logger.info("Using external data embeddings provided.")
            self.store_embeddings(adata, data_embeddings, key="d_emb")
        else:
            has_data_emb, _ = self.check_embeddings(adata)
            if not has_data_emb:
                if self.data_embedder is not None:
                    self.logger.info("Creating data embeddings...")
                    data_embeddings = self.data_embedder.embed(adata)
                    self.store_embeddings(adata, data_embeddings, key="d_emb")
                else:
                    self.logger.error("Data embeddings are missing, and no data embedder is provided.")
                    raise ValueError("Data embeddings are missing, and no data embedder is provided.")

        # Store external context embeddings if provided
        if context_embeddings is not None:
            self.logger.info("Using external context embeddings provided.")
            self.store_embeddings(adata, context_embeddings, key="c_emb")
        else:
            _, has_context_emb = self.check_embeddings(adata)
            if not has_context_emb:
                if self.context_embedder is not None:
                    self.logger.info("Creating context embeddings...")
                    context_embeddings = self.context_embedder.embed(adata)
                    self.store_embeddings(adata, context_embeddings, key="c_emb")
                else:
                    self.logger.error("Context embeddings are missing, and no context embedder is provided.")
                    raise ValueError("Context embeddings are missing, and no context embedder is provided.")

    def store_embeddings(self, adata, embeddings, key):
        """
        Stores embeddings in adata.obsm with the given key, after validating the shape.

        Parameters
        ----------
            adata (AnnData): The AnnData object.
            embeddings (np.ndarray): Embeddings to store.
            key (str): Key under which to store the embeddings in adata.obsm.

        Raises
        ------
            TypeError: If the embeddings are not a numpy.ndarray.
            ValueError: If the number of samples in embeddings does not match adata.n_obs.
        """
        # Check if embeddings are a NumPy array
        if not isinstance(embeddings, np.ndarray):
            logging.error(
                f"The provided {key} embeddings must be a numpy.ndarray, but got {type(embeddings).__name__}."
            )
            raise TypeError(
                f"The provided {key} embeddings must be a numpy.ndarray, but got {type(embeddings).__name__}."
            )

        if embeddings.shape[0] != adata.n_obs:
            logging.error(
                f"The number of samples in the provided {key} embeddings does not match the number of observations in adata."
            )
            raise ValueError(
                f"The number of samples in the provided {key} embeddings does not match the number of observations in adata."
            )
        adata.obsm[key] = embeddings
