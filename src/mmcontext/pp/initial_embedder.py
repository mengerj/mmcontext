import logging
from typing import Optional

import anndata
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# Try importing optional packages for advanced methods
# (They may or may not be installed in your environment)
try:
    import scvi

    SCVI_AVAILABLE = True
except ImportError:
    SCVI_AVAILABLE = False

try:
    import scgpt  # Replace with actual scGPT library name

    SCGPT_AVAILABLE = True
except ImportError:
    SCGPT_AVAILABLE = False


class BaseAnnDataEmbedder:
    """
    Abstract base class for an embedding method that works on AnnData objects.

    All subclasses must implement `fit(adata)` and `embed(adata, obsm_key, ...)`.
    """

    def __init__(self, embedding_dim: int = 50):
        """
        Base constructor for embedder.

        Parameters
        ----------
        embedding_dim : int, optional
            Dimensionality of the output embedding space.
        """
        self.embedding_dim = embedding_dim

    def fit(self, adata: anndata.AnnData) -> None:
        """
        Train or fit the embedding model on `adata`. Some methods may not require training.

        Parameters
        ----------
        adata : anndata.AnnData
            The single-cell data to be used for training/fitting the embedding.
        """
        raise NotImplementedError

    def embed(self, adata: anndata.AnnData, obsm_key: str = "X_pp") -> None:
        """
        Transform the data into the learned embedding space and store in `adata.obsm[obsm_key]`.

        Parameters
        ----------
        adata : anndata.AnnData
            The single-cell data to transform into the embedding space.
        obsm_key : str, optional
            The key under which the embedding will be stored in `adata.obsm`.
        """
        raise NotImplementedError


class PCAEmbedder(BaseAnnDataEmbedder):
    """
    PCA-based embedding for single-cell data stored in AnnData.

    Uses scikit-learn's PCA to reduce the dimensionality of `adata.X`.
    """

    def __init__(self, embedding_dim: int = 50):
        """
        Initialize the PCA embedder.

        Parameters
        ----------
        embedding_dim : int, optional
            Number of principal components to retain.
        """
        super().__init__(embedding_dim)
        self._pca_model = None

    def fit(self, adata: anndata.AnnData) -> None:
        """
        Fit a PCA model to the AnnData object's .X matrix.

        Parameters
        ----------
        adata : anndata.AnnData
            The single-cell data to be used for PCA.
        """
        logger.info("Fitting PCA with %d components.", self.embedding_dim)
        from sklearn.decomposition import PCA

        # Convert adata.X to dense if needed
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()

        pca_model = PCA(n_components=self.embedding_dim)
        pca_model.fit(X)
        self._pca_model = pca_model

    def embed(self, adata: anndata.AnnData, obsm_key: str = "X_pp") -> None:
        """
        Transform the data via PCA and store it in `adata.obsm[obsm_key]`.

        Parameters
        ----------
        adata : anndata.AnnData
            The single-cell data to transform.
        obsm_key : str, optional
            The key under which the PCA embedding will be stored in `adata.obsm`.
        """
        if self._pca_model is None:
            raise RuntimeError("PCA model is not fit yet. Call `fit(adata)` first.")

        X = adata.X
        if sp.issparse(X):
            X = X.toarray()

        logger.info("Transforming data with PCA into %d components.", self.embedding_dim)
        adata.obsm[obsm_key] = self._pca_model.transform(X)


class SCVIEmbedder(BaseAnnDataEmbedder):
    """SCVI Encoder."""

    def __init__(self, embedding_dim: int = 50):
        super().__init__(embedding_dim)
        if not SCVI_AVAILABLE:
            raise ImportError("scvi-tools is not installed. Please install scvi-tools to use SCVIEmbedder.")
        self.model = None

    def fit(self, adata: anndata.AnnData, batch_key, layer_key="counts") -> None:
        """Set up scVI model and train on the data. This is a placeholder skeleton."""
        import scvi

        logger.info("Setting up an scVI model with embedding_dim=%d", self.embedding_dim)
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)
        self.model = scvi.model.SCVI(adata, n_latent=self.embedding_dim)

        logger.info("Training scVI model.")
        self.model.train()

    def embed(self, adata: anndata.AnnData, obsm_key: str = "X_pp") -> None:
        """Use the trained scVI model to compute latent embeddings for each cell."""
        if self.model is None:
            raise RuntimeError("scVI model not trained. Call `fit(adata)` first.")

        logger.info("Computing scVI latent representation.")
        adata.obsm[obsm_key] = self.model.get_latent_representation(adata)


class SCGPTEmbedder(BaseAnnDataEmbedder):
    """Example skeleton for an scGPT-based embedder."""

    def __init__(self, embedding_dim: int = 50):
        super().__init__(embedding_dim)
        if not SCGPT_AVAILABLE:
            raise ImportError("scGPT is not installed. Please install scGPT to use SCGPTEmbedder.")
        self.model = None

    def fit(self, adata: anndata.AnnData) -> None:
        """Placeholder code for scGPT training / fine-tuning."""
        logger.info("Initialize and train scGPT model here (pseudo-code).")
        # e.g., self.model = some_scgpt_lib.Model(...)
        pass

    def embed(self, adata: anndata.AnnData, obsm_key: str = "X_pp") -> None:
        """Placeholder for computing scGPT embeddings."""
        if self.model is None:
            raise RuntimeError("scGPT model not trained. Call `fit(adata)` first.")

        # Example:
        logger.info("Compute scGPT embeddings and store in .obsm.")
        # adata.obsm[obsm_key] = self.model.get_embeddings(adata)
        pass


class InitialEmbedder:
    """Main interface for creating embeddings of single-cell data.

    The user can choose from multiple methods (e.g., 'pca', 'scvi', 'scGPT', etc.).
    If the precomputed_key is provided, the embeddings will be copied from adata.obsm[precomputed_key].

    Examples
    --------
    >>> manager = InitialEmbedder(method="pca", embedding_dim=20)
    >>> manager.fit(adata)  # Fit PCA
    >>> manager.embed(adata, obsm_key="X_pp")  # Store embeddings in adata.obsm["X_pp"]
    """

    def __init__(
        self,
        method: str = "pca",
        embedding_dim: int = 50,
        method_kwargs: dict | None = None,
        precomputed_key: str | None = None,
    ):
        """
        Initialize the manager and select the embedding method.

        Parameters
        ----------
        method : str, optional
            The embedding method to use. One of ["pca", "scvi", "scGPT"].
        embedding_dim : int, optional
            Dimensionality of the output embedding space.
        method_kwargs : dict, optional
            Additional keyword arguments to pass to the embedding method.
        precomputed_key : str, optional
            If provided, the embeddings will be copied from adata.obsm[precomputed]
        """
        self.method = method
        self.embedding_dim = embedding_dim
        self.method_kwargs = method_kwargs or {}
        self.precomputed_key = precomputed_key
        # Dispatch to the correct embedder class
        if method == "pca":
            self.embedder = PCAEmbedder(embedding_dim=embedding_dim)
        elif method == "scvi":
            self.embedder = SCVIEmbedder(embedding_dim=embedding_dim)
        elif method == "scGPT":
            self.embedder = SCGPTEmbedder(embedding_dim=embedding_dim)
        else:
            raise ValueError(f"Unknown embedding method: {method}")

    def fit(self, adata: anndata.AnnData) -> None:
        """
        Fit/train the embedding model on the provided AnnData object.

        Parameters
        ----------
        adata : anndata.AnnData
            Single-cell dataset to be used for fitting.
        """
        logger.info("Fitting method '%s' with embedding_dim=%d", self.method, self.embedding_dim)
        if not self.precomputed_key:
            self.embedder.fit(adata, **self.method_kwargs)
        else:
            logger.info("No fitting required for precomputed embeddings.")

    def embed(self, adata: anndata.AnnData, obsm_key: str = "X_pp") -> None:
        """
        Transform the data into the learned embedding space and store in `adata.obsm[obsm_key]`.

        Parameters
        ----------
        adata : anndata.AnnData
            Single-cell dataset to be transformed.
        obsm_key : str, optional
            The key under which the embedding will be stored in `adata.obsm`.
        """
        if not self.precomputed_key:
            logger.info("Embedding data using method '%s'. Output: adata.obsm['%s']", self.method, obsm_key)
            self.embedder.embed(adata, obsm_key=obsm_key)
        else:
            logger.info(
                "Using precomputed embeddings. Copying from adata.obsm['%s'] to adata.obsm['%s']",
                self.precomputed_key,
                obsm_key,
            )
            adata.obsm[obsm_key] = adata.obsm[self.precomputed_key]
