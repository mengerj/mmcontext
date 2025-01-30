# pp/data_embedder.py

import logging
from abc import ABC, abstractmethod

import anndata
import numpy as np
import scvi
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class DataEmbedder(ABC):
    """Abstract base class for generating data embeddings for AnnData objects."""

    @abstractmethod
    def embed(self, adata: anndata.AnnData) -> np.ndarray:
        """
        Abstract method for generating embeddings for the given AnnData object.

        Parameters
        ----------
        adata
            The AnnData object containing the data.

        Returns
        -------
        The embeddings generated from the data.
        """
        pass


class AnnDataStoredEmbedder(DataEmbedder):
    """
    Simple embedder that fetches existing embeddings from an AnnData object.

    For instance, if you’ve run scVI or some other model and stored the
    latent embeddings in adata.obsm["scvi"], this class can retrieve it.
    """

    def __init__(self, obsm_key: str = "scvi"):
        # super().__init__()
        self.obsm_key = obsm_key

    def embed(self, adata: anndata.AnnData) -> np.ndarray:
        """
        Returns the existing embeddings from adata.obsm[self.obsm_key].

        Parameters
        ----------
        adata : anndata.AnnData
            The AnnData object containing stored embeddings in .obsm.

        Returns
        -------
        np.ndarray
            The embeddings stored in adata.obsm[self.obsm_key].
        """
        if self.obsm_key not in adata.obsm:
            raise ValueError(f"No embeddings found in adata.obsm['{self.obsm_key}'].")
        embeddings = adata.obsm[self.obsm_key]
        logger.info("Fetched existing embeddings from adata.obsm['%s'] of shape %s.", self.obsm_key, embeddings.shape)
        return embeddings


class ScviEmbedder:
    """
    A data embedder that uses a pretrained scVI model for RNA-seq data.

    This class loads a pretrained scVI model (downloaded from a remote source,
    e.g., cellxgene-census) and applies it to an input AnnData object to
    produce latent embeddings. By default, the embeddings will be added to
    `adata.obsm["scvi"]`.

    Parameters
    ----------
    model_path : str
        File path (or directory path) to the pretrained scVI model.
        For example: "2024-02-12-scvi-homo-sapiens/scvi.model".
    add_to_adata : bool, optional
        If True, the latent embeddings will be stored in
        `adata.obsm["scvi"]`. Defaults to True.

    References
    ----------
    The pretrained scVI model is available via cellxgene-census:
    https://cellxgene.cziscience.com/census
    """

    def __init__(self, model_path: str):
        """
        Initializes the ScviEmbedder with a path to the pretrained scVI model.

        Parameters
        ----------
        model_path : str
            File path (or directory) where the scVI model.pt is located.
        """
        self.model_path = model_path

    def embed(self, adata: anndata.AnnData) -> np.ndarray:
        """
        Creates latent embeddings from the input AnnData using the scVI model.

        This method expects the `adata.var` index to contain the gene identifiers
        (usually Ensembl IDs) required by the pretrained model. Any genes missing
        in the query dataset may reduce the fraction of matched genes.

        Parameters
        ----------
        adata : anndata.AnnData
            The input AnnData object to embed, containing the gene expression
            matrix in `adata.X`. `adata.var_names` should correspond to Ensembl
            gene IDs if you are using the default scVI model from cellxgene-census.

        Returns
        -------
        np.ndarray
            The latent representation of shape `(n_cells, latent_dim)`.

        Raises
        ------
        ValueError
            If there is a mismatch in the expected model path or the data cannot
            be embedded for another reason.

        References
        ----------
        The data is drawn from user-provided files, and the pretrained model
        is obtained from:
        s3://cellxgene-contrib-public/models/scvi/2024-02-12/homo_sapiens/model.pt
        (as of the example in the scvi tutorial).
        """
        logger.info("Preparing query AnnData for the pretrained scVI model.")
        # scvi will attempt to match query data with the reference model's genes
        # This step modifies the AnnData in-place to ensure consistency
        scvi.model.SCVI.prepare_query_anndata(adata, self.model_path, return_reference_var_names=True)

        logger.info("Loading query data into the pretrained scVI model from %s", self.model_path)
        vae_q = scvi.model.SCVI.load_query_data(adata, self.model_path)

        # Trick scVI into thinking the model is already trained
        vae_q.is_trained = True

        # Compute the latent representation
        logger.info("Computing latent representation with scVI.")
        latent = vae_q.get_latent_representation()

        logger.info("scVI embedding complete. Latent shape: %s", latent.shape)
        return latent


class PCADataEmbedder(DataEmbedder):
    """
    Data embedder that uses Principal Component Analysis (PCA) to generate embeddings.

    Parameters
    ----------
    n_components
        Number of principal components to retain in the embeddings.
    """

    def __init__(self, n_components: int = 128):
        """Initializes the PCADataEmbedder with the specified number of components."""
        self.n_components = n_components
        self.pca = None

    def embed(self, adata: anndata.AnnData) -> np.ndarray:
        """
        Generates PCA embeddings for the given AnnData object.

        Parameters
        ----------
        adata
            The AnnData object containing the data to be transformed.

        Returns
        -------
        PCA-transformed embeddings.
        """
        # Assuming adata.X contains the raw data matrix
        self.pca = PCA(n_components=self.n_components)
        embeddings = self.pca.fit_transform(adata.X)
        return embeddings


class PlaceholderDataEmbedder(DataEmbedder):
    """Generates random embeddings as a placeholder for real data embeddings."""

    def embed(self, adata: anndata.AnnData) -> np.ndarray:
        """
        Generates random embeddings for the given AnnData object.

        Parameters
        ----------
        adata
            The AnnData object to generate embeddings for.

        Returns
        -------
        Randomly generated embeddings.
        """
        n_samples = adata.n_obs
        embedding_dim = 64  # Example embedding dimension
        embeddings = np.random.rand(n_samples, embedding_dim)
        return embeddings
