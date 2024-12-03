# mmcontext/pp/embedding_normalizer.py

import logging
from abc import ABC, abstractmethod

import numpy as np
from anndata import AnnData


def configure_normalizer(cfg):
    """Configures the normalizer based on the configuration.

    Parameters
    ----------
    cfg
        The configuration object.

    Returns
    -------
    EmbeddingNormalizer
        The configured normalizer.
    """
    if cfg.type == "z-score":
        return ZScoreNormalizer()
    elif cfg.type == "min-max":
        return MinMaxNormalizer()
    elif cfg.type is None:
        return PlaceHolderNormalizer()
    else:
        raise ValueError(f"Unknown normalizer type: {cfg.type}")


class EmbeddingNormalizer(ABC):
    """Abstract base class for normalizing embeddings in an AnnData object."""

    def __init__(self, logger=None):
        """Initializes the EmbeddingNormalizer.

        Parameters
        ----------
        logger
            An optional logger object to use for logging. Defaults to None.
        """
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def normalize(self, adata: AnnData):
        """Abstract method to normalize embeddings in an AnnData object.

        Applies normalization to embeddings in adata.obsm['d_emb'] and adata.obsm['c_emb'],
        storing the normalized embeddings in adata.obsm['d_emb_norm'] and adata.obsm['c_emb_norm'].

        Parameters
        ----------
        adata
            The AnnData object containing the embeddings to normalize.
        """
        pass

    def check_embeddings(self, adata: AnnData):
        """
        Checks that 'd_emb' and 'c_emb' are present in adata.obsm.

        Parameters
        ----------
        adata
            The AnnData object to check for embeddings.

        Raises
        ------
        KeyError
            If either 'd_emb' or 'c_emb' is missing in adata.obsm.
        """
        if "d_emb" not in adata.obsm:
            self.logger.error("Missing 'd_emb' in adata.obsm")
            raise KeyError("Data embeddings 'd_emb' are missing in adata.obsm")
        if "c_emb" not in adata.obsm:
            self.logger.error("Missing 'c_emb' in adata.obsm")
            raise KeyError("Context embeddings 'c_emb' are missing in adata.obsm")


class PlaceHolderNormalizer(EmbeddingNormalizer):
    """A placeholder normalizer that does nothing."""

    def normalize(self, adata: AnnData):
        """Does nothing."""
        self.logger.info(
            "No normalization applied, but still stored in adata.obsm['d_emb_norm'] and adata.obsm['c_emb_norm']"
        )
        adata.obsm["d_emb_norm"] = adata.obsm["d_emb"]
        adata.obsm["c_emb_norm"] = adata.obsm["c_emb"]


class ZScoreNormalizer(EmbeddingNormalizer):
    """Normalizes embeddings using z-score normalization.

    Uses z-score normalization to normalize embeddings in an AnnData object

    Parameters
    ----------
    logger
        An optional logger object to use for logging. Defaults to None.
    """

    def __init__(self, logger=None):
        super().__init__(logger)
        self.means_d = None
        self.stds_d = None
        self.means_c = None
        self.stds_c = None

    def normalize(self, adata: AnnData):
        """Applies z-score normalization.

        Works on embeddings in adata.obsm['d_emb'] and adata.obsm['c_emb'],
        storing the normalized embeddings in adata.obsm['d_emb_norm'] and adata.obsm['c_emb_norm'].

        Parameters
        ----------
        adata
            The AnnData object containing the embeddings to normalize.
        """
        self.logger.info("Normalizing embeddings using z-score normalization...")

        # Check if embeddings exist
        self.check_embeddings(adata)

        # Normalize data embeddings (d_emb)
        d_emb = adata.obsm["d_emb"]

        if set(np.unique(adata.obsm["d_emb"])) == {0, 1}:
            self.logger.info("Data embeddings are already binary.")
            adata.obsm["d_emb_norm"] = d_emb
        else:
            self.means_d = np.mean(d_emb, axis=0)
            self.stds_d = np.std(d_emb, axis=0)
            d_emb_norm = (d_emb - self.means_d) / self.stds_d
            adata.obsm["d_emb_norm"] = d_emb_norm

        # Normalize context embeddings (c_emb)
        c_emb = adata.obsm["c_emb"]
        if set(np.unique(adata.obsm["c_emb"])) == {0, 1}:
            self.logger.info("Context embeddings are already binary.")
            adata.obsm["c_emb_norm"] = c_emb
        else:
            self.means_c = np.mean(c_emb, axis=0)
            self.stds_c = np.std(c_emb, axis=0)
            c_emb_norm = (c_emb - self.means_c) / self.stds_c
            adata.obsm["c_emb_norm"] = c_emb_norm


class MinMaxNormalizer(EmbeddingNormalizer):
    """Normalizes embeddings using min-max normalization."""

    def __init__(self, logger=None):
        """Initializes the MinMaxNormalizer.

        Parameters
        ----------
        logger
            An optional logger object to use for logging. Defaults to None.
        """
        super().__init__(logger)
        self.mins_d = None
        self.maxs_d = None
        self.mins_c = None
        self.maxs_c = None
        self.logger = logging.getLogger(__name__)

    def normalize(self, adata: AnnData):
        """Applies min-max normalization.

        works on embeddings in adata.obsm['d_emb'] and adata.obsm['c_emb'],
        storing the normalized embeddings in adata.obsm['d_emb_norm'] and adata.obsm['c_emb_norm'].

        Parameters
        ----------
        adata
            The AnnData object containing the embeddings to normalize
        """
        self.logger.info("Normalizing embeddings using min-max normalization...")

        # Check if embeddings exist
        self.check_embeddings(adata)

        # Normalize data embeddings (d_emb)
        d_emb = adata.obsm["d_emb"]
        if set(np.unique(adata.obsm["d_emb"])) == {0, 1}:
            self.logger.info("Data embeddings are already binary.")
            adata.obsm["d_emb_norm"] = d_emb
        else:
            self.mins_d = np.min(d_emb, axis=0)
            self.maxs_d = np.max(d_emb, axis=0)
            d_emb_norm = (d_emb - self.mins_d) / (self.maxs_d - self.mins_d)
            adata.obsm["d_emb_norm"] = d_emb_norm

        # Normalize context embeddings (c_emb)
        c_emb = adata.obsm["c_emb"]
        if set(np.unique(adata.obsm["c_emb"])) == {0, 1}:
            self.logger.info("Context embeddings are already binary.")
            adata.obsm["c_emb_norm"] = c_emb
        else:
            self.mins_c = np.min(c_emb, axis=0)
            self.maxs_c = np.max(c_emb, axis=0)
            c_emb_norm = (c_emb - self.mins_c) / (self.maxs_c - self.mins_c)
            adata.obsm["c_emb_norm"] = c_emb_norm
