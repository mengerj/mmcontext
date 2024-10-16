# mmcontext/pp/embedding_normalizer.py

import logging
from abc import ABC, abstractmethod

import numpy as np
from anndata import AnnData


class EmbeddingNormalizer(ABC):
    """Abstract base class for normalizing embeddings in an AnnData object."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def normalize(self, adata: AnnData):
        """Abstract method to normalize embeddings in an AnnData object.

        Applies normalization to embeddings in adata.obsm['d_emb'] and adata.obsm['c_emb'],
        storing the normalized embeddings in adata.obsm['d_emb_norm'] and adata.obsm['c_emb_norm'].
        """
        pass

    def check_embeddings(self, adata: AnnData):
        """
        Checks that 'd_emb' and 'c_emb' are present in adata.obsm.

        Raises
        ------
        KeyError: If either 'd_emb' or 'c_emb' is missing in adata.obsm.
        """
        if "d_emb" not in adata.obsm:
            self.logger.error("Missing 'd_emb' in adata.obsm")
            raise KeyError("Data embeddings 'd_emb' are missing in adata.obsm")
        if "c_emb" not in adata.obsm:
            self.logger.error("Missing 'c_emb' in adata.obsm")
            raise KeyError("Context embeddings 'c_emb' are missing in adata.obsm")


class ZScoreNormalizer(EmbeddingNormalizer):
    """Normalizes embeddings using z-score normalization."""

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
        """
        self.logger.info("Normalizing embeddings using z-score normalization...")

        # Check if embeddings exist
        self.check_embeddings(adata)

        # Normalize data embeddings (d_emb)
        d_emb = adata.obsm["d_emb"]
        self.means_d = np.mean(d_emb, axis=0)
        self.stds_d = np.std(d_emb, axis=0)
        d_emb_norm = (d_emb - self.means_d) / self.stds_d
        adata.obsm["d_emb_norm"] = d_emb_norm

        # Normalize context embeddings (c_emb)
        c_emb = adata.obsm["c_emb"]
        self.means_c = np.mean(c_emb, axis=0)
        self.stds_c = np.std(c_emb, axis=0)
        c_emb_norm = (c_emb - self.means_c) / self.stds_c
        adata.obsm["c_emb_norm"] = c_emb_norm


class MinMaxNormalizer(EmbeddingNormalizer):
    """Normalizes embeddings using min-max normalization."""

    def __init__(self, logger=None):
        """Initializes the MinMaxNormalizer."""
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
        """
        self.logger.info("Normalizing embeddings using min-max normalization...")

        # Check if embeddings exist
        self.check_embeddings(adata)

        # Normalize data embeddings (d_emb)
        d_emb = adata.obsm["d_emb"]
        self.mins_d = np.min(d_emb, axis=0)
        self.maxs_d = np.max(d_emb, axis=0)
        d_emb_norm = (d_emb - self.mins_d) / (self.maxs_d - self.mins_d)
        adata.obsm["d_emb_norm"] = d_emb_norm

        # Normalize context embeddings (c_emb)
        c_emb = adata.obsm["c_emb"]
        self.mins_c = np.min(c_emb, axis=0)
        self.maxs_c = np.max(c_emb, axis=0)
        c_emb_norm = (c_emb - self.mins_c) / (self.maxs_c - self.mins_c)
        adata.obsm["c_emb_norm"] = c_emb_norm
