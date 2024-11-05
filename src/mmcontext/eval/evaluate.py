import logging
from typing import Any

import scanpy as sc
import scib
import scib.metrics
from anndata import AnnData


class Evaluator:
    """
    Evaluates embeddings and reconstructed features using specified metrics.

    This class provides methods to evaluate embeddings and reconstructed features
    using metrics from the scib package. It allows users to specify which metrics
    to compute via a configuration dictionary.

    Parameters
    ----------
    adata
        The AnnData object containing the embeddings and original data.
    batch_key
        Key in `adata.obs` indicating batch labels.
    label_key
        Key in `adata.obs` indicating cell type labels.
    embedding_key
        Key in `adata.obsm` where the embeddings are stored.
    adata_int
        The integrated AnnData object containing reconstructed features. If None,
        `adata` will be used for both original and integrated data.
    config
        Configuration dictionary specifying which metrics to compute.
    logger
        Logger for logging messages. If None, a default logger will be used.
    """

    def __init__(
        self,
        adata: AnnData,
        batch_key: str,
        label_key: str,
        embedding_key: str,
        adata_int: AnnData | None = None,
        config: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.adata = adata
        self.adata_int = adata_int if adata_int is not None else adata
        self.batch_key = batch_key
        self.label_key = label_key
        self.embedding_key = embedding_key
        self.config = config if config is not None else {}
        self.logger = logger or logging.getLogger(__name__)

    def evaluate_embeddings(self):
        """
        Evaluates the embeddings using the specified metrics.

        This method computes the specified metrics using the scib package.
        It prepares the AnnData object by computing the neighborhood graph
        using the embeddings specified by `embedding_key`.

        Returns
        -------
        results
            Dictionary containing the computed metrics.
        """
        # Ensure that the embedding is used to compute the neighborhood graph
        self.logger.info("Computing neighborhood graph using embeddings.")
        sc.pp.neighbors(self.adata, use_rep=self.embedding_key)

        # Prepare parameters for scib.metrics.metrics function
        metrics_params = {
            "adata": self.adata,
            "adata_int": self.adata_int,
            "batch_key": self.batch_key,
            "label_key": self.label_key,
            "embed": self.embedding_key,
        }

        # Set default metrics to False
        metrics_defaults = {
            "isolated_labels_asw_": False,
            "silhouette_": False,
            "hvg_score_": False,
            "graph_conn_": False,
            "pcr_": False,
            "isolated_labels_f1_": False,
            "trajectory_": False,
            "nmi_": False,
            "ari_": False,
            "cell_cycle_": False,
            "kBET_": False,
            "ilisi_": False,
            "clisi_": False,
        }

        # Update metrics_defaults with the metrics specified in the config
        metrics_to_compute = self.config.get("metrics", {})
        for key in metrics_defaults:
            metrics_defaults[key] = metrics_to_compute.get(key, False)

        # Add the metrics flags to the parameters
        metrics_params.update(metrics_defaults)

        self.logger.info("Computing metrics using scib.metrics.metrics.")
        # Compute metrics
        results = scib.metrics.metrics(**metrics_params)

        self.logger.info("Metrics computation completed.")
        return results
