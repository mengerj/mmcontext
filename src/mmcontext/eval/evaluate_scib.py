import concurrent.futures
import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scib import metrics as me
from scib import preprocessing as pp


class scibEvaluator:
    """Evaluates embeddings and reconstructed features using specified metrics."""

    def __init__(
        self,
        adata: AnnData,
        batch_key: str,
        label_key: str,
        embedding_key: str | list[str] = None,
        data_id: str = "",
        n_top_genes: int | None = None,
        max_cells: int | None = None,
        logger: logging.Logger | None = None,
    ):
        self.adata = adata
        self.batch_key = batch_key
        self.label_key = label_key
        self.embedding_key = embedding_key
        self.data_id = data_id
        self.n_top_genes = n_top_genes
        self.max_cells = max_cells
        self.logger = logger or logging.getLogger(__name__)

    def evaluate(self) -> pd.DataFrame:
        """Computes metrics for raw data, embeddings, and reconstructed data."""
        # Suppress specific warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Initialize a list to collect results
        results_list = []

        # Compute metrics on raw data
        self.logger.info("Computing metrics on raw data...")
        raw_metrics = self.compute_metrics(
            adata=self.adata.copy(),
            adata_pre=None,
            adata_post=None,
            use_rep=None,
            type_="full",
            data_type="raw",
        )
        results_list.append(raw_metrics)

        # Compute metrics on reconstructed data
        if "reconstructed" in self.adata.layers:
            self.logger.info("Computing metrics on reconstructed data...")
            adata_post = self.adata.copy()
            adata_post.X = adata_post.layers["reconstructed"]
            reconstructed_metrics = self.compute_metrics(
                adata=adata_post,
                adata_pre=self.adata,
                adata_post=adata_post,
                use_rep=None,
                type_="full",
                data_type="reconstructed",
            )
            results_list.append(reconstructed_metrics)

        # Compute metrics on embeddings
        if self.embedding_key is not None:
            embedding_keys = self.embedding_key if isinstance(self.embedding_key, list) else [self.embedding_key]
            for embedding_key in embedding_keys:
                if embedding_key is None:
                    continue
                self.logger.info(f"Computing metrics on embedding '{embedding_key}'...")
                embedding_metrics = self.compute_metrics(
                    adata=self.adata.copy(),
                    adata_pre=self.adata,
                    adata_post=self.adata,
                    use_rep=embedding_key,
                    type_="embed",
                    data_type=f"embedding_{embedding_key}",
                )
                results_list.append(embedding_metrics)

        # Convert results to DataFrame
        results_df = pd.DataFrame(results_list)

        # Add data_id and hvg columns
        results_df["data_id"] = self.data_id
        results_df["hvg"] = self.n_top_genes if self.n_top_genes is not None else "None"

        # Reorder columns
        cols = ["data_id", "hvg", "type"] + [col for col in results_df.columns if col not in ["data_id", "hvg", "type"]]
        results_df = results_df[cols]

        return results_df

    def compute_metrics(
        self,
        adata: AnnData,
        adata_pre: AnnData | None = None,
        adata_post: AnnData | None = None,
        use_rep: str | None = None,
        cluster_key: str = "cluster",
        type_: str = "full",
        data_type: str = "",
    ) -> dict[str, Any]:
        """Computes metrics on the specified data representation."""
        # Subsample adata if max_cells is specified
        if self.max_cells is not None and adata.n_obs > self.max_cells:
            self.logger.info(f"Subsampling to {self.max_cells} cells for {data_type}...")
            adata = adata[np.random.choice(adata.obs_names, self.max_cells, replace=False)].copy()

        # Preprocess data
        if use_rep is None:
            # Preprocess data: reduce dimensions using PCA and compute neighbors
            if self.n_top_genes is not None:
                self.logger.info(f"Reducing data to {self.n_top_genes} top genes using HVG selection...")
                try:
                    pp.reduce_data(
                        adata, n_top_genes=self.n_top_genes, batch_key=self.batch_key, pca=True, neighbors=True
                    )
                except Exception as e:
                    self.logger.error(f"Error in data reduction: {e}")
                    self.logger.error("Using full data for metrics computation.")
            else:
                # If HVG not performed, compute PCA on all genes
                self.logger.info("Computing PCA on all genes...")
                sc.pp.pca(adata, n_comps=50)
                sc.pp.neighbors(adata, use_rep="X_pca")
            embed_key = "X_pca"
        else:
            # Compute neighbors using the specified representation
            embed_key = use_rep
            try:
                sc.pp.neighbors(adata, use_rep=use_rep)
            except Exception as e:
                self.logger.error(f"Error computing neighbors using embedding '{use_rep}': {e}")

        # Perform clustering
        me.cluster_optimal_resolution(adata, cluster_key=cluster_key, label_key=self.label_key)

        # Define metric functions
        bio_metrics = [
            ("ARI", me.ari, {"cluster_key": cluster_key, "label_key": self.label_key}),
            ("NMI", me.nmi, {"cluster_key": cluster_key, "label_key": self.label_key}),
            ("ASW", me.silhouette, {"label_key": self.label_key, "embed": embed_key}),
            (
                "Isolated_Labels_ASW",
                me.isolated_labels_asw,
                {"label_key": self.label_key, "embed": embed_key, "batch_key": self.batch_key},
            ),
            (
                "Isolated_Labels_F1",
                me.isolated_labels_f1,
                {"label_key": self.label_key, "embed": embed_key, "batch_key": self.batch_key},
            ),
            ("cLISI", me.clisi_graph, {"label_key": self.label_key, "type_": type_, "n_cores": 8, "use_rep": use_rep}),
        ]

        batch_metrics = [
            ("Graph_Connectivity", me.graph_connectivity, {"label_key": self.label_key}),
            (
                "Silhouette_Batch",
                me.silhouette_batch,
                {"batch_key": self.batch_key, "label_key": self.label_key, "embed": embed_key},
            ),
            ("iLISI", me.ilisi_graph, {"batch_key": self.batch_key, "type_": type_, "n_cores": 8, "use_rep": use_rep}),
            (
                "PCR",
                me.pcr_comparison,
                {"adata_pre": adata_pre, "adata_post": adata_post, "covariate": self.batch_key, "embed": embed_key},
            ),
        ]

        # Remove PCR if adata_pre or adata_post is None
        if adata_pre is None or adata_post is None:
            batch_metrics = [m for m in batch_metrics if m[0] != "PCR"]

        # Compute metrics in parallel
        bio_results = self.compute_metrics_in_parallel(adata, bio_metrics)
        batch_results = self.compute_metrics_in_parallel(adata, batch_metrics)

        # Compute average scores
        results = self.compute_average_scores(bio_results, batch_results)
        results["type"] = data_type

        return results

    def compute_metrics_in_parallel(self, adata: AnnData, metrics: list[tuple[str, Any, dict[str, Any]]]):
        """Compute metrics in parallel using a ThreadPoolExecutor."""
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_metric = {
                executor.submit(metric_func, adata, **params): name for name, metric_func, params in metrics
            }
            for future in concurrent.futures.as_completed(future_to_metric):
                metric_name = future_to_metric[future]
                try:
                    result = future.result()
                    results[metric_name] = result
                except Exception as e:
                    self.logger.error(f"Error computing {metric_name}: {e}")
        return results

    def compute_average_scores(self, bio_results: dict[str, Any], batch_results: dict[str, Any]) -> dict[str, Any]:
        """Computes average bio-conservation and batch-integration scores."""
        # Calculate average bio-conservation score
        bio_metrics = ["ARI", "NMI", "ASW", "Isolated_Labels_ASW", "Isolated_Labels_F1", "cLISI"]
        bio_scores = [
            bio_results[metric] for metric in bio_metrics if metric in bio_results and bio_results[metric] is not None
        ]
        bio_avg = sum(bio_scores) / len(bio_scores) if bio_scores else None
        bio_results["Bio_Conservation_Score"] = bio_avg

        # Calculate average batch-integration score
        batch_metrics = ["iLISI", "Graph_Connectivity", "Silhouette_Batch", "PCR"]
        batch_scores = [
            batch_results[metric]
            for metric in batch_metrics
            if metric in batch_results and batch_results[metric] is not None
        ]
        batch_avg = sum(batch_scores) / len(batch_scores) if batch_scores else None
        batch_results["Batch_Integration_Score"] = batch_avg

        # Combine results
        results = {**bio_results, **batch_results}

        # Overall score (optional, adjust weighting as needed)
        if bio_avg is not None and batch_avg is not None:
            overall_score = 0.6 * bio_avg + 0.4 * batch_avg
            results["Overall_Score"] = overall_score
        else:
            results["Overall_Score"] = None

        return results
