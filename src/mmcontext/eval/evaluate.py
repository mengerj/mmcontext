import logging
import warnings
from typing import Any

import pandas as pd
import scanpy as sc
import scib.metrics as me
import scib.preprocessing as pp
from anndata import AnnData


class Evaluator:
    """
    Evaluates embeddings and reconstructed features using specified metrics.

    This class computes metrics on raw data, embeddings, and reconstructed features
    using metrics from the scib package. It outputs a DataFrame with results for each
    evaluation type.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the data. May include embeddings in `.obsm` and
        reconstructed data in `.layers['reconstructed']`.
    batch_key : str
        Key in `adata.obs` indicating batch labels.
    label_key : str
        Key in `adata.obs` indicating cell type labels.
    embedding_key : Union[str, List[str]], optional
        Key(s) in `adata.obsm` where the embeddings are stored. Can be a string or a list of strings.
    data_id : str, optional
        Identifier for the dataset, including preprocessing steps.
    n_top_genes : int, optional
        Number of top genes used for HVG selection. If None, HVG is not performed.
    logger : logging.Logger, optional
        Logger for logging messages. If None, a default logger will be used.
    """

    def __init__(
        self,
        adata: AnnData,
        batch_key: str,
        label_key: str,
        embedding_key: str | list[str] = None,
        data_id: str = "",
        n_top_genes: int | None = None,
        logger: logging.Logger | None = None,
    ):
        self.adata = adata
        self.batch_key = batch_key
        self.label_key = label_key
        self.embedding_key = embedding_key
        self.data_id = data_id
        self.n_top_genes = n_top_genes
        self.logger = logger or logging.getLogger(__name__)

    def evaluate(self) -> pd.DataFrame:
        """
        Computes metrics for raw data, embeddings, and reconstructed data, and returns the results as a DataFrame.

        Returns
        -------
        results_df : pd.DataFrame
            DataFrame containing the computed metrics and dataset information.
        """
        # Suppress specific warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Initialize a list to collect results
        results_list = []

        # Compute metrics on raw data
        self.logger.info("Computing metrics on raw data...")
        raw_metrics = self.compute_feature_space_metrics()
        raw_metrics["type"] = "raw"
        results_list.append(raw_metrics)

        # Check for reconstructed data
        if "reconstructed" in self.adata.layers:
            self.logger.info("Computing metrics on reconstructed data...")
            reconstructed_metrics = self.compute_reconstructed_metrics()
            reconstructed_metrics["type"] = "reconstructed"
            results_list.append(reconstructed_metrics)

        # Compute metrics on embeddings
        if self.embedding_key is not None:
            if isinstance(self.embedding_key, list):
                for key in self.embedding_key:
                    if key is None:
                        continue
                    self.logger.info(f"Computing metrics on embedding '{key}'...")
                    embedding_metrics = self.compute_embedding_metrics(embedding_key=key)
                    embedding_metrics["type"] = f"embedding_{key}"
                    results_list.append(embedding_metrics)
            else:
                self.logger.info(f"Computing metrics on embedding '{self.embedding_key}'...")
                embedding_metrics = self.compute_embedding_metrics(embedding_key=self.embedding_key)
                embedding_metrics["type"] = f"embedding_{self.embedding_key}"
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

    def compute_feature_space_metrics(self) -> dict[str, Any]:
        """
        Computes metrics on the raw feature space of the data.

        Returns
        -------
        results : dict
            Dictionary containing the computed metrics.
        """
        bio_results = {}
        batch_results = {}
        adata = self.adata.copy()

        # Preprocess data: reduce dimensions using PCA and compute neighbors
        if self.n_top_genes is not None:
            self.logger.info(f"Reducing data to {self.n_top_genes} top genes using HVG selection...")
            try:
                pp.reduce_data(adata, n_top_genes=self.n_top_genes, batch_key=self.batch_key, pca=True, neighbors=True)
            except Exception as e:
                self.logger.error(f"Error in data reduction: {e}")
                self.logger.error("Using full data for metrics computation.")
        else:
            # If HVG not performed, compute PCA on all genes
            self.logger.info("Computing PCA on all genes...")
            sc.pp.pca(adata, n_comps=50)
            sc.pp.neighbors(adata, use_rep="X_pca")

        # Bio-conservation metrics
        try:
            me.cluster_optimal_resolution(adata, cluster_key="cluster", label_key=self.label_key)
            ari_score = me.ari(adata, cluster_key="cluster", label_key=self.label_key)
            bio_results["ARI"] = ari_score
        except Exception as e:
            self.logger.error(f"Error computing ARI on raw data: {e}")

        try:
            nmi_score = me.nmi(adata, cluster_key="cluster", label_key=self.label_key)
            bio_results["NMI"] = nmi_score
        except Exception as e:
            self.logger.error(f"Error computing NMI on raw data: {e}")

        try:
            asw_score = me.silhouette(adata, label_key=self.label_key, embed="X_pca")
            bio_results["ASW"] = asw_score
        except Exception as e:
            self.logger.error(f"Error computing ASW on raw data: {e}")

        try:
            isolated_labels_asw = me.isolated_labels_asw(
                adata, label_key=self.label_key, embed="X_pca", batch_key=self.batch_key
            )
            bio_results["Isolated_Labels_ASW"] = isolated_labels_asw
        except Exception as e:
            self.logger.error(f"Error computing Isolated Labels ASW on raw data: {e}")

        try:
            isolated_labels_f1 = me.isolated_labels_f1(
                adata, label_key=self.label_key, batch_key=self.batch_key, embed="X_pca"
            )
            bio_results["Isolated_Labels_F1"] = isolated_labels_f1
        except Exception as e:
            self.logger.error(f"Error computing Isolated Labels F1 on raw data: {e}")

        try:
            clisi_score = me.clisi_graph(adata, label_key=self.label_key, type_="full")
            bio_results["cLISI"] = clisi_score
        except Exception as e:
            self.logger.error(f"Error computing cLISI on raw data: {e}")

        # Batch-integration metrics
        try:
            graph_conn = me.graph_connectivity(adata, label_key=self.label_key)
            batch_results["Graph_Connectivity"] = graph_conn
        except Exception as e:
            self.logger.error(f"Error computing Graph Connectivity on raw data: {e}")

        try:
            silhouette_batch = me.silhouette_batch(
                adata, batch_key=self.batch_key, label_key=self.label_key, embed="X_pca"
            )
            batch_results["Silhouette_Batch"] = silhouette_batch
        except Exception as e:
            self.logger.error(f"Error computing Silhouette Batch on raw data: {e}")

        try:
            ilisi_score = me.ilisi_graph(adata, batch_key=self.batch_key, type_="full")
            batch_results["iLISI"] = ilisi_score
        except Exception as e:
            self.logger.error(f"Error computing iLISI on raw data: {e}")

        # Compute average scores
        results = self.compute_average_scores(bio_results, batch_results)
        return results

    def compute_embedding_metrics(self, embedding_key: str) -> dict[str, Any]:
        """
        Computes metrics on the specified embedding.

        Parameters
        ----------
        embedding_key : str
            Key in `adata.obsm` where the embedding is stored.

        Returns
        -------
        results : dict
            Dictionary containing the computed metrics.
        """
        bio_results = {}
        batch_results = {}
        adata = self.adata.copy()

        # Compute neighbors using the embedding
        try:
            sc.pp.neighbors(adata, use_rep=embedding_key)
        except Exception as e:
            self.logger.error(f"Error computing neighbors using embedding '{embedding_key}': {e}")

        # Bio-conservation metrics
        try:
            me.cluster_optimal_resolution(adata, cluster_key="cluster", label_key=self.label_key)
            ari_score = me.ari(adata, cluster_key="cluster", label_key=self.label_key)
            bio_results["ARI"] = ari_score
        except Exception as e:
            self.logger.error(f"Error computing ARI on embedding '{embedding_key}': {e}")

        try:
            nmi_score = me.nmi(adata, cluster_key="cluster", label_key=self.label_key)
            bio_results["NMI"] = nmi_score
        except Exception as e:
            self.logger.error(f"Error computing NMI on embedding '{embedding_key}': {e}")

        try:
            asw_score = me.silhouette(adata, label_key=self.label_key, embed=embedding_key)
            bio_results["ASW"] = asw_score
        except Exception as e:
            self.logger.error(f"Error computing ASW on embedding '{embedding_key}': {e}")

        try:
            isolated_labels_asw = me.isolated_labels_asw(
                adata, label_key=self.label_key, embed=embedding_key, batch_key=self.batch_key
            )
            bio_results["Isolated_Labels_ASW"] = isolated_labels_asw
        except Exception as e:
            self.logger.error(f"Error computing Isolated Labels ASW on embedding '{embedding_key}': {e}")

        try:
            isolated_labels_f1 = me.isolated_labels_f1(
                adata, label_key=self.label_key, batch_key=self.batch_key, embed=embedding_key
            )
            bio_results["Isolated_Labels_F1"] = isolated_labels_f1
        except Exception as e:
            self.logger.error(f"Error computing Isolated Labels F1 on embedding '{embedding_key}': {e}")

        try:
            clisi_score = me.clisi_graph(adata, label_key=self.label_key, type_="embed", use_rep=embedding_key)
            bio_results["cLISI"] = clisi_score
        except Exception as e:
            self.logger.error(f"Error computing cLISI on embedding '{embedding_key}': {e}")

        # Batch-integration metrics
        try:
            graph_conn = me.graph_connectivity(adata, label_key=self.label_key)
            batch_results["Graph_Connectivity"] = graph_conn
        except Exception as e:
            self.logger.error(f"Error computing Graph Connectivity on embedding '{embedding_key}': {e}")

        try:
            silhouette_batch = me.silhouette_batch(
                adata, batch_key=self.batch_key, label_key=self.label_key, embed=embedding_key
            )
            batch_results["Silhouette_Batch"] = silhouette_batch
        except Exception as e:
            self.logger.error(f"Error computing Silhouette Batch on embedding '{embedding_key}': {e}")

        try:
            ilisi_score = me.ilisi_graph(adata, batch_key=self.batch_key, type_="embed", use_rep=embedding_key)
            batch_results["iLISI"] = ilisi_score
        except Exception as e:
            self.logger.error(f"Error computing iLISI on embedding '{embedding_key}': {e}")

        # PCR Comparison
        try:
            pcr_embedding = me.pcr_comparison(self.adata, self.adata, covariate=self.batch_key, embed=embedding_key)
            batch_results["PCR"] = pcr_embedding
        except Exception as e:
            self.logger.error(f"Error computing PCR Comparison on embedding '{embedding_key}': {e}")

        # Compute average scores
        results = self.compute_average_scores(bio_results, batch_results)
        return results

    def compute_reconstructed_metrics(self) -> dict[str, Any]:
        """
        Computes metrics on the reconstructed data.

        Returns
        -------
        results : dict
            Dictionary containing the computed metrics.
        """
        bio_results = {}
        batch_results = {}
        adata = self.adata.copy()

        # Replace adata.X with reconstructed data
        if "reconstructed" not in adata.layers:
            self.logger.error("Reconstructed data not found in 'adata.layers['reconstructed']'.")
            return {}

        adata.X = adata.layers["reconstructed"]

        # Preprocess data: reduce dimensions using PCA and compute neighbors
        if self.n_top_genes is not None:
            self.logger.info(f"Reducing reconstructed data to {self.n_top_genes} top genes using HVG selection...")
            try:
                pp.reduce_data(adata, n_top_genes=self.n_top_genes, batch_key=self.batch_key, pca=True, neighbors=True)
            except Exception as e:
                self.logger.error(f"Error in data reduction on reconstructed data: {e}")
                self.logger.error("Using full reconstructed data for metrics computation.")
        else:
            # If HVG not performed, compute PCA on all genes
            self.logger.info("Computing PCA on all reconstructed genes...")
            # TODO: custom n_comps
            sc.pp.pca(adata, n_comps=50)
            sc.pp.neighbors(adata, use_rep="X_pca")

        # Bio-conservation metrics
        try:
            me.cluster_optimal_resolution(adata, cluster_key="cluster", label_key=self.label_key)
            ari_score = me.ari(adata, cluster_key="cluster", label_key=self.label_key)
            bio_results["ARI"] = ari_score
        except Exception as e:
            self.logger.error(f"Error computing ARI on reconstructed data: {e}")

        try:
            nmi_score = me.nmi(adata, cluster_key="cluster", label_key=self.label_key)
            bio_results["NMI"] = nmi_score
        except Exception as e:
            self.logger.error(f"Error computing NMI on reconstructed data: {e}")

        try:
            asw_score = me.silhouette(adata, label_key=self.label_key, embed="X_pca")
            bio_results["ASW"] = asw_score
        except Exception as e:
            self.logger.error(f"Error computing ASW on reconstructed data: {e}")

        try:
            isolated_labels_asw = me.isolated_labels_asw(
                adata, label_key=self.label_key, embed="X_pca", batch_key=self.batch_key
            )
            bio_results["Isolated_Labels_ASW"] = isolated_labels_asw
        except Exception as e:
            self.logger.error(f"Error computing Isolated Labels ASW on reconstructed data: {e}")

        try:
            isolated_labels_f1 = me.isolated_labels_f1(
                adata, label_key=self.label_key, batch_key=self.batch_key, embed="X_pca"
            )
            bio_results["Isolated_Labels_F1"] = isolated_labels_f1
        except Exception as e:
            self.logger.error(f"Error computing Isolated Labels F1 on reconstructed data: {e}")

        try:
            clisi_score = me.clisi_graph(adata, label_key=self.label_key, type_="full")
            bio_results["cLISI"] = clisi_score
        except Exception as e:
            self.logger.error(f"Error computing cLISI on reconstructed data: {e}")

        # Batch-integration metrics
        try:
            graph_conn = me.graph_connectivity(adata, label_key=self.label_key)
            batch_results["Graph_Connectivity"] = graph_conn
        except Exception as e:
            self.logger.error(f"Error computing Graph Connectivity on reconstructed data: {e}")

        try:
            silhouette_batch = me.silhouette_batch(
                adata, batch_key=self.batch_key, label_key=self.label_key, embed="X_pca"
            )
            batch_results["Silhouette_Batch"] = silhouette_batch
        except Exception as e:
            self.logger.error(f"Error computing Silhouette Batch on reconstructed data: {e}")

        try:
            ilisi_score = me.ilisi_graph(adata, batch_key=self.batch_key, type_="full")
            batch_results["iLISI"] = ilisi_score
        except Exception as e:
            self.logger.error(f"Error computing iLISI on reconstructed data: {e}")

        # PCR Comparison
        try:
            pcr_reconstructed = me.pcr_comparison(self.adata, adata, covariate=self.batch_key)
            batch_results["PCR"] = pcr_reconstructed
        except Exception as e:
            self.logger.error(f"Error computing PCR Comparison on reconstructed data: {e}")

        # Compute average scores
        results = self.compute_average_scores(bio_results, batch_results)
        return results

    def compute_average_scores(self, bio_results: dict[str, Any], batch_results: dict[str, Any]) -> dict[str, Any]:
        """
        Computes average bio-conservation and batch-integration scores.

        Parameters
        ----------
        bio_results : dict
            Dictionary containing bio-conservation metrics.
        batch_results : dict
            Dictionary containing batch-integration metrics.

        Returns
        -------
        results : dict
            Combined dictionary with average scores added.
        """
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
