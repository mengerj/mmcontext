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


'''
class Evaluator:
    """
    Evaluates embeddings and reconstructed features using specified metrics.

    This class provides methods to evaluate embeddings and reconstructed features
    using metrics from the scib package. It allows users to compute metrics on
    feature space, embedding space, and to compare metrics between two spaces.

    Parameters
    ----------
    adata_pre : AnnData
        The AnnData object before integration (original data).
    adata_post : AnnData
        The AnnData object after integration (integrated data or data with embeddings).
    batch_key : str
        Key in `adata.obs` indicating batch labels.
    label_key : str
        Key in `adata.obs` indicating cell type labels.
    embedding_key : str | list[str]
        Key in `adata.obsm` where the embeddings are stored in `adata_post`. Multiple embeddings can be specified.
    n_top_genes : int, optional
    logger : logging.Logger, optional
        Logger for logging messages. If None, a default logger will be used.
    """

    def __init__(
        self,
        adata_pre: AnnData,
        adata_post: AnnData,
        batch_key: str,
        label_key: str,
        embedding_key: str | list[str],
        n_top_genes: int | None = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.adata_pre = adata_pre
        self.adata_post = adata_post
        self.batch_key = batch_key
        self.label_key = label_key
        self.embedding_key = embedding_key
        self.n_top_genes = n_top_genes
        self.logger = logger or logging.getLogger(__name__)

    def compute_feature_space_metrics(self) -> Dict[str, Any]:
        """
        Computes metrics on the feature space of the integrated data (`adata_post`).

        The following metrics are computed:
        - Adjusted Rand Index (ARI)
        - Normalized Mutual Information (NMI)
        - Silhouette Score (ASW)
        - Isolated Labels ASW
        - Isolated Labels F1
        - Clustering LISI (cLISI)
        - Graph Connectivity
        - Silhouette Batch
        - iLISI

        Returns
        -------
        results : dict
            Dictionary containing the computed metrics with keys indicating the metric and the space.
            For example, 'ARI_feature' for ARI computed on feature space.
        """
        bio_results = {}
        batch_results = {}
        adata = self.adata_post.copy()
        # Preprocess data: reduce dimensions using PCA and compute neighbors
        if self.n_top_genes is not None:
            self.logger.info(f"Reducing data to {self.n_top_genes} top genes...")
            try:
                pp.reduce_data(
                    adata, n_top_genes=self.n_top_genes, batch_key=self.batch_key, pca=True, neighbors=True
             )
            except Exception as e:
                self.logger.error(f"Error in data reduction: {e}")
                self.logger.error("Using full data for metrics computation.")

        # Bio-conservation metrics
        # ARI
        try:
            me.cluster_optimal_resolution(
                adata, cluster_key="cluster", label_key=self.label_key
            )
        except Exception as e:
            self.logger.error(f"Error computing optimal clusters on feature space: {e}")
            raise RuntimeError(f"Error computing optimal clusters on feature space: {e}")

        try:
            ari_score = me.ari(adata, cluster_key="cluster", label_key=self.label_key)
            bio_results['ARI_feature'] = ari_score
        except Exception as e:
            self.logger.error(f"Error computing ARI on feature space: {e}")

        # NMI
        try:
            nmi_score = me.nmi(adata, cluster_key="cluster", label_key=self.label_key)
            bio_results['NMI_feature'] = nmi_score
        except Exception as e:
            self.logger.error(f"Error computing NMI on feature space: {e}")

        # Silhouette Score (ASW)
        try:
            asw_score = me.silhouette(adata, label_key=self.label_key, embed='X_pca')
            bio_results['ASW_feature'] = asw_score
        except Exception as e:
            self.logger.error(f"Error computing ASW on feature space: {e}")

        # Isolated Labels ASW
        try:
            isolated_labels_asw = me.isolated_labels_asw(
                adata, label_key=self.label_key, embed='X_pca', batch_key=self.batch_key
            )
            bio_results['Isolated_Labels_ASW_feature'] = isolated_labels_asw
        except Exception as e:
            self.logger.error(f"Error computing Isolated Labels ASW on feature space: {e}")

        # Isolated Labels F1
        try:
            isolated_labels_f1 = me.isolated_labels_f1(
                adata, label_key=self.label_key, batch_key=self.batch_key, embed='X_pca'
            )
            bio_results['Isolated_Labels_F1_feature'] = isolated_labels_f1
        except Exception as e:
            self.logger.error(f"Error computing Isolated Labels F1 on feature space: {e}")

        # cell-type LISI (cLISI)
        try:
            clisi_score = me.clisi_graph(adata, label_key=self.label_key, type_='full')
            bio_results['cLISI_feature'] = clisi_score
        except Exception as e:
            self.logger.error(f"Error computing cLISI on feature space: {e}")

        # Graph Connectivity
        try:
            graph_conn = me.graph_connectivity(adata, label_key=self.label_key)
            batch_results['Graph_Connectivity_feature'] = graph_conn
        except Exception as e:
            self.logger.error(f"Error computing Graph Connectivity on feature space: {e}")

        # Silhouette Batch
        try:
            silhouette_batch = me.silhouette_batch(
                adata, batch_key=self.batch_key, label_key=self.label_key, embed='X_pca'
            )
            batch_results['Silhouette_Batch_feature'] = silhouette_batch
        except Exception as e:
            self.logger.error(f"Error computing Silhouette Batch on feature space: {e}")

        # iLISI
        try:
            ilisi_score = me.ilisi_graph(adata, batch_key=self.batch_key, type_='full')
            batch_results['iLISI_feature'] = ilisi_score
        except Exception as e:
            self.logger.error(f"Error computing iLISI on feature space: {e}")

        # Compute average bio-conservation and batch-integration scores
        self.current_metric = 'feature'
        results = self.add_average_scores(bio_results, batch_results)
        return results

    def add_average_scores(self, bio_results, batch_results) -> Dict[str, Any]:
        """Add average bio-conservation and batch-integration scores to the results.

        Parameters
        ----------
        bio_results
            Dictionary containing bio-conservation metrics.
        batch_results
            Dictionary containing batch-integration metrics.

        Returns
        ----------
            Dict[str, Any]: Dictionary containing the computed metrics with keys indicating the metric and the space.
        """
        bio_scores = [
            bio_results[key] for key in bio_results
        ]
        if bio_scores:
            bio_avg = sum(bio_scores) / len(bio_scores)
            bio_results[f'Bio_Conservation_Score_{self.current_metric}'] = bio_avg
        else:
            bio_results[f'Bio_Conservation_Score_{self.current_metric}'] = None

        batch_scores = [
            batch_results[key] for key in batch_results
        ]
        if batch_scores:
            batch_avg = sum(batch_scores) / len(batch_scores)
            batch_results[f'Batch_Integration_Score_{self.current_metric}'] = batch_avg
        else:
            batch_results[f'Batch_Integration_Score_{self.current_metric}'] = None

        results = {**bio_results, **batch_results}

        results[f"Overall_Score_{self.current_metric}"] = (0.6*bio_avg + 0.4*batch_avg) / 2
        return results

    def compute_embedding_space_metrics(self) -> Dict[str, Any]:
        """
        Computes metrics on the embedding space of the integrated data (`adata_post`).

        The following metrics are computed:
        - Adjusted Rand Index (ARI)
        - Normalized Mutual Information (NMI)
        - Silhouette Score (ASW)
        - Isolated Labels ASW
        - Isolated Labels F1
        - Clustering LISI (cLISI)
        - Graph Connectivity
        - Silhouette Batch
        - iLISI

        Returns
        -------
        results : dict
            Dictionary containing the computed metrics with keys indicating the metric and the space.
            For example, 'ARI_embedding' for ARI computed on embedding space.
        """
        bio_results = {}
        batch_results = {}
        adata = self.adata_post.copy()

        # Compute neighbors using the embedding
        try:
            sc.pp.neighbors(adata, use_rep=self.embedding_key)
        except Exception as e:
            self.logger.error(f"Error computing neighbors using embedding: {e}")

        # ARI
        try:
            me.cluster_optimal_resolution(
                adata, cluster_key="cluster", label_key=self.label_key
            )
            ari_score = me.ari(adata, cluster_key="cluster", label_key=self.label_key)
            bio_results['ARI_embedding'] = ari_score
        except Exception as e:
            self.logger.error(f"Error computing ARI on embedding space: {e}")

        # NMI
        try:
            nmi_score = me.nmi(adata, cluster_key="cluster", label_key=self.label_key)
            bio_results['NMI_embedding'] = nmi_score
        except Exception as e:
            self.logger.error(f"Error computing NMI on embedding space: {e}")

        # Silhouette Score (ASW)
        try:
            asw_score = me.silhouette(adata, label_key=self.label_key, embed=self.embedding_key)
            bio_results['ASW_embedding'] = asw_score
        except Exception as e:
            self.logger.error(f"Error computing ASW on embedding space: {e}")

        # Isolated Labels ASW
        try:
            isolated_labels_asw = me.isolated_labels_asw(
                adata, label_key=self.label_key, embed=self.embedding_key, batch_key=self.batch_key
            )
            bio_results['Isolated_Labels_ASW_embedding'] = isolated_labels_asw
        except Exception as e:
            self.logger.error(f"Error computing Isolated Labels ASW on embedding space: {e}")

        # Isolated Labels F1
        try:
            isolated_labels_f1 = me.isolated_labels_f1(
                adata, batch_key=self.batch_key, label_key=self.label_key, embed=self.embedding_key
            )
            bio_results['Isolated_Labels_F1_embedding'] = isolated_labels_f1
        except Exception as e:
            self.logger.error(f"Error computing Isolated Labels F1 on embedding space: {e}")

        # cell-type LISI (cLISI)
        try:
            clisi_score = me.clisi_graph(
                adata, label_key=self.label_key, type_='embed', use_rep=self.embedding_key
            )
            bio_results['cLISI_embedding'] = clisi_score
        except Exception as e:
            self.logger.error(f"Error computing cLISI on embedding space: {e}")

        # Graph Connectivity
        try:
            graph_conn = me.graph_connectivity(adata, label_key=self.label_key)
            batch_results['Graph_Connectivity_embedding'] = graph_conn
        except Exception as e:
            self.logger.error(f"Error computing Graph Connectivity on embedding space: {e}")

        # Silhouette Batch
        try:
            silhouette_batch = me.silhouette_batch(
                adata, batch_key=self.batch_key, label_key=self.label_key, embed=self.embedding_key
            )
            batch_results['Silhouette_Batch_embedding'] = silhouette_batch
        except Exception as e:
            self.logger.error(f"Error computing Silhouette Batch on embedding space: {e}")

        # iLISI
        try:
            ilisi_score = me.ilisi_graph(
                adata, batch_key=self.batch_key, type_='embed', use_rep=self.embedding_key
            )
            batch_results['iLISI_embedding'] = ilisi_score
        except Exception as e:
            self.logger.error(f"Error computing iLISI on embedding space: {e}")

        # Compute average bio-conservation and batch-integration scores
        self.current_metric = 'embedding'
        results = self.add_average_scores(bio_results, batch_results)

        return results

    def compute_comparison_metrics(self) -> Dict[str, Any]:
        """
        Computes metrics that compare between `adata_pre` and `adata_post`.

        The following metrics are computed:
        - HVG Overlap
        - PCR Comparison

        Returns
        -------
        results : dict
            Dictionary containing the computed comparison metrics with meaningful keys.
        """
        results = {}

        # HVG Overlap
        try:
            adata_pre = self.adata_pre.copy()
            adata_post = self.adata_post.copy()

            hvg_overlap = me.hvg_overlap(adata_pre, adata_post, batch_key=self.batch_key)
            results['HVG_Overlap'] = hvg_overlap
        except Exception as e:
            self.logger.error(f"Error computing HVG Overlap: {e}")

        # PCR Comparison
        try:
            # Unintegrated vs integrated feature space
            pcr_feature = me.pcr_comparison(
                self.adata_pre, self.adata_post, covariate=self.batch_key
            )
            results['PCR_feature_vs_intfeature'] = pcr_feature
        except Exception as e:
            self.logger.error(f"Error computing PCR Comparison on feature space: {e}")

        try:
            # Unintegrated vs embedding output
            pcr_embedding = me.pcr_comparison(
                self.adata_pre, self.adata_post, covariate=self.batch_key, embed=self.embedding_key
            )
            results['PCR_feature_vs_embedding'] = pcr_embedding
        except Exception as e:
            self.logger.error(f"Error computing PCR Comparison on embedding space: {e}")

        try:
            # integrated vs embedding output
            pcr_embedding = me.pcr_comparison(
                self.adata_post, self.adata_pre, covariate=self.batch_key, embed=self.embedding_key
            )
            results['PCR_intfeature_vs_embedding'] = pcr_embedding
        except Exception as e:
            self.logger.error(f"Error computing PCR Comparison on embedding space: {e}")

        return results

    def evaluate(self) -> Dict[str, Any]:
        """
        Computes all metrics and returns the results, including averages of bio-conservation and batch-integration scores.

        Returns
        -------
        results : dict
            Dictionary containing all computed metrics and average scores.
            The keys are:
            - Metrics computed on feature space (e.g., 'ARI_feature', 'ASW_feature')
            - Metrics computed on embedding space (e.g., 'ARI_embedding', 'ASW_embedding')
            - Comparison metrics (e.g., 'HVG_Overlap', 'PCR_feature', 'PCR_embedding')
            - 'Bio_Conservation_Score': Average score of bio-conservation metrics
            - 'Batch_Integration_Score': Average score of batch-integration metrics
        """
        # Suppress specific warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        results = {}

        # Compute feature space metrics
        self.logger.info("Computing feature space metrics...")
        feature_metrics = self.compute_feature_space_metrics()
        results.update(feature_metrics)

        # Compute embedding space metrics
        # check if embedding key is provided as a list
        if isinstance(self.embedding_key, list):
            all_embeddings = self.embedding_key
            embedding_metrics = {}
            for key in all_embeddings:
                self.embedding_key = key
                self.logger.info(f"Computing embedding space metrics for {key}...")
                metrics = self.compute_embedding_space_metrics()
                #Add the key to all names of the metrics
                metrics = {f"{k}_{key}": v for k, v in metrics.items()}
                embedding_metrics.update(metrics)
            results.update(embedding_metrics)
        else:
            self.logger.info("Computing embedding space metrics...")
            embedding_metrics = self.compute_embedding_space_metrics()
            results.update(embedding_metrics)

        # Compute comparison metrics
        self.logger.info("Computing comparison metrics...")
        comparison_metrics = self.compute_comparison_metrics()
        results.update(comparison_metrics)

        return results
'''
