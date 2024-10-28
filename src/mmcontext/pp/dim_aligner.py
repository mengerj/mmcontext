# mmcontext/pp/dim_aligner.py

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from sklearn.decomposition import PCA


class DimAligner(ABC):
    """
    Abstract base class for aligning dimensions of embeddings.

    This class defines the interface for dimension alignment, including methods for
    reducing and extending embeddings to achieve a consistent target dimension.

    Parameters
    ----------
    latent_dim
        Target dimension for alignment (default: 64).
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        latent_dim: int = 64,
        context_key: str = "c_emb_norm",
        data_key: str = "d_emb_norm",
    ):
        """
        Initializes the DimAligner with the target latent dimension.

        Parameters
        ----------
        latent_dim
            The target dimension to align embeddings to.
        context_key
            The key for context embeddings in adata.obsm. Defaults are normalized embeddings
        data_key
            The key for data embeddings in adata.obsm. Defaults are normalized embeddings
        """
        self.latent_dim = latent_dim
        self.context_key = context_key
        self.data_key = data_key
        self.logger = logger or logging.getLogger(__name__)

    def align(self, adata: AnnData):
        """Aligns the dimensions of data and context embeddings

        Aligns the dimensions of data and context embeddings in an AnnData object to the target dimension.

        Aligned embeddings are stored in 'd_emb_aligned' and 'c_emb_aligned' in adata.obsm.

        If an embedding's dimension is larger than the target dimension, it is reduced.
        If it is smaller, it is extended (padded).

        Parameters
        ----------
        adata
            The AnnData object containing embeddings in 'd_emb' and 'c_emb'.

        Raises
        ------
        ValueError
            If embeddings are missing from adata.obsm.
        """
        if self.data_key not in adata.obsm or self.context_key not in adata.obsm:
            raise ValueError(f"Embeddings {self.data_key} and {self.context_key} must be present in adata.obsm.")

        d_emb = adata.obsm[self.data_key]
        c_emb = adata.obsm[self.context_key]

        # Align data embeddings
        if d_emb.shape[1] > self.latent_dim:
            d_emb_aligned = self.reduce(d_emb)
        elif d_emb.shape[1] < self.latent_dim:
            d_emb_aligned = self.extend(d_emb)
        else:
            d_emb_aligned = d_emb  # Already at target dimension

        # Align context embeddings
        if c_emb.shape[1] > self.latent_dim:
            c_emb_aligned = self.reduce(c_emb)
        elif c_emb.shape[1] < self.latent_dim:
            c_emb_aligned = self.extend(c_emb)
        else:
            c_emb_aligned = c_emb  # Already at target dimension

        # Store aligned embeddings
        adata.obsm["d_emb_aligned"] = d_emb_aligned
        adata.obsm["c_emb_aligned"] = c_emb_aligned

    @abstractmethod
    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduces the dimensions of the embeddings to the target dimension.

        Parameters
        ----------
        embeddings
            The embeddings to reduce.

        Returns
        -------
        The dimensionally reduced embeddings.
        """
        pass

    def extend(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Extends the dimensions of the embeddings to the target dimension by zero-padding.

        Parameters
        ----------
        embeddings
            The embeddings to extend.

        Returns
        -------
        The dimensionally extended embeddings.
        """
        current_dim = embeddings.shape[1]
        if current_dim >= self.latent_dim:
            return embeddings

        padding = np.zeros((embeddings.shape[0], self.latent_dim - current_dim))
        extended_embeddings = np.hstack((embeddings, padding))
        return extended_embeddings


class PCAReducer(DimAligner):
    """
    Aligns dimensions of embeddings using PCA for dimensionality reduction.

    This class reduces the dimensions of embeddings to the target latent dimension using PCA.

    Parameters
    ----------
    latent_dim
        Target dimension for alignment.
    context_key
        Key for context embeddings in adata.obsm.
    data_key
        Key for data embeddings in adata.obsm.
    max_samples
        Maximum number of samples to use for fitting PCA.
    random_state
        Random seed for reproducibility.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *args,
        max_samples: int = 10000,
        random_state: int = None,
        config: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initializes the PCAReducer with the target latent dimension and parameters.

        Parameters
        ----------
        max_samples
            Maximum number of samples to use for fitting PCA.
        random_state
            Random seed for reproducibility.
        logger
            An optional logger object to use for logging.
        *args
            Positional arguments for the base class.
        **kwargs
            Keyword arguments for the base class (e.g., latent_dim, context_key, data_key).
        """
        super().__init__(logger, *args, **kwargs)
        self.max_samples = max_samples
        self.random_state = random_state
        self.config = config or {}

    def reduce(self, embeddings: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
        """
        Reduces the dimensions of the embeddings to the target dimension using PCA.

        Optionally evaluates the PCA model based on the provided configuration.

        Parameters
        ----------
        embeddings : np.ndarray
            The embeddings to reduce.
        config : dict, optional
            Configuration dictionary for PCA evaluation and plotting options.

        Returns
        -------
        np.ndarray
            The dimensionally reduced embeddings.
        """
        config = self.config

        n_samples = embeddings.shape[0]
        if n_samples > self.max_samples:
            # Randomly sample max_samples indices
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(n_samples, size=self.max_samples, replace=False)
            sample_embeddings = embeddings[indices]
        else:
            sample_embeddings = embeddings

        # Fit PCA model
        self.pca = PCA(n_components=self.latent_dim, random_state=self.random_state)
        self.pca.fit(sample_embeddings)

        # Transform the entire embeddings
        reduced_embeddings = self.pca.transform(embeddings)

        # Evaluate PCA if requested
        if config.get("evaluate_pca", False):
            self.evaluate_pca(embeddings=sample_embeddings, config=config)

        return reduced_embeddings

    def evaluate_pca(self, embeddings: np.ndarray, config: dict[str, Any]):
        """
        Evaluates the PCA model by computing and logging various metrics and generating plots.

        Parameters
        ----------
        embeddings : np.ndarray
            The embeddings used to fit the PCA model.
        config : dict
            Configuration dictionary with evaluation options.
        """
        save_path = config.get("save_path", None)
        if save_path:
            os.makedirs(save_path, exist_ok=True)

        # Compute and log Kaiser criterion
        # self.compute_kaiser_criterion()

        # Compute and log KMO measure
        # self.compute_kmo_measure(embeddings)

        # Generate scree plot
        if config.get("scree_plot", False):
            self.plot_scree(save_path)

        # Generate cumulative explained variance plot
        if config.get("cumulative_variance_plot", False):
            self.plot_cumulative_variance(save_path)

        # Generate heatmap of principal component loadings
        if config.get("loadings_heatmap", False):
            self.plot_loadings_heatmap(save_path, config.get("loadings_heatmap_options", {}))

    def plot_scree(self, save_path: str | None):
        """
        Generates and saves a scree plot.

        Parameters
        ----------
        save_path : str, optional
            Path to save the scree plot.
        """
        plt.figure(figsize=(8, 5))
        components = np.arange(1, len(self.pca.explained_variance_ratio_) + 1)
        plt.plot(components, self.pca.explained_variance_ratio_, "o-", linewidth=2)
        plt.title("Scree Plot")
        plt.xlabel("Principal Component")
        plt.ylabel("Variance Explained")
        plt.grid(True)
        if save_path:
            plt.savefig(os.path.join(save_path, "scree_plot.png"))
        plt.close()

    def plot_cumulative_variance(self, save_path: str | None):
        """
        Generates and saves a cumulative explained variance plot.

        Parameters
        ----------
        save_path : str, optional
            Path to save the cumulative variance plot.
        """
        plt.figure(figsize=(8, 5))
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        components = np.arange(1, len(cumulative_variance) + 1)
        plt.plot(components, cumulative_variance, "o-", linewidth=2)
        plt.title("Cumulative Explained Variance")
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Variance Explained")
        plt.grid(True)
        if save_path:
            plt.savefig(os.path.join(save_path, "cumulative_variance_plot.png"))
        plt.close()

    def plot_loadings_heatmap(self, save_path: str | None, options: dict[str, Any]):
        """
        Generates and saves a heatmap of principal component loadings.

        Parameters
        ----------
        save_path : str, optional
            Path to save the heatmap.
        options : dict
            Options for plotting, such as threshold and number of components/variables to display.
        """
        # Extract options
        threshold = options.get("threshold", None)
        top_n_components = options.get("top_n_components", 5)
        top_n_variables = options.get("top_n_variables", None)

        # Get loadings
        loadings = self.pca.components_.T  # Shape: (n_features, n_components)
        n_components = min(top_n_components, loadings.shape[1])
        loadings = loadings[:, :n_components]

        # Create DataFrame for easier handling
        import pandas as pd

        loadings_df = pd.DataFrame(loadings, columns=[f"PC{i+1}" for i in range(n_components)])

        # Apply threshold if provided
        if threshold is not None:
            loadings_df = loadings_df[(loadings_df.abs() > threshold).any(axis=1)]

        # Select top N variables if specified
        if top_n_variables is not None:
            # Compute the sum of squared loadings across selected components
            importance = (loadings_df**2).sum(axis=1)
            loadings_df = loadings_df.loc[importance.nlargest(top_n_variables).index]

        plt.figure(figsize=(10, 8))
        sns.heatmap(loadings_df, cmap="coolwarm", center=0, annot=False)
        plt.title("Principal Component Loadings Heatmap")
        plt.xlabel("Principal Components")
        plt.ylabel("Variables")
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "loadings_heatmap.png"))
        plt.close()


'''
    def compute_kaiser_criterion(self):
        """
        Computes and logs the Kaiser criterion.
        """
        eigenvalues = self.pca.explained_variance_
        num_components_over_one = np.sum(eigenvalues > 1)
        self.logger.info(f"Kaiser Criterion: Number of components with eigenvalue > 1: {num_components_over_one}")

    def compute_kmo_measure(self, data: np.ndarray):
        """
        Computes and logs the Kaiser-Meyer-Olkin (KMO) measure.

        Parameters
        ----------
        data : np.ndarray
            The data matrix used for PCA fitting.
        """
        try:
            kmo_all, kmo_j = self.calculate_kmo(data)
            self.logger.info(f"Kaiser-Meyer-Olkin (KMO) Measure: {kmo_all:.4f}")
        except Exception as e:
            self.logger.error(f"Error computing KMO measure: {e}")

    @staticmethod
    def calculate_kmo(data: np.ndarray):
        """
        Calculates the Kaiser-Meyer-Olkin (KMO) measure.

        Parameters
        ----------
        data : np.ndarray
            The data matrix.

        Returns
        -------
        kmo_all : float
            The overall KMO measure.
        kmo_j : np.ndarray
            The KMO measure for each variable.
        """
        # Standardize data
        scaler = StandardScaler()
        data_std = scaler.fit_transform(data)

        corr_matrix = np.corrcoef(data_std.T)
        inv_corr_matrix = np.linalg.pinv(corr_matrix)
        partial_corr_matrix = -inv_corr_matrix / np.sqrt(np.outer(np.diag(inv_corr_matrix), np.diag(inv_corr_matrix)))
        np.fill_diagonal(partial_corr_matrix, 0)

        partial_corr_squared = partial_corr_matrix ** 2
        corr_squared = corr_matrix ** 2

        partial_corr_sum = np.sum(partial_corr_squared, axis=0)
        corr_sum = np.sum(corr_squared, axis=0) - 1  # Exclude diagonal elements

        kmo_j = corr_sum / (corr_sum + partial_corr_sum)
        kmo_all = np.sum(corr_sum) / (np.sum(corr_sum) + np.sum(partial_corr_sum))

        return kmo_all, kmo_j
'''
