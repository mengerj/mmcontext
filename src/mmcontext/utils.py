# tests/utils.py
import importlib
import json
import logging

import anndata
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def load_logging_config():
    """
    Load the logging configuration from `logging_config.json` in the `mmcontext.conf` module.

    Returns
    -------
    dict
        The parsed logging configuration as a dictionary.
    """
    try:
        # Get the path to the logging_config.json file
        resource_path = importlib.resources.files("mmcontext.conf") / "logging_config.json"

        # Open the resource file
        with resource_path.open("r", encoding="utf-8") as config_file:
            logging_config = json.load(config_file)

        return logging_config
    except FileNotFoundError as err:
        raise RuntimeError("The logging configuration file could not be found.") from err
    except json.JSONDecodeError as err:
        raise ValueError(f"Error decoding the JSON logging configuration: {err}") from err


def setup_logging():
    """Load the logging configuration from the logging_config.json file and configure the logging system."""
    config_dict = load_logging_config()
    # Configure logging
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(__name__)
    logger.info("mmcontext logging configured using the specified configuration file.")


def sample_zinb(mu, theta, pi):
    """
    Samples from the Zero-Inflated Negative Binomial distribution.

    Parameters
    ----------
    - mu (torch.Tensor): Mean of the NB distribution (batch_size, num_genes)
    - theta (torch.Tensor): Dispersion parameter of the NB distribution (batch_size, num_genes)
    - pi (torch.Tensor): Zero-inflation probability (batch_size, num_genes)

    Returns
    -------
    - samples (torch.Tensor): Sampled counts (batch_size, num_genes)
    """
    # Ensure parameters are on the same device and have the same shape
    assert mu.shape == theta.shape == pi.shape

    # Sample zero-inflation indicator z
    bernoulli_dist = torch.distributions.Bernoulli(probs=pi)
    z = bernoulli_dist.sample()

    # Sample from Negative Binomial distribution
    # Compute probability p from mu and theta
    p = theta / (theta + mu)
    # Convert to total_count (r) and probability (1 - p)
    nb_dist = torch.distributions.NegativeBinomial(total_count=theta, probs=1 - p)
    nb_sample = nb_dist.sample()

    # Combine zero-inflation and NB samples
    samples = z * 0 + (1 - z) * nb_sample

    return samples


def compute_cosine_similarity(sample_embeddings, query_embeddings, device="cpu"):
    """
    Compute pairwise cosine similarity between samples and queries using PyTorch.

    Parameters
    ----------
    sample_embeddings : np.ndarray
        2D array of shape (num_samples, embedding_dim) containing
        the sample (omics) embeddings. Source: adata.obsm["omics_emb"].
    query_embeddings : np.ndarray
        2D array of shape (num_queries, embedding_dim) containing
        the query embeddings. Typically from model.encode(...).
    device : str, optional
        The device on which to run the computation. One of ["cpu", "cuda","mps"].

    Returns
    -------
    np.ndarray
        A matrix of shape (num_queries, num_samples), containing
        the cosine similarity scores for each query against each sample.
    """
    logger.info("Converting numpy arrays to torch Tensors.")
    # Convert to torch Tensors
    sample_t = torch.from_numpy(sample_embeddings).float().to(device)
    query_t = torch.from_numpy(query_embeddings).float().to(device)

    logger.info("L2-normalizing embeddings for cosine similarity.")
    # L2 normalize if we want to treat dot product as cosine
    sample_t = sample_t / (sample_t.norm(dim=1, keepdim=True) + 1e-9)
    query_t = query_t / (query_t.norm(dim=1, keepdim=True) + 1e-9)

    logger.info("Performing matrix multiplication on device=%s", device)
    # matrix shape: (num_queries, embedding_dim) x (embedding_dim, num_samples)
    # result -> (num_queries, num_samples)
    sim_t = query_t.mm(sample_t.transpose(0, 1))

    # Move back to CPU, convert to numpy
    sim = sim_t.cpu().numpy()
    return sim


def create_test_anndata(n_samples=20, n_features=100, cell_types=None, tissues=None, batch_categories=None):
    """
    Create a test AnnData object with synthetic data, including batch information.

    Parameters
    ----------
    n_samples : int
        Number of cells (observations). Default is 20.
    n_features : int
        Number of genes (variables). Default is 100.
    cell_types : list, optional
        List of cell types. Defaults to ["B cell", "T cell", "NK cell"].
    tissues : list, optional
        List of tissues. Defaults to ["blood", "lymph"].
    batch_categories : list, optional
        List of batch categories. Defaults to ["Batch1", "Batch2"].

    Returns
    -------
    anndata.AnnData
        Generated AnnData object.
    """
    import numpy as np

    # Set default values for mutable arguments if they are None
    if cell_types is None:
        cell_types = ["B cell", "T cell", "NK cell"]
    if tissues is None:
        tissues = ["blood", "lymph"]
    if batch_categories is None:
        batch_categories = ["Batch1", "Batch2"]

    # Set a random seed for reproducibility
    np.random.seed(42)

    # Determine the number of batches and allocate samples to batches
    n_batches = len(batch_categories)
    samples_per_batch = n_samples // n_batches
    remainder = n_samples % n_batches

    batch_labels = []
    for i, batch in enumerate(batch_categories):
        n = samples_per_batch + (1 if i < remainder else 0)
        batch_labels.extend([batch] * n)

    # Shuffle batch labels
    np.random.shuffle(batch_labels)

    # Generate observation (cell) metadata

    obs = pd.DataFrame(
        {
            "cell_type": np.random.choice(cell_types, n_samples),
            "tissue": np.random.choice(tissues, n_samples),
            "batch": batch_labels,
        }
    )
    obs.index = [f"Cell_{i}" for i in range(n_samples)]
    # transform obs to categorical
    obs = obs.astype("category")
    obs["sample_id"] = np.arange(n_samples)
    # Generate a random data matrix (e.g., gene expression values)
    X = np.zeros((n_samples, n_features))
    for i, batch in enumerate(batch_categories):
        # Get indices of cells in this batch
        idx = obs[obs["batch"] == batch].index
        idx = [obs.index.get_loc(i) for i in idx]

        # Generate data for this batch
        # For simplicity, let's make a mean shift between batches
        mean = np.random.rand(n_features) * (i + 1)  # Different mean for each batch
        X[idx, :] = np.random.normal(loc=mean, scale=1.0, size=(len(idx), n_features))

    # Create variable (gene) metadata
    var = pd.DataFrame({"gene_symbols": [f"Gene_{i}" for i in range(n_features)]})
    var.index = [f"Gene_{i}" for i in range(n_features)]

    # Create the AnnData object
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    return adata


def create_test_emb_anndata(n_samples, emb_dim, data_key="d_emb_aligned", context_key="c_emb_aligned", sample_ids=None):
    """
    Helper function to create a test AnnData object with specified embeddings and sample IDs.

    Args:
        n_samples (int): Number of samples (cells).
        emb_dim (int): Embedding dimension.
        data_key (str): Key for data embeddings in adata.obsm.
        context_key (str): Key for context embeddings in adata.obsm.
        sample_ids (list): List of sample IDs. If None, default IDs are assigned.

    Returns
    -------
        AnnData: The constructed AnnData object.
    """
    adata = create_test_anndata(n_samples=n_samples)
    adata.obsm[data_key] = np.random.rand(n_samples, emb_dim)
    adata.obsm[context_key] = np.random.rand(n_samples, emb_dim)
    if sample_ids is not None:
        adata.obs["sample_id"] = sample_ids
    return adata
