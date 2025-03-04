# tests/utils.py
import json
import logging
import os
import random
import tempfile
from datetime import datetime
from pathlib import Path

import anndata
import h5py
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from sentence_transformers import evaluation, losses

logger = logging.getLogger(__name__)


def load_test_adata_from_hf_dataset(
    test_dataset,
    sample_size=10,
):
    """
    Load an AnnData object from a Hugging Face test dataset that contains a share link to an external `.h5ad` file.

    This function downloads the file to a temporary directory
    and reads it into memory.

    Parameters
    ----------
    test_dataset : dict
        A dictionary-like object representing the HF dataset split (e.g., `test_dataset["train"]`).
        It must contain an 'anndata_ref' field, where each element is a JSON string with a "file_path" key.
    sample_size : int, optional
        Number of random rows to check for file path consistency before download, by default 10.

    Returns
    -------
    anndata.AnnData
        The AnnData object read from the downloaded `.h5ad` file.

    Notes
    -----
    - Data is assumed to come from a Hugging Face dataset with a single unique `file_path` for all rows.
    - The function downloads the file to a temporary directory, which is removed when this function returns.
    - If multiple rows have different `file_path` values, the function raises an error.
    """
    # If the dataset split is large, reduce the sample size to the dataset size
    size_of_dataset = len(test_dataset)
    sample_size = min(sample_size, size_of_dataset)

    # Randomly sample rows to ensure all file paths match
    indices_to_check = random.sample(range(size_of_dataset), sample_size)
    paths = []
    for idx in indices_to_check:
        adata_ref = test_dataset[idx]["anndata_ref"]
        paths.append(adata_ref["file_record"]["dataset_path"])

    # Ensure that all random rows have the same file path
    first_path = paths[0]
    for p in paths[1:]:
        if p != first_path:
            raise ValueError("Not all sampled rows contain the same file path. Please verify the dataset consistency.")

    # Download the file from the share link into a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.h5ad"
        download_file_from_share_link(first_path, str(save_path))
        adata = anndata.read_h5ad(save_path)

    return adata


def download_file_from_share_link(share_link, save_path, chunk_size=8192):
    """
    Downloads a file from a Nextcloud share link and validates it based on its suffix.

    Parameters
    ----------
    share_link : str
        The full share link URL to the file.
    save_path : str
        The local path where the file should be saved.
    chunk_size : int, optional
        Size of each chunk in bytes during streaming; defaults to 8192.

    Returns
    -------
    bool
        True if the download was successful and the file is valid based on its suffix;
        False otherwise.

    References
    ----------
    Data is expected to come from a Nextcloud share link and is validated in memory.
    """
    # Step 1: Stream download the file
    try:
        with requests.get(share_link, stream=True) as response:
            response.raise_for_status()

            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    file.write(chunk)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download the file from '{share_link}': {e}")
        return False

    # Step 2: Validate based on suffix
    file_suffix = os.path.splitext(save_path)[1].lower()

    try:
        if file_suffix == ".h5ad":
            # Validate as an anndata-compatible HDF5 file
            with h5py.File(save_path, "r") as h5_file:
                required_keys = ["X", "obs", "var"]  # Common in .h5ad
                if all(key in h5_file for key in required_keys):
                    logger.info("File is a valid .h5ad file.")
                    return True
                else:
                    logger.warning("File is an HDF5 file but missing required .h5ad keys.")
                    return False

        elif file_suffix == ".npz":
            # Validate as a .npz file (we can at least confirm we can load it)
            try:
                np.load(save_path, allow_pickle=True)
                logger.info("File is a valid .npz file.")
                return True
            except Exception as e:
                logger.error(f"Error while validating the downloaded file: {e}")
                return False

        elif file_suffix == ".npy":
            # Validate as a .npy file
            try:
                np.load(save_path, allow_pickle=True)
                logger.info("File is a valid .npy file.")
                return True
            except Exception as e:
                logger.error(f"Error while validating the downloaded file: {e}")
                return False
        else:
            # If your use-case requires more file types, add them here
            logger.warning(f"No specific validation logic for files of type '{file_suffix}'. Skipping validation.")
            return True

    except Exception as e:
        logger.error(f"Error while validating the downloaded file: {e}")
        return False


def setup_logging(logging_dir="logs"):
    """Set up logging configuration for the module.

    This function configures the root logger to display messages in the console and to write them to a file
    named by the day. The log level is set to INFO.
    """
    # Create the logs directory
    os.makedirs(logging_dir, exist_ok=True)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers (important if function is called multiple times)
    if not logger.hasHandlers():
        # Create a file handler
        log_file = logging.FileHandler(f"logs/{datetime.now().strftime('%Y-%m-%d')}.log")
        log_file.setLevel(logging.INFO)

        # Create a console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # Create a formatter and set it for the handlers
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        log_file.setFormatter(formatter)
        console.setFormatter(formatter)

        # Add the handlers to the root logger
        logger.addHandler(log_file)
        logger.addHandler(console)

    return logger


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


def get_loss(dataset_type: str, loss_name: str = None):
    """
    Return a suitable loss object based on the provided loss name and dataset type.

    Parameters
    ----------
    loss_name : str | None
        A string identifying the loss function to be used.
    model : MMContextEncoder
        The multimodal context encoder model.
    dataset_type : str
        The type of the dataset (e.g., 'pairs', 'triplets').

    Returns
    -------
    losses.Loss
        An instance of a SentenceTransformer loss object.

    Notes
    -----
    This is a stub. You will add compatibility checks later (e.g., if dataset_type
    is 'pairs', maybe only certain losses are supported).
    """
    pairs_losses = ["ContrastiveLoss", "OnlineContrastiveLoss"]
    pairs_losses_default = "ContrastiveLoss"
    multiplets_losses = ["MultipleNegativesRankingLoss", "CachedMultipleNegativesRankingLoss", "CachedGISTEmbedLoss"]
    multiplets_losses_default = "MultipleNegativesRankingLoss"
    if dataset_type == "pairs" and loss_name is None:
        loss_name = pairs_losses_default
    elif dataset_type == "pairs" and loss_name not in pairs_losses:
        raise ValueError(f"Loss '{loss_name}' is not supported for pairs dataset. Choose from {pairs_losses}")
    if dataset_type == "multiplets":
        if loss_name is None:
            loss_name = multiplets_losses_default
        elif loss_name not in multiplets_losses:
            raise ValueError(
                f"Loss '{loss_name}' is not supported for multiplets dataset. Choose from {multiplets_losses}"
            )

    # Dynamically fetch the loss class from the sentence_transformers.losses module
    try:
        LossClass = getattr(losses, loss_name)
    except AttributeError as e:
        raise f"Loss class '{loss_name}' not found in sentence_transformers.losses" from e
    # Instantiate the loss class with the given model
    return LossClass


def get_evaluator(dataset_type: str, dataset, evaluator_name: str | None = None, batch_size: int = 32):
    """
    Return a suitable evaluator object

    It is obtained based on the provided evaluator name, dataset type, and dataset. Has to be one of the evalutors supported
    from the sentence_transformers.evaluation module. Also has to be explicitly defined for the dataset type in this function.

    Parameters
    ----------
    dataset_type : str
        The type of the dataset (e.g., 'pairs', 'multiplets').
    dataset: dict
        The dataset dictionary containing the necessary data for the evaluator.
    evaluator_name : str
        A string identifying the evaluator to be used. If none is given a default will be used depending on the dataset type.

    Returns
    -------
    Evaluator
        An instance of a SentenceTransformer evaluator object.

    Raises
    ------
    ValueError
        If the provided evaluator name is not supported for the given dataset type
        or if the required data keys are missing in the dataset.
    AttributeError
        If the evaluator class is not found in the sentence_transformers.evaluation module.
    """
    pairs_evaluators = ["BinaryClassificationEvaluator"]
    pairs_evaluator_default = "BinaryClassificationEvaluator"
    multiplets_evaluators = ["TripletEvaluator"]
    multiplets_evaluator_default = "TripletEvaluator"

    if dataset_type == "pairs":
        if evaluator_name is None:
            evaluator_name = pairs_evaluator_default
        if evaluator_name not in pairs_evaluators:
            raise ValueError(
                f"Evaluator '{evaluator_name}' is not supported for pairs dataset. Choose from {pairs_evaluators}"
            )

        required_keys = {"anndata_ref", "caption", "label"}
        if not required_keys.issubset(dataset.column_names):
            raise ValueError(f"Dataset for 'pairs' evaluator must contain keys: {required_keys}")
        try:
            EvaluatorClass = getattr(evaluation, evaluator_name)
            evaluator_obj = EvaluatorClass(
                sentences1=dataset["anndata_ref"],
                sentences2=dataset["caption"],
                labels=dataset["label"],
                batch_size=batch_size,
            )
        except AttributeError as e:
            raise f"Evaluator class '{evaluator_name}' not found in sentence_transformers.evaluation" from e
        except TypeError as e:  # Catch potential errors due to missing or incorrect arguments
            raise ValueError(f"Error instantiating {evaluator_name}: {e}") from e

    elif dataset_type == "multiplets":
        if evaluator_name is None:
            evaluator_name = multiplets_evaluator_default
        if evaluator_name not in multiplets_evaluators:
            raise ValueError(
                f"Evaluator '{evaluator_name}' is not supported for multiplets dataset. "
                f"Choose from {multiplets_evaluators}"
            )

        required_keys = {"anndata_ref", "positive", "negative_1"}  # Updated to match TripletEvaluator
        if not required_keys.issubset(dataset.column_names):
            raise ValueError(f"Dataset for 'multiplets' evaluator must contain keys: {required_keys}")
        try:
            EvaluatorClass = getattr(evaluation, evaluator_name)

            evaluator_obj = EvaluatorClass(
                anchors=dataset["anndata_ref"],
                positives=dataset["positive"],
                negatives=dataset["negative_1"],
            )
        except AttributeError as e:
            raise f"Evaluator class '{evaluator_name}' not found in sentence_transformers.evaluation" from e
        except TypeError as e:  # Catch potential errors due to missing or incorrect arguments
            raise ValueError(f"Error instantiating {evaluator_name}: {e}") from e

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return evaluator_obj


def get_device():
    """Helper function to get the available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
