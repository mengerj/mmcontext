import hashlib
import json
import logging
import os
import tempfile

import numpy as np
import scipy.sparse as sp
import torch
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from mmcontext.utils import download_file_from_share_link

logger = logging.getLogger(__name__)


def create_embedding_store(
    dataset: Dataset | DatasetDict,
    embedding_key: str,
    output_dir: str | None = None,
    batch_size: int = 1000,
    column_name: str = "anndata_ref",
    dtype: str = "float32",
    force_recreate: bool = False,
) -> tuple[str, dict[str, int]]:
    """
    Create an embedding store from a dataset.

    Parameters
    ----------
    dataset : Union[Dataset, DatasetDict]
        The input dataset containing anndata references
    embedding_key : str
        The key to use for embeddings (e.g., 'X_hvg', 'X_pca')
    output_dir : Optional[str], default=None
        Directory to store the embeddings. If None, uses a temp directory
    batch_size : int, default=1000
        Batch size for processing large datasets
    column_name : str, default="anndata_ref"
        Column name in the dataset containing the anndata references
    dtype : str, default="float32"
        Data type for storing embeddings. Options: "float32", "float16", "bfloat16"
    force_recreate : bool, default=False
        If True, force recreation of the store even if it already exists

    Returns
    -------
    Tuple[str, Dict[str, int]]
        The path to the embedding store and a mapping from sample_id to index
    """
    # Validate dtype
    valid_dtypes = ["float32", "float16", "bfloat16"]
    if dtype not in valid_dtypes:
        raise ValueError(f"Invalid dtype: {dtype}. Must be one of {valid_dtypes}")

    # Create output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(
            tempfile.gettempdir(), f"mmcontext_data_{hashlib.md5(embedding_key.encode()).hexdigest()}"
        )
    os.makedirs(output_dir, exist_ok=True)

    # Path for the embedding store
    store_path = os.path.join(output_dir, f"{embedding_key}_store_{dtype}.npz")
    index_path = os.path.join(output_dir, f"{embedding_key}_index_{dtype}.json")

    # Collect all unique embedding paths and sample IDs from all splits
    all_embedding_paths = set()
    all_sample_ids = []
    all_examples = []

    # Create a dictionary to track which samples belong to which split
    if isinstance(dataset, DatasetDict):
        split_samples = {split_name: [] for split_name in dataset.keys()}

        for split_name, split_dataset in dataset.items():
            for example in tqdm(split_dataset, desc=f"Collecting data from {split_name} split"):
                anndata_ref = example[column_name]
                sample_id = anndata_ref["sample_id"]

                all_embedding_paths.add(anndata_ref["file_record"]["embeddings"][embedding_key])
                all_sample_ids.append(sample_id)
                all_examples.append((split_name, example))

                # Track which split this sample belongs to
                split_samples[split_name].append(sample_id)

        # Log the number of samples in each split for debugging
        for split_name, samples in split_samples.items():
            logger.info(f"Split {split_name}: {len(samples)} samples")
    else:
        for example in tqdm(dataset, desc="Collecting embedding paths"):
            anndata_ref = example[column_name]
            sample_id = anndata_ref["sample_id"]

            all_embedding_paths.add(anndata_ref["file_record"]["embeddings"][embedding_key])
            all_sample_ids.append(sample_id)
            all_examples.append(("single", example))

    # Check if the store already exists
    if os.path.exists(store_path) and os.path.exists(index_path) and not force_recreate:
        logger.info(f"Loading existing embedding store from {store_path}")
        with open(index_path) as f:
            sample_id_to_idx = json.load(f)

        # Verify that all samples are in the index
        missing_samples = []
        for sample_id in all_sample_ids:
            if sample_id not in sample_id_to_idx:
                missing_samples.append(sample_id)

        if missing_samples:
            logger.warning(f"Found {len(missing_samples)} samples missing from the index. Recreating the store.")
            # Force recreation of the store
            os.remove(store_path)
            os.remove(index_path)

            # Create a new store
            logger.info(f"Creating new embedding store at {store_path}")
            sample_id_to_idx, embedding_matrix = _create_combined_embedding_store(
                all_embedding_paths, all_sample_ids, all_examples, embedding_key, column_name, dtype
            )

            # Save the embedding matrix
            logger.info(f"Saving embedding store to {store_path}")
            np.savez_compressed(
                store_path,
                data=embedding_matrix,
                sample_ids=np.array(list(sample_id_to_idx.keys())),
                dtype=dtype,  # Store dtype information
            )

            # Save the index
            with open(index_path, "w") as f:
                json.dump(sample_id_to_idx, f)
        else:
            logger.info("All samples are in the index.")
    else:
        # Create a new store
        logger.info(f"Creating new embedding store at {store_path}")
        sample_id_to_idx, embedding_matrix = _create_combined_embedding_store(
            all_embedding_paths, all_sample_ids, all_examples, embedding_key, column_name, dtype
        )

        # Save the embedding matrix
        logger.info(f"Saving embedding store to {store_path}")
        np.savez_compressed(
            store_path,
            data=embedding_matrix,
            sample_ids=np.array(list(sample_id_to_idx.keys())),
            dtype=dtype,  # Store dtype information
        )

        # Save the index
        with open(index_path, "w") as f:
            json.dump(sample_id_to_idx, f)

    return store_path, sample_id_to_idx


def transform_dataset(
    dataset: Dataset | DatasetDict,
    store_path: str,
    sample_id_to_idx: dict[str, int],
    dataset_type: str,
    column_name: str = "anndata_ref",
    dtype: str = "float32",
    in_memory: bool = True,
) -> Dataset | DatasetDict:
    """
    Transform a dataset to use sample indices instead of anndata references.

    Parameters
    ----------
    dataset : Union[Dataset, DatasetDict]
        The input dataset containing anndata references
    store_path : str
        Path to the embedding store
    sample_id_to_idx : Dict[str, int]
        Mapping from sample_id to index
    dataset_type : str
        Type of the dataset: "pairs", "multiplets", etc.
    column_name : str, default="anndata_ref"
        Column name in the dataset containing the anndata references
    dtype : str, default="float32"
        Data type for storing embeddings
    in_memory : bool, default=True
        Whether to load embeddings into memory

    Returns
    -------
    Union[Dataset, DatasetDict]
        The transformed dataset
    """
    if isinstance(dataset, DatasetDict):
        processed_dataset = DatasetDict()
        for split_name, split_dataset in dataset.items():
            processed_dataset[split_name] = _transform_dataset_by_type(
                split_dataset, store_path, sample_id_to_idx, dataset_type, column_name, dtype
            )

            # Add store_path as a dataset attribute
            processed_dataset[split_name].store_path = store_path
    else:
        processed_dataset = _transform_dataset_by_type(
            dataset, store_path, sample_id_to_idx, dataset_type, column_name, dtype
        )

        # Add store_path as a dataset attribute
        processed_dataset.store_path = store_path

    # If in_memory is True, load the embeddings into memory
    if in_memory:
        logger.info("Loading embeddings into memory")
        _load_embeddings_into_memory(store_path, dtype)

    return processed_dataset


def _transform_dataset_by_type(
    dataset: Dataset, store_path: str, sample_id_to_idx: dict[str, int], dataset_type: str, column_name: str, dtype: str
) -> Dataset:
    """
    Transform a dataset based on its type.

    Parameters
    ----------
    dataset : Dataset
        The input dataset
    store_path : str
        Path to the embedding store
    sample_id_to_idx : Dict[str, int]
        Mapping from sample_id to index
    dataset_type : str
        Type of the dataset: "pairs", "multiplets", etc.
    column_name : str
        Column name containing the anndata references
    dtype : str
        Data type for storing embeddings

    Returns
    -------
    Dataset
        The transformed dataset
    """
    if dataset_type == "pairs":
        return _transform_pairs_dataset(dataset, store_path, sample_id_to_idx, column_name, dtype)
    elif dataset_type == "multiplets":
        return _transform_multiplets_dataset(dataset, store_path, sample_id_to_idx, column_name, dtype)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def _transform_pairs_dataset(
    dataset: Dataset, store_path: str, sample_id_to_idx: dict[str, int], column_name: str, dtype: str
) -> Dataset:
    """
    Transform a pairs dataset.

    Parameters
    ----------
    dataset : Dataset
        The input dataset
    store_path : str
        Path to the embedding store
    sample_id_to_idx : Dict[str, int]
        Mapping from sample_id to index
    column_name : str
        Column name containing the anndata references
    dtype : str
        Data type for storing embeddings

    Returns
    -------
    Dataset
        The transformed dataset with only essential columns:
        - sample_idx: Index in the embedding store
        - caption: Text caption
        - label: Classification label (if exists)
    """

    def transform_example(example):
        sample_id = example[column_name]["sample_id"]

        # Create a new example with only the essential columns
        new_example = {
            "sample_idx": sample_id_to_idx[sample_id],
            "caption": example.get("text", ""),
        }

        # Add label if it exists
        if "label" in example:
            new_example["label"] = example["label"]

        return new_example

    return dataset.map(
        transform_example,
        desc="Transforming pairs dataset",
        remove_columns=dataset.column_names,  # Remove all original columns
    )


def _transform_multiplets_dataset(
    dataset: Dataset, store_path: str, sample_id_to_idx: dict[str, int], column_name: str, dtype: str
) -> Dataset:
    """
    Transform a multiplets dataset.

    Parameters
    ----------
    dataset : Dataset
        The input dataset
    store_path : str
        Path to the embedding store
    sample_id_to_idx : Dict[str, int]
        Mapping from sample_id to index
    column_name : str
        Column name containing the anndata references
    dtype : str
        Data type for storing embeddings

    Returns
    -------
    Dataset
        The transformed dataset with only essential columns:
        - sample_idx: Index in the embedding store
        - caption: Text caption
        - positives: List of positive sample indices
        - negatives: List of negative sample indices
    """

    def transform_example(example):
        # Transform the anchor
        sample_id = example[column_name]["sample_id"]

        # Transform positives if they exist
        positives = example.get("positives", [])
        positive_indices = []
        if positives:
            for pos in positives:
                if isinstance(pos, dict) and "sample_id" in pos:
                    positive_indices.append(sample_id_to_idx[pos["sample_id"]])
                else:
                    # If positives are not anndata_ref, keep them as is
                    positive_indices.append(pos)

        # Transform negatives if they exist
        negatives = example.get("negatives", [])
        negative_indices = []
        if negatives:
            for neg in negatives:
                if isinstance(neg, dict) and "sample_id" in neg:
                    negative_indices.append(sample_id_to_idx[neg["sample_id"]])
                else:
                    # If negatives are not anndata_ref, keep them as is
                    negative_indices.append(neg)

        # Create a new example with only the essential columns
        new_example = {
            "sample_idx": sample_id_to_idx[sample_id],
            "caption": example.get("text", ""),
        }

        # Add positives and negatives if they exist
        if positive_indices:
            new_example["positives"] = positive_indices
        if negative_indices:
            new_example["negatives"] = negative_indices

        return new_example

    return dataset.map(
        transform_example,
        desc="Transforming multiplets dataset",
        remove_columns=dataset.column_names,  # Remove all original columns
    )


def prepare_efficient_dataset(
    dataset: Dataset | DatasetDict,
    embedding_key: str,
    dataset_type: str,
    output_dir: str | None = None,
    in_memory: bool = True,
    batch_size: int = 1000,
    column_name: str = "anndata_ref",
    dtype: str = "float32",
) -> Dataset | DatasetDict:
    """
    Prepares an efficient dataset by downloading embeddings once and creating a local store.

    Parameters
    ----------
    dataset : Union[Dataset, DatasetDict]
        The input dataset containing anndata references
    embedding_key : str
        The key to use for embeddings (e.g., 'X_hvg', 'X_pca')
    dataset_type : str
        Type of the dataset: "pairs", "multiplets", etc.
    output_dir : Optional[str], default=None
        Directory to store the embeddings. If None, uses a temp directory
    in_memory : bool, default=True
        Whether to keep embeddings in memory or on disk
    batch_size : int, default=1000
        Batch size for processing large datasets
    column_name : str, default="anndata_ref"
        Column name in the dataset containing the anndata references
    dtype : str, default="float32"
        Data type for storing embeddings. Options: "float32", "float16", "bfloat16"

    Returns
    -------
    Union[Dataset, DatasetDict]
        The transformed dataset with sample indices instead of anndata references
    """
    # First, create the embedding store
    store_path, sample_id_to_idx = create_embedding_store(
        dataset=dataset,
        embedding_key=embedding_key,
        output_dir=output_dir,
        batch_size=batch_size,
        column_name=column_name,
        dtype=dtype,
    )

    # Then, transform the dataset based on its type
    processed_dataset = transform_dataset(
        dataset=dataset,
        store_path=store_path,
        sample_id_to_idx=sample_id_to_idx,
        dataset_type=dataset_type,
        column_name=column_name,
        dtype=dtype,
        in_memory=in_memory,
    )

    return processed_dataset, store_path


def _create_combined_embedding_store(
    all_embedding_paths: set, all_sample_ids: list, all_examples: list, embedding_key: str, column_name: str, dtype: str
) -> tuple:
    """Create a combined embedding store from all splits."""
    # Create a mapping from sample_id to index
    sample_id_to_idx = {sample_id: idx for idx, sample_id in enumerate(all_sample_ids)}

    # Download and process each embedding file
    all_embeddings = {}
    all_file_sample_ids = {}

    for embedding_path in tqdm(all_embedding_paths, desc="Downloading embedding files"):
        # Resolve the file path (download if needed)
        local_path = _resolve_file_path(embedding_path)

        # Load the embedding data
        npzfile = np.load(local_path, allow_pickle=True)
        embeddings = npzfile["data"]
        file_sample_ids = npzfile["sample_ids"]

        # Store in dictionaries
        all_embeddings[embedding_path] = embeddings
        all_file_sample_ids[embedding_path] = file_sample_ids

    # Determine embedding dimension
    embedding_dim = next(iter(all_embeddings.values())).shape[1]

    # Create the embedding matrix with the specified dtype
    np_dtype = np.float32 if dtype == "float32" else np.float16
    embedding_matrix = np.zeros((len(all_sample_ids), embedding_dim), dtype=np_dtype)

    # Fill the embedding matrix
    for _i, (_split_name, example) in enumerate(tqdm(all_examples, desc="Creating embedding matrix")):
        anndata_ref = example[column_name]
        sample_id = anndata_ref["sample_id"]
        embedding_path = anndata_ref["file_record"]["embeddings"][embedding_key]

        # Find the index of this sample_id in the file
        file_sample_ids = all_file_sample_ids[embedding_path]
        file_idx = np.where(file_sample_ids == sample_id)[0][0]

        # Get the embedding and convert to the specified dtype
        embedding = all_embeddings[embedding_path][file_idx].astype(np_dtype)

        # Store in the matrix
        matrix_idx = sample_id_to_idx[sample_id]
        embedding_matrix[matrix_idx] = embedding

    return sample_id_to_idx, embedding_matrix


def _resolve_file_path(file_path: str, suffix: str = ".npz") -> str:
    """Resolve a file path, downloading if it's a URL."""
    if file_path.startswith("https"):
        # Generate a unique local filename based on the MD5 hash of the share link
        hash_object = hashlib.md5(file_path.encode())
        file_hash = hash_object.hexdigest()
        local_file = os.path.join(tempfile.gettempdir(), f"{file_hash}{suffix}")

        if not os.path.exists(local_file):
            logger.info(f"Downloading file from share link: {file_path} to {local_file}")
            success = download_file_from_share_link(file_path, local_file)
            if not success:
                raise ValueError(f"Failed to download file from share link: {file_path}")

        return local_file

    return file_path


# Global cache for in-memory embeddings
_EMBEDDING_CACHE = {}


def _load_embeddings_into_memory(store_path: str, dtype: str = "float32") -> None:
    """Load embeddings into memory cache with optimized lookup."""
    if store_path not in _EMBEDDING_CACHE:
        npzfile = np.load(store_path, allow_pickle=True)
        sample_ids = npzfile["sample_ids"]
        stored_dtype = npzfile.get("dtype", "float32")  # Get stored dtype or default to float32

        # Pre-compute the mapping from sample_id to index
        sample_id_to_idx = {sid: idx for idx, sid in enumerate(sample_ids)}

        # Convert to the appropriate torch dtype
        if dtype == "float16" or stored_dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16" or stored_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        # Convert to tensor with the specified dtype
        tensor_data = torch.from_numpy(npzfile["data"]).to(torch_dtype)

        _EMBEDDING_CACHE[store_path] = {
            "data": tensor_data,
            "sample_ids": sample_ids,
            "sample_id_to_idx": sample_id_to_idx,  # Store the mapping
            "dtype": torch_dtype,  # Store the torch dtype
        }


def get_embedding(store_path: str, sample_id: str) -> torch.Tensor:
    """
    Get an embedding for a sample ID from the store.

    Parameters
    ----------
    store_path : str
        Path to the embedding store
    sample_id : str
        Sample ID to retrieve

    Returns
    -------
    torch.Tensor
        The embedding tensor
    """
    # Check if embeddings are in memory
    if store_path in _EMBEDDING_CACHE:
        # Get from memory
        cache = _EMBEDDING_CACHE[store_path]
        sample_ids = cache["sample_ids"]
        data = cache["data"]

        # Find the index
        idx = np.where(sample_ids == sample_id)[0][0]

        # Return the embedding
        return data[idx]
    else:
        # Load from disk
        npzfile = np.load(store_path, allow_pickle=True)
        sample_ids = npzfile["sample_ids"]
        data = npzfile["data"]

        # Find the index
        idx = np.where(sample_ids == sample_id)[0][0]

        # Return the embedding
        return torch.from_numpy(data[idx]).float()


# Example usage:
# processed_dataset, store_path = prepare_efficient_dataset(
#     dataset=my_dataset,
#     embedding_key="X_hvg",
#     output_dir="/path/to/store",
#     in_memory=True
# )


class OptimizedProcessor:
    """
    Processor for efficiently retrieving embeddings from a pre-processed store.

    Parameters
    ----------
    store_path : str
        Path to the embedding store
    in_memory : bool, default=True
        Whether to keep embeddings in memory
    dtype : str, default="float32"
        Data type for storing embeddings. Options: "float32", "float16", "bfloat16"
    """

    def __init__(self, store_path: str, in_memory: bool = True, dtype: str = "float32", **kwargs):
        self.store_path = store_path
        self.in_memory = in_memory
        self.dtype = dtype

        # Validate dtype
        valid_dtypes = ["float32", "float16", "bfloat16"]
        if dtype not in valid_dtypes:
            raise ValueError(f"Invalid dtype: {dtype}. Must be one of {valid_dtypes}")

        if in_memory:
            # Load embeddings into memory
            _load_embeddings_into_memory(store_path, dtype)

    def get_rep(self, data: list[dict]) -> torch.Tensor:
        """
        Retrieve embeddings for a batch of samples using vectorized operations.

        Parameters
        ----------
        data : List[Dict]
            List of dictionaries with sample_id and sample_idx

        Returns
        -------
        torch.Tensor
            Batch of embeddings
        """
        # Extract all sample indices directly from the data
        # This is much faster than looking up by sample_id
        batch_indices = data

        # Check if embeddings are in memory
        if self.store_path in _EMBEDDING_CACHE:
            # Get from memory
            all_embeddings = _EMBEDDING_CACHE[self.store_path]["data"]

            # Get all embeddings at once using the indices
            features = all_embeddings[batch_indices]

        else:
            # Load from disk
            npzfile = np.load(self.store_path, allow_pickle=True)
            all_embeddings = npzfile["data"]

            # Convert to the appropriate torch dtype
            if self.dtype == "float16":
                torch_dtype = torch.float16
            elif self.dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

            # Get all embeddings at once using the indices and convert to the specified dtype
            features = torch.from_numpy(all_embeddings[batch_indices]).to(torch_dtype)

        return features

    def get_embedding(self, store_path: str, sample_id: str) -> torch.Tensor:
        """
        Get an embedding for a sample ID from the store.

        Parameters
        ----------
        store_path : str
            Path to the embedding store
        sample_id : str
            Sample ID to retrieve

        Returns
        -------
        torch.Tensor
            The embedding tensor
        """
        # Check if embeddings are in memory
        if store_path in _EMBEDDING_CACHE:
            # Get from memory
            cache = _EMBEDDING_CACHE[store_path]
            sample_ids = cache["sample_ids"]
            data = cache["data"]

            # Find the index
            idx = np.where(sample_ids == sample_id)[0][0]

            # Return the embedding
            return data[idx]
        else:
            # Load from disk
            npzfile = np.load(store_path, allow_pickle=True)
            sample_ids = npzfile["sample_ids"]
            data = npzfile["data"]

            # Find the index
            idx = np.where(sample_ids == sample_id)[0][0]

            # Return the embedding
            return torch.from_numpy(data[idx]).float()

    def clear_cache(self):
        """Clear the embedding cache."""
        if self.store_path in _EMBEDDING_CACHE:
            del _EMBEDDING_CACHE[self.store_path]


def debug_embedding_cache(store_path: str, dataset: Dataset | DatasetDict) -> None:
    """
    Debug function to check if all samples in the dataset are in the cache.

    Parameters
    ----------
    store_path : str
        Path to the embedding store
    dataset : Union[Dataset, DatasetDict]
        The dataset to check
    """
    if store_path not in _EMBEDDING_CACHE:
        logger.warning(f"Store path {store_path} not in cache")
        return

    cache = _EMBEDDING_CACHE[store_path]
    sample_id_to_idx = cache["sample_id_to_idx"]

    if isinstance(dataset, DatasetDict):
        for split_name, split_dataset in dataset.items():
            missing_samples = []
            for example in split_dataset:
                sample_id = example["sample_id"]
                if sample_id not in sample_id_to_idx:
                    missing_samples.append(sample_id)

            if missing_samples:
                logger.warning(f"Split {split_name}: {len(missing_samples)} samples missing from cache")
                logger.warning(f"First few missing samples: {missing_samples[:5]}")
            else:
                logger.info(f"Split {split_name}: All {len(split_dataset)} samples found in cache")
    else:
        missing_samples = []
        for example in dataset:
            sample_id = example["sample_id"]
            if sample_id not in sample_id_to_idx:
                missing_samples.append(sample_id)

        if missing_samples:
            logger.warning(f"{len(missing_samples)} samples missing from cache")
            logger.warning(f"First few missing samples: {missing_samples[:5]}")
        else:
            logger.info(f"All {len(dataset)} samples found in cache")
