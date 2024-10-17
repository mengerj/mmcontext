# mmcontext/datasets/__init__.py

import logging
import os

import anndata


def load_example_dataset(data_dir: str = "../../data/test_data/") -> anndata.AnnData:
    """
    Loads the example dataset for the mmcontext tutorial.

    Returns
    -------
        AnnData: The loaded AnnData object containing the example dataset.
        data_dir (str): The directory to save the example dataset (default: data/test_data).
    """
    logger = logging.getLogger(__name__)

    dataset_path = os.path.join(data_dir, "small_cellxgene.h5ad")
    logger.info("Loading the example dataset, which is taken from cellxgene...")
    # Create the dataset directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Check if the dataset already exists
    if not os.path.isfile(dataset_path):
        logger.info("Downloading the example dataset...")
        # Download the dataset
        logger.error("Download not implemented yet")
        # print("Download complete.")

    # Load the dataset
    adata = anndata.read_h5ad(dataset_path)
    return adata
