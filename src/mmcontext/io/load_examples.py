# mmcontext/datasets/__init__.py

import importlib.resources
import logging

import anndata


def load_example_dataset() -> anndata.AnnData:
    """
    Loads the example dataset for the mmcontext tutorial.

    Returns
    -------
        AnnData: The loaded AnnData object containing the example dataset.
        data_dir (str): The directory to save the example dataset (default: data/test_data).
    """
    logger = logging.getLogger(__name__)

    logger.info("Loading the example dataset, which is taken from cellxgene...")

    with importlib.resources.path("mmcontext.datasets", "small_cellxgene.h5ad") as dataset_path:
        adata = anndata.read_h5ad(dataset_path)

    return adata
