import logging

import anndata
import numpy as np
from sentence_transformers import SentenceTransformer

from mmcontext.pp import AnnDataSetConstructor

logger = logging.getLogger(__name__)


class MMContextInference:
    """Class to handle inference with trained MMContext models."""

    def __init__(
        self,
        file_path: str,
        constructor: AnnDataSetConstructor,
        model: SentenceTransformer,
        sample_id_key: str | None = None,
    ):
        """
        Initialize the inference class.

        Parameters
        ----------
        file_path
            Path to the AnnData file
        constructor
            Pre-initialized AnnDataSetConstructor
        model
            Pre-trained SentenceTransformer model
        sample_id_key
            Optional key in adata.obs to use for sample IDs
        """
        self.file_path = file_path
        self.model = model

        # Add file to constructor and get inference dataset
        constructor.add_anndata(file_path, sample_id_key=sample_id_key)
        self.metadata_list, self.captions_list, self.sample_ids = constructor.get_inference_dataset()

        # Load AnnData and verify sample order
        self.adata = self._load_anndata(file_path)
        self._verify_sample_order()

    def _load_anndata(self, file_path) -> anndata.AnnData:
        if file_path.endswith(".zarr"):
            adata = anndata.read_zarr(file_path)
        if file_path.endswith(".h5ad"):
            adata = anndata.read_h5ad(file_path)
        return adata

    def _verify_sample_order(self) -> None:
        """
        Verify that sample IDs from the dataset match the order in AnnData.

        Raises
        ------
        ValueError
            If sample orders don't match
        """
        adata_ids = self.adata.obs.index.tolist()
        if not len(self.sample_ids) == len(adata_ids):
            raise ValueError(
                f"Number of samples in dataset ({len(self.sample_ids)}) doesn't match AnnData ({len(adata_ids)})"
            )

        if not all(ds_id == adata_id for ds_id, adata_id in zip(self.sample_ids, adata_ids, strict=False)):
            raise ValueError(
                "Sample IDs order in dataset doesn't match AnnData order. "
                "This would result in incorrect embedding assignment."
            )

        logger.info("Sample order verification successful")

    def encode(self, batch_size: int = 32) -> anndata.AnnData:
        """
        Compute embeddings for omics data and captions and add them to AnnData.

        Parameters
        ----------
        batch_size: int
            Batch size to use for encoding

        Returns
        -------
            AnnData object with new embeddings in .obsm
        """
        logger.info("Computing embeddings for omics data...")
        omics_embeddings = self.model.encode(self.metadata_list, show_progress_bar=True)

        logger.info("Computing embeddings for captions...")
        caption_embeddings = self.model.encode(self.captions_list, show_progress_bar=True)

        # Create new AnnData object to avoid modifying the original
        adata_new = self.adata.copy()

        # Add embeddings to obsm
        adata_new.obsm["omics_emb"] = omics_embeddings.astype(np.float32)
        adata_new.obsm["caption_emb"] = caption_embeddings.astype(np.float32)

        logger.info(
            f"Added embeddings to .obsm: omics_emb {omics_embeddings.shape}, caption_emb {caption_embeddings.shape}"
        )

        return adata_new
