import logging
import random

import anndata
from sentence_transformers.readers.InputExample import InputExample

logger = logging.getLogger(__name__)


class AnnDataSetConstructor:
    """Class to generate a dataset compatible with the SentenceTransformer library from anndata files."""

    def __init__(self, caption_constructor, negatives_per_sample: int = 1):
        """
        Initialize the AnnDataSetConstructor.

        Parameters
        ----------
        caption_constructor
            Constructor for creating captions
        negatives_per_sample
            Number of negative examples to create per positive example
        """
        self.caption_constructor = caption_constructor
        self.negatives_per_sample = negatives_per_sample
        self.anndata_files = []
        self.sample_id_keys = {}
        self.dataset = []

    def _check_sample_id_uniqueness(self, adata: anndata.AnnData, file_path: str, sample_id_key: str | None) -> None:
        """
        Check if sample IDs are unique for the given anndata object.

        Parameters
        ----------
        adata
            AnnData object to check
        file_path
            Path to the anndata file (for error message)
        sample_id_key
            Key in adata.obs to use for sample IDs, if None uses adata.obs.index

        Raises
        ------
            ValueError: If sample IDs are not unique
        """
        sample_ids = adata.obs.index if sample_id_key is None else adata.obs[sample_id_key]
        n_total = len(sample_ids)
        n_unique = len(set(sample_ids))

        if n_unique < n_total:
            duplicates = sample_ids[sample_ids.duplicated()].unique()
            error_msg = (
                f"Found {n_total - n_unique} duplicate sample IDs in {file_path}.\n"
                f"Example duplicates: {list(duplicates)[:3]}...\n"
                "To fix this, either:\n"
                "1. Provide a different sample_id_key that contains unique identifiers, or\n"
                "2. Remove duplicate samples from your dataset"
            )
            if sample_id_key is None:
                error_msg += "\nCurrently using adata.obs.index as sample IDs."
            else:
                error_msg += f"\nCurrently using adata.obs['{sample_id_key}'] as sample IDs."

            logger.error(error_msg)
            raise ValueError(error_msg)

    def add_anndata(self, file_path: str, sample_id_key: str | None = None) -> None:
        """
        Add an anndata file to the constructor.

        Parameters
        ----------
        file_path
            Path to the anndata file
        sample_id_key
            Optional key in adata.obs to use for sample IDs. If None, uses adata.obs.index
        """
        self.is_zarr = False
        self.is_h5ad = False
        # 1. Check extension
        if file_path.endswith(".zarr") or file_path.endswith(".zarr/"):
            self.is_zarr = True
        elif file_path.endswith(".h5ad"):
            self.is_h5ad = True
        else:
            logger.error("Unsupported anndata format for file: %s", file_path)
            raise ValueError(f"File {file_path} does not appear to be .zarr or .h5ad format.")

        # 2. Check for duplicates
        if file_path in self.anndata_files:
            logger.error("File %s has already been added to the constructor.", file_path)
            raise ValueError(f"File {file_path} has already been added.")

        # 3. Check sample ID uniqueness
        if self.is_zarr:
            adata = anndata.read_zarr(file_path)
        if self.is_h5ad:
            adata = anndata.read_h5ad(file_path)
        self._check_sample_id_uniqueness(adata, file_path, sample_id_key)

        self.anndata_files.append(file_path)
        self.sample_id_keys[file_path] = sample_id_key
        logger.info("Successfully added anndata file: %s", file_path)

    def buildCaption(self, file_path: str) -> None:
        """
        Build captions for an anndata file using the provided caption constructor.

        Args:
            file_path: Path to the anndata file
            caption_constructor: Instance of a caption constructor class
        """
        if self.is_zarr:
            adata = anndata.read_zarr(file_path)
        if self.is_h5ad:
            adata = anndata.read_h5ad(file_path)
        self.caption_constructor.construct_captions(adata)
        if self.is_zarr:
            adata.write_zarr(file_path)
        if self.is_h5ad:
            adata.write_h5ad(file_path)

    def getCaption(self, file_path: str) -> dict[str, str]:
        """
        Get a dictionary mapping sample IDs to captions from an anndata file.

        Args:
            file_path: Path to the anndata file

        Returns
        -------
            Dict mapping sample IDs to captions
        """
        if self.is_zarr:
            adata = anndata.read_zarr(file_path)
        if self.is_h5ad:
            adata = anndata.read_h5ad(file_path)
        if "caption" not in adata.obs.columns:
            raise ValueError(f"No 'caption' column found in {file_path}")

        sample_id_key = self.sample_id_keys[file_path]
        sample_ids = adata.obs.index if sample_id_key is None else adata.obs[sample_id_key]

        return dict(zip(sample_ids, adata.obs["caption"], strict=False))

    def _create_negative_example(
        self, current_file: str, current_sample: str, current_caption: str, all_captions: dict[str, dict[str, str]]
    ) -> InputExample:
        """
        Create a negative example by finding a caption that doesn't match the current sample.

        Parameters
        ----------
        current_file
            Current file path
        current_sample
            Current sample ID
        current_caption
            Caption of the current sample
        all_captions
            Nested dict mapping file paths to {sample_id: caption} dicts
        """
        while True:
            # Randomly choose a file
            neg_file = random.choice(self.anndata_files)
            # Randomly choose a sample from that file
            neg_sample = random.choice(list(all_captions[neg_file].keys()))
            neg_caption = all_captions[neg_file][neg_sample]

            # Check if this is actually a negative example
            if neg_caption != current_caption:
                metadata = {"file_path": current_file, "sample_id": current_sample}

                return InputExample(
                    guid=f"{current_file}_{current_sample}_neg", texts=[metadata, neg_caption], label=0.0
                )

    def get_dataset(self) -> list[InputExample]:
        """
        Create and return the dataset containing InputExample instances for all added anndata files.

        Returns
        -------
            List of InputExample instances
        """
        dataset = []
        all_captions = {}  # Nested dict: {file_path: {sample_id: caption}}

        # First, build all captions and store them
        for file_path in self.anndata_files:
            self.buildCaption(file_path)
            all_captions[file_path] = self.getCaption(file_path)

        # Create positive and negative examples
        for file_path in self.anndata_files:
            caption_dict = all_captions[file_path]

            for sample_id, caption in caption_dict.items():
                # Create positive example
                metadata = {"file_path": file_path, "sample_id": sample_id}

                positive_example = InputExample(
                    guid=f"{file_path}_{sample_id}_pos", texts=[metadata, caption], label=1.0
                )
                dataset.append(positive_example)

                # Create negative examples
                for _ in range(self.negatives_per_sample):
                    negative_example = self._create_negative_example(file_path, sample_id, caption, all_captions)
                    dataset.append(negative_example)

        return dataset

    def get_inference_dataset(self) -> tuple[list[dict[str, str]], list[str], list[str]]:
        """Build a dataset from an anndata file suitable for SentenceTransformer.encode.

        Build and return separate lists for inference: a list of metadata dicts, a list of captions,
        and a parallel list of sample IDs (all in the same order).

        The method reads each .zarr file (adding captions if not already present via `buildCaption`),
        then extracts sample IDs and captions. This is useful for inference scenarios where you
        need to maintain the exact order of samples for external reference.

        Returns
        -------
        metadata_list : list of dict
            Each dictionary contains ``{"file_path": <path_to_zarr>, "sample_id": <sample_id>}``.
            This is useful if you need to retrieve the file path and sample ID for downstream processing.
        captions_list : list of str
            A list of caption strings corresponding to each sample in the dataset.
        sample_ids : list of str
            A list of sample IDs in the same index order as the captions_list.

        Notes
        -----
        - This method internally calls ``buildCaption(file_path)`` to ensure each file
        is annotated with a ``"caption"`` column. If the column already exists, the
        constructor logic may simply overwrite or skip as needed.
        - The data is sourced from the .zarr files previously added via ``add_anndata``.
        - Logging messages are issued at various points to indicate progress.
        """
        metadata_list = []
        captions_list = []
        sample_ids = []

        # For each file, ensure captions are built and then retrieved
        for file_path in self.anndata_files:
            logger.info("Building caption for inference from file: %s", file_path)
            self.buildCaption(file_path)  # This will overwrite the .zarr file if new captions were generated

            logger.info("Retrieving captions for inference from file: %s", file_path)
            caption_dict = self.getCaption(file_path)

            # Gather data into parallel lists
            for sid, caption in caption_dict.items():
                metadata_list.append({"file_path": file_path, "sample_id": sid})
                captions_list.append(caption)
                sample_ids.append(sid)

        logger.info("Constructed inference dataset with %d samples.", len(sample_ids))
        return metadata_list, captions_list, sample_ids

    def clear(self) -> None:
        """Clear all stored data in the constructor."""
        self.anndata_files.clear()
        self.sample_id_keys.clear()
        self.dataset.clear()
