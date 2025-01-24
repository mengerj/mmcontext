import logging
import random

import anndata
from sentence_transformers.readers.InputExample import InputExample

logger = logging.getLogger(__name__)


class AnnDataDataSetConstructor:
    """Class to generate a dataset compatible with the SentenceTransformer library from anndata files."""

    def __init__(self, caption_constructor, negatives_per_sample: int = 1):
        """
        Initialize the AnnDataDataSetConstructor.

        Args:
            caption_constructor: Constructor for creating captions
            negatives_per_sample: Number of negative examples to create per positive example
        """
        self.caption_constructor = caption_constructor
        self.negatives_per_sample = negatives_per_sample
        self.anndata_files = []
        self.sample_id_keys = {}
        self.dataset = []

    def add_anndata(self, file_path: str, sample_id_key: str | None = None) -> None:
        """
        Add an anndata file to the constructor.

        Args:
            file_path: Path to the anndata file
            sample_id_key: Optional key in adata.obs to use for sample IDs.
                          If None, uses adata.obs_names
        """
        # 1. Check extension
        if not (file_path.endswith(".zarr")):
            logger.error("Unsupported anndata format for file: %s", file_path)
            raise ValueError(
                f"File {file_path} does not appear to be .zarr format."
                "You can convert it to .zarr using anndata.write_zarr(adata, 'filename.zarr')."
            )

        # 2. Check for duplicates
        if file_path in self.anndata_files:
            logger.error("File %s has already been added to the constructor.", file_path)
            raise ValueError(f"File {file_path} has already been added.")

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
        adata = anndata.read_zarr(file_path)
        self.caption_constructor.construct_captions(adata)
        adata.write_zarr(file_path)  # Overwrite the original file

    def getCaption(self, file_path: str) -> dict[str, str]:
        """
        Get a dictionary mapping sample IDs to captions from an anndata file.

        Args:
            file_path: Path to the anndata file

        Returns
        -------
            Dict mapping sample IDs to captions
        """
        adata = anndata.read_zarr(file_path)
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

        Args:
            current_file: Current file path
            current_sample: Current sample ID
            current_caption: Caption of the current sample
            all_captions: Nested dict mapping file paths to {sample_id: caption} dicts
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

    def clear(self) -> None:
        """Clear all stored data in the constructor."""
        self.anndata_files.clear()
        self.sample_id_keys.clear()
        self.dataset.clear()
