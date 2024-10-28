# pp/context_embedder.py
import gzip
import logging
import os
import pickle
from abc import ABC, abstractmethod

import anndata
import numpy as np
import openai
import pandas as pd
from openai import OpenAI


class ContextEmbedder(ABC):
    """Abstract base class for generating embeddings for AnnData objects."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def embed(self, adata: anndata.AnnData):
        """
        Abstract method for embedding generation.

        Parameters
        ----------
        adata
            The AnnData object to generate embeddings for.
        """
        pass


class PlaceholderContextEmbedder(ContextEmbedder):
    """Generates random embeddings as a placeholder for real embeddings."""

    def embed(self, adata: anndata.AnnData) -> np.ndarray:
        """
        Generates random embeddings for the given AnnData object.

        Parameters
        ----------
        adata
            The AnnData object to generate embeddings for.

        Returns
        -------
            Randomly generated embeddings.
        """
        n_samples = adata.n_obs
        embedding_dim = 128  # Example embedding dimension
        embeddings = np.random.rand(n_samples, embedding_dim)
        return embeddings


class CategoryEmbedder(ContextEmbedder):
    """Embed metadata categories using OpenAI's text embedding model.

    ContextEmbedder Subclass that looks at the individual metadata categories and generates embeddings for each unique value. If multiple classes are chosen,
    the embeddings are combined using the specified `combination_method`.

    Parameters
    ----------
    metadata_categories
        List of metadata categories to process.
    embeddings_file_path
        Path to the dictionary to save/load embeddings.
    model
        Text embedding model to use.
    combination_method
        Method to combine embeddings ('average', 'concatenate').
    one_hot
        Whether to use one-hot encoding for metadata categories.
    unknown_threshold
        Threshold for unknown elements when API key is not available but most classes were known from dictionary.
        If less than the threshold, unkonwn elements will be filled with zeros.
    logger
        logger object to use for logging.

    Raises
    ------
    ValueError
        If a metadata category is missing from adata.obs.
    ValueError
        If no API key is set and unknown elements are greater than threshold.
    """

    def __init__(
        self,
        metadata_categories: list,
        embeddings_file_path: str,
        model: str = "text-embedding-3-small",
        combination_method: str = "concatenate",
        one_hot: bool = False,
        unknown_threshold: int = 20,
        logger: logging.Logger | None = None,
    ):
        super().__init__(logger)
        self.metadata_categories = metadata_categories
        self.embeddings_file_path = embeddings_file_path
        self.model = model
        self.combination_method = combination_method
        self.one_hot = one_hot
        self.metadata_embeddings = self.load_metadata_embeddings()
        self.unknown_threshold = unknown_threshold

    def embed(self, adata: anndata.AnnData) -> np.ndarray:
        """
        Generates or updates embeddings for the AnnData object and combines them.

        Parameters
        ----------
        adata
            The AnnData object containing the data.

        Returns
        -------
        Combined context embeddings.
        """
        self.process_adata_metadata_embeddings(adata)
        combined_embeddings = self.combine_metadata_embeddings(adata)
        return combined_embeddings

    def process_adata_metadata_embeddings(self, adata):
        """
        Processes and updates metadata embeddings for the AnnData object.

        Parameters
        ----------
        adata
            The AnnData object containing the data.
        """
        # Check for required metadata categories
        for category in self.metadata_categories:
            if category not in adata.obs:
                raise ValueError(f"Metadata category '{category}' not found in adata.obs.")

        # Initialize a flag to track if there was an API failure
        api_call_failed = False

        if self.one_hot:
            self.logger.info("One-hot encoding metadata categories...")
            for category in self.metadata_categories:
                adata.obsm[f"{category}_emb"] = pd.get_dummies(adata.obs[category]).values.astype(int)
                self.logger.info(f"Embeddings for '{category}' stored in adata.obsm['{category}_emb']")
            return  # No need to generate embeddings using a model

        # Check for new categories or values that need embeddings
        unique_categories = {}
        for category in self.metadata_categories:
            unique_values = adata.obs[category].unique().tolist()
            if category not in self.metadata_embeddings or any(
                value not in self.metadata_embeddings[category] for value in unique_values
            ):
                unique_categories[category] = unique_values

        # Update embeddings if needed
        if unique_categories:
            try:
                self.generate_update_metadata_embeddings(unique_categories)
            except RuntimeError as e:
                api_call_failed = True
                self.logger.error(f"API call failed: {e}")  # Log the specific API call error
        # Check for total unknown elements and handle accordingly
        total_unknown = sum([len(values) for values in unique_categories.values()])
        if api_call_failed and total_unknown > self.unknown_threshold:
            self.logger.error(
                f"Unknown elements exceed the threshold: {total_unknown} is greater than the threshold: {self.unknown_threshold}."
            )
            raise ValueError(
                f"API call failed. Unknown elements exceed the threshold. {total_unknown} is greater than threshold: {self.unknown_threshold}."
            )
        elif api_call_failed:
            self.logger.warning(
                f"API call failed but {total_unknown} unknown elements is less than the threshold: {self.unknown_threshold}. Filling unknown elements with zeros."
            )
        # Map embeddings to each sample
        for category in self.metadata_categories:
            # Raise an error if no embeddings are available at all for the category
            if category not in self.metadata_embeddings or not self.metadata_embeddings[category]:
                raise ValueError(
                    f"No embeddings found for category '{category}'. Cannot proceed with zero vector filling."
                )

            # Get the embedding dimension from a valid stored embedding
            embedding_dim = None
            for value in self.metadata_embeddings[category].values():
                if value is not None:
                    embedding_dim = len(value)
                    break
            if embedding_dim is None:
                raise ValueError(f"Failed to determine the embedding dimension for category '{category}'.")

            category_embeddings = []
            # Loop through the values in adata.obs for the category
            for value in adata.obs[category]:
                if value in self.metadata_embeddings[category]:
                    # If the embedding exists, use it
                    category_embeddings.append(self.metadata_embeddings[category][value])
                else:
                    # Use zeros if no embedding is found for the value
                    category_embeddings.append(np.zeros(embedding_dim))

            # Store the embeddings in adata.obsm
            adata.obsm[f"{category}_emb"] = np.array(category_embeddings)
            self.logger.info(f"Embeddings for '{category}' stored in adata.obsm['{category}_emb']")

    def combine_metadata_embeddings(self, adata: anndata.AnnData) -> np.ndarray:
        """
        Combines embeddings from specified metadata categories.

        Parameters
        ----------
        adata
            The AnnData object containing the data.

        Returns
        -------
        Combined embeddings.
        """
        # Retrieve embeddings from adata.obsm
        embeddings = [
            adata.obsm[f"{category}_emb"] for category in self.metadata_categories if f"{category}_emb" in adata.obsm
        ]

        # Check that all embeddings are present
        if len(embeddings) != len(self.metadata_categories):
            missing = set(self.metadata_categories) - {key.replace("_emb", "") for key in adata.obsm.keys()}
            raise ValueError(f"Missing embeddings in adata.obsm: {missing}")

        # Combine embeddings
        if self.combination_method == "average":
            combined_embedding = np.mean(embeddings, axis=0)
        elif self.combination_method == "concatenate":
            combined_embedding = np.concatenate(embeddings, axis=1)
        else:
            raise ValueError(f"Unsupported combination method: {self.combination_method}")

        # Optionally store combined embeddings in adata.obsm
        adata.obsm["c_emb"] = combined_embedding
        self.logger.info("Combined context embeddings stored in adata.obsm['c_emb']")

        return combined_embedding

    def generate_update_metadata_embeddings(self, unique_categories: dict):
        """Generates or updates embeddings for metadata categories.

        Parameters
        ----------
        unique_categories
            Dictionary with categories as keys and list of unique values.
        """
        updated = False

        for category, values in unique_categories.items():
            if category not in self.metadata_embeddings:
                self.metadata_embeddings[category] = {}

            for value in values:
                if value not in self.metadata_embeddings[category]:
                    # Generate embedding for new value
                    embedding = self.generate_text_embedding(value)
                    if embedding is not None:
                        self.metadata_embeddings[category][value] = embedding
                        self.logger.info(f"Generated new embedding for '{value}' in '{category}'")
                        updated = True
                    else:
                        # Raise an error instead of logging it here to propagate back to the caller
                        raise RuntimeError(f"Failed to generate embedding for '{value}' in '{category}'")

        # Save embeddings if updated
        if updated:
            self.save_metadata_embeddings()
            self.logger.info("Updated embeddings dictionary saved.")
        else:
            self.logger.info("No new embeddings generated; dictionary loaded from file and not updated.")

    def generate_text_embedding(self, text: str) -> list | None:
        """
        Generates an embedding for the given text using OpenAI's embedding model.

        Parameters
        ----------
        text
            The text for which to generate an embedding.

        Returns
        -------
        A list representing the embedding vector, or None in case of failure.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.logger.warning("OPENAI_API_KEY is not set.")
            return None

        client = OpenAI(api_key=api_key)
        try:
            response = client.embeddings.create(input=text, model=self.model)
            embedding = response.data[0].embedding
            return embedding
        except openai.error.InvalidRequestError as e:
            self.logger.warning(f"Invalid request for embedding generation: {e}")
        except openai.error.AuthenticationError as e:
            self.logger.warning(f"Authentication failed for embedding generation: {e}")
        except openai.error.APIError as e:
            self.logger.warning(f"OpenAI API error occurred: {e}")
        except openai.error.APIConnectionError as e:
            self.logger.warning(f"Network error when trying to generate embedding: {e}")
        except openai.error.RateLimitError as e:
            self.logger.warning(f"Rate limit exceeded for embedding generation: {e}")

        return None

    def load_metadata_embeddings(self) -> dict:
        """
        Loads the metadata embeddings from a compressed file.

        Returns
        -------
        The embeddings dictionary.
        """
        if os.path.exists(self.embeddings_file_path):
            with gzip.open(self.embeddings_file_path, "rb") as f:
                embeddings = pickle.load(f)
            self.logger.info("Loaded embeddings from file.")
            total_elements = sum([len(values) for values in embeddings.values()])
            self.logger.info(
                f"Embeddings dictionary contains the following categories: {embeddings.keys()} with a total of {total_elements} elements."
            )
            return embeddings
        else:
            self.logger.info("No embeddings file found; starting with empty embeddings.")
            return {}

    def save_metadata_embeddings(self):
        """Saves the metadata embeddings to a compressed file."""
        os.makedirs(os.path.dirname(self.embeddings_file_path), exist_ok=True)
        with gzip.open(self.embeddings_file_path, "wb") as f:
            pickle.dump(self.metadata_embeddings, f)
        self.logger.info(f"Embeddings saved to {self.embeddings_file_path}")
