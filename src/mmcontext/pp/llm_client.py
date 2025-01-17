import json
import logging
import os
from abc import ABC, abstractmethod

import requests

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients that generate text embeddings."""

    @abstractmethod
    def generate_embedding(self, text: str) -> list[float] | None:
        """Generates an embedding for the given text using a specific LLM backend.

        Parameters
        ----------
        text : str
            The text for which to generate an embedding.

        Returns
        -------
        list of float or None
            A list representing the embedding vector, or None in case of failure.
        """
        pass


class OpenAILLMClient(BaseLLMClient):
    """LLM client for the OpenAI API."""

    def __init__(self, model: str = "text-embedding-ada-002"):
        """Initializes the OpenAI LLM client with the specified model.

        Parameters
        ----------
        api_key : str
            Your OpenAI API key.
        model : str, optional
            The name of the OpenAI model to use for embedding.
        """
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.logger = logging.getLogger(__name__)
        if not self.api_key:
            self.logger.warning("OPENAI_API_KEY is not set.")
            return None

    def generate_embedding(self, text: str) -> list[float] | None:
        """Generates an embedding for the given text using the OpenAI API."""
        import openai  # keep the import local if you prefer

        if not self.api_key:
            raise RuntimeError("OpenAI API key is not set.")
        client = openai.OpenAI(api_key=self.api_key)
        try:
            response = client.embeddings.create(input=text, model=self.model)
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            self.logger.error("Error calling OpenAI API: %s", e)

        return None


class LocalLLMClient(BaseLLMClient):
    """LLM client for a locally hosted or custom server model."""

    def __init__(self, url: str, token: str | None = None, model: str = "LocalModel"):
        """Initialize the local LLM client with the specified URL, token, and model.

        Parameters
        ----------
        url : str
            The endpoint for your local LLM server or custom model API.
        token : str, optional
            The authentication token (if needed).
        model : str, optional
            The model identifier used in the local or custom API request body.
        """
        self.url = url
        self.token = token
        self.model = model

    def generate_embedding(self, text: str) -> list[float] | None:
        """Generates an embedding for the given text using a locally hosted or custom server model."""
        logger.debug("Generating local LLM embedding for text: %s", text)
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        # Example request body, depends on your local model's API contract
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            # Might also need to indicate you want an embedding, depending on your server setup
        }

        try:
            response = requests.post(self.url, headers=headers, data=json.dumps(body))
            response.raise_for_status()
            # The structure of the response depends on your local server’s design
            data = response.json()
            # For demonstration, assume that 'data["embedding"]' is the embedding
            return data.get("embedding", None)
        except requests.RequestException as e:
            logger.error("Error calling local LLM server: %s", e)
            return None
