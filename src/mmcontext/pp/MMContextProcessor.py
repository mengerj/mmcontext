import numpy as np
import torch
import transformers
from scipy import sparse as sp
from sklearn.decomposition import PCA
from torch import Tensor


class MMContextProcessor:
    """A Processor to create initial embeddings for text and omics data input.

    Uses a tokenizer for text data and a custom processor for omics data.
    The latter can be chosen from several apporaches
    """

    def __init__(self, omics_processor_name="none", text_encoder_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(text_encoder_name)
        self.omics_processor = self._load_omics_processor(omics_processor_name)

    def _load_omics_processor(self, omics_processor_name):
        if omics_processor_name == "none":
            processor = NoneProcessor()
            return processor


class NoneProcessor:
    """Processor that retrieves the raw data without any processing."""

    def encode(self, data):
        """Take the raw data and return it as is.

        Encodes a list of omicsSample objects into a feature dictionary with
        a tensor of shape (batch_size, seq_length, feature_dim).

        Parameters
        ----------
        data : list
            A list of omicsSample objects.

        Returns
        -------
        features : dict
            A dictionary containing the omics embeddings tensor with shape
            (batch_size, seq_length, feature_dim).
        """
        if isinstance(data, list):
            omics_embeddings = []

            for sample in data:
                counts = torch.tensor(sample.get("counts")).unsqueeze(0).float()  # Shape: [1, feature_dim]
                omics_embeddings.append(counts)

            # Concatenate along the batch dimension
            features = torch.cat(omics_embeddings, dim=0)  # Shape: [batch_size, feature_dim]
        else:
            raise ValueError("Data must be a list of omicsSample objects")

        return features


class PCAOmicsProcessor:
    """Processor to encode omics data"""

    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca = None
        self.mean_ = None

    def fit(self, array: np.ndarray | Tensor | sp.spmatrix):
        """Fits PCA on the input array."""
        count_matrix = array
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(count_matrix)
        self.mean_ = np.mean(count_matrix, axis=0)

    def encode(self, count_vector):
        """Encodes a single count vector using the fitted PCA."""
        if self.pca is None or self.mean_ is None:
            raise ValueError("PCA encoder must be fitted before encoding.")
        standardized_vector = count_vector - self.mean_
        return self.pca.transform(standardized_vector.reshape(1, -1))[0]

    def save(self, filepath):
        """Saves the PCA components and mean."""
        np.savez(filepath, components=self.pca.components_, mean=self.mean_)

    def load(self, filepath):
        """Loads the PCA components and mean."""
        data = np.load(filepath)
        self.pca = PCA(n_components=self.n_components)
        self.pca.components_ = data["components"]
        self.mean_ = data["mean"]
