import logging

import numpy as np

# import faiss

logger = logging.getLogger(__name__)


class MultimodalFaissIndex:
    """
    A class to handle both text-based and data-based similarity search using FAISS.

    This class allows two main modes of operation:
    1. Querying a dataset (e.g., RNA-seq samples) with a text query to find the most
       relevant data samples.
    2. Annotating data (e.g., RNA-seq samples) by matching them to text labels.

    Parameters
    ----------
    text_encoder : callable, optional
        A function or callable that takes text input and returns an embedding vector
        (e.g., a trained transformer or any sentence encoder). If not provided,
        the relevant methods for text processing will be unavailable.

    data_encoder : callable, optional
        A function or callable that takes numeric data input (e.g., RNA-seq data) and
        returns an embedding vector. If not provided, the relevant methods for numeric
        data processing will be unavailable.

    Attributes
    ----------
    index : faiss.Index or None
        The FAISS index object once built. Initially None if `build_index` has not
        been called.
    labels : list
        A list of labels corresponding to items in the FAISS index. This is populated
        during the call to `build_index`.
    mode : str or None
        The current mode of the index, either 'text' or 'data'. Determined during
        `build_index`.

    References
    ----------
    The data to be indexed is expected to come from user-provided sources such as
    text descriptions or RNA-seq samples.
    """

    def __init__(self, text_encoder=None, data_encoder=None):
        self.text_encoder = text_encoder
        self.data_encoder = data_encoder
        self.index = None
        self.labels = []
        self.mode = None

    def build_index(self, references, labels, mode="text"):
        """
        Builds a FAISS index from a set of references, either text-based or numeric data.

        If `mode='text'`, the references are assumed to be strings which will be passed
        through `text_encoder`. If `mode='data'`, the references are assumed to be
        numeric data which will be passed through `data_encoder`. The resulting
        embeddings will be normalized and then indexed in FAISS for fast similarity
        search.

        Parameters
        ----------
        references : list
            A list of references to be embedded. The type of elements in this list
            depends on the mode:
            - `mode='text'`: list of strings
            - `mode='data'`: list of numeric data arrays (e.g., RNA-seq vectors)
        labels : list
            A list of labels corresponding to each reference. The length of `labels`
            must match `references`.
        mode : {'text', 'data'}, optional
            The mode for the index. If 'text', references are embedded by `text_encoder`.
            If 'data', references are embedded by `data_encoder`.

        Returns
        -------
        None
            The index is stored in `self.index`, and labels are stored in `self.labels`.
        """
        logger.info("Building FAISS index in %s mode.", mode)
        if mode not in ["text", "data"]:
            raise ValueError("Mode must be either 'text' or 'data'.")

        # Encode the references according to the selected mode
        if mode == "text":
            if self.text_encoder is None:
                raise ValueError("No text_encoder provided, but mode='text' requested.")
            embeddings = [self.text_encoder(r) for r in references]
        else:  # mode == "data"
            if self.data_encoder is None:
                raise ValueError("No data_encoder provided, but mode='data' requested.")
            embeddings = [self.data_encoder(r) for r in references]

        embeddings_array = np.vstack(embeddings)
        self.index = self._build_faiss_index(embeddings_array)
        self.labels = labels
        self.mode = mode

    def query_dataset(self, data_samples, text_query, k=1):
        """Function to query the index with a text query and return the most similar data samples.

        Given the following inputs:
        - A list of data samples (e.g., numeric vectors for RNA-seq).
        - A single text query.

        Embeds the data samples using `data_encoder` and searches for the most similar
        item to the text query in the FAISS index. The text query is encoded using
        `text_encoder`. This method returns similarity scores between the query and
        each data sample, effectively ranking data samples by their relevance to the
        text query.

        Parameters
        ----------
        data_samples : list of numeric arrays
            A list of data samples (e.g., RNA-seq vectors) to be embedded using
            `data_encoder`.
        text_query : str
            A single text query to find the most relevant data sample.
        k : int, optional
            Number of top similar items to return for each query embedding. Default is 1.

        Returns
        -------
        list of lists
            A list containing one inner list (because there is a single text query),
            where each inner list contains tuples of (label, similarity_score) for the
            top `k` matches among the data samples.
        """
        logger.info("Querying dataset with a text query.")
        if self.mode != "data":
            raise ValueError("Index mode is not 'data'. Please build an index with `mode='data'` first.")

        # Embed the data samples using data_encoder
        data_embeddings = [self.data_encoder(sample) for sample in data_samples]
        data_embeddings_array = np.vstack(data_embeddings)

        # Encode the text query
        text_query_embedding = self.text_encoder(text_query).reshape(1, -1)

        # Now we want the similarity of this single text query to each data embedding
        # One approach is to build an in-memory, temporary index for the data
        # or simply compute the dot product. But to reuse FAISS, we can do:
        # 1. Build a new index on data_embeddings
        # 2. Query with text_query_embedding
        temp_index = self._build_faiss_index(data_embeddings_array)
        results = self._query_faiss_index(
            text_query_embedding, temp_index, labels=[f"Sample_{i}" for i in range(len(data_samples))], k=k
        )
        return results

    def annotate_data(self, data_samples, k=1):
        """Annotate an array of data samples with the most similar text labels. From labels pre embedded.

        Given data samples (e.g., RNA-seq vectors), this method uses the existing
        text-based index (containing label embeddings) to find the most similar text
        label for each sample. This effectively "annotates" the data samples with their
        most relevant text label.

        Parameters
        ----------
        data_samples : list of numeric arrays
            A list of data samples (e.g., RNA-seq vectors) for which to find the most
            similar labels in the text-based FAISS index.
        k : int, optional
            Number of top labels to return for each data sample. Default is 1.

        Returns
        -------
        list of lists
            A list of lists, where each inner list contains tuples of
            (label, similarity_score) for the top `k` matches among the text labels.
        """
        logger.info("Annotating data samples with text labels.")
        if self.mode != "text":
            raise ValueError("Index mode is not 'text'. Please build an index with `mode='text'` first.")

        # Embed each data sample
        data_embeddings = [self.data_encoder(sample) for sample in data_samples]
        data_embeddings_array = np.vstack(data_embeddings)

        # Query the stored text label index
        results = self._query_faiss_index(data_embeddings_array, self.index, labels=self.labels, k=k)
        return results


'''
    @staticmethod
    def _build_faiss_index(ref_array):
        """
        Builds a FAISS index for cosine similarity search from a given array of embeddings.

        Parameters
        ----------
        ref_array : numpy.ndarray
            An array of embeddings to be indexed.

        Returns
        -------
        faiss.IndexFlatIP
            A FAISS index object ready for similarity searching.
        """
        # Normalize the embeddings to unit length for cosine similarity
        logger.debug("Normalizing reference array for FAISS index.")
        normalized_ref_array = ref_array / np.linalg.norm(ref_array, axis=1, keepdims=True)

        # Define the dimension of the embeddings
        dimension = normalized_ref_array.shape[1]

        # Create a Faiss index for inner product (equivalent to cosine similarity when inputs are normalized)
        logger.debug("Creating FAISS IndexFlatIP for dimension %d.", dimension)
        index = faiss.IndexFlatIP(dimension)

        # Convert embeddings to float32 as required by Faiss, then add them to the index
        logger.debug("Adding normalized embeddings to index.")
        index.add(normalized_ref_array.astype("float32"))

        return index

    @staticmethod
    def _query_faiss_index(query_array, index, labels, k=1):
        """
        Finds the most similar vectors to the input query array using a FAISS index.

        Parameters
        ----------
        query_array : numpy.ndarray
            An array of floats for which similar items are to be found. Shape should be
            (n_query, embedding_dim).
        index : faiss.Index
            Pre-built FAISS index of reference embeddings for searching.
        labels : list
            List of labels corresponding to the embeddings in the FAISS index.
        k : int, optional
            Number of top similar items to return for each query embedding. Default is 1.

        Returns
        -------
        list of lists
            A list of lists, where each inner list contains tuples of
            (label, similarity_score) for a single query embedding.
        """
        if len(query_array.shape) == 1:
            query_array = query_array.reshape(1, -1)

        logger.debug("Normalizing query array for cosine similarity.")
        # Normalize the input embeddings to unit length for cosine similarity calculation
        norms = np.linalg.norm(query_array, axis=1, keepdims=True)
        query_array_normalized = query_array / norms
        query_array_normalized = query_array_normalized.astype("float32")

        # Faiss returns distances (inner-product scores) and indices of the k nearest neighbors
        distances, indices = index.search(query_array_normalized, k)
        logger.debug("FAISS search complete. Shape of distances: %s, indices: %s", distances.shape, indices.shape)

        # Collect the labels and distances of the most similar items for each query embedding
        all_similar_items = []
        for idx_batch, dist_batch in zip(indices, distances, strict=False):
            similar_items = []
            for idx, dist in zip(idx_batch, dist_batch, strict=False):
                # Check if the index is within the valid range
                if idx < len(labels):
                    similar_items.append((labels[idx], float(dist)))
                else:
                    logger.warning("Index %d out of bounds for labels with size %d", idx, len(labels))
            all_similar_items.append(similar_items)

        return all_similar_items
'''
