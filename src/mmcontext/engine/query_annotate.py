import logging

import numpy as np

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from mmcontext.utils import compute_cosine_similarity

logger = logging.getLogger(__name__)


class OmicsQueryAnnotator:
    """Class to Annotate and Query omics data using text labels.

    A class to handle two main modes of operation:
    1) Annotating omics data with text labels,
    2) Querying omics data from textual descriptions.

    This class uses Faiss indices for efficient similarity lookups.

    Attributes
    ----------
    model : SentenceTransformer or similar
        A model providing `encode(list_of_strings) -> np.ndarray`.
    faiss_index : faiss.Index
        An optional Faiss index for fast similarity lookups.
    embeddings : np.ndarray
        Embeddings currently stored in the class, either label embeddings
        or sample embeddings, depending on the mode.
    labels_ : List[str]
        A list storing the labels that were used to build the Faiss index
        (valid in "label annotation" mode).
    sample_ids_ : List[str]
        A list storing sample IDs for the omics index (valid in "querying" mode).
    is_cosine : bool
        Whether to perform L2 normalization and treat the similarity
        as cosine. If True, uses inner-product (IP) distance on normalized vectors.
    """

    def __init__(self, model, is_cosine=True):
        """
        Initialize the OmicsQueryAnnotator.

        Parameters
        ----------
        model : object
            A model with an `.encode(list_of_strings: List[str]) -> np.ndarray` method.
        is_cosine : bool, optional
            If True, vectors are normalized and we use inner-product for
            similarity (equivalent to cosine similarity on normalized vectors).
        """
        self.model = model
        self.faiss_index = None
        self.embeddings = None
        self.labels_ = []
        self.sample_ids_ = []
        self.is_cosine = is_cosine

    def build_label_index(self, labels):
        """
        Build a Faiss index for text labels.

        Parameters
        ----------
        labels : List[str]
            A list of text labels to be encoded and indexed.
        """
        if not FAISS_AVAILABLE:
            raise ImportError("Faiss is not installed. Please install faiss-cpu or faiss-gpu depending on your system.")
        logger.info("Encoding label text.")
        label_emb = self.model.encode(labels)

        # Store labels so we can map retrieval indices back
        self.labels_ = labels

        # L2-normalize if we're using cosine
        if self.is_cosine:
            label_emb = self._l2_normalize(label_emb)

        self.embeddings = label_emb.astype(np.float32)

        logger.info("Building Faiss index for label embeddings.")
        dim = label_emb.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings)

    def annotate_omics_data(self, adata, labels=None, use_faiss=False, device="cpu", n_top=5):
        """
        Annotate omics data by finding top-matching labels for each sample.

        The scores and the labels are stored in `adata.obs["inferred_labels"]`.
        The single best label per sample is stored in `adata.obs["best_label"]`.

        Parameters
        ----------
        adata : anndata.AnnData
            Anndata object containing omics data. We expect `adata.obsm["omics_emb"]`
            to hold the omics embeddings (dim: n_samples x embed_dim).
        n_top : int, optional
            Number of top matching labels to retrieve per sample.
        labels : List[str], optional
            A list of text labels to use for annotation. If None, the faiss index has to be built first.
        use_faiss : bool, optional
            Whether to use the Faiss index for efficient retrieval. Default is False,
            and matrix multiplication is performed. Only needed for very large datasets.
        device : str, optional
            The device on which to run the computation. One of ["cpu", "cuda", "mps"].

        Notes
        -----
        The source of the omics data is adata.obsm["omics_emb"] as provided by
        some upstream pipeline.
        """
        if "omics_emb" not in adata.obsm:
            raise ValueError(
                "`adata.obsm['omics_emb']` not found. Use MMContextInference first. "
                "Make sure you use the same, pre-trained model as you provide to this class."
            )
        if not use_faiss and labels is None:
            raise ValueError("Labels must be provided if not using Faiss index.")
        if use_faiss and self.faiss_index is None:
            self.build_label_index(labels)

        data_emb = adata.obsm["omics_emb"]
        if self.is_cosine:
            data_emb = self._l2_normalize(data_emb)
        data_emb = data_emb.astype(np.float32)

        # We will store dictionaries of all label scores AND the single best label.
        label_dicts = []
        best_label_list = []

        if use_faiss:
            logger.info("Querying label index for each sample using Faiss.")
            # Retrieve top-n labels for each sample
            distances, indices = self.faiss_index.search(data_emb, n_top)

            for dist_row, idx_row in zip(distances, indices, strict=False):
                # Build a dict from label -> similarity
                sample_label_scores = {
                    self.labels_[idx]: float(dist) for idx, dist in zip(idx_row, dist_row, strict=False)
                }
                label_dicts.append(sample_label_scores)

                # Identify the single best label from the top-n
                best_label = None
                best_score = float("-inf")
                for lbl, score in sample_label_scores.items():
                    if score > best_score:
                        best_score = score
                        best_label = lbl
                best_label_list.append(best_label)

        else:
            # use matrix multiplication to compute similarity
            logger.info("Using matrix multiplication to compute label similarities.")
            label_emb = self.model.encode(labels)
            if self.is_cosine:
                label_emb = self._l2_normalize(label_emb)
            label_emb = label_emb.astype(np.float32)

            # check that second dimension of data_emb and label_emb are the same
            if data_emb.shape[1] != label_emb.shape[1]:
                raise ValueError(
                    f"Dimensions of omics embeddings ({data_emb.shape[1]}) and label embeddings ({label_emb.shape[1]}) do not match."
                )
            # shape: (n_labels, n_samples)
            similarity_matrix = compute_cosine_similarity(data_emb, label_emb, device=device)

            # For each sample row
            for col in similarity_matrix.T:
                # Build dict from label -> similarity
                sample_label_scores = {}
                best_label = None
                best_score = float("-inf")
                for lbl_idx, score in enumerate(col):
                    lbl_name = labels[lbl_idx]
                    float_score = float(score)
                    sample_label_scores[lbl_name] = float_score

                    if float_score > best_score:
                        best_score = float_score
                        best_label = lbl_name

                label_dicts.append(sample_label_scores)
                best_label_list.append(best_label)

        # Store the dictionary of inferred labels in obs
        adata.obs["inferred_labels"] = label_dicts
        adata.obs["best_label"] = best_label_list

    def build_omics_index(self, adata, sample_ids=None):
        """
        Build a Faiss index from the omics embeddings in `adata.obsm["omics_emb"]`.

        Parameters
        ----------
        adata : anndata.AnnData
            Anndata object containing omics data in `adata.obsm["omics_emb"]`.
        sample_ids : List[str], optional
            A parallel list of sample identifiers. If None, we'll default
            to adata.obs_names (row names in AnnData).

        Notes
        -----
        The source of the omics embeddings is `adata.obsm["omics_emb"]` which
        should already exist by the time this method is called.
        """
        if not FAISS_AVAILABLE:
            raise ImportError("Faiss is not installed. Please install faiss-cpu or faiss-gpu depending on your system.")
        if "omics_emb" not in adata.obsm:
            raise ValueError("`adata.obsm['omics_emb']` not found.")

        logger.info("Building Faiss index for omics embeddings.")
        data_emb = adata.obsm["omics_emb"]

        if self.is_cosine:
            data_emb = self._l2_normalize(data_emb)

        data_emb = data_emb.astype(np.float32)
        self.embeddings = data_emb
        # Store sample IDs so we can map retrieval indices back
        if sample_ids is not None:
            self.sample_ids_ = sample_ids
        else:
            self.sample_ids_ = list(adata.obs_names)

        dim = data_emb.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings)

    def query_with_text(self, adata, queries, use_faiss=False, device="cpu", n_top=5):
        """Query omics data with textual queries.

        Compare text query embeddings with omics index and store similarities
        in `adata.obs["query_scores"]`.

        Parameters
        ----------
        adata : anndata.AnnData
            The AnnData object whose samples we want to score against the queries.
        queries : List[str]
            A list of textual queries.
        n_top : int, optional
            Number of top matches to store for each sample (used if you
            want to store or interpret top matches).
        use_faiss : bool, optional
            Whether to use the Faiss index for efficient retrieval. Default is False, and matrix multiplication is performed.
            Only needed for very large datasets.
        device : str, optional
            The device on which to run the computation. One of ["cpu", "cuda"].

        Notes
        -----
        This method expects the faiss_index to be built from omics embeddings
        (e.g. via `build_omics_index`).
        """
        if use_faiss and not FAISS_AVAILABLE:
            raise ImportError("Faiss is not installed. Please install faiss-cpu or faiss-gpu depending on your system.")
        if use_faiss is True and self.faiss_index is None:
            raise ValueError("Omics Faiss index not built. Call `build_omics_index` first.")

        logger.info("Encoding the queries.")
        query_emb = self.model.encode(queries)

        if self.is_cosine:
            query_emb = self._l2_normalize(query_emb)

        query_emb = query_emb.astype(np.float32)
        if use_faiss:
            logger.info("Querying omics index for each textual query.")
            # We'll search for each query individually, or do them all at once if shape allows
            # Distances shape -> (num_queries, num_samples_in_index)
            sample_scores = []
            if n_top > adata.n_obs:
                n_top = adata.n_obs

            # Redo the search with n_top
            distances_top, indices_top = self.faiss_index.search(query_emb, n_top)

            # For each query, we have the top-n matches
            for j, query_str in enumerate(queries):
                for dist, idx in zip(distances_top[j], indices_top[j], strict=False):
                    # Append to that sample's dict
                    sample_scores[idx][query_str] = float(dist)
        else:
            logger.info("Computing cosine similarity between queries and omics data.")
            # Compute cosine similarity between queries and omics data
            if "omics_emb" not in adata.obsm:
                raise ValueError("Omics embeddings not found in adata.obsm.")
            data_emb = adata.obsm["omics_emb"]
            if self.is_cosine:
                data_emb = self._l2_normalize(data_emb)
            similarity_matrix = compute_cosine_similarity(data_emb, query_emb, device=device)
            sample_scores = []
            # queries: length = n_queries
            _n_queries, n_samples = similarity_matrix.shape

            # Initialize a list of dicts — one dict per sample
            sample_scores = [{} for _ in range(n_samples)]

            # For each query row i
            for i, query_str in enumerate(queries):
                # For each sample column s
                for s in range(n_samples):
                    sample_scores[s][query_str] = float(similarity_matrix[i, s])

        adata.obs["query_scores"] = sample_scores

    @staticmethod
    def _l2_normalize(vectors):
        """L2-normalize each row in `vectors`."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-9)
