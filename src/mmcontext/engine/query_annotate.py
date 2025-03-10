import logging

import numpy as np

logger = logging.getLogger(__name__)


class OmicsQueryAnnotator:
    """Class to Annotate and Query omics data using text labels.

    A class to handle two main modes of operation:
    1) Annotating omics data with text labels,
    2) Querying omics data from textual descriptions.

    This class uses matrix multiplication for similarity computations.

    Attributes
    ----------
    model : SentenceTransformer or similar
        A model providing `encode(list_of_strings) -> np.ndarray`.
    embeddings : np.ndarray
        Embeddings currently stored in the class, either label embeddings
        or sample embeddings, depending on the mode.
    labels_ : List[str]
        A list storing the labels that were used for annotation.
    sample_ids_ : List[str]
        A list storing sample IDs for the omics data.
    is_cosine : bool
        Whether to perform L2 normalization and treat the similarity
        as cosine similarity.
    """

    def __init__(self, model, is_cosine=True):
        """
        Initialize the OmicsQueryAnnotator.

        Parameters
        ----------
        model : object
            A model with an `.encode(list_of_strings: List[str]) -> np.ndarray` method.
        is_cosine : bool, optional
            If True, vectors are normalized for cosine similarity computation.
        """
        self.model = model
        self.embeddings = None
        self.labels_ = []
        self.sample_ids_ = []
        self.is_cosine = is_cosine

    def annotate_omics_data(
        self,
        adata,
        labels,
        emb_key="mmcontext_emb",
        n_top=5,
        text_template: str = "{}",
    ):
        """
        Annotate omics data by finding top-matching labels for each sample.

        The scores and the labels are stored in `adata.obs["inferred_labels"]`.
        The single best label per sample is stored in `adata.obs["best_label"]`.

        Parameters
        ----------
        adata : anndata.AnnData
            Anndata object containing omics data. We expect `adata.obsm[emb_key]`
            to hold the omics embeddings (dim: n_samples x embed_dim).
        labels : List[str]
            A list of text labels to use for annotation.
        emb_key : str, optional
            Key in adata.obsm containing the embeddings to use, by default "mmcontext_emb"
        n_top : int, optional
            Number of top matching labels to retrieve per sample.
        text_template : str, optional
            Template string to format labels, by default "{}"
        """
        if emb_key not in adata.obsm:
            raise ValueError(
                f"`adata.obsm['{emb_key}']` not found. Provide the correct key to the embedding "
                "created with an mmcontext model. Make sure you use the same, pre-trained model "
                "as you provide to this class."
            )

        # Format labels with the text template
        labels_to_encode = [text_template.format(lbl) for lbl in labels]

        logger.info(f"Encoding {len(labels)} labels with template: '{text_template}'")
        label_emb = self.model.encode(labels_to_encode)

        # Get omics embeddings
        data_emb = adata.obsm[emb_key]

        # Apply L2 normalization if using cosine similarity
        if self.is_cosine:
            label_emb = self._l2_normalize(label_emb)
            data_emb = self._l2_normalize(data_emb)

        # Convert to float32 for better numerical stability
        label_emb = label_emb.astype(np.float32)
        data_emb = data_emb.astype(np.float32)

        # Check that dimensions match
        if data_emb.shape[1] != label_emb.shape[1]:
            raise ValueError(
                f"Dimensions of omics embeddings ({data_emb.shape[1]}) and "
                f"label embeddings ({label_emb.shape[1]}) do not match."
            )

        # Compute similarity matrix: (n_samples, n_labels)
        logger.info("Computing similarity between omics data and labels")
        similarity_matrix = data_emb @ label_emb.T

        # Store results
        label_dicts = []
        best_label_list = []

        # Process each sample
        for sample_idx in range(similarity_matrix.shape[0]):
            # Get similarities for this sample
            sample_similarities = similarity_matrix[sample_idx]

            # Create dictionary of label -> score
            sample_label_scores = {}
            best_label = None
            best_score = float("-inf")

            # Find top scores and best label
            for label_idx, score in enumerate(sample_similarities):
                label_name = labels[label_idx]
                float_score = float(score)
                sample_label_scores[label_name] = float_score

                if float_score > best_score:
                    best_score = float_score
                    best_label = label_name

            label_dicts.append(sample_label_scores)
            best_label_list.append(best_label)

        # Store the results in adata.obs
        adata.obs["inferred_labels"] = label_dicts
        adata.obs["best_label"] = best_label_list

        logger.info(f"Annotated {len(adata)} samples with {len(labels)} labels")

    def query_with_text(self, adata, queries, emb_key="mmcontext_emb"):
        """Query omics data with textual queries.

        Compare text query embeddings with omics data using matrix multiplication
        and store similarities in `adata.obs["query_scores"]`.

        Parameters
        ----------
        adata : anndata.AnnData
            The AnnData object whose samples we want to score against the queries.
        queries : List[str]
            A list of textual queries.
        emb_key : str, optional
            Key in adata.obsm containing the embeddings to use.
        """
        logger.info("Encoding the queries.")
        query_emb = self.model.encode(queries)

        if self.is_cosine:
            query_emb = self._l2_normalize(query_emb)

        query_emb = query_emb.astype(np.float32)

        logger.info("Computing similarity between queries and omics data.")
        if emb_key not in adata.obsm:
            raise ValueError("Omics embeddings not found in adata.obsm.")

        data_emb = adata.obsm[emb_key]
        if self.is_cosine:
            data_emb = self._l2_normalize(data_emb)

        # Compute similarity matrix
        similarity_matrix = query_emb @ data_emb.T

        # Initialize a list of dicts — one dict per sample
        sample_scores = [{} for _ in range(adata.n_obs)]

        # For each query
        for i, query_str in enumerate(queries):
            # For each sample
            for s in range(adata.n_obs):
                sample_scores[s][query_str] = float(similarity_matrix[i, s])

        adata.obs["query_scores"] = sample_scores

    @staticmethod
    def _l2_normalize(vectors):
        """L2-normalize each row in `vectors`."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-9)
