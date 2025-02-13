import logging
import random

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_emb_pair_dataframe(dataset_split, model, subset_size=100, seed=42):
    """
    Create a small subset dataframe for paired omics-text embeddings.

    Parameters
    ----------
    dataset_split : list or Dataset
        A Hugging Face dataset split (e.g. dataset["train"]) with the columns
        "anndata_ref", "caption", and "label".
        Data source: Hugging Face dataset containing pairs of omics and text data.
    model : sentence_transformers.SentenceTransformer
        A model (or similar) with an .encode method for generating embeddings.
        Make sure it can handle both text (str) and omics inputs (e.g., preprocessed data).
    subset_size : int, optional
        How many positive pairs to sample. By default 100.
    seed : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'embedding': The embedding vector as a list of floats
        - 'embedding_type': Either 'omics' or 'text'
        - 'pair': An integer pairing ID that groups each omics-text pair

    Notes
    -----
    1. We filter the dataset to use only examples where label == 1 (true pairs).
    2. For each pair, we generate two rows in the resulting DataFrame:
       one for the omics embedding, one for the text embedding.
    3. 'subset_size' is used to avoid embedding a huge dataset. If your dataset
       has fewer than 'subset_size' positive pairs, you'll just get all of them.
    4. If your 'anndata_ref' is not directly compatible with `model.encode()`,
       you should adapt the code to handle your omics embedding step
       (e.g., loading an .h5ad file, calling a separate pipeline, etc.).
    """
    # Filter dataset for the correct pairs only
    logger.info("Filtering dataset for label == 1 (true pairs).")
    positive_pairs = [row for row in dataset_split if row["label"] == 1]

    # Sample a subset if the dataset is large
    random.seed(seed)
    if len(positive_pairs) > subset_size:
        logger.info("Sampling %d out of %d positive pairs.", subset_size, len(positive_pairs))
        positive_pairs = random.sample(positive_pairs, k=subset_size)
    else:
        logger.info("Using all %d positive pairs (fewer than subset_size).", len(positive_pairs))

    data_rows = []
    # We create a pairing ID for each row in our subset
    for idx, row in enumerate(positive_pairs):
        # 1) Embed omics
        omics_emb = model.encode([row["anndata_ref"]])  # shape: (1, emb_dim)
        # 2) Embed text
        text_emb = model.encode([row["caption"]])  # shape: (1, emb_dim)

        # omics row
        data_rows.append(
            {
                "embedding": omics_emb[0].tolist(),
                "embedding_type": "omics",
                "pair": idx,  # pairing ID
            }
        )

        # text row
        data_rows.append({"embedding": text_emb[0].tolist(), "embedding_type": "text", "pair": idx})

    df = pd.DataFrame(data_rows)
    logger.info("Created a DataFrame with %d rows (2 per pair).", len(df))
    return df
