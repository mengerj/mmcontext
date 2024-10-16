import gzip
import os
import pickle

import anndata

from mmcontext.pp.context_embedder import CategoryEmbedder


def generate_and_save_embeddings():
    """
    Load a test AnnData object, generate embeddings using CategoryEmbedder, and save the results.

    This function loads an AnnData object from 'data/test_data/test_adata.h5ad',
    generates embeddings using the CategoryEmbedder based on metadata categories like
    "cell_type" and "tissue", and saves the embedding dictionary to
    'data/emb_dicts/test_dict.pkl.gz'. It also optionally saves the updated AnnData object
    with the generated embeddings to 'data/test_adata_with_embeddings.h5ad'.

    Example:
        >>> generate_and_save_embeddings()
        Embeddings generated and saved. Updated AnnData saved to 'data/test_adata_with_embeddings.h5ad'.

    Raises
    ------
        OSError: If there are issues with file creation or reading/writing.
        pickle.PickleError: If there is an error while saving or loading the embeddings dictionary.
    """
    # Load the test AnnData object
    adata = anndata.read_h5ad("data/test_data/test_adata.h5ad")

    # Ensure the embeddings directory exists
    os.makedirs("./data/emb_dicts", exist_ok=True)

    # Set the embeddings file path
    embeddings_file_path = "./data/emb_dicts/test_dict.pkl.gz"

    # Initialize the CategoryEmbedder with the desired parameters
    metadata_categories = ["cell_type", "tissue"]
    context_embedder = CategoryEmbedder(
        metadata_categories=metadata_categories,
        embeddings_file_path=embeddings_file_path,
        model="text-embedding-3-small",
        combination_method="concatenate",
        one_hot=False,
        unknown_threshold=20,  # Set the threshold as desired
    )

    # Run the embedder to generate embeddings and save the dictionary
    context_embedder.embed(adata)

    # Optionally, save the updated AnnData object with embeddings
    adata.write("data/test_adata_with_embeddings.h5ad")
    print("Embeddings generated and saved. Updated AnnData saved to 'data/test_adata_with_embeddings.h5ad'.")


# Run the function to generate and save embeddings
if __name__ == "__main__":
    generate_and_save_embeddings()
    # check if the dictionary was saved
    with gzip.open("./data/emb_dicts/test_dict.pkl.gz", "rb") as f:
        emb_dict = pickle.load(f)
    # print all keys and total amount of values
    print(f"Dictionary keys:{emb_dict.keys()}, with total amount of values: {sum([len(v) for v in emb_dict.values()])}")
