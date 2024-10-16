import os

from mmcontext.utils import create_test_anndata


def save_test_anndata():
    """
    Create and save test AnnData objects to specified files.

    This function generates two test AnnData objects using the `create_test_anndata` function.
    The first AnnData object contains 20 cells and 100 genes, with cell types including "B cell",
    "T cell", and "NK cell", and tissue types of "blood" and "lymph". This object is saved as
    'test_adata.h5ad' in the 'data/test_data' directory.

    The second AnnData object contains 25 cells and 100 genes, with additional cell types such as
    "Dendritic cell" and "Monocyte", and more tissue types, including "bone marrow" and "brain".
    This object is saved as 'new_test_adata.h5ad' in the same directory.

    The function ensures the 'data/test_data' directory exists before saving the files.

    Example:
        >>> save_test_anndata()
        Test AnnData saved to 'data/test_adata.h5ad'.
        Further test AnnData with extra categories saved to 'data/new_test_adata.h5ad'.

    Raises
    ------
        OSError: If there is an issue with file or directory creation.

    """
    # Create the test AnnData object
    adata = create_test_anndata(
        n_cells=20, n_genes=100, cell_types=["B cell", "T cell", "NK cell"], tissues=["blood", "lymph"]
    )

    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)

    # Save the AnnData object to a file
    adata.write("data/test_data/test_adata.h5ad")
    print("Test AnnData saved to 'data/test_adata.h5ad'.")

    adata = create_test_anndata(
        n_cells=25,
        n_genes=100,
        cell_types=["B cell", "T cell", "NK cell", "Dendritic cell", "Monocyte"],  # New cell types
        tissues=["blood", "lymph", "bone marrow", "brain"],  # New tissue types
    )

    # Save the new AnnData object
    adata.write("data/test_data/new_test_adata.h5ad")
    print("Further test AnnData with extra categories saved to 'data/new_test_adata.h5ad'.")


# Run the function to save the dataset
if __name__ == "__main__":
    save_test_anndata()
