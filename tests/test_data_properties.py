'''
import logging

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from mmcontext.eval import DataProperties

# Import the DataProperties class
# Adjust the import statement based on your project structure
# from your_module import DataProperties

# Set up logging
logger = logging.getLogger(__name__)


def test_compute_properties_returns_all_properties():
    """
    Test that compute_properties returns all requested properties.
    """
    logger.info("TEST: test_compute_properties_returns_all_properties")

    # Create a simple numpy array
    data = np.random.rand(100, 50)

    # Create an instance of DataProperties with predefined_subset='all'
    data_properties = DataProperties(predefined_subset="all")

    # Compute properties
    result = data_properties.compute_properties(data)

    # Get the list of properties that should be computed
    expected_properties = data_properties.subset
    # Check that all expected properties are in the result
    missing_properties = [prop for prop in expected_properties if prop not in result]
    assert not missing_properties, f"Properties missing in result: {missing_properties}"


def test_compare_data_properties_returns_dataframe_with_all_properties():
    """
    Test that compare_data_properties returns a DataFrame with all properties.
    """
    logger.info("TEST: test_compare_data_properties_returns_dataframe_with_all_properties")

    # Create original and reconstructed data
    original_data = np.random.rand(100, 50)
    reconstructed_data = np.random.rand(100, 50)
    # Create an instance of DataProperties with predefined_subset='all'
    data_properties = DataProperties(predefined_subset="all")

    # Add original and reconstructed data
    data_properties.add_original_data(original_data)
    data_properties.add_reconstructed_data(reconstructed_data)

    # Compare data properties
    comparison_df = data_properties.compare_data_properties()

    # Get the list of properties that should be compared
    expected_properties = data_properties.subset

    # Expected columns in the comparison DataFrame
    expected_columns = ["id", "recon_id"] + expected_properties + ["meanLog2FC"]

    # Check that all expected columns are present
    missing_columns = [col for col in expected_columns if col not in comparison_df.columns]
    assert not missing_columns, f"Columns missing in comparison DataFrame: {missing_columns}"


def test_reconstructed_data_compared_only_to_its_original_data():
    """
    Test that reconstructed data is only compared to its corresponding original data.
    """
    logger.info("TEST: test_reconstructed_data_compared_only_to_its_original_data")

    # Create original data sets
    original_data1 = np.random.rand(100, 50)
    original_data2 = np.random.rand(100, 50)

    # Create reconstructed data sets
    reconstructed_data1 = np.random.rand(100, 50)
    reconstructed_data2 = np.random.rand(100, 50)

    # Create an instance of DataProperties
    data_properties = DataProperties(predefined_subset="all")

    # Add original data 1 and its reconstruction
    data_properties.add_original_data(original_data1, id="original1")
    data_properties.add_reconstructed_data(reconstructed_data1, id="recon1")

    # Add original data 2 and its reconstruction
    data_properties.add_original_data(original_data2, id="original2")
    data_properties.add_reconstructed_data(reconstructed_data2, id="recon2")

    # Compare data properties
    comparison_df = data_properties.compare_data_properties()

    # Check that reconstructions are only compared within the same 'id'
    for original_id, recon_id in [("original1", "recon1"), ("original2", "recon2")]:
        df_subset = comparison_df[comparison_df["id"] == original_id]
        recon_ids = df_subset["recon_id"].unique()
        assert set(recon_ids) == {recon_id}, (
            f"Reconstructed data for {original_id} includes unexpected reconstructions: {recon_ids}"
        )


def test_compute_properties_with_different_data_types():
    """
    Test compute_properties with different input data types (numpy array, sparse matrix, DataFrame, AnnData object).
    """
    logger.info("TEST: test_compute_properties_with_different_data_types")

    # Create data in different formats
    data_numpy = np.random.rand(100, 50)
    data_sparse = sp.csr_matrix(data_numpy)
    data_df = pd.DataFrame(data_numpy)

    # For AnnData, import anndata
    import anndata

    data_adata = anndata.AnnData(X=data_numpy)

    # Create an instance of DataProperties
    data_properties = DataProperties(predefined_subset="all")

    def test_data_input(data_input, data_type):
        with pytest.raises(ValueError) as exc_info:
            data_properties.compute_properties(data_input)
        assert "must be a numpy array" in str(exc_info.value), f"Expected ValueError for {data_type}"

    # Test with sparse matrix
    test_data_input(data_sparse, "sparse matrix")

    # Test with DataFrame
    test_data_input(data_df, "DataFrame")

    # Test with AnnData object
    test_data_input(data_adata, "AnnData object")


def test_compute_properties_handles_missing_methods():
    """
    Test that compute_properties handles missing 'get_' methods gracefully.
    """
    logger.info("TEST: test_compute_properties_handles_missing_methods")

    # Create data
    data = np.random.rand(100, 50)

    # Create an instance with a custom subset including a non-existent property
    data_properties = DataProperties(custom_subset=["nFeatures", "nonExistentProperty"])

    # Compute properties
    result = data_properties.compute_properties(data)

    # Check that 'nFeatures' is in the result
    assert "nFeatures" in result, "Expected 'nFeatures' to be in result"

    # Check that 'nonExistentProperty' is not in the result (or is NaN)
    assert "nonExistentProperty" not in result or np.isnan(result.get("nonExistentProperty", np.nan)), (
        "Expected 'nonExistentProperty' to be missing or NaN in result"
    )


def test_compute_properties_handles_exceptions_in_methods():
    """
    Test that compute_properties handles exceptions in 'get_' methods gracefully.
    """
    logger.info("TEST: test_compute_properties_handles_exceptions_in_methods")

    # Create data
    data = np.random.rand(100, 50)

    # Create a subclass that raises an exception in one of the 'get_' methods
    class DataPropertiesWithError(DataProperties):
        def get_errorProperty(self):
            raise ValueError("Intentional error in get_errorProperty")

    # Create an instance with the custom property
    data_properties = DataPropertiesWithError(custom_subset=["nFeatures", "errorProperty"])

    # Compute properties
    result = data_properties.compute_properties(data)

    # Check that 'nFeatures' is in the result
    assert "nFeatures" in result, "Expected 'nFeatures' to be in result"

    # Check that 'errorProperty' is in the result and is NaN
    assert "errorProperty" in result, "Expected 'errorProperty' to be in result"
    assert np.isnan(result["errorProperty"]), "Expected 'errorProperty' to be NaN due to exception"


def test_add_reconstructed_data_without_original_data():
    """
    Test that adding reconstructed data without prior original data raises a ValueError.
    """
    logger.info("TEST: test_add_reconstructed_data_without_original_data")

    # Create reconstructed data
    reconstructed_data = np.random.rand(100, 50)

    # Create an instance of DataProperties
    data_properties = DataProperties()

    # Attempt to add reconstructed data without original data
    with pytest.raises(ValueError) as exc_info:
        data_properties.add_reconstructed_data(reconstructed_data)

    assert "Please add an original dataset before adding reconstructed datasets." in str(exc_info.value), (
        "Expected ValueError when adding reconstructed data without original data"
    )


def test_plot_metrics_runs_without_errors(tmp_path):
    """
    Test that plot_metrics runs without errors.
    """
    logger.info("TEST: test_plot_metrics_runs_without_errors")

    # Create original and reconstructed data
    original_data = np.random.rand(100, 50)
    reconstructed_data = np.random.rand(100, 50)

    # Create an instance of DataProperties
    data_properties = DataProperties()

    # Add data
    data_properties.add_original_data(original_data)
    data_properties.add_reconstructed_data(reconstructed_data)

    # Compare data properties
    data_properties.compare_data_properties()

    # Test plot_metrics
    try:
        data_properties.plot_metrics(save_dir=tmp_path)
    except Exception as e:
        pytest.fail(f"plot_metrics raised an exception: {e}")


def test_plot_pca_runs_without_errors(tmp_path):
    """
    Test that plot_pca runs without errors.
    """
    logger.info("TEST: test_plot_pca_runs_without_errors")

    # Create original and reconstructed data
    original_data = np.random.rand(100, 50)
    reconstructed_data = np.random.rand(100, 50)

    # Create an instance of DataProperties
    data_properties = DataProperties()

    # Add data
    data_properties.add_original_data(original_data)
    data_properties.add_reconstructed_data(reconstructed_data)

    # Test plot_pca
    try:
        data_properties.plot_pca(save_path=f"{tmp_path}/pca_plot.png")
    except Exception as e:
        pytest.fail(f"plot_pca raised an exception: {e}")
'''
