import logging
from unittest.mock import MagicMock

import anndata
import numpy as np
import pandas as pd
import pytest

# If your class is in a separate module, you'd import from there, e.g.:
# from my_package.my_module import AnnDataDataSetConstructor

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_caption_constructor():
    """
    Pytest fixture that returns a mock caption constructor.

    The mock simply sets adata.obs["caption"] to be a string derived
    from the obs_names, for demonstration.
    """
    constructor = MagicMock()

    def construct_captions(adata):
        """Simulated construction of captions, stored in adata.obs['caption'].
        For demonstration, each sample gets a caption 'caption_{sample_id}'."""
        captions = [f"caption_{idx}" for idx in adata.obs_names]
        adata.obs["caption"] = captions

    constructor.construct_captions.side_effect = construct_captions
    return constructor


@pytest.fixture
def ann_data_file_1(tmp_path):
    """
    Create a small anndata object and write it to a .zarr file for testing.

    Returns
    -------
    str
        The path to the .zarr file containing the test anndata object.
    """
    # Create a small adata
    obs_data = pd.DataFrame(index=[f"S{i}" for i in range(3)])
    var_data = pd.DataFrame(index=[f"G{i}" for i in range(5)])
    X = np.random.rand(3, 5)
    adata = anndata.AnnData(X=X, obs=obs_data, var=var_data)

    file_path = str(tmp_path / "test1.zarr")
    adata.write_zarr(file_path)
    return file_path


@pytest.fixture
def ann_data_file_2(tmp_path):
    """
    Create another small anndata object and write it to a .zarr file for testing.

    Returns
    -------
    str
        The path to the second .zarr file containing the test anndata object.
    """
    obs_data = pd.DataFrame(index=[f"S{i}" for i in range(2)])
    var_data = pd.DataFrame(index=[f"G{i}" for i in range(4)])
    X = np.random.rand(2, 4)
    adata = anndata.AnnData(X=X, obs=obs_data, var=var_data)

    file_path = str(tmp_path / "test2.zarr")
    adata.write_zarr(file_path)
    return file_path


@pytest.fixture
def ann_data_file_h5ad(tmp_path):
    """
    Create a small anndata object and write it to an .h5ad file for testing.

    Returns
    -------
    str
        The path to the .h5ad file containing the test anndata object.
    """
    # Create a small adata
    obs_data = pd.DataFrame(index=[f"S{i}" for i in range(3)])
    var_data = pd.DataFrame(index=[f"G{i}" for i in range(5)])
    X = np.random.rand(3, 5)
    adata = anndata.AnnData(X=X, obs=obs_data, var=var_data)

    file_path = str(tmp_path / "test_h5ad_input.h5ad")
    adata.write_h5ad(file_path)
    return file_path


@pytest.fixture
def dataset_constructor(mock_caption_constructor):
    """
    Fixture to instantiate the AnnDataDataSetConstructor with a mocked caption constructor.

    Returns
    -------
    AnnDataDataSetConstructor
        A fresh instance for testing.
    """
    from mmcontext.pp import AnnDataDataSetConstructor

    return AnnDataDataSetConstructor(caption_constructor=mock_caption_constructor, negatives_per_sample=1)


def test_add_anndata_success(dataset_constructor, ann_data_file_1, ann_data_file_2):
    """
    Test that we can successfully add distinct anndata files without error.

    References
    ----------
    Simulated anndata from the 'ann_data_file_1' and 'ann_data_file_2' fixtures.
    """
    logger.info("Testing adding anndata files to the constructor.")

    dataset_constructor.add_anndata(ann_data_file_1)
    dataset_constructor.add_anndata(ann_data_file_2)

    assert len(dataset_constructor.anndata_files) == 2
    assert ann_data_file_1 in dataset_constructor.anndata_files
    assert ann_data_file_2 in dataset_constructor.anndata_files


def test_add_anndata_duplicate(dataset_constructor, ann_data_file_1):
    """
    Test that adding the same file twice raises a ValueError.

    References
    ----------
    Simulated anndata from the 'ann_data_file_1' fixture.
    """
    logger.info("Testing duplicate anndata file addition.")

    dataset_constructor.add_anndata(ann_data_file_1)
    with pytest.raises(ValueError) as excinfo:
        dataset_constructor.add_anndata(ann_data_file_1)
    assert "has already been added" in str(excinfo.value)


def test_add_anndata_nonexistent_file(dataset_constructor):
    """
    Test that adding a non-existent file path raises an error when building the dataset.

    We do NOT raise the error in add_anndata itself, but rather rely on buildCaption failing
    or anndata.read_zarr failing. If you'd like to fail earlier, you could check existence
    in add_anndata.

    References
    ----------
    No real file provided, a dummy path is used to test error handling.
    """
    from zarr.errors import PathNotFoundError

    logger.info("Testing adding a non-existent anndata file path.")

    fake_path = "some/non_existent_file.zarr"
    dataset_constructor.add_anndata(fake_path)
    with pytest.raises(PathNotFoundError):
        # Attempting to get the dataset should fail when it tries to read a non-existent file
        dataset_constructor.get_dataset()


def test_no_caption_constructor(ann_data_file_1):
    """
    Test that if no caption constructor is provided, buildCaption might fail or we skip building captions.

    References
    ----------
    Simulated anndata from the 'ann_data_file_1' fixture.
    """
    logger.info("Testing behavior with no caption constructor provided.")

    from mmcontext.pp import AnnDataDataSetConstructor

    # Create constructor without a caption_constructor
    constructor = AnnDataDataSetConstructor(caption_constructor=None, negatives_per_sample=1)
    constructor.add_anndata(ann_data_file_1)

    # If your code does not handle 'None' and tries to call 'construct_captions', it should fail
    with pytest.raises(AttributeError):
        constructor.get_dataset()


def test_caption_constructor_fail(dataset_constructor, ann_data_file_1):
    """
    Test that if the provided caption constructor fails, we catch and raise the exception.

    References
    ----------
    Simulated anndata from the 'ann_data_file_1' fixture.
    """
    logger.info("Testing failing caption constructor scenario.")

    # Force the mock to raise an exception
    def fail_construct_captions(adata):
        raise RuntimeError("Caption constructor error.")

    dataset_constructor.caption_constructor.construct_captions.side_effect = fail_construct_captions

    dataset_constructor.add_anndata(ann_data_file_1)

    with pytest.raises(RuntimeError) as excinfo:
        dataset_constructor.get_dataset()
    assert "Caption constructor error" in str(excinfo.value)


def test_get_dataset_positive_and_negative(dataset_constructor, ann_data_file_1, ann_data_file_2):
    """
    Test that the final dataset includes the correct positive and negative examples.

    Specifically:
    - Positive samples have label=1.0, with matching caption and sample_id
    - Negative samples have label=0.0, with different caption content

    References
    ----------
    Simulated anndata from the 'ann_data_file_1' and 'ann_data_file_2' fixtures.
    """
    logger.info("Testing construction of positive and negative examples in the dataset.")

    # Add two files
    dataset_constructor.add_anndata(ann_data_file_1)
    dataset_constructor.add_anndata(ann_data_file_2)

    # Build dataset
    ds = dataset_constructor.get_dataset()

    # The number of samples in the first file is 3, second file is 2
    # Each sample gets 1 positive and 1 negative => total 3+2 = 5 samples => 10 InputExample entries
    assert len(ds) == 10

    # Check that each positive example has label=1.0 and correct text matches the caption
    pos_examples = [ex for ex in ds if ex.label == 1.0]
    neg_examples = [ex for ex in ds if ex.label == 0.0]

    assert len(pos_examples) == 5
    assert len(neg_examples) == 5

    # Positive example check
    for ex in pos_examples:
        # ex.texts[0] is metadata, ex.texts[1] is the caption
        metadata, caption = ex.texts
        file_path = metadata["file_path"]
        sample_id = metadata["sample_id"]

        # Re-read adata to ensure the caption is correct
        adata = anndata.read_zarr(file_path)
        assert "caption" in adata.obs.columns
        # Ensure the sample's caption matches the example's caption
        assert adata.obs.loc[sample_id, "caption"] == caption

    # Negative example check
    for ex in neg_examples:
        metadata, caption = ex.texts
        file_path = metadata["file_path"]
        sample_id = metadata["sample_id"]

        # Re-read adata to ensure the sample's caption is different
        adata = anndata.read_zarr(file_path)
        original_caption = adata.obs.loc[sample_id, "caption"]
        assert original_caption != caption, f"Negative example has the same caption as the original sample {sample_id}."


def test_clear_method(dataset_constructor, ann_data_file_1):
    """
    Test that the clear method resets all internal data structures.

    References
    ----------
    Simulated anndata from the 'ann_data_file_1' fixture.
    """
    logger.info("Testing the clear method of the constructor.")

    dataset_constructor.add_anndata(ann_data_file_1)
    dataset_constructor.clear()

    assert len(dataset_constructor.anndata_files) == 0
    assert len(dataset_constructor.sample_id_keys) == 0
    assert len(dataset_constructor.dataset) == 0


def test_add_anndata_h5ad_file(dataset_constructor, ann_data_file_h5ad):
    """
    Test behavior when an .h5ad file is provided instead of a .zarr file.
    If the constructor strictly uses anndata.read_zarr(...),
    we expect it to fail with an error related to file format or path.
    """
    logger.info("Testing behavior when an .h5ad file is passed (but code uses read_zarr).")

    with pytest.raises(ValueError) as excinfo:
        dataset_constructor.add_anndata(ann_data_file_h5ad)

    # Optionally, you can inspect the exception message:
    err_msg = str(excinfo.value)
    assert ".zarr format" in err_msg.lower(), f"Unexpected error message: {err_msg}"
