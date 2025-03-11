"""
Test suite for OmicsQueryAnnotator and related functionality.

We use pytest, so run these tests via:
   pytest tests/test_query_annotate.py
"""

import anndata
import numpy as np
import pytest
import torch

from mmcontext.engine import OmicsQueryAnnotator
from mmcontext.file_utils import compute_cosine_similarity

# -------------------------------
# Pytest fixtures
# -------------------------------


@pytest.fixture
def dummy_labels():
    """
    Returns a small list of strings representing text labels.
    """
    return ["cell_type_A", "cell_type_B", "cell_type_C"]


@pytest.fixture
def dummy_adata():
    """
    Returns a small AnnData object with random embeddings in .obsm["mmcontext_emb"].
    """
    n_obs = 5
    n_dim = 4
    np.random.seed(0)
    X = np.random.rand(n_obs, 10)  # just a random data matrix
    adata = anndata.AnnData(X)
    # Create random embeddings
    emb = np.random.rand(n_obs, n_dim)
    adata.obsm["mmcontext_emb"] = emb
    # Optionally set obs_names
    adata.obs_names = [f"sample_{i}" for i in range(n_obs)]
    return adata


@pytest.fixture
def dummy_model():
    """
    Returns a mock model object that has a .encode() method
    resembling a SentenceTransformer. We'll just do random vectors.
    """

    class MockModel:
        def encode(self, list_of_strings):
            # For simplicity, produce deterministic random vectors:
            rng = np.random.default_rng(seed=42)
            # Suppose embedding_dim=4
            n_dim = 4
            return rng.random((len(list_of_strings), n_dim))

    return MockModel()


# -------------------------------
# Tests for annotate_omics_data
# -------------------------------


def test_annotate_omics_data(dummy_model, dummy_labels, dummy_adata):
    """
    Test that annotate_omics_data adds 'inferred_labels' and 'best_label' to adata.obs.
    """
    oq = OmicsQueryAnnotator(model=dummy_model, is_cosine=True)
    oq.annotate_omics_data(dummy_adata, labels=dummy_labels)

    # We expect adata.obs["inferred_labels"] to exist
    assert "inferred_labels" in dummy_adata.obs
    assert "best_label" in dummy_adata.obs
    # Check length: it should match n_obs
    assert len(dummy_adata.obs["inferred_labels"]) == dummy_adata.n_obs
    assert len(dummy_adata.obs["best_label"]) == dummy_adata.n_obs

    # Each element in inferred_labels is a dict
    for d in dummy_adata.obs["inferred_labels"]:
        assert isinstance(d, dict)
        assert len(d) <= len(dummy_labels), "Should not have more scores than labels"

    # Each best_label should be one of the original labels
    for label in dummy_adata.obs["best_label"]:
        assert label in dummy_labels, f"Best label {label} not in original labels"


# -------------------------------
# Tests for query_with_text
# -------------------------------


def test_query_with_text(dummy_model, dummy_adata):
    """
    Check that query_with_text fills adata.obs["query_scores"] properly.
    """
    oq = OmicsQueryAnnotator(model=dummy_model, is_cosine=True)
    queries = ["some text", "some other text"]
    oq.query_with_text(dummy_adata, queries)

    assert "query_scores" in dummy_adata.obs
    assert len(dummy_adata.obs["query_scores"]) == dummy_adata.n_obs

    # Each element in query_scores should be a dict with entries for each query
    for d in dummy_adata.obs["query_scores"]:
        assert isinstance(d, dict)
        assert len(d) == len(queries), "Should have scores for all queries"
        for k, v in d.items():
            assert k in queries, f"Query {k} not in original queries"
            assert isinstance(v, float), "Scores should be floats"


# -------------------------------
# Tests for compute_cosine_similarity
# -------------------------------


def test_compute_cosine_similarity_basic():
    """
    Simple test to confirm correctness of compute_cosine_similarity on CPU.
    """
    # Example small vectors
    # shape: (n_samples, dim) = (3, 2)
    samples = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    # shape: (n_queries, dim) = (2, 2)
    queries = np.array([[1, 1], [0, 1]], dtype=np.float32)

    # Expected: normalize each row. Then do dot products
    # samples normalized -> s1=(1,0), s2=(0,1), s3=(1/sqrt(2),1/sqrt(2))
    # queries normalized -> q1=(1/sqrt(2),1/sqrt(2)), q2=(0,1)
    # q1 dot s1 = 1/sqrt(2)
    # q1 dot s2 = 1/sqrt(2)
    # q1 dot s3 =  (1/sqrt(2)*1/sqrt(2))+(1/sqrt(2)*1/sqrt(2))=1
    # q2 dot s1 = 0
    # q2 dot s2 = 1
    # q2 dot s3 = 1/sqrt(2)
    # So final matrix = [[0.707..., 0.707..., 1.0],
    #                    [0.0,      1.0,      0.707...]]

    sim = compute_cosine_similarity(samples, queries, device="cpu")
    assert sim.shape == (2, 3), "Output shape should be (n_queries, n_samples)."

    # We'll check approximate correctness
    assert pytest.approx(sim[0, 0], 0.001) == 1 / np.sqrt(2)
    assert pytest.approx(sim[0, 1], 0.001) == 1 / np.sqrt(2)
    assert pytest.approx(sim[0, 2], 0.001) == 1.0
    assert pytest.approx(sim[1, 0], 0.001) == 0.0
    assert pytest.approx(sim[1, 1], 0.001) == 1.0
    assert pytest.approx(sim[1, 2], 0.001) == 1 / np.sqrt(2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compute_cosine_similarity_gpu():
    """
    Test compute_cosine_similarity on a CUDA GPU if available.
    """
    samples = np.random.rand(5, 4).astype(np.float32)
    queries = np.random.rand(3, 4).astype(np.float32)

    sim_cpu = compute_cosine_similarity(samples, queries, device="cpu")
    sim_gpu = compute_cosine_similarity(samples, queries, device="cuda")

    # They should be almost identical
    np.testing.assert_allclose(sim_cpu, sim_gpu, atol=1e-5)


@pytest.mark.skipif(
    not getattr(torch.backends, "mps", False) or not torch.backends.mps.is_available(),
    reason="MPS device not available",
)
def test_compute_cosine_similarity_mps():
    """
    Test compute_cosine_similarity on Apple Silicon MPS device if available.
    """
    samples = np.random.rand(5, 4).astype(np.float32)
    queries = np.random.rand(3, 4).astype(np.float32)

    sim_cpu = compute_cosine_similarity(samples, queries, device="cpu")
    sim_mps = compute_cosine_similarity(samples, queries, device="mps")

    np.testing.assert_allclose(sim_cpu, sim_mps, atol=1e-5)


def test_best_label_annotation():
    """
    Test that annotate_omics_data adds a 'best_label' column in adata.obs,
    and that the best_label actually matches the highest-scoring label in
    adata.obs["inferred_labels"].
    """

    # Create a deterministic mock model
    class DeterministicMockModel:
        def encode(self, list_of_strings):
            """
            We'll encode sample embeddings and label embeddings in a way that
            ensures a specific label is guaranteed to be the highest-scoring.
            """
            embeddings = []
            for s in list_of_strings:
                if s.startswith("sample_"):
                    idx = int(s.split("_")[-1])
                    vec = np.zeros(3, dtype=np.float32)
                    if idx < 3:
                        vec[idx] = 1.0
                    embeddings.append(vec)
                elif s.startswith("label_"):
                    idx = int(s.split("_")[-1])
                    vec = np.zeros(3, dtype=np.float32)
                    vec[idx] = 1.0
                    embeddings.append(vec)
                else:
                    embeddings.append(np.zeros(3, dtype=np.float32))
            return np.array(embeddings, dtype=np.float32)

    # Create test data
    X = np.random.rand(2, 5).astype(np.float32)
    adata = anndata.AnnData(X)
    adata.obs_names = ["sample_0", "sample_1"]
    mock_model = DeterministicMockModel()
    sample_emb = mock_model.encode(adata.obs_names.to_list())
    adata.obsm["mmcontext_emb"] = sample_emb

    # Create labels and run annotation
    labels = ["label_0", "label_1", "label_2"]
    oq = OmicsQueryAnnotator(model=mock_model, is_cosine=True)
    oq.annotate_omics_data(adata, labels=labels)

    # Check results
    assert "inferred_labels" in adata.obs
    assert "best_label" in adata.obs

    inferred_labels = adata.obs["inferred_labels"]
    best_labels = adata.obs["best_label"]

    # Verify best_label matches highest score in inferred_labels
    for i in range(2):
        label_scores = inferred_labels[i]
        found_best = best_labels[i]
        assert found_best in label_scores
        max_label = max(label_scores, key=label_scores.get)
        assert found_best == max_label

    # Verify expected behavior based on our encoding scheme
    assert best_labels[0] == "label_0"
    assert best_labels[1] == "label_1"
