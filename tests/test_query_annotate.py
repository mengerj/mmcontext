"""
Test suite for OmicsQueryAnnotator and related functionality.

We use pytest, so run these tests via:
   pytest tests/test_omics_query_annotator.py
"""

import anndata
import numpy as np
import pytest
import torch

from mmcontext.engine import OmicsQueryAnnotator
from mmcontext.utils import compute_cosine_similarity

# Check if faiss is available
try:
    import faiss

    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

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
# Tests for build_label_index / annotate_omics_data
# -------------------------------


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="Faiss is not available")
def test_build_label_index(dummy_model, dummy_labels):
    """
    Ensure that build_label_index stores the labels and creates a Faiss index with the right size.
    """
    oq = OmicsQueryAnnotator(model=dummy_model, is_cosine=True)
    oq.build_label_index(dummy_labels)

    assert oq.labels_ == dummy_labels, "Labels must be stored in the same order."
    assert oq.faiss_index is not None, "Faiss index should be built."
    assert oq.embeddings.shape == (len(dummy_labels), 4), "Embeddings shape should match (n_labels, embed_dim)."


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="Faiss is not available")
def test_annotate_omics_data(dummy_model, dummy_labels, dummy_adata):
    """
    Test that annotate_omics_data adds 'inferred_labels' to adata.obs.
    """
    oq = OmicsQueryAnnotator(model=dummy_model, is_cosine=True)
    oq.build_label_index(dummy_labels)
    oq.annotate_omics_data(dummy_adata, n_top=2, use_faiss=True)

    # We expect adata.obs["inferred_labels"] to exist
    assert "inferred_labels" in dummy_adata.obs
    # Check length: it should match n_obs
    assert len(dummy_adata.obs["inferred_labels"]) == dummy_adata.n_obs

    # Each element is a dict with up to n_top keys
    for d in dummy_adata.obs["inferred_labels"]:
        assert isinstance(d, dict)
        assert len(d) <= 2, "Expected up to 2 top labels per sample."


# -------------------------------
# Tests for build_omics_index / query_with_text
# -------------------------------


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="Faiss is not available")
def test_build_omics_index(dummy_model, dummy_adata):
    """
    Verify building an omics index creates a Faiss index with correct shape.
    """
    oq = OmicsQueryAnnotator(model=dummy_model, is_cosine=True)
    oq.build_omics_index(dummy_adata)

    assert oq.faiss_index is not None, "Faiss index should be built."
    assert oq.embeddings.shape == (dummy_adata.n_obs, 4), "Embeddings should match (n_obs, embed_dim)."
    # sample_ids_ should match obs_names
    assert oq.sample_ids_ == list(dummy_adata.obs_names)


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="Faiss is not available")
def test_query_with_text_faiss(dummy_model, dummy_adata):
    """
    Check that query_with_text using Faiss fills adata.obs["query_scores"] properly.
    """
    oq = OmicsQueryAnnotator(model=dummy_model, is_cosine=True)
    oq.build_omics_index(dummy_adata)

    queries = ["some text", "some other text"]
    oq.query_with_text(dummy_adata, queries, use_faiss=True, n_top=2)

    assert "query_scores" in dummy_adata.obs
    assert len(dummy_adata.obs["query_scores"]) == dummy_adata.n_obs

    # Each element in adata.obs["query_scores"] is a dict with up to 2 keys
    for d in dummy_adata.obs["query_scores"]:
        assert isinstance(d, dict)
        # We won't necessarily always get 2 keys per sample if there's something off,
        # but typically we expect up to n_top keys. Just check type correctness:
        for k, v in d.items():
            assert isinstance(k, str)
            assert isinstance(v, float)


def test_query_with_text_matmul(dummy_model, dummy_adata):
    """
    Check that query_with_text using matrix multiplication fills adata.obs["query_scores"] properly.
    """
    oq = OmicsQueryAnnotator(model=dummy_model, is_cosine=True)

    queries = ["text query 1", "text query 2"]
    oq.query_with_text(dummy_adata, queries, use_faiss=False, device="cpu", n_top=5, emb_key="mmcontext_emb")

    assert "query_scores" in dummy_adata.obs
    # shape: (n_obs,) of dictionaries
    assert len(dummy_adata.obs["query_scores"]) == dummy_adata.n_obs

    for d in dummy_adata.obs["query_scores"]:
        assert isinstance(d, dict)
        # now we should have 2 keys for 2 queries
        assert len(d) == 2
        for k, v in d.items():
            assert k in queries
            assert isinstance(v, float)


# -------------------------------
# Tests for compute_cosine_similarity_torch
# -------------------------------


def test_compute_cosine_similarity_torch_cpu():
    """
    Simple test to confirm correctness of compute_cosine_similarity_torch on CPU.
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
def test_compute_cosine_similarity_torch_gpu():
    """
    Test compute_cosine_similarity_torch on a CUDA GPU if available.
    """
    from mmcontext.utils import compute_cosine_similarity_torch

    samples = np.random.rand(5, 4).astype(np.float32)
    queries = np.random.rand(3, 4).astype(np.float32)

    sim_cpu = compute_cosine_similarity_torch(samples, queries, device="cpu")
    sim_gpu = compute_cosine_similarity_torch(samples, queries, device="cuda")

    # They should be almost identical
    np.testing.assert_allclose(sim_cpu, sim_gpu, atol=1e-5)


@pytest.mark.skipif(
    not getattr(torch.backends, "mps", False) or not torch.backends.mps.is_available(),
    reason="MPS device not available",
)
def test_compute_cosine_similarity_torch_mps():
    """
    Test compute_cosine_similarity_torch on Apple Silicon MPS device if available.
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

    import anndata
    import numpy as np

    from mmcontext.engine import OmicsQueryAnnotator

    # ------------------------------------------------
    # 1. Create a deterministic mock model
    # ------------------------------------------------
    class DeterministicMockModel:
        def encode(self, list_of_strings):
            """
            We'll encode sample embeddings and label embeddings in a way that
            ensures a specific label is guaranteed to be the highest-scoring.

            Strategy:
              - If the input strings look like "sample_0", "sample_1", etc.,
                we'll produce embeddings that favor label_0 or label_1.
              - If the input strings look like "label_0", "label_1", etc.,
                we'll produce specific distinct embeddings.
            """
            # For demonstration, let's say we have 2 samples and 3 labels
            # We'll handle the shape dynamically based on len(list_of_strings).

            embeddings = []
            for s in list_of_strings:
                if s.startswith("sample_"):
                    # We can parse the sample index
                    idx = int(s.split("_")[-1])
                    # Example: sample_0 -> [1,0,0], sample_1 -> [0,1,0], etc.
                    # Just be consistent so we can predict the best label.
                    vec = np.zeros(3, dtype=np.float32)
                    if idx < 3:
                        vec[idx] = 1.0
                    embeddings.append(vec)
                elif s.startswith("label_"):
                    # parse label index
                    idx = int(s.split("_")[-1])
                    # label_0 -> [1,0,0], label_1 -> [0,1,0], label_2 -> [0,0,1]
                    vec = np.zeros(3, dtype=np.float32)
                    vec[idx] = 1.0
                    embeddings.append(vec)
                else:
                    # fallback for random or unrecognized strings
                    # produce a zero vector of dim=3 for simplicity
                    embeddings.append(np.zeros(3, dtype=np.float32))

            return np.array(embeddings, dtype=np.float32)

    # ------------------------------------------------
    # 2. Construct an AnnData with 2 samples
    # ------------------------------------------------
    X = np.random.rand(2, 5).astype(np.float32)  # random dummy gene expression
    adata = anndata.AnnData(X)
    # We'll label the samples "sample_0" and "sample_1"
    adata.obs_names = ["sample_0", "sample_1"]
    # Store the sample embeddings in .obsm["mmcontext_emb"] using the mock format
    # We'll just reuse the model's logic by calling encode() on the sample names
    mock_model = DeterministicMockModel()
    sample_emb = mock_model.encode(adata.obs_names.to_list())  # shape: (2,3)
    adata.obsm["mmcontext_emb"] = sample_emb

    # ------------------------------------------------
    # 3. Create some label strings
    # ------------------------------------------------
    labels = ["label_0", "label_1", "label_2"]  # 3 labels

    # ------------------------------------------------
    # 4. Run annotate_omics_data
    # ------------------------------------------------
    oq = OmicsQueryAnnotator(model=mock_model, is_cosine=True)
    # We'll do a short call, letting 'labels=...' define which label embeddings are used
    # We'll do matrix multiplication (use_faiss=False).
    oq.annotate_omics_data(adata, labels=labels, use_faiss=False, n_top=3)

    # ------------------------------------------------
    # 5. Test the results
    # ------------------------------------------------
    # Check the new columns in adata.obs
    assert "inferred_labels" in adata.obs, "'inferred_labels' was not created."
    assert "best_label" in adata.obs, "'best_label' was not created."

    # We have 2 samples; each element in adata.obs["inferred_labels"] is a dict
    inferred_labels = adata.obs["inferred_labels"]
    best_labels = adata.obs["best_label"]

    assert len(inferred_labels) == 2
    assert len(best_labels) == 2

    # For each sample, check that best_label is the label with the highest score
    for i in range(2):
        label_scores = inferred_labels[i]  # dictionary {label_str: score}
        found_best = best_labels[i]
        # Make sure found_best is actually one of the dictionary keys
        assert found_best in label_scores, f"best_label='{found_best}' not among keys {list(label_scores.keys())}"
        # Verify it has the maximum score in the dict
        max_label = max(label_scores, key=label_scores.get)  # label with highest value
        assert found_best == max_label, f"Expected best_label='{max_label}' but got '{found_best}'."

    # If we want to confirm the logic matches our encoding logic:
    # - sample_0's embedding = [1,0,0]
    #   so it should have highest dot product with label_0 => best_label = 'label_0'
    # - sample_1's embedding = [0,1,0]
    #   so it should match label_1 => best_label = 'label_1'
    assert best_labels[0] == "label_0"
    assert best_labels[1] == "label_1"

    print("Test passed: best_label is correctly identified for each sample.")
