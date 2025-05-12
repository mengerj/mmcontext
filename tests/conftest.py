from unittest.mock import patch

import datasets
import numpy as np
import pandas as pd
import pytest
import torch
from sentence_transformers import SentenceTransformer

from mmcontext.models.MMContextEncoder import MMContextEncoder, MMContextProcessor


# --------------------------------------------------------------------- #
# fast dummy stubs (shared by all tests)
# --------------------------------------------------------------------- #
class _TokStub:
    def __call__(self, texts, padding=True, **kw):
        b = len(texts)
        maxlen = 8
        ids = torch.ones(b, maxlen, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": ids}


@pytest.fixture(scope="session")
def TokStub():
    """Gives tests access to the stub *class*."""
    return _TokStub


class _TextEncStub(torch.nn.Module):
    def __init__(self, hidden=32, model_type="bert"):
        super().__init__()
        self.config = type("cfg", (), {"hidden_size": hidden})()

        # This dummy_param should be registered properly but isn't important for tests
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        # Create structure based on model type
        if model_type == "bert":
            # BERT-like structure with real nn.Module to ensure proper parameter registration
            self.encoder = torch.nn.Module()
            self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(hidden, hidden) for _ in range(3)])
        elif model_type == "roberta":
            # RoBERTa-like structure with real nn.Module to ensure proper parameter registration
            self.roberta = torch.nn.Module()
            self.roberta.encoder = torch.nn.Module()
            self.roberta.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(hidden, hidden) for _ in range(3)])
        else:
            # Unsupported structure for testing warnings
            pass

    def forward(self, input_ids=None, **kw):
        b = input_ids.size(0)
        device = input_ids.device if input_ids is not None else "cpu"
        sequence_length = input_ids.size(1) if input_ids is not None else 8
        last_hidden_state = torch.zeros(b, sequence_length, self.config.hidden_size).to(device)
        return type(
            "o",
            (),
            {
                "pooler_output": torch.zeros(b, self.config.hidden_size).to(device),
                "last_hidden_state": last_hidden_state,
            },
        )  # Mimic the output of a transformer model


@pytest.fixture(scope="session")
def TextEncStub():
    """Gives tests access to the stub *class*."""
    return _TextEncStub


# --------------------------------------------------------------------- #
# Global patches for test efficiency
# --------------------------------------------------------------------- #
@pytest.fixture(scope="session", autouse=True)
def patch_model_loading():
    """Patch model loading functions for the entire test session."""
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=_TextEncStub(model_type="bert")),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_TokStub()),
        patch("safetensors.torch.load_model", return_value=None),
    ):
        # This patch will be active for the entire test session
        yield


# ---------------------------------------------------------------------------
# 1  Embedding helpers
# ---------------------------------------------------------------------------
def _rand_matrix(tokens, hidden=16, seed=0):
    rng = np.random.default_rng(seed)
    return {tok: rng.standard_normal(hidden).astype(np.float32) for tok in tokens}


# ---------------------------------------------------------------------------
# 2  Numeric datasets in THREE formats
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def numeric_df():
    """pandas.DataFrame  –  feature tokens F1‥F3"""
    mapping = _rand_matrix(["F1", "F2", "F3"])
    return pd.DataFrame({"token": list(mapping.keys()), "embedding": list(mapping.values())})


@pytest.fixture(scope="session")
def numeric_df_other_dim():
    """pandas.DataFrame  –  feature tokens F1‥F3"""
    mapping = _rand_matrix(["F1", "F2", "F3"], hidden=4)
    return pd.DataFrame({"token": list(mapping.keys()), "embedding": list(mapping.values())})


@pytest.fixture(scope="session")
def numeric_mapping():
    """dict[str, np.ndarray]  –  sample tokens S1‥S3"""
    return _rand_matrix(["S1", "S2", "S3"], seed=1)


@pytest.fixture(scope="session")
def numeric_hfds():
    """datasets.Dataset  –  mix of F4‥F5, S4"""
    mapping = _rand_matrix(["F4", "F5", "S4"], seed=2)
    return datasets.Dataset.from_dict({"token": list(mapping.keys()), "embedding": list(mapping.values())})


@pytest.fixture(scope="session")
def dummy_dataset():
    """Create a small dummy dataset for testing."""
    return datasets.Dataset.from_dict(
        {
            "omics_tokens": ["F1", "F1", "F2"],
            "caption": [
                "Some sample of interest",
                "A false description of the sample",
                "A another description of the sample",
            ],
            "label": [0, 1, 0],
        }
    )


@pytest.fixture(scope="session")
def dummy_dataset_with_split():
    """Create a small dummy dataset with a split for testing."""
    return datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {
                    "omics_tokens": ["F1", "F1", "F2"],
                    "caption": [
                        "Some sample of interest",
                        "A false description of the sample",
                        "A another description of the sample",
                    ],
                    "label": [0, 1, 0],
                }
            ),
            "val": datasets.Dataset.from_dict(
                {
                    "omics_tokens": ["F3", "F3"],
                    "caption": [
                        "Some sample of interest",
                        "A false description of the sample",
                    ],
                    "label": [0, 1],
                }
            ),
        }
    )


# ---------------------------------------------------------------------------
# 3  Processor + Encoder ready to use
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def text_only_encoder():
    """MMContextEncoder with text-only encoder (no HF download)"""
    enc = MMContextEncoder(
        text_encoder_name="bert-base-uncased", adapter_hidden_dim=4, adapter_output_dim=8, output_token_embeddings=True
    )
    return enc


@pytest.fixture(scope="function")
def bimodal_encoder(numeric_df, numeric_mapping, numeric_hfds):
    """
    MMContextEncoder with:
      • mocked text-encoder (no HF download)
      • processor containing the lookup
      • MiniOmicsModel holding *all* numeric tokens
    """
    enc = MMContextEncoder(
        text_encoder_name="bert-base-uncased", adapter_hidden_dim=4, adapter_output_dim=8, output_token_embeddings=True
    )

    # register three datasets (DF → Mapping → HF DS)
    enc.register_initial_embeddings(numeric_df, data_origin="pca")  # some datatype has to be used
    enc.register_initial_embeddings(numeric_mapping, data_origin="pca")
    enc.register_initial_embeddings(numeric_hfds, data_origin="pca")

    return enc


@pytest.fixture(scope="function")
def no_adapter_encoder():
    """Create an encoder without adapter layers."""
    # No need for patching, it's handled by the global fixture
    encoder = MMContextEncoder(
        text_encoder_name="bert-base-uncased",
        adapter_hidden_dim=None,  # No adapter
    )
    return encoder


@pytest.fixture(scope="function")
def st_text_encoder(text_only_encoder):
    """SentenceTransformer with text-only encoder (no HF download)"""
    return SentenceTransformer(modules=[text_only_encoder])


@pytest.fixture(scope="function")
def st_bimodal_encoder(bimodal_encoder):
    return SentenceTransformer(modules=[bimodal_encoder])


@pytest.fixture(scope="session")
def hf_caption_dataset():
    """
    Default prefix 'sample_idx:' in *omics_tokens* column.

    Row 0 : omics-only  (two feature tokens)
    Row 1 : caption-only
    Row 2 : omics-only  (single sample token)
    Row 3 : caption + omics  (mixed mini-batch test)
    """
    return datasets.Dataset.from_dict(
        {
            "omics_tokens": [
                "sample_idx: F1 F2",  # row 0
                "",  # row 1
                "sample_idx: S1",  # row 2
                "sample_idx: F3 F5",  # row 3
            ],
            "captions": [
                "",  # row 0
                "A plain caption about biology.",  # row 1
                "",  # row 2
                "Another caption mentioning F3.",  # row 3
            ],
        }
    )


@pytest.fixture(scope="session")
def hf_caption_dataset_alt_prefix():
    """
    Same idea but with prefix 'gene_id:' to verify processor prefix switching.
    """
    return datasets.Dataset.from_dict(
        {
            "omics_tokens": [
                "gene_id: S2 S3",  # omics-only
                "gene_id: F4",  # omics-only
            ],
            "captions": [
                "",  # row 0
                "",  # row 1
            ],
        }
    )
