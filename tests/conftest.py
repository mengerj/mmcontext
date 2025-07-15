import json
import logging
import os
from typing import Optional, Union
from unittest.mock import patch

import datasets
import numpy as np
import pandas as pd
import pytest
import torch
from sentence_transformers import SentenceTransformer
from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.auto import configuration_auto, modeling_auto

from mmcontext.models.mmcontextencoder import MMContextEncoder, MMContextProcessor

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# fast dummy stubs (shared by all tests)
# --------------------------------------------------------------------- #
class _TokStub:
    """Minimal tokenizer stand‑in implementing the *few* HF methods that
    `sentence_transformers` relies on during save/load cycles.

    The implementation purposefully avoids inheriting from
    :class:`transformers.PreTrainedTokenizer` to keep dependencies minimal. All
    that is required for unit‑tests is the presence of
    :pymeth:`save_pretrained`, :pymeth:`from_pretrained`, a handful of
    attributes (such as :pyattr:`pad_token_id`), and a ``__call__`` method that
    returns *input IDs*/**attention masks** tensors.
    """

    pad_token_id: int = 0

    def __init__(self, vocab_size: int = 30_522, max_length: int = 8):
        self.vocab_size = vocab_size
        self.max_length = max_length

    # ------------------------------------------------------------------
    # Encode / batch‑encode
    # ------------------------------------------------------------------
    def __call__(
        self,
        texts: str | list[str],
        padding: bool = True,
        truncation: bool = True,  # ignored – output is fixed length
        return_tensors: str | None = "pt",
        **kwargs,
    ) -> dict:
        """Mimic `AutoTokenizer.__call__` behaviour.

        Parameters
        ----------
        texts : str | list[str]
            Input text(s) to *pretend* to tokenize.
        padding : bool, default=True
            Ignored – output is always fixed length.
        truncation : bool, default=True
            Ignored – sequence is never longer than ``max_length``.
        return_tensors : {"pt", None}, default="pt"
            Controls the returned type; only the PyTorch pathway is supported.

        Returns
        -------
        dict
            * ``input_ids`` – ``torch.LongTensor`` with ones.
            * ``attention_mask`` – same shape, filled with ones.
        """
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        ids = torch.ones(batch_size, self.max_length, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": ids}

    # ------------------------------------------------------------------
    # HF‑style Persistence
    # ------------------------------------------------------------------
    def save_pretrained(self, save_directory: str, **kwargs):
        """Persist *nothing but* a tiny JSON config.

        Parameters
        ----------
        save_directory : str
            Target folder; created if missing.
        **kwargs
            Ignored – kept for signature compatibility.
        """
        os.makedirs(save_directory, exist_ok=True)
        cfg = {
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "pad_token_id": self.pad_token_id,
            "_stub": True,
        }
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as fp:
            json.dump(cfg, fp)
        logger.info("Tokenizer stub saved to %s", save_directory)

    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Re‑instantiate from files written by :pymeth:`save_pretrained`."""
        cfg_path = os.path.join(load_directory, "tokenizer_config.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError("Stub tokenizer config not found in " + load_directory)
        with open(cfg_path) as fp:
            cfg = json.load(fp)
        return cls(vocab_size=cfg["vocab_size"], max_length=cfg["max_length"])


@pytest.fixture(scope="session")
def TokStub():
    """Gives tests access to the stub *class*."""
    return _TokStub


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


class StubConfig(PretrainedConfig):
    """Configuration for :class:`TextEncStub`.

    Parameters
    ----------
    hidden_size : int, default=32
        Dimensionality of the hidden representations produced by the encoder.
    num_hidden_layers : int, default=3
        Number of linear layers stacked in the encoder.
    pad_token_id : int, default=0
        Token used for padding; kept for Hugging-Face compatibility.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`transformers.PretrainedConfig`.

    Notes
    -----
    This configuration is intentionally minimal: it includes only the fields
    required by unit-tests that expect a *Hugging Face-compatible* object.
    """

    model_type = "stub"

    def __init__(
        self,
        hidden_size: int = 32,
        num_hidden_layers: int = 3,
        pad_token_id: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers


# Register the StubConfig in transformers CONFIG_MAPPING
configuration_auto.CONFIG_MAPPING.register("stub", StubConfig)


# -----------------------------------------------------------------------------
# Encoder Stub
# -----------------------------------------------------------------------------


class _TextEncStub(PreTrainedModel):
    """A light-weight *transformer-like* encoder for testing pipelines.

    The class **inherits from** :class:`transformers.PreTrainedModel` so that
    downstream libraries (e.g. *sentence-transformers*) can call
    :pymeth:`save_pretrained` / :pymeth:`from_pretrained` without raising
    ``AttributeError``.

    Parameters
    ----------
    config : StubConfig, optional
        Configuration object describing model hyper-parameters.
    model_type : str, optional
        Type of model to simulate ("bert", "roberta", etc.). If provided,
        creates the appropriate attribute structure for that model type.

    Attributes
    ----------
    layers : torch.nn.ModuleList
        Stack of ``num_hidden_layers`` linear projections; they *do nothing* but
        ensure that parameters are registered correctly for optimization.

    Examples
    --------
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>> model = TextEncStub(StubConfig())
    >>> ids = tokenizer("unit test", return_tensors="pt").input_ids
    >>> out = model(ids)
    >>> out.last_hidden_state.shape
    torch.Size([1, ids.size(1), 32])
    """

    config_class = StubConfig

    def __init__(self, config: StubConfig = None, model_type: str = None) -> None:
        if config is None:
            config = StubConfig()
        super().__init__(config)
        self.config = config  # type: ignore[assignment]

        # Dummy parameter to guarantee at least one weight tensor.
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        # Minimal *encoder* consisting of independent linear layers.
        layers = torch.nn.ModuleList(
            [torch.nn.Linear(config.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)]
        )

        # Create the appropriate structure based on model type
        if model_type == "roberta":
            # Create RoBERTa-like structure: roberta.encoder.layer
            self.roberta = torch.nn.Module()
            self.roberta.encoder = torch.nn.Module()
            self.roberta.encoder.layer = layers
        elif model_type == "unsupported":
            # For unsupported architectures, use default structure
            self.layers = layers
        else:
            # Default to BERT-like structure: encoder.layer
            self.encoder = torch.nn.Module()
            self.encoder.layer = layers

        # Ensure weight initialisation follows HF conventions.
        self.post_init()

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor | None = None, **kwargs):
        """Simulate the forward pass of a transformer.

        Parameters
        ----------
        input_ids : torch.Tensor, optional
            Dummy *token IDs*. Only the *batch size* and *sequence length*
            dimensions are used to shape the output; the *values* themselves are
            ignored.
        **kwargs
            Ignored; kept for API compatibility.

        Returns
        -------
        transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
            An *object-like* namespace containing two attributes:

            * **pooler_output** – zeros of shape *(batch, hidden_size)*.
            * **last_hidden_state** – zeros of shape
              *(batch, sequence_length, hidden_size)*.

        Notes
        -----
        The tensors are **allocated on the same device** as ``input_ids`` if
        provided; otherwise on CPU. They contain only zeros, because the stub is
        designed for *unit-tests* where numerical correctness is irrelevant.
        """

        if input_ids is None:
            logger.warning("`input_ids` is None — falling back to a dummy tensor on CPU.")
            batch_size, seq_len, device = 1, 8, torch.device("cpu")
        else:
            batch_size, seq_len = input_ids.size(0), input_ids.size(1)
            device = input_ids.device

        hidden = torch.zeros(batch_size, seq_len, self.config.hidden_size, device=device)
        pooled = torch.zeros(batch_size, self.config.hidden_size, device=device)

        return type(
            "BaseModelOutput",  # choose neutral name (no need for actual HF class)
            (),
            {
                "last_hidden_state": hidden,
                "pooler_output": pooled,
            },
        )()

    # ------------------------------------------------------------------
    # Weight initialisation helper required by PreTrainedModel
    # ------------------------------------------------------------------
    def _init_weights(self, module):  # pylint: disable=unused-argument
        """No-op initialisation (weights are already zeros)."""
        # All tensors have already been initialised to zeros, which is fine for
        # a stub model used only in tests.
        pass


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
        patch("transformers.AutoModel.from_pretrained", return_value=_TextEncStub(StubConfig())),
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
