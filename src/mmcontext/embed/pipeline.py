"""Builder for the mmcontext sentence-transformers pipeline.

The standard obs-level pipeline is::

    [MMContextModule, AdapterModule, Pooling, Normalize]

:func:`build_pipeline` assembles it with a single call so training scripts
(``scripts/train_tiny.py``, ``scripts/train_config.py``) share one definition.
"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Normalize, Pooling

from mmcontext.modules import AdapterModule, MMContextModule

logger = logging.getLogger(__name__)


def build_pipeline(
    text_model: str,
    *,
    omics_dim: int | None = None,
    shared_dim: int = 256,
    adapter_hidden_dim: int | None = 512,
    max_seq_length: int = 512,
) -> SentenceTransformer:
    """Build the MMContext sentence-transformer pipeline.

    Parameters
    ----------
    text_model : str
        HuggingFace model name/path for the text encoder.
    omics_dim : int, optional
        Dimension of omics vectors. If ``None``, the adapter's omics head is
        sized to match the text dimension (text-only / placeholder use).
    shared_dim : int, default 256
        Output dimension of the shared embedding space.
    adapter_hidden_dim : int or None, default 512
        Hidden layer size of each adapter projection. ``None``/``0`` uses a
        single ``Linear -> LayerNorm`` head instead of an MLP.
    max_seq_length : int, default 512
        Maximum sequence length for text tokenization.

    Returns
    -------
    sentence_transformers.SentenceTransformer
        The assembled pipeline.
    """
    mmcontext = MMContextModule(model_name_or_path=text_model, max_seq_length=max_seq_length)
    text_dim = mmcontext.get_word_embedding_dimension()

    if omics_dim is None:
        omics_dim = text_dim

    adapter = AdapterModule(
        text_input_dim=text_dim,
        omics_input_dim=omics_dim,
        shared_dim=shared_dim,
        hidden_dim=adapter_hidden_dim,
    )
    pooling = Pooling(embedding_dimension=shared_dim, pooling_mode="mean")
    normalize = Normalize()

    pipeline = SentenceTransformer(modules=[mmcontext, adapter, pooling, normalize])
    logger.info(
        "Pipeline: text_dim=%d, omics_dim=%d, shared_dim=%d, adapter_hidden_dim=%s, params=%d",
        text_dim,
        omics_dim,
        shared_dim,
        adapter_hidden_dim,
        sum(p.numel() for p in pipeline.parameters()),
    )
    return pipeline
