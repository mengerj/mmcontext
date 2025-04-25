# file: mini_omics_encoder.py
# -------------------------------------------------------------
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

logger = logging.getLogger(__name__)


class MiniOmicsConfig(PretrainedConfig):
    r"""Configuration for :class:`MiniOmicsModel`.

    Parameters
    ----------
    vocab_size : int
        Number of unique sample IDs (rows in the embedding matrix).
    hidden_size : int
        Dimensionality of each embedding vector.
    padding_idx : int, optional
        Index that should broadcast all-zero vectors (default: 0).
    """

    model_type = "mini_omics"

    def __init__(
        self,
        vocab_size: int = 1,
        hidden_size: int = 1,
        padding_idx: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx


class MiniOmicsModel(PreTrainedModel):
    r"""Look-up–only encoder that mimics ``AutoModel`` outputs.

    The forward pass returns

    * **last_hidden_state** – shape ``(B, L, hidden_size)`` – unprocessed
      embeddings (``L`` is the sequence length, usually ``1``).
    * **pooler_output** – shape ``(B, hidden_size)`` – the 0-th token, so
      identical to ``last_hidden_state[:, 0]``.

    Notes
    -----
    *No* transformer layers, just an ``nn.Embedding`` with pretrained weights.
    """

    config_class = MiniOmicsConfig

    def __init__(self, config: MiniOmicsConfig):
        super().__init__(config)
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.padding_idx,
        )
        self.post_init()  # required by 🤗 base class

    # ------------------------------------------------------------------ #
    # Helper constructor: build from a NumPy embedding matrix
    # ------------------------------------------------------------------ #
    @classmethod
    def from_numpy(
        cls,
        embedding_matrix: np.ndarray,
        *,
        padding_idx: int = 0,
        **kw,
    ) -> MiniOmicsModel:
        """Instantiate model and load weights from a NumPy array.

        Parameters
        ----------
        embedding_matrix : np.ndarray, shape (vocab_size, hidden_size)
            Row *i* contains the vector for sample ID *i*.
        padding_idx : int, optional
            Which row should act as the padding vector (all zeros).

        Returns
        -------
        MiniOmicsModel
        """
        vocab_size, hidden_size = embedding_matrix.shape
        cfg = MiniOmicsConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            padding_idx=padding_idx,
        )
        model = cls(cfg)
        with torch.no_grad():
            model.embeddings.weight.copy_(torch.from_numpy(embedding_matrix))
        logger.info(
            "Loaded embedding matrix with shape (%d, %d)",
            vocab_size,
            hidden_size,
        )
        return model

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.LongTensor,
        **unused,
    ) -> BaseModelOutputWithPooling:
        """
        Forward pass

        (no transformer layers, just a look-up table).

        Parameters
        ----------
        input_ids : torch.LongTensor, shape (B, L)
            Integer sample IDs.

        Returns
        -------
        transformers.BaseModelOutputWithPooling
            * ``last_hidden_state`` – raw embeddings (B, L, H)
            * ``pooler_output``     – first token (B, H)
        """
        hidden = self.embeddings(input_ids)  # (B, L, H)
        pooled = hidden[:, 0]  # (B, H)
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden,
            pooler_output=pooled,
        )


try:
    # Modern API ≥ 4.38
    MiniOmicsConfig.register_for_auto_class("AutoConfig")
    MiniOmicsModel.register_for_auto_class("AutoModel")

except AttributeError:
    # Fallback for 4.25 – 4.37, where a decorator helper is exposed instead
    from transformers.utils import register_for_auto_class

    MiniOmicsConfig = register_for_auto_class("AutoConfig")(MiniOmicsConfig)
    MiniOmicsModel = register_for_auto_class("AutoModel")(MiniOmicsModel)
