import logging
from types import SimpleNamespace

import torch
from torch import nn

logger = logging.getLogger(__name__)


class OneHotTextEncoder(nn.Module):
    """
    Embeds each *full sentence* as an independent, learnable vector.

    The module is deliberately BERT-API-compatible: it exposes
    ``config.hidden_size`` and returns an object with
    ``pooler_output`` **and** ``last_hidden_state``.

    Parameters
    ----------
    num_sentences : int
        Upper bound for distinct sentences that may appear during training.
        Unseen sentences are added on-the-fly until this limit is reached.
    embed_dim : int
        Dimension of the sentence embeddings (= ``hidden_size``).
    trainable : bool, default True
        Whether to update the embedding matrix during training.
    """

    def __init__(self, num_sentences: int, embed_dim: int, *, trainable: bool = True) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=embed_dim)
        self.embeddings = nn.Embedding(num_sentences, embed_dim)
        self.embeddings.weight.requires_grad = trainable

    def forward(self, input_ids: torch.Tensor, **_) -> SimpleNamespace:
        """Forward pass for the OneHotTextEncoder."""
        # input_ids shape (B, 1); treat the single “token” as the sentence id
        vec = self.embeddings(input_ids.squeeze(-1))  # (B, E)
        return SimpleNamespace(
            pooler_output=vec,
            last_hidden_state=vec.unsqueeze(1),  # (B, 1, E)
        )
