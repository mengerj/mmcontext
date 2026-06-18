"""Back-compat shim — the embed helpers now live in :mod:`mmcontext.embed.encode`.

This module re-exports the public embed helpers under their historical
``mmcontext.embed.model_utils`` import path so existing consumers (notably the
``sc_language_foundation_models`` ``predict.py`` scripts) keep working without
changes. New code should import from :mod:`mmcontext.embed.encode`.
"""

from __future__ import annotations

from mmcontext.embed.encode import (
    HFIndexedDataset,
    SentenceDataset,
    create_label_dataset,
    embed_labels,
    load_st_model,
    prepare_model_and_embed,
)

__all__ = [
    "HFIndexedDataset",
    "SentenceDataset",
    "create_label_dataset",
    "embed_labels",
    "load_st_model",
    "prepare_model_and_embed",
]
