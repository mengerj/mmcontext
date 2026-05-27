"""OmicsAttentionModule — optional self-attention for omics tokens.

This module sits between :class:`MMContextModule` and :class:`AdapterModule`
in a sentence-transformers pipeline. It applies multi-head self-attention
ONLY to omics tokens (modality_id=1), leaving text tokens unchanged.

This is useful for var-level models where each sample has multiple gene
embeddings that benefit from contextual mixing before projection into the
shared space.

Architecture
------------
A stack of standard ``nn.TransformerEncoderLayer`` modules with pre-LayerNorm.
Only omics tokens are gathered, attended, and scattered back.  Text tokens
(modality_id=0) and pad tokens (modality_id=2) pass through untouched.

For obs-level inputs (L=1), self-attention is trivial — the single token
attends only to itself, which is essentially a feedforward pass. This is
fine: the module adds negligible overhead for obs-level data and can be
left in the pipeline without harm.

Features dict contract::

    # Input (from MMContextModule.forward):
    {
        "token_embeddings": Tensor[B, L, D],
        "attention_mask":   Tensor[B, L],
        "modality_ids":     Tensor[B, L],  # 0=text, 1=omics, 2=pad
    }

    # Output (after OmicsAttentionModule.forward):
    {
        "token_embeddings": Tensor[B, L, D],   # omics tokens attended
        "attention_mask":   Tensor[B, L],       # unchanged
        "modality_ids":     Tensor[B, L],       # unchanged
    }

Example
-------
>>> attn = OmicsAttentionModule(input_dim=768, num_heads=8, num_layers=2)
>>> features = mmcontext_module.forward(...)
>>> features = attn(features)
>>> features = adapter_module(features)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn

# Try v5.4+ import paths first, fall back to v5.0 paths
try:
    from sentence_transformers.base.modules import Module
except ImportError:
    from sentence_transformers.models import Module

logger = logging.getLogger(__name__)

# Modality constants (must match mmcontext_module.py / adapter_module.py)
MODALITY_TEXT = 0
MODALITY_OMICS = 1
MODALITY_PAD = 2


class OmicsAttentionModule(Module):
    """Self-attention module applied only to omics tokens.

    Maintains a stack of ``TransformerEncoderLayer`` blocks. Only omics
    tokens (modality_id=1) are gathered, fed through the attention layers,
    and scattered back into the output tensor. Text and pad tokens pass
    through unchanged.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the token embeddings (must match the upstream
        encoder's hidden size).
    num_heads : int, optional
        Number of attention heads. Must evenly divide ``input_dim``.
        Default: 4.
    num_layers : int, optional
        Number of stacked TransformerEncoderLayer blocks. Default: 1.
    feedforward_dim : int or None, optional
        Hidden size of the feedforward network inside each layer.
        If None, defaults to ``4 * input_dim``. Default: None.
    dropout : float, optional
        Dropout probability for attention weights and feedforward layers.
        Default: 0.1.
    """

    config_keys: list[str] = [
        "input_dim",
        "num_heads",
        "num_layers",
        "feedforward_dim",
        "dropout",
    ]
    config_file_name: str = "omics_attention_module_config.json"

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.feedforward_dim = feedforward_dim if feedforward_dim else 4 * input_dim
        self.dropout = dropout

        # Build attention stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=self.feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-LayerNorm for training stability
        )
        self.attention = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    # ------------------------------------------------------------------
    # Forward (Module abstract method)
    # ------------------------------------------------------------------
    def forward(
        self,
        features: dict[str, torch.Tensor | Any],
        **kwargs,
    ) -> dict[str, torch.Tensor | Any]:
        """Apply self-attention to omics tokens only.

        Text tokens and pad tokens pass through unchanged. Omics tokens
        are gathered per sample, attended, and scattered back.

        Parameters
        ----------
        features : dict
            Must contain ``token_embeddings`` (B, L, D), ``attention_mask``
            (B, L), and ``modality_ids`` (B, L).

        Returns
        -------
        dict
            Updated features with attended omics token embeddings.
        """
        token_embeddings = features["token_embeddings"]
        modality_ids = features["modality_ids"]
        attention_mask = features["attention_mask"]

        B, L, D = token_embeddings.shape
        device = token_embeddings.device

        # Start with a copy of the input
        output = token_embeddings.clone()

        # Check if there are any omics tokens at all
        omics_mask = modality_ids == MODALITY_OMICS  # (B, L)
        if not omics_mask.any():
            return features

        # Process each sample independently — omics sequences can have
        # different lengths across samples in the batch
        attended_samples = []
        sample_indices = []

        for b in range(B):
            omics_positions = omics_mask[b].nonzero(as_tuple=True)[0]  # positions of omics tokens
            if len(omics_positions) == 0:
                continue

            # Gather omics tokens for this sample: (N_omics, D)
            omics_tokens = token_embeddings[b, omics_positions, :].unsqueeze(0)  # (1, N_omics, D)

            # Build attention mask for these tokens
            # The overall attention_mask tells us which positions are real
            omics_attn_mask = attention_mask[b, omics_positions].unsqueeze(0)  # (1, N_omics)

            # Convert to the format nn.TransformerEncoder expects:
            # src_key_padding_mask: True = ignore, False = attend
            padding_mask = omics_attn_mask == 0  # (1, N_omics)

            # Apply attention
            attended = self.attention(
                omics_tokens,
                src_key_padding_mask=padding_mask,
            )  # (1, N_omics, D)

            # Scatter back into output
            output[b, omics_positions, :] = attended.squeeze(0)

        # Zero out pad positions to be safe
        pad_mask = modality_ids == MODALITY_PAD
        if pad_mask.any():
            output[pad_mask] = 0.0

        features = dict(features)  # shallow copy to avoid mutating input
        features["token_embeddings"] = output
        return features

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    def get_sentence_embedding_dimension(self) -> int:
        """Return the output dimensionality (same as input — attention preserves dim)."""
        return self.input_dim

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(
        self,
        output_path: str,
        *args,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        """Save config and weights to disk.

        Parameters
        ----------
        output_path : str
            Directory where files will be written.
        safe_serialization : bool
            If True, use safetensors format for weights.
        """
        from safetensors.torch import save_model as save_safetensors_model

        output_path = str(output_path)
        os.makedirs(output_path, exist_ok=True)

        self.save_config(output_path)

        if safe_serialization:
            save_safetensors_model(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

        logger.info("Saved OmicsAttentionModule to %s", output_path)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> OmicsAttentionModule:
        """Load a saved OmicsAttentionModule from disk.

        Parameters
        ----------
        model_name_or_path : str
            Path to directory containing saved module files.

        Returns
        -------
        OmicsAttentionModule
        """
        from safetensors.torch import load_model as load_safetensors_model

        config = cls.load_config(
            model_name_or_path,
            subfolder=subfolder,
            config_filename=cls.config_file_name,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

        module = cls(**config)

        load_path = model_name_or_path
        if subfolder:
            load_path = os.path.join(model_name_or_path, subfolder)

        safetensors_path = os.path.join(load_path, "model.safetensors")
        bin_path = os.path.join(load_path, "pytorch_model.bin")

        if os.path.isfile(safetensors_path):
            load_safetensors_model(module, safetensors_path)
        elif os.path.isfile(bin_path):
            module.load_state_dict(
                torch.load(bin_path, map_location=torch.device("cpu"))
            )
        else:
            logger.warning(
                "No weight files found in %s — module uses random init.", load_path
            )

        logger.info("Loaded OmicsAttentionModule from %s", model_name_or_path)
        return module

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"OmicsAttentionModule("
            f"dim={self.input_dim}, "
            f"heads={self.num_heads}, "
            f"layers={self.num_layers}, "
            f"ff={self.feedforward_dim})"
        )
