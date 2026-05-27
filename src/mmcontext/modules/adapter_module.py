"""AdapterModule — modality-aware projection for the ST pipeline.

This module sits after :class:`MMContextModule` in a sentence-transformers
pipeline. It reads ``modality_ids`` from the features dict and applies
separate learned projections for text and omics tokens, mapping them into
a shared embedding space of dimension ``D_shared``.

Architecture
------------
Two independent projection heads (``text_proj`` and ``omics_proj``) map
embeddings from their respective input dimensions to a common output
dimension. Three projection modes are supported:

1. **Identity** — when ``force_identity=True`` and all dims match, both
   projections act as ``nn.Identity``.
2. **Linear → BatchNorm** — when ``hidden_dim`` is ``None`` or ``0``.
3. **Linear → ReLU → Linear → BatchNorm** — the default MLP mode.

Pad tokens (``modality_id=2``) pass through as zeros.

Features dict contract::

    # Input (from MMContextModule.forward):
    {
        "token_embeddings": Tensor[B, L, D_text or D_omics],
        "attention_mask":   Tensor[B, L],
        "modality_ids":     Tensor[B, L],  # 0=text, 1=omics, 2=pad
    }

    # Output (after AdapterModule.forward):
    {
        "token_embeddings": Tensor[B, L, D_shared],  # projected
        "attention_mask":   Tensor[B, L],             # unchanged
        "modality_ids":     Tensor[B, L],             # unchanged
    }

Example
-------
>>> adapter = AdapterModule(text_input_dim=768, omics_input_dim=512, shared_dim=256)
>>> features = mmcontext_module.forward(mmcontext_module.preprocess(texts))
>>> projected = adapter(features)
>>> projected["token_embeddings"].shape
torch.Size([B, L, 256])
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

# Modality constants (must match mmcontext_module.py)
MODALITY_TEXT = 0
MODALITY_OMICS = 1
MODALITY_PAD = 2


def _build_projection(
    input_dim: int,
    output_dim: int,
    hidden_dim: int | None,
    force_identity: bool,
) -> nn.Module:
    """Build a single projection head.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the incoming features.
    output_dim : int
        Dimensionality of the projected output.
    hidden_dim : int or None
        Hidden layer size. If None, skip the hidden layer.
    force_identity : bool
        If True and dims match and no hidden layer, use nn.Identity.

    Returns
    -------
    nn.Module
        The projection network.
    """
    if hidden_dim is None and input_dim == output_dim and force_identity:
        return nn.Identity()
    elif hidden_dim is None:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    else:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )


class AdapterModule(Module):
    """Modality-aware projection module for sentence-transformers pipelines.

    Maintains separate projection heads for text and omics tokens,
    dispatching based on ``modality_ids`` in the features dict.

    Parameters
    ----------
    text_input_dim : int
        Dimensionality of text token embeddings (from the text encoder).
    omics_input_dim : int
        Dimensionality of omics token embeddings.
    shared_dim : int
        Output dimensionality for both projections (the shared space).
    hidden_dim : int or None, optional
        Hidden layer size in each MLP projection. If ``None``, uses a
        single linear layer + BatchNorm instead. Default: 512.
    force_identity : bool, optional
        If True and all dims match and ``hidden_dim`` is None, both
        projections act as identity. Default: False.
    """

    config_keys: list[str] = [
        "text_input_dim",
        "omics_input_dim",
        "shared_dim",
        "hidden_dim",
        "force_identity",
    ]
    config_file_name: str = "adapter_module_config.json"

    def __init__(
        self,
        text_input_dim: int,
        omics_input_dim: int,
        shared_dim: int,
        hidden_dim: int | None = 512,
        force_identity: bool = False,
    ) -> None:
        super().__init__()

        self.text_input_dim = text_input_dim
        self.omics_input_dim = omics_input_dim
        self.shared_dim = shared_dim
        self.hidden_dim = hidden_dim if hidden_dim else None
        self.force_identity = force_identity

        # Build independent projection heads
        self.text_proj = _build_projection(
            text_input_dim, shared_dim, self.hidden_dim, force_identity
        )
        self.omics_proj = _build_projection(
            omics_input_dim, shared_dim, self.hidden_dim, force_identity
        )

    # ------------------------------------------------------------------
    # Forward (Module abstract method)
    # ------------------------------------------------------------------
    def forward(
        self,
        features: dict[str, torch.Tensor | Any],
        **kwargs,
    ) -> dict[str, torch.Tensor | Any]:
        """Project token embeddings using modality-specific heads.

        Parameters
        ----------
        features : dict
            Must contain ``token_embeddings`` (B, L, D), ``attention_mask``
            (B, L), and ``modality_ids`` (B, L).

        Returns
        -------
        dict
            Updated features with ``token_embeddings`` projected to
            ``(B, L, shared_dim)``.
        """
        token_embeddings = features["token_embeddings"]
        modality_ids = features["modality_ids"]

        B, L, D = token_embeddings.shape
        device = token_embeddings.device

        # Allocate output
        output = torch.zeros(B, L, self.shared_dim, device=device, dtype=token_embeddings.dtype)

        # Masks for each modality
        text_mask = modality_ids == MODALITY_TEXT
        omics_mask = modality_ids == MODALITY_OMICS
        # pad (modality_id=2) stays zero — no action needed

        # Project text tokens
        if text_mask.any():
            text_tokens = token_embeddings[text_mask]  # (N_text, D)
            output[text_mask] = self._apply_projection(self.text_proj, text_tokens)

        # Project omics tokens
        if omics_mask.any():
            omics_tokens = token_embeddings[omics_mask]  # (N_omics, D)
            output[omics_mask] = self._apply_projection(self.omics_proj, omics_tokens)

        features["token_embeddings"] = output
        return features

    def _apply_projection(
        self, proj: nn.Module, tokens: torch.Tensor
    ) -> torch.Tensor:
        """Apply a projection head, handling BatchNorm's 2D requirement.

        BatchNorm1d expects (N, C) input. Since we gather tokens from
        potentially scattered positions, they are already (N, D) — no
        reshape needed.
        """
        return proj(tokens)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    def get_sentence_embedding_dimension(self) -> int:
        """Return the shared output dimensionality."""
        return self.shared_dim

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

        # Save weights
        if safe_serialization:
            save_safetensors_model(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

        logger.info("Saved AdapterModule to %s", output_path)

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
    ) -> AdapterModule:
        """Load a saved AdapterModule from disk.

        Parameters
        ----------
        model_name_or_path : str
            Path to directory containing saved module files.

        Returns
        -------
        AdapterModule
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

        # Load weights
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

        logger.info("Loaded AdapterModule from %s", model_name_or_path)
        return module

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"AdapterModule("
            f"text={self.text_input_dim}→{self.shared_dim}, "
            f"omics={self.omics_input_dim}→{self.shared_dim}, "
            f"hidden={self.hidden_dim})"
        )
