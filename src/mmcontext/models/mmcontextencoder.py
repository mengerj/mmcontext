"""
MMContextEncoder and MMContextProcessor

Custom encoder and processor for multimodal embedding generation with SentenceTransformers framework. Currently the second modality is handled
by MiniOmicsModel, a lookup-only encoder that mimics the behavior of AutoModel outputs. A dataset should contain sample_ids and corresponding numeric vectors.
These vectors were precomputed using different models and represent a numeric representation of a sample, based on it's measured data.


Sub-modules
-----------
MMContextProcessor
    Joint tokenizer that can digest either plain captions or *omics* sample-IDs.

MMContextEncoder
    Dual-tower Sentence-Transformers backbone with

    * text side   … any Hugging-Face AutoModel
    * omics side  … MiniOmicsModel (lookup-only)
    * adapters    … feed-forward projection heads (applied at token level)
    * pooling     … SentenceTransformers Pooling module for sentence embeddings

"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model
from sentence_transformers.models import Pooling
from transformers import AutoModel, AutoTokenizer

from mmcontext.file_utils import (
    build_embedding_df,
    collect_unique_links,
    download_and_extract_links,
)

from .adapters import AdapterModule
from .omicsencoder import MiniOmicsModel
from .onehot import OneHotTextEncoder

logger = logging.getLogger(__name__)

_PREFIX = "sample_idx:"  # special token that marks omics IDs


# --------------------------------------------------------------------------- #
# 1.  Processor
# --------------------------------------------------------------------------- #
class MMContextProcessor:
    """
    Joint tokenizer for caption-and-omics batches.

    Initially can be created as a text-only processor, with the ability to add
    omics processing capabilities later via register_initial_embeddings.

    Parameters
    ----------
    text_encoder_name : str
        Name or path of a Hugging-Face checkpoint whose tokenizer is used for
        normal captions.
    omics_lookup : Dict[str, int], optional
        Maps **prefixed** ``sample_idx`` strings to integer indices in the
        omics embedding matrix (row numbers). If not provided, the processor
        will only handle text inputs.
    prefix : str, optional
        Tag that distinguishes omics IDs from text (default: ``"sample_idx:"``).
        Only used if omics_lookup is provided.

    Examples
    --------
    >>> # Text-only processor
    >>> proc = MMContextProcessor("bert-base-uncased")
    >>> batch = ["A photo of a cat.", "Another text."]
    >>> enc = proc.tokenize(batch)
    >>> enc.keys()
    dict_keys(['input_ids', 'attention_mask', 'omics_text_info'])

    >>> # With omics capabilities
    >>> proc = MMContextProcessor("bert-base-uncased", lookup_dict)
    >>> batch = ["sample_idx:42", "A photo of a cat."]
    >>> enc = proc.tokenize(batch)
    >>> enc.keys()
    dict_keys(['pixel_values', 'input_ids', 'attention_mask', 'omics_text_info'])
    """

    def __init__(
        self,
        text_encoder_name: str,
        omics_lookup: dict[str, int] = None,
        *,
        prefix: str = _PREFIX,
    ) -> None:
        if text_encoder_name == "one_hot":
            self.text_tok = None  # no HF tokenizer
            self._sentence2id: dict[str, int] = {}
            self._next_id = 1  # 0 is reserved for padding
        else:
            self.text_tok = AutoTokenizer.from_pretrained(text_encoder_name)
        self.lookup = omics_lookup
        self.prefix = prefix
        self._text_only = omics_lookup is None

    def update_omics_lookup(self, omics_lookup: dict[str, int], prefix: str = None):
        """
        Update the processor with omics lookup capabilities.

        Parameters
        ----------
        omics_lookup : Dict[str, int]
            Maps **prefixed** sample_idx strings to integer indices.
        prefix : str, optional
            Tag that distinguishes omics IDs from text. If not provided,
            the existing prefix will be used.
        """
        self.lookup = omics_lookup
        if prefix is not None:
            self.prefix = prefix
        self._text_only = False

    # --------------------------------------------------------------------- #
    # public helpers
    # --------------------------------------------------------------------- #
    def __call__(self, *args, **kw):
        """Alias for :meth:`tokenize`."""
        return self.tokenize(*args, **kw)

    # --------------------------------------------------------------------- #
    # tokenize
    # --------------------------------------------------------------------- #
    def tokenize(
        self,
        texts: Sequence[str | int | np.ndarray | dict],
        *,
        padding: str | bool = True,
        **tok_kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize a batch of captions and/or omics identifiers.

        * **Text-only mode** (no lookup table registered)\
          → every element must be a string and is tokenised by the HF tokenizer.
        * **Bimodal mode** (lookup table present)\
          → elements that *start* with ``self.prefix`` are treated as omics
          strings.  The prefix is stripped, the remainder is split by
          whitespace, and each resulting token is looked-up **individually**.
          Thus a sample may contain one or many omics IDs.

        The method always returns

        ```
        input_ids, attention_mask,                  # ⟵ text part (if any)
        pixel_values              (B, max_omics_len),  # ⟵ PAD-filled
        omics_attention_mask   (B, max_omics_len),  # 1 = real, 0 = PAD
        omics_text_info        (B,)                 # 1 = text, 0 = omics
        ```
        even when a sample holds only a single omics ID.

        Raises
        ------
        TypeError
            If an element is not a string.
        KeyError
            If an omics token is missing from the lookup table.
        """
        # ---------- NEW ----------
        is_one_hot = self.text_tok is None  # set in __init__ when encoder == "one_hot"
        # --------------------------

        # ----------------------------------------------------------------- #
        # TEXT-ONLY MODE
        # ----------------------------------------------------------------- #
        if self._text_only:
            for i, item in enumerate(texts):
                if not isinstance(item, str):
                    raise TypeError(f"In text-only mode all inputs must be strings (got {type(item)} at position {i}).")
            # ---------- NEW ----------
            if is_one_hot:
                ids = torch.tensor(
                    [self._sentence2id.setdefault(t, self._next_id + i) for i, t in enumerate(texts)], dtype=torch.long
                ).unsqueeze(1)  # (B, 1)
                self._next_id += len(texts)  # grow vocab
                tok_out = {
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                }
            else:
                tok_out = self.text_tok(texts, padding=padding, return_tensors="pt", **tok_kwargs)
            # tok["attention_mask"] = tok["attention_mask"].bool()
            tok_out["omics_text_info"] = torch.ones(len(texts), dtype=torch.int8)
            return tok_out

        # ----------------------------------------------------------------- #
        # BIMODAL MODE
        # ----------------------------------------------------------------- #
        pad_id = 0  # PAD for omics
        text_vals = []  # raw caption strings
        text_pos = []  # indices of caption samples
        omics_rows = [[] for _ in range(len(texts))]  # per-sample ID list

        for i, item in enumerate(texts):
            if isinstance(item, str) and item.startswith(self.prefix):
                # strip prefix, split on whitespace → one or many omics IDs
                id_strs = item[len(self.prefix) :].strip().split()
                if not id_strs:
                    raise ValueError(f"No omics IDs found after prefix in sample {i}")

                for tok in id_strs:
                    try:
                        omics_rows[i].append(self.lookup[tok])
                    except KeyError as err:
                        raise KeyError(f"Omics ID '{tok}' not in lookup table") from err
            elif isinstance(item, str):
                text_vals.append(item)
                text_pos.append(i)
            else:
                raise TypeError(f"Unsupported element type {type(item)} at position {i}; expected strings.")

        batch: dict[str, torch.Tensor] = {}

        # -------------- text branch --------------------------------------
        if text_vals:
            if is_one_hot:
                ids = torch.tensor(
                    [self._sentence2id.setdefault(t, self._next_id + i) for i, t in enumerate(text_vals)],
                    dtype=torch.long,
                ).unsqueeze(1)  # (n_text, 1)
                self._next_id += len(text_vals)
                tok_out = {
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                }
            else:
                tok_out = self.text_tok(text_vals, padding=padding, return_tensors="pt", **tok_kwargs)
            for k, v in tok_out.items():  # v shape: (n_text, L)
                full = [torch.zeros_like(v[0])] * len(texts)
                for src, dst in enumerate(text_pos):
                    full[dst] = v[src]
                batch[k] = torch.stack(full)  # (B, L)

        # -------------- omics branch -------------------------------------
        if any(omics_rows):
            max_len = max(len(seq) for seq in omics_rows) or 1

            ids = torch.full((len(texts), max_len), pad_id, dtype=torch.long)
            mask = torch.zeros((len(texts), max_len), dtype=torch.bool)

            for i, row in enumerate(omics_rows):
                if row:  # sample i is omics
                    ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
                    mask[i, : len(row)] = True

            batch["pixel_values"] = ids  # (B, max_len)
            batch["omics_attention_mask"] = mask  # (B, max_len)

        # -------------- modality indicator -------------------------------
        indi = torch.zeros(len(texts), dtype=torch.int8)  # 0 = omics
        indi[text_pos] = 1  # 1 = text
        batch["omics_text_info"] = indi

        return batch


# --------------------------------------------------------------------------- #
# 2.  Encoder
# --------------------------------------------------------------------------- #
class MMContextEncoder(nn.Module):
    """
    Dual-tower encoder that acts as its own tokenizer.

    Can be initialized in three modes:
    1. Text-only mode: Only processes text inputs
    2. Bimodal mode with preloaded omics embeddings
    3. Bimodal mode that will be completed later via register_initial_embeddings

    The adapter layers can be optionally included for dimensionality reduction.
    Adapters are applied at the token level, and pooling is performed using
    SentenceTransformers Pooling module for better compatibility.

    Parameters
    ----------
    text_encoder_name : str | nn.Module
        Name of HuggingFace model or a pre-initialized model
    omics_embedding : Optional[np.ndarray], optional
        Precomputed embeddings for omics data. If None, the model starts as
        text-only and can be extended via register_initial_embeddings.
    adapter_hidden_dim : int or None, optional
        Hidden dimension of the adapter layers. If None, no adapter is used.
    adapter_output_dim : int or None, optional
        Output dimension of the adapter layers. If None, no adapter is used.
        If adapter_hidden_dim is provided but this is None, output_dim will
        match the text encoder's hidden dimension.
    freeze_text_encoder : bool, optional
        Whether to freeze the text encoder weights. Defaults to False.
    unfreeze_last_n_layers : int, optional
        Number of layers to unfreeze from the end of the text encoder.
        Only applies if freeze_text_encoder is True.
    processor : MMContextProcessor | None, optional
        Pre-initialized processor. If None, one will be created.
    registered_data_origin : str | None, optional
        Type of omics data representation. Must be one of:
        ["unregistered", "pca", "hvg", "scvi_fm", "geneformer"].
        Defaults to "unregistered".
    registered_input_dim : int | None, optional
        Input dimension of registered data. Required when loading a model
        that was previously registered with data to initialize adapters correctly.
    output_token_embeddings : bool, optional
        Whether to include token embeddings in the output. Defaults to False. When using
        SentenceTransformers. Training such a model is slower, but allows to continue working with the token embeddings.
    train_lookup : bool, optional
        Whether to train the lookup table. Defaults to False, as we have precomputed representations in the lookup,
        that we don't want to modify at this point.
    pooling_mode : str, optional
        Pooling strategy to use. Defaults to "mean". Options: "mean", "cls", "max", etc.
        See SentenceTransformers Pooling documentation for all options.
    """

    VALID_DATA_ORIGINS = ["unregistered", "pca", "hvg", "scvi_fm", "geneformer", "random"]

    def __init__(
        self,
        text_encoder_name: str | nn.Module,
        omics_embedding: np.ndarray | None = None,
        *,
        adapter_hidden_dim: int | None = None,
        adapter_output_dim: int | None = None,
        freeze_text_encoder: bool = False,
        unfreeze_last_n_layers: int = 0,
        processor: MMContextProcessor | None = None,
        registered_data_origin: str = "unregistered",
        registered_input_dim: int | None = None,
        output_token_embeddings: bool = False,
        train_lookup: bool = False,
        pooling_mode: str = "mean",
    ) -> None:
        super().__init__()
        if registered_data_origin not in self.VALID_DATA_ORIGINS:
            raise ValueError(f"registered_data_origin must be one of {self.VALID_DATA_ORIGINS}")

        # Store the parameters
        self.text_encoder_name = text_encoder_name
        self.omics_input_dim = None if omics_embedding is None else omics_embedding.shape[1]
        self.adapter_hidden_dim = adapter_hidden_dim
        self.adapter_output_dim = adapter_output_dim
        self.freeze_text_encoder = freeze_text_encoder
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self._has_omics = omics_embedding is not None
        self._registered_data_origin = registered_data_origin
        self._registered_input_dim = registered_input_dim
        self.output_token_embeddings = output_token_embeddings
        self.train_lookup = train_lookup
        self.pooling_mode = pooling_mode

        # Determine if adapters should be used
        self._use_adapters = adapter_hidden_dim is not None or adapter_output_dim is not None

        # ------------------------------------------------ text tower
        if isinstance(text_encoder_name, nn.Module):
            self.text_encoder = text_encoder_name
        elif text_encoder_name == "one_hot":
            # Default: 1 million sentence slots, match adapter_output_dim or fall back to 768
            embed_dim = adapter_output_dim or 768
            self.text_encoder = OneHotTextEncoder(
                num_sentences=1_000_000, embed_dim=embed_dim, trainable=not freeze_text_encoder
            )
        else:
            self.text_encoder = AutoModel.from_pretrained(text_encoder_name)

        text_hidden_dim = self.text_encoder.config.hidden_size

        # Determine output dimension
        if self._use_adapters:
            self._output_dim = adapter_output_dim if adapter_output_dim is not None else text_hidden_dim
        else:
            self._output_dim = text_hidden_dim

        # Setup text adapter if requested
        if self._use_adapters:
            self.text_adapter = AdapterModule(
                input_dim=text_hidden_dim,
                hidden_dim=adapter_hidden_dim,
                output_dim=self._output_dim,
            )
        else:
            self.text_adapter = None

        # ------------------------------------------------ pooling layer
        # Use SentenceTransformers Pooling for sentence embeddings
        self.pooling = Pooling(
            word_embedding_dimension=self._output_dim,
            pooling_mode=pooling_mode,
        )

        # ------------------------------------------------ omics tower
        if omics_embedding is not None:
            self._init_omics(omics_embedding)
        elif registered_data_origin != "unregistered" and registered_input_dim:
            # Initialize adapter for registered data without embedding matrix
            self.omics_encoder = None
            if self._use_adapters:
                self.omics_adapter = AdapterModule(
                    input_dim=registered_input_dim,
                    hidden_dim=adapter_hidden_dim,
                    output_dim=self._output_dim,
                )
            else:
                self.omics_adapter = None
            self._registered_input_dim = registered_input_dim
        else:
            self.omics_encoder = None
            self.omics_adapter = None

        # ------------------------------------------------ processor
        if processor is None:
            omics_lookup = None
            if omics_embedding is not None:
                omics_lookup = {
                    i + 1 for i in range(omics_embedding.shape[0])
                }  # add +1 to make sure the idx 0 is reserved for padding

            processor = MMContextProcessor(
                text_encoder_name=text_encoder_name if isinstance(text_encoder_name, str) else "prajjwal1/bert-tiny",
                omics_lookup=omics_lookup,
            )

        self.processor = processor
        self._manage_text_encoder_freezing()

    def _init_omics(self, matrix: np.ndarray):
        """Initialize the omics tower given an embedding matrix."""
        self.omics_encoder = MiniOmicsModel.from_numpy(matrix)
        self.omics_encoder.embeddings.weight.requires_grad = self.train_lookup
        # Initialize or preserve adapter layers
        self._init_or_preserve_adapters(matrix.shape[1])

        self.omics_input_dim = matrix.shape[1]
        self._registered_input_dim = matrix.shape[1]
        self._has_omics = True

    def _update_omics(self, matrix: np.ndarray):
        """Update the omics encoder with a new embedding matrix."""
        # Only update the omics encoder, not the adapter
        self.omics_encoder = MiniOmicsModel.from_numpy(matrix)
        # freeze the embeddings (default) to retrieve exactly the values from the lookup
        self.omics_encoder.embeddings.weight.requires_grad = self.train_lookup
        self._has_omics = True

    def _init_or_preserve_adapters(self, input_dim: int):
        """Initialize or preserve adapter layers for omics input.

        This method handles adapter logic separately from omics encoder creation,
        ensuring adapters are preserved when reloading models.

        Parameters
        ----------
        input_dim : int
            Input dimension for the adapter
        """
        # If we already have an adapter, preserve it
        has_adapter = hasattr(self, "omics_adapter") and self.omics_adapter is not None

        if not has_adapter and self._use_adapters:
            # Create a new adapter only if adapters are requested
            self.omics_adapter = AdapterModule(
                input_dim=input_dim,
                hidden_dim=self.adapter_hidden_dim,
                output_dim=self._output_dim,
            )
        elif not self._use_adapters:
            # Ensure no adapter if not requested
            self.omics_adapter = None

    def _manage_text_encoder_freezing(self):
        """Freezes all parameters in the text encoder if required, and optionally unfreezes the last n layers."""
        if self.freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

            if self.unfreeze_last_n_layers > 0:
                # For BERT-like models
                if hasattr(self.text_encoder, "encoder") and hasattr(self.text_encoder.encoder, "layer"):
                    layers = self.text_encoder.encoder.layer[-self.unfreeze_last_n_layers :]
                # For RoBERTa-like models
                elif hasattr(self.text_encoder, "roberta") and hasattr(self.text_encoder.roberta, "encoder"):
                    layers = self.text_encoder.roberta.encoder.layer[-self.unfreeze_last_n_layers :]
                else:
                    logger.warning(
                        f"Unsupported architecture for {self.text_encoder_name}. Cannot unfreeze last {self.unfreeze_last_n_layers} layers."
                    )
                    return

                for layer in layers:
                    for param in layer.parameters():
                        param.requires_grad = True

    # ---------------------------------------------------------------------
    def tokenize(
        self,
        texts: Sequence[str],
        *,
        padding: str | bool = True,
        **tok_kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize texts and/or omics identifiers.

        Parameters
        ----------
        texts : sequence of str
            Texts and/or omics identifiers.
        padding : str or bool, optional
            Forwarded to the underlying HF tokenizer (default: ``True``).
        **tok_kwargs
            Additional keyword args forwarded to the tokenizer.

        Returns
        -------
        dict
            Tokenized features compatible with the forward method.
        """
        if self.processor is None:
            raise AttributeError("Processor not initialized")
        return self.processor.tokenize(texts, padding=padding, **tok_kwargs)

    # --------------------------------------------------------------------- #
    # forward
    # --------------------------------------------------------------------- #
    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Embed a batch and maintain original ordering.

        New architecture:
        1. Get token embeddings from encoders
        2. Apply adapters at token level (if enabled)
        3. Use SentenceTransformers Pooling for sentence embeddings
        4. Optionally return token embeddings

        Parameters
        ----------
        features : dict
            Output of tokenize method, containing text features and optionally
            omics features.

        Returns
        -------
        torch.Tensor or dict
            If return_tensor=True in features, returns a tensor of sentence embeddings directly.
            Otherwise, dict with 'sentence_embedding' and optionally 'token_embeddings'.

        Raises
        ------
        RuntimeError
            If omics features are provided but the model hasn't been initialized
            with omics capabilities.
        """
        # ------------------------------------------------------------------ #
        # housekeeping
        # ------------------------------------------------------------------ #
        return_tensor = features.pop("return_tensor", False)
        dev, dtype = next(self.parameters()).device, next(self.parameters()).dtype

        # move incoming tensors to the module device
        for k, v in features.items():
            if torch.is_tensor(v):
                features[k] = v.to(dev)

        batch_size = features["omics_text_info"].size(0)
        text_mask = features["omics_text_info"] == 1
        omics_mask = ~text_mask

        # ------------------------------------------------------------------ #
        # Prepare unified token embeddings tensor
        # ------------------------------------------------------------------ #
        # Determine maximum sequence length across modalities
        txt_len = 1
        om_len = 1
        if "input_ids" in features:
            txt_len = features["input_ids"].size(1)
        if "pixel_values" in features:
            om_len = features["pixel_values"].size(1)
        max_len = max(txt_len, om_len)

        # Initialize unified tensors
        token_embeddings = torch.zeros(batch_size, max_len, self._output_dim, device=dev, dtype=dtype)
        attention_mask = torch.zeros(batch_size, max_len, device=dev, dtype=torch.long)

        # ------------------------------------------------------------------ #
        # text branch
        # ------------------------------------------------------------------ #
        if text_mask.any():
            token_type_slice = (
                features["token_type_ids"][text_mask]  # use the mask
                if "token_type_ids" in features and features["token_type_ids"] is not None
                else None  # otherwise skip arg
            )
            txt_out = self.text_encoder(
                input_ids=features["input_ids"][text_mask],
                attention_mask=features["attention_mask"][text_mask],
                token_type_ids=token_type_slice,
            )

            # Get token embeddings and apply adapter if present
            txt_tokens = txt_out.last_hidden_state.to(dtype)  # (n_text, L_txt, H)
            if self.text_adapter is not None:
                # Apply adapter to each token
                B, L, H = txt_tokens.shape
                txt_tokens = self.text_adapter(txt_tokens.view(-1, H)).view(B, L, self._output_dim)

            # Place in unified tensor
            L_txt = txt_tokens.size(1)
            # Ensure dtype consistency for AMP compatibility
            txt_tokens = txt_tokens.to(token_embeddings.dtype)
            token_embeddings[text_mask, :L_txt] = txt_tokens
            attention_mask[text_mask, :L_txt] = features["attention_mask"][text_mask]

        # ------------------------------------------------------------------ #
        # omics branch
        # ------------------------------------------------------------------ #
        if omics_mask.any():
            if not self._has_omics:
                raise RuntimeError("Call `register_initial_embeddings()` first.")

            om_out = self.omics_encoder(  # (n_omics, L_om, H)
                input_ids=features["pixel_values"][omics_mask]
            )

            # Get token embeddings and apply adapter if present
            om_tokens = om_out.last_hidden_state.to(dtype)  # (n_omics, L_om, H)
            if self.omics_adapter is not None:
                # Apply adapter to each token
                B, L, H = om_tokens.shape
                om_tokens = self.omics_adapter(om_tokens.view(-1, H)).view(B, L, self._output_dim)

            # IMPORTANT: Zero out padded positions after adapter application
            # This ensures that padded tokens remain zero even after going through the adapter
            omics_attn_mask = features["omics_attention_mask"][omics_mask]  # (n_omics, L_om)
            om_tokens = om_tokens * omics_attn_mask.unsqueeze(-1).to(dtype)  # Broadcast and mask

            # Place in unified tensor
            L_om = om_tokens.size(1)
            # Ensure dtype consistency for AMP compatibility
            om_tokens = om_tokens.to(token_embeddings.dtype)
            token_embeddings[omics_mask, :L_om] = om_tokens
            attention_mask[omics_mask, :L_om] = omics_attn_mask.long()

        # ------------------------------------------------------------------ #
        # Apply pooling to get sentence embeddings
        # ------------------------------------------------------------------ #
        pooling_features = {
            "token_embeddings": token_embeddings,
            "attention_mask": attention_mask,
        }
        pooled_features = self.pooling(pooling_features)
        sentence_embeddings = pooled_features["sentence_embedding"]

        # ------------------------------------------------------------------ #
        # pack up & return
        # ------------------------------------------------------------------ #
        if return_tensor:
            return sentence_embeddings

        result = {
            "sentence_embedding": sentence_embeddings,
        }

        if self.output_token_embeddings:
            result["token_embeddings"] = token_embeddings
            result["attention_mask"] = attention_mask
            # Optional: add modality information
            modality_ids = torch.full((batch_size, max_len), 2, device=dev, dtype=torch.long)  # 2 = PAD
            modality_ids[text_mask, :txt_len] = 0  # text
            modality_ids[omics_mask, :om_len] = 1  # omics
            result["modality_ids"] = modality_ids

        return result

    def get_sentence_embedding_dimension(self) -> int:
        """
        Returns the dimension of the final sentence embedding.

        Returns
        -------
        int
            The dimension of the final sentence embedding.
        """
        return self._output_dim

    # Alias for SentenceTransformers compatibility
    def _get_sentence_embedding_dimension(self) -> int:
        """Alias for get_sentence_embedding_dimension for SentenceTransformers compatibility."""
        return self.get_sentence_embedding_dimension()

    def _get_config_dict(self) -> dict:
        """
        Returns a configuration dictionary without the omics embedding matrix.

        Returns
        -------
        dict
            A config with essential hyperparameters.
        """
        return {
            "text_encoder_name": self.text_encoder_name
            if isinstance(self.text_encoder_name, str)
            else "prajjwal1/bert-tiny",
            "adapter_hidden_dim": self.adapter_hidden_dim,
            "adapter_output_dim": self.adapter_output_dim,
            "freeze_text_encoder": self.freeze_text_encoder,
            "unfreeze_last_n_layers": self.unfreeze_last_n_layers,
            "registered_data_origin": self._registered_data_origin,
            "registered_input_dim": self._registered_input_dim,
            "output_token_embeddings": self.output_token_embeddings,
            "train_lookup": self.train_lookup,
            "pooling_mode": self.pooling_mode,
        }

    def save(self, output_path: str, safe_serialization: bool = True) -> None:
        """
        Saves the model configuration and state dict, excluding the omics embedding matrix.

        Parameters
        ----------
        output_path : str
            Directory to save model files.
        safe_serialization : bool, optional
            If True, use safetensors; else use torch.save.
        """
        os.makedirs(output_path, exist_ok=True)

        # Save config without omics matrix
        config_path = os.path.join(output_path, "config.json")
        with open(config_path, "w") as fOut:
            json.dump(self._get_config_dict(), fOut, indent=2)

        # Get state dict and filter omics embeddings
        state = self.state_dict()
        if self._has_omics:
            # Remove omics embeddings but keep adapter weights
            state = {k: v for k, v in state.items() if not k.startswith("omics_encoder.embeddings")}

        if safe_serialization:
            model_path = os.path.join(output_path, "model.safetensors")
            # Create a temporary model with the filtered state dict
            temp_model = MMContextEncoder(
                text_encoder_name=self.text_encoder_name,
                adapter_hidden_dim=self.adapter_hidden_dim,
                adapter_output_dim=self.adapter_output_dim,
                registered_data_origin=self._registered_data_origin,
                registered_input_dim=self._registered_input_dim,
                output_token_embeddings=self.output_token_embeddings,
                train_lookup=self.train_lookup,
                pooling_mode=self.pooling_mode,
            )
            # Load the filtered state dict into the temporary model
            temp_model.load_state_dict(state, strict=False)
            # Save using safetensors
            save_safetensors_model(temp_model, model_path)
        else:
            model_path = os.path.join(output_path, "pytorch_model.bin")
            torch.save(state, model_path)

    @staticmethod
    def load(input_path: str, safe_serialization: bool = True):
        """
        Loads the model from disk.

        Parameters
        ----------
        input_path : str
            Directory where the model was saved.
        safe_serialization : bool, optional
            If True, expects safetensors format; else a PyTorch bin.

        Returns
        -------
        MMContextEncoder
            The loaded model instance, ready for data registration.
        """
        config_path = os.path.join(input_path, "config.json")
        with open(config_path) as fIn:
            cfg = json.load(fIn)

        # Create model from config without omics embedding
        model = MMContextEncoder(**cfg)

        # Load state dict
        if safe_serialization and os.path.exists(os.path.join(input_path, "model.safetensors")):
            load_safetensors_model(model, os.path.join(input_path, "model.safetensors"))
        else:
            state_dict = torch.load(os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"))
            model.load_state_dict(state_dict, strict=False)

        if model._registered_data_origin != "unregistered":
            logger.warning(
                f"Loaded encoder was registered for '{model._registered_data_origin}' data. "
                "Call register_initial_embeddings() with compatible data before using it."
            )

        return model

    @property
    def tokenizer(self) -> MMContextProcessor:
        """Convenience property returning the underlying processor object, which includes the text tokenizer and omics data processor."""
        if self.processor is None:
            raise AttributeError("Processor not initialized")
        return self.processor

    def _estimate_memory(self, num_rows: int, dim: int) -> float:
        """
        Estimate memory footprint in gigabytes for a float32 matrix.

        Parameters
        ----------
        num_rows : int
            Number of rows (samples).
        dim : int
            Embedding dimension.

        Returns
        -------
        float
            Memory in gigabytes.
        """
        bytes_total = num_rows * dim * 4  # float32 = 4 bytes
        return bytes_total / 1024**3

    # --------------------------------------------------------------------------- #
    #  register_initial_embeddings
    # --------------------------------------------------------------------------- #
    def register_initial_embeddings(
        self,
        data: pd.DataFrame | HFDataset | Mapping[str, Sequence[float]],
        data_origin: str,
        *,
        id_col: str = "token",
        emb_col: str | None = "embedding",
        return_added: bool = False,
    ) -> dict[str, int]:
        """
        Add *gene / sample* tokens and their initial embeddings to the encoder.

        Parameters
        ----------
        data
            • **pandas.DataFrame / HF Dataset** – `id_col` has the token strings,
            `emb_col` holds a vector (np.ndarray, list, or tuple).
            • **Mapping**  – ``{token_str: embedding_vector}``.#
        data_origin
            Tag describing how the numeric representation was generated
            (``"pca"``, ``"hvg"``, ``"scvi_fm"``, ``"geneformer"`` …).
        id_col, emb_col
            Column names for DataFrame / HF-Dataset input.
        return_added
            If True, return a dictionary of the newly added tokens and their
            corresponding indices in the omics embedding matrix.

        Returns
        -------
        dict
            Only the **new** tokens inserted this call,
            e.g. ``{"EGFR": 12345, "KRAS": 12346}``.
        """
        # ---------- 0. Make sure data origin is compatible -------------------
        if data_origin not in self.VALID_DATA_ORIGINS:
            raise ValueError(f"{data_origin!r} is not in {self.VALID_DATA_ORIGINS}")
        if self._registered_data_origin != "unregistered" and data_origin != self._registered_data_origin:
            raise ValueError(
                f"Model already registered for {self._registered_data_origin!r}; "
                f"cannot register {data_origin!r} as well."
            )
        self._registered_data_origin = data_origin

        # -------- 1. normalise input to two Python lists --------------------
        if isinstance(data, pd.DataFrame):
            tokens = data[id_col].tolist()
            vectors = np.vstack(data[emb_col].to_numpy())
        elif isinstance(data, HFDataset):
            tokens = list(data[id_col])
            vectors = np.vstack(data[emb_col])
        elif isinstance(data, Mapping):
            tokens, vectors = zip(*data.items(), strict=False)
            vectors = np.vstack(vectors)
        else:
            raise TypeError("data must be a pandas.DataFrame, HuggingFace Dataset or Mapping")

        # -------- 2. prepare current registry ------------------------------
        if not hasattr(self, "_omics_lookup"):
            self._omics_lookup = {}
            old_matrix = np.zeros((1, vectors.shape[1]), dtype=np.float32)
        else:
            old_matrix = self.omics_encoder.embeddings.weight.detach().cpu().numpy()  # (V_old, H)

        # -------- 3. merge, skipping duplicates ----------------------------
        new_rows, added, skipped = [], {}, []
        for tok, vec in zip(tokens, vectors, strict=False):
            vec_dim = vec.shape[0]
            if self._registered_input_dim is not None:
                if vec_dim != self._registered_input_dim:
                    raise ValueError(
                        f"Vector dimension mismatch: expected {self._registered_input_dim}, got {vec_dim}."
                    )
            else:
                # first registration – remember expected dim
                self._registered_input_dim = vec_dim
                self._init_or_preserve_adapters(vec_dim)
            if tok in self._omics_lookup:  # already present → skip
                skipped.append(tok)
                continue
            added[tok] = len(old_matrix) + len(new_rows)
            new_rows.append(np.asarray(vec, dtype=np.float32))

        if skipped:
            logger.warning(
                "register_initial_embeddings: %d duplicate tokens skipped: %s",
                len(skipped),
                ", ".join(skipped[:10]) + ("…" if len(skipped) > 10 else ""),
            )

        if not new_rows:  # nothing to do
            if return_added:
                return {}
            return

        # -------- 4. rebuild embedding matrix & lookup ---------------------
        full_matrix = np.vstack([old_matrix, np.vstack(new_rows)])
        self._omics_lookup.update(added)

        if not self._has_omics:
            self._init_omics(full_matrix)
        else:
            self._update_omics(full_matrix)

        # let the processor know
        if hasattr(self, "processor") and hasattr(self.processor, "update_omics_lookup"):
            self.processor.update_omics_lookup(self._omics_lookup)

        added_gb = self._estimate_memory(full_matrix.shape[0], full_matrix.shape[1])
        logger.info(
            f"Registered {full_matrix.shape[0]} new numeric samples (total {full_matrix.shape[0]}). "
            f"≈{added_gb:.3f} GiB added. (Assuming float32 precision.)"
        )

        if return_added:
            return added
        return

    def random_initial_embeddings(
        self,
        tokens: Sequence[str] | pd.DataFrame | HFDataset | DatasetDict | Mapping[str, Any],
        *,
        dim: int = 64,
        rng_seed: int | None = None,
        id_col: str = "token",
        data_origin: str = "random",
    ) -> dict[str, int]:
        """
        Register *tokens* with Gaussian-random vectors instead of real embeddings.

        Parameters
        ----------
        tokens
            • List/tuple of token strings **or**
            • A pandas / HF dataset / mapping containing a column or key named
            *id_col*.
            Any duplicates are collapsed.
        dim
            Embedding dimensionality for every generated vector.
        rng_seed
            Reproducible seed.  If *None* → numpy default RNG.
        id_col
            Column/key that holds the token strings when *tokens* is a table.
        data_origin
            Passed straight through to ``register_initial_embeddings``.
        """
        # 1. flatten to a list[str] ------------------------------------------------
        if isinstance(tokens, list | tuple):
            flat = list(tokens)
        elif isinstance(tokens, pd.DataFrame):
            flat = tokens[id_col].tolist()
        elif isinstance(tokens, HFDataset | DatasetDict):
            if isinstance(tokens, DatasetDict):
                flat = [row[id_col] for split in tokens.values() for row in split]
            else:
                flat = tokens[id_col]
        elif isinstance(tokens, Mapping):
            flat = list(tokens.keys())
        else:
            raise TypeError("Unsupported type for *tokens*")

        flat = sorted(set(map(str, flat)))  # remove duplicates + stable order
        if len(flat) == 0:
            logger.warning("No tokens given – nothing to register.")
            return {}

        # 2. random matrix --------------------------------------------------------
        rng = np.random.default_rng(rng_seed)
        mat = rng.standard_normal((len(flat), dim)).astype(np.float32)

        df = pd.DataFrame({"token": flat, "embedding": list(mat)})

        # 3. reuse the existing pipeline -----------------------------------------
        return self.register_initial_embeddings(
            df,
            data_origin=data_origin,
            id_col="token",
            emb_col="embedding",
            return_added=True,
        )

    # ------------------------------------------------------------------
    # public helper -----------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def get_initial_embeddings(
        hf_dataset: DatasetDict | HFDataset,
        *,
        layer_key: str | None = None,
        axis: Literal["obs", "var"] = "obs",
        download_dir: str | Path = "../../data/downloaded_chunks",
        extract_zip: bool = True,
        overwrite: bool = False,
        link_column: str = "share_link",
    ) -> tuple[pd.DataFrame, dict[str, Path]]:
        """
        Download all embedding chunks referenced in *hf_dataset* and return in a format suitable for registration.

        This assumes the dataset has a column (specified by link_column) in each split,
        which contains either share links to Nextcloud files or local file paths.
        For URLs, the function will download and optionally extract the files.
        For local paths, they are used directly without downloading.

        Parameters
        ----------
        hf_dataset
            A Hugging Face :pyclass:`~datasets.DatasetDict` that contains one or
            more splits *and* a link/path column in each split.
        layer_key
            Name of the embedding to pull –
            * ``.obsm[layer_key]`` if *axis="obs"*
            * ``.varm[layer_key]`` if *axis="var"*
        axis
            ``"obs"`` → use ``adata.obs.index``
            ``"var"`` → use ``adata.var.index``.
        download_dir
            Local root folder where downloaded chunk files (ZIPs or extracted stores)
            will be materialized. Ignored for local paths.
        extract_zip
            If *True* unpack Nextcloud ZIP downloads into ``chunk_<n>.zarr/``.
            If *False* keep the ZIP and let Zarr's *zip-store* backend read it
            directly (requires Zarr ≥ 2.16). Only applies to downloaded files.
        overwrite
            Re-download / re-extract even if the target already exists.
            Only applies to downloaded files.
        link_column
            Column name that stores the share links or local paths.

        Returns
        -------
        tuple[pandas.DataFrame, dict[str, Path]]
            A tuple containing:
            - DataFrame with two columns:
              * ``token`` - *obs* or *var* label (string)
              * ``embedding`` - 1-D :class:`numpy.ndarray` of floats
            - Path mapping from original links to actual file locations
        """
        # --------------------------------------------------------------
        # 1) collect & download unique share links
        # --------------------------------------------------------------
        links, _ = collect_unique_links(hf_dataset, link_column=link_column)
        path_map = download_and_extract_links(
            links,
            target_dir=download_dir,
            extract=extract_zip,
            overwrite=overwrite,
        )
        # if the layer key is none, this means that only the download step was needed. The model will be used as text only without initial embeddings.
        if layer_key is None:
            logger.info("No layer key provided, get_initial_embeddings() is returning empty DataFrame and path map.")
            return None, path_map

        # --------------------------------------------------------------
        # 2) build per-split DataFrames, then concat
        # --------------------------------------------------------------
        if isinstance(hf_dataset, HFDataset):
            hf_dataset = DatasetDict({hf_dataset.split: hf_dataset})
        split_frames: list[pd.DataFrame] = []
        for split_name, ds in hf_dataset.items():
            # translate split-specific links → local paths
            local_map = {lk: path_map[lk] for lk in ds["share_link"]}
            df = build_embedding_df(
                local_map,
                layer_key=layer_key,
                axis=axis,
            )
            df["split"] = split_name  # keep the provenance
            split_frames.append(df)

        full_df = pd.concat(split_frames, ignore_index=True)
        logger.info("Combined embedding DataFrame shape: %s", full_df.shape)
        print("Use the returned DataFrame to register the embeddings with `register_initial_embeddings()`.")
        return full_df, path_map

    def prepare_ds(
        self,
        ds: HFDataset | DatasetDict,
        primary_cell_sentence_col: str,
        *,  # keyword-only below
        prefix: bool = True,
        caption_col: str = "caption",
        positive_col: str = "positive",
        label_col: str = "label",
        negative_prefix: str = "negative",
        index_col: str = "sample_idx",
        keep_index_col: bool = False,
    ) -> HFDataset | DatasetDict:
        """Return a copy ready for SentenceTransformerTrainer.

        Parameters
        ----------
        ds : HFDataset | DatasetDict
            Input dataset to prepare
        primary_cell_sentence_col : str
            Column containing cell/sample representations. References one of the cell_sentence columns in the dataset, that you want to process.
            These will be tokenized by the omics part of the model, if prefix=True, or by the text part of the model, if prefix=False.
            These will be the primary output columns. For multiplets, negative samples will be chosen from this column, if they are sample indices. Other negatives
            are captions and are not modified.
            If prefix=True, these will be prefixed with the processor's prefix.
        prefix : bool, optional
            Whether to add the processor's prefix to primary_cell_sentence_col. If False, only subsetting is performed.
        caption_col : str, optional
            Name of the caption column for pairs
        positive_col : str, optional
            Name of the positive column for multiplets
        label_col : str, optional
            Name of the label column for pairs
        negative_prefix : str, optional
            Prefix for negative columns in multiplets
        index_col : str, optional
            Name of the index column for HFDataset. Only if index_col is provided, it will be left in the output.
        keep_index_col : bool, optional
            Whether to keep the index column in the output. Defaults to False. Is needed for the embedding workflow.

        Returns
        -------
        HFDataset | DatasetDict
            Processed dataset with appropriate columns and optional prefixes
        """

        def _add_prefix(tok: str) -> str:
            p = self.processor.prefix
            return tok if tok.startswith(p) else f"{p}{tok}"

        def _pref(batch, pref_col):
            if isinstance(pref_col, str):
                # Handle single column
                batch[pref_col] = [_add_prefix(t.strip()) for t in batch[pref_col]]
            else:
                # Handle list of columns (for backward compatibility)
                for col in pref_col:
                    batch[col] = [_add_prefix(t.strip()) for t in batch[col]]
            return batch

        def _resolve_negative_indices(split: HFDataset, negative_cols: list[str]) -> HFDataset:
            """Resolve sample indices in negative columns to actual values.

            Odd-numbered negatives (1, 3, 5, ...) map to positive column values.
            Even-numbered negatives (2, 4, 6, ...) map to anchor column values.
            """
            if not index_col or not negative_cols:
                return split

            import re

            # Create mappings from index to both positive and anchor values
            index_to_positive = {}
            index_to_anchor = {}

            for _, row in enumerate(split):
                if index_col in row:
                    idx_val = str(row[index_col])
                    if positive_col in row:
                        index_to_positive[idx_val] = row[positive_col]
                    if primary_cell_sentence_col in row:
                        index_to_anchor[idx_val] = row[primary_cell_sentence_col]

            def _resolve_negatives(batch):
                for neg_col in negative_cols:
                    if neg_col in batch:
                        # Extract the number from column name (e.g., "negative_1_idx" -> 1)
                        match = re.search(r"negative_(\d+)_idx", neg_col)
                        if match:
                            neg_num = int(match.group(1))
                            is_odd = neg_num % 2 == 1

                            resolved_values = []
                            for idx_value in batch[neg_col]:
                                if is_odd and idx_value in index_to_positive:
                                    resolved_values.append(index_to_positive[idx_value])
                                elif not is_odd and idx_value in index_to_anchor:
                                    resolved_values.append(index_to_anchor[idx_value])
                                else:
                                    # Fallback: keep original value if mapping not found
                                    resolved_values.append(idx_value)

                            batch[neg_col] = resolved_values
                return batch

            # Apply the resolution
            proc = split.map(_resolve_negatives, batched=True, desc="Resolving negative indices")

            # Rename columns to remove "_idx" suffix
            for neg_col in negative_cols:
                if neg_col in proc.column_names and neg_col.endswith("_idx"):
                    new_col_name = neg_col.replace("_idx", "")
                    proc = proc.rename_column(neg_col, new_col_name)

            return proc

        def _process_split(split: HFDataset) -> HFDataset:
            is_pairs = label_col in split.column_names and label_col != primary_cell_sentence_col
            is_multiplets = positive_col in split.column_names

            keep_cols = [primary_cell_sentence_col]

            if is_multiplets:
                keep_cols.append(positive_col)
                negative_cols = [c for c in split.column_names if c.startswith(negative_prefix)]
                # Add the renamed negative column names (without "_idx") to keep_cols
                renamed_negative_cols = [c.replace("_idx", "") if c.endswith("_idx") else c for c in negative_cols]
                keep_cols += renamed_negative_cols
            elif is_pairs:
                keep_cols += [caption_col, label_col]
            else:  # single – keep only the main column
                keep_cols = [primary_cell_sentence_col]

            if keep_index_col:
                keep_cols.append(index_col)

            # sanity checks --------------------------------------------------
            if primary_cell_sentence_col not in split.column_names:
                raise TypeError(f"Column '{primary_cell_sentence_col}' missing from dataset split.")

            if split.features[primary_cell_sentence_col].dtype != "string":
                raise TypeError(f"Column '{primary_cell_sentence_col}' must contain strings.")

            # prefix if requested + drop unused columns ----------------------
            if prefix:
                proc = split.map(
                    lambda b: _pref(b, primary_cell_sentence_col),
                    batched=True,
                    desc=f"Prefixing {primary_cell_sentence_col}",
                )
            else:
                proc = split

            # resolve negative indices to actual cell sentence values --------
            # (after prefix is added so resolved values also have prefix)
            if is_multiplets:
                proc = _resolve_negative_indices(proc, negative_cols)

            drop_cols = [c for c in proc.column_names if c not in keep_cols]
            if drop_cols:
                proc = proc.remove_columns(drop_cols)
            if is_pairs:
                proc = proc.rename_column(primary_cell_sentence_col, "sentence_1")
                if caption_col:
                    proc = proc.rename_column(caption_col, "sentence_2")
            elif is_multiplets:
                proc = proc.rename_column(primary_cell_sentence_col, "anchor")
            return proc

        if isinstance(ds, HFDataset):
            return _process_split(ds)
        elif isinstance(ds, DatasetDict):
            return DatasetDict({name: _process_split(s) for name, s in ds.items()})
        else:
            raise TypeError("prepare_ds expects a datasets.Dataset or DatasetDict")

    @property
    def registered_data_origin(self) -> str:
        """Get the type of omics data representation registered with this model."""
        return self._registered_data_origin

    @property
    def registered_input_dim(self) -> int | None:
        """Get the input dimension of registered data."""
        return self._registered_input_dim
