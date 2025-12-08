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

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model
from sentence_transformers.models import Module, Pooling
from transformers import AutoModel, AutoTokenizer

from .adapters import AdapterModule

# Import local dependencies using relative imports (works with package structure)
from .file_utils import (
    build_embedding_df,
    collect_unique_links,
    download_and_extract_links,
)
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
    max_seq_length : int, optional
        Maximum sequence length for text tokenization. If None, will be extracted
        from the tokenizer's configuration. Used for truncation during tokenization.

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
        max_seq_length: int | None = None,
    ) -> None:
        self.text_encoder_name = text_encoder_name
        if text_encoder_name == "one_hot":
            self.text_tok = None  # no HF tokenizer
            self._sentence2id: dict[str, int] = {}
            self._next_id = 1  # 0 is reserved for padding
            # Set max_seq_length for one-hot encoder
            self.max_seq_length = max_seq_length if max_seq_length is not None else 512
        else:
            self.text_tok = AutoTokenizer.from_pretrained(text_encoder_name)
            # Extract max_seq_length from tokenizer config or use provided value
            self._set_max_seq_length(max_seq_length)
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

    def _set_max_seq_length(self, max_seq_length: int | None = None):
        """Extract and set max_seq_length from tokenizer configuration or use provided value."""
        if max_seq_length is not None:
            # Use explicitly provided value
            self.max_seq_length = max_seq_length
            return

        # Try to get from tokenizer config
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.text_encoder_name)
            if hasattr(config, "max_position_embeddings"):
                self.max_seq_length = config.max_position_embeddings
            elif hasattr(config, "max_seq_length"):
                self.max_seq_length = config.max_seq_length
            elif hasattr(config, "n_positions"):
                self.max_seq_length = config.n_positions
            elif hasattr(config, "max_sequence_length"):
                self.max_seq_length = config.max_sequence_length
            else:
                # Fallback to a reasonable default
                self.max_seq_length = 512
        except Exception:
            # Fallback if config loading fails
            self.max_seq_length = 512

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
                tok_out = self.text_tok(
                    texts, padding=padding, return_tensors="pt", truncation=True, max_length=self.max_seq_length
                )  # ** tok_kwargs, removed due to issues with "document" type being passed, as of ST version 5.
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
                tok_out = self.text_tok(
                    text_vals, padding=padding, return_tensors="pt", truncation=True, max_length=self.max_seq_length
                )  # ** tok_kwargs, removed due to issues with "document" type being passed, as of ST version 5.
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
class MMContextEncoder(Module):
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
        ["unregistered", "pca", "hvg", "scvi_fm", "geneformer", "gs", "gs10k"].
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
    joint_adapter_hidden_dim : int | None, optional
        Hidden dimension for the joint adapter that processes sentence embeddings.
        If None or 0, no joint adapter is used. If >0, uses a two-layer MLP with the
        specified hidden dimension. The joint adapter starts frozen and can be
        enabled/unfrozen at a specified epoch during training. Defaults to None.

        Note: Setting hidden_dim=0 would create an identity module (since input_dim
        equals output_dim), which has no effect, so it's treated the same as None.
    max_seq_length : int | None, optional
        Maximum sequence length for tokenization. If None, will be extracted from
        the text encoder's configuration (e.g., max_position_embeddings). If the
        text encoder doesn't have this information, defaults to 512. This property
        is important for SentenceTransformers compatibility.
    text_model_kwargs : dict | None, optional
        Additional keyword arguments to pass to AutoModel.from_pretrained() when
        loading the text encoder. For example, {"attn_implementation": "flash_attention_2"}
        to enable flash attention. Defaults to None (empty dict).
    use_text_adapter : bool, optional
        Whether to create an adapter for the text encoder. If False, the text encoder
        output dimension must match the adapter_output_dim (if specified) or no adapters
        should be used at all. Defaults to True.
    """

    # Configuration attributes for SentenceTransformer Module base class
    config_keys = [
        "text_encoder_name",
        "adapter_hidden_dim",
        "adapter_output_dim",
        "freeze_text_encoder",
        "unfreeze_last_n_layers",
        "registered_data_origin",
        "registered_input_dim",
        "output_token_embeddings",
        "train_lookup",
        "pooling_mode",
        "joint_adapter_hidden_dim",
        "_joint_adapter_was_trained",
        "max_seq_length",
        "text_model_kwargs",
        "use_text_adapter",
        "current_vocab_size",
        "tokenizer_added_tokens",
    ]

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
        joint_adapter_hidden_dim: int | None = None,
        _joint_adapter_was_trained: bool = False,
        max_seq_length: int | None = None,
        text_model_kwargs: dict | None = None,
        use_text_adapter: bool = True,
        current_vocab_size: int | None = None,
        tokenizer_added_tokens: list | str | None = None,
    ) -> None:
        super().__init__()

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
        self.joint_adapter_hidden_dim = joint_adapter_hidden_dim
        self.text_model_kwargs = text_model_kwargs or {}
        self.use_text_adapter = use_text_adapter

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
            self.text_encoder = AutoModel.from_pretrained(text_encoder_name, **self.text_model_kwargs)

        text_hidden_dim = self.text_encoder.config.hidden_size

        # Validate text adapter configuration
        if not use_text_adapter and self._use_adapters:
            # If text adapter is disabled but adapters are requested, check dimension compatibility
            expected_output_dim = adapter_output_dim if adapter_output_dim is not None else text_hidden_dim
            if text_encoder_name != "one_hot":  # Skip validation for one_hot encoder as it's configurable
                # We need to get text_hidden_dim first, so this validation will be done after text encoder initialization
                pass

        # Validate text adapter configuration after getting text_hidden_dim
        if not use_text_adapter and self._use_adapters:
            expected_output_dim = adapter_output_dim if adapter_output_dim is not None else text_hidden_dim
            if text_hidden_dim != expected_output_dim:
                raise ValueError(
                    f"Text adapter is disabled but text encoder output dimension ({text_hidden_dim}) "
                    f"does not match adapter output dimension ({expected_output_dim}). "
                    f"Either set use_text_adapter=True or set adapter_output_dim={text_hidden_dim}."
                )

        # Note: max_seq_length will be set from processor after processor initialization

        # Determine output dimension
        if self._use_adapters:
            self._output_dim = adapter_output_dim if adapter_output_dim is not None else text_hidden_dim
        else:
            self._output_dim = text_hidden_dim

        # Setup text adapter if requested
        if self._use_adapters and use_text_adapter:
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

        # ------------------------------------------------ joint adapter
        # Create joint adapter if requested
        # None or 0 = skip joint adapter altogether (0 would create identity module, so no effect)
        # >0 = use joint adapter with MLP (hidden layer of specified size)
        if joint_adapter_hidden_dim is not None and joint_adapter_hidden_dim > 0:
            self.joint_adapter = AdapterModule(
                input_dim=self._output_dim,
                hidden_dim=joint_adapter_hidden_dim,
                output_dim=self._output_dim,
            )
            # Start with frozen parameters
            for param in self.joint_adapter.parameters():
                param.requires_grad = False
            self._joint_adapter_enabled = False
        else:
            self.joint_adapter = None
            self._joint_adapter_enabled = False

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
                max_seq_length=max_seq_length,
            )

        self.processor = processor
        # Set max_seq_length from processor
        self.max_seq_length = self.processor.max_seq_length
        self._manage_text_encoder_freezing()

        # Track if joint adapter was trained (used for proper loading)
        self._joint_adapter_was_trained = _joint_adapter_was_trained

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
                # For ModernBERT models
                if hasattr(self.text_encoder, "layers"):
                    layers = self.text_encoder.layers[-self.unfreeze_last_n_layers :]
                    logger.info(f"Unfreezing last {self.unfreeze_last_n_layers} layers of ModernBERT model")
                # For BERT-like models
                elif hasattr(self.text_encoder, "encoder") and hasattr(self.text_encoder.encoder, "layer"):
                    layers = self.text_encoder.encoder.layer[-self.unfreeze_last_n_layers :]
                    logger.info(f"Unfreezing last {self.unfreeze_last_n_layers} layers of BERT-like model")
                # For RoBERTa-like models
                elif hasattr(self.text_encoder, "roberta") and hasattr(self.text_encoder.roberta, "encoder"):
                    layers = self.text_encoder.roberta.encoder.layer[-self.unfreeze_last_n_layers :]
                    logger.info(f"Unfreezing last {self.unfreeze_last_n_layers} layers of RoBERTa-like model")
                else:
                    logger.warning(
                        f"Unsupported architecture for {self.text_encoder_name}. Cannot unfreeze last {self.unfreeze_last_n_layers} layers."
                    )
                    return

                for layer in layers:
                    for param in layer.parameters():
                        param.requires_grad = True

                logger.info(
                    f"Successfully unfroze {len(layers)} layers with {sum(p.numel() for layer in layers for p in layer.parameters() if p.requires_grad)} trainable parameters"
                )

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
            # token_type_slice = (
            #    features["token_type_ids"][text_mask]  # use the mask
            #     if "token_type_ids" in features and features["token_type_ids"] is not None
            #     else None  # otherwise skip arg
            # )
            txt_out = self.text_encoder(
                input_ids=features["input_ids"][text_mask],
                attention_mask=features["attention_mask"][text_mask],
                # token_type_ids=token_type_slice,
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
        # Apply joint adapter if enabled
        # ------------------------------------------------------------------ #
        if self.joint_adapter is not None and self._joint_adapter_enabled:
            sentence_embeddings = self.joint_adapter(sentence_embeddings)

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
        # Convert text_model_kwargs to JSON-serializable format
        serializable_kwargs = {}
        for key, value in self.text_model_kwargs.items():
            # Check if this is a torch dtype by checking the type string
            if str(type(value)).startswith("<class 'torch.") and "dtype" in str(type(value)):
                # Handle torch dtypes by converting to string
                serializable_kwargs[key] = str(value)
            else:
                serializable_kwargs[key] = value

        # Check if tokenizer was resized and store vocab size info
        current_vocab_size = None
        tokenizer_added_tokens = None
        if hasattr(self, "processor") and hasattr(self.processor, "text_tok") and self.processor.text_tok is not None:
            current_vocab_size = len(self.processor.text_tok)
            # Get added tokens if any
            try:
                added_tokens = self.processor.text_tok.get_added_vocab()
                if added_tokens:
                    tokenizer_added_tokens = list(added_tokens.keys())
            except Exception:
                # Fallback: just store the vocab size difference
                try:
                    from transformers import AutoConfig

                    config = AutoConfig.from_pretrained(self.text_encoder_name)
                    original_vocab_size = config.vocab_size
                    if current_vocab_size > original_vocab_size:
                        tokenizer_added_tokens = f"<{current_vocab_size - original_vocab_size}_added_tokens>"
                except Exception:
                    pass

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
            "joint_adapter_hidden_dim": self.joint_adapter_hidden_dim,
            "_joint_adapter_was_trained": self._joint_adapter_was_trained,
            "max_seq_length": self.max_seq_length,
            "text_model_kwargs": serializable_kwargs,
            "use_text_adapter": self.use_text_adapter,
            "current_vocab_size": current_vocab_size,
            "tokenizer_added_tokens": tokenizer_added_tokens,
        }

    def save(self, output_path: str, safe_serialization: bool = True, **kwargs) -> None:
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

        # Save config using our custom serialization method
        config_path = os.path.join(output_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(self._get_config_dict(), f, indent=4)

        # Get state dict and filter omics embeddings
        state = self.state_dict()
        if self._has_omics:
            # Remove omics embeddings but keep adapter weights
            state = {k: v for k, v in state.items() if not k.startswith("omics_encoder.embeddings")}

        # Check if tokenizer has been resized (has additional tokens beyond original vocab)
        current_vocab_size = None
        original_vocab_size = None
        tokenizer_was_resized = False

        if hasattr(self, "processor") and hasattr(self.processor, "text_tok") and self.processor.text_tok is not None:
            current_vocab_size = len(self.processor.text_tok)
            # Get original vocab size from the pretrained model config
            try:
                from transformers import AutoConfig

                config = AutoConfig.from_pretrained(self.text_encoder_name)
                original_vocab_size = config.vocab_size
                tokenizer_was_resized = current_vocab_size > original_vocab_size

                if tokenizer_was_resized:
                    logger.info(f"Detected resized tokenizer: {original_vocab_size} -> {current_vocab_size} tokens")
            except Exception as e:
                logger.warning(f"Could not determine original vocab size: {e}")

        # Create a temporary model with the filtered state dict for saving
        temp_model = MMContextEncoder(
            text_encoder_name=self.text_encoder_name,
            adapter_hidden_dim=self.adapter_hidden_dim,
            adapter_output_dim=self.adapter_output_dim,
            registered_data_origin=self._registered_data_origin,
            registered_input_dim=self._registered_input_dim,
            output_token_embeddings=self.output_token_embeddings,
            train_lookup=self.train_lookup,
            pooling_mode=self.pooling_mode,
            joint_adapter_hidden_dim=self.joint_adapter_hidden_dim,
            max_seq_length=self.max_seq_length,
            text_model_kwargs=self.text_model_kwargs,
            use_text_adapter=self.use_text_adapter,
        )

        # If tokenizer was resized, we need to resize the temp model's embeddings too
        if tokenizer_was_resized and current_vocab_size is not None:
            logger.info(f"Resizing temporary model embeddings to match current vocab size: {current_vocab_size}")
            temp_model.text_encoder.resize_token_embeddings(current_vocab_size)

            # Also update the processor's tokenizer to match
            if (
                hasattr(temp_model, "processor")
                and hasattr(temp_model.processor, "text_tok")
                and temp_model.processor.text_tok is not None
            ):
                # Copy the tokenizer state from the original model
                temp_model.processor.text_tok = self.processor.text_tok

        # Load the filtered state dict into the temporary model
        temp_model.load_state_dict(state, strict=False)

        # Save torch weights using base class method
        temp_model.save_torch_weights(output_path, safe_serialization=safe_serialization)

    @classmethod
    def _deserialize_text_model_kwargs(cls, kwargs_dict: dict) -> dict:
        """
        Deserialize text_model_kwargs from JSON-serializable format.

        Parameters
        ----------
        kwargs_dict : dict
            Dictionary with potentially serialized values

        Returns
        -------
        dict
            Dictionary with deserialized values
        """
        import torch

        deserialized = {}
        for key, value in kwargs_dict.items():
            if isinstance(value, str):
                # Try to convert string representations back to torch dtypes
                if value == "torch.float32":
                    deserialized[key] = torch.float32
                elif value == "torch.float16":
                    deserialized[key] = torch.float16
                elif value == "torch.bfloat16":
                    deserialized[key] = torch.bfloat16
                elif value == "torch.int8":
                    deserialized[key] = torch.int8
                elif value == "torch.int16":
                    deserialized[key] = torch.int16
                elif value == "torch.int32":
                    deserialized[key] = torch.int32
                elif value == "torch.int64":
                    deserialized[key] = torch.int64
                elif value == "auto":
                    # Special case for "auto" which is commonly used
                    deserialized[key] = "auto"
                else:
                    # Keep as string if not a recognized torch dtype
                    deserialized[key] = value
            else:
                deserialized[key] = value

        return deserialized

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token=None,
        cache_folder=None,
        revision=None,
        local_files_only=False,
        safe_serialization: bool = True,
        **kwargs,
    ):
        """
        Loads the model from disk.

        Parameters
        ----------
        model_name_or_path : str
            Path to the model directory or the name of the model on Hugging Face.
        subfolder : str, optional
            The subfolder within the model directory to load from. Defaults to ''.
        token : bool | str | None, optional
            Token for authentication. Defaults to None.
        cache_folder : str | None, optional
            Cache folder for the model files. Defaults to None.
        revision : str | None, optional
            Revision of the model to load. Defaults to None.
        local_files_only : bool, optional
            Whether to only load local files. Defaults to False.
        safe_serialization : bool, optional
            If True, expects safetensors format; else a PyTorch bin.

        Returns
        -------
        MMContextEncoder
            The loaded model instance, ready for data registration.
        """
        # Load configuration
        cfg = cls.load_config(
            model_name_or_path,
            subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

        # Deserialize text_model_kwargs if present
        if "text_model_kwargs" in cfg:
            cfg["text_model_kwargs"] = cls._deserialize_text_model_kwargs(cfg["text_model_kwargs"])

        # Extract tokenizer information from config
        current_vocab_size = cfg.pop("current_vocab_size", None)
        tokenizer_added_tokens = cfg.pop("tokenizer_added_tokens", None)

        # Create model from config without omics embedding
        model = cls(**cfg)

        # Handle tokenizer resizing if needed
        if (
            current_vocab_size is not None
            and hasattr(model, "processor")
            and hasattr(model.processor, "text_tok")
            and model.processor.text_tok is not None
        ):
            original_vocab_size = len(model.processor.text_tok)
            if current_vocab_size > original_vocab_size:
                logger.info(f"Restoring tokenizer size from {original_vocab_size} to {current_vocab_size}")

                # Add tokens if we have the token list
                if tokenizer_added_tokens and isinstance(tokenizer_added_tokens, list):
                    logger.info(f"Adding saved tokens: {tokenizer_added_tokens}")
                    num_added = model.processor.text_tok.add_tokens(tokenizer_added_tokens)
                    logger.info(f"Added {num_added} tokens to tokenizer")
                else:
                    # Fallback: add placeholder tokens to match the size
                    tokens_to_add = current_vocab_size - original_vocab_size
                    placeholder_tokens = [f"<PLACEHOLDER_{i}>" for i in range(tokens_to_add)]
                    logger.warning(f"Adding {tokens_to_add} placeholder tokens to match saved vocab size")
                    model.processor.text_tok.add_tokens(placeholder_tokens)

                # Resize model embeddings to match
                model.text_encoder.resize_token_embeddings(len(model.processor.text_tok))
                logger.info(f"Resized model embeddings to {len(model.processor.text_tok)} tokens")

        # Load torch weights
        cls.load_torch_weights(
            model_name_or_path,
            subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
            model=model,
        )

        if model._registered_data_origin != "unregistered":
            logger.warning(
                f"Loaded encoder was registered for '{model._registered_data_origin}' data. "
                "Call register_initial_embeddings() with compatible data before using it."
            )

        # Enable joint adapter if it was trained in the saved model
        if hasattr(model, "_joint_adapter_was_trained") and model._joint_adapter_was_trained:
            if model.joint_adapter is not None:
                model._joint_adapter_enabled = True
                for param in model.joint_adapter.parameters():
                    param.requires_grad = True
                logger.info("Joint adapter enabled for loaded model (was previously trained)")

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
    def get_initial_embeddings_from_adata_link(
        hf_dataset: DatasetDict | HFDataset,
        *,
        layer_key: str | None = None,
        axis: Literal["obs", "var"] = "obs",
        download_dir: str | Path = "../../data/downloaded_chunks",
        extract_zip: bool = True,
        overwrite: bool = False,
        link_column: str = "adata_link",
        zenodo_token: str | None = None,
        force_drafts: bool = False,
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
        zenodo_token : str | None, optional
            Zenodo access token for authenticating draft record downloads.
            Required for draft records, optional for published records.
        force_drafts : bool, default False
            If False, automatically converts Zenodo draft links to published links.
            This is a quick fix for datasets created with draft links that were
            published remotely afterwards. Set to True to use actual draft links.

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
            zenodo_token=zenodo_token,
            force_drafts=force_drafts,
        )
        # if the layer key is none, this means that only the download step was needed. The model will be used as text only without initial embeddings.
        if layer_key is None:
            logger.info(
                "No layer key provided, get_initial_embeddings_from_adata_link() is returning empty DataFrame and path map."
            )
            return None, path_map

        # --------------------------------------------------------------
        # 2) build per-split DataFrames, then concat
        # --------------------------------------------------------------
        if isinstance(hf_dataset, HFDataset):
            hf_dataset = DatasetDict({hf_dataset.split: hf_dataset})
        split_frames: list[pd.DataFrame] = []
        for split_name, ds in hf_dataset.items():
            # translate split-specific links → local paths
            local_map = {lk: path_map[lk] for lk in ds[link_column]}
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

    @staticmethod
    def create_token_dataframe_from_obsm(
        adata: ad.AnnData,
        *,
        obsm_key: str,
        token_col: str = "token",
        embedding_col: str = "embedding",
    ) -> pd.DataFrame:
        """
        Create a token DataFrame from embeddings stored in adata.obsm.

        This method extracts embeddings from the specified obsm_key and returns a DataFrame
        mapping cell IDs to their embeddings, formatted for use with the
        register_initial_embeddings() method.

        Parameters
        ----------
        adata : anndata.AnnData
            Input AnnData object with embeddings stored in adata.obsm.
        obsm_key : str
            Key in adata.obsm containing the embeddings to extract.
        token_col : str, optional
            Name of the column containing cell/sample IDs. Defaults to "token".
        embedding_col : str, optional
            Name of the column containing embedding vectors. Defaults to "embedding".

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - token_col: Cell/sample IDs from adata.obs.index
            - embedding_col: Embedding vectors from adata.obsm[obsm_key]

        Raises
        ------
        KeyError
            If the specified obsm_key is not found in adata.obsm.

        Examples
        --------
        >>> import anndata as ad
        >>> from mmcontext import MMContextEncoder
        >>> # Load your AnnData object with embeddings
        >>> adata = ad.read_h5ad("your_data.h5ad")
        >>> # Create token DataFrame from any obsm key
        >>> token_df = MMContextEncoder.create_token_dataframe_from_obsm(adata, obsm_key="X_pca")
        >>> # Register with your model
        >>> model = MMContextEncoder("bert-base-uncased")
        >>> model.register_initial_embeddings(token_df, data_origin="pca")

        >>> # Or use with gs10k embeddings
        >>> MMContextEncoder.add_gs10k_embeddings_to_adata(adata)
        >>> token_df = MMContextEncoder.create_token_dataframe_from_obsm(adata, obsm_key="gs10k")
        >>> model.register_initial_embeddings(token_df, data_origin="gs10k")
        """
        import anndata as ad

        if obsm_key not in adata.obsm:
            raise KeyError(f"Key '{obsm_key}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}")

        # Extract embeddings from obsm
        embedding_matrix = adata.obsm[obsm_key]

        # Create DataFrame mapping cell IDs to their embeddings
        token_df = pd.DataFrame(
            {
                token_col: adata.obs.index.tolist(),
                embedding_col: [embedding_matrix[i] for i in range(embedding_matrix.shape[0])],
            }
        )

        logger.info(f"Created token DataFrame from adata.obsm['{obsm_key}'] with shape: {token_df.shape}")
        logger.info(f"Each embedding has dimension: {embedding_matrix.shape[1]}")
        logger.info("Use the returned DataFrame to register the embeddings with `register_initial_embeddings()`.")

        return token_df

    def prefix_ds(
        self,
        ds: HFDataset | DatasetDict,
        columns_to_prefix: list[str] | str,
    ) -> HFDataset | DatasetDict:
        """Return a copy ready for SentenceTransformerTrainer with prefixes applied and columns renamed.

        This simplified version only handles:
        1. Adding prefixes to specified columns
        2. Renaming columns based on dataset type (for sentence transformers compatibility)

        Parameters
        ----------
        ds : HFDataset | DatasetDict
            Input dataset to prepare (should have columns already selected and indices resolved)
        columns_to_prefix : list[str]
            List of column names to add the processor's prefix to

        Returns
        -------
        HFDataset | DatasetDict
            Processed dataset with prefixes and renamed columns
        """
        if isinstance(columns_to_prefix, str):
            columns_to_prefix = [columns_to_prefix]

        def _add_prefix(tok: str) -> str:
            p = self.processor.prefix
            return tok if tok.startswith(p) else f"{p}{tok}"

        def _apply_prefixes(batch):
            for col in columns_to_prefix:
                if col in batch:
                    batch[col] = [_add_prefix(t.strip()) for t in batch[col]]
            return batch

        def _process_split(split: HFDataset) -> HFDataset:
            # Apply prefixes to specified columns
            if columns_to_prefix:
                proc = split.map(
                    _apply_prefixes,
                    batched=True,
                    desc=f"Prefixing columns: {columns_to_prefix}",
                )
            else:
                proc = split

            # Note: Column renaming (e.g., primary column to "anchor") is now handled
            # in the resolve_negative_indices_and_rename function for multiplets.
            # This method only handles prefixing.

            return proc

        if isinstance(ds, HFDataset):
            return _process_split(ds)
        elif isinstance(ds, DatasetDict):
            return DatasetDict({name: _process_split(s) for name, s in ds.items()})
        else:
            raise TypeError("prefix_ds expects a datasets.Dataset or DatasetDict")

    @property
    def registered_data_origin(self) -> str:
        """Get the type of omics data representation registered with this model."""
        return self._registered_data_origin

    @property
    def registered_input_dim(self) -> int | None:
        """Get the input dimension of registered data."""
        return self._registered_input_dim
