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
    * adapters    … feed-forward projection heads

"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model
from transformers import AutoModel, AutoTokenizer

from .Adapters import AdapterModule
from .MiniOmicsEncoder import MiniOmicsModel

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
    dict_keys(['omics_ids', 'input_ids', 'attention_mask', 'omics_text_info'])
    """

    def __init__(
        self,
        text_encoder_name: str,
        omics_lookup: dict[str, int] = None,
        *,
        prefix: str = _PREFIX,
    ) -> None:
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
        omics_ids              (B, max_omics_len),  # ⟵ PAD-filled
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
        # ----------------------------------------------------------------- #
        # TEXT-ONLY MODE
        # ----------------------------------------------------------------- #
        if self._text_only:
            for i, item in enumerate(texts):
                if not isinstance(item, str):
                    raise TypeError(f"In text-only mode all inputs must be strings (got {type(item)} at position {i}).")

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
            tok_out = self.text_tok(text_vals, padding=padding, return_tensors="pt", **tok_kwargs)
            # tok["attention_mask"] = tok["attention_mask"].bool()
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

            batch["omics_ids"] = ids  # (B, max_len)
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

        # ------------------------------------------------ text tower
        if isinstance(text_encoder_name, nn.Module):
            self.text_encoder = text_encoder_name
        else:
            self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        text_hidden_dim = self.text_encoder.config.hidden_size

        self._output_dim = adapter_output_dim if adapter_output_dim is not None else text_hidden_dim
        # Setup adapters if requested
        self.text_adapter = AdapterModule(
            input_dim=text_hidden_dim,
            hidden_dim=adapter_hidden_dim,
            output_dim=adapter_output_dim,
        )
        # ------------------------------------------------------------------
        # token-level projection layers (OPTIONAL)
        # ------------------------------------------------------------------
        self._proj_tokens = self.output_token_embeddings  # short flag

        # ---- text branch -------------------------------------------------
        if self._proj_tokens and self._output_dim != text_hidden_dim:
            # only needed when token dim would mismatch the requested output dim
            self.text_token_adapter = nn.Linear(text_hidden_dim, self._output_dim)
        else:
            # either not requested or dims already match ⇒ no extra layer
            self.text_token_adapter = None  # handled safely in forward

        # ---- omics branch ------------------------------------------------
        # (created later if/when an omics matrix is registered)
        self.omics_token_adapter = None

        # ------------------------------------------------ omics tower
        if omics_embedding is not None:
            self._init_omics(omics_embedding)
        elif registered_data_origin != "unregistered" and registered_input_dim:
            # Initialize adapter for registered data without embedding matrix
            self.omics_encoder = None
            self.omics_adapter = AdapterModule(
                input_dim=registered_input_dim,
                hidden_dim=adapter_hidden_dim,
                output_dim=self._output_dim,
            )
            self._registered_input_dim = registered_input_dim
            # token-projection layers only if we might need them
            if self.output_token_embeddings:
                self.omics_token_adapter = nn.Linear(registered_input_dim, adapter_output_dim)
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
        has_adapter = False
        has_token_adapter = False
        if hasattr(self, "omics_adapter") and self.omics_adapter is not None:
            has_adapter = True
        if hasattr(self, "omics_token_adapter") and self.omics_token_adapter is not None:
            has_token_adapter = True
        if not has_adapter:
            # Otherwise, create a new adapter
            # if input_dim != self._output_dim:
            self.omics_adapter = AdapterModule(
                input_dim=input_dim,
                hidden_dim=self.adapter_hidden_dim,
                output_dim=self._output_dim,
            )
            # else:
            #    raise ValueError(
            #        f"Without adapters, omics dimension ({input_dim}) must match "
            #        f"text encoder dimension ({self._output_dim})"
            #    )
        if not has_token_adapter:
            # Create token projection layers if needed
            if self._proj_tokens and self._output_dim != input_dim:
                self.omics_token_adapter = nn.Linear(input_dim, self._output_dim)

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

    def _safe_project(self, last_hidden, adapter, want_proj: bool):
        """
        Return token representations with an *optional* projection layer.

        • If *want_proj* is False OR *adapter* is None  → just return `last_hidden`
        • else                                         → run the adapter first
        """
        if not want_proj or adapter is None:
            return last_hidden
        return adapter(last_hidden)

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

        Parameters
        ----------
        features : dict
            Output of tokenize method, containing text features and optionally
            omics features.

        Returns
        -------
        torch.Tensor or dict
            If return_tensor=True in features, returns a returns tensor of sentence embeddings directly.
            key. Otherwise, dict with 'sentence_embedding.

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

        # ---- set up sentence-level outputs (always) -------------------------
        batch_size = features["omics_text_info"].size(0)
        sent_out = torch.zeros(batch_size, self._output_dim, device=dev, dtype=dtype)
        # ---- prepare token-level placeholders only if we need them ---------
        if self.output_token_embeddings:
            txt_len = 1
            om_len = 1
            if "input_ids" in features:
                txt_len = features["input_ids"].size(1)
            if "omics_ids" in features:
                om_len = features["omics_ids"].size(1)
            max_len = max(txt_len, om_len)

            tok_out = torch.zeros(batch_size, max_len, self._output_dim, device=dev, dtype=dtype)
            attn_out = torch.zeros(batch_size, max_len, device=dev, dtype=torch.long)
            mod_out = torch.full((batch_size, max_len), 2, device=dev, dtype=torch.long)  # 2 = PAD
            # 0 = text, 1 = omics, 2 = pad

        text_mask = features["omics_text_info"] == 1
        omics_mask = ~text_mask

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

            # pooled sentence vector
            sent_out[text_mask] = self.text_adapter(txt_out.pooler_output.to(dtype))

            if self.output_token_embeddings:
                txt_tokens = self._safe_project(
                    txt_out.last_hidden_state.to(dtype),
                    getattr(self, "text_token_adapter", None),
                    self._proj_tokens,
                )
                L = txt_tokens.size(1)
                tok_out[text_mask, :L] = txt_tokens
                attn_out[text_mask, :L] = features["attention_mask"][text_mask]
                mod_out[text_mask, :L] = 0  # text-modality id

        # ------------------------------------------------------------------ #
        # omics branch
        # ------------------------------------------------------------------ #
        omics_mask = ~text_mask
        if omics_mask.any():
            if not self._has_omics:
                raise RuntimeError("Call `register_initial_embeddings()` first.")

            om_out = self.omics_encoder(  # (n, O, H)
                input_ids=features["omics_ids"][omics_mask]
            )

            sent_out[omics_mask] = self.omics_adapter(om_out.pooler_output.to(dtype))

            if self.output_token_embeddings:
                om_tokens = self._safe_project(
                    om_out.last_hidden_state.to(dtype),
                    getattr(self, "omics_token_adapter", None),
                    self._proj_tokens,
                )
                L = om_tokens.size(1)
                tok_out[omics_mask, :L] = om_tokens
                attn_out[omics_mask, :L] = 1
                mod_out[omics_mask, :L] = 1  # omics-modality id

        # ------------------------------------------------------------------ #
        # pack up & return
        # ------------------------------------------------------------------ #
        if return_tensor:
            return sent_out
        features.update(
            sentence_embedding=sent_out,
        )
        if self.output_token_embeddings:
            features.update(
                token_embeddings=tok_out,
                attention_mask=attn_out,
                modality_ids=mod_out,  # optional metadata
            )
        return features

    def _get_sentence_embedding_dimension(self) -> int:
        """
        Returns the dimension of the final sentence embedding.

        Returns
        -------
        int
            The dimension of the final sentence embedding.
        """
        return self._output_dim

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

    def prefix_ds(
        self,
        ds: HFDataset | DatasetDict,
        cols_to_prefix: str | list[str],
        *,  # keyword-only below
        positive_col: str = "positive",
        label_col: str = "label",
        negative_prefix: str = "negative",
    ) -> HFDataset | DatasetDict:
        """Return a copy ready for SentenceTransformerTrainer.

        * Adds ``self.processor.prefix`` once to every token string in *cols_to_prefix*.
        * Detects *single*, *pairs*, *multiplets* by the presence of ``label`` /
        ``positive`` columns and **removes all unused columns** accordingly.
        """
        if isinstance(cols_to_prefix, str):
            cols_to_prefix = [cols_to_prefix]

        def _add_prefix(tok: str) -> str:
            p = self.processor.prefix
            return tok if tok.startswith(p) else f"{p}{tok}"

        def _pref(batch, pref_cols):
            for col in pref_cols:
                batch[col] = [_add_prefix(t.strip()) for t in batch[col]]
            return batch

        def _process_split(split: HFDataset) -> HFDataset:
            is_pairs = label_col in split.column_names
            is_multiplets = positive_col in split.column_names

            pref_cols = list(cols_to_prefix)

            if is_multiplets:
                pref_cols.append(positive_col)
                pref_cols += [c for c in split.column_names if c.startswith(negative_prefix)]
                keep_cols = pref_cols  # anchor + positive + negatives
            elif is_pairs:
                keep_cols = pref_cols + ["captions", label_col]
            else:  # single – keep everything
                keep_cols = list(split.column_names)

            # sanity checks --------------------------------------------------
            missing = [c for c in pref_cols if c not in split.column_names]
            if missing:
                raise TypeError(f"Columns {missing} missing from dataset split.")

            for col in pref_cols:
                if split.features[col].dtype != "string":
                    raise TypeError(f"Column '{col}' must contain strings.")

            # prefix + drop unused columns ----------------------------------
            proc = split.map(
                lambda b: _pref(b, pref_cols),
                batched=True,
                desc=f"Prefixing {', '.join(pref_cols)}",
            )
            drop_cols = [c for c in proc.column_names if c not in keep_cols]
            if drop_cols:
                proc = proc.remove_columns(drop_cols)
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
