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
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
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
    omics processing capabilities later via register_numeric_ds.

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

    def tokenize(
        self,
        texts: Sequence[str | int | np.ndarray | dict],
        *,
        padding: str | bool = True,
        **tok_kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize a batch of texts and/or omics identifiers.

        In text-only mode, all inputs must be strings and will be processed as text.
        In bimodal mode, inputs starting with the prefix are treated as omics IDs.

        Parameters
        ----------
        texts : sequence of str
            Mixture of captions and (if omics lookup is available) ``"sample_idx:<int>"`` strings.
        padding : str or bool, optional
            Forwarded to the underlying HF tokenizer (default: ``True``).
        **tok_kwargs
            Additional keyword args forwarded to the tokenizer.

        Returns
        -------
        dict
            Contains tokenized features and modality information.

        Raises
        ------
        TypeError
            If an input is not a string or doesn't match the expected format.
        """
        if self._text_only:
            # Text-only mode: process all inputs as text
            for i, item in enumerate(texts):
                if not isinstance(item, str):
                    raise TypeError(f"In text-only mode, all inputs must be strings. Got {type(item)} at position {i}")

            # Tokenize all texts at once
            tok_out = self.text_tok(
                texts,
                padding=padding,
                return_tensors="pt",
                **tok_kwargs,
            )

            # Add modality indicator (all are text)
            tok_out["omics_text_info"] = torch.ones(len(texts), dtype=torch.int8)
            return tok_out
        else:
            # Bimodal mode: separate omics IDs from text
            omics_pos, text_pos = [], []
            omics_rows, text_vals = [], []

            for i, item in enumerate(texts):
                if isinstance(item, str) and item.startswith(self.prefix):
                    omics_pos.append(i)
                    try:
                        omics_rows.append(self.lookup[item])
                    except KeyError as err:
                        raise KeyError(f"Omics ID '{item}' not found in lookup table") from err
                elif isinstance(item, str):
                    text_pos.append(i)
                    text_vals.append(item)
                else:
                    raise TypeError(f"Unsupported element type {type(item)} at position {i}. Expected strings.")

            batch: dict[str, torch.Tensor] = {}

            # ---- text branch --------------------------------------------------
            if text_vals:
                tok_out = self.text_tok(
                    text_vals,
                    padding=padding,
                    return_tensors="pt",
                    **tok_kwargs,
                )
                for k, v in tok_out.items():
                    full = [torch.zeros_like(v[0])] * len(texts)
                    for src, dst in enumerate(text_pos):
                        full[dst] = v[src]
                    batch[k] = torch.stack(full)

            # ---- omics branch -------------------------------------------------
            if omics_rows:
                full = torch.zeros(len(texts), dtype=torch.long)
                full[omics_pos] = torch.tensor(omics_rows, dtype=torch.long)
                batch["omics_ids"] = full

            # ---- modality indicator ------------------------------------------
            indi = torch.tensor([0] * len(texts), dtype=torch.int8)
            indi[text_pos] = 1
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
    3. Bimodal mode that will be completed later via register_numeric_ds

    The adapter layers can be optionally included for dimensionality reduction.

    Parameters
    ----------
    text_encoder_name : str | nn.Module
        Name of HuggingFace model or a pre-initialized model
    omics_embedding : Optional[np.ndarray], optional
        Precomputed embeddings for omics data. If None, the model starts as
        text-only and can be extended via register_numeric_ds.
    adapter_hidden_dim : int or None, optional
        Hidden dimension of the adapter layers. If None, no adapter is used.
    adapter_output_dim : int or None, optional
        Output dimension of the adapter layers. If None, no adapter is used.
        If adapter_hidden_dim is provided but this is None, output_dim will
        match the text encoder's hidden dimension.
    freeze_text_encoder : bool, optional
        Whether to freeze the text encoder weights. Defaults to True.
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
    """

    VALID_DATA_ORIGINS = ["unregistered", "pca", "hvg", "scvi_fm", "geneformer"]

    def __init__(
        self,
        text_encoder_name: str | nn.Module,
        omics_embedding: np.ndarray | None = None,
        *,
        adapter_hidden_dim: int | None = 64,
        adapter_output_dim: int | None = 128,
        freeze_text_encoder: bool = True,
        unfreeze_last_n_layers: int = 0,
        processor: MMContextProcessor | None = None,
        registered_data_origin: str = "unregistered",
        registered_input_dim: int | None = None,
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

        # ------------------------------------------------ text tower
        if isinstance(text_encoder_name, nn.Module):
            self.text_encoder = text_encoder_name
        else:
            self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        text_hidden = self.text_encoder.config.hidden_size

        # Setup adapters if requested
        if adapter_hidden_dim is not None:
            output_dim = adapter_output_dim if adapter_output_dim is not None else text_hidden

            self.text_adapter = AdapterModule(
                input_dim=text_hidden,
                hidden_dim=adapter_hidden_dim,
                output_dim=output_dim,
            )
            self._use_adapters = True
            self._output_dim = output_dim
        else:
            self.text_adapter = nn.Identity()
            self._use_adapters = False
            self._output_dim = text_hidden

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
        else:
            self.omics_encoder = None
            self.omics_adapter = None

        # ------------------------------------------------ processor
        if processor is None:
            omics_lookup = None
            if omics_embedding is not None:
                omics_lookup = {f"{_PREFIX}{i}": i for i in range(omics_embedding.shape[0])}

            processor = MMContextProcessor(
                text_encoder_name=text_encoder_name if isinstance(text_encoder_name, str) else "prajjwal1/bert-tiny",
                omics_lookup=omics_lookup,
            )

        self.processor = processor
        self._manage_text_encoder_freezing()

    def _init_omics(self, matrix: np.ndarray):
        """Initialize the omics tower given an embedding matrix."""
        self.omics_encoder = MiniOmicsModel.from_numpy(matrix)

        # Initialize or preserve adapter layers
        self._init_or_preserve_adapters(matrix.shape[1])

        self.omics_input_dim = matrix.shape[1]
        self._registered_input_dim = matrix.shape[1]
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
        if hasattr(self, "omics_adapter") and self.omics_adapter is not None:
            return

        # Otherwise, create a new adapter
        if self._use_adapters:
            self.omics_adapter = AdapterModule(
                input_dim=input_dim,
                hidden_dim=self.adapter_hidden_dim,
                output_dim=self._output_dim,
            )
        else:
            if input_dim != self._output_dim:
                raise ValueError(
                    f"Without adapters, omics dimension ({input_dim}) must match "
                    f"text encoder dimension ({self._output_dim})"
                )
            self.omics_adapter = nn.Identity()

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
        return_tensor = features.pop("return_tensor", False)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype  # Get model's dtype

        for k, v in features.items():
            features[k] = v.to(device)
        batch_size = features["omics_text_info"].size(0)
        out = torch.zeros(batch_size, self._output_dim, device=device, dtype=dtype)

        # ---- text branch --------------------------------------------------
        text_mask = features["omics_text_info"] == 1
        if text_mask.any():
            txt_out = self.text_encoder(
                input_ids=features["input_ids"][text_mask],
                attention_mask=features["attention_mask"][text_mask],
                token_type_ids=features.get("token_type_ids", None),
            )
            # Ensure consistent dtype for text encoder outputs
            txt_embed = self.text_adapter(txt_out.pooler_output.to(dtype))
            out[text_mask] = txt_embed

        # ---- omics branch -------------------------------------------------
        omics_mask = ~text_mask
        if omics_mask.any():
            if not self._has_omics:
                raise RuntimeError("Omics resources are not initialized. Call `register_numeric_ds()` first.")
            omics_out = self.omics_encoder(input_ids=features["omics_ids"][omics_mask].unsqueeze(1))
            # Ensure consistent dtype for omics encoder outputs
            omics_embed = self.omics_adapter(omics_out.pooler_output.to(dtype))
            out[omics_mask] = omics_embed

        if return_tensor:
            return out
        features["sentence_embedding"] = out
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
                "Call register_numeric_ds() with compatible data before using it."
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

    def register_numeric_ds(
        self,
        ds,
        data_origin,
        *,
        id_col="sample_idx",
        rep_col="data_representation",
        prefix: str = _PREFIX,
    ):
        """
        Add omics resources from a HuggingFace dataset. Returns a copy of the dataset with prefixed IDs.

        If the model doesn't have omics capabilities yet, it will be initialized.
        If it already has omics capabilities, new data must match the existing dimension
        and data type.

        Parameters
        ----------
        ds : datasets.Dataset or datasets.DatasetDict
            HuggingFace dataset (single split or dict of splits).
        data_origin : str
            Origin of the data (e.g., "pca", "hvg", "scvi_fm", "geneformer").
        id_col : str, optional
            Name of the column containing IDs (default: "sample_idx").
        rep_col : str, optional
            Name of the column containing the embedding vectors (default: "data_representation").
        prefix : str, optional
            Prefix for omics sample IDs (default: _PREFIX).

        Returns
        -------
        datasets.Dataset or datasets.DatasetDict
            A copy of the input dataset with a new column 'prefixed_id' containing
            the prefixed sample IDs ready for tokenization.

        Raises
        ------
        ValueError
            If dimensions don't match or data types are incompatible.
        ImportError
            If datasets package is not available.

        Notes
        -----
        To change model precision after registration, use PyTorch's native dtype conversion:
        `model.to(torch.float16)` or `model.half()` for half precision.
        """
        if data_origin not in self.VALID_DATA_ORIGINS:
            raise ValueError(f"registered_data_origin must be one of {self.VALID_DATA_ORIGINS}")
        if self._registered_data_origin != "unregistered" and data_origin != self._registered_data_origin:
            raise ValueError(
                f"Cannot register data of type '{data_origin}' with model registered for '{self._registered_data_origin}'. "
                "Models can only be used with one type of data representation."
            )
        else:
            self._registered_data_origin = data_origin

        # 2) Import check
        try:
            from datasets import Dataset, DatasetDict
        except ImportError as err:
            raise ImportError("register_numeric_ds requires the 'datasets' package from HuggingFace.") from err

        # 3) Process data based on type (Dataset or DatasetDict)
        if hasattr(ds, "values") and callable(ds.values):
            # Handle DatasetDict by recursively processing each split
            result_ds = ds.map(lambda x: {**x, "prefixed_id": f"{prefix}{x[id_col]}"}, desc="Adding prefixed IDs")

            # Flatten splits for processing
            rows = []
            for split in ds.values():
                rows.extend(split)
        else:
            # Handle Dataset directly
            result_ds = ds.map(lambda x: {**x, "prefixed_id": f"{prefix}{x[id_col]}"}, desc="Adding prefixed IDs")
            rows = list(ds)

        if not rows:
            logger.warning("Dataset is empty, no samples to register.")
            return result_ds

        # 4) Check dimension consistency with first vector
        first_row = rows[0]
        vec = first_row[rep_col]

        # Handle different input types
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().numpy()
        elif isinstance(vec, list):
            vec = np.array(vec, dtype=np.float32)
        elif not isinstance(vec, np.ndarray):
            try:
                vec = np.array(vec, dtype=np.float32)
            except Exception as err:
                raise TypeError(
                    f"Cannot convert data of type {type(vec)} to numpy array. "
                    "Supported types: numpy.ndarray, torch.Tensor, list of numbers."
                ) from err

        vec_dim = vec.shape[0]

        # If model was registered but has no embedding matrix yet
        # Ensure adapter is created with correct dimensions if needed
        if not self._has_omics and self._registered_input_dim is not None:
            if vec_dim != self._registered_input_dim:
                raise ValueError(
                    f"Input dimension mismatch: got {vec_dim}, expected {self._registered_input_dim}. "
                    "New data must match the dimension of previously registered data."
                )
            # Initialize adapters if they don't already exist
            self._init_or_preserve_adapters(vec_dim)

        # 5) Collect new IDs / vectors, skip those already present
        if self._has_omics and hasattr(self.processor, "lookup"):
            lookup = dict(self.processor.lookup)
        else:
            lookup = {}

        id_to_idx = dict(lookup)
        next_idx = len(id_to_idx)
        new_ids = []
        new_vecs = []

        for row in rows:
            sample_id = row[id_col]
            sample_key = f"{prefix}{sample_id}"
            if sample_key in id_to_idx:
                continue

            # Handle different input types
            vec = row[rep_col]
            if isinstance(vec, torch.Tensor):
                vec = vec.detach().cpu().numpy()
            elif isinstance(vec, list):
                vec = np.array(vec, dtype=np.float32)
            elif not isinstance(vec, np.ndarray):
                try:
                    vec = np.array(vec, dtype=np.float32)
                except Exception as err:
                    raise TypeError(
                        f"Cannot convert data at key '{sample_key}' of type {type(vec)} to numpy array. "
                        "Supported types: numpy.ndarray, torch.Tensor, list of numbers."
                    ) from err

            # Verify dimensions match with existing data or first vector
            if len(new_vecs) > 0 and vec.shape[0] != new_vecs[0].shape[0]:
                raise ValueError(
                    f"Dimension mismatch at key '{sample_key}': first={new_vecs[0].shape[0]}, current={vec.shape[0]}. "
                    f"All numeric vectors must have the same dimension."
                )

            new_ids.append(sample_key)
            new_vecs.append(vec.astype(np.float32))  # Ensure consistent dtype in numpy
            id_to_idx[sample_key] = next_idx
            next_idx += 1

        n_new = len(new_ids)
        if n_new == 0:
            logger.info("No new samples registered (all IDs already present).")
            return result_ds

        # 6) Build or update the omics matrix
        if self._has_omics and hasattr(self.omics_encoder, "embeddings"):
            # Model already has omics - get embeddings from the model
            old_matrix = self.omics_encoder.embeddings.weight.detach().cpu().numpy()

            # Verify dimensions match
            old_dim = old_matrix.shape[1]
            if new_vecs[0].shape[0] != old_dim:
                raise ValueError(
                    f"Dimension mismatch: existing={old_dim}, new={new_vecs[0].shape[0]}. "
                    f"All numeric vectors must have the same dimension."
                )

            new_matrix = np.vstack([old_matrix] + new_vecs)
        else:
            # First time adding omics or re-adding after load
            new_matrix = np.vstack(new_vecs)

        # 7) Initialize or update omics resources
        if not self._has_omics:
            # Update data type and dimension if this is first registration
            self._registered_input_dim = new_matrix.shape[1]
            self._init_omics(new_matrix)
        else:
            # Update existing omics
            self._update_omics(new_matrix)

        # 8) Update processor with new lookup
        if hasattr(self.processor, "update_omics_lookup"):
            self.processor.update_omics_lookup(id_to_idx, prefix)
        else:
            self.processor = MMContextProcessor(
                text_encoder_name=self.text_encoder_name
                if isinstance(self.text_encoder_name, str)
                else "prajjwal1/bert-tiny",
                omics_lookup=id_to_idx,
                prefix=prefix,
            )

        # 9) Report and return
        added_gb = self._estimate_memory(n_new, new_matrix.shape[1])
        logger.info(
            f"Registered {n_new} new numeric samples (total: {new_matrix.shape[0]}). "
            f"Approx. added memory: {added_gb:.3f} GB"
        )
        return result_ds

    def _update_omics(self, matrix: np.ndarray):
        """Update the omics encoder with a new embedding matrix."""
        # Only update the omics encoder, not the adapter
        self.omics_encoder = MiniOmicsModel.from_numpy(matrix)
        self._has_omics = True

    @property
    def registered_data_origin(self) -> str:
        """Get the type of omics data representation registered with this model."""
        return self._registered_data_origin

    @property
    def registered_input_dim(self) -> int | None:
        """Get the input dimension of registered data."""
        return self._registered_input_dim
