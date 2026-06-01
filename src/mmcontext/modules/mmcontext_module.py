"""MMContextModule — InputModule for multimodal text + omics encoding.

This module serves as the first module in a sentence-transformers (>=5.4)
pipeline, handling both text preprocessing (via AutoTokenizer + AutoModel)
and omics vector pass-through (via VectorStore or direct input). It produces
a unified features dict that downstream modules (AdapterModule, Pooling)
consume.

Architecture
------------
The module operates in two modes depending on the input modality:

**Text mode** (``modality="text"``):
    Input strings are tokenized with AutoTokenizer, then forwarded through
    AutoModel to produce contextual token embeddings.

**Omics mode** (``modality="omics"``):
    Omics vectors are either looked up from an attached :class:`VectorStore`
    (using prefixed string IDs) or provided directly as numpy arrays (via dict
    inputs). These vectors are stored as ``input_values`` and passed through to
    ``token_embeddings`` in the forward output without any
    learned transformation — the downstream AdapterModule handles projection.

Features dict contract (after ``forward()``)::

    {
        "token_embeddings": Tensor[B, L, D],  # per-token representations
        "attention_mask":   Tensor[B, L],      # 1 = real, 0 = pad
        "modality_ids":     Tensor[B, L],      # 0 = text, 1 = omics
    }

Example
-------
>>> from mmcontext.modules import MMContextModule
>>> module = MMContextModule("pubmedbert-base")
>>> features = module.preprocess(["A cell with high EGFR expression."])
>>> result = module.forward(features)
>>> result["token_embeddings"].shape
torch.Size([1, 8, 32])
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Try v5.4+ import paths first, fall back to v5.0 paths
try:
    from sentence_transformers.base.modules import InputModule
except ImportError:
    from sentence_transformers.models import InputModule

from mmcontext.io import VectorStore

logger = logging.getLogger(__name__)

# Modality constants used in modality_ids tensor
MODALITY_TEXT = 0
MODALITY_OMICS = 1
MODALITY_PAD = 2


class MMContextModule(InputModule):
    """Multimodal InputModule for text + omics encoding.

    Extends :class:`sentence_transformers.InputModule` (v5.4+) to support
    both text and continuous omics vectors in a single pipeline.

    Parameters
    ----------
    model_name_or_path : str
        Name or path for the text encoder (passed to ``AutoModel.from_pretrained``
        and ``AutoTokenizer.from_pretrained``).
    max_seq_length : int, optional
        Maximum sequence length for text tokenization. Default: 512.
    omics_prefix : str, optional
        String prefix that marks an input as an omics sample ID to be resolved
        via the attached :class:`VectorStore`. Default: ``"omics:"``.
    tokenizer_args : dict, optional
        Extra keyword arguments forwarded to ``AutoTokenizer.from_pretrained``.
    model_args : dict, optional
        Extra keyword arguments forwarded to ``AutoModel.from_pretrained``.
    """

    config_keys: list[str] = [
        "model_name_or_path",
        "max_seq_length",
        "omics_prefix",
    ]
    config_file_name: str = "mmcontext_module_config.json"
    save_in_root: bool = True

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 512,
        omics_prefix: str = "omics:",
        tokenizer_args: dict | None = None,
        model_args: dict | None = None,
    ) -> None:
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self._max_seq_length = max_seq_length
        self.omics_prefix = omics_prefix

        # Text encoder
        model_args = model_args or {}
        self.auto_model = AutoModel.from_pretrained(model_name_or_path, **model_args)

        # Tokenizer (stored as self.tokenizer for InputModule compatibility)
        tokenizer_args = tokenizer_args or {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_args)

        # VectorStore for omics ID resolution (not saved with model)
        self._vector_store: VectorStore | None = None

    # ------------------------------------------------------------------
    # Modality metadata (v5.4+ API)
    # ------------------------------------------------------------------
    @property
    def modalities(self) -> list[str]:
        """Modalities this module can process.

        Returns ``["text", "omics"]``. Note that ``"omics"`` is a custom
        modality not in the standard ST set (text, image, audio, video).
        """
        return ["text", "omics"]

    # ------------------------------------------------------------------
    # Sequence length
    # ------------------------------------------------------------------
    @property
    def max_seq_length(self) -> int:
        """Maximum sequence length for text tokenization."""
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        self._max_seq_length = value

    # ------------------------------------------------------------------
    # VectorStore management
    # ------------------------------------------------------------------
    def set_vector_store(self, store: VectorStore) -> None:
        """Attach a VectorStore for resolving omics sample IDs.

        Parameters
        ----------
        store : VectorStore
            The store mapping sample IDs to embedding vectors.
        """
        self._vector_store = store
        logger.info("Attached VectorStore with %d entries, dim=%d", len(store), store.dim)

    def remove_vector_store(self) -> None:
        """Detach the current VectorStore."""
        self._vector_store = None
        logger.info("Detached VectorStore")

    # ------------------------------------------------------------------
    # Preprocess (v5.4+ API — replaces tokenize())
    # ------------------------------------------------------------------
    def preprocess(
        self,
        inputs: list,
        prompt: str | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | Any]:
        """Preprocess inputs, routing by modality.

        This is the v5.4+ entry point (replaces the deprecated ``tokenize()``).
        The method detects the input modality and dispatches accordingly.

        Parameters
        ----------
        inputs : list[str | dict]
            Inputs to process. Three formats are supported:

            1. **Plain strings** — preprocessed as text via AutoTokenizer.
            2. **Prefixed strings** (e.g., ``"omics:cell_42"``) — the suffix
               after the prefix is looked up in the attached VectorStore.
            3. **Dicts** with ``"omics_values"`` key — the value is either a
               single 1-D numpy array (obs case) or a list of 1-D arrays
               (var case, variable-length gene sequences).

        prompt : str, optional
            Optional prompt to prepend to text inputs.
        **kwargs
            Additional keyword arguments (e.g. ``task``).

        Returns
        -------
        dict[str, Tensor | Any]
            Features dict. For text: ``{input_ids, attention_mask, modality}``.
            For omics: ``{input_values, attention_mask, modality}``.

        Raises
        ------
        ValueError
            If the input list is empty, or if prefixed omics IDs are used
            without an attached VectorStore.
        KeyError
            If an omics ID is not found in the VectorStore.
        """
        if len(inputs) == 0:
            raise ValueError("Empty input list: at least one input is required.")

        # Dispatch based on input type
        if isinstance(inputs[0], dict) and "omics_values" in inputs[0]:
            return self._preprocess_omics_direct(inputs)
        elif isinstance(inputs[0], str) and inputs[0].startswith(self.omics_prefix):
            return self._preprocess_omics_via_store(inputs)
        else:
            return self._preprocess_text(inputs, prompt=prompt)

    def _preprocess_text(
        self,
        texts: list[str],
        prompt: str | None = None,
    ) -> dict[str, torch.Tensor | Any]:
        """Preprocess plain text strings via AutoTokenizer."""
        if prompt:
            texts = [prompt + t for t in texts]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_seq_length,
            return_tensors="pt",
        )
        encoded["modality"] = "text"
        return encoded

    def _preprocess_omics_via_store(
        self, texts: list[str]
    ) -> dict[str, torch.Tensor | Any]:
        """Resolve prefixed omics IDs through VectorStore."""
        if self._vector_store is None:
            raise ValueError(
                "No VectorStore attached. Call set_vector_store() before "
                "preprocessing omics IDs. Received input starting with "
                f"'{self.omics_prefix}'."
            )

        # Strip prefix and look up vectors
        prefix_len = len(self.omics_prefix)
        sample_ids = [t[prefix_len:] for t in texts]

        vectors = []
        for sid in sample_ids:
            vec = self._vector_store[sid]  # raises KeyError if missing
            vectors.append(torch.from_numpy(vec).unsqueeze(0))  # (1, D)

        # Stack into (B, 1, D) — each sample is a single obs-level vector
        input_values = torch.stack(vectors, dim=0)  # (B, 1, D)
        attention_mask = torch.ones(len(texts), 1, dtype=torch.long)

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "modality": "omics",
        }

    def _preprocess_omics_direct(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """Package direct omics vectors into features dict.

        Handles both obs (single vector per sample) and var (list of gene
        vectors per sample, padded to max length in batch).
        """
        all_embeddings = []
        lengths = []

        for item in inputs:
            values = item["omics_values"]

            if isinstance(values, np.ndarray) and values.ndim == 1:
                # Obs case: single vector → (1, D)
                all_embeddings.append(torch.from_numpy(values).unsqueeze(0))
                lengths.append(1)
            elif isinstance(values, (list, tuple)):
                # Var case: list of gene vectors → (N_genes, D)
                gene_tensors = [torch.from_numpy(np.asarray(v)) for v in values]
                all_embeddings.append(torch.stack(gene_tensors, dim=0))
                lengths.append(len(values))
            elif isinstance(values, np.ndarray) and values.ndim == 2:
                # Var case: (N_genes, D) array
                all_embeddings.append(torch.from_numpy(values))
                lengths.append(values.shape[0])
            else:
                raise ValueError(
                    f"Unsupported omics_values type: {type(values)}. "
                    "Expected 1-D array, 2-D array, or list of 1-D arrays."
                )

        # Pad to max length in batch
        max_len = max(lengths)
        dim = all_embeddings[0].shape[-1]
        batch_size = len(inputs)

        input_values = torch.zeros(batch_size, max_len, dim)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, (emb, length) in enumerate(zip(all_embeddings, lengths)):
            input_values[i, :length] = emb
            attention_mask[i, :length] = 1

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "modality": "omics",
        }

    # ------------------------------------------------------------------
    # Forward (Module abstract method)
    # ------------------------------------------------------------------
    def forward(
        self,
        features: dict[str, torch.Tensor | Any],
        **kwargs,
    ) -> dict[str, torch.Tensor | Any]:
        """Process features through the text encoder or pass omics through.

        Parameters
        ----------
        features : dict
            Features dict from :meth:`preprocess`. Must contain either
            ``input_ids`` (text) or ``input_values`` (omics), plus
            ``attention_mask`` and ``modality``.

        Returns
        -------
        dict[str, Tensor | Any]
            Updated features dict with ``token_embeddings``, ``attention_mask``,
            and ``modality_ids``.
        """
        modality = features.get("modality", "text")

        if modality == "text":
            return self._forward_text(features)
        else:
            return self._forward_omics(features)

    def _forward_text(
        self, features: dict[str, torch.Tensor | Any]
    ) -> dict[str, torch.Tensor | Any]:
        """Run text through the transformer encoder."""
        input_ids = features["input_ids"]
        attention_mask = features.get("attention_mask")

        # Forward through text encoder
        model_output = self.auto_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        token_embeddings = model_output.last_hidden_state  # (B, L, D)

        B, L = input_ids.shape
        modality_ids = torch.full(
            (B, L), MODALITY_TEXT, dtype=torch.long, device=input_ids.device
        )

        features["token_embeddings"] = token_embeddings
        features["modality_ids"] = modality_ids
        return features

    def _forward_omics(
        self, features: dict[str, torch.Tensor | Any]
    ) -> dict[str, torch.Tensor | Any]:
        """Pass omics embeddings through unchanged.

        Reads from ``input_values`` (set by preprocess) and writes to
        ``token_embeddings`` (the standard key consumed by downstream modules).
        Using ``input_values`` as the preprocess key ensures compatibility with
        the ST training collator's ``collect_features`` suffix matching.
        """
        token_embeddings = features["input_values"]  # (B, L, D)

        B, L = token_embeddings.shape[:2]
        modality_ids = torch.full(
            (B, L), MODALITY_OMICS, dtype=torch.long, device=token_embeddings.device
        )

        features["token_embeddings"] = token_embeddings
        features["modality_ids"] = modality_ids
        return features

    # ------------------------------------------------------------------
    # Properties for downstream modules
    # ------------------------------------------------------------------
    def get_word_embedding_dimension(self) -> int:
        """Return the hidden size of the text encoder.

        This is used by downstream modules (Pooling, AdapterModule) to
        determine the text embedding dimension.
        """
        return self.auto_model.config.hidden_size

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------
    def freeze_text_encoder(self, num_layers: int | None = None) -> None:
        """Freeze text encoder parameters.

        Parameters
        ----------
        num_layers : int, optional
            If given, freeze only the first ``num_layers`` layers. The
            remaining layers and the pooler (if any) stay trainable.
            If ``None``, freeze all parameters.
        """
        if num_layers is None:
            for param in self.auto_model.parameters():
                param.requires_grad = False
            logger.info("Froze all text encoder parameters")
        else:
            self._freeze_n_layers(num_layers)

    def unfreeze_text_encoder(self) -> None:
        """Unfreeze all text encoder parameters."""
        for param in self.auto_model.parameters():
            param.requires_grad = True
        logger.info("Unfroze all text encoder parameters")

    def _freeze_n_layers(self, num_layers: int) -> None:
        """Freeze the first ``num_layers`` encoder layers.

        Handles both BERT-style (``encoder.layer``) and RoBERTa-style
        (``roberta.encoder.layer``) architectures.
        """
        layers = None
        if hasattr(self.auto_model, "encoder") and hasattr(
            self.auto_model.encoder, "layer"
        ):
            layers = self.auto_model.encoder.layer
        elif hasattr(self.auto_model, "roberta"):
            layers = self.auto_model.roberta.encoder.layer
        elif hasattr(self.auto_model, "bert"):
            layers = self.auto_model.bert.encoder.layer

        if layers is None:
            logger.warning(
                "Could not identify encoder layers for partial freezing. "
                "Freezing all parameters instead."
            )
            for param in self.auto_model.parameters():
                param.requires_grad = False
            return

        for i, layer in enumerate(layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        logger.info("Froze first %d text encoder layers", num_layers)

    # ------------------------------------------------------------------
    # Save / Load (Module abstract methods)
    # ------------------------------------------------------------------
    def save(
        self,
        output_path: str,
        *args,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        """Save module config, text encoder weights, and tokenizer.

        The VectorStore is intentionally NOT saved — it must be reattached
        after loading via :meth:`set_vector_store`.

        Parameters
        ----------
        output_path : str
            Directory where files will be written.
        safe_serialization : bool
            If True, use safetensors format for weights.
        """
        output_path = str(output_path)
        os.makedirs(output_path, exist_ok=True)

        # Save config
        self.save_config(output_path)

        # Save text encoder
        self.auto_model.save_pretrained(output_path, safe_serialization=safe_serialization)

        # Save tokenizer
        self.save_tokenizer(output_path)

        logger.info("Saved MMContextModule to %s", output_path)

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
    ) -> MMContextModule:
        """Load a saved MMContextModule from disk.

        Parameters
        ----------
        model_name_or_path : str
            Path to directory containing saved module files.
        subfolder : str
            Optional subdirectory.
        **kwargs
            Additional arguments (ignored, for API compatibility).

        Returns
        -------
        MMContextModule
        """
        config = cls.load_config(
            model_name_or_path,
            subfolder=subfolder,
            config_filename=cls.config_file_name,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

        load_path = model_name_or_path
        if subfolder:
            load_path = os.path.join(model_name_or_path, subfolder)

        module = cls(
            model_name_or_path=load_path,
            max_seq_length=config.get("max_seq_length", 512),
            omics_prefix=config.get("omics_prefix", "omics:"),
        )

        logger.info("Loaded MMContextModule from %s", load_path)
        return module

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"MMContextModule("
            f"model={self.model_name_or_path}, "
            f"max_seq_length={self.max_seq_length}, "
            f"omics_prefix='{self.omics_prefix}', "
            f"store={'attached' if self._vector_store else 'none'}"
            f")"
        )
