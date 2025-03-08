import json
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model

from mmcontext.pp.MMContextProcessor import MMContextProcessor

logger = logging.getLogger(__name__)


class AdapterModule(nn.Module):
    """
    Adapter to be used to map text and omics encodings into common space.

    Transforms an input tensor (batch_size, input_dim) into a 2048-dimensional
    output via a two-layer MLP with ReLU, followed by BatchNorm1d(2048).

    Parameters
    ----------
    input_dim : int
        Dimension of the input features.
    output_dim : int, optional
        Dimension of the adapter's output, by default 2048.

    References
    ----------
    The data for this module comes from either the text encoder output
    or directly from your raw omics inputs.
    """

    def __init__(self, input_dim: int, hidden_dim: int | None = 512, output_dim: int = 2048):
        super().__init__()
        if hidden_dim:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the adapter module.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            A tensor of shape (batch_size, output_dim).
        """
        return self.net(x)

    def _get_config_dict(self) -> dict:
        """
        Returns a configuration dictionary.

        Can be used to reconstruct this model (useful for saving/loading).

        Returns
        -------
        dict
            A config with essential hyperparameters.
        """
        return {
            "input_dim": self.net[0].in_features,
            "hidden_dim": self.net[0].out_features if len(self.net) > 2 else None,
            "output_dim": self.net[-2].out_features,
        }

    def save(self, output_path: str, safe_serialization: bool = True) -> None:
        """
        Saves the model configuration and state dict.

        Parameters
        ----------
        output_path : str
            Directory to save model files.
        safe_serialization : bool, optional
            If True, use safetensors; else use torch.save.
        """
        os.makedirs(output_path, exist_ok=True)
        if safe_serialization:
            model_path = os.path.join(output_path, "model.safetensors")
            save_safetensors_model(self, model_path)
        else:
            model_path = os.path.join(output_path, "pytorch_model.bin")
            torch.save(self.state_dict(), model_path)

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
        AdapterModule
            The loaded model instance.
        """
        model = AdapterModule(0)
        if safe_serialization and os.path.exists(os.path.join(input_path, "model.safetensors")):
            load_safetensors_model(model, os.path.join(input_path, "model.safetensors"))
        else:
            model.load_state_dict(
                torch.load(os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"))
            )
        return model


class MMContextEncoder(nn.Module):
    """
    A multi-modal encoder that handles text and omics data.

    - Text inputs via a Hugging Face transformer (always enabled).
    - Omics inputs by passing them directly through an adapter module (prevously computed embeddings stored in anndata .obsm)

    The output of each modality is projected to 2048 dimensions via a two-layer
    MLP + ReLU + BatchNorm. The final "sentence_embedding" merges text/omics
    entries in the order specified by 'omics_text_info'.

    Parameters
    ----------
    text_encoder_name : str
        Name of the Hugging Face transformer model (e.g., "bert-base-uncased").
    omics_input_dim : int
        The dimension of the omics input. If you pass a tensor of shape (batch_size, omics_input_dim),
        it will go straight to the omics_adapter.
    processor_obsm_key : str, optional
        The .obsm key the MMContextProcessor will use to retrieve omics data, by default "X_pp".
    freeze_text_encoder : bool, optional
        If True, freezes all parameters of the text encoder.
    unfreeze_last_n_layers : int, optional
        If freeze_text_encoder is True, this unfreezes the last n layers of the encoder.

    Examples
    --------
    >>> model = MMContextEncoder(
    ...     text_encoder_name="bert-base-uncased",
    ...     omics_input_dim=1000,
    ... )
    >>> # Prepare your features dict
    >>> features = {
    ...     "input_ids": torch.randint(0, 1000, (4, 16)),  # (batch, seq_len)
    ...     "attention_mask": torch.ones((4, 16)),
    ...     "omics_representation": torch.randn(4, 1000),
    ...     "omics_text_info": [0, 1, 1, 0],  # For merging
    ... }
    >>> output = model(features)
    >>> output["sentence_embedding"].shape
    torch.Size([4, 2048])
    """

    def __init__(
        self,
        text_encoder_name: str,
        omics_input_dim: int,
        freeze_text_encoder: bool = False,
        unfreeze_last_n_layers: int = 0,
        adapter_hidden_dim: int = 512,
        adapter_output_dim: int = 2048,
        processor_name: str = "precomputed",
        # store_path: str = None,
        processor_obsm_key: str = "X_pca",
        # feature_dtype: str = "float32",
    ):
        super().__init__()
        self.text_encoder_name = text_encoder_name
        self.omics_input_dim = omics_input_dim
        self.freeze_text_encoder = freeze_text_encoder
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.adapter_hidden_dim = adapter_hidden_dim
        self.adapter_output_dim = adapter_output_dim
        self.processor_name = processor_name
        # self.store_path = store_path
        # self.feature_dtype = feature_dtype
        self.processor_obsm_key = processor_obsm_key

        # Initialize the text encoder
        self.text_encoder = transformers.AutoModel.from_pretrained(text_encoder_name)
        self._manage_text_encoder_freezing()

        # Adapters (both produce 2048-d outputs)
        text_hidden_size = self.text_encoder.config.hidden_size
        self.text_adapter = AdapterModule(
            input_dim=text_hidden_size, hidden_dim=adapter_hidden_dim, output_dim=adapter_output_dim
        )
        self.omics_adapter = AdapterModule(
            input_dim=omics_input_dim, hidden_dim=adapter_hidden_dim, output_dim=adapter_output_dim
        )

        # For tokenization / omics retrieval
        # Combine all processor-related parameters
        processor_params = {
            "processor_name": processor_name,
            "text_encoder_name": text_encoder_name,
        }

        # Add store_path and feature_dtype if they're provided
        # if store_path is not None:
        #     processor_params["store_path"] = store_path
        # if feature_dtype is not None:
        #     processor_params["dtype"] = feature_dtype
        if processor_obsm_key is not None:
            processor_params["obsm_key"] = processor_obsm_key

        self.processor = MMContextProcessor(**processor_params)

    def _manage_text_encoder_freezing(self):
        """Freezes all parameters in the text encoder if required, and optionally unfreezes the last n layers."""
        if self.freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

            if self.unfreeze_last_n_layers > 0:
                # For BERT-like models
                if hasattr(self.text_encoder, "encoder"):
                    layers = self.text_encoder.encoder.layer[-self.unfreeze_last_n_layers :]
                # For RoBERTa-like models
                elif hasattr(self.text_encoder, "roberta"):
                    layers = self.text_encoder.roberta.encoder.layer[-self.unfreeze_last_n_layers :]
                else:
                    raise ValueError(f"Unsupported architecture for {self.text_encoder_name}")

                for layer in layers:
                    for param in layer.parameters():
                        param.requires_grad = True

    def forward(self, features: dict) -> dict:
        """
        Forward pass that processes text and omics inputs.

        Each input is mapped to a common dimensionality via its adapter, and the results are merged into one
        'sentence_embedding' tensor.

        Parameters
        ----------
        features : dict
            Dictionary containing:
            - "input_ids", "token_type_ids", "attention_mask": text inputs
            (shape: (num_text_examples, ...))
            - "omics_representation": omics inputs
            (shape: (num_omics_examples, omics_input_dim))
            - "omics_text_info": A sequence of 0s and 1s of length = total_batch_size
            (i.e., num_omics_examples + num_text_examples). A value of 0 indicates
            the corresponding row in the final output is omics; a value of 1 indicates text.

            Data is sourced from: your internal multimodal data loader that batches
            omics and text in parallel.

        Returns
        -------
        dict
            The same 'features' dictionary with a new key "sentence_embedding":
            a tensor of shape (total_batch_size, adapter_output_dim). The row order
            aligns with 'omics_text_info'.
        """
        # 1. Prepare omics embeddings
        omics_embeds = None
        if "omics_representation" in features:
            omics_embeds = self.omics_adapter(features["omics_representation"])
            # shape: (num_omics_examples, adapter_output_dim)

        # 2. Prepare text embeddings
        text_embeds = None
        if "input_ids" in features:
            text_output = self.text_encoder(
                input_ids=features["input_ids"],
                token_type_ids=features.get("token_type_ids"),
                attention_mask=features.get("attention_mask"),
            )
            # text_output[1] is typically the pooled (CLS) embedding for many HF models
            pooled = text_output[1]  # shape: (num_text_examples, hidden_size)
            text_embeds = self.text_adapter(pooled)  # shape: (num_text_examples, adapter_output_dim)

        # 3. Convert omics_text_info to a PyTorch tensor for boolean indexing
        omics_text_info = features["omics_text_info"]

        # If it's already a tensor, great; if not, convert it
        if not isinstance(omics_text_info, torch.Tensor):
            # Use a long tensor for indexing
            # We place it on the same device as omics_embeds or text_embeds, whichever is available
            if omics_embeds is not None:
                device = omics_embeds.device
            elif text_embeds is not None:
                device = text_embeds.device
            else:
                device = torch.device("cpu")
            omics_text_info = torch.tensor(omics_text_info, dtype=torch.long, device=device)

        # 4. Create the final "sentence_embedding" placeholder
        total_batch_size = omics_text_info.shape[0]
        # Figure out the output dimension from whichever embeddings exist
        if omics_embeds is not None:
            adapter_output_dim = omics_embeds.shape[-1]
            dtype = omics_embeds.dtype
            device = omics_embeds.device
        else:
            adapter_output_dim = text_embeds.shape[-1]
            dtype = text_embeds.dtype
            device = text_embeds.device

        sentence_embedding = torch.zeros((total_batch_size, adapter_output_dim), device=device, dtype=dtype)

        # 5. Identify positions of omics (0) vs. text (1)
        zero_positions = (omics_text_info == 0).nonzero(as_tuple=True)[0]  # indices for omics
        one_positions = (omics_text_info == 1).nonzero(as_tuple=True)[0]  # indices for text

        # 6. Assign omics embeddings
        if zero_positions.numel() > 0:
            # Verify omics_embeds is not None and shapes align
            if omics_embeds is None:
                raise ValueError("Found omics entries in omics_text_info but no omics_embeds provided.")
            if zero_positions.shape[0] != omics_embeds.shape[0]:
                raise ValueError(
                    f"Inconsistent shapes: omics_text_info indicates {zero_positions.shape[0]} omics rows, "
                    f"but omics_embeds has {omics_embeds.shape[0]}."
                )
            sentence_embedding[zero_positions] = omics_embeds

        # 7. Assign text embeddings
        if one_positions.numel() > 0:
            # Verify text_embeds is not None and shapes align
            if text_embeds is None:
                raise ValueError("Found text entries in omics_text_info but no text_embeds provided.")
            if one_positions.shape[0] != text_embeds.shape[0]:
                raise ValueError(
                    f"Inconsistent shapes: omics_text_info indicates {one_positions.shape[0]} text rows, "
                    f"but text_embeds has {text_embeds.shape[0]}."
                )
            sentence_embedding[one_positions] = text_embeds

        # 8. Store in features and return
        features["sentence_embedding"] = sentence_embedding
        return features

    def tokenize(self, texts, padding: str | bool = True) -> dict:
        """
        Tokenizes string inputs as text or interprets JSON definitions for omics data. Omics data is then obtained from the anndata reference.

        Then returns a dictionary containing:
        - "input_ids", "attention_mask" (if text was found)
        - "omics_representation" (if omics data was found)
        - "omics_text_info" - which indicates 0=omics, 1=text ordering.

        Parameters
        ----------
        texts : list
            A list that can contain either normal text strings or JSON-like strings
            that define omics data (e.g., containing a "file_path").
        padding : str or bool, optional
            Passed to the Hugging Face tokenizer, by default True.

        Returns
        -------
        dict
            A dictionary with tokenized text fields, omics data, and "omics_text_info".
        """
        omics_reps = []
        text_vals = []
        omics_text_info = []

        for data in texts:
            if isinstance(data, int):
                # integers are sample indices for the datastore and can be used by OptimizedProcessor
                omics_reps.append(data)
                omics_text_info.append(0)
            elif isinstance(data, dict):  # and "file_record" in data.keys():
                # This indicates a JSON definition for omics
                data_dict = data
                omics_reps.append(data_dict)
                omics_text_info.append(0)
            elif isinstance(data, str):
                # Normal text
                text_vals.append(data)
                omics_text_info.append(1)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

        encoding = {}
        # Tokenize text if any
        if text_vals:
            encoding = self.processor.tokenizer(text_vals, return_tensors="pt", padding=padding)

        # Process omics if any
        if omics_reps:
            omics_features = self.processor.omics_processor.get_rep(omics_reps)
            encoding["omics_representation"] = omics_features

        encoding["omics_text_info"] = omics_text_info
        return encoding

    def _get_sentence_embedding_dimension(self) -> int:
        """
        Returns the dimension of the final sentence embedding.

        Returns
        -------
        int
            2048.
        """
        return 2048

    def _get_config_dict(self) -> dict:
        """
        Returns a configuration dictionary

        Can be used to reconstruct this model (useful for saving/loading).

        Returns
        -------
        dict
            A config with essential hyperparameters.
        """
        return {
            "text_encoder_name": self.text_encoder_name,
            "omics_input_dim": self.omics_input_dim,
            "freeze_text_encoder": self.freeze_text_encoder,
            "unfreeze_last_n_layers": self.unfreeze_last_n_layers,
            "adapter_hidden_dim": self.adapter_hidden_dim,
            "adapter_output_dim": self.adapter_output_dim,
            "processor_name": self.processor_name,
            "store_path": self.store_path,
            "feature_dtype": self.feature_dtype,
            # Add any other processor kwargs that should be saved
        }

    def save(self, output_path: str, safe_serialization: bool = True) -> None:
        """
        Saves the model configuration and state dict.

        Parameters
        ----------
        output_path : str
            Directory to save model files.
        safe_serialization : bool, optional
            If True, use safetensors; else use torch.save.
        """
        os.makedirs(output_path, exist_ok=True)
        # You can clear any cache in self.processor if needed
        self.processor.omics_processor.clear_cache()

        config_path = os.path.join(output_path, "config.json")
        with open(config_path, "w") as fOut:
            json.dump(self._get_config_dict(), fOut, indent=2)

        if safe_serialization:
            model_path = os.path.join(output_path, "model.safetensors")
            save_safetensors_model(self, model_path)
        else:
            model_path = os.path.join(output_path, "pytorch_model.bin")
            torch.save(self.state_dict(), model_path)

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
            The loaded model instance.
        """
        config_path = os.path.join(input_path, "config.json")
        with open(config_path) as fIn:
            cfg = json.load(fIn)

        model = MMContextEncoder(**cfg)
        if safe_serialization and os.path.exists(os.path.join(input_path, "model.safetensors")):
            load_safetensors_model(model, os.path.join(input_path, "model.safetensors"))
        else:
            state_dict = torch.load(os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)

        return model

    @property
    def tokenizer(self) -> MMContextProcessor:
        """Convenience property returning the underlying processor object, which includes the text tokenizer and omics data processor."""
        return self.processor


'''
class BaseModel(nn.Module, metaclass=abc.ABCMeta):
    """BaseModel defines common methods for all models."""

    def __init__(self):
        super().__init__()
        #self.logger = logger or logging.getLogger(__name__)

    @abc.abstractmethod
    def forward(self, *inputs):
        """Forward pass to be implemented by subclasses."""
        pass
'''

'''
class MMContextEncoder(nn.Module):
    """A model containing submodels for processing text (context information) and omics data (a count vector for each sample).

    The model is supposed to be used with the SentenceTransformer framework. It processes text data and omics data
    Text is tokenized and embedded with a pre-trained transformer model, while omics data is processed
    with a custom model. The input should be a class, containing a "texts" attribute, which has a pair of omics and text
    data. The omics data, which is a count vector for each sample expected to be a tensor or a numpy.array.

    Parameters
    ----------
    model_name
        Name of the model to be used for text data. Can be a huggingface model name.
    processor_obsm_key
        The processor will retrieve a representation from the AnnData object .obsm with this key. It has to be computed beforehand and by
        default expects to find the represenation in adata.obsm["X_pp"].
    cfg
        Configuration dictionary for the model. Has to have the following keys:
        - text_encoder_name (str): Name of the transformer model to be used for text data.
        - omics_encoder_cfg (dict): Configuration dictionary for the omics encoder model. See help(OmicsEncoder) for the parameters.
    """

    def __init__(self, text_encoder_name, omics_encoder_cfg, processor_obsm_key="X_pp"):
        super().__init__()
        self.text_encoder_name = text_encoder_name
        self.processor_obsm_key = processor_obsm_key
        self.model = self._load_model(text_encoder_name, omics_encoder_cfg)
        self.processor = self._load_processor(text_encoder_name, processor_obsm_key)

    def _load_model(self, text_encoder_name, cfg):
        return MMContextModel(text_encoder_name, cfg)

    def _load_processor(self, text_encoder_name, processor_obsm_key):
        return MMContextProcessor(obsm_key=processor_obsm_key, text_encoder_name=text_encoder_name)

    def forward(self, features):
        """Forward pass for MMContextEncoder.

        The input should be a dictionary containing omics and/or text data.
        These input features are first extracted with the tokenizer, so that they are torch tensors.
        The "omics_representation" string needs to be handled explicitly in the SentenceTransformerTrainer.collect_features method.
        """
        omics_embeds = []
        text_embeds = []

        if "omics_representation" in features:
            # feature_dim = features["omics_representation"].size(1)
            # device = next(self.parameters()).device

            # Update omics_projection if feature_dim changes
            # if feature_dim != self.model.omics_projection.in_features:
            #    self.model.omics_projection = nn.Linear(feature_dim, self.model.omics_encoder.embedding_dim, device=device)

            # features["omics_representation"] = self.model.omics_projection(features["omics_representation"])
            omics_embeds = self.model.omics_encoder(features)

        if "input_ids" in features:
            text_output = self.model.text_encoder(
                input_ids=features["input_ids"],
                token_type_ids=features["token_type_ids"],
                attention_mask=features["attention_mask"],
            )
            text_embeds = self.model.text_projection(text_output[1])

        sentence_embedding = []
        omics_features = iter(omics_embeds)
        text_features = iter(text_embeds)

        for _idx, input_type in enumerate(features["omics_text_info"]):
            if input_type == 0:
                sentence_embedding.append(next(omics_features))
            else:
                sentence_embedding.append(next(text_features))

        features["sentence_embedding"] = torch.stack(sentence_embedding).float()
        return features

    def tokenize(self, texts, padding: str | bool = True) -> dict[str, torch.Tensor]:
        """Tokenizes the input texts and returns the encoded tensors.

        Uses the tokenizer of the specified text encoder model from huggingface. The input texts is a list of multiple python objects.
        Strings will be handled by the text encoder, while dictionaries with the key "is_omics" set to True will be handled by the omics processor.
        """
        omics_representations = []
        texts_values = []
        omics_text_info = []

        for _idx, data in enumerate(texts):
            if '{"file_path"' in data:
                data = json.loads(data)  # convert string to dictionary
                omics_representations.append(data)
                omics_text_info.append(0)
            elif isinstance(data, str):
                texts_values.append(data)
                omics_text_info.append(1)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

        encoding = {}
        if len(texts_values):
            encoding = self.processor.tokenizer(texts_values, return_tensors="pt", padding=padding)

        if len(omics_representations):
            omics_features = self.processor.omics_processor.get_rep(omics_representations)
            encoding["omics_representation"] = omics_features
        # You need to change SentenceTransfomerTrainer.collect_features to include the the "omics_representation" string.
        encoding["omics_text_info"] = omics_text_info
        return dict(encoding)

    def _get_sentence_embedding_dimension(self) -> int:
        return self.model.embedding_dim

    def _get_config_dict(self) -> dict:
        return {
            "text_encoder_name": self.text_encoder_name,
            "processor_obsm_key": self.processor_obsm_key,
            "omics_encoder_cfg": self.model.omics_encoder._get_config_dict(),
        }

    def save(self, output_path: str, safe_serialization: bool = True) -> None:
        """Saves the model configuration and state."""
        os.makedirs(output_path, exist_ok=True)
        # clear the cache of omics processor
        self.processor.omics_processor.clear_cache()
        # Save config
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self._get_config_dict(), fOut, indent=2)

        # Save model state
        if safe_serialization:
            save_safetensors_model(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def load(input_path: str, safe_serialization: bool = True):
        """Loads the model from the given path."""
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        model = MMContextEncoder(**config)

        if os.path.exists(os.path.join(input_path, "model.safetensors")):
            load_safetensors_model(model, os.path.join(input_path, "model.safetensors"))
        else:
            model.load_state_dict(
                torch.load(os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"))
            )
        return model

    @property
    def tokenizer(self) -> MMContextProcessor:
        """Returns the tokenizer."""
        return self.processor


class MMContextModel(nn.Module):
    """A combined model, loading a transformer model for text data and a customizable model based on the OmicsEncoder for omics data.

    The text encoder is a transformer model, which can be loaded from huggingface transfomers.
    The omics encoder is a customizable model, based on TransformerEncoderLayers. It can be configured
    as an MLP only model, or with attention heads.
    """

    def __init__(self, text_encoder_name, omics_encoder_cfg, freeze_text_encoder=False, unfreeze_last_n_layers=0):
        super().__init__()
        self.text_encoder_name = text_encoder_name
        self.omics_encoder_cfg = omics_encoder_cfg
        self.embedding_dim = omics_encoder_cfg["embedding_dim"]

        # Load the text encoder
        self.text_encoder = transformers.AutoModel.from_pretrained(text_encoder_name)

        # Freeze the text encoder parameters
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

            # Unfreeze the last n transformer layers if specified
            if unfreeze_last_n_layers > 0:
                # For BERT-like models
                if hasattr(self.text_encoder, "encoder"):
                    layers = self.text_encoder.encoder.layer[-unfreeze_last_n_layers:]
                # For RoBERTa-like models
                elif hasattr(self.text_encoder, "roberta"):
                    layers = self.text_encoder.roberta.encoder.layer[-unfreeze_last_n_layers:]
                else:
                    raise ValueError(f"Unsupported model architecture for {text_encoder_name}")

                for layer in layers:
                    for param in layer.parameters():
                        param.requires_grad = True

        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, self.embedding_dim)
        self.omics_encoder = OmicsEncoder(**omics_encoder_cfg)

'''
