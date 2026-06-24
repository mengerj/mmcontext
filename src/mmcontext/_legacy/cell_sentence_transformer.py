import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model


class OmicsEncoder(nn.Module):
    """OmicsEncoder builds an encoder model using torchs TransformerEncoder and custom encoder layers.

    This model is supposed to be used with the SentenceTransformer framework. It processes omics data, while
    the input needs to already contain initial embeddings of the omics data. The embedding dimension has to match the input
    dimension of these intial embeddings. The output of the model will be the same dimension as the input embeddings.

    Parameters
    ----------
    embedding_dim
        Dimension of the input embeddings.
    hidden_dim
        Dimension of the feedforward network (MLP).
    num_layers
        Number of encoder layers.
    num_heads
        Number of attention heads.
    use_self_attention
        Whether to include self-attention.
    use_cross_attention
        Whether to include cross-attention. Not sure if this works yet
    activation
        Activation function for the feedforward network. Can be "relu" or a callable.
    dropout
        Dropout probability.
    learn_temperature
        Whether to learn a temperature parameter for constrastive Loss. Not working yet.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        use_self_attention: bool = False,
        use_cross_attention: bool = False,
        activation: str = "relu",
        dropout: float = 0.1,
        learn_temperature: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        self.activation = activation
        self.dropout = dropout
        self.learn_temperature = learn_temperature

        encoder_layer = CustomTransformerEncoderLayer(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            use_self_attention=use_self_attention,
            use_cross_attention=use_cross_attention,
            activation=activation,
            dropout=dropout,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # default True have a warning
        )

        # self.temperature_module = LearnableTemperature()
        # self.temperature = None  # Can be set by downstream methods to set a fixed temperature for the loss function.

    def forward(
        self,
        inputs: dict,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for OmicsEncoder compatible with SentenceTransformer.

        Parameters
        ----------
        inputs : dict
            Dictionary containing input tensors. Expected keys:
            - `token_embeddings` (Tensor): Precomputed embeddings of shape (batch_size, seq_length, embedding_dim).
            - `attention_mask` (Tensor, optional): Attention mask of shape (batch_size, seq_length).
        **kwargs
            Additional arguments for compatibility.

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, seq_length, embedding_dim).
        """
        # Extract relevant inputs
        in_main = inputs.get("omics_representation", None)
        in_main_key_padding_mask = inputs.get("attention_mask", None)
        if in_main_key_padding_mask is not None:
            in_main_key_padding_mask = in_main_key_padding_mask.bool()

        if in_main is None:
            raise ValueError("`omics` must be provided in the input dictionary.")

        # Validate dimensions
        # if in_main.dim() != 3:
        #    raise ValueError(
        #        f"Expected in_main to have 3 dimensions (batch_size, seq_length, embedding_dim), but got {in_main.dim()}."
        #    )
        # Add a sequence dimension if it doesn't exist
        if in_main.dim() == 2:  # Shape is (batch_size, embedding_dim)
            in_main = in_main.unsqueeze(1)  # Add seq_length dimension -> (batch_size, seq_length=1, embedding_dim)

        if in_main.size(2) != self.embedding_dim:
            raise ValueError(
                f"Expected in_main embedding dimension to be {self.embedding_dim}, but got {in_main.size(2)}."
            )

        # Forward pass through the encoder
        output = in_main
        for mod in self.encoder.layers:
            output = mod(
                in_main=output,
                in_main_key_padding_mask=in_main_key_padding_mask,
                **kwargs,
            )
        # remove the 2nd dimension of the output again
        output = output.squeeze(1)

        # Add temperature if enabled
        # if self.learn_temperature:
        #     temperature = self.temperature_module()
        # elif self.temperature:
        #     temperature = self.temperature
        # else:
        #     temperature = 1.0

        return output

    def _get_sentence_embedding_dimension(self) -> int:
        return self.embedding_dim

    def _get_config_dict(self) -> dict:
        return {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "use_self_attention": self.use_self_attention,
            "use_cross_attention": self.use_cross_attention,
            "activation": self.activation,
            "dropout": self.dropout,
            "learn_temperature": self.learn_temperature,
        }

    def save(self, output_path: str, safe_serialization: bool = True) -> None:
        """Saves the model configuration and state."""
        os.makedirs(output_path, exist_ok=True)

        # Save config
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self._get_config_dict(), fOut, indent=2)

        # Save model state
        if safe_serialization:
            save_safetensors_model(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def load(input_path: str):
        """Loads the model from the given path."""
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        model = OmicsEncoder(**config)

        if os.path.exists(os.path.join(input_path, "model.safetensors")):
            load_safetensors_model(model, os.path.join(input_path, "model.safetensors"))
        else:
            model.load_state_dict(
                torch.load(os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"))
            )
        return model


class CustomTransformerEncoderLayer(nn.Module):
    """CustomTransformerEncoderLayer can act as an MLP-only layer, self-attention layer, or cross-attention layer."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        use_self_attention: bool = False,
        use_cross_attention: bool = False,
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        """
        Initializes the CustomTransformerEncoderLayer.

        Parameters
        ----------
        embedding_dim
            Input and output embedding dimension.
        hidden_dim
            Dimension of the feedforward network (MLP).
        num_heads
            Number of attention heads (used if attention is enabled).
        use_self_attention
            Whether to include self-attention.
        use_cross_attention
            Whether to include cross-attention.
        activation
            Activation function for the feedforward network. Can be "relu" or a callable.
        dropout
            Dropout probability.
        """
        super().__init__()

        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        self.embedding_dim = embedding_dim

        # Self-Attention Layer
        if use_self_attention:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        # Cross-Attention Layer
        if use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        # Feedforward network (MLP)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.activation = F.relu if activation == "relu" else activation
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)

        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        if use_self_attention or use_cross_attention:
            self.norm3 = nn.LayerNorm(embedding_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        in_main: torch.Tensor,
        in_main_key_padding_mask: torch.ByteTensor | torch.BoolTensor | None = None,
        in_cross: torch.Tensor | None = None,
        in_cross_key_padding_mask: torch.ByteTensor | torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder layer. Will be called by the MMContextEncoder and passed through several times, depending on the number of layers set there.

        Parameters
        ----------
        in_main
            Input tensor of shape (batch_size, seq_length, embedding_dim).
        in_main_key_padding_mask
            Mask for in_main keys per batch
        in_cross
            Input tensor for cross-attention (batch_size, seq_length, embedding_dim).
        in_cross_key_padding_mask
            Mask for in_cross keys per batch

        Returns
        -------
        Output tensor of shape (batch_size, seq_length, embedding_dim).
        """
        x = in_main

        # Self-Attention
        if self.use_self_attention:
            attn_output, _ = self.self_attn(x, x, x, key_padding_mask=in_main_key_padding_mask)
            x = x + self.dropout(attn_output)
            x = self.norm1(x)

        # Cross-Attention
        if self.use_cross_attention:
            if in_cross is None:
                raise ValueError("in_cross embeddings are required for cross-attention.")
            attn_output, _ = self.cross_attn(x, in_cross, in_cross, key_padding_mask=in_cross_key_padding_mask)
            x = x + self.dropout(attn_output)
            x = self.norm2(x)

        # Feedforward Network (MLP)
        ff_output = self.linear2(self.activation(self.linear1(x)))
        x = x + self.dropout(ff_output)
        if self.use_self_attention or self.use_cross_attention:
            x = self.norm3(x)
        else:
            x = self.norm1(x)  # Use norm1 if no attention is applied

        return x


class LearnableTemperature(nn.Module):
    """LearnableTemperature is a module that learns a temperature parameter for the InfoNCE loss function."""

    def __init__(self, initial_temperature=0.07):
        super().__init__()
        # Initialize temperature as a learnable parameter
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))

    def forward(self):
        """Forward pass returns the temperature as a positive value."""
        # Ensure the temperature is always positive
        return torch.exp(self.temperature)
