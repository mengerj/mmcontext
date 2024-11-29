# engine/models.py

import abc
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


def configure_models(cfg: DictConfig, decoder_out_dim: int):
    """Configures the model based on the configuration.

    Parameters
    ----------
    cfg
        Configuration object.
    output_dim
        Number of samples in a batch.

    Returns
    -------
    encoder
        Encoder model.
    decoder
        Decoder model.
    """
    cfg_e = cfg.encoder
    cfg_d = cfg.decoder
    if cfg_e.type == "mmcontext_encoder":
        encoder = MMContextEncoder(
            embedding_dim=cfg_e.latent_dim,  # this has to be the same dimension as the latent dimension of the aligner
            hidden_dim=cfg_e.hidden_dim,
            num_layers=cfg_e.num_layers,
            num_heads=cfg_e.num_heads,
            use_self_attention=cfg_e.use_self_attention,
            use_cross_attention=cfg_e.use_cross_attention,
            activation=cfg_e.activation,
            dropout=cfg_e.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.type}")
    if cfg_d.type == "zinb_decoder":
        decoder = ZINBDecoder(
            input_dim=cfg_e.latent_dim,
            hidden_dims=cfg_d.hidden_dims,
            output_dim=decoder_out_dim,
        )
    return encoder, decoder


class BaseModel(nn.Module, metaclass=abc.ABCMeta):
    """BaseModel defines common methods for all models."""

    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)

    @abc.abstractmethod
    def forward(self, *inputs):
        """Forward pass to be implemented by subclasses."""
        pass

    def save(self, file_path: str):
        """
        Saves model state dictionary to the specified file path.

        Parameters
        ----------
        file_path
            Path to save the model state dictionary.
        """
        self.logger.info(f"Saving model state dictionary to {file_path}")
        torch.save(self.state_dict(), file_path)

    def load(self, file_path: str):
        """
        Loads model state dictionary from the specified file path.

        Parameters
        ----------
        file_path
            Path to load the model state dictionary from.
        """
        self.logger.info(f"Loading model state dictionary from {file_path}")
        self.load_state_dict(torch.load(file_path, weights_only=True))


class MMContextEncoder(BaseModel):
    """MMContextEncoder builds an encoder model using torchs TransformerEncoder and custom encoder layers."""

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
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
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

        self.use_cross_attention = use_cross_attention
        self.temperature_module = LearnableTemperature()
        self.learn_temperature = (
            False  # Dont learn temperature by default. Will be set true in trainer if "infoNCE" loss is used.
        )
        self.temperature = None  # Can be set by downstream methods to set a fixed temperature for the loss function.
        self.logger.info(
            f"MMContextEncoder initialized with embedding_dim = {embedding_dim}, num_layers = {num_layers}, use_self_attention = {use_self_attention}, use_cross_attention = {use_cross_attention}."
        )

    def forward(
        self,
        in_main: torch.Tensor,
        in_main_key_padding_mask: torch.ByteTensor | torch.BoolTensor | None = None,
        in_cross: torch.Tensor | None = None,
        in_cross_key_padding_mask: torch.ByteTensor | torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for the MMContextEncoder.

        Parameters
        ----------
        in_main
            Source tensor of shape (batch_size, seq_length, embedding_dim). These inputs will be passed through the MLP and self-attention layers.
        in_main_key_padding_mask
            Mask for the source keys per batch.
            If provided, should be a ByteTensor or BoolTensor of shape (batch_size, seq_length) where True values indicate padding positions.
        in_cross
            Input tensor for cross-attention of shape (batch_size, seq_length, embedding_dim).
            Required when `use_cross_attention` is True.
        in_cross_key_padding_mask
            Mask for the in_cross keys per batch. If provided, should be a
            ByteTensor or BoolTensor of shape (batch_size, seq_length).

        Returns
        -------
        Output tensor of shape (batch_size, seq_length, embedding_dim).

        Raises
        ------
        ValueError
            If `use_cross_attention` is True and `in_cross` is None.

        Note
        ----------
        We implement a custom `forward` method to pass the `in_cross` embeddings to each encoder layer during the forward pass.
        This is necessary because the default `nn.TransformerEncoder` does not support passing additional arguments
        like `in_cross` to its layers. By overriding the `forward` method, we ensure that `in_cross` embeddings are
        provided to each layer when cross-attention is used, enabling the model to perform cross-attention operations.
        """
        if self.use_cross_attention and in_cross is None:
            self.logger.error("in_cross embeddings are required when using cross-attention.")
            raise ValueError("in_cross embeddings are required when using cross-attention.")

        # Add dimension checks
        if in_main.dim() != 3:
            self.logger.error(
                f"Expected in_main to have 3 dimensions (batch_size, seq_length, embedding_dim), but got {in_main.dim()} dimensions."
            )
            raise ValueError(
                f"Expected in_main to have 3 dimensions (batch_size, seq_length, embedding_dim), but got {in_main.dim()} dimensions."
            )
        if in_main.size(2) != self.encoder.layers[0].embedding_dim:
            self.logger.error(
                f"Expected in_main embedding dimension to be {self.encoder.layers[0].embedding_dim}, but got {in_main.size(2)}."
            )
            raise ValueError(
                f"Expected in_main embedding dimension to be {self.encoder.layers[0].embedding_dim}, but got {in_main.size(2)}."
            )

        output = in_main
        for mod in self.encoder.layers:
            output = mod(
                in_main=output,
                in_main_key_padding_mask=in_main_key_padding_mask,
                in_cross=in_cross,
                in_cross_key_padding_mask=in_cross_key_padding_mask,
            )
        if self.learn_temperature:
            temperature = self.temperature_module()
        elif self.temperature:
            temperature = self.temperature
        else:
            temperature = 1.0  # Default temperature
        return output, temperature


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


class ZINBDecoder(BaseModel):
    """ZINBDecoder models the parameters of a Zero-Inflated Negative Binomial distribution, for each gene and cell.

    Parameters
    ----------
    input_dim
        Dimensionality of the input (latent space)
    hidden_dims
        List of hidden layer sizes
    output_dim
        Dimensionality of the output (number of genes)
    """

    def __init__(self, input_dim: int, hidden_dims: int, output_dim: int):
        print("ZINBDecoder")
        super().__init__()
        self.logger.info(
            f"ZINBDecoder initialized with input_dim = {input_dim}, hidden_dims = {hidden_dims}, output_dim = {output_dim}."
        )
        # Build the decoder network
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.decoder = nn.Sequential(*layers)

        # Output layers for ZINB parameters
        self.fc_mu = nn.Linear(prev_dim, output_dim)  # Mean parameter μ
        self.fc_theta = nn.Linear(prev_dim, output_dim)  # Dispersion parameter θ
        self.fc_pi = nn.Linear(prev_dim, output_dim)  # Dropout parameter π

    def forward(self, x):
        """
        Forward pass through the decoder.

        Parameters
        ----------
        - x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns
        -------
        - mu (torch.Tensor): Mean of Negative Binomial distribution
        - theta (torch.Tensor): Dispersion parameter
        - pi (torch.Tensor): Probability of zero inflation
        """
        h = self.decoder(x)
        # Ensure parameters are in valid ranges
        mu = torch.nn.functional.softplus(self.fc_mu(h))  # μ > 0
        theta = torch.nn.functional.softplus(self.fc_theta(h))  # θ > 0
        pi = torch.sigmoid(self.fc_pi(h))  # 0 < π < 1

        return {"mu": mu, "theta": theta, "pi": pi}


class PlaceholderModel(BaseModel):
    """A placeholder model that inherits from BaseModel."""

    def __init__(self):
        super().__init__()
        self.logger.info("Placeholder model initialized.")

    def forward(self, *inputs):
        """Forward pass implementation doing nothing but logging."""
        # Assuming input is a tensor or batch of tensors; returns them unchanged
        return inputs if len(inputs) > 1 else inputs[0]

    def save(self, file_path: str):
        """Overriding save to simulate saving without actual disk operation."""
        self.logger.info(f"Placeholder model saving of model to {file_path}")

    def load(self, file_path: str):
        """Overriding load to simulate loading without actual disk operation."""
        self.logger.info(f"Placeholder model loading of model from {file_path}")
