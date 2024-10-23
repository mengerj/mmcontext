# engine/models.py

import abc
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module, metaclass=abc.ABCMeta):
    """BaseModel defines common methods for all models."""

    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)

    @abc.abstractmethod
    def forward(self, *inputs):
        """Forward pass to be implemented by subclasses."""
        pass

    def save(self, file_path):
        """
        Saves model state dictionary to the specified file path.

        Args:
            file_path (str): Path to save the model state dictionary.
        """
        self.logger.info(f"Saving model state dictionary to {file_path}")
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        """
        Loads model state dictionary from the specified file path.

        Args:
            file_path (str): Path to load the model state dictionary from.
        """
        self.logger.info(f"Loading model state dictionary from {file_path}")
        self.load_state_dict(torch.load(file_path, weights_only=True))


class MMContextEncoder(BaseModel):
    """MMContextEncoder builds an encoder model using torchs TransformerEncoder and custom encoder layers."""

    def __init__(
        self,
        embedding_dim,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_self_attention=False,
        use_cross_attention=False,
        activation="relu",
        dropout=0.1,
    ):
        super().__init__()

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
        self.logger.info(
            f"MMContextEncoder initialized with embedding_dim = {embedding_dim}, num_layers = {num_layers}, use_self_attention = {use_self_attention}, use_cross_attention = {use_cross_attention}."
        )

    def forward(self, in_main, in_main_key_padding_mask=None, in_cross=None, in_cross_key_padding_mask=None):
        """
        Forward pass for the MMContextEncoder.

        Args:
            in_main (Tensor): Source tensor (data embeddings) of shape (batch_size, seq_length, embedding_dim).
            in_main_key_padding_mask (Tensor, optional): Mask for the source keys per batch. If provided, should be a
                ByteTensor or BoolTensor of shape (batch_size, seq_length) where True values indicate padding positions.
                Default is None.
            in_cross (Tensor, optional): Input tensor for cross-attention of shape (batch_size, seq_length, embedding_dim).
                Required when `use_cross_attention` is True. Default is None.
            in_cross_key_padding_mask (Tensor, optional): Mask for the in_cross keys per batch. If provided, should be a
                ByteTensor or BoolTensor of shape (batch_size, seq_length). Default is None.

        Returns
        -------
            Tensor: Output tensor of shape (batch_size, seq_length, embedding_dim).

        Raises
        ------
            ValueError: If `use_cross_attention` is True and `in_cross` is None.

        Note:
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
        return output


class CustomTransformerEncoderLayer(nn.Module):
    """CustomTransformerEncoderLayer can act as an MLP-only layer, self-attention layer, or cross-attention layer."""

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        num_heads=4,
        use_self_attention=False,
        use_cross_attention=False,
        activation="relu",
        dropout=0.1,
    ):
        """
        Initializes the CustomTransformerEncoderLayer.

        Args:
            embedding_dim (int): Input and output embedding dimension.
            hidden_dim (int): Dimension of the feedforward network (MLP).
            num_heads (int): Number of attention heads (used if attention is enabled).
            use_self_attention (bool): Whether to include self-attention.
            use_cross_attention (bool): Whether to include cross-attention.
            activation (str or callable): Activation function for the feedforward network.
            dropout (float): Dropout probability.
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

    def forward(self, in_main, in_main_key_padding_mask=None, in_cross=None, in_cross_key_padding_mask=None):
        """
        Forward pass for the encoder layer.

        Args:
            in_main (Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).
            in_main_key_padding_mask (Tensor, optional): Mask for in_main keys per batch (optional).
            in_cross (Tensor, optional): Input tensor for cross-attention (batch_size, seq_length, embedding_dim).
            in_cross_key_padding_mask (Tensor, optional): Mask for in_cross keys per batch (optional).

        Returns
        -------
            Tensor: Output tensor of shape (batch_size, seq_length, embedding_dim).
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
