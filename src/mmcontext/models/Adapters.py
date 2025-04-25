import logging
import os

import torch
import torch.nn as nn
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model

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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

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
