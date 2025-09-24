import logging
import os

import torch
import torch.nn as nn
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model

logger = logging.getLogger(__name__)


class AdapterModule(nn.Module):
    """
    Light-weight projection head

    used for mapping text/omics encodings into a shared space.
    The module is a simple feed-forward network with optional
    batch normalization and ReLU activation.

    Modes
    -----
    1. **Identity** – `hidden_dim in (None, 0)` *and*
       `output_dim == input_dim` *and* `force_identity=True`
       → acts as `nn.Identity`.
    2. **Linear → BN** – `hidden_dim in (None, 0)`
       → single `Linear + BatchNorm1d`.
    3. **Linear → ReLU → Linear → BN** – default MLP.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the incoming features.
    hidden_dim : int | None, optional
        Size of the hidden layer.  If *None* or *0*, the hidden layer is
        skipped altogether.
    output_dim : int | None, optional
        Size of the final output.  If *None* or *0*, falls back to
        `input_dim`.
    force_identity : bool, optional
        If True, forces the use of identity layer when `hidden_dim is None`
        and `output_dim == input_dim`. If False (default), builds normal
        layers even when dimensions match.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = 512,
        output_dim: int | None = 2048,
        force_identity: bool = False,
    ) -> None:
        super().__init__()

        # ------- normalise "None / 0" -----------------------------------
        hidden_dim = None if not hidden_dim else hidden_dim
        output_dim = input_dim if not output_dim else output_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.force_identity = force_identity

        # ------- build the sub-network ----------------------------------
        if hidden_dim is None and output_dim == input_dim and force_identity:
            # Pure identity (only when explicitly forced)
            self.net = nn.Identity()
            self.is_identity = True
        elif hidden_dim is None:
            # Single linear projection
            self.net = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
            )
            self.is_identity = False
        else:
            # Two-layer MLP
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim),
            )
            self.is_identity = False

    # -------------------------------- forward -----------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)

    # -------------------------------- config ------------------------------
    def _get_config_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "force_identity": self.force_identity,
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
