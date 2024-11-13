# engine/losses.py

import abc
import logging

import torch
import torch.nn.functional as F
from omegaconf import DictConfig


class LossFunction(metaclass=abc.ABCMeta):
    """Abstract base class for loss functions."""

    @abc.abstractmethod
    def compute_loss(self, outputs: dict, targets: dict, data_key: str, context_key: str):
        """
        Computes the loss given outputs and targets.

        Parameters
        ----------
        outputs
            The outputs from the model.
        targets
            The target values.
        data_key
            The key for accessing data embeddings.
        context_key
            The key for accessing context embeddings.

        Returns
        -------
        The computed loss.
        """
        pass


class LossManager:
    """Manages multiple loss functions and their weights.

    Has the ability to compute the total loss by combining individual losses.

    Parameters
    ----------
    data_key
        The key for accessing data embeddings.
    context_key
        The key for accessing context embeddings.
    """

    def __init__(
        self,
        data_key: str = "data_embedding",
        context_key: str = "context_embedding",
        logger: logging.Logger | None = None,
    ):
        self.loss_functions = []
        self.data_key = data_key
        self.context_key = context_key
        self.logger = logger or logging.getLogger(__name__)

    def add_loss(self, loss_function: LossFunction, weight: float = 1.0):
        """
        Adds a loss function to the manager.

        Parameters
        ----------
        loss_function
            An instance of a loss function.
        weight
            Weight for the loss function.
        """
        self.logger.info("Adding loss function: %s with weight %.2f", loss_function.description, weight)
        self.loss_functions.append((loss_function, weight))

    def compute_total_loss(self, outputs: dict, targets: dict):
        """
        Computes the total loss by combining individual losses.

        Parameters
        ----------
        outputs
            Dictionary containing model outputs and other necessary data, such as the original 'in_cross' embeddings and the sample ids.
        targets
            Dictionary containing target values, same keys as outputs, but before any modifications.

        Returns
        -------
        The total loss.
        """
        total_loss = 0.0
        for loss_function, weight in self.loss_functions:
            loss = loss_function.compute_loss(outputs, targets, self.data_key, self.context_key)
            total_loss += weight * loss
        return total_loss

    def configure_losses(self, cfg: DictConfig):
        """Configures and adds loss functions based on the provided configuration."""
        losses_config = cfg.get("losses", {})
        for loss_cfg_key in losses_config.keys():
            loss_cfg = losses_config.get(loss_cfg_key)
            if not loss_cfg.get("use"):
                continue  # Skip if 'use' is False

            loss_type = loss_cfg.get("type")
            weight = loss_cfg.get("weight", 1.0)

            if loss_type == "contrastive_loss":
                # Create a ContrastiveLoss instance with specified settings
                loss_fn = ContrastiveLoss(
                    target_mode=loss_cfg.get("target_mode"),
                    current_mode=loss_cfg.get("current_mode"),
                    similarity_metric=loss_cfg.get("similarity_metric"),
                )
            elif loss_type == "reconstruction_loss":
                # Create a ReconstructionLoss instance with specified settings
                loss_fn = ReconstructionLoss(
                    reduction=loss_cfg.get("reduction"),
                )
            else:
                self.logger.error(f"Unknown loss type '{loss_type}'")
                continue  # Skip unknown loss types

            # Add the loss function to the manager
            self.add_loss(loss_fn, weight=weight)


class ContrastiveLoss(LossFunction):
    """
    Computes contrastive loss using similarity matrices.

    This class computes similarity matrices for both target and current embeddings based on specified modes.

    Parameters
    ----------
    target_mode
        Mode for computing the target similarity matrix ('data_data', 'context_context', 'data_context'). Defaults to "context_context".
    current_mode
        Mode for computing the current similarity matrix. Defaults to "data_context".
    similarity_metric
        Similarity metric to use. 'cosine' or 'euclidean'.
    temperature
        Temperature parameter for scaling the similarity matrix. Only used if target_mode = 'infoNCE', to mimic typical contrastive loss.
    """

    def __init__(
        self,
        target_mode: str = "context_context",
        current_mode: str = "data_context",
        similarity_metric: str = "cosine",
        logger: logging.Logger | None = None,
        temperature: float = 0.07,
    ):
        self.target_mode = target_mode
        self.current_mode = current_mode
        self.similarity_metric = similarity_metric
        self.temperature = temperature
        self.logger = logger or logging.getLogger(__name__)
        self.description = f"ContrastiveLoss(target_mode={target_mode}, current_mode={current_mode}, similarity_metric={similarity_metric})"

    def compute_loss(
        self, outputs: dict, targets: dict, data_key: str = "data_embedding", context_key: str = "context_embedding"
    ):
        """
        Computes the contrastive loss between the current and target similarity matrices.

        Parameters
        ----------
        outputs
            Dictionary containing modified embeddings, e.g., 'data_embedding', 'context_embedding'.
        targets
            Dictionary containing original embeddings, e.g., 'data_embedding', 'context_embedding'.
        data_key
            The key for accessing data embeddings.
        context_key
            The key for accessing context embeddings.

        Returns
        -------
        The computed contrastive loss.
        """
        self.data_key = data_key
        self.context_key = context_key
        if "context" in self.current_mode and context_key not in outputs:
            self.logger.error(
                f"{context_key} not found in outputs but current_mode contains context. Either change current mode or provide context embeddings in outputs."
            )
            raise KeyError(
                f"{context_key} not found in outputs but current_mode contains context. Either change current mode or provide context embeddings in outputs."
            )
        if self.target_mode == "infoNCE":
            self.temperature = outputs["temperature"]
            # Enforce current_mode to be 'data_context'
            if self.current_mode not in ["data_context"]:
                self.current_mode = "data_context"
                self.logger.warning(
                    "Mode to calculate current similarity matrix was forced to be between data and context embeddings. When target_mode is 'infoNCE', current_mode must be 'data_context'."
                )

        # Compute target similarity matrix
        target_sim_matrix = self.get_similarity_matrix(targets, self.target_mode)

        # Compute current similarity matrix
        current_sim_matrix = self.get_similarity_matrix(outputs, self.current_mode)

        # Compute loss between current and target similarity matrices
        if self.target_mode == "infoNCE":
            loss = F.cross_entropy(current_sim_matrix, target_sim_matrix)
        else:
            loss = F.mse_loss(current_sim_matrix, target_sim_matrix)

        return loss

    def get_similarity_matrix(
        self, embeddings_dict: dict, mode: str, data_key: str | None = None, context_key: str | None = None
    ) -> torch.Tensor:
        """
        Computes the similarity matrix based on the specified mode.

        Parameters
        ----------
        embeddings_dict
            Dictionary containing embeddings, e.g., 'data_embedding', 'context_embedding'.
        mode
            Mode for computing the similarity matrix ('data_data', 'context_context', 'data_context').

        Returns
        -------
        Similarity matrix.
        """
        if data_key:
            if data_key not in embeddings_dict.keys() and "data" in mode:
                raise KeyError(
                    f"{data_key} not found in embeddings_dict. This should only appear if you tried to call the get_similarity_matrix method directly and provided a wrong key."
                )
            else:
                self.logger.info(f"Using data key from get_similarity_matrix input: {data_key}")
                self.data_key = data_key
        if context_key:
            if context_key not in embeddings_dict.keys() and "context" in mode:
                raise KeyError(
                    f"{context_key} not found in embeddings_dict. This should only appear if you tried to call the get_similarity_matrix method directly and provided a wrong key."
                )
            else:
                self.logger.info(f"Using context key from get_similarity_matrix input: {context_key}")
                self.context_key = context_key
        if mode == "context_data":
            mode = "data_context"
        if mode == "data_data":
            embeddings_a = embeddings_dict[self.data_key]
            embeddings_b = embeddings_dict[self.data_key]
        elif mode == "context_context":
            embeddings_a = embeddings_dict[self.context_key]
            embeddings_b = embeddings_dict[self.context_key]
        elif mode == "data_context":
            embeddings_a = embeddings_dict[self.data_key]
            embeddings_b = embeddings_dict[self.context_key]
        elif mode == "infoNCE":
            embeddings_a = embeddings_dict[self.data_key]
            embeddings_b = embeddings_dict[self.context_key]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # Flatten the embeddings over batch and sequence dimensions
        embeddings_a = embeddings_a.view(-1, embeddings_a.shape[-1])  # Shape: (N, embedding_dim)
        embeddings_b = embeddings_b.view(-1, embeddings_b.shape[-1])  # Shape: (M, embedding_dim)

        # Compute similarity matrix
        sim_matrix = self.compute_similarity(embeddings_a, embeddings_b)

        if self.target_mode == "infoNCE":
            sim_matrix = sim_matrix / self.temperature
            if mode == "data_context":
                return sim_matrix
            if mode == "infoNCE":
                sim_matrix = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)

        return sim_matrix

    def compute_similarity(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor):
        """
        Computes the similarity matrix between two sets of embeddings.

        Parameters
        ----------
        embeddings_a
            Embeddings of shape (N, embedding_dim).
        embeddings_b
            Embeddings of shape (M, embedding_dim).

        Returns
        -------
        Similarity matrix of shape (N, M).
        """
        if self.similarity_metric == "cosine":
            # Normalize embeddings
            embeddings_a_norm = F.normalize(embeddings_a, p=2, dim=1)
            embeddings_b_norm = F.normalize(embeddings_b, p=2, dim=1)
            # Compute cosine similarity
            sim_matrix = torch.matmul(embeddings_a_norm, embeddings_b_norm.t())  # Shape: (N, M)

        elif self.similarity_metric == "euclidean":
            # Compute negative Euclidean distance
            sim_matrix = -torch.cdist(embeddings_a, embeddings_b, p=2)  # Shape: (N, M)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

        return sim_matrix


class ReconstructionLoss(LossFunction):
    """Computes reconstruction loss between outputs and targets."""

    def __init__(self, reduction="mean", logger=None):
        """
        Initializes the ReconstructionLoss.

        Args:
            reduction (str): Specifies the reduction to apply to the output ('mean', 'sum').
        """
        self.reduction = reduction
        self.logger = logger or logging.getLogger(__name__)
        self.description = f"ReconstructionLoss(reduction={reduction})"

    def compute_loss(self, outputs, targets, data_key, context_key):
        """
        Computes the reconstruction loss.

        Args:
            outputs (dict): Should contain 'data_embedding' or 'context_embedding' (modified embeddings).
            targets (dict): Should contain 'data_embedding' or 'context_embedding' (original embeddings).

        Returns
        -------
            Tensor: The reconstruction loss.
        """
        output_embeddings = outputs[data_key]
        target_embeddings = targets[data_key]

        loss = F.mse_loss(output_embeddings, target_embeddings, reduction=self.reduction)
        return loss
