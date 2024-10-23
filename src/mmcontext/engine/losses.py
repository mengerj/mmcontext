# engine/losses.py

import abc
import logging

import torch
import torch.nn.functional as F


class LossManager:
    """Manages multiple loss functions and their weights."""

    def __init__(self, logger=None):
        self.loss_functions = []
        self.logger = logger or logging.getLogger(__name__)

    def add_loss(self, loss_function, weight=1.0):
        """
        Adds a loss function to the manager.

        Args:
            loss_function (LossFunction): An instance of a loss function.
            weight (float): Weight for the loss function.
        """
        self.logger.info("Adding loss function: %s with weight %.2f", loss_function.description, weight)
        self.loss_functions.append((loss_function, weight))

    def compute_total_loss(self, outputs, targets):
        """
        Computes the total loss by combining individual losses.

        Args:
            outputs (dict): Dictionary containing model outputs and other necessary data.
            targets (dict): Dictionary containing target values.

        Returns
        -------
            Tensor: The total loss.
        """
        total_loss = 0.0
        for loss_function, weight in self.loss_functions:
            loss = loss_function.compute_loss(outputs, targets)
            total_loss += weight * loss
        return total_loss


class LossFunction(metaclass=abc.ABCMeta):
    """Abstract base class for loss functions."""

    @abc.abstractmethod
    def compute_loss(self, outputs, targets):
        """
        Computes the loss given outputs and targets.

        Args:
            outputs (Tensor): The outputs from the model.
            targets (Tensor): The target values.

        Returns
        -------
            Tensor: The computed loss.
        """
        pass


class ContrastiveLoss(LossFunction):
    """
    Computes contrastive loss using similarity matrices.

    This class computes similarity matrices for both target and current embeddings based on specified modes.
    """

    def __init__(
        self,
        target_mode="context_context",
        current_mode="data_context",
        similarity_metric="cosine",
        logger=None,
        temperature=0.07,
        data_emb_key="data_embedding",
        context_emb_key="context_embedding",
    ):
        """
        Initializes the ContrastiveLoss.

        Args:
            target_mode (str): Mode for computing the target similarity matrix ('data_data', 'context_context', 'data_context'). Defaults to "context_context".
            current_mode (str): Mode for computing the current similarity matrix. Defaults to "data_context".
            similarity_metric (str): Similarity metric to use ('cosine', 'euclidean').
            temperature (float): Temperature parameter for scaling the similarity matrix (default: 0.07). Only used if target_mode = 'infoNCE', to mimic typical contrastive loss.
            data_emb_key (str): The key for accessing data embeddings (default: 'data_embedding').
            context_emb_key (str): The key for accessing context embeddings (default: 'context_embedding').
        """
        self.target_mode = target_mode
        self.current_mode = current_mode
        self.similarity_metric = similarity_metric
        self.temperature = temperature
        self.data_emb_key = data_emb_key
        self.context_emb_key = context_emb_key
        self.logger = logger or logging.getLogger(__name__)
        self.description = f"ContrastiveLoss(target_mode={target_mode}, current_mode={current_mode}, similarity_metric={similarity_metric})"

    def compute_loss(self, outputs, targets):
        """
        Computes the contrastive loss between the current and target similarity matrices.

        Args:
            outputs (dict): Dictionary containing modified embeddings, e.g., 'data_embedding', 'context_embedding'.
            targets (dict): Dictionary containing original embeddings, e.g., 'data_embedding', 'context_embedding'.

        Returns
        -------
            Tensor: The computed contrastive loss.
        """
        if "context" in self.current_mode and self.context_emb_key not in outputs:
            self.logger.error(
                f"{self.context_emb_key} not found in outputs but current_mode contains context. Either change current mode or provide context embeddings in outputs."
            )
            raise KeyError(
                f"{self.context_emb_key} not found in outputs but current_mode contains context. Either change current mode or provide context embeddings in outputs."
            )
        if self.target_mode == "infoNCE":
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

    def get_similarity_matrix(self, embeddings_dict, mode):
        """
        Computes the similarity matrix based on the specified mode.

        Args:
            embeddings_dict (dict): Dictionary containing embeddings, e.g., 'data_embedding', 'context_embedding'.
            mode (str): Mode for computing the similarity matrix ('data_data', 'context_context', 'data_context').

        Returns
        -------
            Tensor: Similarity matrix.
        """
        if mode == "context_data":
            mode = "data_context"
        if mode == "data_data":
            embeddings_a = embeddings_dict[self.data_emb_key]
            embeddings_b = embeddings_dict[self.data_emb_key]
        elif mode == "context_context":
            embeddings_a = embeddings_dict[self.context_emb_key]
            embeddings_b = embeddings_dict[self.context_emb_key]
        elif mode == "data_context":
            embeddings_a = embeddings_dict[self.data_emb_key]
            embeddings_b = embeddings_dict[self.context_emb_key]
        elif mode == "infoNCE":
            embeddings_a = embeddings_dict[self.data_emb_key]
            embeddings_b = embeddings_dict[self.context_emb_key]
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

    def compute_similarity(self, embeddings_a, embeddings_b):
        """
        Computes the similarity matrix between two sets of embeddings.

        Args:
            embeddings_a (Tensor): Embeddings of shape (N, embedding_dim).
            embeddings_b (Tensor): Embeddings of shape (M, embedding_dim).

        Returns
        -------
            Tensor: Similarity matrix of shape (N, M).
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

    def __init__(
        self, reduction="mean", logger=None, data_emb_key="data_embedding", context_emb_key="context_embedding"
    ):
        """
        Initializes the ReconstructionLoss.

        Args:
            reduction (str): Specifies the reduction to apply to the output ('mean', 'sum').
            data_emb_key (str): The key for accessing data embeddings (default: 'data_embedding').
            context_emb_key (str): The key for accessing context embeddings (default: 'context_embedding').
        """
        self.reduction = reduction
        self.data_emb_key = data_emb_key
        self.context_emb_key = context_emb_key
        self.logger = logger or logging.getLogger(__name__)
        self.description = f"ReconstructionLoss(reduction={reduction})"

    def compute_loss(self, outputs, targets):
        """
        Computes the reconstruction loss.

        Args:
            outputs (dict): Should contain 'data_embedding' or 'context_embedding' (modified embeddings).
            targets (dict): Should contain 'data_embedding' or 'context_embedding' (original embeddings).

        Returns
        -------
            Tensor: The reconstruction loss.
        """
        output_embeddings = outputs[self.data_emb_key]
        target_embeddings = targets[self.data_emb_key]

        loss = F.mse_loss(output_embeddings, target_embeddings, reduction=self.reduction)
        return loss
