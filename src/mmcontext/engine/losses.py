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
        logger: logging.Logger | None = None,
    ):
        self.loss_functions = []
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

    def compute_total_loss(self, outputs: dict, targets: dict, max_batches=64):
        """
        Computes the total loss by combining individual losses.

        Parameters
        ----------
        outputs
            Dictionary containing model outputs and other necessary data, such as the original 'in_cross' embeddings and the sample ids.
        targets
            Dictionary containing target values, same keys as outputs, but before any modifications.
        max_samples
            Maximum number of samples to use for computing the loss. If None, all samples are used.

        Returns
        -------
        The total loss.
        """
        outputs, targets = self.create_subset(outputs, targets, num_batches=max_batches)
        total_loss = 0.0
        for loss_function, weight in self.loss_functions:
            try:
                loss = loss_function.compute_loss(outputs, targets)
            except KeyError as e:
                self.logger.error(f"KeyError while computing loss: {loss_function.description} - {e}")
                raise e
            # self.logger.debug("Loss: %s: %.4f", loss_function.description, loss*weight)
            total_loss += weight * loss
        return total_loss

    def create_subset(self, outputs, targets, num_batches=None):
        """Creates subsets of the matrices in outputs and targets using consistent sampling."""
        if num_batches is None:
            return outputs, targets
        seed = 42  # Fixed seed for reproducibility
        torch.manual_seed(seed)

        # Determine the total number of samples
        keys = list(outputs.keys())
        total_batches = outputs[keys[0]].shape[0]

        # Generate random indices
        indices = torch.randperm(total_batches)[:num_batches]

        # Subset the outputs and targets
        outputs_subset = {}
        targets_subset = {}

        for key, value in outputs.items():
            if not isinstance(value, torch.Tensor) or value.dim() < 2:
                outputs_subset[key] = value
            elif value.dim() == 2:
                outputs_subset[key] = value[indices]
            elif value.dim() == 3:
                outputs_subset[key] = value[indices, :, :]
            else:
                raise ValueError(f"Unsupported number of dimensions ({value.dim()}) for outputs[{key}]")

        for key, value in targets.items():
            if not isinstance(value, torch.Tensor) or value.dim() < 2:
                targets_subset[key] = value
            elif value.dim() == 2:
                targets_subset[key] = value[indices]
            elif value.dim() == 3:
                targets_subset[key] = value[indices, :, :]
            else:
                raise ValueError(f"Unsupported number of dimensions ({value.dim()}) for targets[{key}]")

        return outputs_subset, targets_subset

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
            elif loss_type == "zinb_loss":
                # Create a ZINBLoss instance with specified settings
                loss_fn = ZINBLoss(
                    eps=loss_cfg.get("eps"),
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
        data_key: str = "d_emb",
        context_key: str = "c_emb",
        similarity_metric: str = "cosine",
        logger: logging.Logger | None = None,
        temperature: float = 0.07,
    ):
        self.target_mode = target_mode
        self.current_mode = current_mode
        self.similarity_metric = similarity_metric
        self.temperature = temperature
        self.data_key = data_key
        self.context_key = context_key
        self.logger = logger or logging.getLogger(__name__)
        self.description = f"ContrastiveLoss(target_mode={target_mode}, current_mode={current_mode}, similarity_metric={similarity_metric})"

        if self.target_mode == "infoNCE":
            self.temperature_dependent = True

    def compute_loss(self, outputs: dict, targets: dict):
        """
        Computes the contrastive loss between the current and target similarity matrices.

        Parameters
        ----------
        outputs
            Dictionary containing modified embeddings, e.g., 'data_embedding', 'context_embedding'.
        targets
            Dictionary containing original embeddings, e.g., 'data_embedding', 'context_embedding'.

        Returns
        -------
        The computed contrastive loss.
        """
        if "context" in self.current_mode and self.context_key not in outputs:
            self.logger.error(
                f"{self.context_key} not found in outputs but current_mode contains context. Either change current mode or provide context embeddings in outputs."
            )
            raise KeyError(
                f"{self.context_key} not found in outputs but current_mode contains context. Either change current mode or provide context embeddings in outputs."
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

    def __init__(self, reduction="mean", logger=None, data_key: str = "d_emb"):
        """
        Initializes the ReconstructionLoss.

        Args:
            reduction (str): Specifies the reduction to apply to the output ('mean', 'sum').
        """
        self.reduction = reduction
        self.logger = logger or logging.getLogger(__name__)
        self.description = f"ReconstructionLoss(reduction={reduction}) on {data_key}"
        self.data_key = data_key

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
        output_embeddings = outputs[self.data_key]
        target_embeddings = targets[self.data_key]

        loss = F.mse_loss(output_embeddings, target_embeddings, reduction=self.reduction)
        return loss


class ZINBLoss(LossFunction):
    """Class to compute the Zero-Inflated Negative Binomial (ZINB) negative log-likelihood loss.

    The compute_loss function is used by the LossManger to compute the loss given the model outputs and target values.
    outputs are expected to contain the keys 'mu', 'theta', 'pi' which are the parameters of the ZINB distribution.
    targets are expected to contain the key 'raw_data' which are the target counts.

    Parameters
    ----------
    eps
        Small value to ensure numerical stability.
    """

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.description = "ZINBLoss"

    def compute_loss(self, outputs: dict, targets: dict):
        """
        Computes the ZINB negative log-likelihood loss.

        Parameters
        ----------
        outputs
            A dictionary containing the model outputs with keys 'mu', 'theta', 'pi'.
        targets
            A dictionary containing the target counts with key 'counts'.
        data_key
            Not used in this loss function.
        context_key
            Not used in this loss function.

        Returns
        -------
        The computed ZINB negative log-likelihood loss.
        """
        outputs = outputs["out_distribution"]
        if outputs is None:
            self.logger.error(
                "ZINB loss was chosen but no distribution outputs were found. Maybe forgot to specify the decoder?"
            )
            raise ValueError(
                "ZINB loss was chosen but no distribution outputs were found. Maybe forgot to specify the decoder?"
            )
        # Extract the necessary tensors from outputs and targets
        mu = outputs["mu"]
        theta = outputs["theta"]
        pi = outputs["pi"]
        x = targets["raw_data"]
        x = x.view(-1, x.size(-1))  # Flatten the raw data for the loss computation

        # Ensure tensors are float tensors
        x = x.float()
        mu = mu.float()
        theta = theta.float()
        pi = pi.float()

        # Compute the loss using the provided function
        loss = self.zinb_negative_log_likelihood(x, mu, theta, pi, eps=self.eps)

        return loss

    def nb_positive_log_prob(self, x, mu, theta, eps=1e-7):
        """Computes log probability of x under Negative Binomial distribution."""
        # Ensure numerical stability
        x = torch.clamp(x, min=0)
        mu = torch.clamp(mu, min=eps)
        theta = torch.clamp(theta, min=eps)

        # Compute log probabilities
        log_theta_mu_eps = torch.log(theta + mu + eps)
        res = (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )
        return res

    def zinb_negative_log_likelihood(self, x, mu, theta, pi, eps=1e-7):
        """Computes the negative log-likelihood of the Zero-Inflated Negative Binomial distribution."""
        # Ensure numerical stability
        theta = torch.clamp(theta, min=eps)
        mu = torch.clamp(mu, min=eps)
        pi = torch.clamp(pi, min=eps, max=1 - eps)
        x = torch.clamp(x, min=0)

        # Compute log probabilities
        nb_case = self.nb_positive_log_prob(x, mu, theta, eps)
        nb_zero = self.nb_positive_log_prob(torch.zeros_like(x), mu, theta, eps)

        # Log probability when x == 0
        log_prob_zero = torch.log(pi + (1 - pi) * torch.exp(nb_zero) + eps)

        # Log probability when x > 0
        log_prob_nb = torch.log(1 - pi + eps) + nb_case

        # Combine log probabilities
        log_prob = torch.where(x < eps, log_prob_zero, log_prob_nb)

        nll = -torch.sum(log_prob) / log_prob.size(0)
        return nll
