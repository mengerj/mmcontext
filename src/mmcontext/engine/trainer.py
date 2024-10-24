# engine/trainer.py

import logging

import torch
from torch.utils.data import DataLoader

from mmcontext.engine.losses import ContrastiveLoss, LossManager


class Trainer:
    """Trainer class to handle the training and evaluation of models."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_manager: LossManager | None = None,
        device: torch.device | None = None,
        logger: logging.Logger | None = None,
        input_embeddings: dict | None = None,
        data_emb_key: str = "data_embedding",
        context_emb_key: str = "context_embedding",
        sample_id_key: str = "sample_id",
    ):
        """
        Initializes the Trainer.

        Parameters
        ----------
        model
            The neural network model to train.
        loss_manager
            Manages multiple loss functions. Not needed for inference.
        optimizer
            Optimizer for updating model parameters.
        device
            Device to run the model on 'cuda' or 'cpu'. Defaults to CUDA if available.
        logger
            Logger for tracking training progress. If None, a default logger is created.
        input_embeddings
            Dict of embeddings to pass to the model.
            Should be have keys 'main' and 'cross'. Values should correspond to names of embeddings in the batch.
            Defaults to `{'main': 'data_embedding', 'cross': 'context_embedding'}`. Only the embeddings from main are modified.
            The embeddings corresponding cross are used for cross-attention if the model was configured with use_cross_attention = True.
        data_emb_key
            Key for data embeddings in the batch. Defaults to 'data_embedding'.
        context_emb_key
            Key for context embeddings in the batch. Defaults to 'context_embedding'.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_manager = loss_manager
        self.optimizer = optimizer
        self.data_emb_key = data_emb_key
        self.context_emb_key = context_emb_key
        self.sample_id_key = sample_id_key
        self.logger = logger or logging.getLogger(__name__)

        # Set default input_embeddings if not provided
        if input_embeddings is None:
            self.logger.info(
                "input_embeddings not provided. Using default values. Data embedding is main and context embedding is cross."
            )
            self.input_embeddings = {"main": data_emb_key, "cross": context_emb_key}
        else:
            self.input_embeddings = {"main": None, "cross": None}
            # add the provided input embeddings to the existing dict to ensure the key for both emb is always present, but the values can be None
            for key, value in input_embeddings.items():
                self.input_embeddings[key] = value
        if self.loss_manager is not None:
            self._validate_input_embeddings()

    def _validate_input_embeddings(self):
        for loss_fn, _ in self.loss_manager.loss_functions:
            if isinstance(loss_fn, ContrastiveLoss):
                if loss_fn.current_mode == "context_context":
                    # if 'context_embedding' are not main argument in input embeddings, raise an error
                    if self.input_embeddings["main"] != self.context_emb_key:
                        self.logger.error(
                            "context_embedding should be the main input_embeddings for context_context (current_mode) contrastive loss."
                        )
                        raise ValueError(
                            "context_embedding should be the main input_embeddings for context_context (current_mode) contrastive loss."
                        )
                if loss_fn.current_mode == "data_data":
                    # if 'data_embedding' are not in first position of input embeddings, raise an error
                    if self.input_embeddings["main"] != self.data_emb_key:
                        self.logger.error(
                            "data_embedding should be the main input_embeddings for data_data (current_mode) contrastive loss."
                        )
                        raise ValueError(
                            "data_embedding should be the first element of input_embeddings for data_data (current_mode) contrastive loss."
                        )
                if loss_fn.current_mode == "data_context":
                    # make sure that both main and cross embeddings are provided and not None
                    if self.input_embeddings["main"] is None or self.input_embeddings["cross"] is None:
                        self.logger.error(
                            "in_cross embeddings are required when using data_context (current_mode) contrastive loss. Eventhough cross attention might not be used, the embeddings are required for loss computation."
                        )
                        raise ValueError(
                            "in_cross embeddings are required when using data_context (current_mode) contrastive loss. Eventhough cross attention might not be used, the embeddings are required for loss computation."
                        )

    def train_epoch(self, data_loader: DataLoader) -> float:
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader
            DataLoader for training data.

        Returns
        -------
            Average training loss for the epoch. (dtype `float`)
        """
        # check if loss manager contains any loss functions
        if len(self.loss_manager.loss_functions) == 0:
            raise ValueError("Loss Manager must contain at least one loss function.")
        self.model.train()
        total_loss = 0.0
        num_batches = len(data_loader)

        for batch_idx, batch in enumerate(data_loader):
            # sample_ids = batch["sample_id"]  # Shape: (batch_size, seq_length)  # Not used in loss computation
            in_main = batch[self.input_embeddings["main"]].to(
                self.device
            )  # Shape: (batch_size, seq_length, embedding_dim)
            if len(self.input_embeddings) == 2:
                in_cross = batch[self.input_embeddings["cross"]].to(
                    self.device
                )  # Shape: (batch_size, seq_length, embedding_dim)
            else:
                in_cross = None
            # Forward pass
            out = self.model(in_main=in_main, in_cross=in_cross)
            outputs = {}
            outputs[self.input_embeddings["main"]] = out
            outputs[self.input_embeddings["cross"]] = (
                in_cross  # Pass the original embeddings used for cross attention to allow for data_context contrastive loss
            )

            # Prepare targets
            targets = {self.data_emb_key: batch[self.data_emb_key], self.context_emb_key: batch[self.context_emb_key]}

            # Compute loss
            loss = self.loss_manager.compute_total_loss(outputs=outputs, targets=targets)
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                self.logger.info(f"Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}")

        average_loss = total_loss / num_batches
        self.logger.info(f"Training Epoch Complete. Average Loss: {average_loss:.4f}")
        return average_loss

    def evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluates the model on a validation set.

        Parameters
        ----------
        data_loader
            DataLoader for validation data.

        Returns
        -------
            Average validation loss. (dtype `float`)
        """
        # check if loss manager contains any loss functions
        if len(self.loss_manager.loss_functions) == 0:
            raise ValueError("Loss Manager must contain at least one loss function.")
        self.model.eval()
        total_loss = 0.0
        num_batches = len(data_loader)

        with torch.no_grad():
            for _, batch in enumerate(data_loader):
                # sample_ids = batch["sample_id"]  # Shape: (batch_size, seq_length)  # Not used
                in_main = batch[self.input_embeddings["main"]].to(
                    self.device
                )  # Shape: (batch_size, seq_length, embedding_dim)
                if len(self.input_embeddings) == 2:
                    in_cross = batch[self.input_embeddings["cross"]].to(
                        self.device
                    )  # Shape: (batch_size, seq_length, embedding_dim)
                else:
                    in_cross = None
                # Forward pass
                out = self.model(in_main=in_main, in_cross=in_cross)
                outputs = {}
                outputs[self.input_embeddings["main"]] = out
                outputs[self.input_embeddings["cross"]] = (
                    in_cross  # Pass the original embeddings used for cross attention to allow for data_context contrastive loss
                )

                # Prepare targets
                targets = {
                    self.data_emb_key: batch[self.data_emb_key],
                    self.context_emb_key: batch[self.context_emb_key],
                }

                # Compute loss
                loss = self.loss_manager.compute_total_loss(outputs=outputs, targets=targets)

                total_loss += loss.item()

        average_loss = total_loss / num_batches
        self.logger.info(f"Validation Complete. Average Loss: {average_loss:.4f}")
        return average_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
        save_path: str | None = None,
    ):
        """
        Trains the model for a specified number of epochs and optionally evaluates on a validation set.

        Parameters
        ----------
        train_loader
            DataLoader for training data.
        val_loader
            DataLoader for validation data. Defaults to None.
        epochs
            Number of epochs to train. Defaults to 10.
        save_path
            Path to save the model checkpoints. Defaults to None.

        Raises
        ------
        ValueError
            If Loss Manager does not contain any loss functions
        ValueError
            If Validation DataLoader is not provided when save_path is specified.
        """
        # check if loss manager contains any loss functions
        if len(self.loss_manager.loss_functions) == 0:
            raise ValueError("Loss Manager must contain at least one loss function.")

        if save_path is not None and val_loader is None:
            self.logger.error("Validation DataLoader is required to save the model checkpoints.")
            raise ValueError("Validation DataLoader is required to save the model checkpoints.")
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            self.logger.info(f"Starting Epoch {epoch}/{epochs}")
            train_loss = self.train_epoch(train_loader)

            if val_loader:
                val_loss = self.evaluate(val_loader)
                self.logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Save the model if validation loss has improved
                if val_loss < best_val_loss and save_path:
                    self.logger.info("Attempting to save model...")
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), save_path)
                    self.logger.info(f"Validation loss improved. Model saved to {save_path}")
            else:
                self.logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}")

    def infer(self, data_loader: DataLoader) -> dict[str, torch.Tensor]:
        """
        Generates modified embeddings using the trained model.

        Args:
            data_loader: DataLoader for embeddings to be modified.

        Returns
        -------
            modified_embeddings : Dict[str, torch.Tensor] Dictionary containing lists of embeddings. (dtype `dict`)
        """
        self.model.eval()
        modified_embeddings = {self.data_emb_key: [], self.context_emb_key: [], self.sample_id_key: []}

        with torch.no_grad():
            for batch in data_loader:
                in_main = batch[self.input_embeddings["main"]].to(
                    self.device
                )  # Shape: (batch_size, seq_length, embedding_dim)
                if len(self.input_embeddings) == 2:
                    in_cross = batch[self.input_embeddings["cross"]].to(
                        self.device
                    )  # Shape: (batch_size, seq_length, embedding_dim)
                else:
                    in_cross = None
                sample_ids = batch[self.sample_id_key]  # Shape: (batch_size, seq_length)  # Not used in prediction

                # Forward pass
                out = self.model(in_main=in_main, in_cross=in_cross)
                outputs = {}
                outputs[self.input_embeddings["main"]] = out
                outputs[self.input_embeddings["cross"]] = (
                    in_cross  # Pass the original embeddings used for cross attention to allow for data_context contrastive loss
                )
                # Collect modified_embeddings
                modified_embeddings[self.data_emb_key].append(outputs[self.data_emb_key].cpu())
                modified_embeddings[self.context_emb_key].append(outputs[self.context_emb_key].cpu())
                modified_embeddings[self.sample_id_key].append(sample_ids.cpu())

        # Concatenate all batches
        modified_embeddings = {k: torch.cat(v, dim=0) for k, v in modified_embeddings.items()}
        return modified_embeddings
