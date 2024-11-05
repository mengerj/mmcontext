# engine/trainer.py

import logging

import torch
from torch.utils.data import DataLoader

from mmcontext.engine import ContrastiveLoss, LossManager


class Trainer:
    """
    Trainer class to handle the training and evaluation of models.

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
    data_key
        Key for data embeddings in the batch. Defaults to 'data_embedding'.
    context_key
        Key for context embeddings in the batch. Defaults to 'context_embedding'.

    Raises
    ------
    ValueError
        If context_embedding is not the main input_embeddings for context_context contrastive loss.
    ValueError
        If data_embedding is not the main input_embeddings for data_data contrastive loss.
    ValueError
        If in_cross embeddings are not provided when using current_mode = data_context contrastive loss.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        loss_manager: LossManager | None = None,
        device: torch.device | None = None,
        logger: logging.Logger | None = None,
        input_embeddings: dict | None = None,
        data_key: str = "data_embedding",
        context_key: str = "context_embedding",
        sample_id_key: str = "sample_id",
        temperature: float | None = None,
    ):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_manager = loss_manager
        self.optimizer = optimizer
        self.data_key = data_key
        self.context_key = context_key
        self.sample_id_key = sample_id_key
        self.logger = logger or logging.getLogger(__name__)
        self.seq_length = None  # Placeholder to store seq_length for inference
        self.batch_size = None  # Placeholder to store batch_size for inference
        self.temperature = temperature
        self.model.temperature = temperature

        # Set default input_embeddings if not provided
        if input_embeddings is None:
            self.logger.info(
                "input_embeddings not provided. Using default values. Data embedding is main and context embedding is cross."
            )
            self.input_embeddings = {"main": data_key, "cross": context_key}
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
                    if self.input_embeddings["main"] != self.context_key:
                        self.logger.error(
                            "context_embedding should be the main input_embeddings for context_context (current_mode) contrastive loss."
                        )
                        raise ValueError(
                            "context_embedding should be the main input_embeddings for context_context (current_mode) contrastive loss."
                        )
                if loss_fn.current_mode == "data_data":
                    # if 'data_embedding' are not in first position of input embeddings, raise an error
                    if self.input_embeddings["main"] != self.data_key:
                        self.logger.error(
                            "data_embedding should be the main input_embeddings for data_data (current_mode) contrastive loss."
                        )
                        raise ValueError(
                            "data_embedding should be the main input_embeddings for data_data (current_mode) contrastive loss."
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
                self.logger.info(f"Temperature: {self.temperature}")
                if self.temperature is not None and self.model.learn_temperature:
                    raise ValueError(
                        "Temperature is provided, but model wasn't reloaded. Reinitialize model to ensure temperature won't be learned."
                    )
                if loss_fn.target_mode == "infoNCE" and self.temperature is None:
                    self.model.learn_temperature = True

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

        Raises
        ------
        ValueError
            If Loss Manager does not contain any loss functions
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
            if self.seq_length is None and in_main.dim() == 3:
                self.seq_length = in_main.size(1)
            if self.batch_size is None:
                self.batch_size = in_main.size(0)
            # Forward pass
            out, temp = self.model(in_main=in_main, in_cross=in_cross)
            outputs = {}
            outputs[self.input_embeddings["main"]] = out
            outputs[self.input_embeddings["cross"]] = (
                in_cross  # Pass the original embeddings used for cross attention to allow for data_context contrastive loss
            )
            outputs["temperature"] = temp
            self.temperature = temp
            # Prepare targets
            targets = {self.data_key: batch[self.data_key], self.context_key: batch[self.context_key]}

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
        Average validation loss.
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
                out, temp = self.model(in_main=in_main, in_cross=in_cross)
                outputs = {}
                outputs[self.input_embeddings["main"]] = out
                outputs[self.input_embeddings["cross"]] = (
                    in_cross  # Pass the original embeddings used for cross attention to allow for data_context contrastive loss
                )
                outputs["temperature"] = temp
                # Prepare targets
                targets = {
                    self.data_key: batch[self.data_key],
                    self.context_key: batch[self.context_key],
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
            self.logger.info(f"Temperature: {self.temperature}")

    def infer(self, data_loader: DataLoader) -> dict[str, torch.Tensor]:
        """
        Generates modified embeddings using the trained model.

        Parameters
        ----------
        data_loader
            DataLoader for embeddings to be modified.

        Returns
        -------
        modified_embeddings : Dict[str, torch.Tensor] Dictionary containing lists of embeddings.
        """
        self.model.eval()
        modified_embeddings = {self.data_key: [], self.context_key: [], self.sample_id_key: []}

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
            outputs[self.input_embeddings["cross"]] = in_cross  # Pass the original embeddings used for cross attention

            # Initialize dictionary to collect outputs if not already initialized
            if "modified_embeddings" not in locals():
                modified_embeddings = {self.data_key: [], self.context_key: [], self.sample_id_key: []}

            # Collect outputs, ensuring to remove any singleton dimensions
            # Squeeze only if the dimension is 1 to avoid altering data shape unintentionally
            modified_embeddings[self.data_key].append(
                outputs[self.data_key].squeeze().cpu()
                if outputs[self.data_key].dim() == 1
                else outputs[self.data_key].cpu()
            )
            modified_embeddings[self.context_key].append(
                outputs[self.context_key].squeeze().cpu()
                if outputs[self.context_key].dim() == 1
                else outputs[self.context_key].cpu()
            )
            modified_embeddings[self.sample_id_key].append(
                sample_ids.cpu()
            )  # Assuming sample_ids does not need squeezing

            # Concatenate all batches
            modified_embeddings = {k: torch.cat(v, dim=0) for k, v in modified_embeddings.items()}
        return modified_embeddings

    def infer_adata(self, adata, sample_id_key: str, seq_length: int | None = None, batch_size: int | None = None):
        """
        Generates modified embeddings for the samples in the given AnnData object using the trained model.

        Parameters
        ----------
        adata
            AnnData object containing the data to infer on.
        sample_id_key
            Key in `adata.obs` containing the sample IDs.
        """
        # Import DataSetConstructor
        from mmcontext.pp import DataSetConstructor

        # Initialize DataSetConstructor
        dataset_constructor = DataSetConstructor(in_sample_id_key=sample_id_key, out_sample_id_key=sample_id_key)

        # Add the AnnData object to the dataset
        dataset_constructor.add_anndata(adata)

        # Use the same seq_length as used during training
        if seq_length is None:
            if self.seq_length is None:
                self.logger.error("seq_length not provided. Please provide seq_length.")
                raise ValueError("seq_length not provided. Please provide seq_length.")
            seq_length = self.seq_length
        if batch_size is None:
            if self.batch_size is None:
                self.logger.error("batch_size not provided. Please provide batch_size.")
                raise ValueError("batch_size not provided. Please provide batch_size.")
            batch_size = self.batch_size

        # Construct the dataset
        dataset = dataset_constructor.construct_dataset(seq_length=seq_length)

        # Create a DataLoader with shuffle=False to maintain order
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Initialize lists to collect outputs and sample IDs
        modified_embeddings = {self.data_key: [], sample_id_key: []}

        self.model.eval()

        with torch.no_grad():
            for _batch_idx, batch in enumerate(data_loader):
                # Get sample IDs
                sample_ids = batch[sample_id_key]  # Shape: (batch_size, seq_length)

                # Get inputs
                in_main = batch[self.input_embeddings["main"]].to(self.device)
                if self.input_embeddings["cross"] is not None:
                    in_cross = batch[self.input_embeddings["cross"]].to(self.device)
                else:
                    in_cross = None

                # Forward pass
                out, _ = self.model(in_main=in_main, in_cross=in_cross)

                # Flatten seq_length dimension
                batch_size, seq_len, emb_dim = out.size()
                out_flat = out.view(batch_size * seq_len, emb_dim)
                sample_ids_flat = sample_ids.view(batch_size * seq_len)

                # Collect outputs and sample IDs
                modified_embeddings[self.data_key].append(out_flat.cpu())
                modified_embeddings[sample_id_key].append(sample_ids_flat.cpu())

        # Concatenate all outputs
        modified_embeddings = {k: torch.cat(v, dim=0) for k, v in modified_embeddings.items()}

        # Map sample IDs to embeddings
        id_to_embedding = {}
        for sample_id, embedding in zip(
            modified_embeddings[sample_id_key], modified_embeddings[self.data_key], strict=False
        ):
            sample_id = sample_id.item()
            id_to_embedding[sample_id] = embedding

        # Now, create an array of embeddings aligned with adata

        sample_ids_in_adata = adata.obs[sample_id_key].values
        embeddings_list = []
        sample_ids_to_cut = []
        for sample_id in sample_ids_in_adata:
            embedding = id_to_embedding.get(sample_id)
            if embedding is None:
                self.logger.warning(f"Sample ID {sample_id} not found in inference outputs.")
                sample_ids_to_cut.append(sample_id)
                # jump to the next iteration
                continue
            embeddings_list.append(embedding)

        # remove the samples not found in the inference outputs
        if len(sample_ids_to_cut) > 0:
            self.logger.warning(
                f"Removing {len(sample_ids_to_cut)} samples not found in inference outputs. Most likely because could not be processed due to seq_length."
            )
            idx_to_keep = ~adata.obs[sample_id_key].isin(sample_ids_to_cut)
            adata_new = adata[idx_to_keep].copy()
        else:
            adata_new = adata.copy()
        # Stack embeddings into a tensor
        embeddings_tensor = torch.stack(embeddings_list)
        adata_new.obsm["mod_emb"] = embeddings_tensor.detach().numpy()
        return adata_new
