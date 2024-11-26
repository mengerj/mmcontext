# engine/trainer.py

import logging

import numpy as np
import torch
import zarr as zarr_module
from numcodecs import Blosc
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mmcontext.engine import ContrastiveLoss, LossManager, PlaceholderModel, ZINBLoss


def configure_optimizer(cfg: DictConfig, model_parameters) -> torch.optim.Optimizer | None:
    """
    Configures the optimizer based on the configuration.

    Parameters
    ----------
    cfg
        The configuration object.
    model_parameters
        The model parameters to optimize.

    Returns
    -------
    torch.optim.Optimizer
        The configured optimizer.
    """
    logger = logging.getLogger(__name__)
    config = cfg.get("optimizer", [])
    # shortly check if there are multiple optimizers set to True and issue a warning that the first one will be selected
    if sum([opt_cfg.get("use") for opt_cfg in config]) > 1:
        logger.warning("Multiple optimizers set to True. The first optimizer will be selected.")
    for opt_cfg in config:
        if not opt_cfg.get("use"):
            continue
        opt_type = opt_cfg.get("type")
        if opt_type in ["adam"]:
            return torch.optim.Adam(
                model_parameters,
                lr=cfg.get("lr", 0.001),
                betas=cfg.get("betas", (0.9, 0.999)),
                weight_decay=cfg.get("weight_decay", 0.0),
            )
        elif opt_type in ["sgd"]:
            return torch.optim.SGD(
                model_parameters,
                lr=cfg.get("lr", 0.01),
                momentum=cfg.get("momentum", 0.9),
                weight_decay=cfg.get("weight_decay", 0.0),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")


def configure_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer):
    """Configures the scheduler based on the configuration.

    Parameters
    ----------
    cfg
        The configuration object.
    optimizer
        The optimizer object.
    """
    logger = logging.getLogger(__name__)
    config = cfg.get("scheduler", [])
    # shortly check if there are multiple schedulers set to True and issue a warning that the first one will be selected
    if sum([sch_cfg.get("use") for sch_cfg in config]) > 1:
        logger.warning("Multiple schedulers set to True. The first scheduler will be selected.")
    for sch_cfg in config:
        if not sch_cfg.get("use", True):
            continue
        sch_type = sch_cfg.get("type")
        if sch_type in ["step"]:
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg.get("step_size", 30), gamma=cfg.get("gamma", 0.1)
            )
        elif sch_type in ["cosine"]:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.get("T_max", 10), eta_min=cfg.get("eta_min", 0)
            )
        elif sch_type in ["None", "none"]:
            # Return a lambda scheduler that does nothing
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        else:
            raise ValueError(f"Unknown scheduler type: {sch_type}")
    # If no scheduler is configured, return LambdaLR with constant function
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)


class Trainer:
    """
    Trainer class to handle the training and evaluation of models.

    Parameters
    ----------
    enocder
        The encoder model to train. It should not modify the data dimension but is trained to learn the context.
    decoder
        The decoder model to train. It outputs paramters of a distribution like ZINB which can be used to calculate the loss.
        By default, it is set to a PlaceholderModel which does nothing.
    loss_manager
        Manages multiple loss functions. Not needed for inference.
    optimizer
        Optimizer for updating model parameters.
    scheduler
        Scheduler for adjusting learning rate during training.
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
        encoder: torch.nn.Module,
        decoder: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        loss_manager: LossManager | None = None,
        device: torch.device | None = None,
        logger: logging.Logger | None = None,
        input_embeddings: dict | None = None,
        data_key: str = "data_embedding",
        context_key: str = "context_embedding",
        sample_id_key: str = "sample_id",
        raw_data_key: str = "raw_data",
        temperature: float | None = None,
    ):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        if decoder is None:
            self.decoder = PlaceholderModel()
        self.decoder = decoder.to(self.device)
        self.loss_manager = loss_manager
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_key = data_key
        self.context_key = context_key
        self.sample_id_key = sample_id_key
        self.raw_data_key = raw_data_key
        self.logger = logger or logging.getLogger(__name__)
        self.seq_length = None  # Placeholder to store seq_length for inference
        self.batch_size = None  # Placeholder to store batch_size for inference
        self.temperature = temperature
        self.encoder.temperature = temperature

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
                if self.temperature is not None and self.encoder.learn_temperature:
                    raise ValueError(
                        "Temperature is provided, but encoder wasn't reloaded. Reinitialize the encoder to ensure temperature won't be learned."
                    )
                if loss_fn.target_mode == "infoNCE" and self.temperature is None:
                    self.encoder.learn_temperature = True
            if isinstance(loss_fn, ZINBLoss):
                if isinstance(self.decoder, PlaceholderModel):
                    self.logger.error(
                        "Decoder is a PlaceholderModel. Please provide a decoder model to calculate ZINB loss."
                    )
                    raise ValueError(
                        "Decoder is a PlaceholderModel. Please provide a decoder model to calculate ZINB loss."
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

        Raises
        ------
        ValueError
            If Loss Manager does not contain any loss functions
        """
        # check if loss manager contains any loss functions
        if len(self.loss_manager.loss_functions) == 0:
            raise ValueError("Loss Manager must contain at least one loss function.")
        if self.optimizer is None:
            raise ValueError("Optimizer is not provided.")
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
        self.encoder.train()
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
            out, temp = self.encoder(in_main=in_main, in_cross=in_cross)
            flat_out = out.view(
                -1, out.size(-1)
            )  # Flatten the output for the decoder as we are not using the seq_length dimension
            out_distribution = self.decoder(
                flat_out
            )  # outputs of the decoder are expected to be parameters of a distribution in a dict
            outputs = {}
            outputs[self.input_embeddings["main"]] = out
            outputs[self.input_embeddings["cross"]] = (
                in_cross  # Pass the original embeddings used for cross attention to allow for data_context contrastive loss
            )
            outputs["temperature"] = temp
            outputs["out_distribution"] = out_distribution
            self.temperature = temp

            # Prepare targets
            targets = {
                self.data_key: batch[self.data_key].to(self.device),
                self.context_key: batch[self.context_key].to(self.device),
                self.raw_data_key: batch[self.raw_data_key].to(self.device),
            }

            # Compute loss
            loss = self.loss_manager.compute_total_loss(outputs=outputs, targets=targets)
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                self.logger.info(f"Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}")

        self.scheduler.step()
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
        self.encoder.eval()
        self.decoder.eval()
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
                out, temp = self.encoder(in_main=in_main, in_cross=in_cross)
                flat_out = out.view(
                    -1, out.size(-1)
                )  # Flatten the output for the decoder as we are not using the seq_length dimension
                out_distribution = self.decoder(flat_out)
                outputs = {}
                outputs[self.input_embeddings["main"]] = out
                outputs[self.input_embeddings["cross"]] = (
                    in_cross  # Pass the original embeddings used for cross attention to allow for data_context contrastive loss
                )
                outputs["temperature"] = temp
                outputs["out_distribution"] = out_distribution
                # Prepare targets
                targets = {
                    self.data_key: batch[self.data_key].to(self.device),
                    self.context_key: batch[self.context_key].to(self.device),
                    self.raw_data_key: batch[self.raw_data_key].to(self.device),
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
        save: bool = False,
        n_epochs_stop: int = 10,
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
        save
            Wether to save the model. Validation loader is required and determines the best model. Defaults to False.
        n_epochs_stop
            Number of epochs to wait for improvement in validation loss before stopping training. Defaults to 10.

        Raises
        ------
        ValueError
            If Loss Manager does not contain any loss functions
        ValueError
            If Validation DataLoader is not provided when save is True.
        """
        # check if loss manager contains any loss functions
        if len(self.loss_manager.loss_functions) == 0:
            raise ValueError("Loss Manager must contain at least one loss function.")

        if save and val_loader is None:
            self.logger.error("Validation DataLoader is required to save the model checkpoints.")
            raise ValueError("Validation DataLoader is required to save the model checkpoints.")
        best_val_loss = float("inf")
        epochs_no_improve = 0
        for epoch in range(1, epochs + 1):
            self.logger.info(f"Starting Epoch {epoch}/{epochs}")
            train_loss = self.train_epoch(train_loader)

            if val_loader:
                val_loss = self.evaluate(val_loader)
                self.logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Save the model if validation loss has improved
                if val_loss < best_val_loss:
                    epochs_no_improve = 0
                    if save:
                        best_val_loss = val_loss
                        self.encoder.save(file_path="best_encoder_weights.pth")
                        self.logger.info("Validation loss improved. Encoder weights saved in current working dir.")
                        self.decoder.save(file_path="best_decoder_weights.pth")
                        self.logger.info("Validation loss improved. Decoder weights saved in current working dir.")
                else:
                    epochs_no_improve += 1
                    # If the validation loss has not improved for 10 epochs, stop training
                    if epochs_no_improve >= n_epochs_stop:
                        self.logger.info(
                            f"Validation loss has not improved for {n_epochs_stop} epochs. Stopping training."
                        )
                        break
            else:
                self.logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}")
            self.logger.info(f"Temperature: {self.temperature}")

    def infer_adata(
        self,
        adata,
        sample_id_key: str,
        seq_length: int | None = None,
        batch_size: int | None = None,
        chunk_size: int | None = None,
        output_zarr_path: str = "inferred_adata.zarr",
        n_recon: int = 5,
    ):
        """
        Generates modified embeddings and multiple reconstructed matrices for the samples in the given AnnData object.

        Parameters
        ----------
        adata : AnnData
            AnnData object containing the data to infer on.
        sample_id_key : str
            Key in `adata.obs` containing the sample IDs.
        seq_length : int, optional
            Sequence length for the data loader.
        batch_size : int, optional
            Batch size for the data loader.
        chunk_size : int, optional
            Chunk size for processing data.
        output_zarr_path : str, optional
            Path to store the output Zarr file.
        n_recon : int, optional
            Number of reconstructed matrices to generate for each sample.
        """
        from mmcontext.engine import sample_zinb
        from mmcontext.pp import DataSetConstructor

        if chunk_size is None:
            self.logger.error("chunk_size not provided. Please provide chunk_size to the infer_adata method.")
            raise ValueError("chunk_size not provided. Please provide chunk_size to the infer_adata method.")

        dataset_constructor = DataSetConstructor(
            in_sample_id_key=sample_id_key,
            out_sample_id_key=sample_id_key,
            chunk_size=chunk_size,
            batch_size=batch_size,
        )
        dataset_constructor.add_anndata(adata)

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

        dataset = dataset_constructor.construct_dataset(seq_length=seq_length, return_indices=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        compressor = Blosc(cname="zstd", clevel=3)
        num_cells = adata.n_obs
        emb_dim = self.encoder.embedding_dim
        embeddings_zarr = zarr_module.open_array(
            output_zarr_path + "/embeddings",
            mode="w",
            shape=(num_cells, emb_dim),
            chunks=(chunk_size, emb_dim),
            dtype="float32",
            compressor=compressor,
        )
        embeddings_zarr[:] = np.nan

        all_out_params = (
            {key: [] for key in ["mu", "theta", "pi"]} if not isinstance(self.decoder, PlaceholderModel) else None
        )
        all_indices = []

        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            for _batch_idx, batch in enumerate(data_loader):
                indices = batch["indices"].view(-1).cpu().numpy()
                all_indices.append(indices)
                in_main = batch[self.input_embeddings["main"]].to(self.device)
                in_cross = (
                    batch[self.input_embeddings["cross"]].to(self.device) if self.input_embeddings["cross"] else None
                )

                out, _ = self.encoder(in_main=in_main, in_cross=in_cross)
                batch_size, seq_len, emb_dim = out.size()
                out_flat = out.view(batch_size * seq_len, emb_dim)
                embeddings_zarr[indices, :] = out_flat.cpu().numpy()

                if all_out_params is not None:
                    out_distribution = self.decoder(out_flat)
                    for key in all_out_params:
                        all_out_params[key].append(out_distribution[key])

        # Concatenate collected indices and parameters
        all_indices = np.concatenate(all_indices)
        if all_out_params is not None:
            for key in all_out_params:
                all_out_params[key] = torch.cat(all_out_params[key], dim=0)

        # Sampling and storing reconstructed matrices
        recon_zarrs = []
        if all_out_params is not None:
            genes = adata.var_names
            num_genes = len(genes)
            for i in range(1, n_recon + 1):
                recon_zarr = zarr_module.open_array(
                    output_zarr_path + f"/reconstructed{i}",
                    mode="w",
                    shape=(num_cells, num_genes),
                    chunks=(chunk_size, num_genes),
                    dtype="float32",
                    compressor=compressor,
                )
                recon_zarr[:] = np.nan

                reconstructed = sample_zinb(all_out_params["mu"], all_out_params["theta"], all_out_params["pi"])

                recon_zarr[all_indices, :] = reconstructed

                recon_zarrs.append(recon_zarr)

            # for i in range(n_recon):
            #
            #    zarr = recon_zarrs[i]
            #    for zarr, data in zip(recon_zarrs, reconstructed):
            #        zarr[all_indices, :] = data.cpu().numpy()

        # Remove rows with NaNs
        na_mask = np.isnan(embeddings_zarr).all(axis=1)
        na_idx = np.where(~na_mask)[0]
        adata = adata[na_idx]
        embeddings_zarr = embeddings_zarr[na_idx]

        if all_out_params is not None:
            for i, recon_zarr in enumerate(recon_zarrs, start=1):
                recon_zarr = recon_zarr[na_idx]
                adata.layers[f"reconstructed{i}"] = recon_zarr

        adata.obsm["mod_emb"] = embeddings_zarr
        adata.write_zarr(output_zarr_path)
