# engine/trainer.py

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr as zarr_module
from numcodecs import Blosc
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mmcontext.engine import MMContextEncoder, PlaceholderModel


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
    encoders : Dict[str, torch.nn.Module]
        A dictionary of encoder models to train. Keys are encoder names, values are encoder instances. The outputs of the first encoder will be passed to the decoder.
        The output of the second encoder can be used for contrastive loss.
    decoder : torch.nn.Module, optional
        The decoder model to train. It outputs parameters of a distribution like ZINB which can be used to calculate the loss.
        By default, it is set to a PlaceholderModel which does nothing.
    loss_manager : LossManager, optional
        Manages multiple loss functions. Not needed for inference.
    optimizer : torch.optim.Optimizer, optional
        Optimizer for updating model parameters.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Scheduler for adjusting learning rate during training.
    device : torch.device, optional
        Device to run the model on 'cuda' or 'cpu'. Defaults to CUDA if available.
    logger : logging.Logger, optional
        Logger for tracking training progress. If None, a default logger is created.
    encoder_inputs : Dict[str, str], optional
        Dict mapping encoder names to their respective input keys in the batch.
    decoder_input_key : str, optional
        Key for the input to the decoder in the batch. Only used if no encoders are provided.
    sample_id_key : str, optional
        Key for sample IDs in the batch.
    raw_data_key : str, optional
        Key for raw data in the batch.
    temperature : float, optional
        Temperature parameter for contrastive loss.
    """

    def __init__(
        self,
        encoders: dict[str, torch.nn.Module] | torch.nn.Module | None = None,
        decoder: torch.nn.Module | None = None,
        loss_manager: Any | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        device: torch.device | None = None,
        logger: logging.Logger | None = None,
        encoder_inputs: dict[str, str] | None = None,
        decoder_input_key: str = None,
        sample_id_key: str = "sample_id",
        raw_data_key: str = "raw_data",
        temperature: float | None = None,
    ):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.getLogger(__name__)
        # Initialize encoders
        if encoders is None:
            self.encoders = {}
            self.logger.info(
                f"No encoders provided. Only the decoder can be trained. With the input in {decoder_input_key}."
            )
            self.decoder_input_key = decoder_input_key
        elif isinstance(encoders, torch.nn.Module):
            self.encoders = {"data_encoder": encoders.to(self.device)}
        else:
            self.encoders = {name: encoder.to(self.device) for name, encoder in encoders.items()}

        # Initialize decoder
        if decoder is None:
            decoder = PlaceholderModel()
        self.decoder = decoder.to(self.device)

        self.loss_manager = loss_manager
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sample_id_key = sample_id_key
        self.raw_data_key = raw_data_key
        self.seq_length = None  # Placeholder to store seq_length for inference
        self.batch_size = None  # Placeholder to store batch_size for inference
        self.temperature = temperature

        # Set temperature for encoders
        for encoder in self.encoders.values():
            encoder.temperature = temperature

        # Set default encoder_inputs if not provided
        if encoder_inputs is None:
            if len(self.encoders) > 1:
                raise ValueError(
                    "Multiple encoders provided. No default implemented for multiple encoders. Please provide encoder_inputs."
                )
            if len(self.encoders) > 0 and not all(
                isinstance(encoder, MMContextEncoder) for encoder in self.encoders.values()
            ):
                raise ValueError(
                    "Encoder is not MMContextEncoder. No default implemented for other encoders. Please provide encoder_inputs."
                )
            self.logger.info("encoder_inputs not provided. Using default values.")
            encoder_inputs = {name: {"in_main": "d_emb", "in_cross": "c_emb"} for name in self.encoders.keys()}
        self.encoder_inputs = encoder_inputs

        if self.loss_manager is not None:
            self._validate_encoders()
        if all(isinstance(encoder, PlaceholderModel) for encoder in self.encoders.values()) and isinstance(
            self.decoder, PlaceholderModel
        ):
            raise ValueError("Both encoders and decoder are PlaceholderModels. Please provide at least one model.")

    def _validate_encoders(self):
        for loss_fn, _ in self.loss_manager.loss_functions:
            if hasattr(loss_fn, "required_encoders"):
                for encoder_name in loss_fn.required_encoders:
                    if encoder_name not in self.encoders:
                        self.logger.error(
                            f"Encoder '{encoder_name}' required by loss function '{loss_fn}' is not provided."
                        )
                        raise ValueError(
                            f"Encoder '{encoder_name}' required by loss function '{loss_fn}' is not provided."
                        )
            if hasattr(loss_fn, "temperature_dependent") and loss_fn.temperature_dependent:
                if self.temperature is not None and any(
                    encoder.learn_temperature for encoder in self.encoders.values()
                ):
                    raise ValueError(
                        "Temperature is provided, but encoder wasn't reloaded. Reinitialize the encoder to ensure temperature won't be learned."
                    )
                if self.temperature is None:
                    for encoder in self.encoders.values():
                        encoder.learn_temperature = True

    def process_batch(self, batch: dict[str, Any]) -> (dict[str, Any], dict[str, Any]):
        """
        Process a single batch of data.

        Parameters
        ----------
        batch : dict
            A batch of data from the DataLoader.

        Returns
        -------
        outputs : dict
            Dictionary containing outputs from encoders and decoder.
        targets : dict
            Dictionary containing targets for loss computation.
        """
        outputs = {}
        # Process encoder if any
        if len(self.encoders) > 0:
            # Forward pass through the encoder
            enc_outputs, decoder_input_name = self._run_encoder(batch)
            # Add encoder outputs to outputs
            outputs.update(enc_outputs)
        else:
            # If no encoders, use the input directly for the decoder
            mod_emb = batch[self.decoder_input_key].to(self.device)
            decoder_input_name = "decoder_input"
            outputs[decoder_input_name] = mod_emb
        # From the batch, get the amount of samples based of the first entry of batch
        self.samples_per_batch = list(batch.values())[0].size(0) * list(batch.values())[0].size(1)
        # If decoder is provided, pass the appropriate embeddings through decoder
        if not isinstance(self.decoder, PlaceholderModel):
            mod_emb = outputs[decoder_input_name]
            flat_mod_emb = mod_emb.view(-1, mod_emb.size(-1))
            out_distribution = self.decoder(flat_mod_emb)
            outputs["out_distribution"] = out_distribution

        # Get all embeddings from batch (all elements besides sample id and raw data)
        embeddings = {
            key: batch[key].to(self.device) for key in batch if key not in [self.sample_id_key, self.raw_data_key]
        }
        # Prepare targets, which are embeddings plus the raw data
        targets = {**embeddings, self.raw_data_key: batch[self.raw_data_key].to(self.device)}
        # Check if all elements from the batch, besides sample id and raw data are in the outputs.
        # If not, they were not used in the encoder forward pass, but might be needed for loss computation.
        for key in embeddings:
            if key not in outputs:
                # self.logger.info(f"Key '{key}' not found in outputs. Most likely not used in forward pass. Adding unmodified to outputs.")
                outputs[key] = embeddings[key]
        return outputs, targets

    def _run_encoder(self, batch):
        encoder_in = {}
        all_outputs = {}
        for encoder_name in self.encoders.keys():
            encoder = self.encoders[encoder_name]
            input_dict = self.encoder_inputs.get(encoder_name)
            if input_dict is None:
                self.logger.error(f"No input key provided for encoder '{encoder_name}'.")
                raise ValueError(f"No input key provided for encoder '{encoder_name}'.")
            for input_key in input_dict.keys():
                if input_dict[input_key] not in batch:
                    self.logger.error(f"Input key '{input_dict[input_key]}' not found in batch.")
                    raise ValueError(f"Input key '{input_dict[input_key]}' not found in batch.")
                encoder_in[input_key] = batch[input_dict[input_key]].to(self.device)
            for in_embedding in encoder_in.values():
                if self.seq_length is None and in_embedding.dim() == 3:
                    self.seq_length = in_embedding.size(1)
                if self.batch_size is None:
                    self.batch_size = in_embedding.size(0)
            if isinstance(encoder, MMContextEncoder):
                mod_emb, temp = encoder(in_main=encoder_in["in_main"], in_cross=encoder_in["in_cross"])
                output = {input_dict["in_main"]: mod_emb, "temperature": temp}
                self.temperature = temp
                all_outputs.update(output)
            else:
                raise ValueError("Encoder type not yet supported. Add method to Trainer._run_encoder.")
        enc_name = list(self.encoders.keys())[0]  # Get first encoder name
        # Use in_main of the first encoder as the decoder input
        decoder_input_name = self.encoder_inputs[enc_name]["in_main"]
        return all_outputs, decoder_input_name

    def train_epoch(self, data_loader: DataLoader) -> float:
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader
            DataLoader for training data.

        Returns
        -------
            Average training loss for the epoch.
        """
        if len(self.loss_manager.loss_functions) == 0:
            raise ValueError("Loss Manager must contain at least one loss function.")
        if self.optimizer is None:
            raise ValueError("Optimizer is not provided.")
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
        for encoder in self.encoders.values():
            encoder.train()
        self.decoder.train()
        total_loss = 0.0
        num_batches = len(data_loader)
        for batch_idx, batch in enumerate(data_loader):
            time_start = time.time()
            outputs, targets = self.process_batch(batch)
            # Compute loss
            loss = self.loss_manager.compute_total_loss(outputs=outputs, targets=targets)
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            time_end = time.time()
            time_per_sample = (time_end - time_start) / self.samples_per_batch
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                self.logger.info(
                    f"Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}, Time per sample: {time_per_sample:.5f}s"
                )
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
        if len(self.loss_manager.loss_functions) == 0:
            raise ValueError("Loss Manager must contain at least one loss function.")
        for encoder in self.encoders.values():
            encoder.eval()
        self.decoder.eval()
        total_loss = 0.0
        num_batches = len(data_loader)
        with torch.no_grad():
            for _batch_idx, batch in enumerate(data_loader):
                outputs, targets = self.process_batch(batch)
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
        save_dir: str = "./",
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
            Whether to save the model. Validation loader is required and determines the best model. Defaults to False.
        n_epochs_stop
            Number of epochs to wait for improvement in validation loss before stopping training. Defaults to 10.

        Raises
        ------
        ValueError
            If Loss Manager does not contain any loss functions
        ValueError
            If Validation DataLoader is not provided when save is True.
        """
        save_dir = Path(save_dir)
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
                        for name, encoder in self.encoders.items():
                            encoder.save(file_path=save_dir / f"best_{name}_weights.pth")
                            self.logger.info(f"Validation loss improved. Encoder '{name}' weights saved.")
                        self.decoder.save(file_path=save_dir / "best_decoder_weights.pth")
                        self.logger.info("Validation loss improved. Decoder weights saved.")
                else:
                    epochs_no_improve += 1
                    # If the validation loss has not improved for n_epochs_stop epochs, stop training
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
        emb_keys: dict[str, str] | None = None,
        seq_length: int | None = None,
        batch_size: int | None = None,
        chunk_size: int | None = None,
        output_zarr_path: str = "inferred_adata.zarr",
        n_recon: int = 0,
    ):
        """
        Generates modified embeddings and multiple reconstructed matrices for the samples in the given AnnData object.

        Parameters
        ----------
        adata : AnnData
            AnnData object containing the data to infer on.
        sample_id_key : str
            Key in `adata.obs` containing the sample IDs.
        emb_keys : dict, optional
            Keys from 'adata.obsm' to use as embeddings.
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

        if emb_keys is None:
            emb_keys = {"data_embedding": "d_emb", "context_embedding": "c_emb"}
        if chunk_size is None:
            self.logger.error("chunk_size not provided. Please provide chunk_size to the infer_adata method.")
            raise ValueError("chunk_size not provided. Please provide chunk_size to the infer_adata method.")

        dataset_constructor = DataSetConstructor(
            out_emb_keys=emb_keys,
            out_sample_id_key=sample_id_key,
            chunk_size=chunk_size,
            batch_size=batch_size,
        )
        dataset_constructor.add_anndata(adata, sample_id_key=sample_id_key, emb_keys=emb_keys)

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
        # Step 1: Collect all indices to determine size of the output Zarr arrays
        all_indices = []
        for batch in data_loader:
            indices = batch["indices"].view(-1)
            all_indices.append(indices.cpu().numpy())  # Collect indices without loading other data

        # Step 2: Concatenate indices to get the total
        all_indices = np.concatenate(all_indices)  # Flatten all batches into one array

        # Subset the adata object to only include the samples in the data loader and remove it from memory
        adata = adata[all_indices]

        # Write the filtered adata to Zarr
        adata.write_zarr(output_zarr_path, chunks=[adata.shape[0], chunk_size])
        # To create the reconstructed matrices, we need the genes
        n_genes = adata.n_vars
        del adata

        # Compressor settings to match existing arrays
        compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
        num_cells = len(all_indices)

        # Create embeddings arrays
        for encoder_name in self.encoders.keys():
            embeddings_zarr_path = os.path.join(output_zarr_path, "obsm", f"{encoder_name}_mod_emb")
            os.makedirs(embeddings_zarr_path, exist_ok=True)
            emb_dim = self.encoders[encoder_name].embedding_dim
            # Open the zarr storage and add the embeddings
            embeddings_zarr = zarr_module.open_array(
                embeddings_zarr_path,
                mode="w",
                shape=(num_cells, emb_dim),
                chunks=(chunk_size, emb_dim),
                dtype="float32",
                compressor=compressor,
                fill_value=0.0,
            )
            embeddings_zarr[:] = np.nan

            # Include necessary metadata in .zattrs
            embeddings_zarr.attrs["encoding-type"] = "array"
            embeddings_zarr.attrs["encoding-version"] = "0.2.0"

        all_out_params = (
            {key: [] for key in ["mu", "theta", "pi"]} if not isinstance(self.decoder, PlaceholderModel) else None
        )

        for encoder in self.encoders.values():
            encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            for batch in data_loader:
                indices = batch["indices"].view(-1).cpu().numpy()
                outputs, _targets = self.process_batch(batch)
                # Save embeddings
                for encoder_name in self.encoders.keys():
                    mod_emb = outputs[emb_keys["data_embedding"]]
                    flat_mod_emb = mod_emb.view(-1, mod_emb.size(-1))
                    embeddings_zarr_path = os.path.join(output_zarr_path, "obsm", f"{encoder_name}_mod_emb")
                    embeddings_zarr = zarr_module.open_array(embeddings_zarr_path, mode="r+")
                    embeddings_zarr[indices, :] = flat_mod_emb.cpu().numpy()

                # Collect all the output parameters
                if all_out_params is not None:
                    out_distribution = outputs["out_distribution"]
                    for key in all_out_params:
                        all_out_params[key].append(out_distribution[key])

        # Concatenate all the output parameters
        if all_out_params is not None:
            for key in all_out_params:
                all_out_params[key] = torch.cat(all_out_params[key], dim=0)

            # Sampling and storing reconstructed matrices
            recon_zarr_path = os.path.join(output_zarr_path, "layers")
            os.makedirs(recon_zarr_path, exist_ok=True)
            for i in range(1, n_recon + 1):
                recon_zarr = zarr_module.open_array(
                    f"{recon_zarr_path}/reconstructed{i}/",
                    mode="w",
                    shape=(num_cells, n_genes),
                    chunks=(chunk_size, n_genes),
                    dtype="float32",
                    compressor=compressor,
                    fill_value=0.0,
                )
                # Include necessary metadata in .zattrs
                recon_zarr.attrs["encoding-type"] = "array"
                recon_zarr.attrs["encoding-version"] = "0.2.0"
                # Sample from the ZINB distribution to get the reconstructed matrices
                recon_zarr[:] = (
                    sample_zinb(all_out_params["mu"], all_out_params["theta"], all_out_params["pi"]).cpu().numpy()
                )


'''
class Trainer:
    """
    Trainer class to handle the training and evaluation of models.

    Parameters
    ----------
    encoder
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
        Defaults to `{'main': 'd_emb_aligned', 'cross': 'c_emb_aligned'}`. Only the embeddings from main are modified.
        The embeddings corresponding cross are used for cross-attention if the model was configured with use_cross_attention = True.
    data_key
        Key for data embeddings in the batch. Defaults to 'd_emb_aligned'.
    context_key
        Key for context embeddings in the batch. Defaults to 'c_emb_aligned'.

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
        encoder: torch.nn.Module | None = None,
        decoder: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        loss_manager: LossManager | None = None,
        device: torch.device | None = None,
        logger: logging.Logger | None = None,
        input_embeddings: dict | None = None,
        data_key: str = "d_emb_aligned",
        context_key: str = "c_emb_aligned",
        sample_id_key: str = "sample_id",
        raw_data_key: str = "raw_data",
        temperature: float | None = None,
    ):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if encoder is None:
            encoder = PlaceholderModel()
        self.encoder = encoder.to(self.device)
        if decoder is None:
            decoder = PlaceholderModel()
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
        if isinstance(self.encoder, PlaceholderModel) and isinstance(self.decoder, PlaceholderModel):
            raise ValueError("Both encoder and decoder are PlaceholderModels. Please provide at least one model.")

    def _validate_input_embeddings(self):
        for loss_fn, _ in self.loss_manager.loss_functions:
            if isinstance(loss_fn, ContrastiveLoss):
                if isinstance(self.encoder, PlaceholderModel):
                    self.logger.error(
                        "Encoder is a PlaceholderModel. Please provide an encoder model to calculate contrastive loss."
                    )
                    raise ValueError(
                        "Encoder is a PlaceholderModel. Please provide an encoder model to calculate contrastive loss."
                    )
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
        if not isinstance(self.encoder, PlaceholderModel):
            self.encoder.train()
        total_loss = 0.0
        num_batches = len(data_loader)
        for batch_idx, batch in enumerate(data_loader):
            time_start  = time.time()
            if not isinstance(self.encoder, PlaceholderModel):
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
                # Forward pass through the encoder
                mod_emb, temp = self.encoder(in_main=in_main, in_cross=in_cross)
            else: # If no encoder is used, the input embeddings are to be passed to the decoder
                mod_emb = batch[self.data_key].to(self.device)
                in_cross = None # To avoid error for loss. Not used if only decoder is used
                temp = None
            flat_mod_emb = mod_emb.view(
                -1, mod_emb.size(-1)
            )  # Flatten the output for the decoder as we are not using the seq_length dimension
            out_distribution = self.decoder(
                flat_mod_emb
            )  # outputs of the decoder are expected to be parameters of a distribution in a dict
            outputs = {}
            outputs[self.input_embeddings["main"]] = mod_emb
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
            time_end = time.time()
            time_per_sample = (time_end - time_start) / flat_mod_emb.size(0)
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                self.logger.info(f"Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}, Time per sample: {time_per_sample:.5f}s")

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
                if not isinstance(self.encoder, PlaceholderModel):
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
                    mod_emb, temp = self.encoder(in_main=in_main, in_cross=in_cross)
                else: # If no encoder is used, the input embeddings are to be passed to the decoder
                    mod_emb = batch[self.data_key].to(self.device)
                    in_cross = None
                    temp = None
                flat_mod_emb = mod_emb.view(
                    -1, mod_emb.size(-1)
                )  # Flatten the output for the decoder as we are not using the seq_length dimension
                out_distribution = self.decoder(flat_mod_emb)
                outputs = {}
                outputs[self.input_embeddings["main"]] = mod_emb
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
        save_dir: str = "./",
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
        save_dir = Path(save_dir)
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
                        self.encoder.save(file_path= save_dir / "best_encoder_weights.pth")
                        self.logger.info("Validation loss improved. Encoder weights saved in current working dir.")
                        self.decoder.save(file_path= save_dir / "best_decoder_weights.pth")
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
        emb_keys: dict | None = None,
        seq_length: int | None = None,
        batch_size: int | None = None,
        chunk_size: int | None = None,
        output_zarr_path: str = "inferred_adata.zarr",
        n_recon: int = 0,
    ):
        """
        Generates modified embeddings and multiple reconstructed matrices for the samples in the given AnnData object.

        Parameters
        ----------
        adata : AnnData
            AnnData object containing the data to infer on.
        sample_id_key : str
            Key in `adata.obs` containing the sample IDs.
        emb_keys : dict, optional
            Keys from 'adata.obsm' to use as embeddings. Defaults to `{'data_embedding': 'd_emb_aligned', 'context_embedding': 'c_emb_aligned'}`. If you want these embeddings
            to be passed to the decoder directly, make sure you set encoder=None and decoder to the desired decoder model.
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
        if emb_keys is None:
            emb_keys = {"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"}
        if chunk_size is None:
            self.logger.error("chunk_size not provided. Please provide chunk_size to the infer_adata method.")
            raise ValueError("chunk_size not provided. Please provide chunk_size to the infer_adata method.")

        dataset_constructor = DataSetConstructor(
            out_emb_keys=emb_keys,
            out_sample_id_key=sample_id_key,
            chunk_size=chunk_size,
            batch_size=batch_size,
        )
        dataset_constructor.add_anndata(adata, sample_id_key=sample_id_key, emb_keys=emb_keys)

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
        # Step 1: Collect all indices to determine size of the output Zarr arrays
        all_indices = []
        for batch in data_loader:
            indices = batch["indices"].view(-1)
            all_indices.append(indices.cpu().numpy())  # Collect indices without loading other data

        # Step 2: Concatenate indices to get the total
        all_indices = np.concatenate(all_indices)  # Flatten all batches into one array

        # Subset the adata object to only include the samples in the data loader and remove it from memory
        # Create a subset of adata in memory
        adata = adata[all_indices]

        # Write the filtered adata to Zarr
        adata.write_zarr(output_zarr_path, chunks = [adata.shape[0], chunk_size])
        # To create the reconstructed matrices, we need the genes
        n_genes = adata.n_vars
        # Delete the filtered adata from memory
        del adata

        # Temp path to store inferred data and reconstructed matrices
        #temp_output_zarr_path = output_zarr_path.replace(".", "_temp.")
        # Compressor settings to match existing arrays
        compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
        num_cells = len(all_indices)
        if not isinstance(self.encoder, PlaceholderModel):
            embeddings_zarr_path = os.path.join(output_zarr_path, "obsm", "mod_emb")
            os.makedirs(embeddings_zarr_path, exist_ok=True)
            emb_dim = self.encoder.embedding_dim
            # Open the zarr storage and add the embeddings
            embeddings_zarr = zarr_module.open_array(
                embeddings_zarr_path,
                mode="w",
                shape=(num_cells, emb_dim),
                chunks=(chunk_size, emb_dim),
                dtype="float32",
                compressor=compressor,
                fill_value=0.0,
            )
            embeddings_zarr[:] = np.nan

            # Include necessary metadata in .zattrs
            embeddings_zarr.attrs["encoding-type"] = "array"
            embeddings_zarr.attrs["encoding-version"] = "0.2.0"

        all_out_params = (
            {key: [] for key in ["mu", "theta", "pi"]} if not isinstance(self.decoder, PlaceholderModel) else None
        )

        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            for _batch_idx, batch in enumerate(data_loader):
                indices = batch["indices"].view(-1).cpu().numpy()
                if not isinstance(self.encoder, PlaceholderModel):
                    in_main = batch[self.input_embeddings["main"]].to(self.device)
                    in_cross = (
                        batch[self.input_embeddings["cross"]].to(self.device) if self.input_embeddings["cross"] else None
                    )

                    mod_emb, _ = self.encoder(in_main=in_main, in_cross=in_cross)
                else:
                    mod_emb = batch[emb_keys["data_embedding"]].to(self.device)
                    in_cross = None
                batch_size, seq_len, emb_dim = mod_emb.size()
                flat_mod_emb = mod_emb.view(batch_size * seq_len, emb_dim)
                if not isinstance(self.encoder, PlaceholderModel):
                    embeddings_zarr[indices, :] = flat_mod_emb.cpu().numpy()

                # Collect all the output parameters
                if all_out_params is not None:
                    out_distribution = self.decoder(flat_mod_emb)
                    for key in all_out_params:
                        all_out_params[key].append(out_distribution[key])

        # Concatenate all the output parameters
        if all_out_params is not None:
            for key in all_out_params:
                all_out_params[key] = torch.cat(all_out_params[key], dim=0)

        # Sampling and storing reconstructed matrices
        recon_zarr_path = os.path.join(output_zarr_path, "layers")
        os.makedirs(recon_zarr_path, exist_ok=True)
        if all_out_params is not None:
            for i in range(1, n_recon + 1):
                recon_zarr = zarr_module.open_array(
                    f"{recon_zarr_path}/reconstructed{i}/",
                    mode="w",
                    shape=(num_cells, n_genes),
                    chunks=(chunk_size, n_genes),
                    dtype="float32",
                    compressor=compressor,
                    fill_value=0.0,
                )
                # Include necessary metadata in .zattrs
                recon_zarr.attrs["encoding-type"] = "array"
                recon_zarr.attrs["encoding-version"] = "0.2.0"
                # Sample from the ZINB distribution to get the reconstructed matrices
                recon_zarr[:] = sample_zinb(all_out_params["mu"], all_out_params["theta"], all_out_params["pi"]).cpu().numpy()

'''
