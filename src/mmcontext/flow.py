import importlib.resources
import logging
from pathlib import Path
from typing import Any

import anndata
import yaml
from omegaconf import DictConfig
from torch.utils.data import DataLoader

# from mmcontext.pp.utils import remove_entries, consolidate_low_frequency_categories, split_anndata
# from mmcontext.pp import CategoryEmbedder, Embedder, AnnDataStoredEmbedder, MinMaxNormalizer, PCAReducer, DataSetConstructor
# from mmcontext.engine import MMContextEncoder, LossManager, ContrastiveLoss, Trainer

logger = logging.getLogger(__name__)


class MMContextPipeline:
    """
    A pipeline for building and training a multimodal context embedding model using AnnData as the core data structure.

    This class provides two primary usage patterns:
        1. Loading an existing model and its weights via `load()`.
        2. Providing training and validation AnnData objects to train a new model.

    Attributes
    ----------
    config : dict, optional
        Configuration parameters for the pipeline. Defaults can be overwritten via `configure()` or set manually.
    train_adata : AnnData, optional
        The AnnData object holding the training data, assigned by `add_train_adata()`.
    val_adata : AnnData, optional
        The AnnData object holding the validation data, assigned by `add_val_adata()`.
    encoders : dict
        Dictionary of encoders used by the Trainer. Keys refer to encoder names, values are `MMContextEncoder` instances.
    loss_manager : LossManager
        Manages loss functions used in training.
    trainer : Trainer
        The high-level training engine.
    """

    def __init__(self, config_path: str | Path = None) -> None:
        """
        Initialize the MMContextPipeline.

        Parameters
        ----------
        config_path : dict, optional
            A path to a yaml file with configuration options. See mmcontext.conf.main_default.yaml for an example.
            If not provided, this default will be used.
        """
        self._load_config(config_path=config_path)
        self.pp_initialized = False

        self.train_adata = None
        self.val_adata = None
        self.encoders = {}
        self.loss_manager = None
        self.trainer = None

        logger.info("Initialized MMContextPipeline with configuration.")

    def configure(self, config: dict) -> None:
        """
        Configure the pipeline by overriding default parameters with user-supplied configuration.

        Parameters
        ----------
        config : dict
            Dictionary containing configuration options to override pipeline defaults.
        """
        # Overwrite or update self.cfg with the user-supplied dictionary.
        self.cfg.update(config)
        logger.info("Pipeline configuration updated.")

    def load(self, model_path: str, device: str | None = "cpu") -> None:
        """
        Load a previously trained model from a file.

        Parameters
        ----------
        model_path : str
            Path to the model weights file.
        device : str, optional
            Device to map the model's parameters to (e.g., 'cpu' or 'cuda'). Default is 'cpu'.

        Notes
        -----
        - In frameworks like HuggingFace, model weights and architecture are often saved together.
          In this example, we currently only load weights. Future improvements might store architecture
          in a config file (JSON or YAML) to ensure the model definitions match the weights.
        """
        logger.info(f"Loading model from {model_path} to device {device}.")
        # Example placeholder:
        #   1) Re-initialize model architecture from stored config
        #   2) Load state_dict
        #   3) Map to device
        #
        # Example:
        #   state_dict = torch.load(model_path, map_location=device)
        #   self.encoders["data_encoder"].load_state_dict(state_dict["data_encoder"])
        #   self.encoders["context_encoder"].load_state_dict(state_dict["context_encoder"])
        #
        # self.encoders["data_encoder"].to(device)
        # self.encoders["context_encoder"].to(device)

        # For now, assume the pipeline is set up with an identical architecture
        # The user should have previously defined self.encoders.
        pass

    def add_train_adata(self, adata: Any, sample_id_key: str) -> None:
        """
        Add and preprocess training data (AnnData) to the pipeline.

        This method:
        1. Removes entries and consolidates low-frequency categories.
        2. Creates, normalizes, and aligns context & data embeddings.
        3. Constructs training datasets and dataloaders.

        Parameters
        ----------
        adata : AnnData
            The AnnData object containing the training data. Sourced from user input or a file (e.g., local .h5ad).
        sample_id_key : str
            Key in adata.obs that contains the sample IDs.
        """
        if not self.pp_initialized:
            self._prepare_preprocessing(self.cfg.pp)

        if not self.train_dataset_initialized:
            self.train_dataset_constructor = self._init_dataset_constructor(self.cfg.dataset)
            self.train_dataset_initialized = True

        self.n_genes.append(adata.shape[1])
        if len(set(self.n_genes)) > 1:
            raise ValueError("The number of genes in the datasets is not consistent.")
        logger.info("Adding training AnnData to pipeline.")
        logger.debug("Splitting, removing entries, and consolidating categories.")
        train_adata = self._preprocess(adata)

        logger.debug("Creating embeddings for train and test data.")
        self._create_and_normalize_embeddings(train_adata)

        logger.debug("Aligning embeddings for train and test data.")
        self._align_embeddings(train_adata)

        self.train_dataset_constructor.add_anndata(
            train_adata, sample_id_key=sample_id_key, emb_keys=self.cfg.dataset.in_emb_keys
        )

    def add_val_adata(self, adata: Any, sample_id_key: str) -> None:
        """
        Add and preprocess validation data (AnnData) to the pipeline.

        Parameters
        ----------
        adata : AnnData
            The AnnData object containing validation data.
            Sourced from user input or a file (e.g., local .h5ad).
        sample_id_key : str
            Key in adata.obs that contains the sample IDs.

        Notes
        -----
        - Preprocessing steps should match those done for the training data to ensure consistency.
        - In many scenarios, you might not do an additional split, but you may still want to
          remove entries, consolidate categories, and embed the data.
        """
        if not self.pp_initialized:
            self._prepare_preprocessing(self.cfg.pp)

        if not self.val_dataset_initialized:
            self.val_dataset_constructor = self._init_dataset_constructor(self.cfg.dataset)
            self.val_dataset_initialized = True

        self.n_genes.append(adata.shape[1])
        if len(set(self.n_genes)) > 1:
            raise ValueError("The number of genes in the datasets is not consistent.")
        logger.info("Adding validation AnnData to pipeline.")
        logger.debug("Splitting, removing entries, and consolidating categories.")
        val_adata = self._preprocess(adata)

        logger.debug("Creating embeddings for train and test data.")
        self._create_and_normalize_embeddings(val_adata)

        logger.debug("Aligning embeddings for train and test data.")
        self._align_embeddings(val_adata)

        self.val_dataset_constructor.add_anndata(
            val_adata, sample_id_key=sample_id_key, emb_keys=self.cfg.dataset.in_emb_keys
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
        save: bool = True,
        save_path: str = "model.pt",
    ) -> Any:
        """
        Train the model using provided train and optional validation DataLoaders.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for the training data.
        val_loader : DataLoader, optional
            DataLoader for the validation data. Defaults to None.
        epochs : int, optional
            Number of epochs to train. Defaults to 10.
        save : bool, optional
            If True, save the best-performing model weights during training. Defaults to True.
        save_path : str, optional
            File path to save the model weights if `save=True`. Defaults to 'model.pt'.

        Returns
        -------
        dict
            A dictionary containing training metrics, such as final loss and/or any other information.
        """
        logger.info("Starting training process.")
        if not self.trainer:
            logger.debug("No trainer found. Initializing trainer.")
            self._init_trainer()  # You can specify your encoders, loss, etc.

        # monitor = None
        # from mmcontext.eval.system_usage import SystemMonitor
        # monitor = SystemMonitor(interval=0.1)
        # monitor.start()

        logger.debug("Fitting the model on train_loader and optional val_loader.")
        results = self.trainer.fit(train_loader, val_loader, epochs=epochs, save=save, save_path=save_path)

        # monitor.stop()  # if monitoring system usage
        logger.info("Training complete.")
        return results

    def fine_tune(self, *args, **kwargs) -> Any:
        """
        Fine-tune a loaded model on new data.

        Parameters
        ----------
        *args, **kwargs
            Placeholder for arguments that would be relevant to fine-tuning.

        Returns
        -------
        Any
            Results of the fine-tuning, similarly to `train()`.
        """
        logger.info("Starting fine-tuning process.")
        # Example:
        #   1) Load model architecture + weights (if not done already)
        #   2) Possibly freeze/unfreeze certain layers
        #   3) Proceed with a training loop similar to `train()`
        pass

    # -----------------------------------------------------------------------
    # Internally used helper methods
    # -----------------------------------------------------------------------
    def _load_config(self, config_path: str | Path) -> None:
        """Load config from a YAML file and save as DictConfig in self.cfg."""
        if not config_path:
            # Load the default config file if non is provided
            with importlib.resources.files("mmcontext.conf", "main_default.yaml") as file:
                cfg = yaml.load(file)
            self.cfg = DictConfig(cfg)
        else:
            # load the provided config file
            with open(config_path) as file:
                cfg = yaml.safe_load(file)
            self.cfg = DictConfig(cfg)

    def _prepare_preprocessing(self, pp_cfg):
        """Intialize the objects required for preprocessing.

        Only called the first time preprocessing is done.

        Parameters
        ----------
        pp_cfg : dict
            Configuration for preprocessing.
        """
        from mmcontext.pp import Embedder
        from mmcontext.pp.pp_configurators import configure_aligner, configure_embedder, configure_normalizer

        data_embedder, context_embedder = configure_embedder(pp_cfg.embedder)
        self.embedder = Embedder(data_embedder, context_embedder)

        self.normalizer = configure_normalizer(pp_cfg.normalizer)
        self.aligner = configure_aligner(pp_cfg.aligner)

        self.n_genes = []
        self.pp_initialized = True

    def _init_dataset_constructor(self, dataset_cfg):
        """Initialize the DataSetConstructor object for creating datasets and dataloaders.

        Parameters
        ----------
        dataset_cfg : dict
            Configuration for the dataset constructor.
        """
        from mmcontext.pp import DataSetConstructor

        dataset_constructor = DataSetConstructor(
            out_emb_keys=dataset_cfg.out_emb_keys,
            use_raw=dataset_cfg.use_raw,
            chunk_size=dataset_cfg.chunk_size,
            batch_size=dataset_cfg.batch_size,
            use_dask=dataset_cfg.use_dask,
        )
        return dataset_constructor

    def _preprocess(self, adata: anndata.AnnData, cfg: DictConfig):
        """
        Split the AnnData into train and test sets, then remove and consolidate low-frequency categories.

        Parameters
        ----------
        adata : AnnData
            The AnnData object containing the full dataset.
        train_size : float
            Fraction of data to use for training.

        Returns
        -------
        (AnnData, AnnData)
            The training and test AnnData objects after consolidation.
        """
        from mmcontext.pp.utils import consolidate_low_frequency_categories, remove_entries

        logger.debug("Removing unwanted entries from train and test data.")
        remove_entries(adata)

        logger.debug("Consolidating low-frequency categories for train data.")
        adata_pp = consolidate_low_frequency_categories(
            adata,
            columns=self.cfg.pp.general.categories,
            threshold=self.cfg.pp.general.threshold,
            remove=self.cfg.pp.general.remove,
        )

        return adata_pp

    def _remove_and_consolidate(self, adata):
        """
        Remove entries and consolidate low-frequency categories for validation data.

        Parameters
        ----------
        adata : AnnData
            Validation AnnData to be preprocessed.

        Returns
        -------
        AnnData
            The preprocessed AnnData object.
        """
        # from mmcontext.pp.utils import remove_entries, consolidate_low_frequency_categories
        # remove_entries(adata)
        # adata = consolidate_low_frequency_categories(
        #     adata, columns=["cell_type", "dataset_id"], threshold=5, remove=True
        # )
        return adata

    def _create_and_normalize_embeddings(self, adata):
        """
        Create context and data embeddings the provided AnnData object(s) and normalize them.

        Parameters
        ----------
        *adatas : AnnData
            One or more AnnData objects to process in sequence.

        Notes
        -----
        This uses classes like `CategoryEmbedder` and `AnnDataStoredEmbedder` from `mmcontext.pp.Embedder`,
        followed by `MinMaxNormalizer` from `mmcontext.pp`.
        """
        # Create the embeddings or use precalculated embeddings
        self.embedder.create_embeddings(adata)
        # Normalize the embeddings
        self.normalizer.normalize(adata)

    def _align_embeddings(self, adata):
        """
        Align data and context embeddings to a common dimension using a PCA-based aligner.

        Parameters
        ----------
        adata : AnnData
            AnnData Object whose embeddings need to be aligned to a common dimension.

        Notes
        -----
        Uses `PCAReducer` from `mmcontext.pp`.
        """
        self.aligner.align(adata)

    def _construct_data_loader(self, train_adata, test_adata=None, mode="train"):
        """
        Construct the dataset and DataLoader for training or validation.

        Parameters
        ----------
        train_adata : AnnData
            The AnnData object (or processed subset) for training or validation.
        test_adata : AnnData, optional
            The test AnnData object if in training mode and a separate test set is needed.
        mode : {'train', 'val'}, optional
            Indicates whether constructing for training or validation.

        Returns
        -------
        DataLoader
            A PyTorch DataLoader ready for model training or validation.

        Notes
        -----
        - Uses `DataSetConstructor` from `mmcontext.pp` to build Torch Datasets.
        - This example code is flexible; adapt to your usage.
        """
        # from mmcontext.pp import DataSetConstructor
        # from torch.utils.data import DataLoader

        # batch_size = self.cfg.get("batch_size", 128)
        # seq_length = self.cfg.get("seq_length", 128)

        # dataset_constructor = DataSetConstructor(
        #     out_emb_keys={"data_embedding": "d_emb", "context_embedding": "c_emb"},
        #     use_raw=True,
        #     chunk_size=seq_length * batch_size,
        #     batch_size=batch_size,
        #     use_dask=False
        # )
        # dataset_constructor.add_anndata(
        #     train_adata,
        #     emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
        #     sample_id_key="soma_joinid",
        # )
        # train_dataset = dataset_constructor.construct_dataset(seq_length=seq_length)
        #
        # data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(mode=="train"), num_workers=0)
        # return data_loader
        return None

    def _init_trainer(self) -> None:
        """
        Initialize the Trainer along with the encoders and loss functions, if not already present.

        Notes
        -----
        Encoders are typically instances of `MMContextEncoder`.
        The LossManager can hold multiple losses (e.g., `ContrastiveLoss`).
        """
        # from mmcontext.engine import MMContextEncoder, LossManager, ContrastiveLoss, Trainer
        # import torch

        # self.encoders["data_encoder"] = MMContextEncoder(...)
        # self.encoders["context_encoder"] = MMContextEncoder(...)

        # self.loss_manager = LossManager()
        # self.loss_manager.add_loss(ContrastiveLoss(...))

        # optimizer = torch.optim.Adam(
        #     list(self.encoders["data_encoder"].parameters()) +
        #     list(self.encoders["context_encoder"].parameters())
        # )

        # self.trainer = Trainer(
        #     encoders=self.encoders,
        #     decoder=None,
        #     loss_manager=self.loss_manager,
        #     optimizer=optimizer,
        #     encoder_inputs={
        #         "data_encoder": {"in_main": "d_emb", "in_cross": "c_emb"},
        #         "context_encoder": {"in_main": "c_emb", "in_cross": "d_emb"},
        #     },
        #     temperature=0.07
        # )
        pass
