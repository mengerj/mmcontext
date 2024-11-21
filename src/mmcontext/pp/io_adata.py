import hashlib
import json
import logging
import os
from pathlib import Path

import anndata as ad
import yaml
from omegaconf import DictConfig, OmegaConf


class AnnDataObtainer:
    """
    A class to get anndata. For now only read from disk, but could implement furhter methods.

    Parameters
    ----------
    file_path : str or Path
        Path to the data file.
    cfg : dict
        Configuration dictionary that may contain 'from_cellxgene' flag.
    backed : str, optional
        If 'r' or 'r+', load AnnData in backed mode. Defaults to None.
    """

    def __init__(self, cfg: dict | None = None, backed: str | None = None):
        self.cfg = cfg or {}
        self.backed = backed
        self.adata = None

        # Set from_cellxgene flag, defaulting to False if not specified
        self.from_cellxgene = cfg.get("from_cellxgene", False)

    def get_data(self, file_path: str | Path):
        """
        Load the AnnData object based on the configuration and file path.

        Parameters
        ----------
        file_path
            Path to the data file.
        """
        self.file_path = Path(file_path)
        if self.from_cellxgene:
            self._load_from_cellxgene()
        else:
            self._load_from_file()
        return self.adata

    def _load_from_file(self):
        """Load the AnnData object from a file, handling .h5ad and .zarr formats."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        file_suffix = self.file_path.suffix.lower()
        if file_suffix == ".h5ad":
            self.adata = ad.read_h5ad(self.file_path, backed=self.backed)
        elif file_suffix == ".zarr":
            self.adata = ad.read_zarr(self.file_path)
        else:
            raise ValueError(f"Unsupported file format '{file_suffix}'. Expected '.h5ad' or '.zarr'.")

    ##TODO: Currently not really useful as loop over filenames doesnt allow to consider entries of cfg. So probaly need to first load all datasets and store them in the train/test dirs.
    def _load_from_cellxgene(self):
        """
        Placeholder method to load data from the cellxgene census.

        Not implemented currently.
        """
        # Placeholder for future implementation
        raise NotImplementedError("Loading from cellxgene census is not implemented yet.")


class ConfigWorker:
    """
    Utility class for handling configuration tasks, such as hashing, saving, and loading.

    Parameters
    ----------
    cfg: DictConfig
        The configuration object to work with.
    out_dir: str or Path
        Directory where configurations and related files will be stored.
    """

    def __init__(self, cfg: DictConfig = None, out_dir: str | Path = ""):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.logger = logging.getLogger(__name__)

    def compute_hash(self) -> dict:
        """
        Compute a hash for the configuration and check if it already exists.

        Returns
        -------
        dict: A dictionary containing:
            - 'hash': The computed hash string.
            - 'exists': Whether this hash already exists in the cache.
            - 'path': Suggested path for saving or loading processed data.
        """
        # Convert DictConfig to a plain dictionary and resolve interpolations
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)

        # Convert the configuration to a sorted JSON string and compute the hash
        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        # Define the path for the hash directory
        hash_dir = self.out_dir / config_hash
        os.makedirs(hash_dir, exist_ok=True)

        # Define the settings file path
        hash_file = hash_dir / "settings.yaml"

        # Check if the hash exists
        exists = hash_file.exists()

        return {
            "hash": config_hash,
            "exists": exists,
            "path": hash_file,
        }

    def save_config(self, path: Path | str, config: dict | DictConfig):
        """
        Save a configuration to a YAML file.

        Parameters
        ----------
        path: Path or str
            Path where the configuration will be saved.
        config: dict or DictConfig
            The configuration to save.
        """
        with open(path, "w") as f:
            yaml.dump(config, f, indent=4)
        self.logger.info("Configuration saved to %s", path)

    def load_config(self, path: Path | str) -> dict:
        """
        Load a configuration from a YAML file.

        Parameters
        ----------
        path: Path or str
            Path to the YAML file to load.

        Returns
        -------
        dict: The loaded configuration.
        """
        with open(path) as f:
            config = yaml.safe_load(f)
        self.logger.info("Configuration loaded from %s", path)
        return config

    def check_and_save_config(self) -> dict:
        """
        Check if the configuration hash exists. If not, save the configuration.

        Returns
        -------
        dict: The hash information dictionary returned by `compute_hash`.
        """
        hash_info = self.compute_hash()
        if not hash_info["exists"]:
            # Save the configuration to the corresponding hash file
            config_dict = OmegaConf.to_container(self.cfg, resolve=True)
            self.save_config(hash_info["path"], config_dict)
        else:
            self.logger.info("Hash already exists at %s", hash_info["path"])
        return hash_info

    def transfer_settings(self, settings: dict | DictConfig) -> DictConfig:
        """
        Transfer specific settings from the settings dictionary to the cfg object stored in self.

        Parameters
        ----------
        settings: dict or DictConfig
            The source configuration containing the settings to transfer.

        Returns
        -------
        DictConfig:
            The updated configuration.
        """
        # Convert pp_settings to DictConfig if it's a standard dictionary
        if isinstance(settings, dict):
            settings = OmegaConf.create(settings)

        # Transfer specific settings
        self.cfg.dataset.chunk_size = settings["dataset"]["chunk_size"]
        self.cfg.dataset.seq_length = settings["dataset"]["seq_length"]
        self.cfg.dataset.batch_size = settings["dataset"]["batch_size"]
        self.cfg.engine.models.encoder.latent_dim = settings["pp"]["aligner"]["latent_dim"]
        self.cfg.data.sample_key = settings["data"]["sample_key"]
        self.cfg.data.cell_type_key = settings["data"]["cell_type_key"]
        self.cfg.data.batch_key = settings["data"]["batch_key"]

        return self.cfg


'''
class HasherSaver:
    """
    Utility class to save and hash prepreocessed data.

    Parameters:
    -----------
    cfg: DictConfig
        The configuration dictionary.
    out_dir:
        The directory to save the preprocessed data.
    """
    def __init__(self, cfg: DictConfig, out_dir: str | Path, data_key: str = "d_emb_aligned", context_key: str = "c_emb_aligned"):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.data_key = data_key
        self.context_key = context_key
        self.logger = logging.getLogger(__name__)
    def check_and_save(self, adata, data_type, filename, hash_info):
        """
        Hash the preprocessing conditions and save the preprocessed data.

        Parameters:
        -----------
        adata: anndata.AnnData
            The preprocessed AnnData object.
        data_type: str
            The type of data (train or test).
        filename: str
            The filename to save the data to.
        hash_info: dict
            The hash information dictionary. Computed by get_preprocessing_hash.
        out_dir: str
            The directory to save the preprocessed data.
        """
        # Get the preprocessing hash
        config_hash = hash_info["hash"]
        hash_file = hash_info["path"]
        exists = hash_info["exists"]
        relevant_config = hash_info["relevant_config"]

        # Save the preprocessing conditions if they don't exist
        if not exists:
            self.save_used_configuration(hash_file, relevant_config)
        else:
            self.logger.info("Preprocessing conditions already exist for hash %s", config_hash)
        # Save the preprocessed data
        chunk_size = self.cfg.dataset.chunk_size
        self.save_adata(adata, data_type, filename, config_hash, chunks = [adata.shape[0], chunk_size])

    def save_adata(self, adata, data_type, filename, config_hash, chunks):
        """
        Save the preprocessed data to a file. Saves in zarr format.

        Parameters:
        -----------
        adata: anndata.AnnData
            The preprocessed AnnData object.
        data_type: str
            The type of data (train or test).
        filename: str
            The filename to save the data to.
        config_hash:
            The hash of the preprocessing configuration.
        chunks: dict
            The chunk sizes for the different arrays in the AnnData object.
        """
        save_path = self.out_dir / config_hash / "data" / data_type /f"{filename}.zarr"
        if save_path.exists():
            self.logger.info("Overwriting existing file %s", save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        self.logger.info("Saving preprocessed data to %s", save_path)
        adata.write_zarr(save_path, chunks = chunks)

    def get_config_hash(self):
        """
        Computes a unique hash for preprocessing conditions and checks if it's already cached.

        Returns
        ----------
        dict: A dictionary containing:
            - 'hash': The computed hash string.
            - 'exists': Whether this hash already exists in the cache.
            - 'path': Suggested path for saving or loading processed data.
            - 'relevant_config': The relevant configuration dictionary. Only this part will be used for hashing.
        """

        config_dict = OmegaConf.to_container(self.cfg, resolve=True)

        # Extract relevant preprocessing parameters
        relevant_config = config_dict #Previously I applied some subsetting, but it turned out impractical

        # Convert the configuration to a string
        config_str = json.dumps(relevant_config, sort_keys=True)
        # Compute a unique hash
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        hash_dir = self.out_dir / config_hash

        # Create the cache directory if it doesn't exist
        os.makedirs(hash_dir, exist_ok=True)

        # Define cache file path
        hash_file = os.path.join(hash_dir, "settings.yaml")

        # Check if hash exists
        exists = os.path.exists(hash_file)
        self.logger.info("Current Hash: %s", config_hash)
        return {
            "hash": config_hash,
            "exists": exists,
            "path": hash_file,
            "relevant_config": relevant_config,
        }

    def save_used_configuration(self, hash_file, relevant_config):
        """
        Save preprocessing configuration to a file for future reference.

        Args:
            hash_file (str): Path to the hash file.
            relevant_config (dict): The full configuration dictionary.
        """
        with open(hash_file, "w") as f:
            yaml.dump(relevant_config, f, indent=4)

'''
