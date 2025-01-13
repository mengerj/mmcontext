import logging
import os

import anndata
import torch

from mmcontext.pp import ContextEmbedder, DataEmbedder

logger = logging.getLogger(__name__)


class EncoderPreTrained:
    """Class to use pre-trained encoder models for embedding data and context.

    Class for managing multiple pre-trained encoders (e.g., data encoder and
    context encoder) and applying them to an AnnData object in a two-stage
    embedding process.

    The two-stage embedding process consists of:
    1) Generating an initial embedding using an external embedder.
    2) Passing the initial embedding through a loaded pre-trained encoder model.

    This class is designed to be used in tandem with a FAISS-based indexing
    pipeline. Once `encode_data` and `encode_context` have been called and
    the final embeddings are stored in `adata.obsm`, these embeddings can
    be retrieved in order to build or query a FAISS index.

    Attributes
    ----------
    encoders : Dict[str, torch.nn.Module]
        A dictionary of encoder models, keyed by names such as "data_encoder"
        or "context_encoder". Each encoder must have a compatible forward pass
        with the input embeddings.
    weights_paths : Dict[str, str]
        A dictionary of file paths to the weights for each encoder. The key
        must match the encoder name (e.g., "data_encoder" -> "weights_data.pt").
    data_obsm_key_init : str
        The AnnData `.obsm` key where the initial data embeddings will be stored.
    context_obsm_key_init : str
        The AnnData `.obsm` key where the initial context embeddings will be stored.
    data_obsm_key_final : str
        The AnnData `.obsm` key where the final data embeddings will be stored.
    context_obsm_key_final : str
        The AnnData `.obsm` key where the final context embeddings will be stored.
    """

    def __init__(
        self,
        encoders: dict[str, torch.nn.Module],
        weights_paths: dict[str, str],
        device=torch.device | None,
        data_obsm_key_final: str = "X_data_final",
        context_obsm_key_final: str = "X_context_final",
    ):
        """
        Initialize the EncoderPreTrained instance with given encoders and weights.

        Parameters
        ----------
        encoders : Dict[str, torch.nn.Module]
            Dictionary of encoder models to be used, e.g.,
            {
                "data_encoder": SomeTorchModel(...),
                "context_encoder": AnotherTorchModel(...)
            }
            The user must ensure the architecture of each encoder matches the
            corresponding weights to be loaded.
        weights_paths : Dict[str, str]
            Dictionary of weights paths for each encoder, e.g.,
            {
                "data_encoder": "/path/to/data_encoder_weights.pth",
                "context_encoder": "/path/to/context_encoder_weights.pth"
            }
        data_obsm_key_init : str, optional
            Key under `adata.obsm` for storing initial data embeddings.
            Default is "X_data_init".
        context_obsm_key_init : str, optional
            Key under `adata.obsm` for storing initial context embeddings.
            Default is "X_context_init".
        data_obsm_key_final : str, optional
            Key under `adata.obsm` for storing final data embeddings.
            Default is "X_data_final".
        context_obsm_key_final : str, optional
            Key under `adata.obsm` for storing final context embeddings.
            Default is "X_context_final".
        """
        self.encoders = encoders
        self.weights_paths = weights_paths
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Validate that every encoder has a corresponding weights path
        for encoder_name in encoders:
            if encoder_name not in weights_paths:
                msg = (
                    f"No weights path provided for encoder '{encoder_name}'. "
                    "Please add the correct key in `weights_paths`."
                )
                logger.error(msg)
                raise ValueError(msg)

        # Load weights for each encoder
        self._load_all_weights()

        self.data_obsm_key_final = data_obsm_key_final
        self.context_obsm_key_final = context_obsm_key_final

    def _load_all_weights(self) -> None:
        """
        Internal helper method to load weights for each encoder in self.encoders.

        Raises
        ------
        FileNotFoundError
            If a specified weights file does not exist.
        RuntimeError
            If loading the weights fails or there is a mismatch with the model architecture.
        """
        for name, model in self.encoders.items():
            weights_path = self.weights_paths[name]
            if not os.path.isfile(weights_path):
                logger.error(f"Weights file not found for '{name}' at {weights_path}")
                raise FileNotFoundError(f"{weights_path} does not exist")
            logger.info(f"Loading weights for encoder '{name}' from {weights_path}")
            try:
                model.load_state_dict(torch.load(weights_path))
            except Exception as e:
                logger.error(
                    f"Failed to load weights for encoder '{name}'. "
                    "Ensure the architecture matches the saved weights."
                )
                raise RuntimeError(str(e)) from e

    def encode_data(
        self,
        adata: anndata.AnnData,
        data_embedder: DataEmbedder,
        data_source_info: str = "Unknown data source",
    ) -> None:
        """
        Embed data in two stages: initial embedder + final data encoder.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object containing the dataset to be embedded. Typically
            loaded from a .h5ad file or other single-cell source.
        data_embedder : Callable[[anndata.AnnData], None]
            A function or object that, when called, produces an initial data
            embedding stored in `adata.obsm[self.data_obsm_key_init]`.
        data_source_info : str, optional
            Description of the data source, e.g.,
            "Single-cell dataset from cellxgene link". Default is "Unknown data source".

        Returns
        -------
        None
            This method updates the `adata.obsm` in place by storing the final
            data embedding in `adata.obsm[self.data_obsm_key_final]`.

        Raises
        ------
        ValueError
            If "data_encoder" is not found in self.encoders.
        """
        logger.info(f"Generating initial data embeddings. Source: {data_source_info}")

        # 1) Create initial embeddings using a user-supplied embedder
        initial_data_emb = data_embedder.embed(adata)
        if initial_data_emb is None:
            msg = "No initial data embeddings found. Please check the embedder."
            logger.error(msg)
            raise ValueError(msg)

        # 2) Retrieve the 'data_encoder' from encoders
        if "data_encoder" not in self.encoders:
            msg = "No 'data_encoder' found in self.encoders."
            logger.error(msg)
            raise ValueError(msg)

        data_encoder = self.encoders["data_encoder"]

        # 3) Forward pass through the final data encoder
        logger.info("Passing initial data embeddings through the data encoder.")
        initial_data_emb = torch.tensor(initial_data_emb, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            # The forward signature depends on your model's implementation.
            # Example: model returns an embedding plus some extra output
            final_data_emb, _ = data_encoder.forward(initial_data_emb.unsqueeze(0))
            final_data_emb = final_data_emb.squeeze(0).cpu().numpy()

        # 4) Store the final data embeddings
        adata.obsm[self.data_obsm_key_final] = final_data_emb
        logger.info(f"Final data embeddings stored in adata.obsm['{self.data_obsm_key_final}'].")

    def encode_context(
        self,
        adata: anndata.AnnData,
        context_embedder: ContextEmbedder,
        context_source_info: str = "Unknown context source",
    ) -> None:
        """
        Embed context in two stages: initial embedder + final context encoder.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object containing context data, such as text-based annotations.
        context_embedder : Callable[[anndata.AnnData], None]
            A function or object that, when called, produces an initial context
            embedding stored in `adata.obsm[self.context_obsm_key_init]`.
        context_source_info : str, optional
            Description of the context data source, e.g.,
            "Text data from PubMed abstracts". Default is "Unknown context source".

        Returns
        -------
        None
            This method updates the `adata.obsm` in place by storing the final
            context embedding in `adata.obsm[self.context_obsm_key_final]`.

        Raises
        ------
        ValueError
            If "context_encoder" is not found in self.encoders.
        """
        logger.info(f"Generating initial context embeddings. Source: {context_source_info}")

        # 1) Create initial embeddings using a user-supplied embedder
        initial_context_emb = context_embedder.embed(adata)
        if initial_context_emb is None:
            msg = "No initial context embeddings found. Please check the embedder."
            logger.error(msg)
            raise ValueError(msg)

        # 2) Retrieve the 'context_encoder' from encoders
        if "context_encoder" not in self.encoders:
            msg = "No 'context_encoder' found in self.encoders."
            logger.error(msg)
            raise ValueError(msg)

        context_encoder = self.encoders["context_encoder"]

        # 3) Forward pass through the final context encoder
        logger.info("Passing initial context embeddings through the context encoder.")
        initial_context_emb = torch.tensor(initial_context_emb, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            # The forward signature depends on your model's implementation.
            # Example: model returns an embedding plus some extra output
            final_context_emb, _ = context_encoder.forward(initial_context_emb.unsqueeze(0))
            final_context_emb = final_context_emb.squeeze(0).cpu().numpy()

        # 4) Store the final context embeddings
        adata.obsm[self.context_obsm_key_final] = final_context_emb
        logger.info(f"Final context embeddings stored in adata.obsm['{self.context_obsm_key_final}'].")
