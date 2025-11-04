from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import anndata as ad
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mmcontext.mmcontextencoder import MMContextEncoder as MMEnc

logger = logging.getLogger(__name__)


@dataclass
class SentenceDataset(Dataset):
    """
    A minimal Dataset that returns (index, text) tuples needed for batching.

    Parameters
    ----------
    indices : list[int] | list[str]
        Unique sample identifiers.
    texts : list[str]
        Sentences or tokenised strings.
    """

    indices: list[int]
    texts: list[str]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx], self.texts[idx]


class HFIndexedDataset(torch.utils.data.Dataset):
    """
    Class to create a dataset that is used to embed the data.

    A lightweight view on top of a 🤗 Dataset that returns
    ``(sample_index, text)`` tuples.

    Parameters
    ----------
    ds : datasets.Dataset
        The underlying dataset.
    main_col : str
        Name of the column containing the raw sentence.
    index_col : str
        Name of the identifier column (kept for the final join).
    """

    def __init__(self, ds: Dataset, main_col: str, index_col: str):
        self.ds = ds
        self.main_col = main_col
        self.index_col = index_col

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[int(idx)]
        return row[self.index_col], row[self.main_col]


def load_st_model(model_id: str | Path) -> SentenceTransformer:
    """Load a Sentence-Transformer model from HF Hub or local path with GPU support."""
    import os

    model_id = str(model_id)
    logger.info("Loading model %s", model_id)

    # Check device availability in worker process
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"CUDA available in worker process {os.getpid()}: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        logger.warning(f"CUDA not available in worker process {os.getpid()}, using CPU")

    # Load model and explicitly move to device
    model = SentenceTransformer(model_id, device=device, trust_remote_code=True)
    logger.info(f"Model loaded on device: {model.device}")

    return model


def prepare_model_and_embed(
    st_model: SentenceTransformer,
    data: HFDataset,
    *,
    layer_key: str = None,
    adata_download_dir: str | Path = None,
    main_col: str = "cell_sentence_1",
    index_col: str = "sample_index",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = torch.cuda.is_available(),
    axis: Literal["var", "obs"] = "obs",
    text_only: bool = False,
    overwrite: bool = True,
) -> tuple[pd.DataFrame, dict[str, Path] | None]:
    """
    Create embeddings using a Sentence-Transformer model, with some model-specific preparation.

    Prepare a customised Sentence-Transformer model *if possible* and obtain
    sentence embeddings in mini-batches.

    Parameters
    ----------
    st_model : SentenceTransformer
        The model wrapper.  The *first* module (``st_model[0]``) may expose
        extra methods such as ``prefix_ds``.
    df_subset : DataFrame
        Test-set slice (≤ 5 k rows by default) **must** contain
        ``main_col`` and ``index_col``.
    main_col : str
        Column containing the raw sentences.
    index_col : str, default ``"sample_index"``
        Column to keep as the identifier in the returned table.
    batch_size : int, default 32
        Number of sentences per encode() call.
    num_workers : int, default 0
        Worker processes for the PyTorch data loader.
    pin_memory : bool, default ``True`` if CUDA is available
        Forwarded to DataLoader for speed on GPUs.
    layer_key : str
        The name of the layer key to be used for the embedding.
    axis : str, default ``"obs"``
        The axis to be used for the embedding. Can be either ``"obs"`` or
        ``"var"``.
    text_only : bool, default ``False``
        If ``True``, all input will be treated as text. If "False", the dataset has to contain
        a column share_link, which points to a zarr store that can be used to get initial embeddings for each token.
    overwrite : bool, default ``True``
        If ``True``, the initial embeddings will be re-downloaded even if they already exist.

    Returns
    -------
    tuple[DataFrame, dict[str, Path] | None]
        A tuple containing:
        - DataFrame with columns ``[index_col, "embedding"]`` where *embedding* is a list[float].
        - Path mapping from original links to actual file locations (None if no numeric data)

    Notes
    -----
    * Optional methods are only executed if present.
    * ``token_df`` is **registered** with the model (if possible) but **not**
      returned to the caller.
    """
    if not text_only and (layer_key is None or adata_download_dir is None):
        raise ValueError("layer_key and adata_download_dir must be provided if text_only is False")
    # If using a multiplets dataset, the column with the data representation is called "anchor". For the embedding workflow, only this data is embedded.
    if "anchor" in data.column_names:
        main_col = "anchor"
    ###########################################################
    # --- optional model-specific preparation ----------------
    ###########################################################
    impl = st_model[0] if len(st_model) > 0 else st_model
    has_numeric = ("share_link" in data.column_names) or (
        "adata_link" in data.column_names
    )  # otherwise we cannot download the initial embeddings and register them with the model
    path_map = None
    if data is not None and has_numeric:
        link_column = "share_link" if "share_link" in data.column_names else "adata_link"
        # If the dataset is available, download get the token_df, even if the models doesnt use it. Just so the data is downloaded and the adata subset can be created downstream
        logger.info("Extracting numeric intital embeddings from dataset via it's share_links …")
        token_df, path_map = MMEnc.get_initial_embeddings_from_adata_link(
            data,
            layer_key=layer_key,
            axis=axis,
            download_dir=adata_download_dir,
            overwrite=overwrite,
            link_column=link_column,
        )  # type: ignore[arg-type]
    else:
        logger.info("""While the model supports initial embeddings, the dataset does not provide them.
                    Therefore cell sentences will be analysed as texts.""")
        token_df = None
        # register only if the method exists
    if hasattr(impl, "register_initial_embeddings") and token_df is not None and not text_only:
        logger.info("Registering initial embeddings with MMContextEncoder module of SentenceTransformer Model …")
        impl.register_initial_embeddings(token_df, data_origin=layer_key.split("X_")[1])  # type: ignore[arg-type]

    if hasattr(impl, "prefix_ds") and not text_only:
        logger.info("Calling prefix_ds on model-specific module …")
        # Step 4: Apply prefixes using the new simplified prefix_ds method
        logger.info(f"Applying prefixes to columns: {main_col}")
        ds = impl.prefix_ds(
            ds=data,
            columns_to_prefix=main_col,
        )  # type: ignore[arg-type]
    else:
        ds = data
    ###########################################################
    # --- sentence embeddings via encode() --------------------
    ###########################################################
    # -------- create a torch-compatible dataset ------------------------
    if isinstance(ds, pd.DataFrame):
        torch_ds = SentenceDataset(
            indices=ds[index_col].tolist(),
            texts=ds[main_col].astype(str).tolist(),
        )
    else:  # Hugging-Face Dataset
        torch_ds = HFIndexedDataset(ds, main_col=main_col, index_col=index_col)

    loader = DataLoader(
        torch_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    all_indices: list[int | str] = []
    all_embeddings: list[list[float]] = []

    st_model.eval()
    with torch.inference_mode():
        for batch_indices, batch_texts in tqdm(loader, desc="Encoding"):
            # DataLoader returns tuples → convert to list[str]
            embeddings = st_model.encode(
                list(batch_texts),
                batch_size=len(batch_texts),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            if isinstance(batch_indices, torch.Tensor):
                batch_indices = batch_indices.tolist()
            all_indices.extend(batch_indices)
            all_embeddings.extend(embeddings.tolist())

    # Create DataFrame using from_dict to avoid deprecation warning
    emb_df = pd.DataFrame.from_dict({"sample_idx": all_indices, "embedding": all_embeddings})
    logger.info("Generated %d embeddings", len(emb_df))
    return emb_df, path_map


def create_label_dataset(adata, label_col: str) -> tuple[HFDataset, dict[str, int]]:
    """
    Create a small HF Dataset from unique labels in an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the labels
    label_col : str
        Name of the column in adata.obs containing the labels

    Returns
    -------
    tuple[HFDataset, dict[str, int]]
        A tuple containing:
        - HuggingFace dataset with unique labels and their indices
        - Mapping from label string to embedding index
    """
    from datasets import Dataset

    # Get unique labels only
    unique_labels = adata.obs[label_col].astype(str).unique().tolist()

    # Create mapping from label to embedding index
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    # Create indices for the unique labels (just sequential)
    indices = list(range(len(unique_labels)))

    # Create dataset with unique labels only
    ds = Dataset.from_dict({"sample_idx": indices, "label": unique_labels})

    return ds, label_to_index


def embed_labels(
    st_model: SentenceTransformer,
    adata: ad.AnnData,
    label_col: str,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = torch.cuda.is_available(),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Embed unique labels from an AnnData object using a SentenceTransformer model.

    Parameters
    ----------
    st_model : SentenceTransformer
        The model to use for embedding
    adata : anndata.AnnData
        AnnData object containing the labels
    label_col : str
        Name of the column in adata.obs containing the labels
    batch_size : int, default 32
        Number of labels per encode() call
    num_workers : int, default 0
        Worker processes for the PyTorch data loader
    pin_memory : bool, default True if CUDA is available
        Forwarded to DataLoader for speed on GPUs

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int]]
        A tuple containing:
        - DataFrame with columns ['sample_idx', 'embedding'] containing unique label embeddings
        - Mapping from label string to embedding index in the DataFrame
    """
    # Create dataset from unique labels
    label_ds, label_to_index = create_label_dataset(adata, label_col)

    # Use prepare_model_and_embed with text_only=True since we're just embedding text labels
    emb_df, _ = prepare_model_and_embed(
        st_model,
        data=label_ds,
        main_col="label",
        index_col="sample_idx",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        text_only=True,
    )

    return emb_df, label_to_index
