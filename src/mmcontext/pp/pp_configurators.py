from omegaconf import DictConfig

from mmcontext.pp import (
    AnnDataStoredEmbedder,
    CategoryEmbedder,
    ContextEmbedder,
    DataEmbedder,
    MinMaxNormalizer,
    PCAReducer,
    PlaceHolderNormalizer,
    ZScoreNormalizer,
)


def configure_embedder(cfg: DictConfig) -> tuple[DataEmbedder | None, ContextEmbedder | None, str | None, str | None]:
    """Pass the cfg.embedder subset of the main cfg object to this function to configure the data and context embedders.

    Parameters
    ----------
    cfg
        The configuration object. It should contain only the embedder subset.
    """
    if cfg.context_embedder.type == "categorical":
        # Initialize the CategoryEmbedder
        context_embedder = CategoryEmbedder(
            metadata_categories=cfg.context_embedder.specs.metadata_categories,
            model=cfg.context_embedder.specs.model,
            combination_method=cfg.context_embedder.specs.combination_method,
            embeddings_file_path=cfg.context_embedder.specs.embeddings_file_path,
            one_hot=cfg.context_embedder.specs.one_hot,
        )
    else:
        raise ValueError(
            f"Invalid context embedder class: {cfg.context_embedder.type}. Only 'categorical' is supported."
        )

    if cfg.data_embedder.type == "precalculated":
        data_embedder = AnnDataStoredEmbedder(obsm_key=cfg.data_embedder.specs.precalculated_obsm_key)
    else:
        raise ValueError(f"Invalid data embedder class: {cfg.data_embedder.type}. Only 'precalculated' is supported.")
    return data_embedder, context_embedder


def configure_aligner(cfg: DictConfig):
    """
    Configures a dimension aligner based on the provided configuration.

    Parameters
    ----------
    cfg
        The configuration object. It should contain only the aligner subset.

    Returns
    -------
    DimAligner
        The configured dimension aligner.
    """
    aligner_type = cfg.type
    if aligner_type in ["PCA", "pca"]:
        return PCAReducer(
            latent_dim=cfg.latent_dim,
            max_samples=cfg.max_samples,
            random_state=cfg.random_state,
            config=cfg.additional.pca_eval,
        )
    else:
        raise ValueError(f"Invalid dimension aligner type: {aligner_type}")


def configure_normalizer(cfg):
    """Configures the normalizer based on the configuration.

    Parameters
    ----------
    cfg
        The configuration object.

    Returns
    -------
    EmbeddingNormalizer
        The configured normalizer.
    """
    if cfg.type == "z-score":
        return ZScoreNormalizer()
    elif cfg.type == "min-max":
        return MinMaxNormalizer()
    elif cfg.type in ["none", "None"]:
        return PlaceHolderNormalizer()
    else:
        raise ValueError(f"Unknown normalizer type: {cfg.type}")
