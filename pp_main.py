import os
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="pp_config")
def pp_main(cfg: DictConfig):
    """The preprocessing pipeline, which takes files from folders defined in the config, processes them and stores the processed files.

    The newly saved objects contain dask arrays which allow to construct a memory efficient dataset. They are saved in zarr format.
    """
    from mmcontext.pp import (
        AnnDataObtainer,
        ConfigWorker,
        Embedder,
        configure_aligner,
        configure_embedder,
        configure_normalizer,
        consolidate_low_frequency_categories,
        get_chunk_size,
        remove_entries,
    )

    # Define the output directory
    out_dir = Path(to_absolute_path("out/processed"))  # save in project root
    # Compute the chunk size based on the chosen seq length and batch size
    cfg.dataset.chunk_size = get_chunk_size(cfg)
    # Hash the preprocessing conditions
    cfg_worker = ConfigWorker(cfg=cfg, out_dir=out_dir)
    hash_info = cfg_worker.compute_hash()
    cfg_worker.check_and_save_config()
    # Configure the obtainer
    obtainer = AnnDataObtainer(cfg, backed=None)
    # Configure the Embedder
    data_embedder, context_embedder, data_embedding_key, context_embedding_key = configure_embedder(cfg.pp.embedder)
    # Initialize the Embedder
    embedder = Embedder(context_embedder=context_embedder, data_embedder=data_embedder)
    # Configure the Normalizer
    normalizer = configure_normalizer(cfg.pp.normalizer)
    # Configure the Aligner
    cfg.pp.additional.pca_eval.save_path = out_dir / hash_info["hash"] / "figs" / "pca_eval"
    aligner = configure_aligner(cfg.pp.aligner, cfg.pp.additional)
    # For training data
    data_dirs = {"train": cfg.data.train.subdir, "test": cfg.data.test.subdir}
    # out_data_dirs = {"train": cfg.data.train.out_dir, "test": cfg.data.test.out_dir}
    # Store the number of genes to ensure consistency
    n_genes = []
    for data_type in data_dirs.keys():
        data_dir = data_dirs[data_type]
        for filename in cfg.data.get(data_type).get("filenames"):
            data_path = Path(to_absolute_path(f"{data_dir}/{filename}"))
            adata = obtainer.get_data(data_path)
            # Ensure the same genes are used for the test data
            # remove cells with less than 10 appearances
            adata = consolidate_low_frequency_categories(
                adata,
                columns=cfg.pp.general.categories,
                threshold=cfg.pp.general.threshold,
                remove=cfg.pp.general.remove,
            )
            remove_entries(adata)
            # make sure the number of genes is consistent across all datasets
            n_genes.append(adata.shape[1])
            if len(set(n_genes)) > 1:
                raise ValueError("The number of genes in the datasets is not consistent.")
            data_embeddings = adata.obsm[data_embedding_key] if data_embedding_key else None
            context_embeddings = adata.obsm[context_embedding_key] if context_embedding_key else None
            # Create the embeddings or use precalculated embeddings
            embedder.create_embeddings(adata, data_embeddings=data_embeddings, context_embeddings=context_embeddings)
            # Normalize the embeddings
            normalizer.normalize(adata)
            # Align the dimensions of the embeddings
            aligner.align(adata)
            # remove file ending from filename
            filename = filename.split(".")[0]
            # Save the preprocessed adata files
            out_file = out_dir / hash_info["hash"] / "data" / data_type / f"{filename}.zarr"
            os.makedirs(out_file.parent, exist_ok=True)
            adata.write_zarr(out_file, chunks=[adata.shape[0], cfg.dataset.chunk_size])


if __name__ == "__main__":
    pp_main()
