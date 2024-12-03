## Main script for training a model

import os
from pathlib import Path

import anndata
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mmcontext.engine import LossManager, Trainer, configure_models, configure_optimizer, configure_scheduler
from mmcontext.eval import Evaluator
from mmcontext.pl.plotting import plot_umap
from mmcontext.pp import (
    DataSetConstructor,
    Embedder,
    configure_aligner,
    configure_embedder,
    configure_normalizer,
    consolidate_low_frequency_categories,
    remove_entries,
)


@hydra.main(config_path="configs", config_name="config")
def main_train(cfg: DictConfig) -> None:
    """Main function for training a validating a model, based on a configuration file."""
    # Configure the Embedder
    data_embedder, context_embedder, data_embedding_key, context_embedding_key = configure_embedder(cfg.pp.embedder)
    # Initialize the Embedder
    embedder = Embedder(context_embedder=context_embedder, data_embedder=data_embedder)
    # Configure the Normalizer
    normalizer = configure_normalizer(cfg.pp.normalizer)
    # Configure the Aligner
    aligner = configure_aligner(cfg.pp.aligner)
    # The sequence length for the dataset. Since attention mechanisms will be used, samples can be grouped into sequences to perform attention within the sequence.
    seq_length = cfg.pp.dataset_constructor.seq_length
    batch_size = cfg.pp.dataset_constructor.batch_size
    # For training data
    data_dirs = {"train": cfg.data.train.subdir, "test": cfg.data.test.subdir}
    out_data_dirs = {"train": cfg.data.train.out_dir, "test": cfg.data.test.out_dir}
    # Store the number of genes to ensure consistency
    n_genes = []
    # make an empty dict with the same keys as data_dir
    datasets = {key: [] for key in data_dirs}
    for data_type in data_dirs.keys():
        data_dir = data_dirs[data_type]
        # Initialize the DataSetConstructor. Has to be reinstantiated dataset to be constructed
        dataset_constructor = DataSetConstructor(in_sample_id_key=cfg.data.sample_key)
        for filename in cfg.data.get(data_type).get("filenames"):
            data_path = Path(to_absolute_path(f"{data_dir}/{filename}"))
            adata = anndata.read_h5ad(data_path)
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
            # Save the prcocessed adata files
            # Potentially create the folder
            os.makedirs(Path(to_absolute_path(out_data_dirs[data_type])), exist_ok=True)
            save_path = Path(to_absolute_path(f"{out_data_dirs[data_type]}/{filename}"))
            adata.write_h5ad(save_path)
            # Add the AnnData object to the dataset
            dataset_constructor.add_anndata(adata)
            # if data_type == "train":
            #    train_gene_names.append(adata.var_names)
            # if data_type == "test":
            #    test_gene_names.append(adata.var_names)
        # collapse the train_gene_names list of lists into a single list
        # train_gene_names = [gene for sublist in train_gene_names for gene in sublist]
        # test_gene_names = [gene for sublist in test_gene_names for gene in sublist]
        # intersect of gene names, because we want to use the same genes for training and testing
        # intersect_gene_names = list(set(train_gene_names).intersection(set(test_gene_names)))
        # Construct the dataset
        datasets[data_type] = dataset_constructor.construct_dataset(seq_length=seq_length)

        # Turn into pytorch dataloaders
        if batch_size * seq_length > adata.n_obs:
            raise ValueError("Batch size and sequence length are too large for the dataset.")
        train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=False)
        # For now use the test data as validation data
        val_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)

    # Load the models
    encoder, decoder = configure_models(
        cfg.engine.models, decoder_out_dim=n_genes[0]
    )  # output_dim is number of samples in a batch. Sequence dimension is not used for secoder
    # Configure the loss manager
    loss_manager = LossManager()
    loss_manager.configure_losses(cfg.engine)
    # Get the optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = configure_optimizer(cfg.engine, params)
    # Get the scheduler
    scheduler = configure_scheduler(cfg.engine, optimizer)
    # Inituilize the trainer
    trainer = Trainer(
        encoder=encoder,
        decoder=decoder,
        loss_manager=loss_manager,
        optimizer=optimizer,
        scheduler=scheduler,
        input_embeddings=cfg.engine.trainer.input_embeddings,
        temperature=cfg.engine.trainer.temperature,
    )
    # Train the model
    trainer.fit(train_loader, val_loader, cfg.engine.trainer.epochs, save=True)

    # Use the best model to predict the test data
    encoder.load(file_path="encoder_weights.pth")
    decoder.load(file_path="decoder_weights.pth")
    trainer = Trainer(
        encoder=encoder,
        decoder=decoder,
        input_embeddings=cfg.engine.trainer.input_embeddings,
        temperature=cfg.engine.trainer.temperature,
    )
    # get all files stored in the .out_test.dir and infer the model on them for evaluation
    # This is processed test data
    test_data_dir = Path(to_absolute_path(cfg.data.test.out_dir))
    for filename in os.listdir(test_data_dir):
        test_adata = anndata.read_h5ad(test_data_dir / filename)
        # Ensure the same genes are used for the test data
        inferred_adata = trainer.infer_adata(
            test_adata, sample_id_key="soma_joinid", seq_length=seq_length, batch_size=batch_size
        )
        inferred_adata.write_h5ad(f"inferred_{filename}")

        # Initialize Evaluator
        evaluator = Evaluator(
            adata=inferred_adata,
            batch_key=cfg.data.batch_key,
            label_key=cfg.data.cell_type_key,
            embedding_key=[data_embedding_key, "mod_emb"],
        )

        res = evaluator.evaluate()
        # save results as csv
        res.to_csv(f"evaluation_{filename}.csv")
        # plot umap
        # Remove low frequency categories for better visualization
        inferred_adata = consolidate_low_frequency_categories(
            inferred_adata, columns=[cfg.data.cell_type_key], threshold=10, remove=True
        )
        for emb_key in ["mod_emb", data_embedding_key]:
            plot_umap(
                inferred_adata,
                color_group=cfg.data.cell_type_key,
                embedding_key=emb_key,
                save_plot=True,
                save_dir="figs/",
                nametag=emb_key,
            )


if __name__ == "__main__":
    main_train()
