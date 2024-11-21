## Main script for training a model

import os
import shutil
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mmcontext.engine import LossManager, Trainer, configure_models, configure_optimizer, configure_scheduler
from mmcontext.pp import AnnDataObtainer, ConfigWorker, DataSetConstructor


@hydra.main(config_path="configs", config_name="train_config")
def main_train(cfg: DictConfig) -> None:
    """Main function for training a validating a model, based on a configuration file.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary for one training configuration
    """
    data_dir = Path(to_absolute_path(cfg.data.dir))
    # Configure the obtainer
    obtainer = AnnDataObtainer(cfg, backed=None)
    # Configure the config worker
    cfg_worker = ConfigWorker(cfg=cfg, out_dir=data_dir / "model_out")
    # Update the config with settings used in preprocessing
    pp_settings_file = data_dir / "settings.yaml"
    pp_settings = cfg_worker.load_config(pp_settings_file)
    cfg = cfg_worker.transfer_settings(pp_settings)
    # Compute the hash for the training configuration
    hash_info = cfg_worker.compute_hash()
    # Save the current cfg file with the hash
    cfg_worker.check_and_save_config()
    # save all model outputs in a directory in the data directory it was computed on
    out_dir = data_dir / "model_out" / hash_info["hash"]

    n_genes = []
    datasets = {}
    for data_type in ["train", "test"]:
        dataset_constructor = DataSetConstructor(
            in_sample_id_key=cfg.data.sample_key, chunk_size=cfg.dataset.chunk_size
        )
        file_dir = data_dir / "data" / data_type
        filenames = os.listdir(file_dir)
        for filename in filenames:
            data_path = file_dir / filename
            adata = obtainer.get_data(data_path)
            n_genes.append(adata.shape[1])
            if len(set(n_genes)) > 1:
                raise ValueError("The number of genes in the datasets is not consistent.")
        dataset_constructor.add_anndata(adata)
        # Construct the dataset
        datasets[data_type] = dataset_constructor.construct_dataset(seq_length=cfg.dataset.seq_length)

    train_loader = DataLoader(datasets["train"], batch_size=cfg.dataset.batch_size, shuffle=False)
    # For now use the test data as validation data
    val_loader = DataLoader(datasets["test"], batch_size=cfg.dataset.batch_size, shuffle=False)

    # Load the models
    encoder, decoder = configure_models(
        cfg.engine.models, decoder_out_dim=n_genes[0]
    )  # output_dim is number of samples in a batch. Sequence dimension is not used for decoder
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
    # copy best weights from working directory to out_dir
    shutil.copy("best_encoder_weights.pth", out_dir / "best_encoder_weights.pth")
    shutil.copy("best_decoder_weights.pth", out_dir / "best_decoder_weights.pth")
    # Use the best model to predict the test data
    encoder.load(file_path=out_dir / "best_encoder_weights.pth")
    decoder.load(file_path=out_dir / "best_decoder_weights.pth")
    trainer = Trainer(
        encoder=encoder,
        decoder=decoder,
        input_embeddings=cfg.engine.trainer.input_embeddings,
        temperature=cfg.engine.trainer.temperature,
    )
    # get all files stored in the .out_test.dir and infer the model on them for evaluation
    # This is processed test data
    test_data_dir = data_dir / "data" / "test"
    for filename in os.listdir(test_data_dir):
        test_adata = obtainer.get_data(test_data_dir / filename)
        # Ensure the same genes are used for the test data
        inferred_adata = trainer.infer_adata(
            test_adata,
            sample_id_key=cfg.data.sample_key,
            seq_length=cfg.dataset.seq_length,
            batch_size=cfg.dataset.batch_size,
            chunk_size=cfg.dataset.chunk_size,
        )
        # remove keys from obs that contain "norm" as these are not very interesting
        for key in inferred_adata.obs.keys():
            if "norm" in key:
                del inferred_adata.obs[key]

        # Save the inferred data
        out_data_dir = out_dir / "data"
        os.makedirs(out_data_dir, exist_ok=True)
        inferred_adata.write_zarr(out_data_dir / f"inferred_{filename}")


if __name__ == "__main__":
    main_train()
