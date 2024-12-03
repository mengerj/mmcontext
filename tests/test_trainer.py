# tests/test_trainer.py

import logging

import anndata
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from mmcontext.engine import (
    ContrastiveLoss,
    LossManager,
    MMContextEncoder,
    ReconstructionLoss,
    Trainer,
    ZINBDecoder,
    ZINBLoss,
)
from mmcontext.pp import DataSetConstructor
from mmcontext.utils import create_test_emb_anndata


def create_test_dataloader(
    batch_size=4, seq_length=10, emb_dim=64, n_samples=100, data_key="d_emb", context_key="c_emb"
):
    """
    Creates a dummy DataLoader using random embeddings for testing.

    Args:
        batch_size (int): Number of sequences per batch.
        seq_length (int): Length of each sequence.
        emb_dim (int): Dimension of the embeddings.

    Returns:
        DataLoader: A DataLoader with random embeddings.
    """
    adata1 = create_test_emb_anndata(n_samples=n_samples, emb_dim=emb_dim)
    adata2 = create_test_emb_anndata(n_samples=20, emb_dim=emb_dim, sample_ids=np.arange(n_samples, 120))

    dataset_constructor = DataSetConstructor(
        out_emb_keys={"data_embedding": data_key, "context_embedding": context_key},
        batch_size=batch_size,
        chunk_size=seq_length * batch_size,
        use_raw=True,
    )
    dataset_constructor.add_anndata(
        adata1,
        sample_id_key="sample_id",
        emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
    )
    dataset_constructor.add_anndata(
        adata2,
        sample_id_key="sample_id",
        emb_keys={"data_embedding": "d_emb_aligned", "context_embedding": "c_emb_aligned"},
    )

    dataset = dataset_constructor.construct_dataset(seq_length=seq_length)

    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader


def test_trainer_successful_training():
    """Test that the Trainer can successfully train when all required embeddings are provided."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_trainer_successful_training")
    batch_size = 2
    seq_length = 4
    embedding_dim = 16

    # Create DataLoader
    data_loader = create_test_dataloader(batch_size, seq_length, emb_dim=embedding_dim)

    # Initialize encoder
    encoder = MMContextEncoder(
        embedding_dim=embedding_dim,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        use_self_attention=True,
        use_cross_attention=True,
        activation="relu",
        dropout=0.1,
    )

    # Initialize LossManager and add losses
    loss_manager = LossManager()
    contrastive_loss_fn1 = ContrastiveLoss(
        target_mode="data_data",
        current_mode="data_context",
        similarity_metric="cosine",
        data_key="d_emb",
        context_key="c_emb",
    )
    contrastive_loss_fn2 = ContrastiveLoss(
        target_mode="context_context",
        current_mode="data_context",
        similarity_metric="cosine",
        data_key="d_emb",
        context_key="c_emb",
    )
    reconstruction_loss_fn = ReconstructionLoss(reduction="mean")

    loss_manager.add_loss(contrastive_loss_fn1, weight=1.0)
    loss_manager.add_loss(contrastive_loss_fn2, weight=1.0)
    loss_manager.add_loss(reconstruction_loss_fn, weight=0.5)

    # Initialize optimizer
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    # display one batch of the data loader
    data_key = "d_emb"
    context_key = "c_emb"
    # Initialize Trainer with all required embeddings
    trainer = Trainer(
        encoders=encoder,
        loss_manager=loss_manager,
        optimizer=optimizer,
        device=torch.device("cpu"),
        encoder_inputs={"data_encoder": {"in_main": data_key, "in_cross": context_key}},
    )

    # Perform one training epoch

    train_loss = trainer.train_epoch(data_loader)
    assert train_loss >= 0, "Training loss should be non-negative."


def test_trainer_data_context_missing_cross():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_trainer_data_context_missing_cross")
    data_loader = create_test_dataloader()

    # Create a simplified encoder and loss for the test
    encoder = MMContextEncoder(embedding_dim=16, hidden_dim=16)
    loss_manager = LossManager()
    loss_manager.add_loss(ContrastiveLoss(current_mode="data_context", target_mode="context_context"), weight=1.0)

    # Initialize Trainer without 'cross' embedding
    with pytest.raises(KeyError):
        trainer = Trainer(
            encoders=encoder,
            loss_manager=loss_manager,
            optimizer=torch.optim.Adam(encoder.parameters()),
            encoder_inputs={"data_encoder": {"in_main": "d_emb"}},  # Only 'main' provided
        )
        trainer.train_epoch(data_loader)


def test_trainer_default_values():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_trainer_default_values")
    emb_dim = 64
    data_loader = create_test_dataloader(emb_dim=emb_dim)

    # Model and loss setup
    encoder = MMContextEncoder(embedding_dim=emb_dim)
    loss_manager = LossManager()
    loss_manager.add_loss(ContrastiveLoss())
    # Initialize Trainer with default settings
    trainer = Trainer(encoders=encoder, loss_manager=loss_manager, optimizer=torch.optim.Adam(encoder.parameters()))

    train_loss = trainer.train_epoch(data_loader)
    assert train_loss >= 0, "Training loss should be non-negative."


def test_trainer_evaluation():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_trainer_evaluation")
    emb_dim = 64
    eval_loader = create_test_dataloader(emb_dim=emb_dim)

    # Model and loss setup
    encoder = MMContextEncoder(embedding_dim=emb_dim)
    loss_manager = LossManager()
    loss_manager.add_loss(ContrastiveLoss())
    # Trainer initialization
    trainer = Trainer(encoders=encoder, loss_manager=loss_manager, optimizer=torch.optim.Adam(encoder.parameters()))

    val_loss = trainer.evaluate(eval_loader)
    assert val_loss >= 0, "Validation loss should be non-negative."
    val_loss2 = trainer.evaluate(eval_loader)
    assert val_loss == val_loss2, "Validation loss should be the same for the same data."


def test_trainer_fit_method():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_trainer_fit_method")
    emb_dim = 16
    train_loader = create_test_dataloader(batch_size=5, seq_length=2, emb_dim=emb_dim)
    val_loader = create_test_dataloader(batch_size=5, seq_length=2, emb_dim=emb_dim)

    # Model and loss setup
    encoder = MMContextEncoder(embedding_dim=emb_dim, hidden_dim=16)
    loss_manager = LossManager()
    loss_manager.add_loss(ContrastiveLoss())
    # Trainer initialization
    trainer = Trainer(encoders=encoder, loss_manager=loss_manager, optimizer=torch.optim.Adam(encoder.parameters()))
    val_loss1 = trainer.evaluate(val_loader)
    trainer.fit(train_loader, val_loader, epochs=2)
    val_loss2 = trainer.evaluate(val_loader)
    # assert that losses are different
    assert val_loss1 != val_loss2, "Validation loss should change after training."
    assert trainer  # Simple check to ensure trainer object exists post fit


def test_trainer_save_load_weights(tmp_path):
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_trainer_save_load_weights")
    emb_dim = 16
    train_loader = create_test_dataloader(emb_dim=emb_dim)
    val_loader = create_test_dataloader(emb_dim=emb_dim)

    # Model and loss setup
    encoder = MMContextEncoder(embedding_dim=emb_dim, hidden_dim=16)
    loss_manager = LossManager()
    loss_manager.add_loss(ContrastiveLoss())
    # Trainer initialization
    trainer = Trainer(encoders=encoder, loss_manager=loss_manager, optimizer=torch.optim.Adam(encoder.parameters()))

    # Define save path
    save_dir = tmp_path
    # Run training to have something to save
    trainer.fit(train_loader, val_loader, epochs=1, save=True, save_dir=save_dir)
    default_encoder_name = "data_encoder"
    save_path_enc = save_dir / "best_encoder_weights" / f"best_{default_encoder_name}_weights.pth"
    # Check if file exists
    assert save_path_enc.exists(), "Enocder file was not saved."
    # Load and verify
    new_encoder = MMContextEncoder(embedding_dim=16, hidden_dim=16)
    new_encoder.load(save_path_enc)
    assert new_encoder, "Failed to load the saved encoder."

    # use the new encoder and compare the loss with the old encoder
    new_trainer = Trainer(
        encoders=new_encoder, loss_manager=loss_manager, optimizer=torch.optim.Adam(new_encoder.parameters())
    )
    val_loss1 = trainer.evaluate(val_loader)
    val_loss2 = new_trainer.evaluate(val_loader)
    assert val_loss1 == val_loss2, "Validation loss should be the same when reloading encoder and using same data."


def test_trainer_predictions(tmp_path):
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_trainer_predictions")
    emb_dim = 16
    seq_length = 10
    # Model and loss setup
    encoder = MMContextEncoder(embedding_dim=emb_dim, hidden_dim=16)
    # Trainer initialization
    trainer = Trainer(encoders=encoder, optimizer=torch.optim.Adam(encoder.parameters()))
    # generate test adata
    adata = create_test_emb_anndata(n_samples=100, emb_dim=emb_dim, data_key="d_emb", context_key="c_emb")
    trainer.infer_adata(
        adata,
        sample_id_key="sample_id",
        seq_length=seq_length,
        batch_size=6,
        chunk_size=6 * seq_length,
        output_zarr_path=f"{tmp_path}/test.zarr",
        in_emb_keys={"data_embedding": "d_emb", "context_embedding": "c_emb"},
    )
    new_adata = anndata.read_zarr(f"{tmp_path}/test.zarr")
    assert new_adata, "No predictions returned."
    assert "data_encoder_mod_emb" in new_adata.obsm, "Predictions missing 'mod_emb'."


def test_trainer_predictions_custom_keys(tmp_path):
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_trainer_predictions")
    emb_dim = 16
    seq_length = 10
    # Model and loss setup
    encoder = MMContextEncoder(embedding_dim=emb_dim, hidden_dim=16)
    # Trainer initialization
    trainer = Trainer(encoders=encoder, optimizer=torch.optim.Adam(encoder.parameters()))
    # generate test adata
    adata = create_test_emb_anndata(n_samples=100, emb_dim=emb_dim, data_key="data_emb", context_key="context_emb")
    trainer.infer_adata(
        adata,
        sample_id_key="sample_id",
        seq_length=seq_length,
        batch_size=6,
        chunk_size=6 * seq_length,
        output_zarr_path=f"{tmp_path}/test.zarr",
        out_emb_keys={"data_embedding": "d_emb", "context_embedding": "c_emb"},
        in_emb_keys={"data_embedding": "data_emb", "context_embedding": "context_emb"},
    )
    new_adata = anndata.read_zarr(f"{tmp_path}/test.zarr")
    assert new_adata, "No predictions returned."
    assert "data_encoder_mod_emb" in new_adata.obsm, "Predictions missing 'mod_emb'."


def test_trainer_custom_dict_keys():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_trainer_custom_dict_values")
    emb_dim = 64
    data_loader = create_test_dataloader(emb_dim=emb_dim, data_key="data_emb", context_key="context_emb")

    # Model and loss setup
    encoder = MMContextEncoder(embedding_dim=emb_dim)
    loss_manager = LossManager()
    loss_manager.add_loss(ContrastiveLoss(data_key="data_emb", context_key="context_emb"))
    # Initialize Trainer with default settings
    trainer = Trainer(
        encoders=encoder,
        loss_manager=loss_manager,
        optimizer=torch.optim.Adam(encoder.parameters()),
        encoder_inputs={"data_encoder": {"in_main": "data_emb", "in_cross": "context_emb"}},
    )

    train_loss = trainer.train_epoch(data_loader)
    assert train_loss >= 0, "Training loss should be non-negative."


def test_learnable_temperature():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_learnable_temperature")
    emb_dim = 16
    train_loader = create_test_dataloader(batch_size=5, seq_length=2, emb_dim=emb_dim)
    val_loader = create_test_dataloader(batch_size=5, seq_length=2, emb_dim=emb_dim)

    # Model and loss setup
    encoder = MMContextEncoder(embedding_dim=emb_dim, hidden_dim=16)
    loss_manager = LossManager()
    loss_manager.add_loss(ContrastiveLoss(target_mode="infoNCE"))
    # Trainer initialization
    trainer = Trainer(encoders=encoder, loss_manager=loss_manager, optimizer=torch.optim.Adam(encoder.parameters()))
    val_loss1 = trainer.evaluate(val_loader)
    trainer.fit(train_loader, val_loader, epochs=1)
    temp1 = trainer.temperature
    val_loss2 = trainer.evaluate(val_loader)
    trainer.fit(train_loader, val_loader, epochs=1)
    temp2 = trainer.temperature
    # assert that losses are different
    assert val_loss1 != val_loss2, "Validation loss should change after training."
    assert trainer  # Simple check to ensure trainer object exists post fit
    assert temp1 != temp2, "Temperature should change after training."


def test_fixed_temperature():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_fixed_temperature")
    emb_dim = 16
    train_loader = create_test_dataloader(batch_size=5, seq_length=2, emb_dim=emb_dim)
    val_loader = create_test_dataloader(batch_size=5, seq_length=2, emb_dim=emb_dim)

    # Model and loss setup
    encoder = MMContextEncoder(embedding_dim=emb_dim, hidden_dim=16)
    loss_manager = LossManager()
    loss_manager.add_loss(ContrastiveLoss(target_mode="infoNCE"))
    # Trainer initialization
    trainer = Trainer(
        encoders=encoder, loss_manager=loss_manager, optimizer=torch.optim.Adam(encoder.parameters()), temperature=0.1
    )
    val_loss1 = trainer.evaluate(val_loader)
    trainer.fit(train_loader, val_loader, epochs=1)
    temp1 = trainer.temperature
    val_loss2 = trainer.evaluate(val_loader)
    trainer.fit(train_loader, val_loader, epochs=1)
    temp2 = trainer.temperature
    # assert that losses are different
    assert val_loss1 != val_loss2, "Validation loss should change after training."
    assert trainer  # Simple check to ensure trainer object exists post fit
    assert temp1 == temp2 == 0.1, "Temperature should not change after training."


def test_fit_with_decoder():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_fit_with_decoder")
    emb_dim = 16
    n_samples = 100
    train_loader = create_test_dataloader(batch_size=5, seq_length=2, emb_dim=emb_dim, n_samples=n_samples)
    val_loader = create_test_dataloader(batch_size=5, seq_length=2, emb_dim=emb_dim, n_samples=n_samples)

    # Model and loss setup
    encoder = MMContextEncoder(embedding_dim=emb_dim, hidden_dim=16)
    decoder = ZINBDecoder(input_dim=emb_dim, hidden_dims=[16], output_dim=n_samples)
    loss_manager = LossManager()
    loss_manager.add_loss(ContrastiveLoss(target_mode="infoNCE"))
    loss_manager.add_loss(ZINBLoss())
    # Trainer initialization
    trainer = Trainer(
        encoders=encoder,
        decoder=decoder,
        loss_manager=loss_manager,
        optimizer=torch.optim.Adam(encoder.parameters()),
        temperature=0.1,
    )
    val_loss1 = trainer.evaluate(val_loader)
    trainer.fit(train_loader, val_loader, epochs=1)
    temp1 = trainer.temperature
    val_loss2 = trainer.evaluate(val_loader)
    trainer.fit(train_loader, val_loader, epochs=1)
    temp2 = trainer.temperature
    # assert that losses are different
    assert val_loss1 != val_loss2, "Validation loss should change after training."
    assert trainer  # Simple check to ensure trainer object exists post fit
    assert temp1 == temp2 == 0.1, "Temperature should not change after training."


def test_train_using_only_decoder():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_train_using_only_decoder")
    emb_dim = 16
    n_samples = 100
    train_loader = create_test_dataloader(
        batch_size=5, seq_length=2, emb_dim=emb_dim, n_samples=n_samples, data_key="mod_emb"
    )
    val_loader = create_test_dataloader(
        batch_size=5, seq_length=2, emb_dim=emb_dim, n_samples=n_samples, data_key="mod_emb"
    )

    # Model and loss setup
    decoder = ZINBDecoder(input_dim=emb_dim, hidden_dims=[16], output_dim=n_samples)
    loss_manager = LossManager()
    loss_manager.add_loss(ZINBLoss())
    # Trainer initialization
    trainer = Trainer(
        decoder=decoder,
        loss_manager=loss_manager,
        optimizer=torch.optim.Adam(decoder.parameters()),
        decoder_input_key="mod_emb",
    )
    val_loss1 = trainer.evaluate(val_loader)
    trainer.fit(train_loader, val_loader, epochs=1)
    val_loss2 = trainer.evaluate(val_loader)
    # assert that losses are different
    assert val_loss1 != val_loss2, "Validation loss should change after training."
    assert trainer  # Simple check to ensure trainer object exists post fit


def test_inference_only_decoder(tmp_path):
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_inference_only_decoder")
    emb_dim = 16
    n_samples = 100
    # Model and loss setup
    decoder = ZINBDecoder(input_dim=emb_dim, hidden_dims=[16], output_dim=n_samples)
    # Trainer initialization
    trainer = Trainer(decoder=decoder, optimizer=torch.optim.Adam(decoder.parameters()), decoder_input_key="mod_emb")
    # generate test adata
    adata = create_test_emb_anndata(n_samples=100, emb_dim=emb_dim, data_key="mod_emb")
    trainer.infer_adata(
        adata,
        sample_id_key="sample_id",
        seq_length=10,
        in_emb_keys={"data_embedding": "mod_emb"},
        out_emb_keys={"data_embedding": "mod_emb"},
        batch_size=6,
        chunk_size=6 * 10,
        output_zarr_path=f"{tmp_path}/test.zarr",
        n_recon=1,
    )
    new_adata = anndata.read_zarr(f"{tmp_path}/test.zarr")
    assert new_adata, "No predictions returned."
    assert "reconstructed1" in new_adata.layers, "Predictions missing 'mod_emb'."
