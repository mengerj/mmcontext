# tests/test_trainer.py

import logging

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from mmcontext.engine import ContrastiveLoss, LossManager, MMContextEncoder, ReconstructionLoss, Trainer
from mmcontext.pp import DataSetConstructor
from mmcontext.utils import create_test_emb_anndata


def create_test_dataloader(
    batch_size=4, seq_length=10, emb_dim=64, data_emb_key="data_embedding", context_emb_key="context_embedding"
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
    adata1 = create_test_emb_anndata(n_samples=100, emb_dim=emb_dim)
    adata2 = create_test_emb_anndata(n_samples=20, emb_dim=emb_dim, sample_ids=np.arange(100, 120))

    dataset_constructor = DataSetConstructor()
    dataset_constructor.add_anndata(adata1)
    dataset_constructor.add_anndata(adata2)

    dataset = dataset_constructor.construct_dataset(
        seq_length=seq_length, data_emb_key=data_emb_key, context_emb_key=context_emb_key
    )

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

    # Initialize model
    model = MMContextEncoder(
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
        target_mode="data_data", current_mode="data_context", similarity_metric="cosine"
    )
    contrastive_loss_fn2 = ContrastiveLoss(
        target_mode="context_context", current_mode="data_context", similarity_metric="cosine"
    )
    reconstruction_loss_fn = ReconstructionLoss(reduction="mean")

    loss_manager.add_loss(contrastive_loss_fn1, weight=1.0)
    loss_manager.add_loss(contrastive_loss_fn2, weight=1.0)
    loss_manager.add_loss(reconstruction_loss_fn, weight=0.5)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # display one batch of the data loader
    data_emb_key = "data_embedding"
    context_emb_key = "context_embedding"
    # Initialize Trainer with all required embeddings
    trainer = Trainer(
        model=model,
        loss_manager=loss_manager,
        optimizer=optimizer,
        device=torch.device("cpu"),
        input_embeddings={"main": data_emb_key, "cross": context_emb_key},
        data_emb_key=data_emb_key,
        context_emb_key=context_emb_key,
    )

    # Perform one training epoch

    train_loss = trainer.train_epoch(data_loader)
    assert train_loss >= 0, "Training loss should be non-negative."


def test_trainer_data_context_missing_cross():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_trainer_data_context_missing_cross")
    data_loader = create_test_dataloader()

    # Create a simplified model and loss for the test
    model = MMContextEncoder(embedding_dim=16, hidden_dim=16)
    loss_manager = LossManager()
    loss_manager.add_loss(ContrastiveLoss(current_mode="data_context", target_mode="context_context"), weight=1.0)

    # Initialize Trainer without 'cross' embedding
    with pytest.raises(ValueError):
        trainer = Trainer(
            model=model,
            loss_manager=loss_manager,
            optimizer=torch.optim.Adam(model.parameters()),
            input_embeddings={"main": "data_embedding"},  # Only 'main' provided
        )
        trainer.train_epoch(data_loader)


def test_trainer_default_values():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_trainer_default_values")
    emb_dim = 64
    data_loader = create_test_dataloader(emb_dim=emb_dim)

    # Model and loss setup
    model = MMContextEncoder(embedding_dim=emb_dim)
    loss_manager = LossManager()
    loss_manager.add_loss(ContrastiveLoss())
    # Initialize Trainer with default settings
    trainer = Trainer(model=model, loss_manager=loss_manager, optimizer=torch.optim.Adam(model.parameters()))

    train_loss = trainer.train_epoch(data_loader)
    assert train_loss >= 0, "Training loss should be non-negative."


def test_trainer_evaluation():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_trainer_evaluation")
    emb_dim = 64
    eval_loader = create_test_dataloader(emb_dim=emb_dim)

    # Model and loss setup
    model = MMContextEncoder(embedding_dim=emb_dim)
    loss_manager = LossManager()
    loss_manager.add_loss(ContrastiveLoss())
    # Trainer initialization
    trainer = Trainer(model=model, loss_manager=loss_manager, optimizer=torch.optim.Adam(model.parameters()))

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
    model = MMContextEncoder(embedding_dim=emb_dim, hidden_dim=16)
    loss_manager = LossManager()
    loss_manager.add_loss(ContrastiveLoss())
    # Trainer initialization
    trainer = Trainer(model=model, loss_manager=loss_manager, optimizer=torch.optim.Adam(model.parameters()))
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
    model = MMContextEncoder(embedding_dim=emb_dim, hidden_dim=16)
    loss_manager = LossManager()
    loss_manager.add_loss(ContrastiveLoss())
    # Trainer initialization
    trainer = Trainer(model=model, loss_manager=loss_manager, optimizer=torch.optim.Adam(model.parameters()))

    # Define save path
    save_path = tmp_path / "model.pth"
    # Run training to have something to save
    trainer.fit(train_loader, val_loader, epochs=1, save_path=str(save_path))

    # Check if file exists
    assert save_path.exists(), "Model file was not saved."

    # Load and verify
    new_model = MMContextEncoder(embedding_dim=16, hidden_dim=16)
    new_model.load(save_path)
    assert new_model, "Failed to load the saved model."

    # use the new model and compare the loss with the old model
    new_trainer = Trainer(
        model=new_model, loss_manager=loss_manager, optimizer=torch.optim.Adam(new_model.parameters())
    )
    val_loss1 = trainer.evaluate(val_loader)
    val_loss2 = new_trainer.evaluate(val_loader)
    assert val_loss1 == val_loss2, "Validation loss should be the same when reloading model and using same data."


def test_trainer_predictions():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_trainer_inference")
    emb_dim = 16
    seq_length = 10
    predict_loader = create_test_dataloader(emb_dim=emb_dim, seq_length=seq_length)
    # check contents of predict loader
    for batch in predict_loader:
        logger.info(f"predict loader: {batch.keys()}")
        break
    # Model and loss setup
    model = MMContextEncoder(embedding_dim=emb_dim, hidden_dim=16)

    # Trainer initialization
    trainer = Trainer(model=model, optimizer=torch.optim.Adam(model.parameters()))

    predictions = trainer.infer(predict_loader)
    assert predictions, "No predictions returned."
    assert "data_embedding" in predictions, "Predictions missing 'data_embedding'."
    assert "context_embedding" in predictions, "Predictions missing 'context_embedding'."
    assert "sample_id" in predictions, "Predictions missing 'sample_id'."
    # Check shape of predictions
    assert (
        predictions["data_embedding"].shape[1] == seq_length
    ), "Incorrect sequence length for predicted 'data_embedding'."
    assert predictions["data_embedding"].shape[2] == emb_dim, "Incorrect embedding dim for predicted 'data_embedding'."
    assert (
        predictions["context_embedding"].shape[1] == seq_length
    ), "Incorrect sequence length for predicted 'context_embedding'."
    assert (
        predictions["context_embedding"].shape[2] == emb_dim
    ), "Incorrect embedding for predicted 'context_embedding'."
    # compare total number of samples in data loader and predictions
    total_samples = 0
    for batch in predict_loader:
        total_samples += batch["data_embedding"].shape[0] * batch["data_embedding"].shape[1]
    assert (
        predictions["data_embedding"].shape[0] * predictions["data_embedding"].shape[1] == total_samples
    ), "Incorrect number of samples in predicted 'data_embedding'."


def test_trainer_custom_dict_keys():
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_trainer_default_values")
    emb_dim = 64
    data_loader = create_test_dataloader(emb_dim=emb_dim, data_emb_key="data_emb", context_emb_key="context_emb")

    # Model and loss setup
    model = MMContextEncoder(embedding_dim=emb_dim)
    loss_manager = LossManager()
    loss_manager.add_loss(ContrastiveLoss(data_emb_key="data_emb", context_emb_key="context_emb"))
    # Initialize Trainer with default settings
    trainer = Trainer(
        model=model,
        loss_manager=loss_manager,
        optimizer=torch.optim.Adam(model.parameters()),
        data_emb_key="data_emb",
        context_emb_key="context_emb",
    )

    train_loss = trainer.train_epoch(data_loader)
    assert train_loss >= 0, "Training loss should be non-negative."
