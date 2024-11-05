# tests/test_losses.py
import logging

import pytest
import torch
import torch.nn.functional as F

from mmcontext.engine.losses import ContrastiveLoss, LossManager, ReconstructionLoss


def create_random_embeddings(batch_size, seq_length, embedding_dim):
    return torch.randn(batch_size, seq_length, embedding_dim)


def create_identical_embeddings(batch_size, seq_length, embedding_dim):
    emb = torch.randn(batch_size, seq_length, embedding_dim)
    return emb.clone(), emb.clone()


def test_contrastive_loss_different_modes():
    """Test ContrastiveLoss with different combinations of target and current modes."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_contrastive_loss_different_modes")
    batch_size = 2
    seq_length = 3
    embedding_dim = 4

    # Create sample embeddings
    data_embeddings = create_random_embeddings(batch_size, seq_length, embedding_dim)
    context_embeddings = create_random_embeddings(batch_size, seq_length, embedding_dim)

    # Modified embeddings (outputs from model)
    modified_data_embeddings = data_embeddings + torch.randn_like(data_embeddings) * 0.1
    modified_context_embeddings = context_embeddings + torch.randn_like(context_embeddings) * 0.1

    # Prepare outputs and targets dictionaries
    outputs = {"data_embeddings": modified_data_embeddings, "context_embeddings": modified_context_embeddings}
    targets = {"data_embeddings": data_embeddings, "context_embeddings": context_embeddings}

    # Define different combinations of modes to test
    modes = [
        ("data_data", "data_data"),
        ("context_context", "context_context"),
        ("data_context", "data_context"),
        ("data_data", "data_context"),
        ("context_context", "data_data"),
        ("data_context", "context_context"),
        ("context_context", "data_context"),
    ]

    for target_mode, current_mode in modes:
        loss_fn = ContrastiveLoss(target_mode=target_mode, current_mode=current_mode, similarity_metric="cosine")
        try:
            loss = loss_fn.compute_loss(outputs, targets, data_key="data_embeddings", context_key="context_embeddings")
            assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
            logger.info(
                f"ContrastiveLoss with target_mode='{target_mode}', current_mode='{current_mode}' computed successfully with loss {loss.item()}"
            )
        except RuntimeError as e:
            pytest.fail(f"ContrastiveLoss failed with target_mode='{target_mode}', current_mode='{current_mode}': {e}")


def test_similarity_matrix_symmetry():
    """Test that the similarity matrix is symmetrical and has the expected shape for intra-sequence calculations."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_similarity_matrix_symmetry")
    batch_size = 2
    seq_length = 3
    embedding_dim = 4

    # Create embeddings
    embeddings_a = create_random_embeddings(batch_size, seq_length, embedding_dim)

    # Flatten embeddings
    embeddings_flat = embeddings_a.view(-1, embedding_dim)

    # Create loss function
    loss_fn = ContrastiveLoss(target_mode="data_data", current_mode="data_data", similarity_metric="cosine")
    # compute similarity matrix
    sim_matrix = loss_fn.get_similarity_matrix(
        embeddings_dict={"data_embeddings": embeddings_a}, mode="data_data", data_key="data_embeddings"
    )

    # Check symmetry
    assert torch.allclose(sim_matrix, sim_matrix.t(), atol=1e-6), "Similarity matrix is not symmetric"

    # Check shape
    expected_size = embeddings_flat.size(0)
    assert sim_matrix.shape == (
        expected_size,
        expected_size,
    ), f"Expected similarity matrix shape ({expected_size}, {expected_size}), got {sim_matrix.shape})"


def test_contrastive_loss_missing_modified_context_embeddings():
    """Test that ContrastiveLoss raises an error when current mode requires modified context embeddings but they are not provided."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_contrastive_loss_missing_modified_context_embeddings")
    batch_size = 2
    seq_length = 3
    embedding_dim = 4

    # Create sample embeddings
    data_embeddings = create_random_embeddings(batch_size, seq_length, embedding_dim)
    context_embeddings = create_random_embeddings(batch_size, seq_length, embedding_dim)

    # Modified embeddings (outputs from model), only data_embeddings
    modified_data_embeddings = data_embeddings + torch.randn_like(data_embeddings) * 0.1
    # Modified context embeddings are not provided
    # modified_context_embeddings = context_embeddings + torch.randn_like(context_embeddings) * 0.1

    # Prepare outputs and targets dictionaries
    outputs = {
        "data_embeddings": modified_data_embeddings,
        # 'context_embeddings': modified_context_embeddings  # Not provided
    }
    targets = {"data_embeddings": data_embeddings, "context_embeddings": context_embeddings}

    # Specify a mode that requires modified context embeddings in outputs
    target_mode = "data_data"
    current_mode = "data_context"  # Requires 'context_embeddings' in outputs

    loss_fn = ContrastiveLoss(target_mode=target_mode, current_mode=current_mode, similarity_metric="cosine")

    with pytest.raises(KeyError) as excinfo:
        loss_fn.compute_loss(outputs, targets, data_key="data_embeddings", context_key="context_embeddings")
    logger.info(f"Raised expected KeyError: {excinfo.value}")


def test_contrastive_loss_zero_when_embeddings_identical():
    """Test that the ContrastiveLoss is zero when modified embeddings are identical to the original embeddings."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_contrastive_loss_zero_when_embeddings_identical")
    batch_size = 2
    seq_length = 3
    embedding_dim = 4

    # Create identical embeddings
    data_embeddings = create_random_embeddings(batch_size, seq_length, embedding_dim)
    context_embeddings = create_random_embeddings(batch_size, seq_length, embedding_dim)

    # Modified embeddings are identical to the original embeddings
    modified_data_embeddings = data_embeddings.clone()
    modified_context_embeddings = context_embeddings.clone()

    # Prepare outputs and targets dictionaries
    outputs = {"data_embeddings": modified_data_embeddings, "context_embeddings": modified_context_embeddings}
    targets = {"data_embeddings": data_embeddings, "context_embeddings": context_embeddings}

    # Use any mode
    target_mode = "data_data"
    current_mode = "data_data"

    loss_fn = ContrastiveLoss(target_mode=target_mode, current_mode=current_mode, similarity_metric="cosine")
    loss = loss_fn.compute_loss(outputs, targets, data_key="data_embeddings", context_key="context_embeddings")
    assert torch.isclose(
        loss, torch.tensor(0.0), atol=1e-6
    ), f"Loss should be close to zero when embeddings are identical, got {loss.item()}"


def test_contrastive_loss_positive_when_embeddings_different():
    """Test that the ContrastiveLoss is positive when modified embeddings differ from the original embeddings."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_contrastive_loss_positive_when_embeddings_different")
    batch_size = 2
    seq_length = 3
    embedding_dim = 4

    # Create embeddings
    data_embeddings = create_random_embeddings(batch_size, seq_length, embedding_dim)
    context_embeddings = create_random_embeddings(batch_size, seq_length, embedding_dim)

    # Modified embeddings are different
    modified_data_embeddings = data_embeddings + torch.randn_like(data_embeddings) * 0.5
    modified_context_embeddings = context_embeddings + torch.randn_like(context_embeddings) * 0.5

    # Prepare outputs and targets dictionaries
    outputs = {"data_embeddings": modified_data_embeddings, "context_embeddings": modified_context_embeddings}
    targets = {"data_embeddings": data_embeddings, "context_embeddings": context_embeddings}

    # Use any mode
    target_mode = "data_data"
    current_mode = "data_data"

    loss_fn = ContrastiveLoss(target_mode=target_mode, current_mode=current_mode, similarity_metric="cosine")
    loss = loss_fn.compute_loss(outputs, targets, data_key="data_embeddings", context_key="context_embeddings")
    assert loss.item() > 0, f"Loss should be positive when embeddings differ, got {loss.item()}"


def test_reconstruction_loss():
    """Test that the ReconstructionLoss computes correctly."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_reconstruction_loss")

    batch_size = 2
    seq_length = 3
    embedding_dim = 4

    # Create original and modified embeddings
    data_embeddings = create_random_embeddings(batch_size, seq_length, embedding_dim)
    modified_data_embeddings = data_embeddings + torch.randn_like(data_embeddings) * 0.1

    # Prepare outputs and targets dictionaries
    outputs = {"data_embeddings": modified_data_embeddings}
    targets = {"data_embeddings": data_embeddings}

    # Initialize ReconstructionLoss
    loss_fn = ReconstructionLoss(reduction="mean")
    loss = loss_fn.compute_loss(outputs, targets, data_key="data_embeddings", context_key="context_embeddings")
    assert loss.item() >= 0, f"Reconstruction loss should be non-negative, got {loss.item()}"


def test_loss_manager_combines_losses():
    """Test that the LossManager correctly combines multiple losses."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_loss_manager_combines_losses")

    batch_size = 2
    seq_length = 3
    embedding_dim = 4

    # Create embeddings
    data_embeddings = create_random_embeddings(batch_size, seq_length, embedding_dim)
    context_embeddings = create_random_embeddings(batch_size, seq_length, embedding_dim)
    modified_data_embeddings = data_embeddings + torch.randn_like(data_embeddings) * 0.1
    modified_context_embeddings = context_embeddings + torch.randn_like(context_embeddings) * 0.1

    # Prepare outputs and targets
    outputs = {"data_embeddings": modified_data_embeddings, "context_embeddings": modified_context_embeddings}
    targets = {"data_embeddings": data_embeddings, "context_embeddings": context_embeddings}

    # Initialize LossManager and add losses
    loss_manager = LossManager(data_key="data_embeddings", context_key="context_embeddings")
    contrastive_loss_fn1 = ContrastiveLoss(
        target_mode="data_data", current_mode="data_data", similarity_metric="cosine"
    )
    contrastive_loss_fn2 = ContrastiveLoss(
        target_mode="context_context", current_mode="data_data", similarity_metric="cosine"
    )
    reconstruction_loss_fn = ReconstructionLoss(reduction="mean")

    loss_manager.add_loss(contrastive_loss_fn1, weight=1.0)
    loss_manager.add_loss(contrastive_loss_fn2, weight=1.0)
    loss_manager.add_loss(reconstruction_loss_fn, weight=0.5)

    # Compute total loss
    total_loss = loss_manager.compute_total_loss(outputs, targets)
    assert total_loss.item() >= 0, f"Total loss should be non-negative, got {total_loss.item()}"


def test_contrastive_loss_invalid_mode():
    """Test that ContrastiveLoss raises a ValueError for an unsupported mode."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_contrastive_loss_invalid_mode")
    batch_size = 2
    seq_length = 3
    embedding_dim = 4

    # Create embeddings
    data_embeddings = create_random_embeddings(batch_size, seq_length, embedding_dim)
    modified_data_embeddings = data_embeddings + torch.randn_like(data_embeddings) * 0.1

    # Prepare outputs and targets
    outputs = {"data_embeddings": modified_data_embeddings}
    targets = {"data_embeddings": data_embeddings}

    # Use an invalid mode
    target_mode = "invalid_mode"
    current_mode = "data_data"

    loss_fn = ContrastiveLoss(target_mode=target_mode, current_mode=current_mode, similarity_metric="cosine")

    with pytest.raises(ValueError) as excinfo:
        loss_fn.compute_loss(outputs, targets, data_key="data_embeddings", context_key="context_embeddings")
    logger.info(f"Raised expected ValueError: {excinfo.value}")


def test_contrastive_loss_invalid_similarity_metric():
    """Test that ContrastiveLoss raises a ValueError for an unsupported similarity metric."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_contrastive_loss_invalid_similarity_metric")

    batch_size = 2
    seq_length = 3
    embedding_dim = 4

    # Create embeddings
    data_embeddings = create_random_embeddings(batch_size, seq_length, embedding_dim)
    modified_data_embeddings = data_embeddings + torch.randn_like(data_embeddings) * 0.1

    # Prepare outputs and targets
    outputs = {"data_embeddings": modified_data_embeddings}
    targets = {"data_embeddings": data_embeddings}

    # Use an unsupported similarity metric
    target_mode = "data_data"
    current_mode = "data_data"
    similarity_metric = "unsupported_metric"

    loss_fn = ContrastiveLoss(target_mode=target_mode, current_mode=current_mode, similarity_metric=similarity_metric)

    with pytest.raises(ValueError) as excinfo:
        loss_fn.compute_loss(outputs, targets, data_key="data_embeddings", context_key="context_embeddings")
    logger.info(f"Raised expected ValueError: {excinfo.value}")


def test_contrastive_loss_mismatched_embedding_dims():
    """Test that ContrastiveLoss raises an error when embeddings have mismatched dimensions."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_contrastive_loss_mismatched_embedding_dims")
    batch_size = 2
    seq_length = 3
    embedding_dim_a = 4
    embedding_dim_b = 5  # Different embedding dimension

    # Create embeddings with different dimensions
    data_embeddings = create_random_embeddings(batch_size, seq_length, embedding_dim_a)
    context_embeddings = create_random_embeddings(batch_size, seq_length, embedding_dim_b)

    # Modified embeddings
    modified_data_embeddings = data_embeddings + torch.randn_like(data_embeddings) * 0.1
    modified_context_embeddings = context_embeddings + torch.randn_like(context_embeddings) * 0.1

    # Prepare outputs and targets
    outputs = {"data_embeddings": modified_data_embeddings, "context_embeddings": modified_context_embeddings}
    targets = {"data_embeddings": data_embeddings, "context_embeddings": context_embeddings}

    # Use a mode that requires matching dimensions
    target_mode = "data_context"
    current_mode = "data_context"

    loss_fn = ContrastiveLoss(target_mode=target_mode, current_mode=current_mode, similarity_metric="cosine")

    with pytest.raises(RuntimeError) as excinfo:
        loss_fn.compute_loss(outputs, targets, data_key="data_embeddings", context_key="context_embeddings")
    logger.info(f"Raised expected RuntimeError: {excinfo.value}")


def test_contrastive_loss_infoNCE():
    """Test that the ContrastiveLoss computes InfoNCE loss correctly when target_mode is 'infoNCE'."""
    logger = logging.getLogger(__name__)
    logger.info("TEST: test_contrastive_loss_infoNCE")
    batch_size = 2
    seq_length = 3
    embedding_dim = 4

    # Create random embeddings
    data_embeddings = create_random_embeddings(batch_size, seq_length, embedding_dim)
    context_embeddings = create_random_embeddings(batch_size, seq_length, embedding_dim)

    # Modified embeddings (outputs from model)
    modified_data_embeddings = data_embeddings + torch.randn_like(data_embeddings) * 0.1
    modified_context_embeddings = context_embeddings + torch.randn_like(context_embeddings) * 0.1

    # Prepare outputs and targets dictionaries
    outputs = {
        "data_embeddings": modified_data_embeddings,
        "context_embeddings": modified_context_embeddings,
        "temperature": 0.07,
    }
    targets = {"data_embeddings": data_embeddings, "context_embeddings": context_embeddings}

    # Initialize ContrastiveLoss with 'infoNCE' target_mode
    loss_fn = ContrastiveLoss(target_mode="infoNCE", current_mode="data_context", similarity_metric="cosine")
    loss = loss_fn.compute_loss(outputs, targets, data_key="data_embeddings", context_key="context_embeddings")
    assert loss.item() >= 0, f"InfoNCE loss should be non-negative, got {loss.item()}"

    logits = loss_fn.get_similarity_matrix(
        outputs, "data_context", data_key="data_embeddings", context_key="context_embeddings"
    )
    labels = torch.arange(logits.shape[0], device=logits.device)
    manual_loss = F.cross_entropy(logits, labels)
    assert torch.isclose(
        loss, manual_loss, atol=1e-6
    ), f"InfoNCE loss should match manual computation, got {loss.item()} and {manual_loss.item()}"
