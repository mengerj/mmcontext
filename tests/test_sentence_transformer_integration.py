# test_sentence_transformer_integration.py
import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import datasets
import numpy as np
import pytest
import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from mmcontext.models.mmcontextencoder import MMContextEncoder, MMContextProcessor

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Basic Integration Tests
# --------------------------------------------------------------------- #


def test_text_only_encode(st_text_encoder):
    """Test that we can encode text with a text-only model."""
    texts = ["This is a test", "Another test"]
    # text_only_encoder.text_encoder.__call__.return_value = mock_output

    # Mock the text encoder's forward method to return a proper tensor output
    # with patch.object(st_text_encoder._first_module().text_encoder, '__call__') as mock_forward:
    # Create a proper tensor output with the correct shape
    # mock_output = type('MockOutput', (), {
    #    'pooler_output': torch.zeros(2, 32, dtype=torch.float32)  # batch_size=2, hidden_size=32
    # })
    # mock_forward.return_value = mock_output

    # Encode as tensor
    embeddings = st_text_encoder.encode(texts, convert_to_tensor=True)

    # Check output shape and type
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 8)  # Batch size 2, output_dim 8

    # Encode as numpy
    np_embeddings = st_text_encoder.encode(texts, convert_to_tensor=False)

    # Check output shape and type
    assert isinstance(np_embeddings, np.ndarray)
    assert np_embeddings.shape == (2, 8)  # Batch size 2, output_dim 8


def test_text_encode_token_embeddings():
    """Test that we can encode with text model and output token embeddings."""
    # Mixed input: one omics ID, one text
    inputs = ["This is a test", "Another test"]

    encoder = MMContextEncoder(
        text_encoder_name="bert-base-uncased",
        adapter_hidden_dim=8,
        adapter_output_dim=4,
        output_token_embeddings=True,  # Ensure we get token embeddings
    )
    st_encoder = SentenceTransformer(modules=[encoder])
    # Encode as tensor
    embeddings = st_encoder.encode(inputs, output_value="token_embeddings")
    # embeddings = embeddings[0]  # Get the first (and only) element of the batch
    # Check output shape and type
    for embedding in embeddings:
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (8, 4)


def test_bimodal_encode_token_embeddings(numeric_df):
    """Test that we can encode with bimodal model and output token embeddings."""
    inputs = ["This is a test", "sample_idx:F1 F2 F3"]  # matches numeric_df from conftest

    # Encode as tensor
    encoder = MMContextEncoder(
        text_encoder_name="bert-base-uncased",
        adapter_hidden_dim=8,
        adapter_output_dim=4,
        output_token_embeddings=True,  # Ensure we get token embeddings
    )
    encoder.register_initial_embeddings(numeric_df, data_origin="pca")
    st_encoder_bimodal = SentenceTransformer(modules=[encoder])
    embeddings = st_encoder_bimodal.encode(inputs, output_value="token_embeddings")
    # Check output shape and type
    assert isinstance(embeddings[0], torch.Tensor)
    assert embeddings[0].shape == (8, 4)  # Assuming - will assign 8 tokens for each sentence
    assert isinstance(embeddings[1], torch.Tensor)
    assert embeddings[1].shape == (3, 4)  # Assuming 4 tokens for the omics ID


def test_batch_encoding(st_bimodal_encoder):
    """Test batch encoding with different batch sizes."""
    # Create a larger batch
    prefixed_ids = [f"sample_idx:F{i}" for i in range(1, 4)]
    texts = ["Text sample 1", "Text sample 2", "Text sample 3"]
    mixed_batch = prefixed_ids + texts

    # Encode with default batch size
    embeddings = st_bimodal_encoder.encode(mixed_batch, convert_to_tensor=True)

    # Check output
    assert embeddings.shape == (6, 8)  # 6 inputs, 8-dim embeddings

    # Try with explicit batch size
    embeddings_batched = st_bimodal_encoder.encode(mixed_batch, batch_size=2, convert_to_tensor=True)

    # Results should be the same regardless of batch size
    assert torch.allclose(embeddings, embeddings_batched)


# --------------------------------------------------------------------- #
# Save and Load Tests
# --------------------------------------------------------------------- #


def test_save_load_text_only(st_text_encoder, tmp_path):
    """Test that we can save and load a text-only model."""
    # Encode some text
    texts = ["This is a test", "Another test"]
    original_embeddings = st_text_encoder.encode(texts, convert_to_tensor=True)

    # Save the model
    save_dir = tmp_path / "text_only_model"
    st_text_encoder.save(str(save_dir))

    # Check that necessary files exist
    assert (save_dir / "modules.json").exists()
    assert (save_dir / "0_MMContextEncoder").exists()
    assert (save_dir / "0_MMContextEncoder/config.json").exists()

    # Load the model
    loaded_model = SentenceTransformer(str(save_dir))

    # Encode the same text with the loaded model
    loaded_embeddings = loaded_model.encode(texts, convert_to_tensor=True)

    # Check that embeddings are the same
    assert torch.allclose(original_embeddings, loaded_embeddings, atol=1e-5)


def test_save_load_bimodal(st_bimodal_encoder, numeric_mapping, tmp_path):  # numeric mapping has S1
    """Test that we can save and load a bimodal model and still use it for omics data."""
    # Create mixed input with omics ID and text
    prefixed_id = "sample_idx:S1"
    inputs = [prefixed_id, "This is a test"]

    # Get original embeddings
    original_embeddings = st_bimodal_encoder.encode(inputs, convert_to_tensor=True)

    # Save the model
    save_dir = tmp_path / "bimodal_model"
    st_bimodal_encoder.save(str(save_dir))

    # Check that necessary files exist
    assert (save_dir / "modules.json").exists()
    assert (save_dir / "0_MMContextEncoder").exists()
    assert (save_dir / "0_MMContextEncoder/config.json").exists()

    # Load the model (without data)
    loaded_model = SentenceTransformer(str(save_dir))

    # The first module should be MMContextEncoder
    loaded_encoder = loaded_model._first_module()
    assert isinstance(loaded_encoder, MMContextEncoder)

    # Re-register numeric data with the loaded model
    loaded_encoder.register_initial_embeddings(numeric_mapping, data_origin="pca")

    # Re-encode the same inputs with the loaded model
    loaded_embeddings = loaded_model.encode(inputs, convert_to_tensor=True)

    # Check that embeddings are the same
    assert torch.allclose(original_embeddings, loaded_embeddings, atol=1e-5)


# --------------------------------------------------------------------- #
# Training Tests
# --------------------------------------------------------------------- #


def test_training_text_only(st_text_encoder, dummy_dataset, tmp_path):
    """Test training a text-only model with SentenceTransformerTrainer."""
    # Set up training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=str(tmp_path),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-4,
        evaluation_strategy="steps",
        eval_steps=1,
        save_total_limit=1,
        no_cuda=True,  # Ensure we don't use CUDA for this test
    )

    # Set up loss function
    train_loss = losses.CosineSimilarityLoss(st_text_encoder)

    # Create a simple evaluator
    # without using .prepare_ds() on the ds, the omics_tokens will be treated as text input
    evaluator = EmbeddingSimilarityEvaluator(
        dummy_dataset["omics_tokens"], dummy_dataset["caption"], dummy_dataset["label"]
    )

    # Create trainer
    trainer = SentenceTransformerTrainer(
        model=st_text_encoder,
        args=args,
        train_dataset=dummy_dataset,  # Use dummy_text_dataset instead of dummy_dataset
        loss=train_loss,
        evaluator=evaluator,
    )

    # Train for one epoch
    with patch("transformers.trainer.Trainer.train") as mock_train:
        # Mock the training to avoid actual computation
        mock_train.return_value = None
        trainer.train()
        # Verify train was called
        assert mock_train.called

    # Ensure we can still use the model after training
    texts = ["This is a test", "Another test"]
    embeddings = st_text_encoder.encode(texts, convert_to_tensor=True)
    assert embeddings.shape == (2, 8)  # Fix output dimension to match model


def test_training_bimodal(st_bimodal_encoder, dummy_dataset_with_split, tmp_path):
    """Test training a bimodal model with SentenceTransformerTrainer."""
    # Set up training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=str(tmp_path),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-4,
        evaluation_strategy="no",
        eval_steps=1,
        save_total_limit=1,
        no_cuda=True,  # Ensure we don't use CUDA for this test
    )

    # Create classifier loss
    train_loss = losses.SoftmaxLoss(
        model=st_bimodal_encoder,
        sentence_embedding_dimension=4,
        num_labels=2,
        concatenation_sent_rep=True,  # For single sentence
    )
    ds = st_bimodal_encoder[0].prepare_ds(dummy_dataset_with_split, cell_sentences_cols="omics_tokens")
    # Create trainer with bimodal dataset
    trainer = SentenceTransformerTrainer(
        model=st_bimodal_encoder,
        args=args,
        train_dataset=ds["train"],
        loss=train_loss,
    )

    # Mock the training to avoid actual computation
    with patch("transformers.trainer.Trainer.train") as mock_train:
        mock_train.return_value = None
        trainer.train()
        assert mock_train.called

    # Ensure we can still use the model after training
    inputs = ["sample_idx:S1", "This is a test"]
    embeddings = st_bimodal_encoder.encode(inputs, convert_to_tensor=True)
    assert embeddings.shape == (2, 8)


# --------------------------------------------------------------------- #
# Integration with Framework Features
# --------------------------------------------------------------------- #


def test_normalization_layer(bimodal_encoder):
    """Test adding a normalization layer to SentenceTransformer."""
    from sentence_transformers import models

    # Create model with normalization
    normalize = models.Normalize()

    st_model = SentenceTransformer(modules=[bimodal_encoder, normalize])

    # Test encoding
    inputs = ["sample_idx:S1", "This is a test"]
    embeddings = st_model.encode(inputs, convert_to_tensor=True)

    # Embeddings should be normalized (unit vectors)
    norms = torch.norm(embeddings, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_precision_conversion(st_bimodal_encoder):
    """Test that we can convert the model precision and it affects the output."""
    # Test in FP32
    inputs = ["sample_idx:S1", "This is a test"]
    fp32_embeddings = st_bimodal_encoder.encode(inputs, convert_to_tensor=True)
    assert fp32_embeddings.dtype == torch.float32

    # Convert to FP16
    st_bimodal_encoder.half()

    # Test in FP16
    fp16_embeddings = st_bimodal_encoder.encode(inputs, convert_to_tensor=True)
    assert fp16_embeddings.dtype == torch.float16

    # Convert back to FP32 for other tests
    st_bimodal_encoder.float()


def test_gradient_checkpointing(bimodal_encoder):
    """Test that gradient checkpointing can be enabled for memory efficiency."""
    # Check that we can enable gradient checkpointing on the text encoder
    if hasattr(bimodal_encoder.text_encoder, "gradient_checkpointing_enable"):
        bimodal_encoder.text_encoder.gradient_checkpointing_enable()
        assert bimodal_encoder.text_encoder.is_gradient_checkpointing, "Gradient checkpointing not enabled"
    # Some models might not support gradient checkpointing, so this is conditional


def test_with_problematic_inputs(text_only_encoder):
    """Test the model's robustness with inputs that might cause type issues."""
    # Create a potentially problematic input - this is similar to how
    # SentenceTransformer might manipulate inputs in some scenarios
    text = "This is a test"

    # Mock a tokenized output that has problematic data types
    with patch.object(text_only_encoder, "tokenize") as mock_tokenize:
        # Return a dict with a boolean instead of a tensor for omics_text_info
        mock_tokenize.return_value = {
            "input_ids": torch.tensor([[101, 2023, 2003, 1037, 3231, 102]]),  # Example tokenized text
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
            "omics_text_info": torch.tensor([True]),  # This would cause issues without the fix
        }

        # With the fix in place, this should be handled correctly
        try:
            # Create a SentenceTransformer with the MMContextEncoder
            st_model = SentenceTransformer(modules=[text_only_encoder])

            # This would fail before the fix
            embedding = st_model.encode(text, convert_to_tensor=True)

            # If we reach here, the issue is fixed
            assert isinstance(embedding, torch.Tensor)
            assert embedding.shape[0] == 8  # Expected output shape
        except AttributeError:
            pytest.fail("Model should handle boolean omics_text_info without AttributeError")
