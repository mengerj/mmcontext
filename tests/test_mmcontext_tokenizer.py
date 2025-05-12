# test_tokenize_masks.py
from unittest.mock import patch

import datasets
import numpy as np
import pytest
import torch

from mmcontext.models.MMContextEncoder import MMContextEncoder, MMContextProcessor


# --- fixtures ---------------------------------------------------------
@pytest.fixture(scope="session")
def processor():
    lookup = {"F1": 1, "F2": 2, "S1": 3}  # PAD row 0 reserved
    proc = MMContextProcessor("bert-base-uncased", lookup)
    return proc


@pytest.fixture(scope="session")
def dummy_encoder(processor):
    with patch("transformers.AutoModel.from_pretrained"), patch("transformers.AutoTokenizer.from_pretrained"):
        enc = MMContextEncoder("bert-base-uncased", 8, 8)
        enc.processor = processor
        return enc


# --- the actual test --------------------------------------------------
def test_masks_align_with_padding(processor):
    batch = [
        "sample_idx: F1 F2",  # omics, 2 IDs
        "A caption with words",  # text
        "sample_idx: S1",
    ]  # omics, 1 ID

    feats = processor.tokenize(batch)

    # TEXT part ---------------------------------------------------------
    txt_ids = feats["input_ids"]
    txt_mask = feats["attention_mask"]
    assert txt_ids.shape == txt_mask.shape
    # padded positions must have mask == False
    assert torch.all((txt_ids != 0) == txt_mask)

    # OMICS part --------------------------------------------------------
    om_ids = feats["omics_ids"]
    om_mask = feats["omics_attention_mask"]
    assert om_ids.shape == om_mask.shape
    assert om_mask.dtype is torch.bool
    assert torch.all((om_ids != 0) == om_mask)

    # Sample-level counts (row 0 has 2 IDs, row 2 has 1)
    assert om_mask.sum(dim=1).tolist() == [2, 0, 1]
