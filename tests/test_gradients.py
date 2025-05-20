# test_gradients.py
import pytest
import torch


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _make_mixed_batch(enc):
    """
    Build a 4-item batch:
      • 2 omics tokens  (S1, S2)
      • 2 text captions
    ⇒ text- and omics-adapters each process ≥2 rows → BatchNorm happy.
    """
    omics_samples = ["sample_idx: S1", "sample_idx: S2"]
    text_samples = ["This is caption A.", "This is caption B."]
    return enc.tokenize(omics_samples + text_samples)


# ---------------------------------------------------------------------
# 1.  Gradients flow through adapters
# ---------------------------------------------------------------------
def test_adapter_grads_exist(bimodal_encoder):
    enc = bimodal_encoder
    enc.train()

    feats = _make_mixed_batch(enc)
    out = enc(feats)

    # simple scalar loss: L = sum(embeddings²)
    loss = out["sentence_embedding"].pow(2).sum()
    loss.backward()

    # —— every adapter parameter should now have a grad ---------------
    adapters = list(enc.text_adapter.parameters()) + list(enc.omics_adapter.parameters())
    assert adapters, "No adapters found"

    grads = [p.grad for p in adapters]
    assert all(g is not None for g in grads), "Some adapter grads are None"
    assert any(g.abs().sum() > 0 for g in grads), "All adapter grads are zero"


# ---------------------------------------------------------------------
# 2.  Optimiser step actually changes adapter weights
# ---------------------------------------------------------------------
@pytest.mark.parametrize("lr", [1e-1])  # lr as a param → easy to tweak
def test_adapter_weights_update(bimodal_encoder, lr):
    enc = bimodal_encoder
    enc.train()

    # -----------------------------------------------------------------
    #  stash *all* trainable params before the step
    before = {n: p.detach().clone() for n, p in enc.named_parameters() if p.requires_grad}

    # one SGD step -----------------------------------------------------
    opt = torch.optim.SGD(enc.parameters(), lr=lr)
    feats = _make_mixed_batch(enc)
    loss = enc(feats)["sentence_embedding"].pow(2).sum()

    opt.zero_grad()
    loss.backward()
    opt.step()

    # -----------------------------------------------------------------
    #  compare after – at least one adapter tensor must differ
    changed = []
    for name, p in enc.named_parameters():
        if "adapter" in name and p.requires_grad:
            delta = (p.detach() - before[name]).abs().max().item()
            changed.append(delta)

    assert any(d > 0 for d in changed), "No adapter weight changed after optimiser step"
