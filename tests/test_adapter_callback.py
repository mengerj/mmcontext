"""Tests for UnfreezeTextEncoderCallback and UnfreezeAdapterCallback."""

import logging
from types import SimpleNamespace

import pytest
import torch.nn as nn

from mmcontext.callback import UnfreezeAdapterCallback, UnfreezeTextEncoderCallback
from mmcontext.modules.adapter_module import AdapterModule
from mmcontext.modules.mmcontext_module import MMContextModule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(epoch: float) -> SimpleNamespace:
    return SimpleNamespace(epoch=epoch)


def _all_frozen(module: nn.Module) -> bool:
    return all(not p.requires_grad for p in module.parameters())


def _all_trainable(module: nn.Module) -> bool:
    return all(p.requires_grad for p in module.parameters())


class _FakePipeline(nn.Module):
    """Lightweight SentenceTransformer-like container that supports model[i] indexing."""

    def __init__(self, *modules: nn.Module):
        super().__init__()
        self._modules_list = nn.ModuleList(modules)

    def __getitem__(self, idx: int) -> nn.Module:
        return self._modules_list[idx]

    def children(self):
        return iter(self._modules_list)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mmcontext_module() -> MMContextModule:
    """MMContextModule with patched auto_model (already patched session-wide in conftest)."""
    return MMContextModule("bert-base-uncased")


@pytest.fixture()
def adapter_module() -> AdapterModule:
    return AdapterModule(text_input_dim=32, omics_input_dim=32, shared_dim=16, hidden_dim=None)


@pytest.fixture()
def pipeline_without_attention(mmcontext_module, adapter_module) -> _FakePipeline:
    """MMContextModule → AdapterModule (adapter at index 1)."""
    return _FakePipeline(mmcontext_module, adapter_module)


@pytest.fixture()
def pipeline_with_attention(mmcontext_module, adapter_module) -> _FakePipeline:
    """MMContextModule → dummy attention module → AdapterModule (adapter at index 2)."""
    dummy_attention = nn.Linear(16, 16)  # not an AdapterModule, stands in for OmicsAttentionModule
    return _FakePipeline(mmcontext_module, dummy_attention, adapter_module)


@pytest.fixture()
def pipeline_no_adapter(mmcontext_module) -> _FakePipeline:
    """Pipeline without any AdapterModule."""
    return _FakePipeline(mmcontext_module)


# ---------------------------------------------------------------------------
# UnfreezeTextEncoderCallback
# ---------------------------------------------------------------------------


class TestUnfreezeTextEncoderCallback:
    def _call_epoch(self, callback, model, epoch: float):
        """Simulate on_epoch_begin for a given epoch."""
        state = _make_state(epoch)
        callback.on_epoch_begin(args=None, state=state, control=None, model=model)

    def test_text_encoder_frozen_before_unfreeze_epoch(self, pipeline_without_attention):
        """Text encoder parameters must stay frozen before the unfreeze epoch."""
        model = pipeline_without_attention
        model[0].freeze_all_but_top_layers(0)

        cb = UnfreezeTextEncoderCallback(unfreeze_epoch=2.0)
        self._call_epoch(cb, model, epoch=0.0)
        self._call_epoch(cb, model, epoch=1.0)

        assert _all_frozen(model[0].auto_model), "Text encoder should still be frozen before epoch 2"
        assert not cb.unfrozen

    def test_text_encoder_unfreezes_at_target_epoch(self, pipeline_without_attention):
        """Text encoder should be unfrozen exactly at the configured epoch."""
        model = pipeline_without_attention
        model[0].freeze_all_but_top_layers(0)

        cb = UnfreezeTextEncoderCallback(unfreeze_epoch=2.0)
        self._call_epoch(cb, model, epoch=2.0)

        assert _all_trainable(model[0].auto_model), "Text encoder should be unfrozen at epoch 2"
        assert cb.unfrozen

    def test_text_encoder_unfreezes_past_target_epoch(self, pipeline_without_attention):
        """Unfreezing also triggers when epoch exceeds the target (e.g., epoch 3 > target 2)."""
        model = pipeline_without_attention
        model[0].freeze_all_but_top_layers(0)

        cb = UnfreezeTextEncoderCallback(unfreeze_epoch=2.0)
        self._call_epoch(cb, model, epoch=3.0)

        assert _all_trainable(model[0].auto_model)
        assert cb.unfrozen

    def test_text_encoder_unfreezes_only_once(self, pipeline_without_attention):
        """The unfreeze action should happen exactly once regardless of subsequent epochs."""
        model = pipeline_without_attention
        model[0].freeze_all_but_top_layers(0)

        cb = UnfreezeTextEncoderCallback(unfreeze_epoch=1.0)
        self._call_epoch(cb, model, epoch=1.0)

        # Manually re-freeze to check the callback doesn't unfreeze again
        model[0].freeze_all_but_top_layers(0)
        self._call_epoch(cb, model, epoch=2.0)

        assert _all_frozen(model[0].auto_model), "Should not re-unfreeze after unfrozen=True is set"

    def test_warns_when_no_compatible_module(self, caplog):
        """A warning is logged when model[0] has no auto_model attribute."""
        plain_nn = nn.Linear(4, 4)
        model = _FakePipeline(plain_nn)

        cb = UnfreezeTextEncoderCallback(unfreeze_epoch=0.0)
        with caplog.at_level(logging.WARNING):
            self._call_epoch(cb, model, epoch=0.0)

        assert any("compatible text encoder" in r.message for r in caplog.records)
        assert not cb.unfrozen

    def test_different_unfreeze_epochs(self, pipeline_without_attention):
        """Two callbacks with different epochs unfreeze independently."""
        model = pipeline_without_attention
        model[0].freeze_all_but_top_layers(0)

        cb_epoch1 = UnfreezeTextEncoderCallback(unfreeze_epoch=1.0)
        cb_epoch3 = UnfreezeTextEncoderCallback(unfreeze_epoch=3.0)

        # At epoch 1 — only the epoch-1 callback fires
        self._call_epoch(cb_epoch1, model, epoch=1.0)
        assert cb_epoch1.unfrozen
        assert not cb_epoch3.unfrozen

        model[0].freeze_all_but_top_layers(0)  # re-freeze for the next check
        # At epoch 3 — the epoch-3 callback fires
        self._call_epoch(cb_epoch3, model, epoch=3.0)
        assert cb_epoch3.unfrozen


# ---------------------------------------------------------------------------
# UnfreezeAdapterCallback
# ---------------------------------------------------------------------------


class TestUnfreezeAdapterCallback:
    def _train_begin(self, callback, model):
        callback.on_train_begin(args=None, state=None, control=None, model=model)

    def _call_epoch(self, callback, model, epoch: float):
        state = _make_state(epoch)
        callback.on_epoch_begin(args=None, state=state, control=None, model=model)

    # --- initialization ---

    def test_freezes_both_on_train_begin(self, pipeline_without_attention, adapter_module):
        cb = UnfreezeAdapterCallback(freeze_text_adapter=True, freeze_omics_adapter=True)
        self._train_begin(cb, pipeline_without_attention)

        assert _all_frozen(adapter_module.text_proj)
        assert _all_frozen(adapter_module.omics_proj)
        assert cb.adapters_initialized

    def test_freezes_only_text_on_train_begin(self, pipeline_without_attention, adapter_module):
        cb = UnfreezeAdapterCallback(freeze_text_adapter=True, freeze_omics_adapter=False)
        self._train_begin(cb, pipeline_without_attention)

        assert _all_frozen(adapter_module.text_proj)
        assert _all_trainable(adapter_module.omics_proj)

    def test_freezes_only_omics_on_train_begin(self, pipeline_without_attention, adapter_module):
        cb = UnfreezeAdapterCallback(freeze_text_adapter=False, freeze_omics_adapter=True)
        self._train_begin(cb, pipeline_without_attention)

        assert _all_trainable(adapter_module.text_proj)
        assert _all_frozen(adapter_module.omics_proj)

    # --- epoch-based unfreezing ---

    def test_text_adapter_unfreezes_at_epoch_1(self, pipeline_without_attention, adapter_module):
        cb = UnfreezeAdapterCallback(
            freeze_text_adapter=True,
            freeze_omics_adapter=True,
            unfreeze_text_adapter_epoch=1.0,
        )
        self._train_begin(cb, pipeline_without_attention)

        # Before epoch 1 — still frozen
        self._call_epoch(cb, pipeline_without_attention, epoch=0.0)
        assert _all_frozen(adapter_module.text_proj)
        assert not cb.text_adapter_unfrozen

        # At epoch 1 — text unfrozen, omics still frozen
        self._call_epoch(cb, pipeline_without_attention, epoch=1.0)
        assert _all_trainable(adapter_module.text_proj)
        assert _all_frozen(adapter_module.omics_proj)
        assert cb.text_adapter_unfrozen
        assert not cb.omics_adapter_unfrozen

    def test_omics_adapter_unfreezes_at_epoch_2(self, pipeline_without_attention, adapter_module):
        cb = UnfreezeAdapterCallback(
            freeze_text_adapter=True,
            freeze_omics_adapter=True,
            unfreeze_text_adapter_epoch=1.0,
            unfreeze_omics_adapter_epoch=2.0,
        )
        self._train_begin(cb, pipeline_without_attention)

        self._call_epoch(cb, pipeline_without_attention, epoch=1.0)  # unfreeze text
        assert cb.text_adapter_unfrozen
        assert not cb.omics_adapter_unfrozen

        self._call_epoch(cb, pipeline_without_attention, epoch=2.0)  # unfreeze omics
        assert _all_trainable(adapter_module.omics_proj)
        assert cb.omics_adapter_unfrozen

    def test_adapters_stay_frozen_without_unfreeze_epoch(self, pipeline_without_attention, adapter_module):
        """When unfreeze_*_epoch is None, frozen adapters remain frozen forever."""
        cb = UnfreezeAdapterCallback(freeze_text_adapter=True, freeze_omics_adapter=True)
        self._train_begin(cb, pipeline_without_attention)

        for epoch in [0.0, 1.0, 5.0, 100.0]:
            self._call_epoch(cb, pipeline_without_attention, epoch=epoch)

        assert _all_frozen(adapter_module.text_proj)
        assert _all_frozen(adapter_module.omics_proj)

    def test_unfreezes_only_once(self, pipeline_without_attention, adapter_module):
        """Each adapter is unfrozen at most once."""
        cb = UnfreezeAdapterCallback(
            freeze_text_adapter=True,
            freeze_omics_adapter=True,
            unfreeze_text_adapter_epoch=1.0,
            unfreeze_omics_adapter_epoch=1.0,
        )
        self._train_begin(cb, pipeline_without_attention)
        self._call_epoch(cb, pipeline_without_attention, epoch=1.0)

        assert cb.text_adapter_unfrozen
        assert cb.omics_adapter_unfrozen

        # Re-freeze manually and simulate another epoch — callback should not unfreeze again
        adapter_module.freeze_text_proj()
        adapter_module.freeze_omics_proj()
        self._call_epoch(cb, pipeline_without_attention, epoch=2.0)

        assert _all_frozen(adapter_module.text_proj)
        assert _all_frozen(adapter_module.omics_proj)

    # --- adapter at position 2 (with OmicsAttentionModule) ---

    def test_finds_adapter_at_position_2(self, pipeline_with_attention, adapter_module):
        """Adapter is found even when it's at index 2 (behind a dummy attention module)."""
        cb = UnfreezeAdapterCallback(
            freeze_text_adapter=True,
            freeze_omics_adapter=True,
            unfreeze_text_adapter_epoch=1.0,
            unfreeze_omics_adapter_epoch=2.0,
        )
        self._train_begin(cb, pipeline_with_attention)

        assert cb.adapters_initialized
        assert cb._adapter is adapter_module

        self._call_epoch(cb, pipeline_with_attention, epoch=1.0)
        assert _all_trainable(adapter_module.text_proj)
        assert _all_frozen(adapter_module.omics_proj)

        self._call_epoch(cb, pipeline_with_attention, epoch=2.0)
        assert _all_trainable(adapter_module.omics_proj)

    # --- no adapter present ---

    def test_warns_when_no_adapter(self, pipeline_no_adapter, caplog):
        """A warning is emitted and adapters_initialized stays False when no AdapterModule found."""
        cb = UnfreezeAdapterCallback(freeze_text_adapter=True, freeze_omics_adapter=True)

        with caplog.at_level(logging.WARNING):
            self._train_begin(cb, pipeline_no_adapter)

        assert any("No AdapterModule found" in r.message for r in caplog.records)
        assert not cb.adapters_initialized

    def test_epoch_begin_noop_without_initialization(self, pipeline_no_adapter):
        """on_epoch_begin is a no-op when adapters_initialized is False."""
        cb = UnfreezeAdapterCallback(
            freeze_text_adapter=True,
            unfreeze_text_adapter_epoch=0.0,
        )
        # Deliberately skip on_train_begin so adapters_initialized stays False
        state = _make_state(5.0)
        result = cb.on_epoch_begin(args=None, state=state, control=None, model=pipeline_no_adapter)
        assert result is None  # control passed through unchanged

    # --- different epoch scenarios ---

    def test_text_and_omics_unfreeze_at_different_epochs(self, pipeline_without_attention, adapter_module):
        """Text unfreezes at epoch 1, omics at epoch 3 — verify each fires independently."""
        cb = UnfreezeAdapterCallback(
            freeze_text_adapter=True,
            freeze_omics_adapter=True,
            unfreeze_text_adapter_epoch=1.0,
            unfreeze_omics_adapter_epoch=3.0,
        )
        self._train_begin(cb, pipeline_without_attention)

        self._call_epoch(cb, pipeline_without_attention, epoch=0.0)
        assert _all_frozen(adapter_module.text_proj)
        assert _all_frozen(adapter_module.omics_proj)

        self._call_epoch(cb, pipeline_without_attention, epoch=1.0)
        assert _all_trainable(adapter_module.text_proj)
        assert _all_frozen(adapter_module.omics_proj)

        self._call_epoch(cb, pipeline_without_attention, epoch=2.0)
        assert _all_trainable(adapter_module.text_proj)
        assert _all_frozen(adapter_module.omics_proj)

        self._call_epoch(cb, pipeline_without_attention, epoch=3.0)
        assert _all_trainable(adapter_module.text_proj)
        assert _all_trainable(adapter_module.omics_proj)

    def test_fractional_unfreeze_epoch(self, pipeline_without_attention, adapter_module):
        """Fractional unfreeze epochs work correctly (e.g., epoch 0.5)."""
        cb = UnfreezeAdapterCallback(
            freeze_omics_adapter=True,
            unfreeze_omics_adapter_epoch=0.5,
        )
        self._train_begin(cb, pipeline_without_attention)

        self._call_epoch(cb, pipeline_without_attention, epoch=0.0)
        assert _all_frozen(adapter_module.omics_proj)

        self._call_epoch(cb, pipeline_without_attention, epoch=0.5)
        assert _all_trainable(adapter_module.omics_proj)
