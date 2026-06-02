import logging

from transformers import TrainerCallback

from mmcontext.modules.adapter_module import AdapterModule
from mmcontext.modules.mmcontext_module import MMContextModule

logger = logging.getLogger(__name__)


class UnfreezeTextEncoderCallback(TrainerCallback):
    """
    A TrainerCallback to unfreeze the text encoder at a specified epoch.

    Parameters
    ----------
    unfreeze_epoch : float
        The epoch (or fraction thereof) at which the text encoder should be unfrozen.
        For example, if unfreeze_epoch is 0.5, then once the training reaches 0.5 epochs,
        the text encoder parameters will be unfrozen.
    """

    def __init__(self, unfreeze_epoch: float):
        self.unfreeze_epoch = unfreeze_epoch
        self.unfrozen = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        """
        Called at the beginning of each epoch.

        If the current epoch is at or past the unfreeze_epoch and the text encoder is still frozen, then unfreeze its parameters.
        """
        if not self.unfrozen and state.epoch >= self.unfreeze_epoch:
            model = kwargs["model"]
            if isinstance(model[0], MMContextModule):
                model[0].unfreeze_text_encoder()
                self.unfrozen = True
                logger.info(f"Text encoder unfrozen at epoch {state.epoch:.2f}")
            elif hasattr(model[0], "auto_model"):
                for param in model[0].auto_model.parameters():
                    param.requires_grad = True
                self.unfrozen = True
                logger.info(f"Text encoder unfrozen at epoch {state.epoch:.2f}")
            else:
                logger.warning("Model does not have a compatible text encoder at model[0].")
        return control


class UnfreezeAdapterCallback(TrainerCallback):
    """
    A TrainerCallback to freeze/unfreeze text and omics adapter projections at specified epochs.

    This callback provides fine-grained control over adapter training by allowing you to:
    1. Start with frozen adapter projection heads and unfreeze them at specific epochs
    2. Control text and omics projections independently
    3. Handle cases where the AdapterModule is at any position in the pipeline

    The callback searches ``model.children()`` for an :class:`AdapterModule` instance,
    so it works regardless of whether an OmicsAttentionModule is included in the pipeline.

    Parameters
    ----------
    freeze_text_adapter : bool, optional
        Whether to start with the text projection head frozen. Defaults to False.
    freeze_omics_adapter : bool, optional
        Whether to start with the omics projection head frozen. Defaults to False.
    unfreeze_text_adapter_epoch : float, optional
        The epoch at which to unfreeze the text projection head. If None and freeze_text_adapter
        is True, the projection remains frozen. Defaults to None.
    unfreeze_omics_adapter_epoch : float, optional
        The epoch at which to unfreeze the omics projection head. If None and freeze_omics_adapter
        is True, the projection remains frozen. Defaults to None.

    Examples
    --------
    >>> # Freeze both projections initially, unfreeze text at epoch 1, omics at epoch 2
    >>> callback = UnfreezeAdapterCallback(
    ...     freeze_text_adapter=True,
    ...     freeze_omics_adapter=True,
    ...     unfreeze_text_adapter_epoch=1.0,
    ...     unfreeze_omics_adapter_epoch=2.0,
    ... )

    >>> # Only control omics projection (useful for text-only datasets)
    >>> callback = UnfreezeAdapterCallback(freeze_omics_adapter=True, unfreeze_omics_adapter_epoch=1.5)
    """

    def __init__(
        self,
        freeze_text_adapter: bool = False,
        freeze_omics_adapter: bool = False,
        unfreeze_text_adapter_epoch: float = None,
        unfreeze_omics_adapter_epoch: float = None,
    ):
        self.freeze_text_adapter = freeze_text_adapter
        self.freeze_omics_adapter = freeze_omics_adapter
        self.unfreeze_text_adapter_epoch = unfreeze_text_adapter_epoch
        self.unfreeze_omics_adapter_epoch = unfreeze_omics_adapter_epoch

        self.text_adapter_unfrozen = False
        self.omics_adapter_unfrozen = False
        self.adapters_initialized = False
        self._adapter: AdapterModule | None = None

    def _find_adapter(self, model) -> AdapterModule | None:
        """Find the AdapterModule by iterating model.children()."""
        for m in model.children():
            if isinstance(m, AdapterModule):
                return m
        return None

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training to initialize adapter freezing state."""
        model = kwargs["model"]

        adapter = self._find_adapter(model)
        if adapter is None:
            logger.warning("No AdapterModule found in model. Adapter callback will have no effect.")
            return control

        self._adapter = adapter

        if self.freeze_text_adapter:
            adapter.freeze_text_proj()

        if self.freeze_omics_adapter:
            adapter.freeze_omics_proj()

        self.adapters_initialized = True
        logger.info("Adapter callback initialized — AdapterModule found and freeze state applied.")
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch to check if adapters should be unfrozen."""
        if not self.adapters_initialized:
            return control

        if (
            not self.text_adapter_unfrozen
            and self.unfreeze_text_adapter_epoch is not None
            and state.epoch >= self.unfreeze_text_adapter_epoch
        ):
            self._adapter.unfreeze_text_proj()
            self.text_adapter_unfrozen = True
            logger.info(f"Text adapter projection unfrozen at epoch {state.epoch:.2f}")

        if (
            not self.omics_adapter_unfrozen
            and self.unfreeze_omics_adapter_epoch is not None
            and state.epoch >= self.unfreeze_omics_adapter_epoch
        ):
            self._adapter.unfreeze_omics_proj()
            self.omics_adapter_unfrozen = True
            logger.info(f"Omics adapter projection unfrozen at epoch {state.epoch:.2f}")

        return control
