import logging

from transformers import TrainerCallback

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
            # Assuming model[0] is your MMContextEncoder containing a text_encoder attribute
            if hasattr(model[0], "text_encoder"):
                for param in model[0].text_encoder.parameters():
                    param.requires_grad = True
                self.unfrozen = True
                logger.info(f"Text encoder unfrozen at epoch {state.epoch:.2f}")
            else:
                logger.warning("Model does not have a 'text_encoder' attribute at model[0].")
        return control


class UnfreezeAdapterCallback(TrainerCallback):
    """
    A TrainerCallback to freeze/unfreeze text and omics adapters at specified epochs.

    This callback provides fine-grained control over adapter training by allowing you to:
    1. Start with frozen adapters and unfreeze them at specific epochs
    2. Control text and omics adapters independently
    3. Handle cases where only one type of adapter is present

    Parameters
    ----------
    freeze_text_adapter : bool, optional
        Whether to start with the text adapter frozen. Defaults to False.
    freeze_omics_adapter : bool, optional
        Whether to start with the omics adapter frozen. Defaults to False.
    unfreeze_text_adapter_epoch : float, optional
        The epoch at which to unfreeze the text adapter. If None and freeze_text_adapter
        is True, the adapter remains frozen. Defaults to None.
    unfreeze_omics_adapter_epoch : float, optional
        The epoch at which to unfreeze the omics adapter. If None and freeze_omics_adapter
        is True, the adapter remains frozen. Defaults to None.

    Examples
    --------
    >>> # Freeze both adapters initially, unfreeze text at epoch 1, omics at epoch 2
    >>> callback = UnfreezeAdapterCallback(
    ...     freeze_text_adapter=True,
    ...     freeze_omics_adapter=True,
    ...     unfreeze_text_adapter_epoch=1.0,
    ...     unfreeze_omics_adapter_epoch=2.0,
    ... )

    >>> # Only control omics adapter (useful for text-only datasets)
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

        # Track unfreezing state
        self.text_adapter_unfrozen = False
        self.omics_adapter_unfrozen = False

        # Track initialization state
        self.adapters_initialized = False

    def _freeze_adapter(self, adapter, adapter_name: str):
        """Freeze all parameters in an adapter."""
        if adapter is not None:
            for param in adapter.parameters():
                param.requires_grad = False
            logger.info(f"{adapter_name} adapter frozen")
        else:
            logger.debug(f"{adapter_name} adapter not present, skipping freeze")

    def _unfreeze_adapter(self, adapter, adapter_name: str):
        """Unfreeze all parameters in an adapter."""
        if adapter is not None:
            for param in adapter.parameters():
                param.requires_grad = True
            logger.info(f"{adapter_name} adapter unfrozen")
        else:
            logger.debug(f"{adapter_name} adapter not present, skipping unfreeze")

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training to initialize adapter freezing state."""
        model = kwargs["model"]

        # Check if we have MMContextEncoder at model[0]
        if not hasattr(model[0], "text_adapter") and not hasattr(model[0], "omics_adapter"):
            logger.warning("Model does not have adapter attributes. Adapter callback will have no effect.")
            # Don't set adapters_initialized = True when no adapters are present
            return control

        # Initialize freezing state for adapters
        if self.freeze_text_adapter and hasattr(model[0], "text_adapter"):
            self._freeze_adapter(model[0].text_adapter, "Text")

        if self.freeze_omics_adapter and hasattr(model[0], "omics_adapter"):
            self._freeze_adapter(model[0].omics_adapter, "Omics")

        self.adapters_initialized = True

        # Log current adapter status
        text_adapter_present = hasattr(model[0], "text_adapter") and model[0].text_adapter is not None
        omics_adapter_present = hasattr(model[0], "omics_adapter") and model[0].omics_adapter is not None

        logger.info(
            f"Adapter callback initialized - Text adapter: {'present' if text_adapter_present else 'absent'}, "
            f"Omics adapter: {'present' if omics_adapter_present else 'absent'}"
        )

        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch to check if adapters should be unfrozen."""
        if not self.adapters_initialized:
            return control

        model = kwargs["model"]

        # Check text adapter unfreezing
        if (
            not self.text_adapter_unfrozen
            and self.unfreeze_text_adapter_epoch is not None
            and state.epoch >= self.unfreeze_text_adapter_epoch
        ):
            if hasattr(model[0], "text_adapter"):
                self._unfreeze_adapter(model[0].text_adapter, "Text")
                self.text_adapter_unfrozen = True
                logger.info(f"Text adapter unfrozen at epoch {state.epoch:.2f}")

        # Check omics adapter unfreezing
        if (
            not self.omics_adapter_unfrozen
            and self.unfreeze_omics_adapter_epoch is not None
            and state.epoch >= self.unfreeze_omics_adapter_epoch
        ):
            if hasattr(model[0], "omics_adapter"):
                self._unfreeze_adapter(model[0].omics_adapter, "Omics")
                self.omics_adapter_unfrozen = True
                logger.info(f"Omics adapter unfrozen at epoch {state.epoch:.2f}")

        return control
