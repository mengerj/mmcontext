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
