import numpy as np

from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.utils import io_utils


@keras_export("keras.callbacks.TerminateOnNaN")
class TerminateOnNaN(Callback):
    """Callback that terminates training when a NaN loss is encountered."""

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                io_utils.print_msg(
                    f"Batch {batch}: Invalid loss, terminating training"
                )
                self.model.stop_training = True
