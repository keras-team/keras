from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.utils import io_utils
from keras.src.utils.progbar import Progbar


@keras_export("keras.callbacks.ProgbarLogger")
class ProgbarLogger(Callback):
    """Callback that prints metrics to stdout.

    Args:
        count_mode: One of `"steps"` or `"samples"`.
            Whether the progress bar should
            count samples seen or steps (batches) seen.

    Raises:
        ValueError: In case of invalid `count_mode`.
    """

    def __init__(self):
        super().__init__()
        self.seen = 0
        self.progbar = None
        self.target = None
        self.verbose = 1
        self.epochs = 1

        self._called_in_fit = False

    def set_params(self, params):
        verbose = params["verbose"]
        if verbose == "auto":
            verbose = 1
        self.verbose = verbose
        self.epochs = params["epochs"]
        self.target = params["steps"]

    def on_train_begin(self, logs=None):
        # When this logger is called inside `fit`, validation is silent.
        self._called_in_fit = True

    def on_test_begin(self, logs=None):
        if not self._called_in_fit:
            self._reset_progbar()
            self._maybe_init_progbar()

    def on_predict_begin(self, logs=None):
        self._reset_progbar()
        self._maybe_init_progbar()

    def on_epoch_begin(self, epoch, logs=None):
        self._reset_progbar()
        self._maybe_init_progbar()
        if self.verbose and self.epochs > 1:
            io_utils.print_msg(f"Epoch {epoch + 1}/{self.epochs}")

    def on_train_batch_end(self, batch, logs=None):
        self._update_progbar(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        if not self._called_in_fit:
            self._update_progbar(batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        # Don't pass prediction results.
        self._update_progbar(batch, None)

    def on_epoch_end(self, epoch, logs=None):
        self._finalize_progbar(logs)

    def on_test_end(self, logs=None):
        if not self._called_in_fit:
            self._finalize_progbar(logs)

    def on_predict_end(self, logs=None):
        self._finalize_progbar(logs)

    def _reset_progbar(self):
        self.seen = 0
        self.progbar = None

    def _maybe_init_progbar(self):
        if self.progbar is None:
            self.progbar = Progbar(
                target=self.target, verbose=self.verbose, unit_name="step"
            )

    def _update_progbar(self, batch, logs=None):
        """Updates the progbar."""
        logs = logs or {}
        self._maybe_init_progbar()
        self.seen = batch + 1  # One-indexed.

        if self.verbose == 1:
            self.progbar.update(self.seen, list(logs.items()), finalize=False)

    def _finalize_progbar(self, logs):
        logs = logs or {}
        if self.target is None:
            self.target = self.seen
            self.progbar.target = self.target
        self.progbar.update(self.target, list(logs.items()), finalize=True)
