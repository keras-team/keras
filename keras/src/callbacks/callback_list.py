import concurrent.futures

from keras.src import backend
from keras.src import tree
from keras.src import utils
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.callbacks.history import History
from keras.src.callbacks.progbar_logger import ProgbarLogger
from keras.src.utils import python_utils


@keras_export("keras.callbacks.CallbackList")
class CallbackList(Callback):
    """Container abstracting a list of callbacks."""

    def __init__(
        self,
        callbacks=None,
        add_history=False,
        add_progbar=False,
        model=None,
        **params,
    ):
        """Container for `Callback` instances.

        This object wraps a list of `Callback` instances, making it possible
        to call them all at once via a single endpoint
        (e.g. `callback_list.on_epoch_end(...)`).

        Args:
            callbacks: List of `Callback` instances.
            add_history: Whether a `History` callback should be added, if one
                does not already exist in the `callbacks` list.
            add_progbar: Whether a `ProgbarLogger` callback should be added, if
                one does not already exist in the `callbacks` list.
            model: The `Model` these callbacks are used with.
            **params: If provided, parameters will be passed to each `Callback`
                via `Callback.set_params`.
        """
        self.callbacks = tree.flatten(callbacks) if callbacks else []
        self._executor = None
        self._async_train = False
        self._async_test = False
        self._async_predict = False
        self._futures = []
        self._configure_async_dispatch(callbacks)
        self._add_default_callbacks(add_history, add_progbar)
        self.set_model(model)
        self.set_params(params)

    def set_params(self, params):
        self.params = params
        if params:
            for callback in self.callbacks:
                callback.set_params(params)

    def _configure_async_dispatch(self, callbacks):
        # Determine whether callbacks can be dispatched asynchronously.
        if not backend.IS_THREAD_SAFE:
            return
        async_train = True
        async_test = True
        async_predict = True
        if callbacks:
            if isinstance(callbacks, (list, tuple)):
                for cbk in callbacks:
                    if getattr(cbk, "async_safe", False):
                        # Callbacks that expose self.async_safe == True
                        # will be assumed safe for async dispatch.
                        continue
                    if not utils.is_default(cbk.on_batch_end):
                        async_train = False
                    if not utils.is_default(cbk.on_train_batch_end):
                        async_train = False
                    if not utils.is_default(cbk.on_test_batch_end):
                        async_test = False
                    if not utils.is_default(cbk.on_predict_batch_end):
                        async_predict = False

        if async_train or async_test or async_predict:
            self._executor = concurrent.futures.ThreadPoolExecutor()

        self._async_train = async_train
        self._async_test = async_test
        self._async_predict = async_predict

    def _add_default_callbacks(self, add_history, add_progbar):
        """Adds `Callback`s that are always present."""
        self._progbar = None
        self._history = None

        for cb in self.callbacks:
            if isinstance(cb, ProgbarLogger):
                self._progbar = cb
            elif isinstance(cb, History):
                self._history = cb

        if self._history is None and add_history:
            self._history = History()
            self.callbacks.append(self._history)

        if self._progbar is None and add_progbar:
            self._progbar = ProgbarLogger()
            self.callbacks.append(self._progbar)

    def set_model(self, model):
        if not model:
            return
        super().set_model(model)
        if self._history:
            model.history = self._history
        for callback in self.callbacks:
            callback.set_model(model)

    def _async_dispatch(self, fn, *args):
        for future in self._futures:
            if future.done():
                future.result()
                self._futures.remove(future)
        future = self._executor.submit(fn, *args)
        self._futures.append(future)

    def _clear_futures(self):
        for future in self._futures:
            future.result()
        self._futures = []

    def on_batch_begin(self, batch, logs=None):
        logs = python_utils.pythonify_logs(logs)
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        logs = python_utils.pythonify_logs(logs)
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        if self._async_train:
            self._clear_futures()

        logs = python_utils.pythonify_logs(logs)
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        logs = python_utils.pythonify_logs(logs)
        for callback in self.callbacks:
            callback.on_train_batch_begin(batch, logs=logs)

    def on_test_batch_begin(self, batch, logs=None):
        logs = python_utils.pythonify_logs(logs)
        for callback in self.callbacks:
            callback.on_test_batch_begin(batch, logs=logs)

    def on_predict_batch_begin(self, batch, logs=None):
        logs = python_utils.pythonify_logs(logs)
        for callback in self.callbacks:
            callback.on_predict_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        if self._async_train:
            self._async_dispatch(self._on_batch_end, batch, logs)
        else:
            self._on_batch_end(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        if self._async_train:
            self._async_dispatch(self._on_train_batch_end, batch, logs)
        else:
            self._on_train_batch_end(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        if self._async_test:
            self._async_dispatch(self._on_test_batch_end, batch, logs)
        else:
            self._on_test_batch_end(batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        if self._async_predict:
            self._async_dispatch(self._on_predict_batch_end, batch, logs)
        else:
            self._on_predict_batch_end(batch, logs)

    def _on_batch_end(self, batch, logs=None):
        logs = python_utils.pythonify_logs(logs)
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs=logs)

    def _on_train_batch_end(self, batch, logs=None):
        logs = python_utils.pythonify_logs(logs)
        for callback in self.callbacks:
            callback.on_train_batch_end(batch, logs=logs)

    def _on_test_batch_end(self, batch, logs=None):
        logs = python_utils.pythonify_logs(logs)
        for callback in self.callbacks:
            callback.on_test_batch_end(batch, logs=logs)

    def _on_predict_batch_end(self, batch, logs=None):
        logs = python_utils.pythonify_logs(logs)
        for callback in self.callbacks:
            callback.on_predict_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        logs = python_utils.pythonify_logs(logs)
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        if self._async_train:
            self._clear_futures()

        logs = python_utils.pythonify_logs(logs)
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_test_begin(self, logs=None):
        logs = python_utils.pythonify_logs(logs)
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        if self._async_test:
            self._clear_futures()

        logs = python_utils.pythonify_logs(logs)
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_predict_begin(self, logs=None):
        logs = python_utils.pythonify_logs(logs)
        for callback in self.callbacks:
            callback.on_predict_begin(logs)

    def on_predict_end(self, logs=None):
        if self._async_predict:
            self._clear_futures()

        logs = python_utils.pythonify_logs(logs)
        for callback in self.callbacks:
            callback.on_predict_end(logs)
