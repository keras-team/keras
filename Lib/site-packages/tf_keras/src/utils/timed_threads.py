# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Thread utilities."""

import abc
import threading

from absl import logging
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.utils.TimedThread", v1=[])
class TimedThread:
    """Time-based interval Threads.

    Runs a timed thread every x seconds. It can be used to run a threaded
    function alongside model training or any other snippet of code.

    Args:
        interval: The interval, in seconds, to wait between calls to the
            `on_interval` function.
        **kwargs: additional args that are passed to `threading.Thread`. By
            default, `Thread` is started as a `daemon` thread unless
            overridden by the user in `kwargs`.

    Examples:

    ```python
    class TimedLogIterations(keras.utils.TimedThread):
        def __init__(self, model, interval):
            self.model = model
            super().__init__(interval)

        def on_interval(self):
            # Logs Optimizer iterations every x seconds
            try:
                opt_iterations = self.model.optimizer.iterations.numpy()
                print(f"Epoch: {epoch}, Optimizer Iterations: {opt_iterations}")
            except Exception as e:
                print(str(e))  # To prevent thread from getting killed

    # `start` and `stop` the `TimerThread` manually. If the `on_interval` call
    # requires access to `model` or other objects, override `__init__` method.
    # Wrap it in a `try-except` to handle exceptions and `stop` the thread run.
    timed_logs = TimedLogIterations(model=model, interval=5)
    timed_logs.start()
    try:
        model.fit(...)
    finally:
        timed_logs.stop()

    # Alternatively, run the `TimedThread` in a context manager
    with TimedLogIterations(model=model, interval=5):
        model.fit(...)

    # If the timed thread instance needs access to callback events,
    # subclass both `TimedThread` and `Callback`.  Note that when calling
    # `super`, they will have to called for each parent class if both of them
    # have the method that needs to be run. Also, note that `Callback` has
    # access to `model` as an attribute and need not be explictly provided.
    class LogThreadCallback(
        keras.utils.TimedThread, keras.callbacks.Callback
    ):
        def __init__(self, interval):
            self._epoch = 0
            keras.utils.TimedThread.__init__(self, interval)
            keras.callbacks.Callback.__init__(self)

        def on_interval(self):
            if self.epoch:
                opt_iter = self.model.optimizer.iterations.numpy()
                logging.info(f"Epoch: {self._epoch}, Opt Iteration: {opt_iter}")

        def on_epoch_begin(self, epoch, logs=None):
            self._epoch = epoch

    with LogThreadCallback(interval=5) as thread_callback:
        # It's required to pass `thread_callback` to also `callbacks` arg of
        # `model.fit` to be triggered on callback events.
        model.fit(..., callbacks=[thread_callback])
    ```
    """

    def __init__(self, interval, **kwargs):
        self.interval = interval
        self.daemon = kwargs.pop("daemon", True)
        self.thread_kwargs = kwargs
        self.thread = None
        self.thread_stop_event = None

    def _call_on_interval(self):
        # Runs indefinitely once thread is started
        while not self.thread_stop_event.is_set():
            self.on_interval()
            self.thread_stop_event.wait(self.interval)

    def start(self):
        """Creates and starts the thread run."""
        if self.thread and self.thread.is_alive():
            logging.warning("Thread is already running.")
            return
        self.thread = threading.Thread(
            target=self._call_on_interval,
            daemon=self.daemon,
            **self.thread_kwargs
        )
        self.thread_stop_event = threading.Event()
        self.thread.start()

    def stop(self):
        """Stops the thread run."""
        if self.thread_stop_event:
            self.thread_stop_event.set()

    def is_alive(self):
        """Returns True if thread is running. Otherwise returns False."""
        if self.thread:
            return self.thread.is_alive()
        return False

    def __enter__(self):
        # Starts the thread in context manager
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        # Stops the thread run.
        self.stop()

    @abc.abstractmethod
    def on_interval(self):
        """User-defined behavior that is called in the thread."""
        raise NotImplementedError(
            "Runs every x interval seconds. Needs to be "
            "implemented in subclasses of `TimedThread`"
        )

