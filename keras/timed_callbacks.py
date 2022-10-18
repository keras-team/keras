# flake8: noqa
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Timed Callbacks: utilities called at user-specified intervals during model training."""


import abc
import threading
import time

import keras
from keras.callbacks import Callback

# isort: off
from tensorflow.python.training import (
    coordinator,
)  # pylint: disable=g-direct-tensorflow-import


class TimedCallback(Callback, metaclass=abc.ABCMeta):
    """Abstract class used to build new time-based callbacks."""

    def __init__(self, interval=5):
        """Initialize Timed Callback object.

        Args:
            interval: the interval, in seconds, to wait between
              calls to the thread function.
        """
        self.interval = interval
        self._coord = coordinator.Coordinator()

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
        self.thread = threading.Thread(
            target=self._call_function_on_interval, daemon=True
        )  # needed in order for threads to shut down successfully
        self.thread.start()

    def _call_function_on_interval(self):
        while not self._coord.should_stop():
            time.sleep(self.interval)
            self.thread_function()

    @abc.abstractmethod
    def thread_function(self, *args):
        """User-defined behavior that is called in the thread.

        Args:
            *args:
        """

        raise NotImplementedError("Must be implemented in subclasses")

    def on_train_end(self, _):
        self._coord.request_stop()
