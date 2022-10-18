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
"""Tests for timed_callbacks."""


import time

import numpy as np
import tensorflow.compat.v2 as tf

import keras
from keras import timed_callbacks
from keras.testing_infra import test_combinations


@test_combinations.run_with_all_model_types
@test_combinations.run_all_keras_modes
class WaitCallback(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        time.sleep(1)


class TimedCallbacksTest(test_combinations.TestCase):
    def test_timed_callback(self):
        class Counter(timed_callbacks.TimedCallback):
            def __init__(self, interval=1):
                super().__init__(interval=interval)
                self.count = 0

            def thread_function(self, *args):
                self.count += 1

        data = np.random.random((500, 1))
        labels = np.where(data > 0.5, 1, 0)
        model = keras.models.Sequential(
            (
                keras.layers.Dense(1, input_dim=1, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            )
        )
        model.compile(
            optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"]
        )
        counter = Counter(interval=0.5)  # Half of one second interval
        waiter = WaitCallback()
        model.fit(data, labels, epochs=10, callbacks=[counter, waiter])
        # Counter should be called at least 20 times, more from overhead
        self.assertGreaterEqual(counter.count, 20)


if __name__ == "__main__":
    tf.test.main()
