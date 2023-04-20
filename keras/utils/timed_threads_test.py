# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for timed_threads."""

import time

import tensorflow.compat.v2 as tf
from absl import logging

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import timed_threads


@test_utils.run_v2_only
class TimedThreadTest(test_combinations.TestCase):
    def test_timed_thread_run(self):
        class LogThread(timed_threads.TimedThread):
            def on_interval(self):
                logging.info("Thread Run")

        log_thread = LogThread(interval=0.1)
        with self.assertLogs(level="INFO") as logs:
            log_thread.start()
            time.sleep(1)
            self.assertTrue(log_thread.is_alive())
            log_thread.stop()
        self.assertIn("INFO:absl:Thread Run", logs.output)
        time.sleep(0.1)
        self.assertFalse(log_thread.is_alive())

    def test_timed_thread_restart(self):
        # Verfiy that thread can be started and stopped multiple times.
        class LogThread(timed_threads.TimedThread):
            def on_interval(self):
                logging.info("Thread Run")

        log_thread = LogThread(interval=0.1)
        for _ in range(2):
            self.assertFalse(log_thread.is_alive())
            with self.assertLogs(level="INFO") as logs:
                log_thread.start()
                time.sleep(1)
                self.assertTrue(log_thread.is_alive())
                log_thread.stop()
            self.assertIn("INFO:absl:Thread Run", logs.output)
            time.sleep(0.1)
            self.assertFalse(log_thread.is_alive())

    def test_timed_thread_running_warning(self):
        # Verfiy thread start warning if its already running
        class LogThread(timed_threads.TimedThread):
            def on_interval(self):
                logging.info("Thread Run")

        log_thread = LogThread(interval=0.1)
        self.assertFalse(log_thread.is_alive())
        with self.assertLogs(level="INFO") as logs:
            log_thread.start()
            time.sleep(1)
            self.assertTrue(log_thread.is_alive())
            self.assertIn("INFO:absl:Thread Run", logs.output)
        with self.assertLogs(level="WARNING") as logs:
            log_thread.start()
            self.assertIn(
                "WARNING:absl:Thread is already running.", logs.output
            )
            self.assertTrue(log_thread.is_alive())
        log_thread.stop()
        time.sleep(0.1)
        self.assertFalse(log_thread.is_alive())

    def test_timed_thread_callback_model_fit(self):
        class LogThreadCallback(
            timed_threads.TimedThread, keras.callbacks.Callback
        ):
            def __init__(self, interval):
                self._epoch = 0
                timed_threads.TimedThread.__init__(self, interval=interval)
                keras.callbacks.Callback.__init__(self)

            def on_interval(self):
                if self._epoch:
                    # Verify that `model` is accessible.
                    _ = self.model.optimizer.iterations.numpy()
                    logging.info(f"Thread Run Epoch: {self._epoch}")

            def on_epoch_begin(self, epoch, logs=None):
                self._epoch = epoch
                time.sleep(1)

        x = tf.random.normal((32, 2))
        y = tf.ones((32, 1), dtype=tf.float32)
        model = keras.Sequential([keras.layers.Dense(1)])
        model.compile(loss="mse")
        with self.assertLogs(level="INFO") as logs, LogThreadCallback(
            interval=0.1
        ) as log_thread_callback:
            self.assertIsNone(log_thread_callback.model)
            model.fit(x, y, epochs=2, callbacks=[log_thread_callback])
            self.assertIsNotNone(log_thread_callback.model)
            self.assertIn("INFO:absl:Thread Run Epoch: 1", logs.output)


if __name__ == "__main__":
    tf.test.main()
