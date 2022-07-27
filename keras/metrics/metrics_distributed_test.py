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
"""Tests for Keras metrics within tf.distribute.Strategy contexts."""

import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras import backend
from keras import metrics


class KerasDistributedMetricsTest(tf.test.TestCase, parameterized.TestCase):
    def test_one_device_strategy_cpu(self):
        with self.cached_session():
            with tf.distribute.OneDeviceStrategy("/CPU:0").scope():
                metric = metrics.Sum()
                output = metric.update_state(1)
                self.assertEqual(backend.eval(output), 1)

    def test_one_device_strategy_gpu(self):
        with self.cached_session():
            with tf.distribute.OneDeviceStrategy("/GPU:0").scope():
                metric = metrics.Sum()
                output = metric.update_state(1)
                self.assertEqual(backend.eval(output), 1)

    def test_mirrored_strategy(self):
        with self.cached_session():
            with tf.distribute.MirroredStrategy().scope():
                metric = metrics.Sum()
                output = metric.update_state(1)
                self.assertEqual(backend.eval(output), 1)


if __name__ == "__main__":
    tf.test.main()
