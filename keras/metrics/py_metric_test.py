# Copyright 2023 The Keras Authors. All Rights Reserved.
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
"""Tests for Keras PyMetric classes."""


import tensorflow.compat.v2 as tf

from keras import metrics
from keras.testing_infra import test_combinations


class KTrimmedMean(metrics.PyMetric):
    """An example PyMetric which computes the trimmed mean of `y_pred`."""

    def __init__(self, k=0.1, name="k_trimmed_mean", **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = y_true.numpy()

        if sample_weight is not None:
            y_true *= sample_weight.numpy()

        # Insert y_pred into our values list (keeping the list sorted)
        index = 0
        for i, element in enumerate(self.values):
            if y_true > element:
                index = i
                break
        self.values = self.values[:index] + [y_true] + self.values[index:]

    def reset_state(self):
        self.values = []

    def result(self):
        k = int(self.k * len(self.values))
        return tf.reduce_mean(self.values[k:-k])

    def get_config(self):
        config = super().get_config()
        config.update({"k": self.k})
        return config


class Mean(metrics.PyMetric):
    """An example PyMetric which computes the mean of `y_pred`."""

    def __init__(self, name="mean", **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.values.append(y_true)

    def reset_state(self):
        self.values = []

    def result(self):
        return tf.reduce_mean(tf.concat(self.values, axis=0))


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class PyMetricsTest(tf.test.TestCase):
    def test_config(self):
        ktm_object = KTrimmedMean(name="ktm", k=0.2, dtype=tf.float16)
        self.assertEqual(ktm_object.name, "ktm")
        self.assertEqual(ktm_object.k, 0.2)
        self.assertEqual(ktm_object.dtype, tf.float16)

        # Check save and restore config
        ktm_object2 = KTrimmedMean.from_config(ktm_object.get_config())
        self.assertEqual(ktm_object2.name, "ktm")
        self.assertEqual(ktm_object.k, 0.2)
        self.assertEqual(ktm_object2.dtype, tf.float16)

    def test_unweighted(self):
        ktm_object = KTrimmedMean(k=0.2)

        for y_true in [-100, -10, 1, 2, 3, 4, 5, 6, 14, 9001]:
            self.evaluate(
                ktm_object.update_state(
                    tf.constant(y_true, dtype=tf.float32),
                    y_pred=tf.constant(0, dtype=tf.float32),
                )
            )

        result = ktm_object.result()
        self.assertEqual(3.5, self.evaluate(result))

    def test_weighted(self):
        ktm_object = KTrimmedMean(k=0.2)

        for y_true in [-100, -10, 1, 2, 3, 4, 5, 6, 14, 9001]:
            self.evaluate(
                ktm_object.update_state(
                    tf.constant(y_true, dtype=tf.float32),
                    y_pred=tf.constant(0, dtype=tf.float32),
                    sample_weight=tf.constant(2, dtype=tf.float32),
                )
            )

        result = ktm_object.result()
        self.assertEqual(7, self.evaluate(result))

    def test_state_stored_on_cpu_host(self):
        with tf.device("/device:GPU:0"):
            mean_obj = Mean()

            y_true_0 = tf.constant([0, 1, 2], dtype=tf.float32)
            y_true_1 = tf.constant([3, 4], dtype=tf.float32)
            self.evaluate(
                mean_obj.update_state(
                    y_true=y_true_0, y_pred=tf.constant(0, dtype=tf.float32)
                )
            )
            self.evaluate(
                mean_obj.update_state(
                    y_true=y_true_1, y_pred=tf.constant(0, dtype=tf.float32)
                )
            )

        self.assertEqual(2, self.evaluate(mean_obj.result()))

        if not tf.executing_eagerly():
            self.assertEndsWith(y_true_0.device, "/device:GPU:0")
            self.assertEndsWith(y_true_1.device, "/device:GPU:0")

        self.assertEndsWith(mean_obj.values[0].device, "/device:CPU:0")
        self.assertEndsWith(mean_obj.values[1].device, "/device:CPU:0")


if __name__ == "__main__":
    tf.test.main()
