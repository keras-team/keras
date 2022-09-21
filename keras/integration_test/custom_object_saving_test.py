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
"""Test for custom Keras object saving with `register_keras_serializable`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras.testing_infra import test_utils
from keras.utils import get_custom_objects


# `tf.print` message is only available in stderr in TF2, which this test checks.
@test_utils.run_v2_only
class CustomObjectSavingTest(tf.test.TestCase, parameterized.TestCase):
    """Test for custom Keras object saving with
    `register_keras_serializable`."""

    def setUp(self):
        super().setUp()
        get_custom_objects().clear()

    def test_register_keras_serializable_correct_class(self):
        train_step_message = "This is my training step"
        temp_dir = os.path.join(self.get_temp_dir(), "my_model")

        @tf.keras.utils.register_keras_serializable("CustomModelX")
        class CustomModelX(tf.keras.Model):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.dense1 = MyDense(
                    1,
                    kernel_regularizer=MyRegularizer(0.01),
                    activity_regularizer=MyRegularizer(0.01),
                )

            def call(self, inputs):
                return self.dense1(inputs)

            def train_step(self, data):
                tf.print(train_step_message)
                x, y = data
                with tf.GradientTape() as tape:
                    y_pred = self(x)
                    loss = self.compiled_loss(y, y_pred)

                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients, self.trainable_variables)
                )
                return {}

            def one(self):
                return 1

        @tf.keras.utils.register_keras_serializable("MyDense")
        class MyDense(tf.keras.layers.Dense):
            def two(self):
                return 2

        @tf.keras.utils.register_keras_serializable("MyAdam")
        class MyAdam(tf.keras.optimizers.Adam):
            def three(self):
                return 3

        @tf.keras.utils.register_keras_serializable("MyLoss")
        class MyLoss(tf.keras.losses.MeanSquaredError):
            def four(self):
                return 4

        @tf.keras.utils.register_keras_serializable("MyMetric")
        class MyMetric(tf.keras.metrics.MeanAbsoluteError):
            def five(self):
                return 5

        @tf.keras.utils.register_keras_serializable("MyRegularizer")
        class MyRegularizer(tf.keras.regularizers.L2):
            def six(self):
                return 6

        @tf.keras.utils.register_keras_serializable("my_sq_diff")
        def my_sq_diff(y_true, y_pred):
            y_pred = tf.convert_to_tensor(y_pred)
            y_true = tf.cast(y_true, y_pred.dtype)
            sq_diff_plus_x = tf.math.squared_difference(y_pred, y_true)
            return tf.reduce_mean(sq_diff_plus_x, axis=-1)

        subclassed_model = CustomModelX()
        subclassed_model.compile(
            optimizer=MyAdam(), loss=MyLoss(), metrics=[MyMetric(), my_sq_diff]
        )

        x = np.random.random((100, 32))
        y = np.random.random((100, 1))
        subclassed_model.fit(x, y, epochs=1)
        subclassed_model.save(temp_dir, save_format="tf")

        loaded_model = tf.keras.models.load_model(temp_dir)

        # `tf.print` writes to stderr.
        with self.captureWritesToStream(sys.stderr) as printed:
            loaded_model.fit(x, y, epochs=1)
            self.assertRegex(printed.contents(), train_step_message)

        # Check that the custom classes do get used.
        self.assertIs(loaded_model.__class__, CustomModelX)
        self.assertIs(loaded_model.optimizer.__class__, MyAdam)
        self.assertIs(loaded_model.compiled_loss._losses[0].__class__, MyLoss)
        self.assertIs(
            loaded_model.compiled_metrics._metrics[0].__class__, MyMetric
        )
        self.assertIs(loaded_model.compiled_metrics._metrics[1], my_sq_diff)
        self.assertIs(loaded_model.layers[0].__class__, MyDense)
        self.assertIs(
            loaded_model.layers[0].activity_regularizer.__class__, MyRegularizer
        )
        self.assertIs(
            loaded_model.layers[0].kernel_regularizer.__class__, MyRegularizer
        )

        # Check that the custom methods are available.
        self.assertEqual(loaded_model.one(), 1)
        self.assertEqual(loaded_model.layers[0].two(), 2)
        self.assertEqual(loaded_model.optimizer.three(), 3)
        self.assertEqual(loaded_model.compiled_loss._losses[0].four(), 4)
        self.assertEqual(loaded_model.compiled_metrics._metrics[0].five(), 5)
        self.assertEqual(loaded_model.layers[0].activity_regularizer.six(), 6)
        self.assertEqual(loaded_model.layers[0].kernel_regularizer.six(), 6)
        self.assertEqual(loaded_model.compiled_metrics._metrics[1]([1], [3]), 4)


if __name__ == "__main__":
    tf.test.main()
