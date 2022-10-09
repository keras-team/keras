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
"""Tests for dropout layer."""

import os

import numpy as np
import tensorflow.compat.v2 as tf

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.run_all_keras_modes
class DropoutTest(test_combinations.TestCase):
    def test_dropout(self):
        test_utils.layer_test(
            keras.layers.Dropout, kwargs={"rate": 0.5}, input_shape=(3, 2)
        )

        test_utils.layer_test(
            keras.layers.Dropout,
            kwargs={"rate": 0.5, "noise_shape": [3, 1]},
            input_shape=(3, 2),
        )

    def test_dropout_supports_masking(self):
        dropout = keras.layers.Dropout(0.5)
        self.assertEqual(True, dropout.supports_masking)

    def test_dropout_partial_noise_shape(self):
        inputs = keras.Input(shape=(5, 10))
        layer = keras.layers.Dropout(0.5, noise_shape=(None, 1, None))
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)
        out = model(np.ones((20, 5, 10)), training=True)
        out_np = keras.backend.get_value(out)
        # Test that dropout mask is shared across second dim.
        self.assertAllClose(out_np[:, 0, :], out_np[:, 1, :])

    @test_utils.run_v2_only
    def test_dropout_with_zero_rate(self):
        inputs = np.ones((20, 5, 10))
        dropout = keras.layers.Dropout(0.0, force_generator=True)
        dropout.build((20, 5, 10))
        # Make sure we don't use the RNG when the dropout rate is 0
        # (for performance).
        rng_state_var = tf.constant(
            dropout._random_generator._generator._state_var
        )
        output = dropout(inputs, training=True)
        self.assertAllClose(inputs, output)
        self.assertAllClose(
            rng_state_var, dropout._random_generator._generator._state_var
        )

    def test_dropout_with_savemodel(self):
        inputs = keras.Input(shape=(5, 10))
        layer = keras.layers.Dropout(0.5, force_generator=True)
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)
        train = model(np.ones((20, 5, 10)), training=True)
        predict = model(np.ones((20, 5, 10)))
        # Make sure the weights from tf.random.Generator is not present in the
        # model which will cause weight loading issue for existing application
        # models if it contains dropout layer.
        self.assertEmpty(layer.get_weights())
        self.assertEmpty(model.get_weights())

        # Make sure the layer does dropout value when training
        self.assertNotAllClose(train, predict)

        model.save(
            os.path.join(self.get_temp_dir(), "savedmodel"), save_format="tf"
        )
        loaded_model = keras.models.load_model(
            os.path.join(self.get_temp_dir(), "savedmodel")
        )
        predict2 = loaded_model(np.ones((20, 5, 10)))

        self.assertAllClose(predict, predict2)
        # Make sure the model dropout different value after loading
        train2 = loaded_model(np.ones((20, 5, 10)), training=True)
        self.assertNotAllClose(train, train2)
        self.assertIsNotNone(loaded_model.layers[1]._random_generator)

        # Also make sure the checkpoint doesn't contain any variable from the
        # dropout layer, to keep the backward compatibility.
        checkpoint = tf.train.Checkpoint(model)
        save_path = checkpoint.save(
            os.path.join(self.get_temp_dir(), "checkpoint")
        )
        checkpoint_var_names = [
            name_value_tuple[0]
            for name_value_tuple in tf.train.list_variables(save_path)
        ]
        for name in checkpoint_var_names:
            self.assertNotIn("dropout", name)

        # Make sure the checkpoint can be loaded
        clone_model = keras.models.clone_model(model)
        checkpoint = tf.train.Checkpoint(clone_model)
        status = checkpoint.restore(
            os.path.join(self.get_temp_dir(), "checkpoint-1")
        )
        self.assertTrue(status.assert_consumed())
        self.assertTrue(status.assert_existing_objects_matched())
        # Make sure the output is differnt from the original model, since
        # the StateVar is not preserved.
        train3 = clone_model(np.ones((20, 5, 10)), training=True)
        self.assertNotAllClose(train3, train2)

    @test_utils.run_v2_only
    def test_state_variable_name(self):
        inputs = keras.Input(shape=(5, 10))
        layer = keras.layers.Dropout(
            0.5, force_generator=True, name="dropout_layer"
        )
        layer(inputs)
        self.assertEqual(
            layer._random_generator._generator._state_var.name,
            "dropout_layer/StateVar:0",
        )


if __name__ == "__main__":
    tf.test.main()
