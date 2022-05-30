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
"""Tests for cropping layers."""

import numpy as np
import tensorflow.compat.v2 as tf

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.run_all_keras_modes
class CroppingTest(test_combinations.TestCase):
    def test_cropping_1d(self):
        num_samples = 2
        time_length = 4
        input_len_dim1 = 2
        inputs = np.random.rand(num_samples, time_length, input_len_dim1)

        with self.cached_session():
            test_utils.layer_test(
                keras.layers.Cropping1D,
                kwargs={"cropping": (1, 1)},
                input_shape=inputs.shape,
            )

        # test incorrect use
        with self.assertRaises(ValueError):
            keras.layers.Cropping1D(cropping=(1, 1, 1))
        with self.assertRaises(ValueError):
            keras.layers.Cropping1D(cropping=None)
        with self.assertRaises(ValueError):
            input_layer = keras.layers.Input(
                shape=(num_samples, time_length, input_len_dim1)
            )
            keras.layers.Cropping1D(cropping=(2, 3))(input_layer)

    def test_cropping_2d(self):
        num_samples = 2
        stack_size = 2
        input_len_dim1 = 9
        input_len_dim2 = 9
        cropping = ((2, 2), (3, 3))

        for data_format in ["channels_first", "channels_last"]:
            if data_format == "channels_first":
                inputs = np.random.rand(
                    num_samples, stack_size, input_len_dim1, input_len_dim2
                )
            else:
                inputs = np.random.rand(
                    num_samples, input_len_dim1, input_len_dim2, stack_size
                )
            with self.cached_session():
                # basic test
                test_utils.layer_test(
                    keras.layers.Cropping2D,
                    kwargs={"cropping": cropping, "data_format": data_format},
                    input_shape=inputs.shape,
                )
                # correctness test
                layer = keras.layers.Cropping2D(
                    cropping=cropping, data_format=data_format
                )
                layer.build(inputs.shape)
                output = layer(keras.backend.variable(inputs))
                if tf.executing_eagerly():
                    np_output = output.numpy()
                else:
                    np_output = keras.backend.eval(output)
                # compare with numpy
                if data_format == "channels_first":
                    expected_out = inputs[
                        :,
                        :,
                        cropping[0][0] : -cropping[0][1],
                        cropping[1][0] : -cropping[1][1],
                    ]
                else:
                    expected_out = inputs[
                        :,
                        cropping[0][0] : -cropping[0][1],
                        cropping[1][0] : -cropping[1][1],
                        :,
                    ]
                np.testing.assert_allclose(np_output, expected_out)

        for data_format in ["channels_first", "channels_last"]:
            if data_format == "channels_first":
                inputs = np.random.rand(
                    num_samples, stack_size, input_len_dim1, input_len_dim2
                )
            else:
                inputs = np.random.rand(
                    num_samples, input_len_dim1, input_len_dim2, stack_size
                )
            # another correctness test (no cropping)
            with self.cached_session():
                cropping = ((0, 0), (0, 0))
                layer = keras.layers.Cropping2D(
                    cropping=cropping, data_format=data_format
                )
                layer.build(inputs.shape)
                output = layer(keras.backend.variable(inputs))
                if tf.executing_eagerly():
                    np_output = output.numpy()
                else:
                    np_output = keras.backend.eval(output)
                # compare with input
                np.testing.assert_allclose(np_output, inputs)

        # test incorrect use
        with self.assertRaises(ValueError):
            keras.layers.Cropping2D(cropping=(1, 1, 1))
        with self.assertRaises(ValueError):
            keras.layers.Cropping2D(cropping=None)
        with self.assertRaises(ValueError):
            input_layer = keras.layers.Input(
                shape=(num_samples, input_len_dim1, input_len_dim2, stack_size)
            )
            keras.layers.Cropping2D(cropping=((5, 4), (3, 4)))(input_layer)

    def test_cropping_3d(self):
        num_samples = 2
        stack_size = 2
        input_len_dim1 = 8
        input_len_dim2 = 8
        input_len_dim3 = 8
        croppings = [((2, 2), (1, 1), (2, 3)), 3, (0, 1, 1)]

        for cropping in croppings:
            for data_format in ["channels_last", "channels_first"]:
                if data_format == "channels_first":
                    inputs = np.random.rand(
                        num_samples,
                        stack_size,
                        input_len_dim1,
                        input_len_dim2,
                        input_len_dim3,
                    )
                else:
                    inputs = np.random.rand(
                        num_samples,
                        input_len_dim1,
                        input_len_dim2,
                        input_len_dim3,
                        stack_size,
                    )
                # basic test
                with self.cached_session():
                    test_utils.layer_test(
                        keras.layers.Cropping3D,
                        kwargs={
                            "cropping": cropping,
                            "data_format": data_format,
                        },
                        input_shape=inputs.shape,
                    )

                if len(croppings) == 3 and len(croppings[0]) == 2:
                    # correctness test
                    with self.cached_session():
                        layer = keras.layers.Cropping3D(
                            cropping=cropping, data_format=data_format
                        )
                        layer.build(inputs.shape)
                        output = layer(keras.backend.variable(inputs))
                        if tf.executing_eagerly():
                            np_output = output.numpy()
                        else:
                            np_output = keras.backend.eval(output)
                        # compare with numpy
                        if data_format == "channels_first":
                            expected_out = inputs[
                                :,
                                :,
                                cropping[0][0] : -cropping[0][1],
                                cropping[1][0] : -cropping[1][1],
                                cropping[2][0] : -cropping[2][1],
                            ]
                        else:
                            expected_out = inputs[
                                :,
                                cropping[0][0] : -cropping[0][1],
                                cropping[1][0] : -cropping[1][1],
                                cropping[2][0] : -cropping[2][1],
                                :,
                            ]
                        np.testing.assert_allclose(np_output, expected_out)

        # test incorrect use
        with self.assertRaises(ValueError):
            keras.layers.Cropping3D(cropping=(1, 1))
        with self.assertRaises(ValueError):
            keras.layers.Cropping3D(cropping=None)


if __name__ == "__main__":
    tf.test.main()
