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

import tensorflow as tf
from absl.testing import parameterized

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


class SpectralNormalizationTest(test_combinations.TestCase):
    @test_combinations.run_all_keras_modes
    def test_basic_spectralnorm(self):
        test_utils.layer_test(
            keras.layers.SpectralNormalization,
            kwargs={"layer": keras.layers.Dense(2), "input_shape": (3, 4)},
            input_data=tf.random.uniform((10, 3, 4)),
        )

    @test_combinations.run_all_keras_modes
    def test_from_to_config(self):
        base_layer = keras.layers.Dense(1)
        sn = keras.layers.SpectralNormalization(base_layer)
        config = sn.get_config()

        new_sn = keras.layers.SpectralNormalization.from_config(config)
        self.assertEqual(sn.power_iterations, new_sn.power_iterations)

    @test_combinations.run_all_keras_modes
    def test_save_load_model(self):
        base_layer = keras.layers.Dense(1)
        input_shape = [1]

        inputs = keras.layers.Input(shape=input_shape)
        sn_layer = keras.layers.SpectralNormalization(base_layer)
        model = keras.models.Sequential(layers=[inputs, sn_layer])

        # initialize model
        model.predict(tf.random.uniform((2, 1)))

        model.save("test.h5")
        new_model = keras.models.load_model("test.h5")

        self.assertEqual(
            model.layers[0].get_config(), new_model.layers[0].get_config()
        )

    @test_combinations.run_all_keras_modes
    def test_normalization(self):
        inputs = keras.layers.Input(shape=[2, 2, 1])

        base_layer = keras.layers.Conv2D(
            1, (2, 2), kernel_initializer=tf.constant_initializer(value=2)
        )
        sn_layer = keras.layers.SpectralNormalization(base_layer)
        model = keras.models.Sequential(layers=[inputs, sn_layer])

        weights = tf.squeeze(model.layers[0].w.numpy())
        # This wrapper normalizes weights by the maximum eigen value
        eigen_val, _ = tf.linalg.eig(weights)
        weights_normalized = weights / tf.reduce_max(eigen_val)

        for training in [False, True]:
            _ = model(
                tf.constant(tf.ones((1, 2, 2, 1), dtype=tf.float32)),
                training=training,
            )
            if training:
                w = weights_normalized
            else:
                w = weights
            self.assertAllClose(w, tf.squeeze(model.layers[0].w.numpy()))

    @test_combinations.run_all_keras_modes
    def test_apply_layer(self):
        images = tf.ones((1, 2, 2, 1))
        sn_wrapper = keras.layers.SpectralNormalization(
            keras.layers.Conv2D(
                1, [2, 2], kernel_initializer=tf.constant_initializer(value=1)
            ),
            input_shape=(2, 2, 1),
        )

        result = sn_wrapper(images, training=False)
        result_train = sn_wrapper(images, training=True)
        expected_output = tf.constant([[[[4.0]]]], dtype=tf.float32)

        self.assertAllClose(result, expected_output)
        # max eigen value of 2x2 matrix of ones is 2
        self.assertAllClose(result_train, expected_output / 2)
        self.assertTrue(hasattr(sn_wrapper, "u"))

    @test_combinations.run_all_keras_modes
    def test_no_layer(self):
        images = tf.random.uniform((2, 4, 43))
        with self.assertRaises(AssertionError):
            keras.layers.SpectralNormalization(images)

    @test_combinations.run_all_keras_modes
    def test_no_kernel(self):
        with self.assertRaises(AttributeError):
            keras.layers.SpectralNormalization(
                keras.layers.MaxPooling2D(2, 2)
            ).build((2, 2))

    @parameterized.parameters(
        [
            (lambda: keras.layers.Dense(2), [3, 2]),
            (
                lambda: keras.layers.Conv2D(3, (2, 2), padding="same"),
                [4, 4, 3],
            ),
            (lambda: keras.layers.Embedding(2, 10), [2]),
        ],
    )
    @test_combinations.run_all_keras_modes
    def test_model_build(self, base_layer_fn, input_shape):
        inputs = keras.layers.Input(shape=input_shape)
        base_layer = base_layer_fn()
        sn_layer = keras.layers.SpectralNormalization(base_layer)
        model = keras.models.Sequential(layers=[inputs, sn_layer])
        model.build()
        self.assertTrue(hasattr(model.layers[0], "vector_u"))

    @parameterized.parameters(
        [
            (lambda: keras.layers.Dense(2), [3, 2], [3, 2]),
            (
                lambda: keras.layers.Conv2D(3, (2, 2), padding="same"),
                [4, 4, 3],
                [4, 4, 3],
            ),
            (lambda: keras.layers.Embedding(2, 10), [2], [2, 10]),
        ],
    )
    @test_combinations.run_all_keras_modes
    def test_model_fit(self, base_layer_fn, input_shape, output_shape):
        inputs = keras.layers.Input(shape=input_shape)
        base_layer = base_layer_fn()

        sn_layer = keras.layers.SpectralNormalization(base_layer)
        model = keras.models.Sequential(layers=[inputs, sn_layer])
        model.add(keras.layers.Activation("relu"))

        model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
            loss="mse",
        )
        model.fit(
            tf.random.uniform((2, *input_shape)),
            tf.random.uniform((2, *output_shape)),
            epochs=3,
            batch_size=10,
            verbose=0,
        )
        self.assertTrue(hasattr(model.layers[0], "vector_u"))
