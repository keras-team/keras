import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from keras_core import layers
from keras_core import testing


class AveragePoolingBasicTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        (2, 1, "valid", "channels_last", (3, 5, 4), (3, 4, 4)),
        (2, 1, "same", "channels_first", (3, 5, 4), (3, 5, 4)),
        ((2,), (2,), "valid", "channels_last", (3, 5, 4), (3, 2, 4)),
    )
    def test_average_pooling1d(
        self,
        pool_size,
        strides,
        padding,
        data_format,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.AveragePooling1D,
            init_kwargs={
                "pool_size": pool_size,
                "strides": strides,
                "padding": padding,
                "data_format": data_format,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @parameterized.parameters(
        (2, 1, "valid", "channels_last", (3, 5, 5, 4), (3, 4, 4, 4)),
        (2, 1, "same", "channels_first", (3, 5, 5, 4), (3, 5, 5, 4)),
        ((2, 3), (2, 2), "valid", "channels_last", (3, 5, 5, 4), (3, 2, 2, 4)),
    )
    def test_average_pooling2d(
        self,
        pool_size,
        strides,
        padding,
        data_format,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.AveragePooling2D,
            init_kwargs={
                "pool_size": pool_size,
                "strides": strides,
                "padding": padding,
                "data_format": data_format,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @parameterized.parameters(
        (2, 1, "valid", "channels_last", (3, 5, 5, 5, 4), (3, 4, 4, 4, 4)),
        (2, 1, "same", "channels_first", (3, 5, 5, 5, 4), (3, 5, 5, 5, 4)),
        (
            (2, 3, 2),
            (2, 2, 1),
            "valid",
            "channels_last",
            (3, 5, 5, 5, 4),
            (3, 2, 2, 4, 4),
        ),
    )
    def test_average_pooling3d(
        self,
        pool_size,
        strides,
        padding,
        data_format,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.AveragePooling3D,
            init_kwargs={
                "pool_size": pool_size,
                "strides": strides,
                "padding": padding,
                "data_format": data_format,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )


class AveragePoolingCorrectnessTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        (2, 1, "valid", "channels_last"),
        (2, 1, "same", "channels_first"),
        ((2,), (2,), "valid", "channels_last"),
    )
    def test_average_pooling1d(self, pool_size, strides, padding, data_format):
        inputs = np.arange(24, dtype=np.float).reshape((2, 3, 4))

        layer = layers.AveragePooling1D(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )
        tf_keras_layer = tf.keras.layers.AveragePooling1D(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )

        outputs = layer(inputs)
        expected = tf_keras_layer(inputs)
        self.assertAllClose(outputs, expected)

    @parameterized.parameters(
        (2, 1, "valid", "channels_last"),
        ((2, 3), (2, 2), "same", "channels_last"),
    )
    def test_average_pooling2d(self, pool_size, strides, padding, data_format):
        inputs = np.arange(300, dtype=np.float).reshape((3, 5, 5, 4))

        layer = layers.AveragePooling2D(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )
        tf_keras_layer = tf.keras.layers.AveragePooling2D(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )

        outputs = layer(inputs)
        expected = tf_keras_layer(inputs)
        self.assertAllClose(outputs, expected)

    @parameterized.parameters(
        (2, 1, "valid", "channels_last"),
        (2, 1, "same", "channels_first"),
        ((2, 3, 2), (2, 2, 1), "valid", "channels_last"),
    )
    def test_average_pooling3d(self, pool_size, strides, padding, data_format):
        inputs = np.arange(240, dtype=np.float).reshape((2, 3, 4, 5, 2))

        layer = layers.AveragePooling3D(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )
        tf_keras_layer = tf.keras.layers.AveragePooling3D(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )

        outputs = layer(inputs)
        expected = tf_keras_layer(inputs)
        self.assertAllClose(outputs, expected)
