import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_core import backend
from keras_core import layers
from keras_core import testing


class ConvTransposeBasicTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        {
            "filters": 5,
            "kernel_size": 2,
            "strides": 2,
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": 1,
            "input_shape": (2, 8, 4),
            "output_shape": (2, 16, 5),
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": 3,
            "padding": "same",
            "output_padding": 2,
            "data_format": "channels_last",
            "dilation_rate": (1,),
            "input_shape": (2, 8, 4),
            "output_shape": (2, 23, 6),
        },
        {
            "filters": 6,
            "kernel_size": (2,),
            "strides": (2,),
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": 1,
            "input_shape": (2, 8, 4),
            "output_shape": (2, 16, 6),
        },
    )
    @pytest.mark.requires_trainable_backend
    def test_conv1d_transpose_basic(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.Conv1DTranspose,
            init_kwargs={
                "filters": filters,
                "kernel_size": kernel_size,
                "strides": strides,
                "padding": padding,
                "output_padding": output_padding,
                "data_format": data_format,
                "dilation_rate": dilation_rate,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @parameterized.parameters(
        {
            "filters": 5,
            "kernel_size": 2,
            "strides": 2,
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": 1,
            "input_shape": (2, 8, 8, 4),
            "output_shape": (2, 16, 16, 5),
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": 3,
            "padding": "same",
            "output_padding": 2,
            "data_format": "channels_last",
            "dilation_rate": (1, 1),
            "input_shape": (2, 8, 8, 4),
            "output_shape": (2, 23, 23, 6),
        },
        {
            "filters": 6,
            "kernel_size": (2, 3),
            "strides": (2, 1),
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_first",
            "dilation_rate": (1, 1),
            "input_shape": (2, 4, 8, 8),
            "output_shape": (2, 6, 16, 10),
        },
        {
            "filters": 2,
            "kernel_size": (7, 7),
            "strides": (16, 16),
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": (1, 1),
            "input_shape": (1, 14, 14, 2),
            "output_shape": (1, 224, 224, 2),
        },
    )
    @pytest.mark.requires_trainable_backend
    def test_conv2d_transpose_basic(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
        input_shape,
        output_shape,
    ):
        if (
            data_format == "channels_first"
            and backend.backend() == "tensorflow"
        ):
            pytest.skip("channels_first unsupported on CPU with TF")

        self.run_layer_test(
            layers.Conv2DTranspose,
            init_kwargs={
                "filters": filters,
                "kernel_size": kernel_size,
                "strides": strides,
                "padding": padding,
                "output_padding": output_padding,
                "data_format": data_format,
                "dilation_rate": dilation_rate,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @parameterized.parameters(
        {
            "filters": 5,
            "kernel_size": 2,
            "strides": 2,
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": 1,
            "input_shape": (2, 8, 8, 8, 4),
            "output_shape": (2, 16, 16, 16, 5),
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": 3,
            "padding": "same",
            "output_padding": 2,
            "data_format": "channels_last",
            "dilation_rate": (1, 1, 1),
            "input_shape": (2, 8, 8, 8, 4),
            "output_shape": (2, 23, 23, 23, 6),
        },
        {
            "filters": 6,
            "kernel_size": (2, 2, 3),
            "strides": (2, 1, 2),
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": (1, 1, 1),
            "input_shape": (2, 8, 8, 8, 4),
            "output_shape": (2, 16, 9, 17, 6),
        },
    )
    @pytest.mark.requires_trainable_backend
    def test_conv3d_transpose_basic(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.Conv3DTranspose,
            init_kwargs={
                "filters": filters,
                "kernel_size": kernel_size,
                "strides": strides,
                "padding": padding,
                "output_padding": output_padding,
                "data_format": data_format,
                "dilation_rate": dilation_rate,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    def test_bad_init_args(self):
        # `filters` is not positive.
        with self.assertRaises(ValueError):
            layers.Conv1DTranspose(filters=0, kernel_size=1)

        # `kernel_size` has 0.
        with self.assertRaises(ValueError):
            layers.Conv2DTranspose(filters=2, kernel_size=(1, 0))

        # `strides` has 0.
        with self.assertRaises(ValueError):
            layers.Conv2DTranspose(
                filters=2, kernel_size=(2, 2), strides=(1, 0)
            )

        # `dilation_rate > 1` while `strides > 1`.
        with self.assertRaises(ValueError):
            layers.Conv2DTranspose(
                filters=2, kernel_size=(2, 2), strides=2, dilation_rate=(2, 1)
            )


class ConvTransposeCorrectnessTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        {
            "filters": 5,
            "kernel_size": 2,
            "strides": 2,
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": 1,
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": 3,
            "padding": "same",
            "output_padding": 2,
            "data_format": "channels_last",
            "dilation_rate": (1,),
        },
        {
            "filters": 6,
            "kernel_size": (2,),
            "strides": (2,),
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": 1,
        },
    )
    def test_conv1d_transpose(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
    ):
        layer = layers.Conv1DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )
        tf_keras_layer = tf.keras.layers.Conv1DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )

        inputs = np.random.normal(size=[2, 8, 4])
        layer.build(input_shape=inputs.shape)
        tf_keras_layer.build(input_shape=inputs.shape)

        kernel_shape = layer.kernel.shape
        kernel_weights = np.random.normal(size=kernel_shape)
        bias_weights = np.random.normal(size=(filters,))
        layer.kernel.assign(kernel_weights)
        tf_keras_layer.kernel.assign(kernel_weights)

        layer.bias.assign(bias_weights)
        tf_keras_layer.bias.assign(bias_weights)

        outputs = layer(inputs)
        expected = tf_keras_layer(inputs)
        self.assertAllClose(outputs, expected)

    @parameterized.parameters(
        {
            "filters": 5,
            "kernel_size": 2,
            "strides": 2,
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": 1,
        },
        {
            "filters": 6,
            "kernel_size": 7,
            "strides": 16,
            "padding": "same",
            "output_padding": 2,
            "data_format": "channels_last",
            "dilation_rate": (1, 1),
        },
        {
            "filters": 6,
            "kernel_size": (2, 3),
            "strides": (2, 1),
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": (1, 1),
        },
        {
            "filters": 2,
            "kernel_size": (7, 7),
            "strides": (16, 16),
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": (1, 1),
        },
    )
    def test_conv2d_transpose(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
    ):
        layer = layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )
        tf_keras_layer = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )

        inputs = np.random.normal(size=[2, 14, 14, 4])
        layer.build(input_shape=inputs.shape)
        tf_keras_layer.build(input_shape=inputs.shape)

        kernel_shape = layer.kernel.shape
        kernel_weights = np.random.normal(size=kernel_shape)
        bias_weights = np.random.normal(size=(filters,))
        layer.kernel.assign(kernel_weights)
        tf_keras_layer.kernel.assign(kernel_weights)

        layer.bias.assign(bias_weights)
        tf_keras_layer.bias.assign(bias_weights)

        outputs = layer(inputs)
        expected = tf_keras_layer(inputs)
        self.assertAllClose(outputs, expected)

    @parameterized.parameters(
        {
            "filters": 5,
            "kernel_size": 2,
            "strides": 2,
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": 1,
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": 3,
            "padding": "same",
            "output_padding": 2,
            "data_format": "channels_last",
            "dilation_rate": (1, 1, 1),
        },
        {
            "filters": 6,
            "kernel_size": (2, 2, 3),
            "strides": (2, 1, 2),
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": (1, 1, 1),
        },
    )
    def test_conv3d_transpose(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
    ):
        layer = layers.Conv3DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )
        tf_keras_layer = tf.keras.layers.Conv3DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )

        inputs = np.random.normal(size=[2, 8, 8, 8, 4])
        layer.build(input_shape=inputs.shape)
        tf_keras_layer.build(input_shape=inputs.shape)

        kernel_shape = layer.kernel.shape
        kernel_weights = np.random.normal(size=kernel_shape)
        bias_weights = np.random.normal(size=(filters,))
        layer.kernel.assign(kernel_weights)
        tf_keras_layer.kernel.assign(kernel_weights)

        layer.bias.assign(bias_weights)
        tf_keras_layer.bias.assign(bias_weights)
        outputs = layer(inputs)
        expected = tf_keras_layer(inputs)
        self.assertAllClose(outputs, expected, atol=1e-5)
