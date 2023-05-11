import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from keras_core import layers
from keras_core import testing


class DepthwiseConvBasicTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        {
            "depth_multiplier": 5,
            "kernel_size": 2,
            "strides": 1,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "input_shape": (3, 5, 4),
            "output_shape": (3, 4, 20),
        },
        {
            "depth_multiplier": 6,
            "kernel_size": 2,
            "strides": 1,
            "padding": "same",
            "data_format": "channels_last",
            "dilation_rate": (2,),
            "input_shape": (3, 4, 4),
            "output_shape": (3, 4, 24),
        },
        {
            "depth_multiplier": 6,
            "kernel_size": 2,
            "strides": (2,),
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "input_shape": (3, 5, 4),
            "output_shape": (3, 2, 24),
        },
    )
    def test_depthwise_conv1d_basic(
        self,
        depth_multiplier,
        kernel_size,
        strides,
        padding,
        data_format,
        dilation_rate,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.DepthwiseConv1D,
            init_kwargs={
                "depth_multiplier": depth_multiplier,
                "kernel_size": kernel_size,
                "strides": strides,
                "padding": padding,
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
            "depth_multiplier": 5,
            "kernel_size": 2,
            "strides": 1,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "input_shape": (3, 5, 5, 4),
            "output_shape": (3, 4, 4, 20),
        },
        {
            "depth_multiplier": 6,
            "kernel_size": 2,
            "strides": 1,
            "padding": "same",
            "data_format": "channels_last",
            "dilation_rate": (2, 2),
            "input_shape": (3, 4, 4, 4),
            "output_shape": (3, 4, 4, 24),
        },
        {
            "depth_multiplier": 6,
            "kernel_size": (2, 2),
            "strides": (2, 2),
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": (1, 1),
            "input_shape": (3, 5, 5, 4),
            "output_shape": (3, 2, 2, 24),
        },
    )
    def test_depthwise_conv2d_basic(
        self,
        depth_multiplier,
        kernel_size,
        strides,
        padding,
        data_format,
        dilation_rate,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.DepthwiseConv2D,
            init_kwargs={
                "depth_multiplier": depth_multiplier,
                "kernel_size": kernel_size,
                "strides": strides,
                "padding": padding,
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
        # `depth_multiplier` is not positive.
        with self.assertRaises(ValueError):
            layers.DepthwiseConv1D(depth_multiplier=0, kernel_size=1)

        # `kernel_size` has 0.
        with self.assertRaises(ValueError):
            layers.DepthwiseConv2D(depth_multiplier=2, kernel_size=(1, 0))

        # `strides` has 0.
        with self.assertRaises(ValueError):
            layers.DepthwiseConv2D(
                depth_multiplier=2, kernel_size=(2, 2), strides=(1, 0)
            )

        # `dilation_rate > 1` while `strides > 1`.
        with self.assertRaises(ValueError):
            layers.DepthwiseConv2D(
                depth_multiplier=2,
                kernel_size=(2, 2),
                strides=2,
                dilation_rate=(2, 1),
            )


class DepthwiseConvCorrectnessTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        {
            "depth_multiplier": 5,
            "kernel_size": 2,
            "strides": 1,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
        },
        {
            "depth_multiplier": 6,
            "kernel_size": 2,
            "strides": 1,
            "padding": "same",
            "data_format": "channels_last",
            "dilation_rate": (2,),
        },
        {
            "depth_multiplier": 6,
            "kernel_size": (2,),
            "strides": (2,),
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
        },
    )
    def test_depthwise_conv1d(
        self,
        depth_multiplier,
        kernel_size,
        strides,
        padding,
        data_format,
        dilation_rate,
    ):
        layer = layers.DepthwiseConv1D(
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )
        tf_keras_layer = tf.keras.layers.DepthwiseConv1D(
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )

        inputs = np.random.normal(size=[2, 8, 4])
        layer.build(input_shape=inputs.shape)
        tf_keras_layer.build(input_shape=inputs.shape)

        kernel_shape = layer.kernel.shape
        kernel_weights = np.random.normal(size=kernel_shape)
        bias_weights = np.random.normal(size=(depth_multiplier * 4,))
        layer.kernel.assign(kernel_weights)
        tf_keras_layer.depthwise_kernel.assign(kernel_weights)

        layer.bias.assign(bias_weights)
        tf_keras_layer.bias.assign(bias_weights)

        outputs = layer(inputs)
        expected = tf_keras_layer(inputs)
        self.assertAllClose(outputs, expected)

    @parameterized.parameters(
        {
            "depth_multiplier": 5,
            "kernel_size": 2,
            "strides": 1,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
        },
        {
            "depth_multiplier": 6,
            "kernel_size": 2,
            "strides": 1,
            "padding": "same",
            "data_format": "channels_last",
            "dilation_rate": (2, 2),
        },
        {
            "depth_multiplier": 6,
            "kernel_size": (2, 2),
            "strides": (2, 2),
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": (1, 1),
        },
    )
    def test_depthwise_conv2d(
        self,
        depth_multiplier,
        kernel_size,
        strides,
        padding,
        data_format,
        dilation_rate,
    ):
        layer = layers.DepthwiseConv2D(
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )
        tf_keras_layer = tf.keras.layers.DepthwiseConv2D(
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )

        inputs = np.random.normal(size=[2, 8, 8, 4])
        layer.build(input_shape=inputs.shape)
        tf_keras_layer.build(input_shape=inputs.shape)

        kernel_shape = layer.kernel.shape
        kernel_weights = np.random.normal(size=kernel_shape)
        bias_weights = np.random.normal(size=(depth_multiplier * 4,))
        layer.kernel.assign(kernel_weights)
        tf_keras_layer.depthwise_kernel.assign(kernel_weights)

        layer.bias.assign(bias_weights)
        tf_keras_layer.bias.assign(bias_weights)

        outputs = layer(inputs)
        expected = tf_keras_layer(inputs)
        self.assertAllClose(outputs, expected)
