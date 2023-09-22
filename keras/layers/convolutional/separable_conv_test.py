import numpy as np
import pytest
from absl.testing import parameterized

from keras import layers
from keras import testing
from keras.layers.convolutional.conv_test import np_conv1d
from keras.layers.convolutional.conv_test import np_conv2d
from keras.layers.convolutional.depthwise_conv_test import np_depthwise_conv1d
from keras.layers.convolutional.depthwise_conv_test import np_depthwise_conv2d


class SeparableConvBasicTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        {
            "depth_multiplier": 5,
            "filters": 5,
            "kernel_size": 2,
            "strides": 1,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "input_shape": (3, 5, 4),
            "output_shape": (3, 4, 5),
        },
        {
            "depth_multiplier": 6,
            "filters": 6,
            "kernel_size": 2,
            "strides": 1,
            "padding": "same",
            "data_format": "channels_last",
            "dilation_rate": (2,),
            "input_shape": (3, 4, 4),
            "output_shape": (3, 4, 6),
        },
        {
            "depth_multiplier": 6,
            "filters": 6,
            "kernel_size": 2,
            "strides": (2,),
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "input_shape": (3, 5, 4),
            "output_shape": (3, 2, 6),
        },
    )
    @pytest.mark.requires_trainable_backend
    def test_separable_conv1d_basic(
        self,
        depth_multiplier,
        filters,
        kernel_size,
        strides,
        padding,
        data_format,
        dilation_rate,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.SeparableConv1D,
            init_kwargs={
                "depth_multiplier": depth_multiplier,
                "filters": filters,
                "kernel_size": kernel_size,
                "strides": strides,
                "padding": padding,
                "data_format": data_format,
                "dilation_rate": dilation_rate,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @parameterized.parameters(
        {
            "depth_multiplier": 5,
            "filters": 5,
            "kernel_size": 2,
            "strides": 1,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "input_shape": (3, 5, 5, 4),
            "output_shape": (3, 4, 4, 5),
        },
        {
            "depth_multiplier": 6,
            "filters": 6,
            "kernel_size": 2,
            "strides": 1,
            "padding": "same",
            "data_format": "channels_last",
            "dilation_rate": (2, 2),
            "input_shape": (3, 4, 4, 4),
            "output_shape": (3, 4, 4, 6),
        },
        {
            "depth_multiplier": 6,
            "filters": 6,
            "kernel_size": (2, 2),
            "strides": (2, 2),
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": (1, 1),
            "input_shape": (3, 5, 5, 4),
            "output_shape": (3, 2, 2, 6),
        },
    )
    @pytest.mark.requires_trainable_backend
    def test_separable_conv2d_basic(
        self,
        depth_multiplier,
        filters,
        kernel_size,
        strides,
        padding,
        data_format,
        dilation_rate,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.SeparableConv2D,
            init_kwargs={
                "depth_multiplier": depth_multiplier,
                "filters": filters,
                "kernel_size": kernel_size,
                "strides": strides,
                "padding": padding,
                "data_format": data_format,
                "dilation_rate": dilation_rate,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    def test_bad_init_args(self):
        # `depth_multiplier` is not positive.
        with self.assertRaises(ValueError):
            layers.SeparableConv1D(depth_multiplier=0, filters=1, kernel_size=1)

        # `filters` is not positive.
        with self.assertRaises(ValueError):
            layers.SeparableConv1D(depth_multiplier=1, filters=0, kernel_size=1)

        # `kernel_size` has 0.
        with self.assertRaises(ValueError):
            layers.SeparableConv2D(
                depth_multiplier=2, filters=2, kernel_size=(1, 0)
            )

        # `strides` has 0.
        with self.assertRaises(ValueError):
            layers.SeparableConv2D(
                depth_multiplier=2,
                filters=2,
                kernel_size=(2, 2),
                strides=(1, 0),
            )

        # `dilation_rate > 1` while `strides > 1`.
        with self.assertRaises(ValueError):
            layers.SeparableConv2D(
                depth_multiplier=2,
                filters=2,
                kernel_size=(2, 2),
                strides=2,
                dilation_rate=(2, 1),
            )


class SeparableConvCorrectnessTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        {
            "depth_multiplier": 5,
            "filters": 5,
            "kernel_size": 2,
            "strides": 1,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
        },
        {
            "depth_multiplier": 6,
            "filters": 6,
            "kernel_size": 2,
            "strides": 1,
            "padding": "same",
            "data_format": "channels_last",
            "dilation_rate": (2,),
        },
        {
            "depth_multiplier": 6,
            "filters": 6,
            "kernel_size": (2,),
            "strides": (2,),
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
        },
    )
    def test_separable_conv1d(
        self,
        depth_multiplier,
        filters,
        kernel_size,
        strides,
        padding,
        data_format,
        dilation_rate,
    ):
        layer = layers.SeparableConv1D(
            depth_multiplier=depth_multiplier,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )

        inputs = np.random.normal(size=[2, 8, 4])
        layer.build(input_shape=inputs.shape)

        depthwise_kernel_shape = layer.depthwise_kernel.shape
        depthwise_kernel_weights = np.random.normal(size=depthwise_kernel_shape)
        layer.depthwise_kernel.assign(depthwise_kernel_weights)

        pointwise_kernel_shape = layer.pointwise_kernel.shape
        pointwise_kernel_weights = np.random.normal(size=pointwise_kernel_shape)
        layer.pointwise_kernel.assign(pointwise_kernel_weights)

        bias_weights = np.random.normal(size=(filters,))
        layer.bias.assign(bias_weights)

        outputs = layer(inputs)
        expected_depthwise = np_depthwise_conv1d(
            inputs,
            depthwise_kernel_weights,
            np.zeros(4 * depth_multiplier),
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )
        expected = np_conv1d(
            expected_depthwise,
            pointwise_kernel_weights,
            bias_weights,
            strides=1,
            padding=padding,
            data_format=data_format,
            dilation_rate=1,
            groups=1,
        )

        self.assertAllClose(outputs.shape, expected.shape)
        self.assertAllClose(outputs, expected, rtol=1e-5, atol=1e-5)

    @parameterized.parameters(
        {
            "depth_multiplier": 5,
            "filters": 5,
            "kernel_size": 2,
            "strides": 1,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
        },
        {
            "depth_multiplier": 6,
            "filters": 6,
            "kernel_size": 2,
            "strides": 1,
            "padding": "same",
            "data_format": "channels_last",
            "dilation_rate": (2, 2),
        },
        {
            "depth_multiplier": 6,
            "filters": 6,
            "kernel_size": (2, 2),
            "strides": (2, 2),
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": (1, 1),
        },
    )
    def test_separable_conv2d(
        self,
        depth_multiplier,
        filters,
        kernel_size,
        strides,
        padding,
        data_format,
        dilation_rate,
    ):
        layer = layers.SeparableConv2D(
            depth_multiplier=depth_multiplier,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )

        inputs = np.random.normal(size=[2, 8, 8, 4])
        layer.build(input_shape=inputs.shape)

        depthwise_kernel_shape = layer.depthwise_kernel.shape
        depthwise_kernel_weights = np.random.normal(size=depthwise_kernel_shape)
        layer.depthwise_kernel.assign(depthwise_kernel_weights)

        pointwise_kernel_shape = layer.pointwise_kernel.shape
        pointwise_kernel_weights = np.random.normal(size=pointwise_kernel_shape)
        layer.pointwise_kernel.assign(pointwise_kernel_weights)

        bias_weights = np.random.normal(size=(filters,))
        layer.bias.assign(bias_weights)

        outputs = layer(inputs)
        expected_depthwise = np_depthwise_conv2d(
            inputs,
            depthwise_kernel_weights,
            np.zeros(4 * depth_multiplier),
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )
        expected = np_conv2d(
            expected_depthwise,
            pointwise_kernel_weights,
            bias_weights,
            strides=1,
            padding=padding,
            data_format=data_format,
            dilation_rate=1,
            groups=1,
        )

        self.assertAllClose(outputs.shape, expected.shape)
        self.assertAllClose(outputs, expected, rtol=1e-5, atol=1e-5)
