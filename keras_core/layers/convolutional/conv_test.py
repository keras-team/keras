import math

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized
from numpy.lib.stride_tricks import as_strided

from keras_core import layers
from keras_core import testing


class ConvBasicTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        {
            "filters": 5,
            "kernel_size": 2,
            "strides": 1,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "groups": 1,
            "input_shape": (3, 5, 4),
            "output_shape": (3, 4, 5),
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": 1,
            "padding": "same",
            "data_format": "channels_last",
            "dilation_rate": (2,),
            "groups": 2,
            "input_shape": (3, 4, 4),
            "output_shape": (3, 4, 6),
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": 1,
            "padding": "causal",
            "data_format": "channels_last",
            "dilation_rate": (2,),
            "groups": 2,
            "input_shape": (3, 4, 4),
            "output_shape": (3, 4, 6),
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": (2,),
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "groups": 2,
            "input_shape": (3, 5, 4),
            "output_shape": (3, 2, 6),
        },
    )
    @pytest.mark.requires_trainable_backend
    def test_conv1d_basic(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        data_format,
        dilation_rate,
        groups,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.Conv1D,
            init_kwargs={
                "filters": filters,
                "kernel_size": kernel_size,
                "strides": strides,
                "padding": padding,
                "data_format": data_format,
                "dilation_rate": dilation_rate,
                "groups": groups,
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
            "strides": 1,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "groups": 1,
            "input_shape": (3, 5, 5, 4),
            "output_shape": (3, 4, 4, 5),
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": 1,
            "padding": "same",
            "data_format": "channels_last",
            "dilation_rate": (2, 2),
            "groups": 2,
            "input_shape": (3, 4, 4, 4),
            "output_shape": (3, 4, 4, 6),
        },
        {
            "filters": 6,
            "kernel_size": (2, 2),
            "strides": (2, 1),
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": (1, 1),
            "groups": 2,
            "input_shape": (3, 5, 5, 4),
            "output_shape": (3, 2, 4, 6),
        },
    )
    @pytest.mark.requires_trainable_backend
    def test_conv2d_basic(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        data_format,
        dilation_rate,
        groups,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.Conv2D,
            init_kwargs={
                "filters": filters,
                "kernel_size": kernel_size,
                "strides": strides,
                "padding": padding,
                "data_format": data_format,
                "dilation_rate": dilation_rate,
                "groups": groups,
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
            "strides": 1,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "groups": 1,
            "input_shape": (3, 5, 5, 5, 4),
            "output_shape": (3, 4, 4, 4, 5),
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": 1,
            "padding": "same",
            "data_format": "channels_last",
            "dilation_rate": (2, 2, 2),
            "groups": 2,
            "input_shape": (3, 4, 4, 4, 4),
            "output_shape": (3, 4, 4, 4, 6),
        },
        {
            "filters": 6,
            "kernel_size": (2, 2, 3),
            "strides": (2, 1, 2),
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": (1, 1, 1),
            "groups": 2,
            "input_shape": (3, 5, 5, 5, 4),
            "output_shape": (3, 2, 4, 2, 6),
        },
    )
    @pytest.mark.requires_trainable_backend
    def test_conv3d_basic(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        data_format,
        dilation_rate,
        groups,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.Conv3D,
            init_kwargs={
                "filters": filters,
                "kernel_size": kernel_size,
                "strides": strides,
                "padding": padding,
                "data_format": data_format,
                "dilation_rate": dilation_rate,
                "groups": groups,
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
            layers.Conv1D(filters=0, kernel_size=1)

        # `kernel_size` has 0.
        with self.assertRaises(ValueError):
            layers.Conv2D(filters=2, kernel_size=(1, 0))

        # `strides` has 0.
        with self.assertRaises(ValueError):
            layers.Conv2D(filters=2, kernel_size=(2, 2), strides=(1, 0))

        # `dilation_rate > 1` while `strides > 1`.
        with self.assertRaises(ValueError):
            layers.Conv2D(
                filters=2, kernel_size=(2, 2), strides=2, dilation_rate=(2, 1)
            )

        # `filters` cannot be divided by `groups`.
        with self.assertRaises(ValueError):
            layers.Conv2D(filters=5, kernel_size=(2, 2), groups=2)


class ConvCorrectnessTest(testing.TestCase, parameterized.TestCase):
    def _same_padding(self, input_size, kernel_size, stride):
        # P = ((S-1)*W-S+K)/2, with K = kernel size, S = stride, W = input size
        padding = int(
            math.ceil(((stride - 1) * input_size - stride + kernel_size) / 2)
        )
        return padding

    def _np_conv1d(
        self,
        x,
        kernel_weights,
        bias_weights,
        strides,
        padding,
        data_format,
        dilation_rate,
        groups,
    ):
        if data_format == "channels_first":
            x = x.swapaxes(1, 2)
        if isinstance(strides, (tuple, list)):
            h_stride = strides[0]
        else:
            h_stride = strides
        if isinstance(dilation_rate, (tuple, list)):
            dilation_rate = dilation_rate[0]
        kernel_size, ch_in, ch_out = kernel_weights.shape

        if dilation_rate > 1:
            new_kernel_size = kernel_size + (dilation_rate - 1) * (
                kernel_size - 1
            )
            new_kernel_weights = np.zeros(
                (new_kernel_size, ch_in, ch_out), dtype=kernel_weights.dtype
            )
            new_kernel_weights[::dilation_rate] = kernel_weights
            kernel_weights = new_kernel_weights
            kernel_size = kernel_weights.shape[0]

        if padding != "valid":
            n_batch, h_x, _ = x.shape
            h_pad = self._same_padding(h_x, kernel_size, h_stride)
            npad = [(0, 0)] * x.ndim
            if padding == "causal":
                npad[1] = (h_pad * 2, 0)
            else:
                npad[1] = (h_pad, h_pad)
            x = np.pad(x, pad_width=npad, mode="constant", constant_values=0)

        n_batch, h_x, _ = x.shape
        h_out = int((h_x - kernel_size) / h_stride) + 1

        kernel_weights = kernel_weights.reshape(-1, ch_out)
        bias_weights = bias_weights.reshape(1, ch_out)

        out_grps = []
        for grp in range(1, groups + 1):
            x_in = x[..., (grp - 1) * ch_in : grp * ch_in]
            stride_shape = (n_batch, h_out, kernel_size, ch_in)
            strides = (
                x_in.strides[0],
                h_stride * x_in.strides[1],
                x_in.strides[1],
                x_in.strides[2],
            )
            inner_dim = kernel_size * ch_in
            x_strided = as_strided(
                x_in, shape=stride_shape, strides=strides
            ).reshape(n_batch, h_out, inner_dim)
            ch_out_groups = ch_out // groups
            kernel_weights_grp = kernel_weights[
                ..., (grp - 1) * ch_out_groups : grp * ch_out_groups
            ]
            bias_weights_grp = bias_weights[
                ..., (grp - 1) * ch_out_groups : grp * ch_out_groups
            ]
            out_grps.append(x_strided @ kernel_weights_grp + bias_weights_grp)
        return np.concatenate(out_grps, axis=-1)

    @parameterized.parameters(
        {
            "filters": 5,
            "kernel_size": 2,
            "strides": 1,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "groups": 1,
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": 1,
            "padding": "same",
            "data_format": "channels_last",
            "dilation_rate": (2,),
            "groups": 2,
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": 1,
            "padding": "causal",
            "data_format": "channels_last",
            "dilation_rate": (2,),
            "groups": 2,
        },
        {
            "filters": 6,
            "kernel_size": (2,),
            "strides": (2,),
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "groups": 2,
        },
    )
    def test_conv1d(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        data_format,
        dilation_rate,
        groups,
    ):
        layer = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
        )

        inputs = np.random.normal(size=[2, 8, 4])
        layer.build(input_shape=inputs.shape)

        kernel_shape = layer.kernel.shape
        kernel_weights = np.random.normal(size=kernel_shape)
        bias_weights = np.random.normal(size=(filters,))
        layer.kernel.assign(kernel_weights)
        layer.bias.assign(bias_weights)

        outputs = layer(inputs)
        expected = self._np_conv1d(
            inputs,
            kernel_weights,
            bias_weights,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
        )
        self.assertAllClose(outputs, expected)

    @parameterized.parameters(
        {
            "filters": 5,
            "kernel_size": 2,
            "strides": 1,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "groups": 1,
        },
        {
            "filters": 4,
            "kernel_size": 3,
            "strides": 2,
            "padding": "same",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "groups": 1,
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": 1,
            "padding": "same",
            "data_format": "channels_last",
            "dilation_rate": (2, 2),
            "groups": 2,
        },
        {
            "filters": 6,
            "kernel_size": (4, 3),
            "strides": (2, 1),
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": (1, 1),
            "groups": 2,
        },
    )
    def test_conv2d(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        data_format,
        dilation_rate,
        groups,
    ):
        layer = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
        )
        tf_keras_layer = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
        )

        inputs = np.random.normal(size=[2, 8, 8, 4])
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
        self.assertAllClose(outputs, expected, rtol=5e-4)

    @parameterized.parameters(
        {
            "filters": 5,
            "kernel_size": 2,
            "strides": 1,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "groups": 1,
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": 1,
            "padding": "same",
            "data_format": "channels_last",
            "dilation_rate": (2, 2, 2),
            "groups": 2,
        },
        {
            "filters": 6,
            "kernel_size": (2, 2, 3),
            "strides": (2, 1, 2),
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": (1, 1, 1),
            "groups": 2,
        },
    )
    def test_conv3d(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        data_format,
        dilation_rate,
        groups,
    ):
        layer = layers.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
        )
        tf_keras_layer = tf.keras.layers.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
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
        self.assertAllClose(outputs, expected, rtol=5e-4)
