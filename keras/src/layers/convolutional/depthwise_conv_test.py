import numpy as np
import pytest
from absl.testing import parameterized
from numpy.lib.stride_tricks import as_strided

from keras.src import layers
from keras.src import testing


def _same_padding(input_size, kernel_size, stride):
    if input_size % stride == 0:
        padding = max(kernel_size - stride, 0)
    else:
        padding = max(kernel_size - (input_size % stride), 0)
    return padding // 2, padding - padding // 2


def np_depthwise_conv1d(
    x,
    kernel_weights,
    bias_weights,
    strides,
    padding,
    data_format,
    dilation_rate,
):
    if data_format == "channels_first":
        x = x.transpose((0, 2, 1))
    if isinstance(strides, (tuple, list)):
        h_stride = strides[0]
    else:
        h_stride = strides
    if isinstance(dilation_rate, (tuple, list)):
        h_dilation = dilation_rate[0]
    else:
        h_dilation = dilation_rate
    h_kernel, ch_in, ch_out = kernel_weights.shape

    if h_dilation > 1:
        new_h_kernel = h_kernel + (h_dilation - 1) * (h_kernel - 1)
        new_kernel_weights = np.zeros(
            (new_h_kernel, ch_in, ch_out),
            dtype=kernel_weights.dtype,
        )
        new_kernel_weights[::h_dilation] = kernel_weights
        kernel_weights = new_kernel_weights
        h_kernel = kernel_weights.shape[0]

    if padding == "same":
        n_batch, h_x, _ = x.shape
        h_pad = _same_padding(h_x, h_kernel, h_stride)
        npad = [(0, 0)] * x.ndim
        npad[1] = h_pad
        x = np.pad(x, pad_width=npad, mode="constant", constant_values=0)

    n_batch, h_x, _ = x.shape
    h_out = int((h_x - h_kernel) / h_stride) + 1

    out_grps = []
    bias_weights = bias_weights.reshape(ch_in, ch_out)
    for ch_in_idx in range(ch_in):
        for ch_out_idx in range(ch_out):
            x_in = np.ascontiguousarray(x[..., ch_in_idx])
            stride_shape = (n_batch, h_out, h_kernel)
            strides = (
                x_in.strides[0],
                h_stride * x_in.strides[1],
                x_in.strides[1],
            )
            inner_dim = h_kernel
            x_strided = as_strided(
                x_in, shape=stride_shape, strides=strides
            ).reshape(-1, inner_dim)
            kernel_weights_grp = kernel_weights[
                ..., ch_in_idx, ch_out_idx
            ].reshape(-1, 1)
            bias_weights_grp = bias_weights[..., ch_in_idx, ch_out_idx]
            out_grps.append(
                (x_strided @ kernel_weights_grp + bias_weights_grp).reshape(
                    n_batch, h_out, 1
                )
            )
    out = np.concatenate(out_grps, axis=-1)
    if data_format == "channels_first":
        out = out.transpose((0, 2, 1))
    return out


def np_depthwise_conv2d(
    x,
    kernel_weights,
    bias_weights,
    strides,
    padding,
    data_format,
    dilation_rate,
):
    if data_format == "channels_first":
        x = x.transpose((0, 2, 3, 1))
    if isinstance(strides, (tuple, list)):
        h_stride, w_stride = strides
    else:
        h_stride = strides
        w_stride = strides
    if isinstance(dilation_rate, (tuple, list)):
        h_dilation, w_dilation = dilation_rate
    else:
        h_dilation = dilation_rate
        w_dilation = dilation_rate
    h_kernel, w_kernel, ch_in, ch_out = kernel_weights.shape

    if h_dilation > 1 or w_dilation > 1:
        new_h_kernel = h_kernel + (h_dilation - 1) * (h_kernel - 1)
        new_w_kernel = w_kernel + (w_dilation - 1) * (w_kernel - 1)
        new_kenel_size_tuple = (new_h_kernel, new_w_kernel)
        new_kernel_weights = np.zeros(
            (*new_kenel_size_tuple, ch_in, ch_out),
            dtype=kernel_weights.dtype,
        )
        new_kernel_weights[::h_dilation, ::w_dilation] = kernel_weights
        kernel_weights = new_kernel_weights
        h_kernel, w_kernel = kernel_weights.shape[:2]

    if padding == "same":
        n_batch, h_x, w_x, _ = x.shape
        h_pad = _same_padding(h_x, h_kernel, h_stride)
        w_pad = _same_padding(w_x, w_kernel, w_stride)
        npad = [(0, 0)] * x.ndim
        npad[1] = h_pad
        npad[2] = w_pad
        x = np.pad(x, pad_width=npad, mode="constant", constant_values=0)

    n_batch, h_x, w_x, _ = x.shape
    h_out = int((h_x - h_kernel) / h_stride) + 1
    w_out = int((w_x - w_kernel) / w_stride) + 1

    out_grps = []
    bias_weights = bias_weights.reshape(ch_in, ch_out)
    for ch_in_idx in range(ch_in):
        for ch_out_idx in range(ch_out):
            x_in = np.ascontiguousarray(x[..., ch_in_idx])
            stride_shape = (n_batch, h_out, w_out, h_kernel, w_kernel)
            strides = (
                x_in.strides[0],
                h_stride * x_in.strides[1],
                w_stride * x_in.strides[2],
                x_in.strides[1],
                x_in.strides[2],
            )
            inner_dim = h_kernel * w_kernel
            x_strided = as_strided(
                x_in, shape=stride_shape, strides=strides
            ).reshape(-1, inner_dim)
            kernel_weights_grp = kernel_weights[
                ..., ch_in_idx, ch_out_idx
            ].reshape(-1, 1)
            bias_weights_grp = bias_weights[..., ch_in_idx, ch_out_idx]
            out_grps.append(
                (x_strided @ kernel_weights_grp + bias_weights_grp).reshape(
                    n_batch, h_out, w_out, 1
                )
            )
    out = np.concatenate(out_grps, axis=-1)
    if data_format == "channels_first":
        out = out.transpose((0, 3, 1, 2))
    return out


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
    @pytest.mark.requires_trainable_backend
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
    @pytest.mark.requires_trainable_backend
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
        with self.assertRaisesRegex(
            ValueError,
            "Invalid value for argument `depth_multiplier`. "
            "Expected a strictly positive value. Received "
            "depth_multiplier=0.",
        ):
            layers.DepthwiseConv1D(depth_multiplier=0, kernel_size=1)

        # `kernel_size` has 0.
        with self.assertRaisesRegex(
            ValueError,
            r"The `kernel_size` argument must be a tuple of 2 "
            r"integers. Received kernel_size=\(1, 0\), including values "
            r"\{0\} that do not satisfy `value > 0`",
        ):
            layers.DepthwiseConv2D(depth_multiplier=2, kernel_size=(1, 0))

        # `strides` has 0.
        with self.assertRaisesRegex(
            ValueError,
            r"The `strides` argument must be a tuple of \d+ "
            r"integers. Received strides=\(1, 0\), including values \{0\} "
            r"that do not satisfy `value > 0`",
        ):
            layers.DepthwiseConv2D(
                depth_multiplier=2, kernel_size=(2, 2), strides=(1, 0)
            )

        # `dilation_rate > 1` while `strides > 1`.
        with self.assertRaisesRegex(
            ValueError,
            r"`strides > 1` not supported in conjunction with "
            r"`dilation_rate > 1`. Received: strides=\(2, 2\) and "
            r"dilation_rate=\(2, 1\)",
        ):
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

        inputs = np.random.normal(size=[2, 8, 4])
        layer.build(input_shape=inputs.shape)

        kernel_shape = layer.kernel.shape
        kernel_weights = np.random.normal(size=kernel_shape)
        bias_weights = np.random.normal(size=(depth_multiplier * 4,))
        layer.kernel.assign(kernel_weights)
        layer.bias.assign(bias_weights)

        outputs = layer(inputs)
        expected = np_depthwise_conv1d(
            inputs,
            kernel_weights,
            bias_weights,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )
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

        inputs = np.random.normal(size=[2, 8, 8, 4])
        layer.build(input_shape=inputs.shape)

        kernel_shape = layer.kernel.shape
        kernel_weights = np.random.normal(size=kernel_shape)
        bias_weights = np.random.normal(size=(depth_multiplier * 4,))
        layer.kernel.assign(kernel_weights)
        layer.bias.assign(bias_weights)

        outputs = layer(inputs)
        expected = np_depthwise_conv2d(
            inputs,
            kernel_weights,
            bias_weights,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )
        self.assertAllClose(outputs.shape, expected.shape)
        self.assertAllClose(outputs, expected, atol=1e-5)
