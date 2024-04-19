import os

import numpy as np
import pytest
from absl.testing import parameterized
from numpy.lib.stride_tricks import as_strided

from keras.src import backend
from keras.src import constraints
from keras.src import layers
from keras.src import models
from keras.src import saving
from keras.src import testing


def _same_padding(input_size, kernel_size, stride):
    if input_size % stride == 0:
        padding = max(kernel_size - stride, 0)
    else:
        padding = max(kernel_size - (input_size % stride), 0)
    return padding // 2, padding - padding // 2


def np_conv1d(
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
        new_kernel_size = kernel_size + (dilation_rate - 1) * (kernel_size - 1)
        new_kernel_weights = np.zeros(
            (new_kernel_size, ch_in, ch_out), dtype=kernel_weights.dtype
        )
        new_kernel_weights[::dilation_rate] = kernel_weights
        kernel_weights = new_kernel_weights
        kernel_size = kernel_weights.shape[0]

    if padding != "valid":
        n_batch, h_x, _ = x.shape
        h_pad = _same_padding(h_x, kernel_size, h_stride)
        npad = [(0, 0)] * x.ndim
        if padding == "causal":
            npad[1] = (h_pad[0] + h_pad[1], 0)
        else:
            npad[1] = h_pad
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
    out = np.concatenate(out_grps, axis=-1)
    if data_format == "channels_first":
        out = out.swapaxes(1, 2)
    return out


def np_conv2d(
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
    for grp in range(1, groups + 1):
        x_in = x[..., (grp - 1) * ch_in : grp * ch_in]
        stride_shape = (n_batch, h_out, w_out, h_kernel, w_kernel, ch_in)
        strides = (
            x_in.strides[0],
            h_stride * x_in.strides[1],
            w_stride * x_in.strides[2],
            x_in.strides[1],
            x_in.strides[2],
            x_in.strides[3],
        )
        inner_dim = h_kernel * w_kernel * ch_in
        x_strided = as_strided(
            x_in, shape=stride_shape, strides=strides
        ).reshape(-1, inner_dim)
        ch_out_groups = ch_out // groups
        kernel_weights_grp = kernel_weights[
            ..., (grp - 1) * ch_out_groups : grp * ch_out_groups
        ].reshape(-1, ch_out_groups)
        bias_weights_grp = bias_weights[
            ..., (grp - 1) * ch_out_groups : grp * ch_out_groups
        ]
        out_grps.append(x_strided @ kernel_weights_grp + bias_weights_grp)
    out = np.concatenate(out_grps, axis=-1).reshape(
        n_batch, h_out, w_out, ch_out
    )
    if data_format == "channels_first":
        out = out.transpose((0, 3, 1, 2))
    return out


def np_conv3d(
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
        x = x.transpose((0, 2, 3, 4, 1))
    if isinstance(strides, (tuple, list)):
        h_stride, w_stride, d_stride = strides
    else:
        h_stride = strides
        w_stride = strides
        d_stride = strides
    if isinstance(dilation_rate, (tuple, list)):
        h_dilation, w_dilation, d_dilation = dilation_rate
    else:
        h_dilation = dilation_rate
        w_dilation = dilation_rate
        d_dilation = dilation_rate

    h_kernel, w_kernel, d_kernel, ch_in, ch_out = kernel_weights.shape

    if h_dilation > 1 or w_dilation > 1 or d_dilation > 1:
        new_h_kernel = h_kernel + (h_dilation - 1) * (h_kernel - 1)
        new_w_kernel = w_kernel + (w_dilation - 1) * (w_kernel - 1)
        new_d_kernel = d_kernel + (d_dilation - 1) * (d_kernel - 1)
        new_kenel_size_tuple = (new_h_kernel, new_w_kernel, new_d_kernel)
        new_kernel_weights = np.zeros(
            (*new_kenel_size_tuple, ch_in, ch_out),
            dtype=kernel_weights.dtype,
        )
        new_kernel_weights[::h_dilation, ::w_dilation, ::d_dilation] = (
            kernel_weights
        )
        kernel_weights = new_kernel_weights
        h_kernel, w_kernel, d_kernel = kernel_weights.shape[:3]

    if padding == "same":
        n_batch, h_x, w_x, d_x, _ = x.shape
        h_pad = _same_padding(h_x, h_kernel, h_stride)
        w_pad = _same_padding(w_x, w_kernel, w_stride)
        d_pad = _same_padding(d_x, d_kernel, d_stride)
        npad = [(0, 0)] * x.ndim
        npad[1] = h_pad
        npad[2] = w_pad
        npad[3] = d_pad
        x = np.pad(x, pad_width=npad, mode="constant", constant_values=0)

    n_batch, h_x, w_x, d_x, _ = x.shape
    h_out = int((h_x - h_kernel) / h_stride) + 1
    w_out = int((w_x - w_kernel) / w_stride) + 1
    d_out = int((d_x - d_kernel) / d_stride) + 1

    out_grps = []
    for grp in range(1, groups + 1):
        x_in = x[..., (grp - 1) * ch_in : grp * ch_in]
        stride_shape = (
            n_batch,
            h_out,
            w_out,
            d_out,
            h_kernel,
            w_kernel,
            d_kernel,
            ch_in,
        )
        strides = (
            x_in.strides[0],
            h_stride * x_in.strides[1],
            w_stride * x_in.strides[2],
            d_stride * x_in.strides[3],
            x_in.strides[1],
            x_in.strides[2],
            x_in.strides[3],
            x_in.strides[4],
        )
        inner_dim = h_kernel * w_kernel * d_kernel * ch_in
        x_strided = as_strided(
            x_in, shape=stride_shape, strides=strides
        ).reshape(-1, inner_dim)
        ch_out_groups = ch_out // groups
        kernel_weights_grp = kernel_weights[
            ..., (grp - 1) * ch_out_groups : grp * ch_out_groups
        ].reshape(-1, ch_out_groups)
        bias_weights_grp = bias_weights[
            ..., (grp - 1) * ch_out_groups : grp * ch_out_groups
        ]
        out_grps.append(x_strided @ kernel_weights_grp + bias_weights_grp)
    out = np.concatenate(out_grps, axis=-1).reshape(
        n_batch, h_out, w_out, d_out, ch_out
    )
    if data_format == "channels_first":
        out = out.transpose((0, 4, 1, 2, 3))
    return out


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
        with self.assertRaisesRegex(
            ValueError,
            "Invalid value for argument `filters`. Expected a "
            "strictly positive value. Received filters=0.",
        ):
            layers.Conv1D(filters=0, kernel_size=1)

        # `kernel_size` has 0.
        with self.assertRaisesRegex(
            ValueError,
            r"The `kernel_size` argument must be a tuple of \d+ "
            r"integers. Received kernel_size=\(1, 0\), including values \{0\} "
            r"that do not satisfy `value > 0`",
        ):
            layers.Conv2D(filters=2, kernel_size=(1, 0))

        # `strides` has 0.
        with self.assertRaisesRegex(
            ValueError,
            r"The `strides` argument must be a tuple of \d+ "
            r"integers. Received strides=\(1, 0\), including values \{0\} that "
            r"do not satisfy `value > 0`",
        ):
            layers.Conv2D(filters=2, kernel_size=(2, 2), strides=(1, 0))

        # `dilation_rate > 1` while `strides > 1`.
        with self.assertRaisesRegex(
            ValueError,
            r"`strides > 1` not supported in conjunction with "
            r"`dilation_rate > 1`. Received: strides=\(2, 2\) and "
            r"dilation_rate=\(2, 1\)",
        ):
            layers.Conv2D(
                filters=2, kernel_size=(2, 2), strides=2, dilation_rate=(2, 1)
            )

        # `groups` is not strictly positive.
        with self.assertRaisesRegex(
            ValueError,
            "The number of groups must be a positive integer. "
            "Received: groups=0.",
        ):
            layers.Conv2D(filters=5, kernel_size=(2, 2), groups=0)

        # `filters` cannot be divided by `groups`.
        with self.assertRaisesRegex(
            ValueError,
            "The number of filters must be evenly divisible by the"
            " number of groups. Received: groups=2, filters=5.",
        ):
            layers.Conv2D(filters=5, kernel_size=(2, 2), groups=2)

    @parameterized.named_parameters(
        {
            "testcase_name": "conv1d_kernel_size3_strides1",
            "conv_cls": layers.Conv1D,
            "filters": 6,
            "kernel_size": 3,
            "strides": 1,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "groups": 1,
            "input_shape": (None, 5, 4),
            "output_shape": (None, 3, 6),
        },
        {
            "testcase_name": "conv1d_kernel_size2_strides2",
            "conv_cls": layers.Conv1D,
            "filters": 6,
            "kernel_size": 2,
            "strides": 2,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "groups": 2,
            "input_shape": (None, 5, 4),
            "output_shape": (None, 2, 6),
        },
        {
            "testcase_name": "conv2d_kernel_size3_strides1",
            "conv_cls": layers.Conv2D,
            "filters": 6,
            "kernel_size": 3,
            "strides": 1,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "groups": 1,
            "input_shape": (None, 5, 5, 4),
            "output_shape": (None, 3, 3, 6),
        },
        {
            "testcase_name": "conv2d_kernel_size2_strides2",
            "conv_cls": layers.Conv2D,
            "filters": 6,
            "kernel_size": 2,
            "strides": 2,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "groups": 2,
            "input_shape": (None, 5, 5, 4),
            "output_shape": (None, 2, 2, 6),
        },
        {
            "testcase_name": "conv3d_kernel_size3_strides1",
            "conv_cls": layers.Conv3D,
            "filters": 6,
            "kernel_size": 3,
            "strides": 1,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "groups": 1,
            "input_shape": (None, 5, 5, 5, 4),
            "output_shape": (None, 3, 3, 3, 6),
        },
        {
            "testcase_name": "conv3d_kernel_size2_strides2",
            "conv_cls": layers.Conv3D,
            "filters": 6,
            "kernel_size": 2,
            "strides": 2,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "groups": 2,
            "input_shape": (None, 5, 5, 5, 4),
            "output_shape": (None, 2, 2, 2, 6),
        },
    )
    @pytest.mark.requires_trainable_backend
    def test_enable_lora(
        self,
        conv_cls,
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
        if conv_cls not in (layers.Conv1D, layers.Conv2D, layers.Conv3D):
            raise TypeError
        layer = conv_cls(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
        )
        layer.build(input_shape)
        layer.enable_lora(2)
        self.assertLen(layer.trainable_weights, 3)
        self.assertLen(layer.non_trainable_weights, 1)
        if backend.backend() == "torch":
            self.assertLen(layer.torch_params, 4)
        # Try eager call
        x = np.random.random((64,) + input_shape[1:])
        y = np.random.random((64,) + output_shape[1:])
        _ = layer(x[:2])

        init_lora_a_kernel_value = layer.lora_kernel_a.numpy()
        init_lora_b_kernel_value = layer.lora_kernel_b.numpy()

        # Try calling fit()
        model = models.Sequential([layer])
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y)

        final_lora_a_kernel_value = layer.lora_kernel_a.numpy()
        final_lora_b_kernel_value = layer.lora_kernel_b.numpy()
        diff_a = np.max(
            np.abs(init_lora_a_kernel_value - final_lora_a_kernel_value)
        )
        diff_b = np.max(
            np.abs(init_lora_b_kernel_value - final_lora_b_kernel_value)
        )
        self.assertGreater(diff_a, 0.0)
        self.assertGreater(diff_b, 0.0)

        # Try saving and reloading the model
        temp_filepath = os.path.join(self.get_temp_dir(), "lora_model.keras")
        model.save(temp_filepath)

        new_model = saving.load_model(temp_filepath)
        self.assertTrue(new_model.layers[0].lora_enabled)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "lora_model.weights.h5"
        )
        model.save_weights(temp_filepath)

        # Load the file into a fresh, non-lora model
        new_model = models.Sequential(
            [
                conv_cls(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    dilation_rate=dilation_rate,
                    groups=groups,
                )
            ]
        )
        new_model.build(input_shape)
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try loading a normal checkpoint into a lora model
        new_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    @pytest.mark.requires_trainable_backend
    def test_lora_weight_name(self):

        class MyModel(models.Model):
            def __init__(self):
                super().__init__(name="mymodel")
                self.conv2d = layers.Conv2D(4, 3, name="conv2d")

            def build(self, input_shape):
                self.conv2d.build(input_shape)

            def call(self, x):
                return self.conv2d(x)

        model = MyModel()
        model.build((None, 5, 5, 4))
        model.conv2d.enable_lora(2)
        self.assertEqual(
            model.conv2d.lora_kernel_a.path, "mymodel/conv2d/lora_kernel_a"
        )

    @pytest.mark.requires_trainable_backend
    def test_lora_rank_argument(self):
        self.run_layer_test(
            layers.Conv2D,
            init_kwargs={
                "filters": 5,
                "kernel_size": 3,
                "activation": "sigmoid",
                "data_format": "channels_last",
                "kernel_regularizer": "l2",
                "lora_rank": 2,
            },
            input_shape=(2, 5, 5, 4),
            expected_output_shape=(2, 3, 3, 5),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=1,
            expected_num_seed_generators=0,
            expected_num_losses=2,  # we have 2 regularizers.
            supports_masking=False,
        )


class ConvCorrectnessTest(testing.TestCase, parameterized.TestCase):
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
        {
            "filters": 6,
            "kernel_size": (2,),
            "strides": (2,),
            "padding": "valid",
            "data_format": "channels_first",
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
        expected = np_conv1d(
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
            "kernel_size": 2,
            "strides": 1,
            "padding": "same",
            "data_format": "channels_last",
            "dilation_rate": (2, 3),
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
        {
            "filters": 6,
            "kernel_size": (4, 3),
            "strides": (2, 1),
            "padding": "valid",
            "data_format": "channels_first",
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

        inputs = np.random.normal(size=[2, 8, 8, 4])
        layer.build(input_shape=inputs.shape)

        kernel_shape = layer.kernel.shape
        kernel_weights = np.random.normal(size=kernel_shape)
        bias_weights = np.random.normal(size=(filters,))
        layer.kernel.assign(kernel_weights)
        layer.bias.assign(bias_weights)

        outputs = layer(inputs)
        expected = np_conv2d(
            inputs,
            kernel_weights,
            bias_weights,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
        )
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
            "kernel_size": 2,
            "strides": 1,
            "padding": "same",
            "data_format": "channels_last",
            "dilation_rate": (2, 3, 4),
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
        {
            "filters": 6,
            "kernel_size": (2, 2, 3),
            "strides": (2, 1, 2),
            "padding": "valid",
            "data_format": "channels_first",
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

        inputs = np.random.normal(size=[2, 8, 8, 8, 4])
        layer.build(input_shape=inputs.shape)

        kernel_shape = layer.kernel.shape
        kernel_weights = np.random.normal(size=kernel_shape)
        bias_weights = np.random.normal(size=(filters,))
        layer.kernel.assign(kernel_weights)
        layer.bias.assign(bias_weights)

        outputs = layer(inputs)
        expected = np_conv3d(
            inputs,
            kernel_weights,
            bias_weights,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
        )
        self.assertAllClose(outputs, expected, rtol=1e-3)

    def test_conv_constraints(self):
        layer = layers.Conv2D(
            filters=4,
            kernel_size=3,
            kernel_constraint="non_neg",
        )
        layer.build((None, 5, 5, 3))
        self.assertIsInstance(layer.kernel.constraint, constraints.NonNeg)
        layer = layers.Conv2D(
            filters=4,
            kernel_size=3,
            bias_constraint="non_neg",
        )
        layer.build((None, 5, 5, 3))
        self.assertIsInstance(layer.bias.constraint, constraints.NonNeg)
