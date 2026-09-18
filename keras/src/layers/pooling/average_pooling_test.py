import numpy as np
from absl.testing import parameterized
from numpy.lib.stride_tricks import as_strided

from keras.src import layers
from keras.src import testing


def _same_padding(input_size, pool_size, stride):
    if input_size % stride == 0:
        return max(pool_size - stride, 0)
    else:
        return max(pool_size - (input_size % stride), 0)


def np_avgpool1d(x, pool_size, strides, padding, data_format):
    if data_format == "channels_first":
        x = x.swapaxes(1, 2)
    if isinstance(pool_size, (tuple, list)):
        pool_size = pool_size[0]
    if isinstance(strides, (tuple, list)):
        h_stride = strides[0]
    else:
        h_stride = strides

    if padding == "same":
        n_batch, h_x, ch_x = x.shape
        pad_value = _same_padding(h_x, pool_size, h_stride)
        npad = [(0, 0)] * x.ndim
        npad[1] = (0, pad_value)
        x = np.pad(x, pad_width=npad, mode="edge")

    n_batch, h_x, ch_x = x.shape
    out_h = int((h_x - pool_size) / h_stride) + 1

    stride_shape = (n_batch, out_h, ch_x, pool_size)
    strides = (
        x.strides[0],
        h_stride * x.strides[1],
        x.strides[2],
        x.strides[1],
    )
    windows = as_strided(x, shape=stride_shape, strides=strides)
    out = np.mean(windows, axis=(3,))
    if data_format == "channels_first":
        out = out.swapaxes(1, 2)
    return out


def np_avgpool2d(x, pool_size, strides, padding, data_format):
    if data_format == "channels_first":
        x = x.transpose((0, 2, 3, 1))
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)
    if isinstance(strides, int):
        strides = (strides, strides)

    h_pool_size, w_pool_size = pool_size
    h_stride, w_stride = strides
    if padding == "same":
        n_batch, h_x, w_x, ch_x = x.shape
        h_padding = _same_padding(h_x, h_pool_size, h_stride)
        w_padding = _same_padding(w_x, w_pool_size, w_stride)
        npad = [(0, 0)] * x.ndim
        npad[1] = (0, h_padding)
        npad[2] = (0, w_padding)
        x = np.pad(x, pad_width=npad, mode="edge")

    n_batch, h_x, w_x, ch_x = x.shape
    out_h = int((h_x - h_pool_size) / h_stride) + 1
    out_w = int((w_x - w_pool_size) / w_stride) + 1

    stride_shape = (n_batch, out_h, out_w, ch_x, *pool_size)
    strides = (
        x.strides[0],
        h_stride * x.strides[1],
        w_stride * x.strides[2],
        x.strides[3],
        x.strides[1],
        x.strides[2],
    )
    windows = as_strided(x, shape=stride_shape, strides=strides)
    out = np.mean(windows, axis=(4, 5))
    if data_format == "channels_first":
        out = out.transpose((0, 3, 1, 2))
    return out


def np_avgpool3d(x, pool_size, strides, padding, data_format):
    if data_format == "channels_first":
        x = x.transpose((0, 2, 3, 4, 1))

    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size, pool_size)
    if isinstance(strides, int):
        strides = (strides, strides, strides)

    h_pool_size, w_pool_size, d_pool_size = pool_size
    h_stride, w_stride, d_stride = strides

    if padding == "same":
        n_batch, h_x, w_x, d_x, ch_x = x.shape
        h_padding = _same_padding(h_x, h_pool_size, h_stride)
        w_padding = _same_padding(w_x, w_pool_size, w_stride)
        d_padding = _same_padding(d_x, d_pool_size, d_stride)
        npad = [(0, 0)] * x.ndim
        npad[1] = (0, h_padding)
        npad[2] = (0, w_padding)
        npad[3] = (0, d_padding)
        # Was "symmetric"; changed to "edge" to match np_avgpool1d/2d's
        # convention above. NOTE: every "same"-padding case actually
        # exercised by this file only ever produces a pad width of 0 or 1,
        # and at width <= 1 "edge" and "symmetric" are mathematically
        # identical (both just replicate/mirror the single adjacent
        # boundary element) -- verified this holds for every parameterized
        # case below, so this change does not alter any currently-passing
        # test's expected value. This only aligns the three helpers'
        # stated convention for padding widths >= 2, which nothing here
        # currently tests; whether "edge" (vs "symmetric") is the version
        # that actually matches the real backend's `SAME`-padding
        # semantics at that scale was NOT independently verified -- an
        # attempt to check this against a real AveragePooling3D layer at
        # pad width 3 did not match either convention, suggesting the
        # real backend's algorithm differs from this reference's
        # pad-then-average approach in a way not captured here. Treat this
        # as "removes an unexplained inconsistency between the three
        # helpers," not as "proven correct for large padding."
        x = np.pad(x, pad_width=npad, mode="edge")

    n_batch, h_x, w_x, d_x, ch_x = x.shape
    out_h = int((h_x - h_pool_size) / h_stride) + 1
    out_w = int((w_x - w_pool_size) / w_stride) + 1
    out_d = int((d_x - d_pool_size) / d_stride) + 1

    stride_shape = (n_batch, out_h, out_w, out_d, ch_x, *pool_size)
    strides = (
        x.strides[0],
        h_stride * x.strides[1],
        w_stride * x.strides[2],
        d_stride * x.strides[3],
        x.strides[4],
        x.strides[1],
        x.strides[2],
        x.strides[3],
    )
    windows = as_strided(x, shape=stride_shape, strides=strides)
    out = np.mean(windows, axis=(5, 6, 7))
    if data_format == "channels_first":
        out = out.transpose((0, 4, 1, 2, 3))
    return out


class AveragePoolingBasicTest(testing.TestCase):
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
            run_mixed_precision_check=False,
            assert_built_after_instantiation=True,
        )

    @parameterized.parameters(
        (2, 1, "valid", "channels_last", (3, 5, 5, 4), (3, 4, 4, 4)),
        (2, 1, "same", "channels_last", (3, 5, 5, 4), (3, 5, 5, 4)),
        (2, 1, "valid", "channels_first", (3, 5, 5, 4), (3, 5, 4, 3)),
        (2, 1, "same", "channels_first", (3, 5, 5, 4), (3, 5, 5, 4)),
        ((2, 3), (2, 2), "valid", "channels_last", (3, 5, 5, 4), (3, 2, 2, 4)),
        ((2, 3), (2, 2), "same", "channels_last", (3, 5, 5, 4), (3, 3, 3, 4)),
        ((2, 3), (3, 3), "same", "channels_first", (3, 5, 5, 4), (3, 5, 2, 2)),
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
            run_mixed_precision_check=False,
            assert_built_after_instantiation=True,
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
            # Incomplete op support on tensorflow.
            run_mixed_precision_check=False,
            assert_built_after_instantiation=True,
        )


class AveragePoolingCorrectnessTest(testing.TestCase):
    @parameterized.parameters(
        (2, 1, "valid", "channels_last"),
        (2, 1, "valid", "channels_first"),
        ((2,), (2,), "valid", "channels_last"),
        ((2,), (2,), "valid", "channels_first"),
    )
    def test_average_pooling1d(self, pool_size, strides, padding, data_format):
        inputs = np.arange(24, dtype="float32").reshape((2, 3, 4))

        layer = layers.AveragePooling1D(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )
        outputs = layer(inputs)
        expected = np_avgpool1d(
            inputs, pool_size, strides, padding, data_format
        )
        self.assertAllClose(outputs, expected)

    @parameterized.parameters(
        (2, 1, "same", "channels_last"),
        (2, 1, "same", "channels_first"),
        ((2,), (2,), "same", "channels_last"),
        ((2,), (2,), "same", "channels_first"),
    )
    def test_average_pooling1d_same_padding(
        self, pool_size, strides, padding, data_format
    ):
        inputs = np.arange(24, dtype="float32").reshape((2, 3, 4))

        layer = layers.AveragePooling1D(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )
        outputs = layer(inputs)
        expected = np_avgpool1d(
            inputs, pool_size, strides, padding, data_format
        )
        self.assertAllClose(outputs, expected)

    @parameterized.parameters(
        (2, 1, "valid", "channels_last"),
        ((2, 3), (2, 2), "valid", "channels_last"),
    )
    def test_average_pooling2d(self, pool_size, strides, padding, data_format):
        inputs = np.arange(16, dtype="float32").reshape((1, 4, 4, 1))
        layer = layers.AveragePooling2D(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )
        outputs = layer(inputs)
        expected = np_avgpool2d(
            inputs, pool_size, strides, padding, data_format
        )
        self.assertAllClose(outputs, expected)

    @parameterized.parameters(
        (2, (2, 1), "same", "channels_last"),
        (2, (2, 1), "same", "channels_first"),
        ((2, 2), (2, 2), "same", "channels_last"),
        ((2, 2), (2, 2), "same", "channels_first"),
    )
    def test_average_pooling2d_same_padding(
        self, pool_size, strides, padding, data_format
    ):
        inputs = np.arange(16, dtype="float32").reshape((1, 4, 4, 1))
        layer = layers.AveragePooling2D(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )
        outputs = layer(inputs)
        expected = np_avgpool2d(
            inputs, pool_size, strides, padding, data_format
        )
        self.assertAllClose(outputs, expected)

    @parameterized.parameters(
        (2, 1, "valid", "channels_last"),
        (2, 1, "valid", "channels_first"),
        ((2, 3, 2), (2, 2, 1), "valid", "channels_last"),
        ((2, 3, 2), (2, 2, 1), "valid", "channels_first"),
    )
    def test_average_pooling3d(self, pool_size, strides, padding, data_format):
        inputs = np.arange(240, dtype="float32").reshape((2, 3, 4, 5, 2))

        layer = layers.AveragePooling3D(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )
        outputs = layer(inputs)
        expected = np_avgpool3d(
            inputs, pool_size, strides, padding, data_format
        )
        self.assertAllClose(outputs, expected)

    @parameterized.parameters(
        (2, 1, "same", "channels_last"),
        (2, 1, "same", "channels_first"),
        ((2, 2, 2), (2, 2, 1), "same", "channels_last"),
        ((2, 2, 2), (2, 2, 1), "same", "channels_first"),
    )
    def test_average_pooling3d_same_padding(
        self, pool_size, strides, padding, data_format
    ):
        inputs = np.arange(240, dtype="float32").reshape((2, 3, 4, 5, 2))

        layer = layers.AveragePooling3D(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )
        outputs = layer(inputs)
        expected = np_avgpool3d(
            inputs, pool_size, strides, padding, data_format
        )
        self.assertAllClose(outputs, expected)
