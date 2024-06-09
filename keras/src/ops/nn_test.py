import numpy as np
import pytest
from absl.testing import parameterized

import keras
from keras.src import backend
from keras.src import layers
from keras.src import losses
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.backend.common import dtypes
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.layers.convolutional.conv_test import np_conv1d
from keras.src.layers.convolutional.conv_test import np_conv2d
from keras.src.layers.convolutional.conv_test import np_conv3d
from keras.src.layers.convolutional.conv_transpose_test import (
    np_conv1d_transpose,
)
from keras.src.layers.convolutional.conv_transpose_test import (
    np_conv2d_transpose,
)
from keras.src.layers.convolutional.depthwise_conv_test import (
    np_depthwise_conv2d,
)
from keras.src.layers.pooling.average_pooling_test import np_avgpool1d
from keras.src.layers.pooling.average_pooling_test import np_avgpool2d
from keras.src.layers.pooling.max_pooling_test import np_maxpool1d
from keras.src.layers.pooling.max_pooling_test import np_maxpool2d
from keras.src.ops import nn as knn
from keras.src.ops import numpy as knp
from keras.src.testing.test_utils import named_product


class NNOpsDynamicShapeTest(testing.TestCase, parameterized.TestCase):
    def test_relu(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.relu(x).shape, (None, 2, 3))

    def test_relu6(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.relu6(x).shape, (None, 2, 3))

    def test_sigmoid(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.sigmoid(x).shape, (None, 2, 3))

    def test_softplus(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.softplus(x).shape, (None, 2, 3))

    def test_softsign(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.softsign(x).shape, (None, 2, 3))

    def test_silu(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.silu(x).shape, (None, 2, 3))

    def test_log_sigmoid(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.log_sigmoid(x).shape, (None, 2, 3))

    def test_leaky_relu(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.leaky_relu(x).shape, (None, 2, 3))

    def test_hard_sigmoid(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.hard_sigmoid(x).shape, (None, 2, 3))

    def test_hard_silu(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.hard_silu(x).shape, (None, 2, 3))

    def test_elu(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.elu(x).shape, (None, 2, 3))

    def test_selu(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.selu(x).shape, (None, 2, 3))

    def test_gelu(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.gelu(x).shape, (None, 2, 3))

    def test_softmax(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.softmax(x).shape, (None, 2, 3))
        self.assertEqual(knn.softmax(x, axis=1).shape, (None, 2, 3))
        self.assertEqual(knn.softmax(x, axis=-1).shape, (None, 2, 3))

    def test_softmax_in_graph(self):
        class SoftmaxLayer(keras.Layer):
            def call(self, x):
                return ops.softmax(x, axis=-1)

        class Model(keras.Model):
            def __init__(self):
                x = keras.Input(shape=(None,))
                y = SoftmaxLayer()(x)
                super().__init__(inputs=x, outputs=y)

        # Make sure Keras is able to compile the model graph
        model = Model()
        x = ops.array([[1.0, 2.0, 3.0, 4.0]])
        model.predict(x)

    def test_log_softmax(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.log_softmax(x).shape, (None, 2, 3))
        self.assertEqual(knn.log_softmax(x, axis=1).shape, (None, 2, 3))
        self.assertEqual(knn.log_softmax(x, axis=-1).shape, (None, 2, 3))

    def test_max_pool(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_shape = (None, 8, 3)
        else:
            input_shape = (None, 3, 8)
        x = KerasTensor(input_shape)
        self.assertEqual(
            knn.max_pool(x, 2, 1).shape,
            (None, 7, 3) if data_format == "channels_last" else (None, 3, 7),
        )
        self.assertEqual(
            knn.max_pool(x, 2, 2, padding="same").shape,
            (None, 4, 3) if data_format == "channels_last" else (None, 3, 4),
        )

        if data_format == "channels_last":
            input_shape = (None, 8, None, 3)
        else:
            input_shape = (None, 3, 8, None)
        x = KerasTensor(input_shape)
        (
            self.assertEqual(knn.max_pool(x, 2, 1).shape, (None, 7, None, 3))
            if data_format == "channels_last"
            else (None, 3, 7, None)
        )
        self.assertEqual(
            knn.max_pool(x, 2, 2, padding="same").shape,
            (
                (None, 4, None, 3)
                if data_format == "channels_last"
                else (None, 3, 4, None)
            ),
        )
        self.assertEqual(
            knn.max_pool(x, (2, 2), (2, 2), padding="same").shape,
            (
                (None, 4, None, 3)
                if data_format == "channels_last"
                else (None, 3, 4, None)
            ),
        )

    def test_average_pool(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_shape = (None, 8, 3)
        else:
            input_shape = (None, 3, 8)
        x = KerasTensor(input_shape)
        self.assertEqual(
            knn.average_pool(x, 2, 1).shape,
            (None, 7, 3) if data_format == "channels_last" else (None, 3, 7),
        )
        self.assertEqual(
            knn.average_pool(x, 2, 2, padding="same").shape,
            (None, 4, 3) if data_format == "channels_last" else (None, 3, 4),
        )

        if data_format == "channels_last":
            input_shape = (None, 8, None, 3)
        else:
            input_shape = (None, 3, 8, None)
        x = KerasTensor(input_shape)
        self.assertEqual(
            knn.average_pool(x, 2, 1).shape,
            (
                (None, 7, None, 3)
                if data_format == "channels_last"
                else (None, 3, 7, None)
            ),
        )
        self.assertEqual(
            knn.average_pool(x, 2, 2, padding="same").shape,
            (
                (None, 4, None, 3)
                if data_format == "channels_last"
                else (None, 3, 4, None)
            ),
        )
        self.assertEqual(
            knn.average_pool(x, (2, 2), (2, 2), padding="same").shape,
            (
                (None, 4, None, 3)
                if data_format == "channels_last"
                else (None, 3, 4, None)
            ),
        )

    def test_multi_hot(self):
        x = KerasTensor([None, 3, 1])
        self.assertEqual(knn.multi_hot(x, 5).shape, (None, 1, 5))
        self.assertEqual(knn.multi_hot(x, 5, 1).shape, (None, 3, 1))
        self.assertEqual(knn.multi_hot(x, 5, 2).shape, (None, 5, 1))
        self.assertSparse(knn.multi_hot(x, 5, sparse=True))

    @parameterized.product(dtype=["float32", "int32"])
    def test_multi_hot_dtype(self, dtype):
        # dtype tests
        x = np.arange(5)
        out = knn.multi_hot(x, 5, axis=0, dtype=dtype)
        self.assertEqual(backend.standardize_dtype(out.dtype), dtype)

    def test_conv(self):
        data_format = backend.config.image_data_format()
        # Test 1D conv.
        if data_format == "channels_last":
            input_shape = (None, 20, 3)
        else:
            input_shape = (None, 3, 20)
        inputs_1d = KerasTensor(input_shape)
        kernel = KerasTensor([4, 3, 2])
        for padding in ["valid", "VALID"]:
            self.assertEqual(
                knn.conv(inputs_1d, kernel, 1, padding=padding).shape,
                (
                    (None, 17, 2)
                    if data_format == "channels_last"
                    else (None, 2, 17)
                ),
            )
        for padding in ["same", "SAME"]:
            self.assertEqual(
                knn.conv(inputs_1d, kernel, 1, padding=padding).shape,
                (
                    (None, 20, 2)
                    if data_format == "channels_last"
                    else (None, 2, 20)
                ),
            )
        self.assertEqual(
            knn.conv(inputs_1d, kernel, (2,), dilation_rate=2).shape,
            (None, 7, 2) if data_format == "channels_last" else (None, 2, 7),
        )

        # Test 2D conv.
        if data_format == "channels_last":
            input_shape = (None, 10, None, 3)
        else:
            input_shape = (None, 3, 10, None)
        inputs_2d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 2, 3, 2])
        for padding in ["valid", "VALID"]:
            self.assertEqual(
                knn.conv(inputs_2d, kernel, 1, padding=padding).shape,
                (
                    (None, 9, None, 2)
                    if data_format == "channels_last"
                    else (None, 2, 9, None)
                ),
            )
        for padding in ["same", "SAME"]:
            self.assertEqual(
                knn.conv(inputs_2d, kernel, 1, padding=padding).shape,
                (
                    (None, 10, None, 2)
                    if data_format == "channels_last"
                    else (None, 2, 10, None)
                ),
            )
        self.assertEqual(
            knn.conv(inputs_2d, kernel, (2, 1), dilation_rate=(2, 1)).shape,
            (
                (None, 4, None, 2)
                if data_format == "channels_last"
                else (None, 2, 4, None)
            ),
        )

        # Test 2D conv - H, W specified
        if data_format == "channels_last":
            input_shape = (None, 10, 10, 3)
        else:
            input_shape = (None, 3, 10, 10)
        inputs_2d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 2, 3, 2])
        for padding in ["valid", "VALID"]:
            self.assertEqual(
                knn.conv(inputs_2d, kernel, 1, padding=padding).shape,
                (
                    (None, 9, 9, 2)
                    if data_format == "channels_last"
                    else (None, 2, 9, 9)
                ),
            )
        for padding in ["same", "SAME"]:
            self.assertEqual(
                knn.conv(inputs_2d, kernel, 1, padding=padding).shape,
                (
                    (None, 10, 10, 2)
                    if data_format == "channels_last"
                    else (None, 2, 10, 10)
                ),
            )
        self.assertEqual(
            knn.conv(inputs_2d, kernel, (2, 1), dilation_rate=(2, 1)).shape,
            (
                (None, 4, 9, 2)
                if data_format == "channels_last"
                else (None, 2, 4, 9)
            ),
        )

        # Test 3D conv.
        if data_format == "channels_last":
            input_shape = (None, 8, None, 8, 3)
        else:
            input_shape = (None, 3, 8, None, 8)
        inputs_3d = KerasTensor(input_shape)
        kernel = KerasTensor([3, 3, 3, 3, 2])
        for padding in ["valid", "VALID"]:
            self.assertEqual(
                knn.conv(inputs_3d, kernel, 1, padding=padding).shape,
                (
                    (None, 6, None, 6, 2)
                    if data_format == "channels_last"
                    else (None, 2, 6, None, 6)
                ),
            )
        for padding in ["same", "SAME"]:
            self.assertEqual(
                knn.conv(inputs_3d, kernel, (2, 1, 2), padding=padding).shape,
                (
                    (None, 4, None, 4, 2)
                    if data_format == "channels_last"
                    else (None, 2, 4, None, 4)
                ),
            )
        self.assertEqual(
            knn.conv(
                inputs_3d, kernel, 1, padding="valid", dilation_rate=(1, 2, 2)
            ).shape,
            (
                (None, 6, None, 4, 2)
                if data_format == "channels_last"
                else (None, 2, 6, None, 4)
            ),
        )

    def test_depthwise_conv(self):
        data_format = backend.config.image_data_format()
        # Test 1D depthwise conv.
        if data_format == "channels_last":
            input_shape = (None, 20, 3)
        else:
            input_shape = (None, 3, 20)
        inputs_1d = KerasTensor(input_shape)
        kernel = KerasTensor([4, 3, 1])
        for padding in ["valid", "VALID"]:
            self.assertEqual(
                knn.depthwise_conv(inputs_1d, kernel, 1, padding=padding).shape,
                (
                    (None, 17, 3)
                    if data_format == "channels_last"
                    else (None, 3, 17)
                ),
            )
        for padding in ["same", "SAME"]:
            self.assertEqual(
                knn.depthwise_conv(
                    inputs_1d, kernel, (1,), padding=padding
                ).shape,
                (
                    (None, 20, 3)
                    if data_format == "channels_last"
                    else (None, 3, 20)
                ),
            )
        self.assertEqual(
            knn.depthwise_conv(inputs_1d, kernel, 2, dilation_rate=2).shape,
            (None, 7, 3) if data_format == "channels_last" else (None, 3, 7),
        )

        # Test 2D depthwise conv.
        if data_format == "channels_last":
            input_shape = (None, 10, 10, 3)
        else:
            input_shape = (None, 3, 10, 10)
        inputs_2d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 2, 3, 1])
        for padding in ["valid", "VALID"]:
            self.assertEqual(
                knn.depthwise_conv(inputs_2d, kernel, 1, padding=padding).shape,
                (
                    (None, 9, 9, 3)
                    if data_format == "channels_last"
                    else (None, 3, 9, 9)
                ),
            )
        for padding in ["same", "SAME"]:
            self.assertEqual(
                knn.depthwise_conv(
                    inputs_2d, kernel, (1, 2), padding=padding
                ).shape,
                (
                    (None, 10, 5, 3)
                    if data_format == "channels_last"
                    else (None, 3, 10, 5)
                ),
            )
        self.assertEqual(
            knn.depthwise_conv(inputs_2d, kernel, 2, dilation_rate=2).shape,
            (
                (None, 4, 4, 3)
                if data_format == "channels_last"
                else (None, 3, 4, 4)
            ),
        )
        self.assertEqual(
            knn.depthwise_conv(
                inputs_2d, kernel, 2, dilation_rate=(2, 1)
            ).shape,
            (
                (None, 4, 5, 3)
                if data_format == "channels_last"
                else (None, 3, 4, 5)
            ),
        )

    def test_separable_conv(self):
        data_format = backend.config.image_data_format()
        # Test 1D separable conv.
        if data_format == "channels_last":
            input_shape = (None, 20, 3)
        else:
            input_shape = (None, 3, 20)
        inputs_1d = KerasTensor(input_shape)
        kernel = KerasTensor([4, 3, 2])
        pointwise_kernel = KerasTensor([1, 6, 5])
        self.assertEqual(
            knn.separable_conv(
                inputs_1d, kernel, pointwise_kernel, 1, padding="valid"
            ).shape,
            (None, 17, 5) if data_format == "channels_last" else (None, 5, 17),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_1d, kernel, pointwise_kernel, 1, padding="same"
            ).shape,
            (None, 20, 5) if data_format == "channels_last" else (None, 5, 20),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_1d, kernel, pointwise_kernel, 2, dilation_rate=2
            ).shape,
            (None, 7, 5) if data_format == "channels_last" else (None, 5, 7),
        )

        # Test 2D separable conv.
        if data_format == "channels_last":
            input_shape = (None, 10, 10, 3)
        else:
            input_shape = (None, 3, 10, 10)
        inputs_2d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 2, 3, 2])
        pointwise_kernel = KerasTensor([1, 1, 6, 5])
        self.assertEqual(
            knn.separable_conv(
                inputs_2d, kernel, pointwise_kernel, 1, padding="valid"
            ).shape,
            (
                (None, 9, 9, 5)
                if data_format == "channels_last"
                else (None, 5, 9, 9)
            ),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_2d, kernel, pointwise_kernel, (1, 2), padding="same"
            ).shape,
            (
                (None, 10, 5, 5)
                if data_format == "channels_last"
                else (None, 5, 10, 5)
            ),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_2d, kernel, pointwise_kernel, 2, dilation_rate=(2, 1)
            ).shape,
            (
                (None, 4, 5, 5)
                if data_format == "channels_last"
                else (None, 5, 4, 5)
            ),
        )

    def test_conv_transpose(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_shape = (None, 4, 3)
        else:
            input_shape = (None, 3, 4)
        inputs_1d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 5, 3])
        self.assertEqual(
            knn.conv_transpose(inputs_1d, kernel, 2).shape,
            (None, 8, 5) if data_format == "channels_last" else (None, 5, 8),
        )
        self.assertEqual(
            knn.conv_transpose(inputs_1d, kernel, 2, padding="same").shape,
            (None, 8, 5) if data_format == "channels_last" else (None, 5, 8),
        )
        self.assertEqual(
            knn.conv_transpose(
                inputs_1d, kernel, 5, padding="valid", output_padding=4
            ).shape,
            (None, 21, 5) if data_format == "channels_last" else (None, 5, 21),
        )

        if data_format == "channels_last":
            input_shape = (None, 4, 4, 3)
        else:
            input_shape = (None, 3, 4, 4)
        inputs_2d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 2, 5, 3])
        self.assertEqual(
            knn.conv_transpose(inputs_2d, kernel, 2).shape,
            (
                (None, 8, 8, 5)
                if data_format == "channels_last"
                else (None, 5, 8, 8)
            ),
        )
        self.assertEqual(
            knn.conv_transpose(inputs_2d, kernel, (2, 2), padding="same").shape,
            (
                (None, 8, 8, 5)
                if data_format == "channels_last"
                else (None, 5, 8, 8)
            ),
        )
        self.assertEqual(
            knn.conv_transpose(
                inputs_2d, kernel, (5, 5), padding="valid", output_padding=4
            ).shape,
            (
                (None, 21, 21, 5)
                if data_format == "channels_last"
                else (None, 5, 21, 21)
            ),
        )

    def test_one_hot(self):
        x = KerasTensor([None, 3, 1])
        self.assertEqual(knn.one_hot(x, 5).shape, (None, 3, 1, 5))
        self.assertEqual(knn.one_hot(x, 5, 1).shape, (None, 5, 3, 1))
        self.assertEqual(knn.one_hot(x, 5, 2).shape, (None, 3, 5, 1))
        self.assertSparse(knn.one_hot(x, 5, sparse=True))

    @parameterized.product(dtype=["float32", "int32"])
    def test_one_hot_dtype(self, dtype):
        # dtype tests
        x = np.arange(5)
        out = knn.one_hot(x, 5, axis=0, dtype=dtype)
        self.assertEqual(backend.standardize_dtype(out.dtype), dtype)

    def test_moments(self):
        x = KerasTensor([None, 3, 4])
        self.assertEqual(knn.moments(x, axes=[0])[0].shape, (3, 4))
        self.assertEqual(knn.moments(x, axes=[0, 1])[0].shape, (4,))
        self.assertEqual(
            knn.moments(x, axes=[0, 1], keepdims=True)[0].shape, (1, 1, 4)
        )

        self.assertEqual(knn.moments(x, axes=[1])[0].shape, (None, 4))
        self.assertEqual(knn.moments(x, axes=[1, 2])[0].shape, (None,))
        self.assertEqual(
            knn.moments(x, axes=[1, 2], keepdims=True)[0].shape, (None, 1, 1)
        )

    def test_batch_normalization(self):
        x = KerasTensor([None, 3, 4])
        mean = KerasTensor([4])
        variance = KerasTensor([4])
        self.assertEqual(
            knn.batch_normalization(x, mean, variance, axis=-1).shape,
            (None, 3, 4),
        )

        x = KerasTensor([None, 3, 4, 5])
        self.assertEqual(
            knn.batch_normalization(x, mean, variance, axis=2).shape,
            (None, 3, 4, 5),
        )

        mean = KerasTensor([3])
        variance = KerasTensor([3])
        self.assertEqual(
            knn.batch_normalization(x, mean, variance, axis=1).shape,
            (None, 3, 4, 5),
        )

        # Test wrong offset shape
        self.assertRaisesRegex(
            ValueError,
            "`offset` must be a vector of length",
            knn.batch_normalization,
            KerasTensor([None, 3, 4, 5]),
            KerasTensor([5]),
            KerasTensor([5]),
            axis=-1,
            offset=KerasTensor([3]),
            scale=KerasTensor([5]),
        )

        # Test wrong scale shape
        self.assertRaisesRegex(
            ValueError,
            "`scale` must be a vector of length",
            knn.batch_normalization,
            KerasTensor([None, 3, 4, 5]),
            KerasTensor([5]),
            KerasTensor([5]),
            axis=-1,
            offset=KerasTensor([5]),
            scale=KerasTensor([3]),
        )

    def test_ctc_decode(self):
        # Test strategy="greedy"
        inputs = KerasTensor([None, 2, 3])
        sequence_lengths = KerasTensor([None])
        decoded, scores = knn.ctc_decode(inputs, sequence_lengths)
        self.assertEqual(decoded.shape, (1, None, 2))
        self.assertEqual(scores.shape, (None, 1))

        # Test strategy="beam_search"
        inputs = KerasTensor([None, 2, 3])
        sequence_lengths = KerasTensor([None])
        decoded, scores = knn.ctc_decode(
            inputs, sequence_lengths, strategy="beam_search", top_paths=2
        )
        self.assertEqual(decoded.shape, (2, None, 2))
        self.assertEqual(scores.shape, (None, 2))

    def test_normalize(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.normalize(x).shape, (None, 2, 3))

    def test_psnr(self):
        x1 = KerasTensor([None, 2, 3])
        x2 = KerasTensor([None, 5, 6])
        out = knn.psnr(x1, x2, max_val=224)
        self.assertEqual(out.shape, ())


class NNOpsStaticShapeTest(testing.TestCase):
    def test_relu(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.relu(x).shape, (1, 2, 3))

    def test_relu6(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.relu6(x).shape, (1, 2, 3))

    def test_sigmoid(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.sigmoid(x).shape, (1, 2, 3))

    def test_softplus(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.softplus(x).shape, (1, 2, 3))

    def test_softsign(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.softsign(x).shape, (1, 2, 3))

    def test_silu(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.silu(x).shape, (1, 2, 3))

    def test_log_sigmoid(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.log_sigmoid(x).shape, (1, 2, 3))

    def test_leaky_relu(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.leaky_relu(x).shape, (1, 2, 3))

    def test_hard_sigmoid(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.hard_sigmoid(x).shape, (1, 2, 3))

    def test_hard_silu(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.hard_silu(x).shape, (1, 2, 3))

    def test_elu(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.elu(x).shape, (1, 2, 3))

    def test_selu(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.selu(x).shape, (1, 2, 3))

    def test_gelu(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.gelu(x).shape, (1, 2, 3))

    def test_softmax(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.softmax(x).shape, (1, 2, 3))
        self.assertEqual(knn.softmax(x, axis=1).shape, (1, 2, 3))
        self.assertEqual(knn.softmax(x, axis=-1).shape, (1, 2, 3))

    def test_log_softmax(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.log_softmax(x).shape, (1, 2, 3))
        self.assertEqual(knn.log_softmax(x, axis=1).shape, (1, 2, 3))
        self.assertEqual(knn.log_softmax(x, axis=-1).shape, (1, 2, 3))

    def test_max_pool(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_shape = (1, 8, 3)
        else:
            input_shape = (1, 3, 8)
        x = KerasTensor(input_shape)
        self.assertEqual(
            knn.max_pool(x, 2, 1).shape,
            (1, 7, 3) if data_format == "channels_last" else (1, 3, 7),
        )
        self.assertEqual(
            knn.max_pool(x, 2, 2, padding="same").shape,
            (1, 4, 3) if data_format == "channels_last" else (1, 3, 4),
        )

        if data_format == "channels_last":
            input_shape = (1, 8, 8, 3)
        else:
            input_shape = (1, 3, 8, 8)
        x = KerasTensor(input_shape)
        self.assertEqual(
            knn.max_pool(x, 2, 1).shape,
            (1, 7, 7, 3) if data_format == "channels_last" else (1, 3, 7, 7),
        )
        self.assertEqual(
            knn.max_pool(x, 2, 2, padding="same").shape,
            (1, 4, 4, 3) if data_format == "channels_last" else (1, 3, 4, 4),
        )
        self.assertEqual(
            knn.max_pool(x, (2, 2), (2, 2), padding="same").shape,
            (1, 4, 4, 3) if data_format == "channels_last" else (1, 3, 4, 4),
        )

    def test_average_pool(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_shape = (1, 8, 3)
        else:
            input_shape = (1, 3, 8)
        x = KerasTensor(input_shape)
        self.assertEqual(
            knn.average_pool(x, 2, 1).shape,
            (1, 7, 3) if data_format == "channels_last" else (1, 3, 7),
        )
        self.assertEqual(
            knn.average_pool(x, 2, 2, padding="same").shape,
            (1, 4, 3) if data_format == "channels_last" else (1, 3, 4),
        )

        if data_format == "channels_last":
            input_shape = (1, 8, 8, 3)
        else:
            input_shape = (1, 3, 8, 8)
        x = KerasTensor(input_shape)
        self.assertEqual(
            knn.average_pool(x, 2, 1).shape,
            (1, 7, 7, 3) if data_format == "channels_last" else (1, 3, 7, 7),
        )
        self.assertEqual(
            knn.average_pool(x, 2, 2, padding="same").shape,
            (1, 4, 4, 3) if data_format == "channels_last" else (1, 3, 4, 4),
        )
        self.assertEqual(
            knn.average_pool(x, (2, 2), (2, 2), padding="same").shape,
            (1, 4, 4, 3) if data_format == "channels_last" else (1, 3, 4, 4),
        )

    def test_conv(self):
        data_format = backend.config.image_data_format()
        # Test 1D conv.
        if data_format == "channels_last":
            input_shape = (2, 20, 3)
        else:
            input_shape = (2, 3, 20)
        inputs_1d = KerasTensor(input_shape)
        kernel = KerasTensor([4, 3, 2])
        self.assertEqual(
            knn.conv(inputs_1d, kernel, 1, padding="valid").shape,
            (2, 17, 2) if data_format == "channels_last" else (2, 2, 17),
        )
        self.assertEqual(
            knn.conv(inputs_1d, kernel, 1, padding="same").shape,
            (2, 20, 2) if data_format == "channels_last" else (2, 2, 20),
        )
        self.assertEqual(
            knn.conv(inputs_1d, kernel, (2,), dilation_rate=2).shape,
            (2, 7, 2) if data_format == "channels_last" else (2, 2, 7),
        )

        # Test 2D conv.
        if data_format == "channels_last":
            input_shape = (2, 10, 10, 3)
        else:
            input_shape = (2, 3, 10, 10)
        inputs_2d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 2, 3, 2])
        self.assertEqual(
            knn.conv(inputs_2d, kernel, 1, padding="valid").shape,
            (2, 9, 9, 2) if data_format == "channels_last" else (2, 2, 9, 9),
        )
        self.assertEqual(
            knn.conv(inputs_2d, kernel, 1, padding="same").shape,
            (
                (2, 10, 10, 2)
                if data_format == "channels_last"
                else (2, 2, 10, 10)
            ),
        )
        self.assertEqual(
            knn.conv(inputs_2d, kernel, (2, 1), dilation_rate=(2, 1)).shape,
            (2, 4, 9, 2) if data_format == "channels_last" else (2, 2, 4, 9),
        )

        # Test 3D conv.
        if data_format == "channels_last":
            input_shape = (2, 8, 8, 8, 3)
        else:
            input_shape = (2, 3, 8, 8, 8)
        inputs_3d = KerasTensor(input_shape)
        kernel = KerasTensor([3, 3, 3, 3, 2])
        self.assertEqual(
            knn.conv(inputs_3d, kernel, 1, padding="valid").shape,
            (
                (2, 6, 6, 6, 2)
                if data_format == "channels_last"
                else (2, 2, 6, 6, 6)
            ),
        )
        self.assertEqual(
            knn.conv(inputs_3d, kernel, (2, 1, 2), padding="same").shape,
            (
                (2, 4, 8, 4, 2)
                if data_format == "channels_last"
                else (2, 2, 4, 8, 4)
            ),
        )
        self.assertEqual(
            knn.conv(
                inputs_3d, kernel, 1, padding="valid", dilation_rate=(1, 2, 2)
            ).shape,
            (
                (2, 6, 4, 4, 2)
                if data_format == "channels_last"
                else (2, 2, 6, 4, 4)
            ),
        )

    def test_depthwise_conv(self):
        data_format = backend.config.image_data_format()
        # Test 1D depthwise conv.
        if data_format == "channels_last":
            input_shape = (2, 20, 3)
        else:
            input_shape = (2, 3, 20)
        inputs_1d = KerasTensor(input_shape)
        kernel = KerasTensor([4, 3, 1])
        self.assertEqual(
            knn.depthwise_conv(inputs_1d, kernel, 1, padding="valid").shape,
            (2, 17, 3) if data_format == "channels_last" else (2, 3, 17),
        )
        self.assertEqual(
            knn.depthwise_conv(inputs_1d, kernel, (1,), padding="same").shape,
            (2, 20, 3) if data_format == "channels_last" else (2, 3, 20),
        )
        self.assertEqual(
            knn.depthwise_conv(inputs_1d, kernel, 2, dilation_rate=2).shape,
            (2, 7, 3) if data_format == "channels_last" else (2, 3, 7),
        )

        # Test 2D depthwise conv.
        if data_format == "channels_last":
            input_shape = (2, 10, 10, 3)
        else:
            input_shape = (2, 3, 10, 10)
        inputs_2d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 2, 3, 1])
        self.assertEqual(
            knn.depthwise_conv(inputs_2d, kernel, 1, padding="valid").shape,
            (2, 9, 9, 3) if data_format == "channels_last" else (2, 3, 9, 9),
        )
        self.assertEqual(
            knn.depthwise_conv(inputs_2d, kernel, (1, 2), padding="same").shape,
            (2, 10, 5, 3) if data_format == "channels_last" else (2, 3, 10, 5),
        )
        self.assertEqual(
            knn.depthwise_conv(inputs_2d, kernel, 2, dilation_rate=2).shape,
            (2, 4, 4, 3) if data_format == "channels_last" else (2, 3, 4, 4),
        )
        self.assertEqual(
            knn.depthwise_conv(
                inputs_2d, kernel, 2, dilation_rate=(2, 1)
            ).shape,
            (2, 4, 5, 3) if data_format == "channels_last" else (2, 3, 4, 5),
        )

    def test_separable_conv(self):
        data_format = backend.config.image_data_format()
        # Test 1D max pooling.
        if data_format == "channels_last":
            input_shape = (2, 20, 3)
        else:
            input_shape = (2, 3, 20)
        inputs_1d = KerasTensor(input_shape)
        kernel = KerasTensor([4, 3, 2])
        pointwise_kernel = KerasTensor([1, 6, 5])
        self.assertEqual(
            knn.separable_conv(
                inputs_1d, kernel, pointwise_kernel, 1, padding="valid"
            ).shape,
            (2, 17, 5) if data_format == "channels_last" else (2, 5, 17),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_1d, kernel, pointwise_kernel, 1, padding="same"
            ).shape,
            (2, 20, 5) if data_format == "channels_last" else (2, 5, 20),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_1d, kernel, pointwise_kernel, 2, dilation_rate=2
            ).shape,
            (2, 7, 5) if data_format == "channels_last" else (2, 5, 7),
        )

        # Test 2D separable conv.
        if data_format == "channels_last":
            input_shape = (2, 10, 10, 3)
        else:
            input_shape = (2, 3, 10, 10)
        inputs_2d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 2, 3, 2])
        pointwise_kernel = KerasTensor([1, 1, 6, 5])
        self.assertEqual(
            knn.separable_conv(
                inputs_2d, kernel, pointwise_kernel, 1, padding="valid"
            ).shape,
            (2, 9, 9, 5) if data_format == "channels_last" else (2, 5, 9, 9),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_2d, kernel, pointwise_kernel, (1, 2), padding="same"
            ).shape,
            (2, 10, 5, 5) if data_format == "channels_last" else (2, 5, 10, 5),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_2d, kernel, pointwise_kernel, 2, dilation_rate=(2, 1)
            ).shape,
            (2, 4, 5, 5) if data_format == "channels_last" else (2, 5, 4, 5),
        )

    def test_conv_transpose(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_shape = (2, 4, 3)
        else:
            input_shape = (2, 3, 4)
        inputs_1d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 5, 3])
        self.assertEqual(
            knn.conv_transpose(inputs_1d, kernel, 2).shape,
            (2, 8, 5) if data_format == "channels_last" else (2, 5, 8),
        )
        self.assertEqual(
            knn.conv_transpose(inputs_1d, kernel, 2, padding="same").shape,
            (2, 8, 5) if data_format == "channels_last" else (2, 5, 8),
        )
        self.assertEqual(
            knn.conv_transpose(
                inputs_1d, kernel, 5, padding="valid", output_padding=4
            ).shape,
            (2, 21, 5) if data_format == "channels_last" else (2, 5, 21),
        )

        if data_format == "channels_last":
            input_shape = (2, 4, 4, 3)
        else:
            input_shape = (2, 3, 4, 4)
        inputs_2d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 2, 5, 3])
        self.assertEqual(
            knn.conv_transpose(inputs_2d, kernel, 2).shape,
            (2, 8, 8, 5) if data_format == "channels_last" else (2, 5, 8, 8),
        )
        self.assertEqual(
            knn.conv_transpose(inputs_2d, kernel, (2, 2), padding="same").shape,
            (2, 8, 8, 5) if data_format == "channels_last" else (2, 5, 8, 8),
        )
        self.assertEqual(
            knn.conv_transpose(
                inputs_2d, kernel, (5, 5), padding="valid", output_padding=4
            ).shape,
            (
                (2, 21, 21, 5)
                if data_format == "channels_last"
                else (2, 5, 21, 21)
            ),
        )

    def test_batched_and_unbatched_inputs_multi_hot(self):
        x = KerasTensor([2, 3, 1])
        unbatched_input = KerasTensor(
            [
                5,
            ]
        )
        self.assertEqual(knn.multi_hot(unbatched_input, 5, -1).shape, (5,))
        self.assertEqual(knn.multi_hot(x, 5).shape, (2, 1, 5))
        self.assertEqual(knn.multi_hot(x, 5, 1).shape, (2, 3, 1))
        self.assertEqual(knn.multi_hot(x, 5, 2).shape, (2, 5, 1))

    def test_one_hot(self):
        x = KerasTensor([2, 3, 1])
        self.assertEqual(knn.one_hot(x, 5).shape, (2, 3, 1, 5))
        self.assertEqual(knn.one_hot(x, 5, 1).shape, (2, 5, 3, 1))
        self.assertEqual(knn.one_hot(x, 5, 2).shape, (2, 3, 5, 1))
        self.assertSparse(knn.one_hot(x, 5, sparse=True))

    def test_binary_crossentropy(self):
        x1 = KerasTensor([2, 3, 1])
        x2 = KerasTensor([2, 3, 1])
        self.assertEqual(knn.binary_crossentropy(x1, x2).shape, (2, 3, 1))

    def test_categorical_crossentropy(self):
        x1 = KerasTensor([2, 3, 4])
        x2 = KerasTensor([2, 3, 4])
        self.assertEqual(knn.categorical_crossentropy(x1, x2).shape, (2, 3))

    def test_sparse_categorical_crossentropy(self):
        x1 = KerasTensor([2, 3], dtype="int32")
        x2 = KerasTensor([2, 3, 4])
        self.assertEqual(
            knn.sparse_categorical_crossentropy(x1, x2).shape, (2, 3)
        )

    def test_moments(self):
        x = KerasTensor([2, 3, 4])
        self.assertEqual(knn.moments(x, axes=[0])[0].shape, (3, 4))
        self.assertEqual(knn.moments(x, axes=[0, 1])[0].shape, (4,))
        self.assertEqual(
            knn.moments(x, axes=[0, 1], keepdims=True)[0].shape, (1, 1, 4)
        )

    def test_batch_normalization(self):
        x = KerasTensor([10, 3, 4])
        mean = KerasTensor([4])
        variance = KerasTensor([4])
        self.assertEqual(
            knn.batch_normalization(x, mean, variance, axis=-1).shape,
            (10, 3, 4),
        )

        x = KerasTensor([10, 3, 4, 5])
        self.assertEqual(
            knn.batch_normalization(x, mean, variance, axis=2).shape,
            (10, 3, 4, 5),
        )

        mean = KerasTensor([3])
        variance = KerasTensor([3])
        self.assertEqual(
            knn.batch_normalization(x, mean, variance, axis=1).shape,
            (10, 3, 4, 5),
        )

    def test_ctc_loss(self):
        x = KerasTensor([10, 3, 4])
        y = KerasTensor([10, 3], dtype="int32")
        x_lengths = KerasTensor([10], dtype="int32")
        y_lengths = KerasTensor([10], dtype="int32")
        self.assertEqual(knn.ctc_loss(x, y, x_lengths, y_lengths).shape, (10,))

    def test_ctc_decode(self):
        # Test strategy="greedy"
        inputs = KerasTensor([10, 2, 3])
        sequence_lengths = KerasTensor([10])
        decoded, scores = knn.ctc_decode(inputs, sequence_lengths)
        self.assertEqual(decoded.shape, (1, 10, 2))
        self.assertEqual(scores.shape, (10, 1))

        # Test strategy="beam_search"
        inputs = KerasTensor([10, 2, 3])
        sequence_lengths = KerasTensor([10])
        decoded, scores = knn.ctc_decode(
            inputs, sequence_lengths, strategy="beam_search", top_paths=2
        )
        self.assertEqual(decoded.shape, (2, 10, 2))
        self.assertEqual(scores.shape, (10, 2))

    def test_normalize(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.normalize(x).shape, (1, 2, 3))

    def test_psnr(self):
        x1 = KerasTensor([1, 2, 3])
        x2 = KerasTensor([5, 6, 7])
        out = knn.psnr(x1, x2, max_val=224)
        self.assertEqual(out.shape, ())


class NNOpsCorrectnessTest(testing.TestCase, parameterized.TestCase):
    def test_relu(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(knn.relu(x), [0, 0, 1, 2, 3])

    def test_relu6(self):
        x = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
        self.assertAllClose(knn.relu6(x), [0, 0, 1, 2, 3, 4, 5, 6, 6])

    def test_sigmoid(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.sigmoid(x), [0.26894143, 0.5, 0.7310586, 0.880797, 0.95257413]
        )

    def test_softplus(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.softplus(x),
            [0.31326166, 0.6931472, 1.3132616, 2.126928, 3.0485873],
        )

    def test_softsign(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(knn.softsign(x), [-0.5, 0, 0.5, 0.6666667, 0.75])

    def test_silu(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.silu(x),
            [-0.26894143, 0, 0.7310586, 1.7615942, 2.8577223],
        )

    def test_log_sigmoid(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.log_sigmoid(x),
            [-1.3132616, -0.6931472, -0.31326166, -0.126928, -0.04858732],
        )

    def test_leaky_relu(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.leaky_relu(x),
            [-0.2, 0, 1, 2, 3],
        )

    def test_hard_sigmoid(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.hard_sigmoid(x),
            [0.33333334, 0.5, 0.6666667, 0.8333334, 1.0],
        )

    def test_hard_silu(self):
        x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.hard_silu(x),
            [-0.0, -0.333333, -0.333333, 0.0, 0.6666667, 1.6666667, 3.0],
        )

    def test_elu(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.elu(x),
            [-0.63212055, 0, 1, 2, 3],
        )
        self.assertAllClose(
            knn.elu(x, alpha=0.5),
            [-0.31606027, 0, 1, 2, 3],
        )

    def test_selu(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.selu(x),
            [-1.1113307, 0.0, 1.050701, 2.101402, 3.152103],
        )

    def test_gelu(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.gelu(x),
            [-0.15880796, 0.0, 0.841192, 1.9545977, 2.9963627],
        )

    def test_softmax(self):
        x = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
        self.assertAllClose(
            knn.softmax(x, axis=None),  # Reduce on all axes.
            [[0.045015, 0.122364, 0.33262], [0.045015, 0.122364, 0.33262]],
        )
        self.assertAllClose(
            knn.softmax(x, axis=0),
            [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        )
        self.assertAllClose(
            knn.softmax(x, axis=-1),
            [
                [0.09003057, 0.24472848, 0.66524094],
                [0.09003057, 0.24472848, 0.66524094],
            ],
        )
        self.assertAllClose(
            knn.softmax(x),  # Default axis should be -1.
            [
                [0.09003057, 0.24472848, 0.66524094],
                [0.09003057, 0.24472848, 0.66524094],
            ],
        )

    def test_log_softmax(self):
        x = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
        self.assertAllClose(
            knn.log_softmax(x, axis=None),  # Reduce on all axes.
            [
                [-3.100753, -2.100753, -1.100753],
                [-3.100753, -2.100753, -1.100753],
            ],
        )
        self.assertAllClose(
            knn.log_softmax(x, axis=0),
            [
                [-0.693147, -0.693147, -0.693147],
                [-0.693147, -0.693147, -0.693147],
            ],
        )
        self.assertAllClose(
            knn.log_softmax(x, axis=-1),
            [
                [-2.407606, -1.407606, -0.407606],
                [-2.407606, -1.407606, -0.407606],
            ],
        )
        self.assertAllClose(
            knn.log_softmax(x),  # Default axis should be -1.
            [
                [-2.407606, -1.407606, -0.407606],
                [-2.407606, -1.407606, -0.407606],
            ],
        )

    def test_max_pool(self):
        data_format = backend.config.image_data_format()
        # Test 1D max pooling.
        if data_format == "channels_last":
            input_shape = (2, 20, 3)
        else:
            input_shape = (2, 3, 20)
        x = np.arange(120, dtype=float).reshape(input_shape)
        self.assertAllClose(
            knn.max_pool(x, 2, 1, padding="valid"),
            np_maxpool1d(x, 2, 1, padding="valid", data_format=data_format),
        )
        self.assertAllClose(
            knn.max_pool(x, 2, 2, padding="same"),
            np_maxpool1d(x, 2, 2, padding="same", data_format=data_format),
        )

        # Test 2D max pooling.
        if data_format == "channels_last":
            input_shape = (2, 10, 9, 3)
        else:
            input_shape = (2, 3, 10, 9)
        x = np.arange(540, dtype=float).reshape(input_shape)
        self.assertAllClose(
            knn.max_pool(x, 2, 1, padding="valid"),
            np_maxpool2d(x, 2, 1, padding="valid", data_format=data_format),
        )
        self.assertAllClose(
            knn.max_pool(x, 2, (2, 1), padding="same"),
            np_maxpool2d(x, 2, (2, 1), padding="same", data_format=data_format),
        )

    def test_average_pool_valid_padding(self):
        data_format = backend.config.image_data_format()
        # Test 1D max pooling.
        if data_format == "channels_last":
            input_shape = (2, 20, 3)
        else:
            input_shape = (2, 3, 20)
        x = np.arange(120, dtype=float).reshape(input_shape)
        self.assertAllClose(
            knn.average_pool(x, 2, 1, padding="valid"),
            np_avgpool1d(x, 2, 1, padding="valid", data_format=data_format),
        )

        # Test 2D max pooling.
        if data_format == "channels_last":
            input_shape = (2, 10, 9, 3)
        else:
            input_shape = (2, 3, 10, 9)
        x = np.arange(540, dtype=float).reshape(input_shape)
        self.assertAllClose(
            knn.average_pool(x, 2, 1, padding="valid"),
            np_avgpool2d(x, 2, 1, padding="valid", data_format=data_format),
        )

    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Torch outputs differently from TF when using `same` padding.",
    )
    def test_average_pool_same_padding(self):
        data_format = backend.config.image_data_format()
        # Test 1D max pooling.
        if data_format == "channels_last":
            input_shape = (2, 20, 3)
        else:
            input_shape = (2, 3, 20)
        x = np.arange(120, dtype=float).reshape(input_shape)

        self.assertAllClose(
            knn.average_pool(x, 2, 2, padding="same"),
            np_avgpool1d(x, 2, 2, padding="same", data_format=data_format),
        )

        # Test 2D max pooling.
        if data_format == "channels_last":
            input_shape = (2, 10, 9, 3)
        else:
            input_shape = (2, 3, 10, 9)
        x = np.arange(540, dtype=float).reshape(input_shape)
        self.assertAllClose(
            knn.average_pool(x, 2, (2, 1), padding="same"),
            np_avgpool2d(x, 2, (2, 1), padding="same", data_format=data_format),
        )

    @parameterized.product(
        strides=(1, 2, 3),
        padding=("valid", "same"),
        dilation_rate=(1, 2),
    )
    def test_conv_1d(self, strides, padding, dilation_rate):
        if strides > 1 and dilation_rate > 1:
            pytest.skip("Unsupported configuration")

        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 20, 3)
        else:
            input_shape = (2, 3, 20)
        inputs_1d = np.arange(120, dtype=float).reshape(input_shape)
        kernel = np.arange(24, dtype=float).reshape([4, 3, 2])

        outputs = knn.conv(
            inputs_1d,
            kernel,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
        )
        expected = np_conv1d(
            inputs_1d,
            kernel,
            bias_weights=np.zeros((2,)),
            strides=strides,
            padding=padding.lower(),
            data_format=backend.config.image_data_format(),
            dilation_rate=dilation_rate,
            groups=1,
        )
        self.assertAllClose(outputs, expected)

    @parameterized.product(strides=(1, 2, (1, 2)), padding=("valid", "same"))
    def test_conv_2d(self, strides, padding):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 10, 10, 3)
        else:
            input_shape = (2, 3, 10, 10)
        inputs_2d = np.arange(600, dtype=float).reshape(input_shape)
        kernel = np.arange(24, dtype=float).reshape([2, 2, 3, 2])

        outputs = knn.conv(inputs_2d, kernel, strides, padding=padding)
        expected = np_conv2d(
            inputs_2d,
            kernel,
            bias_weights=np.zeros((2,)),
            strides=strides,
            padding=padding,
            data_format=backend.config.image_data_format(),
            dilation_rate=1,
            groups=1,
        )
        self.assertAllClose(outputs, expected)

    @parameterized.product(strides=(1, 2), dilation_rate=(1, (2, 1)))
    def test_conv_2d_group_2(self, strides, dilation_rate):
        if (
            backend.backend() == "tensorflow"
            and strides == 2
            and dilation_rate == (2, 1)
        ):
            # This case is not supported by the TF backend.
            return
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 10, 10, 4)
        else:
            input_shape = (2, 4, 10, 10)
        inputs_2d = np.ones(input_shape)
        kernel = np.ones([2, 2, 2, 6])
        outputs = knn.conv(
            inputs_2d,
            kernel,
            strides,
            padding="same",
            dilation_rate=dilation_rate,
        )
        expected = np_conv2d(
            inputs_2d,
            kernel,
            bias_weights=np.zeros((6,)),
            strides=strides,
            padding="same",
            data_format=backend.config.image_data_format(),
            dilation_rate=dilation_rate,
            groups=1,
        )
        self.assertAllClose(outputs, expected)

    @parameterized.product(
        strides=(1, (1, 1, 1), 2),
        padding=("valid", "same"),
        data_format=("channels_first", "channels_last"),
    )
    def test_conv_3d(self, strides, padding, data_format):
        if data_format == "channels_last":
            input_shape = (2, 8, 8, 8, 3)
        else:
            input_shape = (2, 3, 8, 8, 8)
        inputs_3d = np.arange(3072, dtype=float).reshape(input_shape)
        kernel = np.arange(162, dtype=float).reshape([3, 3, 3, 3, 2])

        outputs = knn.conv(
            inputs_3d, kernel, strides, padding=padding, data_format=data_format
        )
        expected = np_conv3d(
            inputs_3d,
            kernel,
            bias_weights=np.zeros((2,)),
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=1,
            groups=1,
        )
        self.assertAllClose(outputs, expected, rtol=1e-5, atol=1e-5)

    @parameterized.product(
        strides=(1, (1, 1), (2, 2)),
        padding=("valid", "same"),
        dilation_rate=(1, (2, 2)),
    )
    def test_depthwise_conv_2d(self, strides, padding, dilation_rate):
        if (
            backend.backend() == "tensorflow"
            and strides == (2, 2)
            and dilation_rate == (2, 2)
        ):
            # This case is not supported by the TF backend.
            return
        print(strides, padding, dilation_rate)
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 10, 10, 3)
        else:
            input_shape = (2, 3, 10, 10)
        inputs_2d = np.arange(600, dtype=float).reshape(input_shape)
        kernel = np.arange(24, dtype=float).reshape([2, 2, 3, 2])

        outputs = knn.depthwise_conv(
            inputs_2d,
            kernel,
            strides,
            padding=padding,
            dilation_rate=dilation_rate,
        )
        expected = np_depthwise_conv2d(
            inputs_2d,
            kernel,
            bias_weights=np.zeros((6,)),
            strides=strides,
            padding=padding,
            data_format=backend.config.image_data_format(),
            dilation_rate=dilation_rate,
        )
        self.assertAllClose(outputs, expected)

    @parameterized.product(
        strides=(1, 2),
        padding=("valid", "same"),
        dilation_rate=(1, (2, 2)),
    )
    def test_separable_conv_2d(self, strides, padding, dilation_rate):
        if (
            backend.backend() == "tensorflow"
            and strides == 2
            and dilation_rate == (2, 2)
        ):
            # This case is not supported by the TF backend.
            return
        # Test 2D conv.
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 10, 10, 3)
        else:
            input_shape = (2, 3, 10, 10)
        inputs_2d = np.arange(600, dtype=float).reshape(input_shape)
        depthwise_kernel = np.arange(24, dtype=float).reshape([2, 2, 3, 2])
        pointwise_kernel = np.arange(72, dtype=float).reshape([1, 1, 6, 12])

        outputs = knn.separable_conv(
            inputs_2d,
            depthwise_kernel,
            pointwise_kernel,
            strides,
            padding=padding,
            dilation_rate=dilation_rate,
        )
        # Depthwise followed by pointwise conv
        expected_depthwise = np_depthwise_conv2d(
            inputs_2d,
            depthwise_kernel,
            np.zeros(6),
            strides=strides,
            padding=padding,
            data_format=backend.config.image_data_format(),
            dilation_rate=dilation_rate,
        )
        expected = np_conv2d(
            expected_depthwise,
            pointwise_kernel,
            np.zeros(6 * 12),
            strides=1,
            padding=padding,
            data_format=backend.config.image_data_format(),
            dilation_rate=dilation_rate,
            groups=1,
        )
        self.assertAllClose(outputs, expected)

    @parameterized.product(padding=("valid", "same"))
    def test_conv_transpose_1d(self, padding):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 4, 3)
        else:
            input_shape = (2, 3, 4)
        inputs_1d = np.arange(24, dtype=float).reshape(input_shape)
        kernel = np.arange(30, dtype=float).reshape([2, 5, 3])
        outputs = knn.conv_transpose(inputs_1d, kernel, 2, padding=padding)
        expected = np_conv1d_transpose(
            inputs_1d,
            kernel,
            bias_weights=np.zeros(5),
            strides=2,
            output_padding=None,
            padding=padding,
            data_format=backend.config.image_data_format(),
            dilation_rate=1,
        )
        self.assertAllClose(outputs, expected)

    @parameterized.product(strides=(2, (2, 2)), padding=("valid", "same"))
    def test_conv_transpose_2d(self, strides, padding):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 4, 4, 3)
        else:
            input_shape = (2, 3, 4, 4)
        inputs_2d = np.arange(96, dtype=float).reshape(input_shape)
        kernel = np.arange(60, dtype=float).reshape([2, 2, 5, 3])

        outputs = knn.conv_transpose(
            inputs_2d, kernel, strides, padding=padding
        )
        expected = np_conv2d_transpose(
            inputs_2d,
            kernel,
            bias_weights=np.zeros(5),
            strides=strides,
            output_padding=None,
            padding=padding,
            data_format=backend.config.image_data_format(),
            dilation_rate=1,
        )
        self.assertAllClose(outputs, expected)

    @parameterized.named_parameters(
        [
            {"testcase_name": "dense", "sparse": False},
            {"testcase_name": "sparse", "sparse": True},
        ]
    )
    def test_one_hot(self, sparse):
        if sparse and not backend.SUPPORTS_SPARSE_TENSORS:
            pytest.skip("Backend does not support sparse tensors")
        # Test 1D one-hot.
        indices_1d = np.array([0, 1, 2, 3])
        output_1d = knn.one_hot(indices_1d, 4, sparse=sparse)
        self.assertAllClose(output_1d, np.eye(4)[indices_1d])
        self.assertSparse(output_1d, sparse)
        output_1d = knn.one_hot(indices_1d, 4, axis=0, sparse=sparse)
        self.assertAllClose(output_1d, np.eye(4)[indices_1d])
        self.assertSparse(output_1d, sparse)

        # Test 1D list one-hot.
        indices_1d = [0, 1, 2, 3]
        output_1d = knn.one_hot(indices_1d, 4, sparse=sparse)
        self.assertAllClose(output_1d, np.eye(4)[indices_1d])
        self.assertSparse(output_1d, sparse)
        output_1d = knn.one_hot(indices_1d, 4, axis=0, sparse=sparse)
        self.assertAllClose(output_1d, np.eye(4)[indices_1d])
        self.assertSparse(output_1d, sparse)

        # Test 2D one-hot.
        indices_2d = np.array([[0, 1], [2, 3]])
        output_2d = knn.one_hot(indices_2d, 4, sparse=sparse)
        self.assertAllClose(output_2d, np.eye(4)[indices_2d])
        self.assertSparse(output_2d, sparse)
        output_2d = knn.one_hot(indices_2d, 4, axis=2, sparse=sparse)
        self.assertAllClose(output_2d, np.eye(4)[indices_2d])
        self.assertSparse(output_2d, sparse)
        output_2d = knn.one_hot(indices_2d, 4, axis=1, sparse=sparse)
        self.assertAllClose(
            output_2d, np.transpose(np.eye(4)[indices_2d], (0, 2, 1))
        )
        self.assertSparse(output_2d, sparse)

        # Test 1D one-hot with 1 extra dimension.
        indices_1d = np.array([[0], [1], [2], [3]])
        output_1d = knn.one_hot(indices_1d, 4, sparse=sparse)
        self.assertAllClose(output_1d, np.eye(4)[indices_1d])
        self.assertSparse(output_1d, sparse)
        output_1d = knn.one_hot(indices_1d, 4, axis=0, sparse=sparse)
        self.assertAllClose(output_1d, np.eye(4)[indices_1d].swapaxes(1, 2))
        self.assertSparse(output_1d, sparse)

        # Test 1D one-hot with negative inputs
        indices_1d = np.array([0, -1, -1, 3])
        output_1d = knn.one_hot(indices_1d, 4, sparse=sparse)
        self.assertAllClose(
            output_1d,
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            ),
        )
        self.assertSparse(output_1d, sparse)

    def test_binary_crossentropy(self):
        # Test with from_logits=False
        target = np.array([[0.1], [0.9], [0.2], [1.0]])
        output = np.array([[0.1], [0.2], [0.3], [0.4]])
        result = knn.binary_crossentropy(target, output, from_logits=False)
        self.assertAllClose(
            result,
            np.array([[0.32508277], [1.47080801], [0.52613434], [0.91629048]]),
        )

        # Test with from_logits=True
        target = np.array([[0.1], [0.9], [0.2], [1.0]])
        output = np.array([[0.1], [0.2], [0.3], [0.4]])
        result = knn.binary_crossentropy(target, output, from_logits=True)
        self.assertAllClose(
            result,
            np.array([[0.73439666], [0.61813887], [0.79435524], [0.51301525]]),
        )

        # Test with output clipping
        target = np.array([[0.1], [0.9], [0.2], [1.0]])
        output = np.array([[0.99], [-0.2], [0.9], [-0.4]])
        result = knn.binary_crossentropy(target, output, from_logits=True)
        self.assertAllClose(
            result,
            np.array([[1.206961], [0.778139], [1.061154], [0.913015]]),
        )

    def test_categorical_crossentropy(self):
        target = np.array(
            [
                [0.33008796, 0.0391289, 0.9503603],
                [0.80376694, 0.92363342, 0.19147756],
            ]
        )
        output = np.array(
            [
                [0.23446431, 0.35822914, 0.06683268],
                [0.3413979, 0.05420256, 0.81619654],
            ]
        )

        # Test from_logits=False
        result = knn.categorical_crossentropy(
            target, output, from_logits=False, axis=-1
        )
        self.assertAllClose(result, np.array([2.54095299, 3.96374412]))

        # Test axis
        result = knn.categorical_crossentropy(
            target, output, from_logits=False, axis=0
        )
        self.assertAllClose(
            result, np.array([0.71683073, 1.87988172, 2.46810762])
        )

        # Test from_logits=True
        result = knn.categorical_crossentropy(
            target, output, from_logits=True, axis=-1
        )
        self.assertAllClose(result, np.array([1.59419954, 2.49880593]))

        # Test with output clipping
        output = np.array(
            [
                [1.23446431, -0.35822914, 1.06683268],
                [0.3413979, -0.05420256, 0.81619654],
            ]
        )
        result = knn.categorical_crossentropy(
            target, output, from_logits=True, axis=-1
        )
        self.assertAllClose(result, np.array([1.16825923, 2.55436813]))

    def test_sparse_categorical_crossentropy(self):
        target = np.array([0, 1, 2])
        output = np.array(
            [[0.9, 0.05, 0.05], [0.05, 0.89, 0.06], [0.05, 0.01, 0.94]]
        )
        result = knn.sparse_categorical_crossentropy(target, output)
        self.assertAllClose(result, [0.105361, 0.116534, 0.061875])

        output = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        result = knn.sparse_categorical_crossentropy(
            target, output, from_logits=True
        )
        self.assertAllClose(result, [0.001822, 0.000459, 0.169846])

    @parameterized.named_parameters(
        [
            {"testcase_name": "dense", "sparse": False},
            {"testcase_name": "sparse", "sparse": True},
        ]
    )
    def test_multi_hot(self, sparse):
        if sparse and not backend.SUPPORTS_SPARSE_TENSORS:
            pytest.skip("Backend does not support sparse tensors")

        # Test 1D multi-hot.
        indices_1d = np.array([0, 1, 2, 3])
        expected_output_1d = np.array([1, 1, 1, 1])
        output_1d = knn.multi_hot(indices_1d, 4, sparse=sparse)
        self.assertAllClose(output_1d, expected_output_1d)
        self.assertSparse(output_1d, sparse)

        # Test 2D multi-hot.
        indices_2d = np.array([[0, 1], [2, 3]])
        expected_output_2d = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
        output_2d = knn.multi_hot(indices_2d, 4, sparse=sparse)
        self.assertAllClose(output_2d, expected_output_2d)
        self.assertSparse(output_2d, sparse)

        # Test 1D multi-hot with negative inputs
        indices_1d = np.array([0, -1, -1, 3])
        expected_output_1d = np.array([1, 0, 0, 1])
        output_1d = knn.multi_hot(indices_1d, 4, sparse=sparse)
        self.assertAllClose(output_1d, expected_output_1d)
        self.assertSparse(output_1d, sparse)

    def test_moments(self):
        # Test 1D moments
        x = np.array([0, 1, 2, 3, 4, 100, -200]).astype(np.float32)
        mean, variance = knn.moments(x, axes=[0])
        self.assertAllClose(mean, np.mean(x), atol=1e-5, rtol=1e-5)
        self.assertAllClose(variance, np.var(x), atol=1e-5, rtol=1e-5)

        # Test batch statistics for 4D moments (batch, height, width, channels)
        x = np.random.uniform(size=(2, 28, 28, 3)).astype(np.float32)
        mean, variance = knn.moments(x, axes=[0])
        self.assertAllClose(mean, np.mean(x, axis=0), atol=1e-5, rtol=1e-5)
        self.assertAllClose(variance, np.var(x, axis=0), atol=1e-5, rtol=1e-5)

        # Test global statistics for 4D moments (batch, height, width, channels)
        x = np.random.uniform(size=(2, 28, 28, 3)).astype(np.float32)
        mean, variance = knn.moments(x, axes=[0, 1, 2])
        expected_mean = np.mean(x, axis=(0, 1, 2))
        expected_variance = np.var(x, axis=(0, 1, 2))
        self.assertAllClose(mean, expected_mean, atol=1e-5, rtol=1e-5)
        self.assertAllClose(variance, expected_variance, atol=1e-5, rtol=1e-5)

        # Test keepdims
        x = np.random.uniform(size=(2, 28, 28, 3)).astype(np.float32)
        mean, variance = knn.moments(x, axes=[0, 1, 2], keepdims=True)
        expected_mean = np.mean(x, axis=(0, 1, 2), keepdims=True)
        expected_variance = np.var(x, axis=(0, 1, 2), keepdims=True)
        self.assertAllClose(mean, expected_mean, atol=1e-5, rtol=1e-5)
        self.assertAllClose(variance, expected_variance, atol=1e-5, rtol=1e-5)

        # Test float16 which causes overflow
        x = np.array(
            [-741.0, 353.2, 1099.0, -1807.0, 502.8, -83.4, 333.5, -130.9],
            dtype=np.float16,
        )
        mean, variance = knn.moments(x, axes=[0])
        expected_mean = np.mean(x.astype(np.float32)).astype(np.float16)
        # the output variance is clipped to the max value of np.float16 because
        # it is overflowed
        expected_variance = np.finfo(np.float16).max
        self.assertAllClose(mean, expected_mean, atol=1e-5, rtol=1e-5)
        self.assertAllClose(variance, expected_variance, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="synchronized=True only implemented for TF backend",
    )
    def test_moments_sync(self):
        # Test batch statistics for 4D moments (batch, height, width, channels)
        x = np.random.uniform(size=(2, 28, 28, 3)).astype(np.float32)
        mean, variance = knn.moments(x, axes=[0], synchronized=True)
        self.assertAllClose(mean, np.mean(x, axis=0), atol=1e-5, rtol=1e-5)
        self.assertAllClose(variance, np.var(x, axis=0), atol=1e-5, rtol=1e-5)

        # Test global statistics for 4D moments (batch, height, width, channels)
        x = np.random.uniform(size=(2, 28, 28, 3)).astype(np.float32)
        mean, variance = knn.moments(x, axes=[0, 1, 2], synchronized=True)
        expected_mean = np.mean(x, axis=(0, 1, 2))
        expected_variance = np.var(x, axis=(0, 1, 2))
        self.assertAllClose(mean, expected_mean, atol=1e-5, rtol=1e-5)
        self.assertAllClose(variance, expected_variance, atol=1e-5, rtol=1e-5)

        # Test keepdims
        x = np.random.uniform(size=(2, 28, 28, 3)).astype(np.float32)
        mean, variance = knn.moments(
            x, axes=[0, 1, 2], keepdims=True, synchronized=True
        )
        expected_mean = np.mean(x, axis=(0, 1, 2), keepdims=True)
        expected_variance = np.var(x, axis=(0, 1, 2), keepdims=True)
        self.assertAllClose(mean, expected_mean, atol=1e-5, rtol=1e-5)
        self.assertAllClose(variance, expected_variance, atol=1e-5, rtol=1e-5)

    @parameterized.product(dtype=["float16", "float32"])
    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="synchronized=True only implemented for TF backend",
    )
    def test_moments_sync_with_distribution_strategy(self, dtype):
        from keras.src.utils.module_utils import tensorflow as tf

        # Config 2 CPUs for testing.
        logical_cpus = tf.config.list_logical_devices("CPU")
        if len(logical_cpus) == 1:
            from tensorflow.python.eager import context

            context._reset_context()
            tf.config.set_logical_device_configuration(
                tf.config.list_physical_devices("CPU")[0],
                [
                    tf.config.LogicalDeviceConfiguration(),
                    tf.config.LogicalDeviceConfiguration(),
                ],
            )

        @tf.function()
        def test_on_moments(inputs):
            return knn.moments(
                inputs, axes=-1, keepdims=True, synchronized=True
            )

        # Test output of moments.
        inputs = tf.constant([5.0, 9.0, 1.0, 3.0], dtype=dtype)
        strategy = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1"])
        with strategy.scope():
            mean, variance = strategy.run(test_on_moments, args=(inputs,))
            self.assertEqual(mean.values[0], 4.5)
            self.assertEqual(variance.values[0], 8.75)
            self.assertEqual(variance.values[0], 8.75)

    def test_batch_normalization(self):
        x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mean = np.array([0.2, 0.3, 0.4])
        variance = np.array([4.0, 16.0, 64.0])
        output = knn.batch_normalization(
            x,
            mean,
            variance,
            axis=-1,
            offset=np.array([5.0, 10.0, 15.0]),
            scale=np.array([10.0, 20.0, 30.0]),
            epsilon=1e-7,
        )
        expected_output = np.array([[4.5, 9.5, 14.625], [6.0, 11.0, 15.75]])
        self.assertAllClose(output, expected_output)

        output = knn.batch_normalization(
            x,
            mean,
            variance,
            axis=1,
            epsilon=1e-7,
        )
        expected_output = np.array(
            [[-0.05, -0.025, -0.0125], [0.1, 0.05, 0.025]]
        )
        self.assertAllClose(output, expected_output)

        output = knn.batch_normalization(
            np.random.uniform(size=[2, 3, 3, 5]),
            np.random.uniform(size=[5]),
            np.random.uniform(size=[5]),
            axis=3,
            offset=np.random.uniform(size=[5]),
            scale=np.random.uniform(size=[5]),
        )
        self.assertEqual(tuple(output.shape), (2, 3, 3, 5))

    def test_ctc_loss(self):
        labels = np.array([[1, 2, 1], [1, 2, 2]])
        outputs = np.array(
            [
                [[0.4, 0.8, 0.4], [0.2, 0.8, 0.3], [0.9, 0.4, 0.5]],
                [[0.4, 0.8, 0.4], [0.2, 0.3, 0.3], [0.4, 0.3, 0.2]],
            ]
        )

        label_length = np.array([3, 2])
        output_length = np.array([3, 2])

        result = knn.ctc_loss(labels, outputs, label_length, output_length)
        self.assertAllClose(result, np.array([3.4411672, 1.91680186]))

    def test_ctc_decode(self):
        inputs = np.array(
            [
                [
                    [0.1, 0.4, 0.2, 0.4],
                    [0.3, -0.3, 0.4, 0.2],
                    [0.3, 0.2, 0.4, 0.3],
                ],
                [
                    [0.7, 0.4, 0.3, 0.2],
                    [0.3, 0.3, 0.4, 0.1],
                    [0.6, -0.1, 0.1, 0.5],
                ],
                [
                    [0.1, 0.4, 0.2, 0.7],
                    [0.3, 0.3, -0.2, 0.7],
                    [0.3, 0.2, 0.4, 0.1],
                ],
            ]
        )
        labels = np.array([[1, 2, -1], [2, -1, -1], [3, -1, -1]])
        score_labels = np.array([[-1.2], [-1.7], [-0.7]])
        repeated_labels = np.array([[1, 2, 2], [2, -1, -1], [3, -1, -1]])

        # Test strategy="greedy" and merge_repeated=True
        (decoded,), scores = knn.ctc_decode(
            inputs,
            sequence_lengths=[3, 3, 1],
            strategy="greedy",
            mask_index=0,
        )
        self.assertAllClose(decoded, labels)
        self.assertAllClose(scores, score_labels)

        # Test strategy="greedy" and merge_repeated=False
        (decoded,), scores = knn.ctc_decode(
            inputs,
            sequence_lengths=[3, 3, 1],
            strategy="greedy",
            merge_repeated=False,
            mask_index=0,
        )
        self.assertAllClose(decoded, repeated_labels)
        self.assertAllClose(scores, score_labels)

        if backend.backend() == "torch":
            self.skipTest("torch doesn't support 'beam_search' strategy")

        labels = np.array(
            [
                [[1, 2, -1], [2, -1, -1], [3, -1, -1]],
                [[2, -1, -1], [3, -1, -1], [1, -1, -1]],
            ]
        )
        score_labels = np.array(
            [
                [-2.426537, -2.435596],
                [-2.127681, -2.182338],
                [-1.063386, -1.363386],
            ]
        )
        beam_width = 4
        top_paths = 2

        # Test strategy="beam_search"
        decoded, scores = knn.ctc_decode(
            inputs,
            sequence_lengths=[3, 3, 1],
            strategy="beam_search",
            beam_width=beam_width,
            top_paths=top_paths,
            mask_index=0,
        )
        self.assertAllClose(decoded, labels)
        self.assertAllClose(scores, score_labels)

    def test_normalize(self):
        x = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
        self.assertAllClose(
            knn.normalize(x, axis=None),
            [
                [0.18898225, 0.3779645, 0.56694674],
                [0.18898225, 0.3779645, 0.56694674],
            ],
        )
        self.assertAllClose(
            knn.normalize(x, axis=0),
            [
                [0.70710677, 0.70710677, 0.70710677],
                [0.70710677, 0.70710677, 0.70710677],
            ],
        )
        self.assertAllClose(
            knn.normalize(x, axis=-1),
            [
                [0.26726124, 0.53452247, 0.8017837],
                [0.26726124, 0.53452247, 0.8017837],
            ],
        )
        self.assertAllClose(
            knn.normalize(x, order=3),
            [
                [0.30285344, 0.6057069, 0.9085603],
                [0.30285344, 0.6057069, 0.9085603],
            ],
        )

    def test_psnr(self):
        x1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        x2 = np.array([[0.2, 0.2, 0.3], [0.4, 0.6, 0.6]])
        max_val = 1.0
        expected_psnr_1 = 20 * np.log10(max_val) - 10 * np.log10(
            np.mean(np.square(x1 - x2))
        )
        psnr_1 = knn.psnr(x1, x2, max_val)
        self.assertAlmostEqual(psnr_1, expected_psnr_1)

        x3 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        x4 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        max_val = 1.0
        expected_psnr_2 = 20 * np.log10(max_val) - 10 * np.log10(
            np.mean(np.square(x3 - x4))
        )
        psnr_2 = knn.psnr(x3, x4, max_val)
        self.assertAlmostEqual(psnr_2, expected_psnr_2)


class NNOpsDtypeTest(testing.TestCase, parameterized.TestCase):
    """Test the dtype to verify that the behavior matches JAX."""

    FLOAT_DTYPES = dtypes.FLOAT_TYPES

    def setUp(self):
        from jax.experimental import enable_x64

        self.jax_enable_x64 = enable_x64()
        self.jax_enable_x64.__enter__()
        return super().setUp()

    def tearDown(self) -> None:
        self.jax_enable_x64.__exit__(None, None, None)
        return super().tearDown()

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_elu(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.elu(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.elu(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Elu().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_gelu(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)

        # approximate = True
        expected_dtype = standardize_dtype(jnn.gelu(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.gelu(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Gelu().symbolic_call(x).dtype),
            expected_dtype,
        )
        # approximate = False
        expected_dtype = standardize_dtype(jnn.gelu(x_jax, False).dtype)

        self.assertEqual(
            standardize_dtype(knn.gelu(x, False).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Gelu(False).symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_hard_sigmoid(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.hard_sigmoid(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.hard_sigmoid(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.HardSigmoid().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_hard_silu(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.hard_silu(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.hard_silu(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.HardSilu().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_leaky_relu(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.leaky_relu(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.leaky_relu(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.LeakyRelu().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_log_sigmoid(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.log_sigmoid(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.log_sigmoid(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.LogSigmoid().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_log_softmax(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((10,), dtype=dtype)
        x_jax = jnp.ones((10,), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.log_softmax(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.log_softmax(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.LogSoftmax().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_relu(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.relu(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.relu(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Relu().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_relu6(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.relu6(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.relu6(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Relu6().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_selu(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.selu(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.selu(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Selu().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_sigmoid(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.sigmoid(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.sigmoid(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Sigmoid().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_silu(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.silu(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.silu(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Silu().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_softplus(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.softplus(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.softplus(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Softplus().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_softmax(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((10,), dtype=dtype)
        x_jax = jnp.ones((10,), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.softmax(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.softmax(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Softmax().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_softsign(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.soft_sign(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.softsign(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Softsign().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_ctc_loss(self, dtype):
        labels = knp.array([[1, 2, 1]], dtype="int32")
        outputs = knp.array(
            [[[0.4, 0.8, 0.4], [0.2, 0.8, 0.3], [0.9, 0.4, 0.5]]], dtype=dtype
        )
        label_length = knp.array([3])
        output_length = knp.array([3])
        expected_dtype = (
            "float32" if dtype in ("float16", "bfloat16") else dtype
        )

        self.assertEqual(
            standardize_dtype(
                knn.ctc_loss(labels, outputs, label_length, output_length).dtype
            ),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(
                knn.CTCLoss()
                .symbolic_call(labels, outputs, label_length, output_length)
                .dtype
            ),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_ctc_decode(self, dtype):
        inputs = knp.array(
            [[[0.4, 0.8, 0.4], [0.2, 0.8, 0.3], [0.9, 0.4, 0.5]]], dtype=dtype
        )
        sequence_length = knp.array([3])
        expected_dtype = backend.result_type(dtype, "float32")

        # Test strategy="greedy"
        decoded, scores = knn.ctc_decode(
            inputs, sequence_length, strategy="greedy"
        )
        self.assertEqual(standardize_dtype(decoded.dtype), "int32")
        self.assertEqual(standardize_dtype(scores.dtype), expected_dtype)
        decoded, scores = knn.CTCDecode(strategy="greedy").symbolic_call(
            inputs, sequence_length
        )
        self.assertEqual(standardize_dtype(decoded.dtype), "int32")
        self.assertEqual(standardize_dtype(scores.dtype), expected_dtype)

        if backend.backend() == "torch":
            self.skipTest("torch doesn't support 'beam_search' strategy")

        # Test strategy="beam_search"
        decoded, scores = knn.ctc_decode(
            inputs, sequence_length, strategy="beam_search"
        )
        self.assertEqual(standardize_dtype(decoded.dtype), "int32")
        self.assertEqual(standardize_dtype(scores.dtype), expected_dtype)
        decoded, scores = knn.CTCDecode(strategy="beam_search").symbolic_call(
            inputs, sequence_length
        )
        self.assertEqual(standardize_dtype(decoded.dtype), "int32")
        self.assertEqual(standardize_dtype(scores.dtype), expected_dtype)


class NNOpsBehaviorTest(testing.TestCase, parameterized.TestCase):
    def test_logit_recovery_binary_crossentropy(self):
        layer = layers.Dense(
            4, activation="sigmoid", use_bias=False, kernel_initializer="ones"
        )
        loss = losses.BinaryCrossentropy()
        x = np.array([[1.4, 1.6, 0.8]])
        y = np.array([[0.2, 0.6, 0.1, 0.3]])
        loss_value = loss(y, layer(x))
        self.assertAllClose(loss_value, 2.682124)

        model = models.Sequential([layer])
        model.compile(loss="binary_crossentropy", optimizer="sgd")
        out = model.evaluate(x, y)
        self.assertAllClose(out, 2.682124)

    def test_softmax_on_axis_with_size_one_warns(self):
        x = np.array([[1.0]])
        # Applying softmax on the second axis, which has size 1
        axis = 1

        # Expected warning message
        expected_warning_regex = (
            r"You are using a softmax over axis 1 "
            r"of a tensor of shape \(1, 1\)\. This axis "
            r"has size 1\. The softmax operation will always return "
            r"the value 1, which is likely not what you intended\. "
            r"Did you mean to use a sigmoid instead\?"
        )

        with self.assertWarnsRegex(UserWarning, expected_warning_regex):
            knn.softmax(x, axis)

    def test_normalize_order_validation(self):
        # Test with a non-integer order
        with self.assertRaisesRegex(
            ValueError, "Argument `order` must be an int >= 1"
        ):
            knn.normalize(np.array([1, 2, 3]), order="a")

        # Test with a negative integer
        with self.assertRaisesRegex(
            ValueError, "Argument `order` must be an int >= 1"
        ):
            knn.normalize(np.array([1, 2, 3]), order=-1)

        # Test with zero
        with self.assertRaisesRegex(
            ValueError, "Argument `order` must be an int >= 1"
        ):
            knn.normalize(np.array([1, 2, 3]), order=0)

        # Test with a floating-point number
        with self.assertRaisesRegex(
            ValueError, "Argument `order` must be an int >= 1"
        ):
            knn.normalize(np.array([1, 2, 3]), order=2.5)

    def test_check_shape_first_dim_mismatch(self):
        name1, shape1 = "labels", (2, 3)
        name2, shape2 = "logits", (3, 4, 5)
        ctc_loss_instance = knn.CTCLoss(mask_index=-1)
        with self.assertRaisesRegex(
            ValueError, "must have the same first dimension"
        ):
            ctc_loss_instance._check_shape_first_dim(
                name1, shape1, name2, shape2
            )

    def test_invalid_strategy_ctc_decode(self):
        inputs = np.array(
            [
                [
                    [0.1, 0.4, 0.2, 0.4],
                    [0.3, 0.3, 0.4, 0.2],
                    [0.3, 0.2, 0.4, 0.3],
                ]
            ]
        )
        beam_width = 4
        top_paths = 2
        with self.assertRaisesRegex(ValueError, "Invalid strategy"):
            knn.ctc_decode(
                inputs,
                sequence_lengths=[3, 3, 1],
                strategy="invalid",
                beam_width=beam_width,
                top_paths=top_paths,
            )
