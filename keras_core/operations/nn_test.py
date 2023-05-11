import numpy as np
import pytest
import tensorflow as tf

from keras_core import backend
from keras_core import testing
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.operations import nn as knn


@pytest.mark.skipif(
    not backend.DYNAMIC_SHAPES_OK,
    reason="Backend does not support dynamic shapes",
)
class NNOpsDynamicShapeTest(testing.TestCase):
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

    def test_swish(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.swish(x).shape, (None, 2, 3))

    def test_log_sigmoid(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.log_sigmoid(x).shape, (None, 2, 3))

    def test_leaky_relu(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.leaky_relu(x).shape, (None, 2, 3))

    def test_hard_sigmoid(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.hard_sigmoid(x).shape, (None, 2, 3))

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

    def test_log_softmax(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.log_softmax(x).shape, (None, 2, 3))
        self.assertEqual(knn.log_softmax(x, axis=1).shape, (None, 2, 3))
        self.assertEqual(knn.log_softmax(x, axis=-1).shape, (None, 2, 3))

    def test_max_pool(self):
        x = KerasTensor([None, 8, 3])
        self.assertEqual(knn.max_pool(x, 2, 1).shape, (None, 7, 3))
        self.assertEqual(
            knn.max_pool(x, 2, 2, padding="same").shape, (None, 4, 3)
        )

        x = KerasTensor([None, 8, 8, 3])
        self.assertEqual(knn.max_pool(x, 2, 1).shape, (None, 7, 7, 3))
        self.assertEqual(
            knn.max_pool(x, 2, 2, padding="same").shape, (None, 4, 4, 3)
        )
        self.assertEqual(
            knn.max_pool(x, (2, 2), (2, 2), padding="same").shape,
            (None, 4, 4, 3),
        )

    def test_average_pool(self):
        x = KerasTensor([None, 8, 3])
        self.assertEqual(knn.average_pool(x, 2, 1).shape, (None, 7, 3))
        self.assertEqual(
            knn.average_pool(x, 2, 2, padding="same").shape, (None, 4, 3)
        )

        x = KerasTensor([None, 8, 8, 3])
        self.assertEqual(knn.average_pool(x, 2, 1).shape, (None, 7, 7, 3))
        self.assertEqual(
            knn.average_pool(x, 2, 2, padding="same").shape, (None, 4, 4, 3)
        )
        self.assertEqual(
            knn.average_pool(x, (2, 2), (2, 2), padding="same").shape,
            (None, 4, 4, 3),
        )

    def test_conv(self):
        # Test 1D conv.
        inputs_1d = KerasTensor([None, 20, 3])
        kernel = KerasTensor([4, 3, 2])
        self.assertEqual(
            knn.conv(inputs_1d, kernel, 1, padding="valid").shape, (None, 17, 2)
        )
        self.assertEqual(
            knn.conv(inputs_1d, kernel, 1, padding="same").shape, (None, 20, 2)
        )
        self.assertEqual(
            knn.conv(inputs_1d, kernel, (2,), dilation_rate=2).shape,
            (None, 7, 2),
        )

        # Test 2D conv.
        inputs_2d = KerasTensor([None, 10, 10, 3])
        kernel = KerasTensor([2, 2, 3, 2])
        self.assertEqual(
            knn.conv(inputs_2d, kernel, 1, padding="valid").shape,
            (None, 9, 9, 2),
        )
        self.assertEqual(
            knn.conv(inputs_2d, kernel, 1, padding="same").shape,
            (None, 10, 10, 2),
        )
        self.assertEqual(
            knn.conv(inputs_2d, kernel, (2, 1), dilation_rate=(2, 1)).shape,
            (None, 4, 9, 2),
        )

        # Test 3D conv.
        inputs_3d = KerasTensor([None, 8, 8, 8, 3])
        kernel = KerasTensor([3, 3, 3, 3, 2])
        self.assertEqual(
            knn.conv(inputs_3d, kernel, 1, padding="valid").shape,
            (None, 6, 6, 6, 2),
        )
        self.assertEqual(
            knn.conv(inputs_3d, kernel, (2, 1, 2), padding="same").shape,
            (None, 4, 8, 4, 2),
        )
        self.assertEqual(
            knn.conv(
                inputs_3d, kernel, 1, padding="valid", dilation_rate=(1, 2, 2)
            ).shape,
            (None, 6, 4, 4, 2),
        )

    def test_depthwise_conv(self):
        # Test 1D depthwise conv.
        inputs_1d = KerasTensor([None, 20, 3])
        kernel = KerasTensor([4, 3, 1])
        self.assertEqual(
            knn.depthwise_conv(inputs_1d, kernel, 1, padding="valid").shape,
            (None, 17, 3),
        )
        self.assertEqual(
            knn.depthwise_conv(inputs_1d, kernel, (1,), padding="same").shape,
            (None, 20, 3),
        )
        self.assertEqual(
            knn.depthwise_conv(inputs_1d, kernel, 2, dilation_rate=2).shape,
            (None, 7, 3),
        )

        # Test 2D depthwise conv.
        inputs_2d = KerasTensor([None, 10, 10, 3])
        kernel = KerasTensor([2, 2, 3, 1])
        self.assertEqual(
            knn.depthwise_conv(inputs_2d, kernel, 1, padding="valid").shape,
            (None, 9, 9, 3),
        )
        self.assertEqual(
            knn.depthwise_conv(inputs_2d, kernel, (1, 2), padding="same").shape,
            (None, 10, 5, 3),
        )
        self.assertEqual(
            knn.depthwise_conv(inputs_2d, kernel, 2, dilation_rate=2).shape,
            (None, 4, 4, 3),
        )
        self.assertEqual(
            knn.depthwise_conv(
                inputs_2d, kernel, 2, dilation_rate=(2, 1)
            ).shape,
            (None, 4, 5, 3),
        )

    def test_separable_conv(self):
        # Test 1D separable conv.
        inputs_1d = KerasTensor([None, 20, 3])
        kernel = KerasTensor([4, 3, 2])
        pointwise_kernel = KerasTensor([1, 6, 5])
        self.assertEqual(
            knn.separable_conv(
                inputs_1d, kernel, pointwise_kernel, 1, padding="valid"
            ).shape,
            (None, 17, 5),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_1d, kernel, pointwise_kernel, 1, padding="same"
            ).shape,
            (None, 20, 5),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_1d, kernel, pointwise_kernel, 2, dilation_rate=2
            ).shape,
            (None, 7, 5),
        )

        # Test 2D separable conv.
        inputs_2d = KerasTensor([None, 10, 10, 3])
        kernel = KerasTensor([2, 2, 3, 2])
        pointwise_kernel = KerasTensor([1, 1, 6, 5])
        self.assertEqual(
            knn.separable_conv(
                inputs_2d, kernel, pointwise_kernel, 1, padding="valid"
            ).shape,
            (None, 9, 9, 5),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_2d, kernel, pointwise_kernel, (1, 2), padding="same"
            ).shape,
            (None, 10, 5, 5),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_2d, kernel, pointwise_kernel, 2, dilation_rate=(2, 1)
            ).shape,
            (None, 4, 5, 5),
        )

    def test_conv_transpose(self):
        inputs_1d = KerasTensor([None, 4, 3])
        kernel = KerasTensor([2, 5, 3])
        self.assertEqual(
            knn.conv_transpose(inputs_1d, kernel, 2).shape, (None, 8, 5)
        )
        self.assertEqual(
            knn.conv_transpose(inputs_1d, kernel, 2, padding="same").shape,
            (None, 8, 5),
        )
        self.assertEqual(
            knn.conv_transpose(
                inputs_1d, kernel, 5, padding="valid", output_padding=4
            ).shape,
            (None, 21, 5),
        )

        inputs_2d = KerasTensor([None, 4, 4, 3])
        kernel = KerasTensor([2, 2, 5, 3])
        self.assertEqual(
            knn.conv_transpose(inputs_2d, kernel, 2).shape, (None, 8, 8, 5)
        )
        self.assertEqual(
            knn.conv_transpose(inputs_2d, kernel, (2, 2), padding="same").shape,
            (None, 8, 8, 5),
        )
        self.assertEqual(
            knn.conv_transpose(
                inputs_2d, kernel, (5, 5), padding="valid", output_padding=4
            ).shape,
            (None, 21, 21, 5),
        )

    def test_one_hot(self):
        x = KerasTensor([None, 3, 1])
        self.assertEqual(knn.one_hot(x, 5).shape, (None, 3, 1, 5))
        self.assertEqual(knn.one_hot(x, 5, 1).shape, (None, 5, 3, 1))
        self.assertEqual(knn.one_hot(x, 5, 2).shape, (None, 3, 5, 1))


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

    def test_swish(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.swish(x).shape, (1, 2, 3))

    def test_log_sigmoid(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.log_sigmoid(x).shape, (1, 2, 3))

    def test_leaky_relu(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.leaky_relu(x).shape, (1, 2, 3))

    def test_hard_sigmoid(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.hard_sigmoid(x).shape, (1, 2, 3))

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
        x = KerasTensor([1, 8, 3])
        self.assertEqual(knn.max_pool(x, 2, 1).shape, (1, 7, 3))
        self.assertEqual(knn.max_pool(x, 2, 2, padding="same").shape, (1, 4, 3))

        x = KerasTensor([1, 8, 8, 3])
        self.assertEqual(knn.max_pool(x, 2, 1).shape, (1, 7, 7, 3))
        self.assertEqual(
            knn.max_pool(x, 2, 2, padding="same").shape, (1, 4, 4, 3)
        )
        self.assertEqual(
            knn.max_pool(x, (2, 2), (2, 2), padding="same").shape, (1, 4, 4, 3)
        )

    def test_average_pool(self):
        x = KerasTensor([1, 8, 3])
        self.assertEqual(knn.average_pool(x, 2, 1).shape, (1, 7, 3))
        self.assertEqual(
            knn.average_pool(x, 2, 2, padding="same").shape, (1, 4, 3)
        )

        x = KerasTensor([1, 8, 8, 3])
        self.assertEqual(knn.average_pool(x, 2, 1).shape, (1, 7, 7, 3))
        self.assertEqual(
            knn.average_pool(x, 2, 2, padding="same").shape, (1, 4, 4, 3)
        )
        self.assertEqual(
            knn.average_pool(x, (2, 2), (2, 2), padding="same").shape,
            (1, 4, 4, 3),
        )

    def test_conv(self):
        # Test 1D conv.
        inputs_1d = KerasTensor([2, 20, 3])
        kernel = KerasTensor([4, 3, 2])
        self.assertEqual(
            knn.conv(inputs_1d, kernel, 1, padding="valid").shape, (2, 17, 2)
        )
        self.assertEqual(
            knn.conv(inputs_1d, kernel, 1, padding="same").shape, (2, 20, 2)
        )
        self.assertEqual(
            knn.conv(inputs_1d, kernel, (2,), dilation_rate=2).shape, (2, 7, 2)
        )

        # Test 2D conv.
        inputs_2d = KerasTensor([2, 10, 10, 3])
        kernel = KerasTensor([2, 2, 3, 2])
        self.assertEqual(
            knn.conv(inputs_2d, kernel, 1, padding="valid").shape, (2, 9, 9, 2)
        )
        self.assertEqual(
            knn.conv(inputs_2d, kernel, 1, padding="same").shape, (2, 10, 10, 2)
        )
        self.assertEqual(
            knn.conv(inputs_2d, kernel, (2, 1), dilation_rate=(2, 1)).shape,
            (2, 4, 9, 2),
        )

        # Test 3D conv.
        inputs_3d = KerasTensor([2, 8, 8, 8, 3])
        kernel = KerasTensor([3, 3, 3, 3, 2])
        self.assertEqual(
            knn.conv(inputs_3d, kernel, 1, padding="valid").shape,
            (2, 6, 6, 6, 2),
        )
        self.assertEqual(
            knn.conv(inputs_3d, kernel, (2, 1, 2), padding="same").shape,
            (2, 4, 8, 4, 2),
        )
        self.assertEqual(
            knn.conv(
                inputs_3d, kernel, 1, padding="valid", dilation_rate=(1, 2, 2)
            ).shape,
            (2, 6, 4, 4, 2),
        )

    def test_depthwise_conv(self):
        # Test 1D depthwise conv.
        inputs_1d = KerasTensor([2, 20, 3])
        kernel = KerasTensor([4, 3, 1])
        self.assertEqual(
            knn.depthwise_conv(inputs_1d, kernel, 1, padding="valid").shape,
            (2, 17, 3),
        )
        self.assertEqual(
            knn.depthwise_conv(inputs_1d, kernel, (1,), padding="same").shape,
            (2, 20, 3),
        )
        self.assertEqual(
            knn.depthwise_conv(inputs_1d, kernel, 2, dilation_rate=2).shape,
            (2, 7, 3),
        )

        # Test 2D depthwise conv.
        inputs_2d = KerasTensor([2, 10, 10, 3])
        kernel = KerasTensor([2, 2, 3, 1])
        self.assertEqual(
            knn.depthwise_conv(inputs_2d, kernel, 1, padding="valid").shape,
            (2, 9, 9, 3),
        )
        self.assertEqual(
            knn.depthwise_conv(inputs_2d, kernel, (1, 2), padding="same").shape,
            (2, 10, 5, 3),
        )
        self.assertEqual(
            knn.depthwise_conv(inputs_2d, kernel, 2, dilation_rate=2).shape,
            (2, 4, 4, 3),
        )
        self.assertEqual(
            knn.depthwise_conv(
                inputs_2d, kernel, 2, dilation_rate=(2, 1)
            ).shape,
            (2, 4, 5, 3),
        )

    def test_separable_conv(self):
        # Test 1D separable conv.
        inputs_1d = KerasTensor([2, 20, 3])
        kernel = KerasTensor([4, 3, 2])
        pointwise_kernel = KerasTensor([1, 6, 5])
        self.assertEqual(
            knn.separable_conv(
                inputs_1d, kernel, pointwise_kernel, 1, padding="valid"
            ).shape,
            (2, 17, 5),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_1d, kernel, pointwise_kernel, 1, padding="same"
            ).shape,
            (2, 20, 5),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_1d, kernel, pointwise_kernel, 2, dilation_rate=2
            ).shape,
            (2, 7, 5),
        )

        # Test 2D separable conv.
        inputs_2d = KerasTensor([2, 10, 10, 3])
        kernel = KerasTensor([2, 2, 3, 2])
        pointwise_kernel = KerasTensor([1, 1, 6, 5])
        self.assertEqual(
            knn.separable_conv(
                inputs_2d, kernel, pointwise_kernel, 1, padding="valid"
            ).shape,
            (2, 9, 9, 5),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_2d, kernel, pointwise_kernel, (1, 2), padding="same"
            ).shape,
            (2, 10, 5, 5),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_2d, kernel, pointwise_kernel, 2, dilation_rate=(2, 1)
            ).shape,
            (2, 4, 5, 5),
        )

    def test_conv_transpose(self):
        inputs_1d = KerasTensor([2, 4, 3])
        kernel = KerasTensor([2, 5, 3])
        self.assertEqual(
            knn.conv_transpose(inputs_1d, kernel, 2).shape, (2, 8, 5)
        )
        self.assertEqual(
            knn.conv_transpose(inputs_1d, kernel, 2, padding="same").shape,
            (2, 8, 5),
        )
        self.assertEqual(
            knn.conv_transpose(
                inputs_1d, kernel, 5, padding="valid", output_padding=4
            ).shape,
            (2, 21, 5),
        )

        inputs_2d = KerasTensor([2, 4, 4, 3])
        kernel = KerasTensor([2, 2, 5, 3])
        self.assertEqual(
            knn.conv_transpose(inputs_2d, kernel, 2).shape, (2, 8, 8, 5)
        )
        self.assertEqual(
            knn.conv_transpose(inputs_2d, kernel, (2, 2), padding="same").shape,
            (2, 8, 8, 5),
        )
        self.assertEqual(
            knn.conv_transpose(
                inputs_2d, kernel, (5, 5), padding="valid", output_padding=4
            ).shape,
            (2, 21, 21, 5),
        )

    def test_one_hot(self):
        x = KerasTensor([2, 3, 1])
        self.assertEqual(knn.one_hot(x, 5).shape, (2, 3, 1, 5))
        self.assertEqual(knn.one_hot(x, 5, 1).shape, (2, 5, 3, 1))
        self.assertEqual(knn.one_hot(x, 5, 2).shape, (2, 3, 5, 1))

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


class NNOpsCorrectnessTest(testing.TestCase):
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

    def test_swish(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.swish(x), [-0.26894143, 0.0, 0.7310586, 1.7615943, 2.8577223]
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

    def test_elu(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.elu(x),
            [-0.63212055, 0, 1, 2, 3],
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
            knn.softmax(x),
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

    def test_log_softmax(self):
        x = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
        self.assertAllClose(
            knn.log_softmax(x),
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

    def test_max_pool(self):
        # Test 1D max pooling.
        x = np.arange(120, dtype=float).reshape([2, 20, 3])
        self.assertAllClose(
            knn.max_pool(x, 2, 1, padding="valid"),
            tf.nn.max_pool1d(x, 2, 1, padding="VALID"),
        )
        self.assertAllClose(
            knn.max_pool(x, 2, 2, padding="same"),
            tf.nn.max_pool1d(x, 2, 2, padding="SAME"),
        )

        # Test 2D max pooling.
        x = np.arange(540, dtype=float).reshape([2, 10, 9, 3])
        self.assertAllClose(
            knn.max_pool(x, 2, 1, padding="valid"),
            tf.nn.max_pool2d(x, 2, 1, padding="VALID"),
        )
        self.assertAllClose(
            knn.max_pool(x, 2, (2, 1), padding="same"),
            tf.nn.max_pool2d(x, 2, (2, 1), padding="SAME"),
        )

    def test_average_pool(self):
        # Test 1D max pooling.
        x = np.arange(120, dtype=float).reshape([2, 20, 3])
        self.assertAllClose(
            knn.average_pool(x, 2, 1, padding="valid"),
            tf.nn.avg_pool1d(x, 2, 1, padding="VALID"),
        )
        self.assertAllClose(
            knn.average_pool(x, 2, 2, padding="same"),
            tf.nn.avg_pool1d(x, 2, 2, padding="SAME"),
        )

        # Test 2D max pooling.
        x = np.arange(540, dtype=float).reshape([2, 10, 9, 3])
        self.assertAllClose(
            knn.average_pool(x, 2, 1, padding="valid"),
            tf.nn.avg_pool2d(x, 2, 1, padding="VALID"),
        )
        self.assertAllClose(
            knn.average_pool(x, 2, (2, 1), padding="same"),
            tf.nn.avg_pool2d(x, 2, (2, 1), padding="SAME"),
        )

    def test_conv(self):
        # Test 1D conv.
        inputs_1d = np.arange(120, dtype=float).reshape([2, 20, 3])
        kernel = np.arange(24, dtype=float).reshape([4, 3, 2])

        outputs = knn.conv(inputs_1d, kernel, 1, padding="valid")
        expected = tf.nn.conv1d(inputs_1d, kernel, 1, padding="VALID")
        self.assertAllClose(outputs, expected)

        outputs = knn.conv(inputs_1d, kernel, 2, padding="same")
        expected = tf.nn.conv1d(inputs_1d, kernel, 2, padding="SAME")
        self.assertAllClose(outputs, expected)

        outputs = knn.conv(
            inputs_1d, kernel, 1, padding="same", dilation_rate=2
        )
        expected = tf.nn.conv1d(
            inputs_1d, kernel, 1, padding="SAME", dilations=2
        )
        self.assertAllClose(outputs, expected)

        # Test 2D conv.
        inputs_2d = np.arange(600, dtype=float).reshape([2, 10, 10, 3])
        kernel = np.arange(24, dtype=float).reshape([2, 2, 3, 2])

        outputs = knn.conv(inputs_2d, kernel, 1, padding="valid")
        expected = tf.nn.conv2d(inputs_2d, kernel, 1, padding="VALID")
        self.assertAllClose(outputs, expected)

        outputs = knn.conv(inputs_2d, kernel, (1, 2), padding="valid")
        expected = tf.nn.conv2d(inputs_2d, kernel, (1, 2), padding="VALID")
        self.assertAllClose(outputs, expected)

        outputs = knn.conv(inputs_2d, kernel, 2, padding="same")
        expected = tf.nn.conv2d(inputs_2d, kernel, 2, padding="SAME")
        self.assertAllClose(outputs, expected)

        outputs = knn.conv(
            inputs_2d, kernel, 1, padding="same", dilation_rate=2
        )
        expected = tf.nn.conv2d(
            inputs_2d, kernel, 1, padding="SAME", dilations=2
        )
        self.assertAllClose(outputs, expected)

        outputs = knn.conv(
            inputs_2d,
            kernel,
            1,
            padding="same",
            dilation_rate=(2, 1),
        )
        expected = tf.nn.conv2d(
            inputs_2d,
            kernel,
            1,
            padding="SAME",
            dilations=(2, 1),
        )
        self.assertAllClose(outputs, expected)

        # Test 3D conv.
        inputs_3d = np.arange(3072, dtype=float).reshape([2, 8, 8, 8, 3])
        kernel = np.arange(162, dtype=float).reshape([3, 3, 3, 3, 2])

        outputs = knn.conv(inputs_3d, kernel, 1, padding="valid")
        expected = tf.nn.conv3d(
            inputs_3d, kernel, (1, 1, 1, 1, 1), padding="VALID"
        )
        self.assertAllClose(outputs, expected, rtol=1e-5, atol=1e-5)

        outputs = knn.conv(
            inputs_3d,
            kernel,
            (1, 1, 1),
            padding="valid",
            dilation_rate=(1, 1, 1),
        )
        expected = tf.nn.conv3d(
            inputs_3d,
            kernel,
            (1, 1, 1, 1, 1),
            padding="VALID",
            dilations=(1, 1, 1, 1, 1),
        )
        self.assertAllClose(outputs, expected, rtol=1e-5, atol=1e-5)

        outputs = knn.conv(inputs_3d, kernel, 2, padding="same")
        expected = tf.nn.conv3d(
            inputs_3d, kernel, (1, 2, 2, 2, 1), padding="SAME"
        )
        self.assertAllClose(outputs, expected, rtol=1e-5, atol=1e-5)

    def test_depthwise_conv(self):
        # Test 2D conv.
        inputs_2d = np.arange(600, dtype=float).reshape([2, 10, 10, 3])
        kernel = np.arange(24, dtype=float).reshape([2, 2, 3, 2])

        outputs = knn.depthwise_conv(inputs_2d, kernel, 1, padding="valid")
        expected = tf.nn.depthwise_conv2d(
            inputs_2d, kernel, (1, 1, 1, 1), padding="VALID"
        )
        self.assertAllClose(outputs, expected)

        outputs = knn.depthwise_conv(inputs_2d, kernel, (1, 1), padding="valid")
        expected = tf.nn.depthwise_conv2d(
            inputs_2d, kernel, (1, 1, 1, 1), padding="VALID"
        )
        self.assertAllClose(outputs, expected)

        outputs = knn.depthwise_conv(inputs_2d, kernel, (2, 2), padding="same")
        expected = tf.nn.depthwise_conv2d(
            inputs_2d, kernel, (1, 2, 2, 1), padding="SAME"
        )
        self.assertAllClose(outputs, expected)

        outputs = knn.depthwise_conv(
            inputs_2d, kernel, 1, padding="same", dilation_rate=(2, 2)
        )
        expected = tf.nn.depthwise_conv2d(
            inputs_2d, kernel, (1, 1, 1, 1), padding="SAME", dilations=(2, 2)
        )
        self.assertAllClose(outputs, expected)

    def test_separable_conv(self):
        # Test 2D conv.
        inputs_2d = np.arange(600, dtype=float).reshape([2, 10, 10, 3])
        depthwise_kernel = np.arange(24, dtype=float).reshape([2, 2, 3, 2])
        pointwise_kernel = np.arange(72, dtype=float).reshape([1, 1, 6, 12])

        outputs = knn.separable_conv(
            inputs_2d, depthwise_kernel, pointwise_kernel, 1, padding="valid"
        )
        expected = tf.nn.separable_conv2d(
            inputs_2d,
            depthwise_kernel,
            pointwise_kernel,
            (1, 1, 1, 1),
            padding="VALID",
        )
        self.assertAllClose(outputs, expected)

        outputs = knn.separable_conv(
            inputs_2d,
            depthwise_kernel,
            pointwise_kernel,
            (1, 1),
            padding="valid",
        )
        expected = tf.nn.separable_conv2d(
            inputs_2d,
            depthwise_kernel,
            pointwise_kernel,
            (1, 1, 1, 1),
            padding="VALID",
        )
        self.assertAllClose(outputs, expected)

        outputs = knn.separable_conv(
            inputs_2d, depthwise_kernel, pointwise_kernel, 2, padding="same"
        )
        expected = tf.nn.separable_conv2d(
            inputs_2d,
            depthwise_kernel,
            pointwise_kernel,
            (1, 2, 2, 1),
            padding="SAME",
        )
        self.assertAllClose(outputs, expected)

        outputs = knn.separable_conv(
            inputs_2d,
            depthwise_kernel,
            pointwise_kernel,
            1,
            padding="same",
            dilation_rate=(2, 2),
        )
        expected = tf.nn.separable_conv2d(
            inputs_2d,
            depthwise_kernel,
            pointwise_kernel,
            (1, 1, 1, 1),
            padding="SAME",
            dilations=(2, 2),
        )
        self.assertAllClose(outputs, expected)

    def test_conv_transpose(self):
        # Test 1D conv.
        inputs_1d = np.arange(24, dtype=float).reshape([2, 4, 3])
        kernel = np.arange(30, dtype=float).reshape([2, 5, 3])
        outputs = knn.conv_transpose(inputs_1d, kernel, 2, padding="valid")
        expected = tf.nn.conv_transpose(
            inputs_1d, kernel, [2, 8, 5], 2, padding="VALID"
        )
        self.assertAllClose(outputs, expected)

        outputs = knn.conv_transpose(inputs_1d, kernel, 2, padding="same")
        expected = tf.nn.conv_transpose(
            inputs_1d, kernel, [2, 8, 5], 2, padding="SAME"
        )
        self.assertAllClose(outputs, expected)

        # Test 2D conv.
        inputs_2d = np.arange(96, dtype=float).reshape([2, 4, 4, 3])
        kernel = np.arange(60, dtype=float).reshape([2, 2, 5, 3])

        outputs = knn.conv_transpose(inputs_2d, kernel, (2, 2), padding="valid")
        expected = tf.nn.conv_transpose(
            inputs_2d, kernel, [2, 8, 8, 5], (2, 2), padding="VALID"
        )
        self.assertAllClose(outputs, expected)

        outputs = knn.conv_transpose(inputs_2d, kernel, 2, padding="same")
        expected = tf.nn.conv_transpose(
            inputs_2d, kernel, [2, 8, 8, 5], 2, padding="SAME"
        )
        self.assertAllClose(outputs, expected)

    def test_one_hot(self):
        # Test 1D one-hot.
        indices_1d = np.array([0, 1, 2, 3])
        self.assertAllClose(
            knn.one_hot(indices_1d, 4), tf.one_hot(indices_1d, 4)
        )
        self.assertAllClose(
            knn.one_hot(indices_1d, 4, axis=0),
            tf.one_hot(indices_1d, 4, axis=0),
        )

        # Test 2D one-hot.
        indices_2d = np.array([[0, 1], [2, 3]])
        self.assertAllClose(
            knn.one_hot(indices_2d, 4), tf.one_hot(indices_2d, 4)
        )
        self.assertAllClose(
            knn.one_hot(indices_2d, 4, axis=2),
            tf.one_hot(indices_2d, 4, axis=2),
        )
        self.assertAllClose(
            knn.one_hot(indices_2d, 4, axis=1),
            tf.one_hot(indices_2d, 4, axis=1),
        )

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
