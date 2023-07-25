import numpy as np
import pytest

from keras_core import layers
from keras_core import losses
from keras_core import models
from keras_core import ops
from keras_core import optimizers
from keras_core import testing
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.ops import core


class CoreOpsStaticShapeTest(testing.TestCase):
    def test_scatter(self):
        indices = KerasTensor((5, 2))
        values = KerasTensor((5,))
        shape = (4, 4)
        self.assertEqual(core.scatter(indices, values, shape).shape, (4, 4))

    def test_scatter_update(self):
        inputs = KerasTensor((4, 4))
        indices = KerasTensor((5, 2))
        updates = KerasTensor((5,))
        self.assertEqual(
            core.scatter_update(inputs, indices, updates).shape, (4, 4)
        )

        inputs = KerasTensor((4, 4, 4))
        indices = KerasTensor((5, 2))
        updates = KerasTensor((5, 4))
        self.assertEqual(
            core.scatter_update(inputs, indices, updates).shape, (4, 4, 4)
        )

    def test_slice_update(self):
        inputs = KerasTensor((4, 4))
        start_indices = KerasTensor((2,))
        updates = KerasTensor((2, 2))
        self.assertEqual(
            core.slice_update(inputs, start_indices, updates).shape, (4, 4)
        )

        inputs = KerasTensor((4, 4, 4))
        start_indices = KerasTensor((3,))
        updates = KerasTensor((2, 2, 2))
        self.assertEqual(
            core.slice_update(inputs, start_indices, updates).shape, (4, 4, 4)
        )

    def test_fori_loop(self):
        def body_fun(i, x):
            return x + i

        initial_value = KerasTensor((3, 5, 7))
        result = core.fori_loop(0, 10, body_fun, initial_value)
        self.assertEqual(result.shape, (3, 5, 7))


class CoreOpsCorrectnessTest(testing.TestCase):
    def test_scatter(self):
        # Test 1D
        indices = np.array([[1], [3], [4], [7]])
        values = np.array([9, 10, 11, 12])
        self.assertAllClose(
            core.scatter(indices, values, (8,)),
            [0, 9, 0, 10, 11, 0, 0, 12],
        )
        # Test 2D
        indices = np.array([[0, 1], [2, 0]])
        values = np.array([5, 10])
        self.assertAllClose(
            core.scatter(indices, values, (3, 2)), [[0, 5], [0, 0], [10, 0]]
        )
        # Test 3D
        indices = np.array([[1], [3]])
        values = np.array(
            [
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            ]
        )
        self.assertAllClose(
            core.scatter(indices, values, (4, 4, 4)),
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            ],
        )
        # Test slices
        indices = np.array([[2], [4]])
        values = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertAllClose(
            core.scatter(indices, values, (6, 3)),
            [[0, 0, 0], [0, 0, 0], [1, 2, 3], [0, 0, 0], [4, 5, 6], [0, 0, 0]],
        )
        # Duplicate indices
        indices = np.array([[0], [0]])
        values = np.array([1, 1])
        self.assertAllClose(core.scatter(indices, values, (1,)), [2])

    def test_scatter_update(self):
        # Test 1D.
        inputs = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        indices = [[1], [3], [4], [7]]
        updates = np.array([9, 10, 11, 12])
        self.assertAllClose(
            core.scatter_update(inputs, indices, updates),
            [0, 9, 0, 10, 11, 0, 0, 12],
        )

        # Test 2D.
        inputs = np.array([[1, 1], [1, 1], [1, 1]])
        indices = [[0, 1], [2, 0]]
        updates = np.array([5, 10])
        self.assertAllClose(
            core.scatter_update(inputs, indices, updates),
            [[1, 5], [1, 1], [10, 1]],
        )

        # Test updates has multiple dimension.
        inputs = np.ones([4, 4, 4])
        indices = [[1, 1], [2, 2]]
        updates = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.float64)
        outputs = core.scatter_update(inputs, indices, updates)
        self.assertAllClose(outputs[1, 1, :], [0, 1, 2, 3])
        self.assertAllClose(outputs[2, 2, :], [3, 2, 1, 0])

    def test_slice(self):
        # Test 1D.
        inputs = np.arange(10)
        start_indices = np.array([1])
        shape = np.array([4])
        self.assertAllClose(
            core.slice(inputs, start_indices, shape),
            [1, 2, 3, 4],
        )

        # Test 2D.
        inputs = np.broadcast_to(np.arange(10), (4, 10))
        start_indices = np.array([1, 1])
        shape = np.array([2, 4])
        self.assertAllClose(
            core.slice(inputs, start_indices, shape),
            [[1, 2, 3, 4], [1, 2, 3, 4]],
        )

        # Test N-D.
        inputs = np.broadcast_to(np.arange(10), (4, 4, 4, 10))
        start_indices = np.array([1, 1, 1, 1])
        shape = np.array([1, 2, 3, 4])
        outputs = core.slice(inputs, start_indices, shape)
        expected = np.broadcast_to(np.arange(1, 5), (1, 2, 3, 4))
        self.assertAllClose(outputs, expected)

    def test_dynamic_slice(self):
        def cond(index, inputs, sum):
            return index < 10

        def body(index, inputs, sum):
            sum = sum + core.slice(inputs, [index], [1])
            index = index + 1
            return index, inputs, sum

        index, inputs, sum = 0, np.arange(10), np.array([0])
        index, inputs, sum = core.while_loop(cond, body, (index, inputs, sum))
        self.assertAllClose(sum, [45])

    def test_slice_update(self):
        # Test 1D.
        inputs = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        start_indices = np.array([1])
        updates = np.array([9, 10, 11, 12])
        self.assertAllClose(
            core.slice_update(inputs, start_indices, updates),
            [0, 9, 10, 11, 12, 0, 0, 0],
        )

        # Test 2D.
        inputs = np.array([[1, 1], [1, 1], [1, 1]])
        start_indices = [1, 0]
        updates = np.array([[2, 2], [2, 2]])
        self.assertAllClose(
            core.slice_update(inputs, start_indices, updates),
            [[1, 1], [2, 2], [2, 2]],
        )

        # Test N-D.
        inputs = np.ones([4, 4, 4, 4])
        start_indices = [1, 1, 2, 2]
        updates = np.zeros([2, 2, 2, 2])
        outputs = core.slice_update(inputs, start_indices, updates)
        self.assertAllClose(outputs[1:3, 1:3, 2:4, 2:4], np.zeros([2, 2, 2, 2]))

    def test_while_loop(self):
        def cond(x, y):
            return x[0, 0] < 10

        def body(x, y):
            return x + 1, y + 1

        x = np.ones((2, 3))
        y = np.ones((3, 2))
        x, y = core.while_loop(cond, body, (x, y))
        self.assertAllClose(x, np.ones((2, 3)) * 10)
        self.assertAllClose(y, np.ones((3, 2)) * 10)

        x = np.ones((2, 3))
        y = np.ones((3, 2))
        x, y = core.while_loop(cond, body, (x, y), maximum_iterations=5)
        self.assertAllClose(x, np.ones((2, 3)) * 6)
        self.assertAllClose(y, np.ones((3, 2)) * 6)

    def test_fori_loop(self):
        def body_fun(i, x):
            return x + i

        initial_value = np.array(0)
        result = core.fori_loop(0, 10, body_fun, initial_value)
        self.assertAllClose(result, 45)

    @pytest.mark.requires_trainable_backend
    def test_stop_gradient(self):
        class ExampleLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w = self.add_weight(shape=(1,), initializer="zeros")
                self.b = self.add_weight(shape=(1,), initializer="zeros")

            def call(self, x, training=False):
                return x * ops.stop_gradient(self.w.value) + self.b

        model = models.Sequential([ExampleLayer()])
        model.compile(
            optimizer=optimizers.SGD(), loss=losses.MeanSquaredError()
        )
        rng = np.random.default_rng(0)
        x = np.ones((2, 4), dtype=np.float32)
        y = rng.standard_normal((2, 4), dtype=np.float32)
        model.fit(x, y, epochs=1, batch_size=2)
        self.assertEqual(model.layers[0].w.numpy(), 0.0)
        self.assertNotEqual(model.layers[0].b.numpy(), 0.0)

    def test_shape(self):
        x = np.ones((2, 3, 7, 1))
        self.assertAllEqual(core.shape(x), (2, 3, 7, 1))

        x = KerasTensor((None, 3, None, 1))
        self.assertAllEqual(core.shape(x), (None, 3, None, 1))

    def test_convert_to_tensor(self):
        x = np.ones((2,))
        x = ops.convert_to_tensor(x)
        x = ops.convert_to_numpy(x)
        self.assertAllEqual(x, (1, 1))
        self.assertIsInstance(x, np.ndarray)

        # Partially converted.
        x = ops.convert_to_tensor((1, ops.array(2), 3))
        self.assertAllEqual(x, (1, 2, 3))

        with self.assertRaises(ValueError):
            ops.convert_to_numpy(KerasTensor((2,)))

    def test_cond(self):
        t = ops.cond(True, lambda: 0, lambda: 1)
        self.assertEqual(t, 0)
        f = ops.cond(False, lambda: 0, lambda: 1)
        self.assertEqual(f, 1)

        for val in [True, False]:
            out = ops.cond(
                val,
                lambda: KerasTensor((16, 3)),
                lambda: KerasTensor((16, 3)),
            )
            self.assertEqual((16, 3), out.shape)

        out = ops.cond(
            KerasTensor((), dtype="bool"),
            lambda: ops.ones((1, 3)),
            lambda: ops.zeros((1, 3)),
        )
        self.assertEqual((1, 3), out.shape)

        out = ops.cond(
            KerasTensor((), dtype="bool"),
            lambda: KerasTensor((3,)),
            lambda: KerasTensor((3,)),
        )
        self.assertEqual((3,), out.shape)

        with self.assertRaises(ValueError):
            ops.cond(
                KerasTensor((), dtype="bool"),
                lambda: KerasTensor((3,)),
                lambda: KerasTensor((4,)),
            )
