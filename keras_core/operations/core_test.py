import numpy as np

from keras_core import testing
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.operations import core


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

    def test_block_update(self):
        inputs = KerasTensor((4, 4))
        start_indices = KerasTensor((2,))
        updates = KerasTensor((2, 2))
        self.assertEqual(
            core.block_update(inputs, start_indices, updates).shape, (4, 4)
        )

        inputs = KerasTensor((4, 4, 4))
        start_indices = KerasTensor((3,))
        updates = KerasTensor((2, 2, 2))
        self.assertEqual(
            core.block_update(inputs, start_indices, updates).shape, (4, 4, 4)
        )


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

    def test_block_update(self):
        # Test 1D.
        inputs = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        start_indices = [1]
        updates = np.array([9, 10, 11, 12])
        self.assertAllClose(
            core.block_update(inputs, start_indices, updates),
            [0, 9, 10, 11, 12, 0, 0, 0],
        )

        # Test 2D.
        inputs = np.array([[1, 1], [1, 1], [1, 1]])
        start_indices = [1, 0]
        updates = np.array([[2, 2], [2, 2]])
        self.assertAllClose(
            core.block_update(inputs, start_indices, updates),
            [[1, 1], [2, 2], [2, 2]],
        )

        # Test N-D.
        inputs = np.ones([4, 4, 4, 4])
        start_indices = [1, 1, 2, 2]
        updates = np.zeros([2, 2, 2, 2])
        outputs = core.block_update(inputs, start_indices, updates)
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
