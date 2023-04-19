import numpy as np

from keras_core import constraints
from keras_core import testing


def get_example_array():
    np.random.seed(3537)
    example_array = np.random.random((100, 100)) * 100.0 - 50.0
    example_array[0, 0] = 0.0  # Possible edge case
    return example_array


class ConstraintsTest(testing.TestCase):
    def test_max_norm(self):
        constraint_fn = constraints.MaxNorm(2.0)
        x = np.array([[0, 0, 0], [1.0, 0, 0], [3, 0, 0], [3, 3, 3]]).T
        target = np.array(
            [
                [0, 0, 0],
                [1.0, 0, 0],
                [2.0, 0, 0],
                [2.0 / np.sqrt(3), 2.0 / np.sqrt(3), 2.0 / np.sqrt(3)],
            ]
        ).T
        output = constraint_fn(x)
        self.assertAllClose(target, output)

    def test_non_neg(self):
        constraint_fn = constraints.NonNeg()
        output = constraint_fn(get_example_array())
        output = np.array(output)
        self.assertTrue((np.min(output, axis=1) >= 0.0).all())

    def test_unit_norm(self):
        constraint_fn = constraints.UnitNorm()
        output = constraint_fn(get_example_array())
        l2 = np.sqrt(np.sum(np.square(output), axis=0))
        self.assertAllClose(l2, 1.0)

    def test_min_max_norm(self):
        constraint_fn = constraints.MinMaxNorm(min_value=0.2, max_value=0.5)
        output = constraint_fn(get_example_array())
        l2 = np.sqrt(np.sum(np.square(output), axis=0))
        self.assertFalse(l2[l2 < 0.2])
        self.assertFalse(l2[l2 > 0.5 + 1e-6])
