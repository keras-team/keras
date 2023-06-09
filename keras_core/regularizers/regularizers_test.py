import numpy as np

from keras_core import backend
from keras_core import regularizers
from keras_core import testing


# TODO: serialization tests
class RegularizersTest(testing.TestCase):
    def test_l1(self):
        value = np.random.random((4, 4))
        x = backend.Variable(value)
        y = regularizers.L1(0.1)(x)
        self.assertAllClose(y, 0.1 * np.sum(np.abs(value)))

    def test_l2(self):
        value = np.random.random((4, 4))
        x = backend.Variable(value)
        y = regularizers.L2(0.1)(x)
        self.assertAllClose(y, 0.1 * np.sum(np.square(value)))

    def test_l1_l2(self):
        value = np.random.random((4, 4))
        x = backend.Variable(value)
        y = regularizers.L1L2(l1=0.1, l2=0.2)(x)
        self.assertAllClose(
            y, 0.1 * np.sum(np.abs(value)) + 0.2 * np.sum(np.square(value))
        )

    def test_orthogonal_regularizer(self):
        value = np.random.random((4, 4))
        x = backend.Variable(value)
        regularizers.OrthogonalRegularizer(factor=0.1, mode="rows")(x)
        # TODO

    def test_get_method(self):
        obj = regularizers.get("l1l2")
        self.assertIsInstance(obj, regularizers.L1L2)

        obj = regularizers.get("l1")
        self.assertIsInstance(obj, regularizers.L1)

        obj = regularizers.get("l2")
        self.assertIsInstance(obj, regularizers.L2)

        obj = regularizers.get("orthogonal_regularizer")
        self.assertIsInstance(obj, regularizers.OrthogonalRegularizer)

        obj = regularizers.get(None)
        self.assertEqual(obj, None)

        with self.assertRaises(ValueError):
            regularizers.get("typo")
