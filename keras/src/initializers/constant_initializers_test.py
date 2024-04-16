import numpy as np

from keras.src import backend
from keras.src import initializers
from keras.src import testing


class ConstantInitializersTest(testing.TestCase):
    def test_zeros_initializer(self):
        shape = (3, 3)

        initializer = initializers.Zeros()
        values = initializer(shape=shape)
        self.assertEqual(values.shape, shape)
        np_values = backend.convert_to_numpy(values)
        self.assertAllClose(np_values, np.zeros(shape=shape))

        self.run_class_serialization_test(initializer)

    def test_ones_initializer(self):
        shape = (3, 3)

        initializer = initializers.Ones()
        values = initializer(shape=shape)
        self.assertEqual(values.shape, shape)
        np_values = backend.convert_to_numpy(values)
        self.assertAllClose(np_values, np.ones(shape=shape))

        self.run_class_serialization_test(initializer)

    def test_constant_initializer(self):
        shape = (3, 3)
        constant_value = 6.0

        initializer = initializers.Constant(value=constant_value)
        values = initializer(shape=shape)
        self.assertEqual(values.shape, shape)
        np_values = backend.convert_to_numpy(values)
        self.assertAllClose(
            np_values, np.full(shape=shape, fill_value=constant_value)
        )

        self.run_class_serialization_test(initializer)

    def test_constant_initializer_array_value(self):
        shape = (3, 3)
        constant_value = np.random.random((3, 3))

        initializer = initializers.Constant(value=constant_value)
        values = initializer(shape=shape)
        self.assertEqual(values.shape, shape)
        np_values = backend.convert_to_numpy(values)
        self.assertAllClose(
            np_values, np.full(shape=shape, fill_value=constant_value)
        )

        self.run_class_serialization_test(initializer)

    def test_identity_initializer(self):
        shape = (3, 3)
        gain = 2

        initializer = initializers.Identity(gain=gain)
        values = initializer(shape=shape)
        self.assertEqual(values.shape, shape)
        np_values = backend.convert_to_numpy(values)
        self.assertAllClose(np_values, np.eye(*shape) * gain)

        self.run_class_serialization_test(initializer)
