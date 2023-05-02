import numpy as np

from keras_core import initializers
from keras_core import testing


class InitializersTest(testing.TestCase):
    # TODO: missing many initializer tests.

    def test_random_normal(self):
        shape = (5, 5)
        mean = 0.0
        stddev = 1.0
        seed = 1234
        initializer = initializers.RandomNormal(
            mean=mean, stddev=stddev, seed=seed
        )
        values = initializer(shape=shape)
        self.assertEqual(initializer.mean, mean)
        self.assertEqual(initializer.stddev, stddev)
        self.assertEqual(initializer.seed, seed)
        self.assertEqual(values.shape, shape)

        self.run_class_serialization_test(initializer)

    def test_random_uniform(self):
        shape = (5, 5)
        minval = -1.0
        maxval = 1.0
        seed = 1234
        initializer = initializers.RandomUniform(
            minval=minval, maxval=maxval, seed=seed
        )
        values = initializer(shape=shape)
        self.assertEqual(initializer.minval, minval)
        self.assertEqual(initializer.maxval, maxval)
        self.assertEqual(initializer.seed, seed)
        self.assertEqual(values.shape, shape)
        self.assertGreaterEqual(np.min(values), minval)
        self.assertLess(np.max(values), maxval)

        self.run_class_serialization_test(initializer)

    def test_orthogonal_initializer(self):
        shape = (5, 5)
        gain = 2.0
        seed = 1234
        initializer = initializers.OrthogonalInitializer(gain=gain, seed=seed)
        _ = initializer(shape=shape)
        # TODO: test correctness

        self.run_class_serialization_test(initializer)
