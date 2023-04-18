import numpy as np

from keras_core import initializers
from keras_core import testing


class InitializersTest(testing.TestCase):
    def test_random_normal(self):
        shape = (5, 5)
        mean = 0.0
        stddev = 1.0
        seed = 1234
        external_config = {"mean": 1.0, "stddev": 0.5, "seed": 42}
        initializer = initializers.RandomNormal(
            mean=mean, stddev=stddev, seed=seed
        )
        values = initializer(shape=shape)
        self.assertEqual(initializer.mean, mean)
        self.assertEqual(initializer.stddev, stddev)
        self.assertEqual(initializer.seed, seed)
        self.assertEqual(values.shape, shape)
        self.assert_idempotent_config(initializer, external_config)

    def test_random_uniform(self):
        shape = (5, 5)
        minval = -1.0
        maxval = 1.0
        seed = 1234
        external_config = {"minval": 0.0, "maxval": 1.0, "seed": 42}
        initializer = initializers.RandomUniform(
            minval=minval, maxval=maxval, seed=seed
        )
        values = initializer(shape=shape)
        self.assertEqual(initializer.minval, minval)
        self.assertEqual(initializer.maxval, maxval)
        self.assertEqual(initializer.seed, seed)
        self.assertEqual(values.shape, shape)
        self.assert_idempotent_config(initializer, external_config)
        self.assertGreaterEqual(np.min(values), minval)
        self.assertLess(np.max(values), maxval)

    def assert_idempotent_config(self, initializer, config):
        initializer = initializer.from_config(config)
        self.assertEqual(initializer.get_config(), config)
