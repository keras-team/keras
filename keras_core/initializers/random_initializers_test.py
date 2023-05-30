import numpy as np

from keras_core import backend
from keras_core import initializers
from keras_core import testing
from keras_core import utils


class InitializersTest(testing.TestCase):
    def test_random_normal(self):
        utils.set_random_seed(1337)
        shape = (25, 20)
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
        self.assertAllClose(np.std(values), stddev, atol=1e-1)

        self.run_class_serialization_test(initializer)

        # Test serialization with SeedGenerator
        initializer = initializers.RandomNormal(
            mean=mean, stddev=stddev, seed=backend.random.SeedGenerator(1337)
        )
        values = initializer(shape=shape)

        # Test that unseeded generator gets different results after cloning
        initializer = initializers.RandomNormal(
            mean=mean, stddev=stddev, seed=None
        )
        values = initializer(shape=shape)
        cloned_initializer = initializers.RandomNormal.from_config(
            initializer.get_config()
        )
        new_values = cloned_initializer(shape=shape)
        self.assertNotAllClose(values, new_values)

        # Test that seeded generator gets same results after cloning
        initializer = initializers.RandomNormal(
            mean=mean, stddev=stddev, seed=1337
        )
        values = initializer(shape=shape)
        cloned_initializer = initializers.RandomNormal.from_config(
            initializer.get_config()
        )
        new_values = cloned_initializer(shape=shape)
        self.assertAllClose(values, new_values)

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

    def test_variance_scaling(self):
        utils.set_random_seed(1337)
        shape = (25, 20)
        scale = 2.0
        seed = 1234
        initializer = initializers.VarianceScaling(
            scale=scale, seed=seed, mode="fan_in"
        )
        values = initializer(shape=shape)
        self.assertEqual(initializer.scale, scale)
        self.assertEqual(initializer.seed, seed)
        self.assertEqual(values.shape, shape)
        self.assertAllClose(np.std(values), np.sqrt(scale / 25), atol=1e-1)
        self.run_class_serialization_test(initializer)

        initializer = initializers.VarianceScaling(
            scale=scale, seed=seed, mode="fan_out"
        )
        values = initializer(shape=shape)
        self.assertEqual(initializer.scale, scale)
        self.assertEqual(initializer.seed, seed)
        self.assertEqual(values.shape, shape)
        self.assertAllClose(np.std(values), np.sqrt(scale / 20), atol=1e-1)
        self.run_class_serialization_test(initializer)

    def test_orthogonal_initializer(self):
        shape = (5, 5)
        gain = 2.0
        seed = 1234
        initializer = initializers.OrthogonalInitializer(gain=gain, seed=seed)
        _ = initializer(shape=shape)
        # TODO: test correctness

        self.run_class_serialization_test(initializer)

    def test_get_method(self):
        obj = initializers.get("glorot_normal")
        self.assertTrue(obj, initializers.GlorotNormal)

        obj = initializers.get(None)
        self.assertEqual(obj, None)

        with self.assertRaises(ValueError):
            initializers.get("typo")
