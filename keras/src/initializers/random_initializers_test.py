import numpy as np

from conftest import skip_if_backend
from keras.src import backend
from keras.src import initializers
from keras.src import random
from keras.src import testing
from keras.src import utils


class RandomInitializersTest(testing.TestCase):
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
        self.assertAllClose(
            np.std(backend.convert_to_numpy(values)), stddev, atol=1e-1
        )

        self.run_class_serialization_test(initializer)

        # Test that a fixed seed yields the same results each call.
        initializer = initializers.RandomNormal(
            mean=mean, stddev=stddev, seed=1337
        )
        values = initializer(shape=shape)
        next_values = initializer(shape=shape)
        self.assertAllClose(values, next_values)

        # Test that a SeedGenerator yields different results each call.
        initializer = initializers.RandomNormal(
            mean=mean, stddev=stddev, seed=backend.random.SeedGenerator(1337)
        )
        values = initializer(shape=shape)
        next_values = initializer(shape=shape)
        self.assertNotAllClose(values, next_values)

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
        values = backend.convert_to_numpy(values)
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
        self.assertAllClose(
            np.std(backend.convert_to_numpy(values)),
            np.sqrt(scale / 25),
            atol=1e-1,
        )
        self.run_class_serialization_test(initializer)

        initializer = initializers.VarianceScaling(
            scale=scale, seed=seed, mode="fan_out"
        )
        values = initializer(shape=shape)
        self.assertEqual(initializer.scale, scale)
        self.assertEqual(initializer.seed, seed)
        self.assertEqual(values.shape, shape)
        self.assertAllClose(
            np.std(backend.convert_to_numpy(values)),
            np.sqrt(scale / 20),
            atol=1e-1,
        )
        self.run_class_serialization_test(initializer)

    @skip_if_backend("openvino", "openvino backend does not support `qr`")
    def test_orthogonal(self):
        shape = (5, 5)
        gain = 2.0
        seed = 1234
        initializer = initializers.Orthogonal(gain=gain, seed=seed)
        values = initializer(shape=shape)
        self.assertEqual(initializer.seed, seed)
        self.assertEqual(initializer.gain, gain)

        self.assertEqual(values.shape, shape)
        array = backend.convert_to_numpy(values)
        # Making sure that the columns have gain * unit norm value
        for column in array.T:
            self.assertAlmostEqual(np.linalg.norm(column), gain * 1.0)

        # Making sure that each column is orthonormal to the other column
        for i in range(array.shape[-1]):
            for j in range(i + 1, array.shape[-1]):
                self.assertAlmostEqual(
                    np.dot(array[..., i], array[..., j]), 0.0
                )

        self.run_class_serialization_test(initializer)

        # Test compatible class_name
        initializer = initializers.get("OrthogonalInitializer")
        self.assertIsInstance(initializer, initializers.Orthogonal)

    def test_get_method(self):
        obj = initializers.get("glorot_normal")
        self.assertTrue(obj, initializers.GlorotNormal)

        obj = initializers.get(None)
        self.assertEqual(obj, None)

        with self.assertRaises(ValueError):
            initializers.get("typo")

    @skip_if_backend(
        "openvino", "openvino backend does not support `uniform` with None seed"
    )
    def test_get_method_with_tensor(self):
        shape = (5, 5)

        # Test backend tensor
        tensor = random.uniform(shape=shape)
        initializer = initializers.get(tensor)
        values = initializer(shape=shape)
        self.assertAllClose(values, tensor)

        # Test numpy array
        tensor = np.random.uniform(size=shape).astype("float32")
        initializer = initializers.get(tensor)
        values = initializer(shape=shape)
        self.assertAllClose(values, tensor)

        # Test bad `shape` argument
        with self.assertRaisesRegex(ValueError, r"Expected `shape` to be"):
            initializer(shape=(10, 10))

    def test_variance_scaling_invalid_scale(self):
        seed = 1234

        with self.assertRaisesRegex(
            ValueError, "Argument `scale` must be positive float."
        ):
            initializers.VarianceScaling(scale=-1.0, seed=seed, mode="fan_in")

    def test_variance_scaling_invalid_mode(self):
        scale = 2.0
        seed = 1234

        with self.assertRaisesRegex(ValueError, "Invalid `mode` argument:"):
            initializers.VarianceScaling(
                scale=scale, seed=seed, mode="invalid_mode"
            )

    def test_variance_scaling_invalid_distribution(self):
        scale = 2.0
        seed = 1234

        with self.assertRaisesRegex(
            ValueError, "Invalid `distribution` argument:"
        ):
            initializers.VarianceScaling(
                scale=scale,
                seed=seed,
                mode="fan_in",
                distribution="invalid_dist",
            )

    def test_serialization_with_seed_generator(self):
        seed = random.SeedGenerator()
        initializer = initializers.Orthogonal(seed=seed)
        self.run_class_serialization_test(initializer)

        seed = random.SeedGenerator()
        initializer = initializers.VarianceScaling(seed=seed)
        self.run_class_serialization_test(initializer)

        seed = random.SeedGenerator()
        initializer = initializers.RandomUniform(seed=seed)
        self.run_class_serialization_test(initializer)

        seed = random.SeedGenerator()
        initializer = initializers.RandomNormal(seed=seed)
        self.run_class_serialization_test(initializer)
