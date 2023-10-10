import numpy as np
import pytest
from absl.testing import parameterized

import keras
from keras import backend
from keras import ops
from keras import testing
from keras.random import random
from keras.random import seed_generator


class RandomTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        {"seed": 10, "shape": (5,), "mean": 0, "stddev": 1},
        {"seed": 10, "shape": (2, 3), "mean": 0, "stddev": 1},
        {"seed": 10, "shape": (2, 3, 4), "mean": 0, "stddev": 1},
        {"seed": 10, "shape": (2, 3), "mean": 10, "stddev": 1},
        {"seed": 10, "shape": (2, 3), "mean": 10, "stddev": 3},
    )
    def test_normal(self, seed, shape, mean, stddev):
        np.random.seed(seed)
        np_res = np.random.normal(loc=mean, scale=stddev, size=shape)
        res = random.normal(shape, mean=mean, stddev=stddev, seed=seed)
        self.assertEqual(res.shape, shape)
        self.assertEqual(res.shape, np_res.shape)

    @parameterized.parameters(
        {"seed": 10, "shape": (5,), "minval": 0, "maxval": 1},
        {"seed": 10, "shape": (2, 3), "minval": 0, "maxval": 1},
        {"seed": 10, "shape": (2, 3, 4), "minval": 0, "maxval": 2},
        {"seed": 10, "shape": (2, 3), "minval": -1, "maxval": 1},
        {"seed": 10, "shape": (2, 3), "minval": 1, "maxval": 3},
    )
    def test_uniform(self, seed, shape, minval, maxval):
        np.random.seed(seed)
        np_res = np.random.uniform(low=minval, high=maxval, size=shape)
        res = random.uniform(shape, minval=minval, maxval=maxval, seed=seed)
        self.assertEqual(res.shape, shape)
        self.assertEqual(res.shape, np_res.shape)
        self.assertLessEqual(ops.max(res), maxval)
        self.assertGreaterEqual(ops.max(res), minval)

    @parameterized.parameters(
        {"seed": 10, "num_samples": 1, "batch_size": 1},
        {"seed": 10, "num_samples": 5, "batch_size": 2},
        {"seed": 10, "num_samples": 10, "batch_size": 4},
        {"seed": 10, "num_samples": 15, "batch_size": 8},
    )
    def test_categorical(self, seed, num_samples, batch_size):
        np.random.seed(seed)
        # Create logits that definitely favors the batch index after a softmax
        # is applied. Without a softmax, this would be close to random.
        logits = np.eye(batch_size) * 1e5 + 1e6
        res = random.categorical(logits, num_samples, seed=seed)
        # Outputs should have shape `(batch_size, num_samples)`, where each
        # output index matches the batch index.
        self.assertEqual(res.shape, (batch_size, num_samples))
        expected = np.tile(np.arange(batch_size)[:, None], (1, num_samples))
        self.assertAllClose(res, expected)

    def test_categorical_errors(self):
        with self.assertRaises(ValueError):
            random.categorical(np.ones((5,)), 5)
        with self.assertRaises(ValueError):
            random.categorical(np.ones((5, 5, 5)), 5)

    @parameterized.parameters(
        {"seed": 10, "shape": (5,), "min": 0, "max": 10, "dtype": "uint16"},
        {"seed": 10, "shape": (2, 3), "min": 0, "max": 10, "dtype": "uint32"},
        {"seed": 10, "shape": (2, 3, 4), "min": 0, "max": 2, "dtype": "int8"},
        {"seed": 10, "shape": (2, 3), "min": -1, "max": 1, "dtype": "int16"},
        {"seed": 10, "shape": (2, 3), "min": 1, "max": 3, "dtype": "int32"},
    )
    def test_randint(self, seed, shape, min, max, dtype):
        np.random.seed(seed)
        np_res = np.random.randint(low=min, high=max, size=shape)
        res = random.randint(
            shape, minval=min, maxval=max, seed=seed, dtype=dtype
        )
        self.assertEqual(res.shape, shape)
        self.assertEqual(res.shape, np_res.shape)
        self.assertLessEqual(ops.max(res), max)
        self.assertGreaterEqual(ops.max(res), min)
        # Torch has incomplete dtype support for uints; will remap some dtypes.
        if keras.backend.backend() != "torch":
            self.assertEqual(backend.standardize_dtype(res.dtype), dtype)

    @parameterized.parameters(
        {"seed": 10, "shape": (5,), "mean": 0, "stddev": 1},
        {"seed": 10, "shape": (2, 3), "mean": 0, "stddev": 1},
        {"seed": 10, "shape": (2, 3, 4), "mean": 0, "stddev": 1},
        {"seed": 10, "shape": (2, 3), "mean": 10, "stddev": 1},
        {"seed": 10, "shape": (2, 3), "mean": 10, "stddev": 3},
        # Test list shapes.
        {"seed": 10, "shape": [2, 3], "mean": 10, "stddev": 3},
    )
    def test_truncated_normal(self, seed, shape, mean, stddev):
        np.random.seed(seed)
        np_res = np.random.normal(loc=mean, scale=stddev, size=shape)
        res = random.truncated_normal(
            shape, mean=mean, stddev=stddev, seed=seed
        )
        self.assertEqual(res.shape, tuple(shape))
        self.assertEqual(res.shape, np_res.shape)
        self.assertLessEqual(ops.max(res), mean + 2 * stddev)
        self.assertGreaterEqual(ops.max(res), mean - 2 * stddev)

    def test_dropout(self):
        x = ops.ones((3, 5))
        self.assertAllClose(random.dropout(x, rate=0, seed=0), x)
        x_res = random.dropout(x, rate=0.8, seed=0)
        self.assertGreater(ops.max(x_res), ops.max(x))
        self.assertGreater(ops.sum(x_res == 0), 2)

    @pytest.mark.skipif(
        keras.backend.backend() != "jax",
        reason="This test requires `jax` as the backend.",
    )
    def test_dropout_jax_jit_stateless(self):
        import jax
        import jax.numpy as jnp

        x = ops.ones(3)

        @jax.jit
        def train_step(x):
            with keras.backend.StatelessScope():
                x = keras.layers.Dropout(rate=0.1)(x, training=True)
            return x

        x = train_step(x)
        self.assertIsInstance(x, jnp.ndarray)

    def test_dropout_noise_shape(self):
        inputs = ops.ones((2, 3, 5, 7))
        x = random.dropout(
            inputs, rate=0.3, noise_shape=[None, 3, 5, None], seed=0
        )
        self.assertEqual(x.shape, (2, 3, 5, 7))

    @pytest.mark.skipif(
        keras.backend.backend() != "jax",
        reason="This test requires `jax` as the backend.",
    )
    def test_jax_rngkey_seed(self):
        import jax
        import jax.numpy as jnp

        seed = 1234
        rng = jax.random.PRNGKey(seed)
        self.assertEqual(rng.shape, (2,))
        self.assertEqual(rng.dtype, jnp.uint32)
        x = random.randint((3, 5), 0, 10, seed=rng)
        self.assertIsInstance(x, jnp.ndarray)

    @pytest.mark.skipif(
        keras.backend.backend() != "jax",
        reason="This test requires `jax` as the backend.",
    )
    def test_jax_unseed_disallowed_during_tracing(self):
        import jax

        @jax.jit
        def jit_fn():
            return random.randint((2, 2), 0, 10, seed=None)

        with self.assertRaisesRegex(
            ValueError, "you should only use seeded random ops"
        ):
            jit_fn()

    def test_global_seed_generator(self):
        # Check that unseeded RNG calls use and update global_rng_state()

        def random_numbers(seed):
            rng_state = seed_generator.global_seed_generator().state
            rng_state.assign(seed)
            x = random.normal((), seed=None)
            y = random.normal((), seed=None)
            return x, y, rng_state.value

        if backend.backend() == "tensorflow":
            import tensorflow as tf

            random_numbers = tf.function(jit_compile=True)(random_numbers)

        seed = ops.zeros((2,))
        seed0 = ops.convert_to_numpy(seed)
        x1, y1, seed = random_numbers(seed)
        x1 = ops.convert_to_numpy(x1)
        y1 = ops.convert_to_numpy(y1)
        seed1 = ops.convert_to_numpy(seed)
        x2, y2, seed = random_numbers(seed)
        x2 = ops.convert_to_numpy(x2)
        y2 = ops.convert_to_numpy(y2)
        seed2 = ops.convert_to_numpy(seed)
        x3, y3, seed = random_numbers(seed)
        x3 = ops.convert_to_numpy(x3)
        y3 = ops.convert_to_numpy(y3)
        seed3 = ops.convert_to_numpy(seed)

        self.assertNotEqual(seed0[1], seed1[1])
        self.assertNotEqual(seed1[1], seed2[1])
        self.assertNotEqual(seed2[1], seed3[1])

        self.assertGreater(np.abs(x1 - y1), 1e-4)
        self.assertGreater(np.abs(x1 - y1), 1e-4)
        self.assertGreater(np.abs(x2 - y2), 1e-4)
        self.assertGreater(np.abs(x3 - y3), 1e-4)
        self.assertGreater(np.abs(x1 - x2), 1e-4)
        self.assertGreater(np.abs(x1 - x3), 1e-4)
        self.assertGreater(np.abs(x2 - x3), 1e-4)
        self.assertGreater(np.abs(y1 - y2), 1e-4)
        self.assertGreater(np.abs(y1 - y3), 1e-4)
        self.assertGreater(np.abs(y2 - y3), 1e-4)

        seed_generator.global_seed_generator().state.assign(seed)

    def test_shuffle(self):
        x = np.arange(100).reshape(10, 10)

        # Test axis=0
        y = random.shuffle(x, seed=0)

        self.assertFalse(np.all(x == ops.convert_to_numpy(y)))
        self.assertAllClose(np.sum(x, axis=0), ops.sum(y, axis=0))
        self.assertNotAllClose(np.sum(x, axis=1), ops.sum(y, axis=1))

        # Test axis=1
        y = random.shuffle(x, axis=1, seed=0)

        self.assertFalse(np.all(x == ops.convert_to_numpy(y)))
        self.assertAllClose(np.sum(x, axis=1), ops.sum(y, axis=1))
        self.assertNotAllClose(np.sum(x, axis=0), ops.sum(y, axis=0))

    def test_randint_dtype_validation(self):
        with self.assertRaisesRegex(
            ValueError, "`keras.random.randint` requires an integer `dtype`."
        ):
            random.randint((3, 4), minval=0, maxval=10, dtype="float64")
