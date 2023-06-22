import numpy as np
import pytest
from absl.testing import parameterized

import keras_core
from keras_core import backend
from keras_core import testing
from keras_core.operations import numpy as knp
from keras_core.random import random


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
        self.assertLessEqual(knp.max(res), maxval)
        self.assertGreaterEqual(knp.max(res), minval)

    @parameterized.parameters(
        {"seed": 10, "num_samples": 1, "batch_size": 1},
        {"seed": 10, "num_samples": 5, "batch_size": 2},
        {"seed": 10, "num_samples": 10, "batch_size": 4},
        {"seed": 10, "num_samples": 15, "batch_size": 8},
    )
    def test_categorical(self, seed, num_samples, batch_size):
        np.random.seed(seed)
        # Definitively favor the batch index.
        logits = np.eye(batch_size) * 1e9
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
        self.assertLessEqual(knp.max(res), max)
        self.assertGreaterEqual(knp.max(res), min)
        # Torch has incomplete dtype support for uints; will remap some dtypes.
        if keras_core.backend.backend() != "torch":
            self.assertEqual(backend.standardize_dtype(res.dtype), dtype)

    @parameterized.parameters(
        {"seed": 10, "shape": (5,), "mean": 0, "stddev": 1},
        {"seed": 10, "shape": (2, 3), "mean": 0, "stddev": 1},
        {"seed": 10, "shape": (2, 3, 4), "mean": 0, "stddev": 1},
        {"seed": 10, "shape": (2, 3), "mean": 10, "stddev": 1},
        {"seed": 10, "shape": (2, 3), "mean": 10, "stddev": 3},
    )
    def test_truncated_normal(self, seed, shape, mean, stddev):
        np.random.seed(seed)
        np_res = np.random.normal(loc=mean, scale=stddev, size=shape)
        res = random.truncated_normal(
            shape, mean=mean, stddev=stddev, seed=seed
        )
        self.assertEqual(res.shape, shape)
        self.assertEqual(res.shape, np_res.shape)
        self.assertLessEqual(knp.max(res), mean + 2 * stddev)
        self.assertGreaterEqual(knp.max(res), mean - 2 * stddev)

    def test_dropout(self):
        x = knp.ones((3, 5))
        self.assertAllClose(random.dropout(x, rate=0, seed=0), x)
        x_res = random.dropout(x, rate=0.8, seed=0)
        self.assertGreater(knp.max(x_res), knp.max(x))
        self.assertGreater(knp.sum(x_res == 0), 2)

    @pytest.mark.skipif(
        keras_core.backend.backend() != "jax",
        reason="This test requires `jax` as the backend.",
    )
    def test_dropout_jax_jit_stateless(self):
        import jax
        import jax.numpy as jnp

        x = knp.ones(3)

        @jax.jit
        def train_step(x):
            with keras_core.backend.StatelessScope():
                x = keras_core.layers.Dropout(rate=0.1)(x, training=True)
            return x

        keras_core.utils.traceback_utils.disable_traceback_filtering()
        x = train_step(x)
        assert isinstance(x, jnp.ndarray)

    def test_dropout_noise_shape(self):
        inputs = knp.ones((2, 3, 5, 7))
        x = random.dropout(
            inputs, rate=0.3, noise_shape=[None, 3, 5, None], seed=0
        )
        self.assertEqual(x.shape, (2, 3, 5, 7))
