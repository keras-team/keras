import numpy as np
import pytest
from absl.testing import parameterized

import keras_core
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
