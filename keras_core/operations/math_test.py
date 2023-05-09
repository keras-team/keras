import numpy as np
import pytest

from keras_core import backend
from keras_core import testing
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.operations import math as kmath


@pytest.mark.skipif(
    not backend.DYNAMIC_SHAPES_OK,
    reason="Backend does not support dynamic shapes",
)
class MathOpsDynamicShapeTest(testing.TestCase):
    def test_topk(self):
        x = KerasTensor([None, 2, 3])
        values, indices = kmath.top_k(x, k=1)
        self.assertEqual(values.shape, (None, 2, 1))
        self.assertEqual(indices.shape, (None, 2, 1))

    def test_logsumexp(self):
        x = KerasTensor([None, 2, 3], dtype="float32")
        result = kmath.logsumexp(x)
        self.assertEqual(result.shape, ())


class MathOpsStaticShapeTest(testing.TestCase):
    def test_topk(self):
        x = KerasTensor([1, 2, 3])
        values, indices = kmath.top_k(x, k=1)
        self.assertEqual(values.shape, (1, 2, 1))
        self.assertEqual(indices.shape, (1, 2, 1))

    def test_logsumexp(self):
        x = KerasTensor([1, 2, 3], dtype="float32")
        result = kmath.logsumexp(x)
        self.assertEqual(result.shape, ())


class MathOpsCorrectnessTest(testing.TestCase):
    def test_topk(self):
        x = np.array([0, 4, 2, 1, 3, -1], dtype=np.float32)
        values, indices = kmath.top_k(x, k=2)
        self.assertAllClose(values, [4, 3])
        self.assertAllClose(indices, [1, 4])

    def test_logsumexp(self):
        x_np = np.random.rand(5, 5)
        y_tf_np = kmath.logsumexp(x_np)
        y_np = np.log(np.sum(np.exp(x_np)))
        self.assertAllClose(y_tf_np, y_np)
