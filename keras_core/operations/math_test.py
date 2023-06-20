import numpy as np
import pytest
import tensorflow as tf

from keras_core import backend
from keras_core import testing
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.operations import math as kmath


class MathOpsDynamicShapeTest(testing.TestCase):
    def test_segment_sum(self):
        data = KerasTensor((None, 4), dtype="float32")
        segment_ids = KerasTensor((10,), dtype="int32")
        outputs = kmath.segment_sum(data, segment_ids)
        self.assertEqual(outputs.shape, (None, 4))

        data = KerasTensor((None, 4), dtype="float32")
        segment_ids = KerasTensor((10,), dtype="int32")
        outputs = kmath.segment_sum(data, segment_ids, num_segments=5)
        self.assertEqual(outputs.shape, (5, 4))

    def test_top_k(self):
        x = KerasTensor((None, 2, 3))
        values, indices = kmath.top_k(x, k=1)
        self.assertEqual(values.shape, (None, 2, 1))
        self.assertEqual(indices.shape, (None, 2, 1))

    def test_in_top_k(self):
        targets = KerasTensor((None,))
        predictions = KerasTensor((None, 10))
        self.assertEqual(
            kmath.in_top_k(targets, predictions, k=1).shape, (None,)
        )

    def test_logsumexp(self):
        x = KerasTensor((None, 2, 3), dtype="float32")
        result = kmath.logsumexp(x)
        self.assertEqual(result.shape, ())

    def test_qr(self):
        x = KerasTensor((None, 4, 3), dtype="float32")
        q, r = kmath.qr(x, mode="reduced")
        qref, rref = np.linalg.qr(np.ones((2, 4, 3)), mode="reduced")
        qref_shape = (None,) + qref.shape[1:]
        rref_shape = (None,) + rref.shape[1:]
        self.assertEqual(q.shape, qref_shape)
        self.assertEqual(r.shape, rref_shape)

        q, r = kmath.qr(x, mode="complete")
        qref, rref = np.linalg.qr(np.ones((2, 4, 3)), mode="complete")
        qref_shape = (None,) + qref.shape[1:]
        rref_shape = (None,) + rref.shape[1:]
        self.assertEqual(q.shape, qref_shape)
        self.assertEqual(r.shape, rref_shape)


class MathOpsStaticShapeTest(testing.TestCase):
    @pytest.mark.skipif(
        backend.backend() == "jax",
        reason="JAX does not support `num_segments=None`.",
    )
    def test_segment_sum(self):
        data = KerasTensor((10, 4), dtype="float32")
        segment_ids = KerasTensor((10,), dtype="int32")
        outputs = kmath.segment_sum(data, segment_ids)
        self.assertEqual(outputs.shape, (None, 4))

    def test_segment_sum_explicit_num_segments(self):
        data = KerasTensor((10, 4), dtype="float32")
        segment_ids = KerasTensor((10,), dtype="int32")
        outputs = kmath.segment_sum(data, segment_ids, num_segments=5)
        self.assertEqual(outputs.shape, (5, 4))

    def test_topk(self):
        x = KerasTensor((1, 2, 3))
        values, indices = kmath.top_k(x, k=1)
        self.assertEqual(values.shape, (1, 2, 1))
        self.assertEqual(indices.shape, (1, 2, 1))

    def test_in_top_k(self):
        targets = KerasTensor((5,))
        predictions = KerasTensor((5, 10))
        self.assertEqual(kmath.in_top_k(targets, predictions, k=1).shape, (5,))

    def test_logsumexp(self):
        x = KerasTensor((1, 2, 3), dtype="float32")
        result = kmath.logsumexp(x)
        self.assertEqual(result.shape, ())

    def test_qr(self):
        x = KerasTensor((4, 3), dtype="float32")
        q, r = kmath.qr(x, mode="reduced")
        qref, rref = np.linalg.qr(np.ones((4, 3)), mode="reduced")
        self.assertEqual(q.shape, qref.shape)
        self.assertEqual(r.shape, rref.shape)

        q, r = kmath.qr(x, mode="complete")
        qref, rref = np.linalg.qr(np.ones((4, 3)), mode="complete")
        self.assertEqual(q.shape, qref.shape)
        self.assertEqual(r.shape, rref.shape)


class MathOpsCorrectnessTest(testing.TestCase):
    @pytest.mark.skipif(
        backend.backend() == "jax",
        reason="JAX does not support `num_segments=None`.",
    )
    def test_segment_sum(self):
        # Test 1D case.
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        segment_ids = np.array([0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        outputs = kmath.segment_sum(data, segment_ids)
        expected = tf.math.segment_sum(data, segment_ids)
        self.assertAllClose(outputs, expected)

        # Test N-D case.
        data = np.random.rand(9, 3, 3)
        segment_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        outputs = kmath.segment_sum(data, segment_ids)
        expected = tf.math.segment_sum(data, segment_ids)
        self.assertAllClose(outputs, expected)

    def test_segment_sum_explicit_num_segments(self):
        # Test 1D case.
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        segment_ids = np.array([0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        outputs = kmath.segment_sum(data, segment_ids, num_segments=4)
        expected = tf.math.unsorted_segment_sum(
            data, segment_ids, num_segments=4
        )
        self.assertAllClose(outputs, expected)

        # Test 1D with -1 case.
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        segment_ids = np.array([0, 0, 1, 1, -1, 2, 2, -1], dtype=np.int32)
        outputs = kmath.segment_sum(data, segment_ids, num_segments=4)
        expected = tf.math.unsorted_segment_sum(
            data, segment_ids, num_segments=4
        )
        self.assertAllClose(outputs, expected)

        # Test N-D case.
        data = np.random.rand(9, 3, 3)
        segment_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        outputs = kmath.segment_sum(data, segment_ids, num_segments=4)
        expected = tf.math.unsorted_segment_sum(
            data, segment_ids, num_segments=4
        )
        self.assertAllClose(outputs, expected)

    def test_top_k(self):
        x = np.array([0, 4, 2, 1, 3, -1], dtype=np.float32)
        values, indices = kmath.top_k(x, k=2)
        self.assertAllClose(values, [4, 3])
        self.assertAllClose(indices, [1, 4])

        x = np.array([0, 4, 2, 1, 3, -1], dtype=np.float32)
        values, indices = kmath.top_k(x, k=2, sorted=False)
        # Any order ok when `sorted=False`.
        self.assertEqual(set(backend.convert_to_numpy(values)), set([4, 3]))
        self.assertEqual(set(backend.convert_to_numpy(indices)), set([1, 4]))

        x = np.random.rand(5, 5)
        outputs = kmath.top_k(x, k=2)
        expected = tf.math.top_k(x, k=2)
        self.assertAllClose(outputs[0], expected[0])
        self.assertAllClose(outputs[1], expected[1])

    def test_in_top_k(self):
        targets = np.array([1, 0, 2])
        predictions = np.array(
            [
                [0.1, 0.9, 0.8, 0.8],
                [0.05, 0.95, 0, 1],
                [0.1, 0.8, 0.3, 1],
            ]
        )
        self.assertAllEqual(
            kmath.in_top_k(targets, predictions, k=1), [True, False, False]
        )
        self.assertAllEqual(
            kmath.in_top_k(targets, predictions, k=2), [True, False, False]
        )
        self.assertAllEqual(
            kmath.in_top_k(targets, predictions, k=3), [True, True, True]
        )

        # Test tie cases.
        targets = np.array([1, 0, 2])
        predictions = np.array(
            [
                [0.1, 0.9, 0.8, 0.8],
                [0.95, 0.95, 0, 0.95],
                [0.1, 0.8, 0.8, 0.95],
            ]
        )
        self.assertAllEqual(
            kmath.in_top_k(targets, predictions, k=1), [True, True, False]
        )
        self.assertAllEqual(
            kmath.in_top_k(targets, predictions, k=2), [True, True, True]
        )
        self.assertAllEqual(
            kmath.in_top_k(targets, predictions, k=3), [True, True, True]
        )

    def test_logsumexp(self):
        x = np.random.rand(5, 5)
        outputs = kmath.logsumexp(x)
        expected = np.log(np.sum(np.exp(x)))
        self.assertAllClose(outputs, expected)

        outputs = kmath.logsumexp(x, axis=1)
        expected = np.log(np.sum(np.exp(x), axis=1))
        self.assertAllClose(outputs, expected)

    def test_qr(self):
        x = np.random.random((4, 5))
        q, r = kmath.qr(x, mode="reduced")
        qref, rref = np.linalg.qr(x, mode="reduced")
        self.assertAllClose(qref, q)
        self.assertAllClose(rref, r)

        q, r = kmath.qr(x, mode="complete")
        qref, rref = np.linalg.qr(x, mode="complete")
        self.assertAllClose(qref, q)
        self.assertAllClose(rref, r)
