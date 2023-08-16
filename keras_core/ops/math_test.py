import numpy as np
import pytest
import scipy.signal
from absl.testing import parameterized

from keras_core import backend
from keras_core import testing
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.ops import math as kmath


def _stft(
    x, sequence_length, sequence_stride, fft_length, window="hann", center=True
):
    # pure numpy version of stft
    if backend.standardize_dtype(x.dtype) not in {"float32", "float64"}:
        raise TypeError(
            "Invalid input type. Expected `float32` or `float64`. "
            f"Received: input type={x.dtype}"
        )
    if fft_length < sequence_length:
        raise ValueError(
            "`fft_length` must equal or larger than `sequence_length`. "
            f"Received: sequence_length={sequence_length}, "
            f"fft_length={fft_length}"
        )

    if center:
        pad_width = [(0, 0) for _ in range(len(x.shape))]
        pad_width[-1] = (fft_length // 2, fft_length // 2)
        if backend.backend() != "torch" or len(x.shape) < 3:
            x = np.pad(x, pad_width, mode="reflect")
        else:
            # torch not support reflect padding for N-D cases when N >= 3
            x = np.pad(x, pad_width, mode="constant")

    # extract_sequences
    *batch_shape, _ = x.shape
    batch_shape = list(batch_shape)
    shape = x.shape[:-1] + (
        (x.shape[-1] - (fft_length - sequence_stride)) // sequence_stride,
        fft_length,
    )
    strides = x.strides[:-1] + (sequence_stride * x.strides[-1], x.strides[-1])
    x = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    x = np.reshape(x, (*batch_shape, *x.shape[-2:]))

    if window is not None:
        if isinstance(window, str):
            window = scipy.signal.get_window(window, sequence_length)
        win = np.array(window, dtype=x.dtype)
        if len(win.shape) != 1 or win.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win.shape}"
            )
        l_pad = (fft_length - sequence_length) // 2
        r_pad = fft_length - sequence_length - l_pad
        win = np.pad(win, [[l_pad, r_pad]])
        x = np.multiply(x, win)

    x = np.fft.rfft(x, fft_length)
    return np.real(x), np.imag(x)


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

    def test_segment_max(self):
        data = KerasTensor((None, 4), dtype="float32")
        segment_ids = KerasTensor((10,), dtype="int32")
        outputs = kmath.segment_max(data, segment_ids)
        self.assertEqual(outputs.shape, (None, 4))

        data = KerasTensor((None, 4), dtype="float32")
        segment_ids = KerasTensor((10,), dtype="int32")
        outputs = kmath.segment_max(data, segment_ids, num_segments=5)
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

    def test_extract_sequences(self):
        # Defined dimension
        x = KerasTensor((None, 32), dtype="float32")
        sequence_length = 3
        sequence_stride = 2
        outputs = kmath.extract_sequences(x, sequence_length, sequence_stride)
        num_sequences = 1 + (x.shape[-1] - sequence_length) // sequence_stride
        self.assertEqual(outputs.shape, (None, num_sequences, sequence_length))

        # Undefined dimension
        x = KerasTensor((None, None), dtype="float32")
        sequence_length = 3
        sequence_stride = 2
        outputs = kmath.extract_sequences(x, sequence_length, sequence_stride)
        self.assertEqual(outputs.shape, (None, None, sequence_length))

    def test_fft(self):
        real = KerasTensor((None, 4, 3), dtype="float32")
        imag = KerasTensor((None, 4, 3), dtype="float32")
        real_output, imag_output = kmath.fft((real, imag))
        ref = np.fft.fft(np.ones((2, 4, 3)))
        ref_shape = (None,) + ref.shape[1:]
        self.assertEqual(real_output.shape, ref_shape)
        self.assertEqual(imag_output.shape, ref_shape)

    def test_fft2(self):
        real = KerasTensor((None, 4, 3), dtype="float32")
        imag = KerasTensor((None, 4, 3), dtype="float32")
        real_output, imag_output = kmath.fft2((real, imag))
        ref = np.fft.fft2(np.ones((2, 4, 3)))
        ref_shape = (None,) + ref.shape[1:]
        self.assertEqual(real_output.shape, ref_shape)
        self.assertEqual(imag_output.shape, ref_shape)

    def test_rfft(self):
        x = KerasTensor((None, 4, 3), dtype="float32")
        real_output, imag_output = kmath.rfft(x)
        ref = np.fft.rfft(np.ones((2, 4, 3)))
        ref_shape = (None,) + ref.shape[1:]
        self.assertEqual(real_output.shape, ref_shape)
        self.assertEqual(imag_output.shape, ref_shape)

    def test_stft(self):
        x = KerasTensor((None, 32), dtype="float32")
        sequence_length = 10
        sequence_stride = 3
        fft_length = 15
        real_output, imag_output = kmath.stft(
            x, sequence_length, sequence_stride, fft_length
        )
        real_ref, imag_ref = _stft(
            np.ones((2, 32)), sequence_length, sequence_stride, fft_length
        )
        real_ref_shape = (None,) + real_ref.shape[1:]
        imag_ref_shape = (None,) + imag_ref.shape[1:]
        self.assertEqual(real_output.shape, real_ref_shape)
        self.assertEqual(imag_output.shape, imag_ref_shape)

    def test_rsqrt(self):
        x = KerasTensor([None, 3])
        self.assertEqual(kmath.rsqrt(x).shape, (None, 3))


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

    @pytest.mark.skipif(
        backend.backend() == "jax",
        reason="JAX does not support `num_segments=None`.",
    )
    def test_segment_max(self):
        data = KerasTensor((10, 4), dtype="float32")
        segment_ids = KerasTensor((10,), dtype="int32")
        outputs = kmath.segment_max(data, segment_ids)
        self.assertEqual(outputs.shape, (None, 4))

    def test_segment_max_explicit_num_segments(self):
        data = KerasTensor((10, 4), dtype="float32")
        segment_ids = KerasTensor((10,), dtype="int32")
        outputs = kmath.segment_max(data, segment_ids, num_segments=5)
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

    def test_extract_sequences(self):
        x = KerasTensor((10, 16), dtype="float32")
        sequence_length = 3
        sequence_stride = 2
        outputs = kmath.extract_sequences(x, sequence_length, sequence_stride)
        num_sequences = 1 + (x.shape[-1] - sequence_length) // sequence_stride
        self.assertEqual(outputs.shape, (10, num_sequences, sequence_length))

    def test_fft(self):
        real = KerasTensor((2, 4, 3), dtype="float32")
        imag = KerasTensor((2, 4, 3), dtype="float32")
        real_output, imag_output = kmath.fft((real, imag))
        ref = np.fft.fft(np.ones((2, 4, 3)))
        self.assertEqual(real_output.shape, ref.shape)
        self.assertEqual(imag_output.shape, ref.shape)

    def test_fft2(self):
        real = KerasTensor((2, 4, 3), dtype="float32")
        imag = KerasTensor((2, 4, 3), dtype="float32")
        real_output, imag_output = kmath.fft2((real, imag))
        ref = np.fft.fft2(np.ones((2, 4, 3)))
        self.assertEqual(real_output.shape, ref.shape)
        self.assertEqual(imag_output.shape, ref.shape)

    def test_rfft(self):
        x = KerasTensor((2, 4, 3), dtype="float32")
        real_output, imag_output = kmath.rfft(x)
        ref = np.fft.rfft(np.ones((2, 4, 3)))
        self.assertEqual(real_output.shape, ref.shape)
        self.assertEqual(imag_output.shape, ref.shape)

    def test_rsqrt(self):
        x = KerasTensor([4, 3], dtype="float32")
        self.assertEqual(kmath.rsqrt(x).shape, (4, 3))

    def test_stft(self):
        x = KerasTensor((2, 32), dtype="float32")
        sequence_length = 10
        sequence_stride = 3
        fft_length = 15
        real_output, imag_output = kmath.stft(
            x, sequence_length, sequence_stride, fft_length
        )
        real_ref, imag_ref = _stft(
            np.ones((2, 32)), sequence_length, sequence_stride, fft_length
        )
        self.assertEqual(real_output.shape, real_ref.shape)
        self.assertEqual(imag_output.shape, imag_ref.shape)


class MathOpsCorrectnessTest(testing.TestCase, parameterized.TestCase):
    @pytest.mark.skipif(
        backend.backend() == "jax",
        reason="JAX does not support `num_segments=None`.",
    )
    def test_segment_sum(self):
        # Test 1D case.
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        segment_ids = np.array([0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        outputs = kmath.segment_sum(data, segment_ids)

        # Segment 0: 1 + 2 = 3
        # Segment 1: 3 + 4 + 5 = 12
        # Segment 2: 6 + 7 + 8 = 21
        expected = np.array([3, 12, 21], dtype=np.float32)
        self.assertAllClose(outputs, expected)

        # Test N-D case.
        data = np.random.rand(9, 3, 3)
        segment_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        outputs = kmath.segment_sum(data, segment_ids)

        expected = np.zeros((3, 3, 3))
        for i in range(data.shape[0]):
            segment_id = segment_ids[i]
            expected[segment_id] += data[i]

        self.assertAllClose(outputs, expected)

    def test_segment_sum_explicit_num_segments(self):
        # Test 1D case.
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        segment_ids = np.array([0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        outputs = kmath.segment_sum(data, segment_ids, num_segments=4)
        expected = np.array([3, 12, 21, 0], dtype=np.float32)
        self.assertAllClose(outputs, expected)

        # Test 1D with -1 case.
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        segment_ids = np.array([0, 0, 1, 1, -1, 2, 2, -1], dtype=np.int32)
        outputs = kmath.segment_sum(data, segment_ids, num_segments=4)

        # Segment ID 0: First two elements (1 + 2) = 3
        # Segment ID 1: Next two elements (3 + 4) = 7
        # Segment ID -1: Ignore the next two elements, because segment ID is -1.
        # Segment ID 2: Next two elements (6 + 7) = 13
        # Segment ID 3: No elements, so output is 0.
        expected = np.array([3, 7, 13, 0], dtype=np.float32)
        self.assertAllClose(outputs, expected)

        # Test N-D case.
        data = np.random.rand(9, 3, 3)
        segment_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        outputs = kmath.segment_sum(data, segment_ids, num_segments=4)

        expected = np.zeros((4, 3, 3))
        for i in range(data.shape[0]):
            segment_id = segment_ids[i]
            if segment_id != -1:
                expected[segment_id] += data[i]

        self.assertAllClose(outputs, expected)

    @pytest.mark.skipif(
        backend.backend() == "jax",
        reason="JAX does not support `num_segments=None`.",
    )
    def test_segment_max(self):
        # Test 1D case.
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        segment_ids = np.array([0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        outputs = kmath.segment_max(data, segment_ids)

        # Segment ID 0: Max of the first two elements = 2
        # Segment ID 1: Max of the next three elements = 5
        # Segment ID 2: Max of the next three elements = 8
        expected = np.array([2, 5, 8], dtype=np.float32)

        self.assertAllClose(outputs, expected)

        # Test N-D case.
        data = np.random.rand(9, 3, 3)
        segment_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        outputs = kmath.segment_max(data, segment_ids)
        expected = np.zeros((3, 3, 3))
        for i in range(data.shape[0]):
            segment_id = segment_ids[i]
            expected[segment_id] = np.maximum(expected[segment_id], data[i])

        self.assertAllClose(outputs, expected)

    def test_segment_max_explicit_num_segments(self):
        # Test 1D case.
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        segment_ids = np.array([0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        outputs = kmath.segment_max(data, segment_ids, num_segments=3)

        # Segment ID 0: Max of the first two elements = 2
        # Segment ID 1: Max of the next three elements = 5
        # Segment ID 2: Max of the next three elements = 8
        expected = np.array([2, 5, 8], dtype=np.float32)

        self.assertAllClose(outputs, expected)

        # Test 1D with -1 case.
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        segment_ids = np.array([0, 0, 1, 1, -1, 2, 2, -1], dtype=np.int32)
        outputs = kmath.segment_max(data, segment_ids, num_segments=3)
        expected = np.array([2, 4, 7], dtype=np.float32)

        self.assertAllClose(outputs, expected)

        # Test N-D case.
        data = np.random.rand(9, 3, 3)
        segment_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        outputs = kmath.segment_max(data, segment_ids, num_segments=3)

        expected = np.full((3, 3, 3), -np.inf)
        for i in range(data.shape[0]):
            segment_id = segment_ids[i]
            expected[segment_id] = np.maximum(expected[segment_id], data[i])

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

        expected_values = np.zeros((5, 2))
        expected_indices = np.zeros((5, 2), dtype=np.int32)

        for i in range(x.shape[0]):
            top_k_indices = np.argsort(x[i])[-2:][::-1]
            expected_values[i] = x[i, top_k_indices]
            expected_indices[i] = top_k_indices

        self.assertAllClose(outputs[0], expected_values)
        self.assertAllClose(outputs[1], expected_indices)

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

    def test_extract_sequences(self):
        # Test 1D case.
        x = np.random.random((10,))
        sequence_length = 3
        sequence_stride = 2
        output = kmath.extract_sequences(x, sequence_length, sequence_stride)

        num_sequences = 1 + (x.shape[-1] - sequence_length) // sequence_stride
        expected = np.zeros(shape=(num_sequences, sequence_length))
        pos = 0
        for i in range(num_sequences):
            expected[i] = x[pos : pos + sequence_length]
            pos += sequence_stride
        self.assertAllClose(output, expected)

        # Test N-D case.
        x = np.random.random((4, 8))
        sequence_length = 3
        sequence_stride = 2
        output = kmath.extract_sequences(x, sequence_length, sequence_stride)

        num_sequences = 1 + (x.shape[-1] - sequence_length) // sequence_stride
        expected = np.zeros(shape=(4, num_sequences, sequence_length))
        pos = 0
        for i in range(num_sequences):
            expected[:, i] = x[:, pos : pos + sequence_length]
            pos += sequence_stride
        self.assertAllClose(output, expected)

    def test_fft(self):
        real = np.random.random((2, 4, 3))
        imag = np.random.random((2, 4, 3))
        complex_arr = real + 1j * imag

        real_output, imag_output = kmath.fft((real, imag))
        ref = np.fft.fft(complex_arr)
        real_ref = np.real(ref)
        imag_ref = np.imag(ref)
        self.assertAllClose(real_ref, real_output)
        self.assertAllClose(imag_ref, imag_output)

    def test_fft2(self):
        real = np.random.random((2, 4, 3))
        imag = np.random.random((2, 4, 3))
        complex_arr = real + 1j * imag

        real_output, imag_output = kmath.fft2((real, imag))
        ref = np.fft.fft2(complex_arr)
        real_ref = np.real(ref)
        imag_ref = np.imag(ref)
        self.assertAllClose(real_ref, real_output)
        self.assertAllClose(imag_ref, imag_output)

    @parameterized.parameters([(None,), (3,), (15,)])
    def test_rfft(self, n):
        # Test 1D.
        x = np.random.random((10,))
        real_output, imag_output = kmath.rfft(x, fft_length=n)
        ref = np.fft.rfft(x, n=n)
        real_ref = np.real(ref)
        imag_ref = np.imag(ref)
        self.assertAllClose(real_ref, real_output, atol=1e-5, rtol=1e-5)
        self.assertAllClose(imag_ref, imag_output, atol=1e-5, rtol=1e-5)

        # Test N-D case.
        x = np.random.random((2, 3, 10))
        real_output, imag_output = kmath.rfft(x, fft_length=n)
        ref = np.fft.rfft(x, n=n)
        real_ref = np.real(ref)
        imag_ref = np.imag(ref)
        self.assertAllClose(real_ref, real_output, atol=1e-5, rtol=1e-5)
        self.assertAllClose(imag_ref, imag_output, atol=1e-5, rtol=1e-5)

    @parameterized.parameters(
        [
            (32, 8, 32, "hann", True),
            (8, 8, 16, "hann", True),
            (4, 4, 7, "hann", True),
            (32, 8, 32, "hamming", True),
            (32, 8, 32, "hann", False),
            (32, 8, 32, np.ones((32,)), True),
            (32, 8, 32, None, True),
        ]
    )
    def test_stft(
        self, sequence_length, sequence_stride, fft_length, window, center
    ):
        # Test 1D case.
        x = np.random.random((32,))
        real_output, imag_output = kmath.stft(
            x, sequence_length, sequence_stride, fft_length, window, center
        )
        real_ref, imag_ref = _stft(
            x, sequence_length, sequence_stride, fft_length, window, center
        )
        self.assertAllClose(real_ref, real_output, atol=1e-5, rtol=1e-5)
        self.assertAllClose(imag_ref, imag_output, atol=1e-5, rtol=1e-5)

        # Test N-D case.
        x = np.random.random((2, 3, 32))
        real_output, imag_output = kmath.stft(
            x, sequence_length, sequence_stride, fft_length, window, center
        )
        real_ref, imag_ref = _stft(
            x, sequence_length, sequence_stride, fft_length, window, center
        )
        self.assertAllClose(real_ref, real_output, atol=1e-5, rtol=1e-5)
        self.assertAllClose(imag_ref, imag_output, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="Numpy does not support rsqrt.",
    )
    def test_rsqrt(self):
        x = np.array([[1, 4, 9], [16, 25, 36]], dtype="float32")
        self.assertAllClose(kmath.rsqrt(x), 1 / np.sqrt(x))
        self.assertAllClose(kmath.Rsqrt()(x), 1 / np.sqrt(x))
