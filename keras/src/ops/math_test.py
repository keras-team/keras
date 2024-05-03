import math

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.signal
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.backend.common import dtypes
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.ops import math as kmath


def _stft(
    x, sequence_length, sequence_stride, fft_length, window="hann", center=True
):
    # pure numpy version of stft that matches librosa's implementation
    x = np.array(x)
    ori_dtype = x.dtype

    if center:
        pad_width = [(0, 0) for _ in range(len(x.shape))]
        pad_width[-1] = (fft_length // 2, fft_length // 2)
        x = np.pad(x, pad_width, mode="reflect")

    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        if isinstance(window, str):
            window = scipy.signal.get_window(window, sequence_length)
        win = np.array(window, dtype=x.dtype)
        win = np.pad(win, [[l_pad, r_pad]])
    else:
        win = np.ones((sequence_length + l_pad + r_pad), dtype=x.dtype)

    x = scipy.signal.stft(
        x,
        fs=1.0,
        window=win,
        nperseg=(sequence_length + l_pad + r_pad),
        noverlap=(sequence_length + l_pad + r_pad - sequence_stride),
        nfft=fft_length,
        boundary=None,
        padded=False,
    )[-1]

    # scale and swap to (..., num_sequences, fft_bins)
    x = x / np.sqrt(1.0 / win.sum() ** 2)
    x = np.swapaxes(x, -2, -1)
    return np.real(x).astype(ori_dtype), np.imag(x).astype(ori_dtype)


def _istft(
    x,
    sequence_length,
    sequence_stride,
    fft_length,
    length=None,
    window="hann",
    center=True,
):
    # pure numpy version of istft that matches librosa's implementation
    complex_input = x[0] + 1j * x[1]
    x = np.fft.irfft(
        complex_input, n=fft_length, axis=-1, norm="backward"
    ).astype(x[0].dtype)

    expected_output_len = fft_length + sequence_stride * (x.shape[-2] - 1)

    if window is not None:
        if isinstance(window, str):
            win = np.array(
                scipy.signal.get_window(window, sequence_length), dtype=x.dtype
            )
        else:
            win = np.array(window, dtype=x.dtype)
        l_pad = (fft_length - sequence_length) // 2
        r_pad = fft_length - sequence_length - l_pad
        win = np.pad(win, [[l_pad, r_pad]])

        # square and sum
        _sequence_length = sequence_length + l_pad + r_pad
        denom = np.square(win)
        overlaps = -(-_sequence_length // sequence_stride)
        denom = np.pad(
            denom, [(0, overlaps * sequence_stride - _sequence_length)]
        )
        denom = np.reshape(denom, [overlaps, sequence_stride])
        denom = np.sum(denom, 0, keepdims=True)
        denom = np.tile(denom, [overlaps, 1])
        denom = np.reshape(denom, [overlaps * sequence_stride])
        win = np.divide(win, denom[:_sequence_length])
        x = np.multiply(x, win)

    # overlap_sequences
    def _overlap_sequences(x, sequence_stride):
        *batch_shape, num_sequences, sequence_length = x.shape
        flat_batchsize = math.prod(batch_shape)
        x = np.reshape(x, (flat_batchsize, num_sequences, sequence_length))
        output_size = sequence_stride * (num_sequences - 1) + sequence_length
        nstep_per_segment = 1 + (sequence_length - 1) // sequence_stride
        padded_segment_len = nstep_per_segment * sequence_stride
        x = np.pad(
            x, ((0, 0), (0, 0), (0, padded_segment_len - sequence_length))
        )
        x = np.reshape(
            x,
            (flat_batchsize, num_sequences, nstep_per_segment, sequence_stride),
        )
        x = x.transpose((0, 2, 1, 3))
        x = np.pad(x, ((0, 0), (0, 0), (0, num_sequences), (0, 0)))
        shrinked = x.shape[2] - 1
        x = np.reshape(x, (flat_batchsize, -1))
        x = x[:, : (nstep_per_segment * shrinked * sequence_stride)]
        x = np.reshape(
            x, (flat_batchsize, nstep_per_segment, shrinked * sequence_stride)
        )
        x = np.sum(x, axis=1)[:, :output_size]
        return np.reshape(x, tuple(batch_shape) + (-1,))

    x = _overlap_sequences(x, sequence_stride)

    if backend.backend() in {"numpy", "jax"}:
        x = np.nan_to_num(x)

    start = 0 if center is False else fft_length // 2
    if length is not None:
        end = start + length
    elif center:
        end = -(fft_length // 2)
    else:
        end = expected_output_len
    return x[..., start:end]


class MathOpsDynamicShapeTest(testing.TestCase, parameterized.TestCase):
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

    @parameterized.parameters([(None,), (1,), (5,)])
    def test_rfft(self, fft_length):
        x = KerasTensor((None, 4, 3), dtype="float32")
        real_output, imag_output = kmath.rfft(x, fft_length=fft_length)
        ref = np.fft.rfft(np.ones((2, 4, 3)), n=fft_length)
        ref_shape = (None,) + ref.shape[1:]
        self.assertEqual(real_output.shape, ref_shape)
        self.assertEqual(imag_output.shape, ref_shape)

    @parameterized.parameters([(None,), (1,), (5,)])
    def test_irfft(self, fft_length):
        real = KerasTensor((None, 4, 3), dtype="float32")
        imag = KerasTensor((None, 4, 3), dtype="float32")
        output = kmath.irfft((real, imag), fft_length=fft_length)
        ref = np.fft.irfft(np.ones((2, 4, 3)), n=fft_length)
        ref_shape = (None,) + ref.shape[1:]
        self.assertEqual(output.shape, ref_shape)

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

    def test_istft(self):
        sequence_length = 10
        sequence_stride = 3
        fft_length = 15
        real = KerasTensor((None, 32), dtype="float32")
        imag = KerasTensor((None, 32), dtype="float32")
        output = kmath.istft(
            (real, imag), sequence_length, sequence_stride, fft_length
        )
        ref = _istft(
            (np.ones((5, 32)), np.ones((5, 32))),
            sequence_length,
            sequence_stride,
            fft_length,
        )
        ref_shape = (None,) + ref.shape[1:]
        self.assertEqual(output.shape, ref_shape)

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

    def test_irfft(self):
        real = KerasTensor((2, 4, 3), dtype="float32")
        imag = KerasTensor((2, 4, 3), dtype="float32")
        output = kmath.irfft((real, imag))
        ref = np.fft.irfft(np.ones((2, 4, 3)))
        self.assertEqual(output.shape, ref.shape)

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

    def test_istft(self):
        # sequence_stride must <= x[0].shape[-1]
        # sequence_stride must >= fft_length / num_sequences
        sequence_length = 10
        sequence_stride = 3
        fft_length = 15
        num_sequences = fft_length // sequence_stride + 1
        real = KerasTensor((num_sequences, 32), dtype="float32")
        imag = KerasTensor((num_sequences, 32), dtype="float32")
        output = kmath.istft(
            (real, imag), sequence_length, sequence_stride, fft_length
        )
        ref = _istft(
            (np.ones((num_sequences, 32)), np.ones((num_sequences, 32))),
            sequence_length,
            sequence_stride,
            fft_length,
        )
        self.assertEqual(output.shape, ref.shape)


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

    @parameterized.parameters([(None,), (3,), (15,)])
    def test_irfft(self, n):
        # Test 1D.
        real = np.random.random((10,))
        imag = np.random.random((10,))
        complex_arr = real + 1j * imag
        output = kmath.irfft((real, imag), fft_length=n)
        ref = np.fft.irfft(complex_arr, n=n)
        self.assertAllClose(output, ref, atol=1e-5, rtol=1e-5)

        # Test N-D case.
        real = np.random.random((2, 3, 10))
        imag = np.random.random((2, 3, 10))
        complex_arr = real + 1j * imag
        output = kmath.irfft((real, imag), fft_length=n)
        ref = np.fft.irfft(complex_arr, n=n)
        self.assertAllClose(output, ref, atol=1e-5, rtol=1e-5)

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

    @parameterized.parameters(
        [
            (32, 8, 32, "hann", True),
            (8, 8, 16, "hann", True),
            (4, 4, 7, "hann", True),
            (32, 8, 32, "hamming", True),
            (8, 4, 8, "hann", False),
            (32, 8, 32, np.ones((32,)), True),
            (32, 8, 32, None, True),
        ]
    )
    def test_istft(
        self, sequence_length, sequence_stride, fft_length, window, center
    ):
        # sequence_stride must <= x[0].shape[-1]
        # sequence_stride must >= fft_length / num_sequences
        # Test 1D case.
        x = np.random.random((256,))
        real_x, imag_x = _stft(
            x, sequence_length, sequence_stride, fft_length, window, center
        )
        output = kmath.istft(
            (real_x, imag_x),
            sequence_length,
            sequence_stride,
            fft_length,
            window=window,
            center=center,
        )
        ref = _istft(
            (real_x, imag_x),
            sequence_length,
            sequence_stride,
            fft_length,
            window=window,
            center=center,
        )
        if backend.backend() in ("numpy", "jax", "torch"):
            # these backends have different implementation for the boundary of
            # the output, so we need to truncate 5% befroe assertAllClose
            truncated_len = int(output.shape[-1] * 0.05)
            output = output[..., truncated_len:-truncated_len]
            ref = ref[..., truncated_len:-truncated_len]
        self.assertAllClose(output, ref, atol=1e-5, rtol=1e-5)

        # Test N-D case.
        x = np.random.random((2, 3, 256))
        real_x, imag_x = _stft(
            x, sequence_length, sequence_stride, fft_length, window, center
        )
        output = kmath.istft(
            (real_x, imag_x),
            sequence_length,
            sequence_stride,
            fft_length,
            window=window,
            center=center,
        )
        ref = _istft(
            (real_x, imag_x),
            sequence_length,
            sequence_stride,
            fft_length,
            window=window,
            center=center,
        )
        if backend.backend() in ("numpy", "jax", "torch"):
            # these backends have different implementation for the boundary of
            # the output, so we need to truncate 5% befroe assertAllClose
            truncated_len = int(output.shape[-1] * 0.05)
            output = output[..., truncated_len:-truncated_len]
            ref = ref[..., truncated_len:-truncated_len]
        self.assertAllClose(output, ref, atol=1e-5, rtol=1e-5)

    def test_rsqrt(self):
        x = np.array([[1, 4, 9], [16, 25, 36]], dtype="float32")
        self.assertAllClose(kmath.rsqrt(x), 1 / np.sqrt(x))
        self.assertAllClose(kmath.Rsqrt()(x), 1 / np.sqrt(x))

    def test_erf_operation_basic(self):
        # Sample values for testing
        sample_values = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

        # Expected output using numpy's approximation of the error function
        expected_output = scipy.special.erf(sample_values)

        # Output from the erf operation in keras_core
        output_from_erf_op = kmath.erf(sample_values)

        # Assert that the outputs are close
        self.assertAllClose(expected_output, output_from_erf_op, atol=1e-4)

    def test_erf_operation_dtype(self):
        # Test for float32 and float64 data types
        for dtype in ("float32", "float64"):
            sample_values = np.array(
                [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=dtype
            )
            expected_output = scipy.special.erf(sample_values)
            output_from_erf_op = kmath.erf(sample_values)
            self.assertAllClose(expected_output, output_from_erf_op, atol=1e-4)

    def test_erf_operation_edge_cases(self):
        # Test for edge cases
        edge_values = np.array([1e5, -1e5, 1e-5, -1e-5], dtype=np.float64)
        expected_output = scipy.special.erf(edge_values)
        output_from_edge_erf_op = kmath.erf(edge_values)
        self.assertAllClose(expected_output, output_from_edge_erf_op, atol=1e-4)

    def test_erfinv_operation_basic(self):
        # Sample values for testing
        sample_values = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

        # Expected output using numpy's approximation of the error function
        expected_output = scipy.special.erfinv(sample_values)

        # Output from the erf operation in keras_core
        output_from_erfinv_op = kmath.erfinv(sample_values)

        # Assert that the outputs are close
        self.assertAllClose(expected_output, output_from_erfinv_op, atol=1e-4)

    def test_erfinv_operation_dtype(self):
        # Test for float32 and float64 data types
        for dtype in ("float32", "float64"):
            sample_values = np.array(
                [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=dtype
            )
            expected_output = scipy.special.erfinv(sample_values)
            output_from_erfinv_op = kmath.erfinv(sample_values)
            self.assertAllClose(
                expected_output, output_from_erfinv_op, atol=1e-4
            )

    def test_erfinv_operation_edge_cases(self):
        # Test for edge cases
        edge_values = np.array([1e5, -1e5, 1e-5, -1e-5], dtype=np.float64)
        expected_output = scipy.special.erfinv(edge_values)
        output_from_edge_erfinv_op = kmath.erfinv(edge_values)
        self.assertAllClose(
            expected_output, output_from_edge_erfinv_op, atol=1e-4
        )


class MathDtypeTest(testing.TestCase, parameterized.TestCase):
    """Test the floating dtype to verify that the behavior matches JAX."""

    # TODO: Using uint64 will lead to weak type promotion (`float`),
    # resulting in different behavior between JAX and Keras. Currently, we
    # are skipping the test for uint64
    ALL_DTYPES = [
        x for x in dtypes.ALLOWED_DTYPES if x not in ["string", "uint64"]
    ] + [None]
    INT_DTYPES = [x for x in dtypes.INT_TYPES if x != "uint64"]
    FLOAT_DTYPES = dtypes.FLOAT_TYPES

    if backend.backend() == "torch":
        # TODO: torch doesn't support uint16, uint32 and uint64
        ALL_DTYPES = [
            x for x in ALL_DTYPES if x not in ["uint16", "uint32", "uint64"]
        ]
        INT_DTYPES = [
            x for x in INT_DTYPES if x not in ["uint16", "uint32", "uint64"]
        ]

    def setUp(self):
        from jax.experimental import enable_x64

        self.jax_enable_x64 = enable_x64()
        self.jax_enable_x64.__enter__()
        return super().setUp()

    def tearDown(self) -> None:
        self.jax_enable_x64.__exit__(None, None, None)
        return super().tearDown()


class ExtractSequencesOpTest(testing.TestCase):
    def test_extract_sequences_init_length_1_stride_1(self):
        extract_op = kmath.ExtractSequences(
            sequence_length=1, sequence_stride=1
        )
        self.assertIsNotNone(extract_op)
        self.assertEqual(extract_op.sequence_length, 1)
        self.assertEqual(extract_op.sequence_stride, 1)

    def test_extract_sequences_init_length_5_stride_2(self):
        extract_op = kmath.ExtractSequences(
            sequence_length=5, sequence_stride=2
        )
        self.assertIsNotNone(extract_op)
        self.assertEqual(extract_op.sequence_length, 5)
        self.assertEqual(extract_op.sequence_stride, 2)

    def test_compute_output_spec_low_rank(self):
        extract_op = kmath.ExtractSequences(
            sequence_length=5, sequence_stride=1
        )
        low_rank_input = np.array(42)
        error_message = r"Input should have rank >= 1. Received: .*"
        with self.assertRaisesRegex(ValueError, error_message):
            extract_op.compute_output_spec(low_rank_input)

    def test_extract_sequences_call(self):
        sequence_length, sequence_stride = 5, 2
        extract_op = kmath.ExtractSequences(sequence_length, sequence_stride)
        test_input = np.random.rand(10, 20)
        result = extract_op.call(test_input)

        expected_shape = self.calculate_expected_shape(
            test_input.shape, sequence_length, sequence_stride
        )
        self.assertEqual(result.shape, expected_shape)

    def calculate_expected_shape(
        self, input_shape, sequence_length, sequence_stride
    ):
        num_sequences = (
            (input_shape[1] - sequence_length) // sequence_stride
        ) + 1
        return (input_shape[0], num_sequences, sequence_length)


class SegmentSumTest(testing.TestCase):
    def test_segment_sum_call(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        segment_ids = np.array([0, 0, 1], dtype=np.int32)
        num_segments = 2
        sorted_segments = False
        segment_sum_op = kmath.SegmentSum(
            num_segments=num_segments, sorted=sorted_segments
        )
        output = segment_sum_op.call(data, segment_ids)
        expected_output = np.array([[5, 7, 9], [7, 8, 9]], dtype=np.float32)
        self.assertAllClose(output, expected_output)


class SegmentMaxTest(testing.TestCase):
    def test_segment_max_call(self):
        data = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]], dtype=np.float32)
        segment_ids = np.array([0, 0, 1], dtype=np.int32)
        num_segments = 2
        sorted_segments = False
        segment_max_op = kmath.SegmentMax(
            num_segments=num_segments, sorted=sorted_segments
        )
        output = segment_max_op.call(data, segment_ids)
        expected_output = np.array([[2, 5, 8], [3, 6, 9]], dtype=np.float32)
        self.assertAllClose(output, expected_output)


class TopKTest(testing.TestCase):
    def test_top_k_call_values(self):
        data = np.array([[1, 3, 2], [4, 6, 5]], dtype=np.float32)
        k = 2
        sorted_flag = True
        top_k_op = kmath.TopK(k=k, sorted=sorted_flag)
        values, _ = top_k_op.call(data)
        expected_values = np.array([[3, 2], [6, 5]], dtype=np.float32)
        self.assertAllClose(values, expected_values)

    def test_top_k_call_indices(self):
        data = np.array([[1, 3, 2], [4, 6, 5]], dtype=np.float32)
        k = 2
        sorted_flag = True
        top_k_op = kmath.TopK(k=k, sorted=sorted_flag)
        _, indices = top_k_op.call(data)
        expected_indices = np.array([[1, 2], [1, 2]], dtype=np.int32)
        self.assertAllClose(indices, expected_indices)


class InTopKTest(testing.TestCase):
    def test_in_top_k_call(self):
        targets = np.array([2, 0, 1], dtype=np.int32)
        predictions = np.array(
            [[0.1, 0.2, 0.7], [1.0, 0.2, 0.3], [0.2, 0.6, 0.2]],
            dtype=np.float32,
        )
        k = 2
        in_top_k_op = kmath.InTopK(k=k)
        output = in_top_k_op.call(targets, predictions)
        expected_output = np.array([True, True, True], dtype=bool)
        self.assertAllEqual(output, expected_output)


class LogsumexpTest(testing.TestCase):
    def test_logsumexp_call(self):
        x = np.array([[1, 2], [3, 4]], dtype=np.float32)
        axis = 0
        keepdims = True
        logsumexp_op = kmath.Logsumexp(axis=axis, keepdims=keepdims)
        output = logsumexp_op.call(x)
        expected_output = np.log(
            np.sum(np.exp(x), axis=axis, keepdims=keepdims)
        )
        self.assertAllClose(output, expected_output)


class FFTTest(testing.TestCase):
    def test_fft_input_not_tuple_or_list(self):
        fft_op = kmath.FFT()
        with self.assertRaisesRegex(
            ValueError, "Input `x` should be a tuple of two tensors"
        ):
            fft_op.compute_output_spec(np.array([1, 2, 3]))

    def test_fft_input_parts_different_shapes(self):
        fft_op = kmath.FFT()
        real = np.array([1, 2, 3])
        imag = np.array([1, 2])
        with self.assertRaisesRegex(
            ValueError,
            "Both the real and imaginary parts should have the same shape",
        ):
            fft_op.compute_output_spec((real, imag))

    def test_fft_input_not_1d(self):
        fft_op = kmath.FFT()
        real = np.array(1)
        imag = np.array(1)
        with self.assertRaisesRegex(ValueError, "Input should have rank >= 1"):
            fft_op.compute_output_spec((real, imag))

    def test_fft_last_axis_not_fully_defined(self):
        fft_op = kmath.FFT()
        real = KerasTensor(shape=(None,), dtype="float32")
        imag = KerasTensor(shape=(None,), dtype="float32")
        with self.assertRaisesRegex(
            ValueError, "Input should have its -1th axis fully-defined"
        ):
            fft_op.compute_output_spec((real, imag))

    def test_fft_init_default_axis(self):
        fft_op = kmath.FFT()
        self.assertEqual(fft_op.axis, -1, "Default axis should be -1")


class FFT2Test(testing.TestCase):
    def test_fft2_correct_input(self):
        fft2_op = kmath.FFT2()
        real_part = np.random.rand(2, 3, 4)
        imag_part = np.random.rand(2, 3, 4)
        # This should not raise any errors
        fft2_op.compute_output_spec((real_part, imag_part))

    def test_fft2_incorrect_input_type(self):
        fft2_op = kmath.FFT2()
        incorrect_input = np.array([1, 2, 3])  # Not a tuple or list
        with self.assertRaisesRegex(
            ValueError, "should be a tuple of two tensors"
        ):
            fft2_op.compute_output_spec(incorrect_input)

    def test_fft2_mismatched_shapes(self):
        fft2_op = kmath.FFT2()
        real_part = np.random.rand(2, 3, 4)
        imag_part = np.random.rand(2, 3)  # Mismatched shape
        with self.assertRaisesRegex(
            ValueError,
            "Both the real and imaginary parts should have the same shape",
        ):
            fft2_op.compute_output_spec((real_part, imag_part))

    def test_fft2_low_rank(self):
        fft2_op = kmath.FFT2()
        low_rank_input = np.random.rand(3)  # Rank of 1
        with self.assertRaisesRegex(ValueError, "Input should have rank >= 2"):
            fft2_op.compute_output_spec((low_rank_input, low_rank_input))

    def test_fft2_undefined_dimensions(self):
        fft2_op = kmath.FFT2()
        real_part = KerasTensor(shape=(None, None, 3), dtype="float32")
        imag_part = KerasTensor(shape=(None, None, 3), dtype="float32")
        with self.assertRaisesRegex(
            ValueError, "Input should have its .* axes fully-defined"
        ):
            fft2_op.compute_output_spec((real_part, imag_part))


class RFFTTest(testing.TestCase):
    def test_rfft_low_rank_input(self):
        rfft_op = kmath.RFFT()
        low_rank_input = np.array(5)
        with self.assertRaisesRegex(ValueError, "Input should have rank >= 1"):
            rfft_op.compute_output_spec(low_rank_input)

    def test_rfft_defined_fft_length(self):
        fft_length = 10
        rfft_op = kmath.RFFT(fft_length=fft_length)
        input_tensor = np.random.rand(3, 8)

        expected_last_dimension = fft_length // 2 + 1
        expected_shape = input_tensor.shape[:-1] + (expected_last_dimension,)

        output_tensors = rfft_op.compute_output_spec(input_tensor)
        for output_tensor in output_tensors:
            self.assertEqual(output_tensor.shape, expected_shape)

        def test_rfft_undefined_fft_length_defined_last_dim(self):
            rfft_op = kmath.RFFT()
            input_tensor = np.random.rand(3, 8)
            expected_last_dimension = input_tensor.shape[-1] // 2 + 1
            expected_shape = input_tensor.shape[:-1] + (
                expected_last_dimension,
            )
            output_tensors = rfft_op.compute_output_spec(input_tensor)
            for output_tensor in output_tensors:
                self.assertEqual(output_tensor.shape, expected_shape)

    def test_rfft_undefined_fft_length_undefined_last_dim(self):
        rfft_op = kmath.RFFT()
        input_tensor = KerasTensor(shape=(None, None), dtype="float32")
        expected_shape = input_tensor.shape[:-1] + (None,)
        output_tensors = rfft_op.compute_output_spec(input_tensor)
        for output_tensor in output_tensors:
            self.assertEqual(output_tensor.shape, expected_shape)


class ISTFTTest(testing.TestCase):
    def test_istft_incorrect_input_type(self):
        istft_op = kmath.ISTFT(
            sequence_length=5, sequence_stride=2, fft_length=10
        )
        incorrect_input = np.array([1, 2, 3])
        with self.assertRaisesRegex(
            ValueError, "should be a tuple of two tensors"
        ):
            istft_op.compute_output_spec(incorrect_input)

    def test_istft_mismatched_shapes(self):
        istft_op = kmath.ISTFT(
            sequence_length=5, sequence_stride=2, fft_length=10
        )
        real_part = np.random.rand(2, 3, 4)
        imag_part = np.random.rand(2, 3)
        with self.assertRaisesRegex(
            ValueError,
            "Both the real and imaginary parts should have the same shape",
        ):
            istft_op.compute_output_spec((real_part, imag_part))

    def test_istft_low_rank_input(self):
        istft_op = kmath.ISTFT(
            sequence_length=5, sequence_stride=2, fft_length=10
        )
        low_rank_input = np.random.rand(3)
        with self.assertRaisesRegex(ValueError, "Input should have rank >= 2"):
            istft_op.compute_output_spec((low_rank_input, low_rank_input))

    def test_input_not_tuple_or_list_raises_error(self):
        irfft_op = kmath.IRFFT()
        invalid_input = np.array([1, 2, 3])
        with self.assertRaisesRegex(
            ValueError, "Input `x` should be a tuple of two tensors"
        ):
            irfft_op.compute_output_spec(invalid_input)

    def test_input_tuple_with_less_than_two_elements_raises_error(self):
        irfft_op = kmath.IRFFT()
        too_short_input = (np.array([1, 2, 3]),)
        with self.assertRaisesRegex(
            ValueError, "Input `x` should be a tuple of two tensors"
        ):
            irfft_op.compute_output_spec(too_short_input)

    def test_input_tuple_with_more_than_two_elements_raises_error(self):
        irfft_op = kmath.IRFFT()
        too_long_input = (
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9]),
        )
        with self.assertRaisesRegex(
            ValueError, "Input `x` should be a tuple of two tensors"
        ):
            irfft_op.compute_output_spec(too_long_input)

    def test_mismatched_shapes_input_validation(self):
        irfft_op = kmath.IRFFT()

        # Create real and imaginary parts with mismatched shapes
        real_part = np.array([1, 2, 3])
        imag_part = np.array([[1, 2], [3, 4]])

        with self.assertRaisesRegex(
            ValueError,
            "Both the real and imaginary parts should have the same shape",
        ):
            irfft_op.compute_output_spec((real_part, imag_part))

    def test_insufficient_rank_input_validation(self):
        irfft_op = kmath.IRFFT()

        # Create real and imaginary parts with insufficient rank (0D)
        real_part = np.array(1)
        imag_part = np.array(1)

        with self.assertRaisesRegex(ValueError, "Input should have rank >= 1"):
            irfft_op.compute_output_spec((real_part, imag_part))

    def test_with_specified_fft_length(self):
        fft_length = 10
        irfft_op = kmath.IRFFT(fft_length=fft_length)

        real_part = np.random.rand(4, 8)
        imag_part = np.random.rand(4, 8)

        expected_shape = real_part.shape[:-1] + (fft_length,)
        output_shape = irfft_op.compute_output_spec(
            (real_part, imag_part)
        ).shape

        self.assertEqual(output_shape, expected_shape)

    def test_inferred_fft_length_with_defined_last_dimension(self):
        irfft_op = kmath.IRFFT()

        real_part = np.random.rand(4, 8)
        imag_part = np.random.rand(4, 8)

        inferred_fft_length = 2 * (real_part.shape[-1] - 1)
        expected_shape = real_part.shape[:-1] + (inferred_fft_length,)
        output_shape = irfft_op.compute_output_spec(
            (real_part, imag_part)
        ).shape

        self.assertEqual(output_shape, expected_shape)

    def test_undefined_fft_length_and_last_dimension(self):
        irfft_op = kmath.IRFFT()

        real_part = KerasTensor(shape=(4, None), dtype="float32")
        imag_part = KerasTensor(shape=(4, None), dtype="float32")

        output_spec = irfft_op.compute_output_spec((real_part, imag_part))
        expected_shape = real_part.shape[:-1] + (None,)

        self.assertEqual(output_spec.shape, expected_shape)


class TestMathErrors(testing.TestCase):

    @pytest.mark.skipif(
        backend.backend() != "jax", reason="Testing Jax errors only"
    )
    def test_segment_sum_no_num_segments(self):
        data = jnp.array([1, 2, 3, 4])
        segment_ids = jnp.array([0, 0, 1, 1])
        with self.assertRaisesRegex(
            ValueError,
            "Argument `num_segments` must be set when using the JAX backend.",
        ):
            kmath.segment_sum(data, segment_ids)

    @pytest.mark.skipif(
        backend.backend() != "jax", reason="Testing Jax errors only"
    )
    def test_segment_max_no_num_segments(self):
        data = jnp.array([1, 2, 3, 4])
        segment_ids = jnp.array([0, 0, 1, 1])
        with self.assertRaisesRegex(
            ValueError,
            "Argument `num_segments` must be set when using the JAX backend.",
        ):
            kmath.segment_max(data, segment_ids)

    def test_stft_invalid_input_type(self):
        # backend agnostic error message
        x = np.array([1, 2, 3, 4])
        sequence_length = 2
        sequence_stride = 1
        fft_length = 4
        with self.assertRaisesRegex(TypeError, "`float32` or `float64`"):
            kmath.stft(x, sequence_length, sequence_stride, fft_length)

    def test_invalid_fft_length(self):
        # backend agnostic error message
        x = np.array([1.0, 2.0, 3.0, 4.0])
        sequence_length = 4
        sequence_stride = 1
        fft_length = 2
        with self.assertRaisesRegex(ValueError, "`fft_length` must equal or"):
            kmath.stft(x, sequence_length, sequence_stride, fft_length)

    def test_stft_invalid_window(self):
        # backend agnostic error message
        x = np.array([1.0, 2.0, 3.0, 4.0])
        sequence_length = 2
        sequence_stride = 1
        fft_length = 4
        window = "invalid_window"
        with self.assertRaisesRegex(ValueError, "If a string is passed to"):
            kmath.stft(
                x, sequence_length, sequence_stride, fft_length, window=window
            )

    def test_stft_invalid_window_shape(self):
        # backend agnostic error message
        x = np.array([1.0, 2.0, 3.0, 4.0])
        sequence_length = 2
        sequence_stride = 1
        fft_length = 4
        window = np.ones((sequence_length + 1))
        with self.assertRaisesRegex(ValueError, "The shape of `window` must"):
            kmath.stft(
                x, sequence_length, sequence_stride, fft_length, window=window
            )

    def test_istft_invalid_window_shape_2D_inputs(self):
        # backend agnostic error message
        x = (np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]))
        sequence_length = 2
        sequence_stride = 1
        fft_length = 4
        incorrect_window = np.ones((sequence_length + 1,))
        with self.assertRaisesRegex(
            ValueError, "The shape of `window` must be equal to"
        ):
            kmath.istft(
                x,
                sequence_length,
                sequence_stride,
                fft_length,
                window=incorrect_window,
            )
