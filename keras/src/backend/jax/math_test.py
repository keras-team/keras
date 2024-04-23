import jax.numpy as jnp
import pytest

from keras.src import backend
from keras.src import testing
from keras.src.backend.jax.math import _get_complex_tensor_from_tuple
from keras.src.backend.jax.math import istft
from keras.src.backend.jax.math import qr
from keras.src.backend.jax.math import segment_max
from keras.src.backend.jax.math import segment_sum
from keras.src.backend.jax.math import stft


@pytest.mark.skipif(
    backend.backend() != "jax", reason="Testing Jax functions only"
)
class TestJaxMathErrors(testing.TestCase):

    def test_segment_sum_no_num_segments(self):
        data = jnp.array([1, 2, 3, 4])
        segment_ids = jnp.array([0, 0, 1, 1])
        with self.assertRaisesRegex(
            ValueError,
            "Argument `num_segments` must be set when using the JAX backend.",
        ):
            segment_sum(data, segment_ids)

    def test_segment_max_no_num_segments(self):
        data = jnp.array([1, 2, 3, 4])
        segment_ids = jnp.array([0, 0, 1, 1])
        with self.assertRaisesRegex(
            ValueError,
            "Argument `num_segments` must be set when using the JAX backend.",
        ):
            segment_max(data, segment_ids)

    def test_qr_invalid_mode(self):
        x = jnp.array([[1, 2], [3, 4]])
        invalid_mode = "invalid_mode"
        with self.assertRaisesRegex(
            ValueError, "Expected one of {'reduced', 'complete'}."
        ):
            qr(x, mode=invalid_mode)

    def test_get_complex_tensor_from_tuple_valid_input(self):
        real = jnp.array([[1.0, 2.0, 3.0]])
        imag = jnp.array([[4.0, 5.0, 6.0]])
        complex_tensor = _get_complex_tensor_from_tuple((real, imag))
        self.assertTrue(jnp.iscomplexobj(complex_tensor))
        self.assertTrue(jnp.array_equal(jnp.real(complex_tensor), real))
        self.assertTrue(jnp.array_equal(jnp.imag(complex_tensor), imag))

    def test_invalid_get_complex_tensor_from_tuple_input_type(self):
        with self.assertRaisesRegex(ValueError, "Input `x` should be a tuple"):
            _get_complex_tensor_from_tuple(jnp.array([1.0, 2.0, 3.0]))

    def test_invalid_get_complex_tensor_from_tuple_input_length(self):
        with self.assertRaisesRegex(ValueError, "Input `x` should be a tuple"):
            _get_complex_tensor_from_tuple(
                (
                    jnp.array([1.0, 2.0, 3.0]),
                    jnp.array([4.0, 5.0, 6.0]),
                    jnp.array([7.0, 8.0, 9.0]),
                )
            )

    def test_mismatched_shapes(self):
        real = jnp.array([1.0, 2.0, 3.0])
        imag = jnp.array([4.0, 5.0])
        with self.assertRaisesRegex(ValueError, "Both the real and imaginary"):
            _get_complex_tensor_from_tuple((real, imag))

    def test_stft_invalid_input_type(self):
        x = jnp.array([1, 2, 3, 4])
        sequence_length = 2
        sequence_stride = 1
        fft_length = 4
        with self.assertRaisesRegex(TypeError, "`float32` or `float64`"):
            stft(x, sequence_length, sequence_stride, fft_length)

    def test_invalid_fft_length(self):
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        sequence_length = 4
        sequence_stride = 1
        fft_length = 2
        with self.assertRaisesRegex(ValueError, "`fft_length` must equal or"):
            stft(x, sequence_length, sequence_stride, fft_length)

    def test_stft_invalid_window(self):
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        sequence_length = 2
        sequence_stride = 1
        fft_length = 4
        window = "invalid_window"
        with self.assertRaisesRegex(ValueError, "If a string is passed to"):
            stft(x, sequence_length, sequence_stride, fft_length, window=window)

    def test_stft_invalid_window_shape(self):
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        sequence_length = 2
        sequence_stride = 1
        fft_length = 4
        window = jnp.ones((sequence_length + 1))
        with self.assertRaisesRegex(ValueError, "The shape of `window` must"):
            stft(x, sequence_length, sequence_stride, fft_length, window=window)

    def test_invalid_not_float_get_complex_tensor_from_tuple_dtype(self):
        real = jnp.array([[1, 2, 3]])
        imag = jnp.array([[4.0, 5.0, 6.0]])
        expected_message = "is not of type float"
        with self.assertRaisesRegex(ValueError, expected_message):
            _get_complex_tensor_from_tuple((real, imag))

    def test_istft_invalid_window_shape2(self):
        x = (jnp.array([[1.0, 2.0]]), jnp.array([[3.0, 4.0]]))
        sequence_length = 2
        sequence_stride = 1
        fft_length = 4
        incorrect_window = jnp.ones(
            (sequence_length + 1,)
        )  # Incorrect window length
        with self.assertRaisesRegex(
            ValueError, "The shape of `window` must be equal to"
        ):
            istft(
                x,
                sequence_length,
                sequence_stride,
                fft_length,
                window=incorrect_window,
            )
