import numpy as np
import scipy.signal

from conftest import skip_if_backend
from keras.src import backend
from keras.src import initializers
from keras.src import testing


class ConstantInitializersTest(testing.TestCase):
    def test_zeros_initializer(self):
        shape = (3, 3)

        initializer = initializers.Zeros()
        values = initializer(shape=shape)
        self.assertEqual(values.shape, shape)
        np_values = backend.convert_to_numpy(values)
        self.assertAllClose(np_values, np.zeros(shape=shape))

        self.run_class_serialization_test(initializer)

    def test_ones_initializer(self):
        shape = (3, 3)

        initializer = initializers.Ones()
        values = initializer(shape=shape)
        self.assertEqual(values.shape, shape)
        np_values = backend.convert_to_numpy(values)
        self.assertAllClose(np_values, np.ones(shape=shape))

        self.run_class_serialization_test(initializer)

    def test_constant_initializer(self):
        shape = (3, 3)
        constant_value = 6.0

        initializer = initializers.Constant(value=constant_value)
        values = initializer(shape=shape)
        self.assertEqual(values.shape, shape)
        np_values = backend.convert_to_numpy(values)
        self.assertAllClose(
            np_values, np.full(shape=shape, fill_value=constant_value)
        )

        self.run_class_serialization_test(initializer)

    def test_constant_initializer_array_value(self):
        shape = (3, 3)
        constant_value = np.random.random((3, 3))

        initializer = initializers.Constant(value=constant_value)
        values = initializer(shape=shape)
        self.assertEqual(values.shape, shape)
        np_values = backend.convert_to_numpy(values)
        self.assertAllClose(
            np_values, np.full(shape=shape, fill_value=constant_value)
        )

        self.run_class_serialization_test(initializer)

    @skip_if_backend("openvino", "openvino backend does not support `eye`")
    def test_identity_initializer(self):
        shape = (3, 3)
        gain = 2

        initializer = initializers.Identity(gain=gain)
        values = initializer(shape=shape)
        self.assertEqual(values.shape, shape)
        np_values = backend.convert_to_numpy(values)
        self.assertAllClose(np_values, np.eye(*shape) * gain)

        self.run_class_serialization_test(initializer)

        # Test compatible class_name
        initializer = initializers.get("IdentityInitializer")
        self.assertIsInstance(initializer, initializers.Identity)

    @skip_if_backend("openvino", "openvino backend does not support `arange`")
    def test_stft_initializer(self):
        shape = (256, 1, 513)
        time_range = np.arange(256).reshape((-1, 1, 1))
        freq_range = (np.arange(513) / 1024.0).reshape((1, 1, -1))
        pi = np.arccos(np.float32(-1))
        args = -2 * pi * time_range * freq_range
        tol_kwargs = {"atol": 1e-4, "rtol": 1e-6}

        initializer = initializers.STFT("real", None)
        values = backend.convert_to_numpy(initializer(shape))
        self.assertAllClose(np.cos(args), values, atol=1e-4)
        self.run_class_serialization_test(initializer)

        initializer = initializers.STFT(
            "real",
            "hamming",
            None,
            True,
        )
        window = scipy.signal.windows.get_window("hamming", 256, True)
        window = window.astype("float32").reshape((-1, 1, 1))
        values = backend.convert_to_numpy(initializer(shape, "float32"))
        self.assertAllClose(np.cos(args) * window, values, **tol_kwargs)
        self.run_class_serialization_test(initializer)

        initializer = initializers.STFT(
            "imag",
            "tukey",
            "density",
            False,
        )
        window = scipy.signal.windows.get_window("tukey", 256, False)
        window = window.astype("float32").reshape((-1, 1, 1))
        window = window / np.sqrt(np.sum(window**2))
        values = backend.convert_to_numpy(initializer(shape, "float32"))
        self.assertAllClose(np.sin(args) * window, values, **tol_kwargs)
        self.run_class_serialization_test(initializer)

        initializer = initializers.STFT(
            "imag",
            list(range(1, 257)),
            "spectrum",
        )
        window = np.arange(1, 257)
        window = window.astype("float32").reshape((-1, 1, 1))
        window = window / np.sum(window)
        values = backend.convert_to_numpy(initializer(shape, "float32"))
        self.assertAllClose(np.sin(args) * window, values, **tol_kwargs)
        self.run_class_serialization_test(initializer)

        with self.assertRaises(ValueError):
            initializers.STFT("imaginary")
        with self.assertRaises(ValueError):
            initializers.STFT("real", scaling="l2")
        with self.assertRaises(ValueError):
            initializers.STFT("real", window="unknown")

        # Test compatible class_name
        initializer = initializers.get("STFTInitializer")
        self.assertIsInstance(initializer, initializers.STFT)
