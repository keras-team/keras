import numpy as np
import pytest
import scipy.signal
import tensorflow as tf

from keras import Input
from keras import Sequential
from keras.src import backend
from keras.src import layers
from keras.src import testing


class TestSpectrogram(testing.TestCase):
    @staticmethod
    def _calc_spectrograms(
        X, mode, scaling, window, periodic, frame_length, frame_step, fft_length
    ):
        dtype = "float64"

        layer = Sequential(
            [
                Input(shape=(160000, 1), dtype=dtype),
                layers.Spectrogram(
                    mode=mode,
                    frame_length=frame_length,
                    frame_step=frame_step,
                    fft_length=fft_length,
                    window=window,
                    scaling=scaling,
                    periodic=periodic,
                    dtype=dtype,
                ),
            ]
        )
        Y = layer.predict(X, verbose=0)

        window_arr = scipy.signal.get_window(window, frame_length, periodic)
        _, _, S = scipy.signal.spectrogram(
            X[..., 0].astype(np.float64),
            window=window_arr,
            nperseg=frame_length,
            noverlap=frame_length - frame_step,
            mode=mode,
            scaling=scaling,
            detrend=False,
            nfft=fft_length,
        )
        Y_true = np.transpose(S, [0, 2, 1])
        return Y_true, Y

    @pytest.mark.requires_trainable_backend
    def test_spectrogram_basics(self):
        self.run_layer_test(
            layers.Spectrogram,
            init_kwargs={
                "frame_length": 500,
                "frame_step": 25,
                "fft_length": 1024,
            },
            input_shape=(2, 16000, 1),
            expected_output_shape=(2, 15500 // 25 + 1, 513),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

        self.run_layer_test(
            layers.Spectrogram,
            init_kwargs={
                "frame_length": 150,
                "frame_step": 71,
                "fft_length": 4096,
                "mode": "real",
            },
            input_shape=(2, 160000, 1),
            expected_output_shape=(2, 159850 // 71 + 1, 2049),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

        self.run_layer_test(
            layers.Spectrogram,
            init_kwargs={
                "frame_length": 150,
                "frame_step": 43,
                "fft_length": 512,
                "mode": "imag",
                "padding": "same",
            },
            input_shape=(2, 160000, 1),
            expected_output_shape=(2, 160000 // 43 + 1, 257),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )
        self.run_layer_test(
            layers.Spectrogram,
            init_kwargs={
                "frame_length": 150,
                "frame_step": 10,
                "fft_length": 512,
                "trainable": False,
                "padding": "same",
            },
            input_shape=(2, 160000, 1),
            expected_output_shape=(2, 160000 // 10 + 1, 257),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @pytest.mark.requires_trainable_backend
    def test_spectrogram_error(self):
        rnd = np.random.RandomState(41)
        X = rnd.uniform(low=-1, high=1, size=(4, 160000, 1)).astype(np.float64)
        names = [
            "scaling",
            "window",
            "periodic",
            "frame_length",
            "frame_step",
            "fft_length",
        ]
        for args in [
            ("density", "hann", False, 512, 256, 1024),
            ("spectrum", "blackman", True, 512, 32, 1024),
            ("spectrum", "hamming", True, 256, 192, 512),
            ("spectrum", "tukey", False, 512, 128, 512),
            ("density", "hamming", True, 256, 256, 256),
            ("density", "hann", True, 256, 128, 256),
        ]:
            init_args = dict(zip(names, args))

            tol_kwargs = {"atol": 5e-4, "rtol": 1e-6}

            init_args["mode"] = "magnitude"
            Y_true, Y = self._calc_spectrograms(X, **init_args)
            self.assertEqual(np.shape(Y_true), np.shape(Y))
            self.assertAllClose(Y_true, Y, **tol_kwargs)

            init_args["mode"] = "psd"
            Y_true, Y = self._calc_spectrograms(X, **init_args)
            self.assertEqual(np.shape(Y_true), np.shape(Y))
            self.assertAllClose(Y_true, Y, **tol_kwargs)

            init_args["mode"] = "angle"
            Y_true, Y = self._calc_spectrograms(X, **init_args)

            tol_kwargs = {"atol": 5e-4, "rtol": 1e-5}

            PI = np.arccos(np.float128(-1)).astype(Y_true.dtype)
            mask = np.isclose(Y_true, Y, **tol_kwargs)
            mask |= np.isclose(Y_true, Y + 2 * PI, **tol_kwargs)
            mask |= np.isclose(Y_true, Y - 2 * PI, **tol_kwargs)

            self.assertAllClose(np.cos(Y_true), np.cos(Y), **tol_kwargs)
            self.assertAllClose(np.sin(Y_true), np.sin(Y), **tol_kwargs)
            self.assertTrue(np.all(mask))

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Requires TF tensors."
    )
    def test_tf_data_compatibility(self):
        input_shape = (2, 16000, 1)
        output_shape = (2, 16000 // 128 + 1, 257)
        layer = layers.Spectrogram(
            frame_length=256, frame_step=128, fft_length=512, padding="same"
        )
        input_data = np.random.random(input_shape)
        ds = tf.data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output = output.numpy()
        self.assertEqual(tuple(output.shape), output_shape)

    def test_exceptions(self):
        with self.assertRaises(ValueError):
            layers.Spectrogram(
                frame_length=256, frame_step=1024, fft_length=512
            )
        with self.assertRaises(ValueError):
            layers.Spectrogram(frame_length=256, frame_step=32, fft_length=128)
        with self.assertRaises(ValueError):
            layers.Spectrogram(padding="mypadding")
        with self.assertRaises(ValueError):
            layers.Spectrogram(scaling="l2")
        with self.assertRaises(ValueError):
            layers.Spectrogram(mode="spectrogram")
