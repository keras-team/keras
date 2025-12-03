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
    DTYPE = "float32"

    @staticmethod
    def _calc_spectrograms(
        x, mode, scaling, window, periodic, frame_length, frame_step, fft_length
    ):
        data_format = backend.image_data_format()
        input_shape = (None, 1) if data_format == "channels_last" else (1, None)

        layer = Sequential(
            [
                Input(shape=input_shape, dtype=TestSpectrogram.DTYPE),
                layers.STFTSpectrogram(
                    mode=mode,
                    frame_length=frame_length,
                    frame_step=frame_step,
                    fft_length=fft_length,
                    window=window,
                    scaling=scaling,
                    periodic=periodic,
                    dtype=TestSpectrogram.DTYPE,
                ),
            ]
        )
        if data_format == "channels_first":
            y = layer.predict(np.transpose(x, [0, 2, 1]), verbose=0)
            y = np.transpose(y, [0, 2, 1])
        else:
            y = layer.predict(x, verbose=0)

        window_arr = scipy.signal.get_window(window, frame_length, periodic)
        _, _, spec = scipy.signal.spectrogram(
            x[..., 0].astype(TestSpectrogram.DTYPE),
            window=window_arr.astype(TestSpectrogram.DTYPE),
            nperseg=frame_length,
            noverlap=frame_length - frame_step,
            mode=mode,
            scaling=scaling,
            detrend=False,
            nfft=fft_length,
        )
        y_true = np.transpose(spec, [0, 2, 1])
        return y_true, y

    @pytest.mark.requires_trainable_backend
    def test_spectrogram_channels_broadcasting(self):
        rnd = np.random.RandomState(41)
        audio = rnd.uniform(-1, 1, size=(3, 16000, 7))

        layer_last = Sequential(
            [
                Input(shape=(None, 7), dtype=self.DTYPE),
                layers.STFTSpectrogram(
                    mode="psd", dtype=self.DTYPE, data_format="channels_last"
                ),
            ]
        )
        layer_single = Sequential(
            [
                Input(shape=(None, 1), dtype=self.DTYPE),
                layers.STFTSpectrogram(
                    mode="psd", dtype=self.DTYPE, data_format="channels_last"
                ),
            ]
        )

        layer_expand = Sequential(
            [
                Input(shape=(None, 7), dtype=self.DTYPE),
                layers.STFTSpectrogram(
                    mode="psd",
                    dtype=self.DTYPE,
                    data_format="channels_last",
                    expand_dims=True,
                ),
            ]
        )

        y_last = layer_last.predict(audio, verbose=0)
        y_expanded = layer_expand.predict(audio, verbose=0)
        y_singles = [
            layer_single.predict(audio[..., i : i + 1], verbose=0)
            for i in range(audio.shape[-1])
        ]

        self.assertAllClose(y_last, np.concatenate(y_singles, axis=-1))
        self.assertAllClose(y_expanded, np.stack(y_singles, axis=-1))

    @pytest.mark.skipif(
        backend.backend() == "tensorflow",
        reason="TF doesn't support channels_first",
    )
    @pytest.mark.requires_trainable_backend
    def test_spectrogram_channels_first(self):
        rnd = np.random.RandomState(41)
        audio = rnd.uniform(-1, 1, size=(3, 16000, 7))

        layer_first = Sequential(
            [
                Input(shape=(7, None), dtype=self.DTYPE),
                layers.STFTSpectrogram(
                    mode="psd", dtype=self.DTYPE, data_format="channels_first"
                ),
            ]
        )
        layer_last = Sequential(
            [
                Input(shape=(None, 7), dtype=self.DTYPE),
                layers.STFTSpectrogram(
                    mode="psd", dtype=self.DTYPE, data_format="channels_last"
                ),
            ]
        )
        layer_single = Sequential(
            [
                Input(shape=(None, 1), dtype=self.DTYPE),
                layers.STFTSpectrogram(
                    mode="psd", dtype=self.DTYPE, data_format="channels_last"
                ),
            ]
        )
        layer_expand = Sequential(
            [
                Input(shape=(7, None), dtype=self.DTYPE),
                layers.STFTSpectrogram(
                    mode="psd",
                    dtype=self.DTYPE,
                    data_format="channels_first",
                    expand_dims=True,
                ),
            ]
        )

        y_singles = [
            layer_single.predict(audio[..., i : i + 1], verbose=0)
            for i in range(audio.shape[-1])
        ]
        y_expanded = layer_expand.predict(
            np.transpose(audio, [0, 2, 1]), verbose=0
        )
        y_last = layer_last.predict(audio, verbose=0)
        y_first = layer_first.predict(np.transpose(audio, [0, 2, 1]), verbose=0)
        self.assertAllClose(np.transpose(y_first, [0, 2, 1]), y_last)
        self.assertAllClose(y_expanded, np.stack(y_singles, axis=1))
        self.assertAllClose(
            y_first,
            np.transpose(np.concatenate(y_singles, axis=-1), [0, 2, 1]),
        )
        self.run_layer_test(
            layers.STFTSpectrogram,
            init_kwargs={
                "frame_length": 150,
                "frame_step": 10,
                "fft_length": 512,
                "trainable": False,
                "padding": "same",
                "expand_dims": True,
                "data_format": "channels_first",
            },
            input_shape=(2, 3, 160000),
            expected_output_shape=(2, 3, 160000 // 10, 257),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @pytest.mark.requires_trainable_backend
    def test_spectrogram_basics(self):
        self.run_layer_test(
            layers.STFTSpectrogram,
            init_kwargs={
                "frame_length": 500,
                "frame_step": 25,
                "fft_length": 1024,
                "mode": "stft",
                "data_format": "channels_last",
            },
            input_shape=(2, 16000, 1),
            expected_output_shape=(2, 15500 // 25 + 1, 513 * 2),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

        self.run_layer_test(
            layers.STFTSpectrogram,
            init_kwargs={
                "frame_length": 150,
                "frame_step": 71,
                "fft_length": 4096,
                "mode": "real",
                "data_format": "channels_last",
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
            layers.STFTSpectrogram,
            init_kwargs={
                "frame_length": 150,
                "frame_step": 43,
                "fft_length": 512,
                "mode": "imag",
                "padding": "same",
                "data_format": "channels_last",
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
            layers.STFTSpectrogram,
            init_kwargs={
                "frame_length": 150,
                "frame_step": 10,
                "fft_length": 512,
                "trainable": False,
                "padding": "same",
                "data_format": "channels_last",
            },
            input_shape=(2, 160000, 3),
            expected_output_shape=(2, 160000 // 10, 257 * 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )
        self.run_layer_test(
            layers.STFTSpectrogram,
            init_kwargs={
                "frame_length": 150,
                "frame_step": 10,
                "fft_length": 512,
                "trainable": False,
                "padding": "same",
                "expand_dims": True,
                "data_format": "channels_last",
            },
            input_shape=(2, 160000, 3),
            expected_output_shape=(2, 160000 // 10, 257, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Backend does not support dynamic shapes",
    )
    def test_spectrogram_dynamic_shape(self):
        model = Sequential(
            [
                Input(shape=(None, 1), dtype=TestSpectrogram.DTYPE),
                layers.STFTSpectrogram(
                    frame_length=500,
                    frame_step=25,
                    fft_length=1024,
                    mode="stft",
                    data_format="channels_last",
                ),
            ]
        )

        def generator():
            yield (np.random.random((2, 16000, 1)),)
            yield (np.random.random((3, 8000, 1)),)

        model.predict(generator())

    @pytest.mark.requires_trainable_backend
    def test_spectrogram_error(self):
        rnd = np.random.RandomState(41)
        x = rnd.uniform(low=-1, high=1, size=(4, 160000, 1)).astype(self.DTYPE)
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

            if testing.uses_tpu():
                tol_kwargs = {"atol": 5e-2, "rtol": 1e-3}
            else:
                tol_kwargs = {"atol": 5e-4, "rtol": 1e-6}

            init_args["mode"] = "magnitude"
            y_true, y = self._calc_spectrograms(x, **init_args)
            self.assertEqual(np.shape(y_true), np.shape(y))
            self.assertAllClose(y_true, y, **tol_kwargs)

            init_args["mode"] = "psd"
            y_true, y = self._calc_spectrograms(x, **init_args)
            self.assertEqual(np.shape(y_true), np.shape(y))
            self.assertAllClose(y_true, y, **tol_kwargs)

            init_args["mode"] = "angle"
            y_true, y = self._calc_spectrograms(x, **init_args)

            mask = np.isclose(y, y_true, **tol_kwargs)
            mask |= np.isclose(y + 2 * np.pi, y_true, **tol_kwargs)
            mask |= np.isclose(y - 2 * np.pi, y_true, **tol_kwargs)
            mask |= np.isclose(np.cos(y), np.cos(y_true), **tol_kwargs)
            mask |= np.isclose(np.sin(y), np.sin(y_true), **tol_kwargs)

            self.assertLess(np.mean(~mask), 2e-4)

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Requires TF tensors for TF-data module.",
    )
    def test_tf_data_compatibility(self):
        input_shape = (2, 16000, 1)
        output_shape = (2, 16000 // 128, 358)
        layer = layers.STFTSpectrogram(
            frame_length=256,
            frame_step=128,
            fft_length=715,
            padding="same",
            scaling=None,
        )
        input_data = np.random.random(input_shape)
        ds = tf.data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output = output.numpy()
        self.assertEqual(tuple(output.shape), output_shape)

    def test_exceptions(self):
        with self.assertRaises(ValueError):
            layers.STFTSpectrogram(
                frame_length=256, frame_step=1024, fft_length=512
            )
        with self.assertRaises(ValueError):
            layers.STFTSpectrogram(
                frame_length=256, frame_step=0, fft_length=512
            )
        with self.assertRaises(ValueError):
            layers.STFTSpectrogram(
                frame_length=256, frame_step=32, fft_length=128
            )
        with self.assertRaises(ValueError):
            layers.STFTSpectrogram(padding="mypadding")
        with self.assertRaises(ValueError):
            layers.STFTSpectrogram(scaling="l2")
        with self.assertRaises(ValueError):
            layers.STFTSpectrogram(mode="spectrogram")
        with self.assertRaises(ValueError):
            layers.STFTSpectrogram(window="unknowable")
        with self.assertRaises(ValueError):
            layers.STFTSpectrogram(scaling="l2")
        with self.assertRaises(ValueError):
            layers.STFTSpectrogram(padding="divide")
        with self.assertRaises(TypeError):
            layers.STFTSpectrogram()(
                np.random.randint(0, 255, size=(2, 16000, 1))
            )
