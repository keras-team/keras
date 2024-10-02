import math
import warnings

from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.utils.module_utils import scipy


class STFTInitializer(initializers.Initializer):
    def __init__(self, side):
        if side not in ["real", "imag"]:
            raise ValueError(f"side should be 'real' or 'imag', not {side}")
        self.side = side

    def __call__(self, shape, dtype=None):
        dtype = backend.standardize_dtype(dtype)
        frame_length, _, fft_length = shape

        _fft_length = (fft_length - 1) * 2
        freq = (
            ops.reshape(ops.arange(fft_length, dtype=dtype), (1, 1, fft_length))
            / _fft_length
        )
        time = ops.reshape(
            ops.arange(frame_length, dtype=dtype), (frame_length, 1, 1)
        )
        args = -2 * time * freq * ops.arccos(ops.cast(-1, dtype))

        if self.side == "real":
            return ops.cast(ops.cos(args), dtype)
        elif self.side == "imag":
            return ops.cast(ops.sin(args), dtype)

    def get_config(self):
        return {"side": self.side}


@keras_export("keras.layers.Spectrogram")
class Spectrogram(layers.Layer):
    """
    A layer that computes Spectrograms of the input signal to produce
    a spectrogram. This layers utilizes Short-Time Fourier Transform (STFT) by
    utilizing convolution kernels, which allows parallelization on GPUs and
    trainable kernels for fine-tuning support. This layer allows
    different modes of output (e.g., log-scaled magnitude, phase, power
    spectral density, etc.) and provides flexibility in windowing, padding,
    and scaling options for the STFT calculation.

    Args:
        mode (str): The output type of the spectrogram. Can be one of
            'psd' (Power Spectral Density), 'magnitude', 'real', 'imag',
            'angle', or 'log'. Default is 'log'.
        frame_length (int): The length of each frame (window) for STFT in
            samples. Default is 256.
        frame_step (int, optional): The step size (hop length) between
            consecutive frames. If not provided, defaults to half the
            frame_length.
        fft_length (int, optional): The size of the FFT to apply to each frame.
            Should be a power of two and greater than or equal to
            `frame_length`.  Defaults to the smallest power of two that is
            greater than or equal to `frame_length`.
        window (str or array_like): The windowing function to apply to each
            frame. Can be 'hann' (default), 'hamming', or a custom window
            provided as an array_like.
        periodic (bool): If True, the window function will be treated as
            periodic. Default is False.
        scaling (str): Type of scaling applied to the window. Can be 'density',
            'spectrum', or None. Default is 'density'.
        padding (str): Padding strategy. Can be 'valid' or 'same'.
            Default is 'valid'.
        padding_mode (str): The padding mode to use when padding is applied.
            Default is 'constant'.

    Raises:
        ValueError: If an invalid value is provided for 'mode', 'scaling',
            'padding', or other input arguments.
        TypeError: If the input data type is not one of 'float16', 'float32',
            or 'float64'.

    Input shape:
        A 3D tensor of shape (batch_size, time_length, num_channels), where
        `time_length` is the length of the input signal. Currently,
        only `num_channels=1` is supported.

    Output shape:
        A 3D tensor of shape (batch_size, new_time_length, output_features),
        where new_time_length depends on the padding, and output_features is
        the number of FFT bins (fft_length // 2 + 1).

    Example:
        ```
        spectrogram_layer = keras.layers.Spectrogram(
            mode='log', frame_length=256, fft_length=512
        )
        output = spectrogram_layer(input_signal)
        ```

    """

    def __init__(
        self,
        mode="log",
        frame_length=256,
        frame_step=None,
        fft_length=None,
        window="hann",
        periodic=False,
        scaling="density",
        padding="valid",
        padding_mode="constant",
        **kwargs,
    ):
        if fft_length is not None and (fft_length & -fft_length) != fft_length:
            warnings.warn(
                "`fft_length` is recommended to be a power of two. "
                f"Received fft_length={fft_length}"
            )

        if frame_step is not None and (
            frame_step > frame_length or frame_step < 0
        ):
            raise ValueError(
                "`frame_step` should not be greater than `frame_length`"
            )

        if fft_length is not None and (
            fft_length < frame_length or fft_length < 0 or fft_length % 2 != 0
        ):
            raise ValueError(
                "`fft_length` should be an even integer and "
                "not less than `frame_length`"
            )

        all_modes = ["psd", "magnitude", "real", "imag", "angle", "log"]

        if mode not in all_modes:
            raise ValueError(
                "Output mode is invalid, it must be one of "
                f"{', '.join(all_modes)}. Received: mode={mode}"
            )

        if scaling is not None and scaling not in ["density", "spectrum"]:
            raise ValueError(
                "Scaling is invalid, it must be `None`, 'density' "
                f"or 'spectrum'. Received scaling={scaling}"
            )

        if padding not in ["valid", "same"]:
            raise ValueError(
                "Padding is invalid, it should be 'valid', 'same'. "
                f"Received: padding={padding}"
            )

        super().__init__(**kwargs)

        self.mode = mode

        self.frame_length = frame_length
        self.frame_step = frame_step
        self._frame_step = frame_step or self.frame_length // 2
        self.fft_length = fft_length
        self._fft_length = fft_length or (
            2 ** int(math.ceil(math.log2(frame_length)))
        )

        self.window = window
        self.periodic = periodic
        self.scaling = scaling

        self.padding = padding
        self.padding_mode = padding_mode
        self._padding_length = 0
        if self.padding == "same":
            self._padding_length = self.frame_length

        self.input_spec = layers.input_spec.InputSpec(
            ndim=3, shape=(None, None, 1)
        )

    def build(self, input_shape):
        if self.mode != "imag":
            self.real_kernel = self.add_weight(
                name="real_kernel",
                shape=(self.frame_length, 1, self._fft_length // 2 + 1),
                initializer=STFTInitializer("real"),
            )
        if self.mode != "real":
            self.imag_kernel = self.add_weight(
                name="imag_kernel",
                shape=(self.frame_length, 1, self._fft_length // 2 + 1),
                initializer=STFTInitializer("imag"),
            )
        self.built = True

    def call(self, inputs):
        dtype = inputs.dtype
        if backend.standardize_dtype(dtype) not in {
            "float16",
            "float32",
            "float64",
        }:
            raise TypeError(
                "Invalid input type. Expected `float16`, `float32` or "
                f"`float64`. Received: input type={dtype}"
            )
        win = None
        scaling = 1
        if self.window is not None:
            win = self.window
            if isinstance(win, str):
                # Using SciPy since it provides more windowing functions,
                # easier to be compatible with multiple backends,
                # and this is a one-time operation to store the constant
                # window tensor in the graph
                win = scipy.signal.get_window(
                    win, self.frame_length, self.periodic
                )
            win = ops.convert_to_tensor(win, dtype=dtype)
            if len(win.shape) != 1 or win.shape[-1] != self.frame_length:
                raise ValueError(
                    "The shape of `window` must be equal to [frame_length]."
                    f"Received: window shape={win.shape}"
                )
            win = ops.reshape(win, [-1, 1, 1])
            if self.scaling == "density":
                scaling = ops.sqrt(ops.sum(ops.square(win)) + backend.epsilon())
            elif self.scaling == "spectrum":
                scaling = ops.abs(ops.sum(win)) + backend.epsilon()

        if self.padding != "valid":
            assert self.frame_length > 0
            pad_value = self._padding_length // 2
            inputs = ops.pad(
                inputs,
                [[0, 0], [pad_value, pad_value], [0, 0]],
                mode=self.padding_mode,
            )

        real_signal = None
        imag_signal = None
        power = None
        if self.mode != "imag":
            real_kernel = ops.cast(
                self.real_kernel if win is None else self.real_kernel * win,
                dtype,
            )
            real_signal = (
                ops.conv(
                    inputs,
                    real_kernel,
                    strides=self._frame_step,
                    data_format="channels_last",
                )
                / scaling
            )

        if self.mode != "real":
            imag_kernel = ops.cast(
                self.imag_kernel if win is None else self.imag_kernel * win,
                dtype,
            )
            imag_signal = (
                ops.conv(
                    inputs,
                    imag_kernel,
                    strides=self._frame_step,
                    data_format="channels_last",
                )
                / scaling
            )
        if self.mode == "real":
            return real_signal
        elif self.mode == "imag":
            return imag_signal
        elif self.mode == "angle":
            return ops.arctan2(imag_signal, real_signal)
        else:
            power = ops.square(real_signal) + ops.square(imag_signal)

        if self.mode == "psd":
            return power + ops.pad(power[..., 1:-1], [[0, 0], [0, 0], [1, 1]])

        linear_stft = ops.sqrt(power + backend.epsilon())

        if self.mode == "magnitude":
            return linear_stft
        elif self.mode == "log":
            return ops.log(backend.epsilon() + linear_stft)

        raise NotImplementedError(f"{self.mode} mode is not implemented")

    def compute_output_shape(self, input_shape):
        batch_size, time_length, channels = input_shape
        pad_value = self._padding_length // 2

        new_time_length = None
        if time_length is not None:
            new_time_length = (
                time_length - self.frame_length + pad_value * 2
            ) // self._frame_step + 1

        output_features = self._fft_length // 2 + 1

        return batch_size, new_time_length, output_features

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mode": self.mode,
                "frame_length": self.frame_length,
                "frame_step": self.frame_step,
                "fft_length": self.fft_length,
                "window": self.window,
                "periodic": self.periodic,
                "scaling": self.scaling,
                "padding": self.padding,
                "padding_mode": self.padding_mode,
            }
        )
        return config
