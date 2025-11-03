from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.data_layer import DataLayer

# mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


@keras_export("keras.layers.MelSpectrogram")
class MelSpectrogram(DataLayer):
    """A preprocessing layer to convert raw audio signals to Mel spectrograms.

    This layer takes `float32`/`float64` single or batched audio signal as
    inputs and computes the Mel spectrogram using Short-Time Fourier Transform
    and Mel scaling. The input should be a 1D (unbatched) or 2D (batched) tensor
    representing audio signals. The output will be a 2D or 3D tensor
    representing Mel spectrograms.

    A spectrogram is an image-like representation that shows the frequency
    spectrum of a signal over time. It uses x-axis to represent time, y-axis to
    represent frequency, and each pixel to represent intensity.
    Mel spectrograms are a special type of spectrogram that use the mel scale,
    which approximates how humans perceive sound. They are commonly used in
    speech and music processing tasks like speech recognition, speaker
    identification, and music genre classification.

    **Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline
    (independently of which backend you're using).

    References:
    - [Spectrogram](https://en.wikipedia.org/wiki/Spectrogram),
    - [Mel scale](https://en.wikipedia.org/wiki/Mel_scale).

    Args:
        fft_length: Integer, size of the FFT window.
        sequence_stride: Integer, number of samples between successive STFT
            columns.
        sequence_length: Integer, size of the window used for applying
            `window` to each audio frame. If `None`, defaults to `fft_length`.
        window: String, name of the window function to use. Available values
            are `"hann"` and `"hamming"`. If `window` is a tensor, it will be
            used directly as the window and its length must be
            `sequence_length`. If `window` is `None`, no windowing is
            used. Defaults to `"hann"`.
        sampling_rate: Integer, sample rate of the input signal.
        num_mel_bins: Integer, number of mel bins to generate.
        min_freq: Float, minimum frequency of the mel bins.
        max_freq: Float, maximum frequency of the mel bins.
            If `None`, defaults to `sampling_rate / 2`.
        power_to_db: If True, convert the power spectrogram to decibels.
        top_db: Float, minimum negative cut-off `max(10 * log10(S)) - top_db`.
        mag_exp: Float, exponent for the magnitude spectrogram.
            1 for magnitude, 2 for power, etc. Default is 2.
        ref_power: Float, the power is scaled relative to it
            `10 * log10(S / ref_power)`.
        min_power: Float, minimum value for power and `ref_power`.

    Examples:

    **Unbatched audio signal**

    >>> layer = keras.layers.MelSpectrogram(num_mel_bins=64,
    ...                                     sampling_rate=8000,
    ...                                     sequence_stride=256,
    ...                                     fft_length=2048)
    >>> layer(keras.random.uniform(shape=(16000,))).shape
    (64, 63)

    **Batched audio signal**

    >>> layer = keras.layers.MelSpectrogram(num_mel_bins=80,
    ...                                     sampling_rate=8000,
    ...                                     sequence_stride=128,
    ...                                     fft_length=2048)
    >>> layer(keras.random.uniform(shape=(2, 16000))).shape
    (2, 80, 125)

    Input shape:
        1D (unbatched) or 2D (batched) tensor with shape:`(..., samples)`.

    Output shape:
        2D (unbatched) or 3D (batched) tensor with
        shape:`(..., num_mel_bins, time)`.

    """

    def __init__(
        self,
        fft_length=2048,
        sequence_stride=512,
        sequence_length=None,
        window="hann",
        sampling_rate=16000,
        num_mel_bins=128,
        min_freq=20.0,
        max_freq=None,
        power_to_db=True,
        top_db=80.0,
        mag_exp=2.0,
        min_power=1e-10,
        ref_power=1.0,
        **kwargs,
    ):
        self.fft_length = fft_length
        self.sequence_stride = sequence_stride
        self.sequence_length = sequence_length or fft_length
        self.window = window
        self.sampling_rate = sampling_rate
        self.num_mel_bins = num_mel_bins
        self.min_freq = min_freq
        self.max_freq = max_freq or int(sampling_rate / 2)
        self.power_to_db = power_to_db
        self.top_db = top_db
        self.mag_exp = mag_exp
        self.min_power = min_power
        self.ref_power = ref_power
        super().__init__(**kwargs)

    def call(self, inputs):
        dtype = (
            "float32"
            if self.compute_dtype not in ["float32", "float64"]
            else self.compute_dtype
        )  # jax, tf supports only "float32" and "float64" in stft
        inputs = self.backend.convert_to_tensor(inputs, dtype=dtype)
        outputs = self._spectrogram(inputs)
        outputs = self._melscale(outputs)
        if self.power_to_db:
            outputs = self._dbscale(outputs)
        # swap time & freq axis to have shape of (..., num_mel_bins, time)
        outputs = self.backend.numpy.swapaxes(outputs, -1, -2)
        outputs = self.backend.cast(outputs, self.compute_dtype)
        return outputs

    def _spectrogram(self, inputs):
        real, imag = self.backend.math.stft(
            inputs,
            sequence_length=self.sequence_length,
            sequence_stride=self.sequence_stride,
            fft_length=self.fft_length,
            window=self.window,
            center=True,
        )
        # abs of complex  = sqrt(real^2 + imag^2)
        spec = self.backend.numpy.sqrt(
            self.backend.numpy.add(
                self.backend.numpy.square(real), self.backend.numpy.square(imag)
            )
        )
        spec = self.backend.numpy.power(spec, self.mag_exp)
        return spec

    def _melscale(self, inputs):
        matrix = self.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bins,
            num_spectrogram_bins=self.backend.shape(inputs)[-1],
            sampling_rate=self.sampling_rate,
            lower_edge_hertz=self.min_freq,
            upper_edge_hertz=self.max_freq,
        )
        return self.backend.numpy.tensordot(inputs, matrix, axes=1)

    def _dbscale(self, inputs):
        log_spec = 10.0 * (
            self.backend.numpy.log10(
                self.backend.numpy.maximum(inputs, self.min_power)
            )
        )
        ref_value = self.backend.numpy.abs(
            self.backend.convert_to_tensor(self.ref_power)
        )
        log_spec -= 10.0 * self.backend.numpy.log10(
            self.backend.numpy.maximum(ref_value, self.min_power)
        )
        log_spec = self.backend.numpy.maximum(
            log_spec, self.backend.numpy.max(log_spec) - self.top_db
        )
        return log_spec

    def _hertz_to_mel(self, frequencies_hertz):
        """Converts frequencies in `frequencies_hertz` in Hertz to the
            mel scale.

        Args:
            frequencies_hertz: A tensor of frequencies in Hertz.
            name: An optional name for the operation.

        Returns:
            A tensor of the same shape and type of `frequencies_hertz`
            containing frequencies in the mel scale.
        """
        return _MEL_HIGH_FREQUENCY_Q * self.backend.numpy.log(
            1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ)
        )

    def linear_to_mel_weight_matrix(
        self,
        num_mel_bins=20,
        num_spectrogram_bins=129,
        sampling_rate=8000,
        lower_edge_hertz=125.0,
        upper_edge_hertz=3800.0,
        dtype="float32",
    ):
        """Returns a matrix to warp linear scale spectrograms to the mel scale.

        Returns a weight matrix that can be used to re-weight a tensor
        containing `num_spectrogram_bins` linearly sampled frequency information
        from `[0, sampling_rate / 2]` into `num_mel_bins` frequency information
        from `[lower_edge_hertz, upper_edge_hertz]` on the mel scale.

        This function follows the [Hidden Markov Model Toolkit (HTK)](
        http://htk.eng.cam.ac.uk/) convention, defining the mel scale in
        terms of a frequency in hertz according to the following formula:

        ```mel(f) = 2595 * log10( 1 + f/700)```

        In the returned matrix, all the triangles (filterbanks) have a peak
        value of 1.0.

        For example, the returned matrix `A` can be used to right-multiply a
        spectrogram `S` of shape `[frames, num_spectrogram_bins]` of linear
        scale spectrum values (e.g. STFT magnitudes) to generate a
        "mel spectrogram" `M` of shape `[frames, num_mel_bins]`.

        ```
        # `S` has shape [frames, num_spectrogram_bins]
        # `M` has shape [frames, num_mel_bins]
        M = keras.ops.matmul(S, A)
        ```

        The matrix can be used with `keras.ops.tensordot` to convert an
        arbitrary rank `Tensor` of linear-scale spectral bins into the
        mel scale.

        ```
        # S has shape [..., num_spectrogram_bins].
        # M has shape [..., num_mel_bins].
        M = keras.ops.tensordot(S, A, 1)
        ```

        References:
        - [Mel scale (Wikipedia)](https://en.wikipedia.org/wiki/Mel_scale)

        Args:
            num_mel_bins: Python int. How many bands in the resulting
                mel spectrum.
            num_spectrogram_bins: An integer `Tensor`. How many bins there are
                in the source spectrogram data, which is understood to be
                `fft_size // 2 + 1`, i.e. the spectrogram only contains the
                nonredundant FFT bins.
            sampling_rate: An integer or float `Tensor`. Samples per second of
                the input signal used to create the spectrogram. Used to figure
                out the frequencies corresponding to each spectrogram bin,
                which dictates how they are mapped into the mel scale.
            lower_edge_hertz: Python float. Lower bound on the frequencies to be
                included in the mel spectrum. This corresponds to the lower
                edge of the lowest triangular band.
            upper_edge_hertz: Python float. The desired top edge of the highest
                frequency band.
            dtype: The `DType` of the result matrix. Must be a floating point
                type.

        Returns:
            A tensor of shape `[num_spectrogram_bins, num_mel_bins]`.
        """

        # This function can be constant folded by graph optimization since
        # there are no Tensor inputs.
        sampling_rate = self.backend.cast(sampling_rate, dtype)
        lower_edge_hertz = self.backend.convert_to_tensor(
            lower_edge_hertz,
            dtype,
        )
        upper_edge_hertz = self.backend.convert_to_tensor(
            upper_edge_hertz,
            dtype,
        )
        zero = self.backend.convert_to_tensor(0.0, dtype)

        # HTK excludes the spectrogram DC bin.
        bands_to_zero = 1
        nyquist_hertz = sampling_rate / 2.0
        linear_frequencies = self.backend.numpy.linspace(
            zero, nyquist_hertz, num_spectrogram_bins
        )[bands_to_zero:]
        spectrogram_bins_mel = self.backend.numpy.expand_dims(
            self._hertz_to_mel(linear_frequencies), 1
        )

        # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
        # center of each band is the lower and upper edge of the adjacent bands.
        # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
        # num_mel_bins + 2 pieces.
        band_edges_mel = self.backend.math.extract_sequences(
            self.backend.numpy.linspace(
                self._hertz_to_mel(lower_edge_hertz),
                self._hertz_to_mel(upper_edge_hertz),
                num_mel_bins + 2,
            ),
            sequence_length=3,
            sequence_stride=1,
        )

        # Split the triples up and reshape them into [1, num_mel_bins] tensors.
        lower_edge_mel, center_mel, upper_edge_mel = tuple(
            self.backend.numpy.reshape(t, [1, num_mel_bins])
            for t in self.backend.numpy.split(band_edges_mel, 3, axis=1)
        )

        # Calculate lower and upper slopes for every spectrogram bin.
        # Line segments are linear in the mel domain, not Hertz.
        lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
            center_mel - lower_edge_mel
        )
        upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
            upper_edge_mel - center_mel
        )

        # Intersect the line segments with each other and zero.
        mel_weights_matrix = self.backend.numpy.maximum(
            zero, self.backend.numpy.minimum(lower_slopes, upper_slopes)
        )

        # Re-add the zeroed lower bins we sliced out above.
        return self.backend.numpy.pad(
            mel_weights_matrix,
            [[bands_to_zero, 0], [0, 0]],
        )

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 1:
            output_shape = [
                self.num_mel_bins,
                (
                    (input_shape[0] + self.sequence_stride + 1)
                    // self.sequence_stride
                    if input_shape[0] is not None
                    else None
                ),
            ]
        else:
            output_shape = [
                input_shape[0],
                self.num_mel_bins,
                (
                    (input_shape[1] + self.sequence_stride + 1)
                    // self.sequence_stride
                    if input_shape[1] is not None
                    else None
                ),
            ]
        return output_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "fft_length": self.fft_length,
                "sequence_stride": self.sequence_stride,
                "sequence_length": self.sequence_length,
                "window": self.window,
                "sampling_rate": self.sampling_rate,
                "num_mel_bins": self.num_mel_bins,
                "min_freq": self.min_freq,
                "max_freq": self.max_freq,
                "power_to_db": self.power_to_db,
                "top_db": self.top_db,
                "mag_exp": self.mag_exp,
                "min_power": self.min_power,
                "ref_power": self.ref_power,
            }
        )
        return config
