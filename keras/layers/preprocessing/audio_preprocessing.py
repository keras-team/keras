import math

from keras import ops
from keras.api_export import keras_export
from keras.layers.layer import Layer


def gcd(a, b):
    """Returns the greatest common divisor via Euclid's algorithm.

    Args:
        a: The dividend. A scalar integer `Tensor`.
        b: The divisor. A scalar integer `Tensor`.

    Returns:
        A scalar `Tensor` representing the greatest common divisor
        between `a` and `b`.

    Raises:
        ValueError: If `a` or `b` are not scalar integers.
    """
    a = ops.convert_to_tensor(a)
    b = ops.convert_to_tensor(b)

    a, b = ops.while_loop(
        lambda _, b: ops.greater(b, ops.zeros_like(b)),
        lambda a, b: [b, ops.mod(a, b)],
        [a, b],
    )
    return a


def frame(
    signal,
    frame_length,
    frame_step,
    pad_end=False,
    pad_value=0,
    axis=-1,
    name=None,
):
    signal = ops.convert_to_tensor(signal)
    frame_length = ops.convert_to_tensor(frame_length)
    frame_step = ops.convert_to_tensor(frame_step)
    axis = ops.convert_to_tensor(axis)

    signal_shape = ops.shape(signal)
    # Axis can be negative. Convert it to positive.
    axis = ops.arange(ops.ndim(signal_shape))[axis]
    outer_dimensions, length_samples, inner_dimensions = ops.split(
        ops.array(signal_shape), [axis, axis + 1]
    )
    length_samples = ops.reshape(length_samples, [])
    num_outer_dimensions = ops.size(outer_dimensions)
    num_inner_dimensions = ops.size(inner_dimensions)

    # If padding is requested, pad the input signal tensor with pad_value.
    if pad_end:
        pad_value = ops.convert_to_tensor(pad_value, signal.dtype)

        # Calculate number of frames, using double negatives to round up.
        num_frames = -(-length_samples // frame_step)

        # Pad the signal by up to frame_length samples based on how many samples
        # are remaining starting from last_frame_position.
        pad_samples = ops.maximum(
            0, frame_length + frame_step * (num_frames - 1) - length_samples
        )

        # Pad the inner dimension of signal by pad_samples.
        paddings = ops.concatenate(
            [
                ops.zeros([num_outer_dimensions, 2], dtype=pad_samples.dtype),
                ops.convert_to_tensor([[0, pad_samples]]),
                ops.zeros([num_inner_dimensions, 2], dtype=pad_samples.dtype),
            ],
            0,
        )
        signal = ops.pad(signal, paddings, constant_values=pad_value)

        signal_shape = ops.shape(signal)
        length_samples = signal_shape[axis]
    else:
        num_frames = ops.maximum(
            ops.convert_to_tensor(0, dtype=frame_length.dtype),
            1 + (length_samples - frame_length) // frame_step,
        )

    subframe_length = gcd(frame_length, frame_step)
    subframes_per_frame = frame_length // subframe_length
    subframes_per_hop = frame_step // subframe_length
    num_subframes = length_samples // subframe_length

    slice_shape = ops.concatenate(
        [outer_dimensions, [num_subframes * subframe_length], inner_dimensions],
        0,
    )
    subframe_shape = ops.concatenate(
        [outer_dimensions, [num_subframes, subframe_length], inner_dimensions],
        0,
    )
    subframes = ops.reshape(
        ops.slice(signal, ops.zeros_like(signal_shape), slice_shape),
        ops.convert_to_numpy(subframe_shape).tolist(),
    )

    # frame_selector is a [num_frames, subframes_per_frame] tensor
    # that indexes into the appropriate frame in subframes. For example:
    # [[0, 0, 0, 0], [2, 2, 2, 2], [4, 4, 4, 4]]
    frame_selector = ops.reshape(
        ops.arange(num_frames, dtype=frame_length.dtype) * subframes_per_hop,
        [num_frames, 1],
    )

    # subframe_selector is a [num_frames, subframes_per_frame] tensor
    # that indexes into the appropriate subframe within a frame. For example:
    # [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    subframe_selector = ops.reshape(
        ops.arange(subframes_per_frame, dtype=frame_length.dtype),
        [1, subframes_per_frame],
    )

    # Adding the 2 selector tensors together produces a [num_frames,
    # subframes_per_frame] tensor of indices to use with tf.gather to select
    # subframes from subframes. We then reshape the inner-most
    # subframes_per_frame dimension to stitch the subframes together into
    # frames. For example: [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]].
    selector = frame_selector + subframe_selector

    # Dtypes have to match.
    outer_dimensions = ops.convert_to_tensor(outer_dimensions)
    inner_dimensions = ops.convert_to_tensor(
        inner_dimensions, dtype=outer_dimensions.dtype
    )
    mid_dimensions = ops.convert_to_tensor(
        [num_frames, frame_length], dtype=outer_dimensions.dtype
    )

    new_shape = ops.concatenate(
        [outer_dimensions, mid_dimensions, inner_dimensions], 0
    )
    frames = ops.reshape(
        ops.take(subframes, selector, axis=axis),
        ops.convert_to_numpy(new_shape).tolist(),
    )

    return frames


# mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def _mel_to_hertz(mel_values, name=None):
    """Converts frequencies in `mel_values` from the mel scale to linear scale.

    Args:
    mel_values: A `Tensor` of frequencies in the mel scale.
    name: An optional name for the operation.

    Returns:
    A `Tensor` of the same shape and type as `mel_values` containing linear
    scale frequencies in Hertz.
    """
    mel_values = ops.convert_to_tensor(mel_values)
    return _MEL_BREAK_FREQUENCY_HERTZ * (
        ops.exp(mel_values / _MEL_HIGH_FREQUENCY_Q) - 1.0
    )


def _hertz_to_mel(frequencies_hertz, name=None):
    """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale.

    Args:
    frequencies_hertz: A `Tensor` of frequencies in Hertz.
    name: An optional name for the operation.

    Returns:
    A `Tensor` of the same shape and type of `frequencies_hertz` containing
    frequencies in the mel scale.
    """
    frequencies_hertz = ops.convert_to_tensor(frequencies_hertz)
    return _MEL_HIGH_FREQUENCY_Q * ops.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ)
    )


def linear_to_mel_weight_matrix(
    num_mel_bins=20,
    num_spectrogram_bins=129,
    sample_rate=8000,
    lower_edge_hertz=125.0,
    upper_edge_hertz=3800.0,
    dtype="float32",
):
    r"""Returns a matrix to warp linear scale spectrograms
        to the [mel scale][mel].

    Returns a weight matrix that can be used to re-weight a `Tensor` containing
    `num_spectrogram_bins` linearly sampled frequency information from
    `[0, sample_rate / 2]` into `num_mel_bins` frequency information from
    `[lower_edge_hertz, upper_edge_hertz]` on the [mel scale][mel].

    This function follows the [Hidden Markov Model Toolkit
    (HTK)](http://htk.eng.cam.ac.uk/) convention, defining the mel scale in
    terms of a frequency in hertz according to the following formula:

        $$\textrm{mel}(f) = 2595 * \textrm{log}_{10}(1 + \frac{f}{700})$$

    In the returned matrix, all the triangles (filterbanks) have a peak value
    of 1.0.

    For example, the returned matrix `A` can be used to right-multiply a
    spectrogram `S` of shape `[frames, num_spectrogram_bins]` of linear
    scale spectrum values (e.g. STFT magnitudes) to generate a "mel spectrogram"
    `M` of shape `[frames, num_mel_bins]`.

        # `S` has shape [frames, num_spectrogram_bins]
        # `M` has shape [frames, num_mel_bins]
        M = keras.ops.matmul(S, A)

    The matrix can be used with `tf.tensordot` to convert an arbitrary rank
    `Tensor` of linear-scale spectral bins into the mel scale.

        # S has shape [..., num_spectrogram_bins].
        # M has shape [..., num_mel_bins].
        M = keras.ops.tensordot(S, A, 1)

    Args:
        num_mel_bins: Python int. How many bands in the resulting mel spectrum.
        num_spectrogram_bins: An integer `Tensor`. How many bins there are
            in the source spectrogram data, which is understood to be
            `fft_size // 2 + 1`, i.e. the spectrogram only contains the
            nonredundant FFT bins.
        sample_rate: An integer or float `Tensor`. Samples per second of the
            input signal used to create the spectrogram. Used to figure out the
            frequencies corresponding to each spectrogram bin, which dictates
            how they are mapped into the mel scale.
        lower_edge_hertz: Python float. Lower bound on the frequencies to be
            included in the mel spectrum. This corresponds to the lower edge of
            the lowest triangular band.
        upper_edge_hertz: Python float. The desired top edge of the highest
            frequency band.
        dtype: The `DType` of the result matrix. Must be a floating point type.

    Returns:
        A `Tensor` of shape `[num_spectrogram_bins, num_mel_bins]`.

    [mel]: https://en.wikipedia.org/wiki/Mel_scale
    """

    # This function can be constant folded by graph optimization since there are
    # no Tensor inputs.
    sample_rate = ops.cast(sample_rate, dtype)
    lower_edge_hertz = ops.convert_to_tensor(
        lower_edge_hertz,
        dtype,
    )
    upper_edge_hertz = ops.convert_to_tensor(
        upper_edge_hertz,
        dtype,
    )
    zero = ops.convert_to_tensor(0.0, dtype)

    # HTK excludes the spectrogram DC bin.
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = ops.linspace(
        zero, nyquist_hertz, num_spectrogram_bins
    )[bands_to_zero:]
    spectrogram_bins_mel = ops.expand_dims(_hertz_to_mel(linear_frequencies), 1)

    # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
    # center of each band is the lower and upper edge of the adjacent bands.
    # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
    # num_mel_bins + 2 pieces.
    band_edges_mel = frame(
        ops.linspace(
            _hertz_to_mel(lower_edge_hertz),
            _hertz_to_mel(upper_edge_hertz),
            num_mel_bins + 2,
        ),
        frame_length=3,
        frame_step=1,
    )

    # Split the triples up and reshape them into [1, num_mel_bins] tensors.
    lower_edge_mel, center_mel, upper_edge_mel = tuple(
        ops.reshape(t, [1, num_mel_bins])
        for t in ops.split(band_edges_mel, 3, axis=1)
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
    mel_weights_matrix = ops.maximum(
        zero, ops.minimum(lower_slopes, upper_slopes)
    )

    # Re-add the zeroed lower bins we sliced out above.
    return ops.pad(
        mel_weights_matrix,
        [[bands_to_zero, 0], [0, 0]],
    )


@keras_export("keras.layers.MelSpectrogram")
class MelSpectrogram(Layer):
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

    For more information on spectrograms, you may refer to this resources:
    [spectrogram](https://en.wikipedia.org/wiki/Spectrogram),
    [mel scale](https://en.wikipedia.org/wiki/Mel_scale).

    Examples:

    **Unbatched audio signal**

    >>> layer = keras.layers.MelSpectrogram(num_mel_bins=64,
    ...                                     sampling_rate=8000,
    ...                                     fft_stride=256,
    ...                                     num_fft_bins=2048)
    >>> layer(keras.random.uniform(shape=(16000,)))
    <tf.Tensor: shape=(64, 63), dtype=float32, numpy=
    array([[ 27.002258 ,  25.82557  ,  24.530018 , ...,  29.447317 ,
            22.576866 ,   4.1253514],
        [ 28.87952  ,  27.97433  ,  26.31767  , ...,  26.986296 ,
            19.680676 ,   1.4215486],
        [ 27.325214 ,  27.779736 ,  27.688522 , ...,  24.389975 ,
            17.259838 ,  -1.4919776],
        ...,
        [ 33.47647  ,  33.61223  ,  32.695736 , ...,  23.01299  ,
            11.53509  , -14.854364 ],
        [ 32.956306 ,  33.096653 ,  33.092117 , ...,  22.744698 ,
            12.255739 , -13.4533825],
        [ 32.777905 ,  33.19077  ,  34.070583 , ...,  22.417168 ,
            12.26748  , -10.877618 ]], dtype=float32)>

    **Batched audio signal**

    >>> layer = keras.layers.MelSpectrogram(num_mel_bins=80,
    ...                                     sampling_rate=8000,
    ...                                     fft_stride=128,
    ...                                     num_fft_bins=2048)
    >>> layer(keras.random.uniform(shape=(2, 16000)))
    <tf.Tensor: shape=(2, 80, 125), dtype=float32, numpy=
    array([[[ 23.235947  ,  22.86543   ,  22.391176  , ...,  21.741554  ,
            15.451798  ,   2.4877253 ],
            [ 20.351486  ,  20.715343  ,  21.380817  , ...,  18.518717  ,
            11.968127  ,   0.46634886],
            [ 20.032818  ,  21.016296  ,  22.054605  , ...,  16.514582  ,
            9.840168  ,  -2.1056828 ],
            ...,
            [ 31.911928  ,  31.824018  ,  31.787327  , ...,  13.440712  ,
            4.3596454 ,  -9.938191  ],
            [ 32.92584   ,  33.671867  ,  34.169167  , ...,  10.817527  ,
            2.873957  , -10.4782915 ],
            [ 30.819023  ,  31.034756  ,  31.179695  , ...,   9.792138  ,
            0.40263397, -14.491785  ]],

        [[ 23.705862  ,  24.24318   ,  24.736097  , ...,  21.071415  ,
            14.810348  ,   1.906768  ],
            [ 23.744732  ,  23.76305   ,  23.666683  , ...,  17.193201  ,
            10.598775  ,  -0.476082  ],
            [ 23.355988  ,  23.098003  ,  22.922604  , ...,  13.369602  ,
            7.3324995 ,  -3.4960124 ],
            ...,
            [ 31.444962  ,  31.662983  ,  31.764927  , ...,  12.580458  ,
            4.8858614 ,  -8.833308  ],
            [ 31.369892  ,  31.349333  ,  31.308098  , ...,  13.320463  ,
            4.72253   , -10.279094  ],
            [ 31.86178   ,  31.784441  ,  31.860874  , ...,   7.7960706 ,
            -0.7777866 , -15.290524  ]]], dtype=float32)>

    Input shape:
        1D (unbatched) or 2D (batched) tensor with shape:`(..., samples)`.

    Output shape:
        2D (unbatched) or 3D (batched) tensor with
        shape:`(..., num_mel_bins, time)`.

    Args:
        num_fft_bins: Integer, size of the FFT window.
        fft_stride: Integer, number of samples between successive STFT columns.
        window_size: Integer, size of the window used for applying `window_fn`
            to each audio frame. If `None`, defaults to `num_fft_bins`.
        window_fn: String, name of the window function to use.
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
    """

    def __init__(
        self,
        num_fft_bins=2048,
        fft_stride=512,
        window_size=None,
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
        self.num_fft_bins = num_fft_bins
        self.fft_stride = fft_stride
        self.window_size = window_size or num_fft_bins
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
        inputs = ops.convert_to_tensor(inputs, dtype=self.compute_dtype)
        outputs = self.spectrogram(inputs)
        outputs = self.melscale(outputs)
        if self.power_to_db:
            outputs = self.dbscale(outputs)
        # swap time & freq axis to have shape of (..., num_mel_bins, time)
        outputs = ops.swapaxes(outputs, -1, -2)
        outputs = ops.cast(outputs, self.compute_dtype)
        return outputs

    def spectrogram(self, inputs):
        real, imag = ops.stft(
            inputs,
            sequence_length=self.window_size,
            sequence_stride=self.fft_stride,
            fft_length=self.num_fft_bins,
            window=self.window,
            center=True,
        )
        # abs of complex  = sqrt(real^2 + imag^2)
        spec = ops.sqrt(ops.add(ops.square(real), ops.square(imag)))
        spec = ops.power(spec, self.mag_exp)
        return spec

    def melscale(self, inputs):
        matrix = linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bins,
            num_spectrogram_bins=ops.shape(inputs)[-1],
            sample_rate=self.sampling_rate,
            lower_edge_hertz=self.min_freq,
            upper_edge_hertz=self.max_freq,
        )
        return ops.tensordot(inputs, matrix, axes=1)

    def dbscale(self, inputs):
        log_spec = 10.0 * (
            ops.log(ops.maximum(inputs, self.min_power)) / ops.log(10.0)
        )
        if callable(self.ref_power):
            ref_value = self.ref_power(log_spec)
        else:
            ref_value = ops.abs(self.ref_power)
        log_spec -= (
            10.0
            * ops.log(ops.maximum(ref_value, self.min_power))
            / ops.log(10.0)
        )
        log_spec = ops.maximum(log_spec, ops.max(log_spec) - self.top_db)
        return log_spec

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 1:
            output_shape = [
                self.num_mel_bins,
                int(math.ceil(input_shape[0] / self.fft_stride)),
            ]
        else:
            output_shape = [
                input_shape[0],
                self.num_mel_bins,
                int(math.ceil(input_shape[1] / self.fft_stride)),
            ]
        return output_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_fft_bins": self.num_fft_bins,
                "fft_stride": self.fft_stride,
                "window_size": self.window_size,
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
