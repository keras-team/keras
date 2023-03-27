# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras audio preprocessing layers."""

import math

import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export

from keras.engine import base_layer
from keras.engine import base_preprocessing_layer
from keras.layers.preprocessing import preprocessing_utils as utils


@keras_export("keras.layers.experimental.MelSpectrogram")
class MelSpectrogram(base_layer.Layer):
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

    >>> layer = tf.keras.layers.experimental.MelSpectrogram(num_mel_bins=64,
    ...                                                     sampling_rate=8000,
    ...                                                     fft_stride=256,
    ...                                                     num_fft_bins=2048)
    >>> layer(tf.random.uniform(shape=(16000,)))
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

    >>> layer = tf.keras.layers.experimental.MelSpectrogram(num_mel_bins=80,
    ...                                                     sampling_rate=8000,
    ...                                                     fft_stride=128,
    ...                                                     num_fft_bins=2048)
    >>> layer(tf.random.uniform(shape=(2, 16000)))
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
        window="hann_window",
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
        base_preprocessing_layer.keras_kpl_gauge.get_cell("MelSpectrogram").set(
            True
        )

    def call(self, inputs):
        inputs = utils.ensure_tensor(inputs, dtype=self.compute_dtype)
        outputs = self.spectrogram(inputs)
        outputs = self.melscale(outputs)
        if self.power_to_db:
            outputs = self.dbscale(outputs)
        # swap time & freq axis to have shape of (..., num_mel_bins, time)
        outputs = tf.linalg.matrix_transpose(outputs)
        outputs = tf.cast(outputs, self.compute_dtype)
        return outputs

    def spectrogram(self, inputs):
        spec = tf.signal.stft(
            inputs,
            frame_length=self.window_size,
            frame_step=self.fft_stride,
            fft_length=self.num_fft_bins,
            window_fn=getattr(tf.signal, self.window),
            pad_end=True,
        )
        spec = tf.math.pow(tf.math.abs(spec), self.mag_exp)
        return spec

    def melscale(self, inputs):
        matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bins,
            num_spectrogram_bins=tf.shape(inputs)[-1],
            sample_rate=self.sampling_rate,
            lower_edge_hertz=self.min_freq,
            upper_edge_hertz=self.max_freq,
        )
        return tf.tensordot(inputs, matrix, axes=1)

    def dbscale(self, inputs):
        log_spec = 10.0 * (
            tf.math.log(tf.math.maximum(inputs, self.min_power))
            / tf.math.log(10.0)
        )
        if callable(self.ref_power):
            ref_value = self.ref_power(log_spec)
        else:
            ref_value = tf.math.abs(self.ref_power)
        log_spec -= (
            10.0
            * tf.math.log(tf.math.maximum(ref_value, self.min_power))
            / tf.math.log(10.0)
        )
        log_spec = tf.math.maximum(
            log_spec, tf.math.reduce_max(log_spec) - self.top_db
        )
        return log_spec

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
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
        return tf.TensorShape(output_shape)

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
