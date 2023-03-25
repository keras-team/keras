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

    A spectrogram is a visual representation of the spectrum of frequencies
    of a signal as it varies with time. It is image-lke representation of signal
    where the x-axis represents time, the y-axis represents frequency, and
    each pixel represents intensity. Mel spectrograms are a type of
    spectrogram that use the mel scale, which is a perceptual scale that
    approximates human hearing. By converting the frequency axis to
    the Mel scale, Mel spectrograms can reduce the dimensionality of audio data
    while preserving important information for human perception. They are
    commonly used in speech and music processing tasks, such as
    speech recognition, speaker identification, and music genre classification.
    For more information on spectrograms and the mel scale,
    refer to the provided resources. For more in

    Input shape:
        1D (unbatched) or 2D (batched) tensor with shape:`(..., samples)`.

    Output shape:
        2D (unbatched) or 3D (batched) tensor with shape:`(..., n_mels, time)`.

    Args:
        n_fft: Integer, size of the FFT window.
        hop_length: Integer, number of samples between successive STFT columns.
        win_length: Integer, size of the STFT window.
            If `None`, defaults to `n_fft`.
        window_fn: String, name of the window function to use.
        sr: Integer, sample rate of the input signal.
        n_mels: Integer, number of mel bins to generate.
        fmin: Float, minimum frequency of the mel bins.
        fmax: Float, maximum frequency of the mel bins.
            If `None`, defaults to `sr / 2`.
        power_to_db: If True, convert the power spectrogram to decibels.
        top_db: Float, minimum negative cut-off `max(10 * log10(S)) - top_db`.
        power: Float, exponent for the magnitude spectrogram.
            1 for magnitude, 2 for power, etc. Default is 2.
        ref: Float, the power is scaled relative to it `10 * log10(S / ref)`.
        amin: Float, minimum value for power and `ref`.
    """

    def __init__(
        self,
        n_fft=2048,
        hop_length=512,
        win_length=None,
        window="hann_window",
        sr=16000,
        n_mels=128,
        fmin=20.0,
        fmax=None,
        power_to_db=True,
        top_db=80.0,
        power=2.0,
        amin=1e-10,
        ref=1.0,
        **kwargs,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or int(sr / 2)
        self.power_to_db = power_to_db
        self.top_db = top_db
        self.power = power
        self.amin = amin
        self.ref = ref
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
        # swap time & freq axis to have shape of (..., n_mels, time)
        outputs = tf.linalg.matrix_transpose(outputs)
        outputs = tf.cast(outputs, self.compute_dtype)
        return outputs

    def spectrogram(self, inputs):
        spec = tf.signal.stft(
            inputs,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
            window_fn=getattr(tf.signal, self.window),
            pad_end=True,
        )
        spec = tf.math.pow(tf.math.abs(spec), self.power)
        return spec

    def melscale(self, inputs):
        matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=tf.shape(inputs)[-1],
            sample_rate=self.sr,
            lower_edge_hertz=self.fmin,
            upper_edge_hertz=self.fmax,
        )
        return tf.tensordot(inputs, matrix, axes=1)

    def dbscale(self, inputs):
        log_spec = 10.0 * (
            tf.math.log(tf.math.maximum(inputs, self.amin)) / tf.math.log(10.0)
        )
        if callable(self.ref):
            ref_value = self.ref(log_spec)
        else:
            ref_value = tf.math.abs(self.ref)
        log_spec -= (
            10.0
            * tf.math.log(tf.math.maximum(ref_value, self.amin))
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
                self.n_mels,
                int(math.ceil(input_shape[0] / self.hop_length)),
            ]
        else:
            output_shape = [
                input_shape[0],
                self.n_mels,
                int(math.ceil(input_shape[1] / self.hop_length)),
            ]
        return tf.TensorShape(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "win_length": self.win_length,
                "window": self.window,
                "sr": self.sr,
                "n_mels": self.n_mels,
                "fmin": self.fmin,
                "fmax": self.fmax,
                "power_to_db": self.power_to_db,
                "top_db": self.top_db,
                "power": self.power,
                "amin": self.amin,
                "ref": self.ref,
            }
        )
        return config
