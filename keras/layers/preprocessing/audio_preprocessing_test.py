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
"""Tests for audio preprocessing layers."""

import math

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras.layers.preprocessing import audio_preprocessing
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.run_all_keras_modes(always_skip_v1=True)
class MelSpectrogramTest(test_combinations.TestCase):
    def _run_test(self, kwargs, num_fft_bins, fft_stride, num_mel_bins):
        np.random.seed(1337)
        num_samples = 2
        audio_len = 16000
        kwargs.update(
            {
                "num_fft_bins": num_fft_bins,
                "fft_stride": fft_stride,
                "num_mel_bins": num_mel_bins,
            }
        )
        with test_utils.use_gpu():
            test_utils.layer_test(
                audio_preprocessing.MelSpectrogram,
                kwargs=kwargs,
                input_shape=(num_samples, audio_len),
                expected_output_shape=(
                    None,
                    num_mel_bins,
                    int(math.ceil(audio_len / fft_stride)),
                ),
            )

    @parameterized.named_parameters(
        ("audio_to_spec_1", {"sampling_rate": 8000}, 2048, 512, 128),
        ("audio_to_spec_2", {"sampling_rate": 4000}, 1024, 256, 64),
        ("audio_to_spec_3", {"mag_exp": 1.0}, 1024, 256, 64),
        ("audio_to_spec_4", {"power_to_db": False}, 1024, 256, 64),
        ("audio_to_spec_5", {"ref_power": tf.math.reduce_max}, 1024, 256, 64),
        ("audio_to_spec_6", {"min_freq": 0, "max_freq": 8000}, 1024, 256, 64),
    )
    def test_params(self, kwargs, num_fft_bins, fft_stride, num_mel_bins):
        self._run_test(kwargs, num_fft_bins, fft_stride, num_mel_bins)

    def test_config_with_custom_name(self):
        layer = audio_preprocessing.MelSpectrogram(name="audio_to_spec")
        config = layer.get_config()
        layer_1 = audio_preprocessing.MelSpectrogram.from_config(config)
        self.assertEqual(layer_1.name, layer.name)

    def test_all_zeros_audio(self):
        with test_utils.use_gpu():
            audio = np.zeros((2, 16000), dtype="float32")
            layer = audio_preprocessing.MelSpectrogram()
            spec = layer(audio)
            self.assertFalse(tf.math.reduce_any(tf.math.is_inf(spec)))

    def test_unbatched_audio(self):
        with test_utils.use_gpu():
            audio = np.random.uniform(size=(16000,)).astype("float32")
            layer = audio_preprocessing.MelSpectrogram(
                num_mel_bins=128, fft_stride=512
            )
            spec = layer(audio)
            expected_shape = [128, int(math.ceil(16000 / 512))]
            self.assertAllEqual(spec.shape, expected_shape)

    @test_utils.run_v2_only
    def test_output_dtypes(self):
        inputs = np.random.uniform(size=(2, 16000)).astype("float32")
        layer = audio_preprocessing.MelSpectrogram()
        self.assertAllEqual(layer(inputs).dtype, "float32")
        inputs = np.random.uniform(size=(2, 16000)).astype("float64")
        self.assertAllEqual(layer(inputs).dtype, "float32")


if __name__ == "__main__":
    tf.test.main()
