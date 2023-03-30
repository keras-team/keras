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

import functools

import numpy as np
import math
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.engine import sequential
from keras.layers.preprocessing import audio_preprocessing
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils

# isort: off
from tensorflow.python.ops import stateless_random_ops


@test_combinations.run_all_keras_modes(always_skip_v1=True)
class MelSpectrogramTest(test_combinations.TestCase):
    def _run_test(self, kwargs, num_fft_bins, fft_stride, num_mel_bins):
        np.random.seed(1337)
        num_samples = 2
        audio_len = 16000
        kwargs.update({
            "num_fft_bins": num_fft_bins,
            "fft_stride": fft_stride,
            "num_mel_bins": num_mel_bins,
            })
        with test_utils.use_gpu():
            test_utils.layer_test(
                audio_preprocessing.MelSpectrogram,
                kwargs=kwargs,
                input_shape=(num_samples, audio_len),
                expected_output_shape=(
                    None,
                    num_mel_bins,
                    int(math.ceil(audio_len / fft_stride))
                ),
            )