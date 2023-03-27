# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for audio_dataset when tfio is available."""

import os
import shutil

import numpy as np
import tensorflow.compat.v2 as tf

from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import audio_dataset


@test_utils.run_v2_only
class AudioDatasetFromDirectoryWithTfioTest(test_combinations.TestCase):
    def _get_audio_samples(self, count=16, different_sequence_lengths=False):
        sequence_length = 30
        num_channels = 1
        audio_samples = []
        for _ in range(count):
            if different_sequence_lengths:
                random_sequence_length = np.random.randint(
                    10, sequence_length + 1
                )
                audio = np.random.random((random_sequence_length, num_channels))
            else:
                audio = np.random.random((sequence_length, num_channels))
            audio_samples.append(tf.audio.encode_wav(audio, 1000))
        return audio_samples

    def _prepare_directory(
        self,
        num_classes=2,
        nested_dirs=False,
        count=16,
        different_sequence_lengths=False,
    ):
        # Get a unique temp directory
        temp_dir = os.path.join(
            self.get_temp_dir(), str(np.random.randint(1e6))
        )
        os.mkdir(temp_dir)
        self.addCleanup(shutil.rmtree, temp_dir)

        # Generate paths to class subdirectories
        paths = []
        for class_index in range(num_classes):
            class_directory = f"class_{class_index}"
            if nested_dirs:
                class_paths = [
                    class_directory,
                    os.path.join(class_directory, "subfolder_1"),
                    os.path.join(class_directory, "subfolder_2"),
                    os.path.join(
                        class_directory, "subfolder_1", "sub-subfolder"
                    ),
                ]
            else:
                class_paths = [class_directory]
            for path in class_paths:
                os.mkdir(os.path.join(temp_dir, path))
            paths += class_paths

        # Save audio samples to the paths
        i = 0
        for audio in self._get_audio_samples(
            count=count, different_sequence_lengths=different_sequence_lengths
        ):
            path = paths[i % len(paths)]
            ext = "wav"
            filename = os.path.join(path, f"audio_{i}.{ext}")
            with open(os.path.join(temp_dir, filename), "wb") as f:
                f.write(audio.numpy())
            i += 1
        return temp_dir

    def test_audio_dataset_from_directory_standalone_with_resampling(self):
        # Test retrieving audio samples withouts labels from a directory and its
        # subdirs where we double the sampling rate.
        # Save a few extra audio in the parent directory.
        directory = self._prepare_directory(count=7, num_classes=2)
        for i, audio in enumerate(self._get_audio_samples(3)):
            filename = f"audio_{i}.wav"
            with open(os.path.join(directory, filename), "wb") as f:
                f.write(audio.numpy())

        dataset = audio_dataset.audio_dataset_from_directory(
            directory,
            batch_size=5,
            output_sequence_length=30,
            labels=None,
            sampling_rate=2000,  # Twice the original sample rate.
        )
        batch = next(iter(dataset))
        # We return plain audio. Expect twice as many samples now.
        self.assertEqual(batch.shape, (5, 60, 1))
        self.assertEqual(batch.dtype.name, "float32")
        # Count samples
        batch_count = 0
        sample_count = 0
        for batch in dataset:
            batch_count += 1
            sample_count += batch.shape[0]
        self.assertEqual(batch_count, 2)
        self.assertEqual(sample_count, 10)


if __name__ == "__main__":
    try:
        import tensorflow_io  # noqa: F401

        # Only run these tests if tensorflow_io is installed.
        tf.test.main()
    except ImportError:
        pass
