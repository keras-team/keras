# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmark for TF-Keras image preprocessing layer."""

import functools
import time

import numpy as np
import tensorflow.compat.v2 as tf

import tf_keras.src as keras
from tf_keras.src.layers.preprocessing import image_preprocessing

LOWER = 0.2
UPPER = 0.4
BATCH_SIZE = 32


def rotate(inputs):
    """rotate image."""
    inputs_shape = tf.shape(inputs)
    batch_size = inputs_shape[0]
    img_hd = tf.cast(inputs_shape[1], tf.float32)
    img_wd = tf.cast(inputs_shape[2], tf.float32)
    min_angle = LOWER * 2.0 * np.pi
    max_angle = UPPER * 2.0 * np.pi
    angles = tf.random.uniform(
        shape=[batch_size], minval=min_angle, maxval=max_angle
    )
    return image_preprocessing.transform(
        inputs, image_preprocessing.get_rotation_matrix(angles, img_hd, img_wd)
    )


def zoom(inputs):
    """zoom image."""
    inputs_shape = tf.shape(inputs)
    batch_size = inputs_shape[0]
    img_hd = tf.cast(inputs_shape[1], tf.float32)
    img_wd = tf.cast(inputs_shape[2], tf.float32)
    height_zoom = tf.random.uniform(
        shape=[batch_size, 1], minval=1.0 + LOWER, maxval=1.0 + UPPER
    )
    width_zoom = tf.random.uniform(
        shape=[batch_size, 1], minval=1.0 + LOWER, maxval=1.0 + UPPER
    )
    zooms = tf.cast(
        tf.concat([width_zoom, height_zoom], axis=1), dtype=tf.float32
    )
    return image_preprocessing.transform(
        inputs, image_preprocessing.get_zoom_matrix(zooms, img_hd, img_wd)
    )


def image_augmentation(inputs, batch_size):
    """image augmentation."""
    img = inputs
    img = tf.image.resize(img, size=[224, 224])
    img = tf.image.random_crop(img, size=[batch_size, 224, 224, 3])
    img = rotate(img)
    img = zoom(img)
    return img


class BenchmarkLayer(tf.test.Benchmark):
    """Benchmark the layer forward pass."""

    def run_dataset_implementation(self, batch_size):
        num_repeats = 5
        starts = []
        ends = []
        for _ in range(num_repeats):
            ds = tf.data.Dataset.from_tensor_slices(
                np.random.random((batch_size, 256, 256, 3))
            )
            ds = ds.shuffle(batch_size * 100)
            ds = ds.batch(batch_size)
            ds = ds.prefetch(batch_size)
            img_augmentation = functools.partial(
                image_augmentation, batch_size=batch_size
            )
            ds = ds.map(img_augmentation, num_parallel_calls=8)
            starts.append(time.time())
            count = 0
            # Benchmarked code begins here.
            for i in ds:
                _ = i
                count += 1
            # Benchmarked code ends here.
            ends.append(time.time())

        avg_time = np.mean(np.array(ends) - np.array(starts)) / count
        return avg_time

    def bm_layer_implementation(self, batch_size):
        with tf.device("/gpu:0"):
            img = keras.Input(shape=(256, 256, 3), dtype=tf.float32)
            preprocessor = keras.Sequential(
                [
                    image_preprocessing.Resizing(224, 224),
                    image_preprocessing.RandomCrop(height=224, width=224),
                    image_preprocessing.RandomRotation(factor=(0.2, 0.4)),
                    image_preprocessing.RandomFlip(mode="horizontal"),
                    image_preprocessing.RandomZoom(0.2, 0.2),
                ]
            )
            _ = preprocessor(img)

            num_repeats = 5
            starts = []
            ends = []
            for _ in range(num_repeats):
                ds = tf.data.Dataset.from_tensor_slices(
                    np.random.random((batch_size, 256, 256, 3))
                )
                ds = ds.shuffle(batch_size * 100)
                ds = ds.batch(batch_size)
                ds = ds.prefetch(batch_size)
                starts.append(time.time())
                count = 0
                # Benchmarked code begins here.
                for i in ds:
                    _ = preprocessor(i)
                    count += 1
                # Benchmarked code ends here.
                ends.append(time.time())

        avg_time = np.mean(np.array(ends) - np.array(starts)) / count
        name = f"image_preprocessing|batch_{batch_size}"
        baseline = self.run_dataset_implementation(batch_size)
        extras = {
            "dataset implementation baseline": baseline,
            "delta seconds": (baseline - avg_time),
            "delta percent": ((baseline - avg_time) / baseline) * 100,
        }
        self.report_benchmark(
            iters=num_repeats, wall_time=avg_time, extras=extras, name=name
        )

    def benchmark_vocab_size_by_batch(self):
        for batch in [32, 64, 256]:
            self.bm_layer_implementation(batch_size=batch)


if __name__ == "__main__":
    tf.test.main()

