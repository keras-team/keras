# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import tensorflow as tf

from keras.src import layers
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.testing import TestCase


class AddOneToInputs(BaseImagePreprocessingLayer):
    """Add 1 to all image values, for testing purposes."""

    def __init__(self):
        super().__init__()
        self.call_counter = tf.Variable(initial_value=0)

    def call(self, inputs):
        self.call_counter.assign_add(1)
        return inputs + 1

    def transform_images(self, images, transformation=None, training=True):
        return images + 1

    def transform_labels(self, labels, transformation=None, training=True):
        return labels + 1

    def transform_bounding_boxes(self, bboxes, transformation=None, training=True):
        return bboxes + 1

    def transform_segmentation_masks(self, masks, transformation=None, training=True):
        return masks + 1


class RandomChoiceTest(TestCase):
    def test_calls_layer_augmentation_per_image(self):
        layer = AddOneToInputs()
        pipeline = layers.RandomChoice(layers=[layer])
        xs = tf.random.uniform((2, 5, 5, 3), 0, 100, dtype=tf.float32)
        os = pipeline(xs)

        self.assertAllClose(xs + 1, os)

    @pytest.mark.tf_keras_only
    def test_calls_layer_augmentation_in_graph(self):
        layer = AddOneToInputs()
        pipeline = layers.RandomChoice(layers=[layer])

        @tf.function()
        def call_pipeline(xs):
            return pipeline(xs)

        xs = tf.random.uniform((2, 5, 5, 3), 0, 100, dtype=tf.float32)
        os = call_pipeline(xs)

        self.assertAllClose(xs + 1, os)

    def test_batchwise(self):
        layer = AddOneToInputs()
        pipeline = layers.RandomChoice(layers=[layer], batchwise=True)
        xs = tf.random.uniform((4, 5, 5, 3), 0, 100, dtype=tf.float32)
        os = pipeline(xs)

        self.assertAllClose(xs + 1, os)
        tf.reduce_all(tf.equal(layer.call_counter.numpy(), 1))

    def test_works_with_random_flip(self):
        pipeline = layers.RandomChoice(
            layers=[layers.RandomFlip("vertical", data_format="channels_last", seed=42)], batchwise=True
        )
        xs = tf.random.uniform((4, 5, 5, 3), 0, 100, dtype=tf.float32)
        pipeline(xs)

    def test_calls_layer_augmentation_single_image(self):
        layer = AddOneToInputs()
        pipeline = layers.RandomChoice(layers=[layer])
        xs = tf.random.uniform((5, 5, 3), 0, 100, dtype=tf.float32)
        os = pipeline(xs)

        self.assertAllClose(xs + 1, os)

    def test_calls_choose_one_layer_augmentation(self):
        batch_size = 10
        pipeline = layers.RandomChoice(
            layers=[AddOneToInputs(), AddOneToInputs()]
        )
        xs = tf.random.uniform((batch_size, 5, 5, 3), 0, 100, dtype=tf.float32)
        os = pipeline(xs)

        self.assertAllClose(xs + 1, os)

        total_calls = (
            pipeline.layers[0].call_counter + pipeline.layers[1].call_counter
        )
        tf.reduce_all(tf.equal(total_calls.numpy(), batch_size))