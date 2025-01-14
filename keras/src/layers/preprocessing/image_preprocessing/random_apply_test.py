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
from absl.testing import parameterized

import keras.src.random as random
from keras.src import layers
from keras.src import ops
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.testing import TestCase


class ZeroOut(BaseImagePreprocessingLayer):
    """Layer that zeros out tensors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.built = True

    def call(self, inputs):
        return ops.zeros_like(inputs)

    def transform_images(self, images, transformation=None, training=True):
        return ops.zeros_like(images)

    def transform_segmentation_masks(
        self, masks, transformation=None, training=True
    ):
        return ops.zeros_like(masks)

    def transform_bounding_boxes(
        self, bboxes, transformation=None, training=True
    ):
        return ops.zeros_like(bboxes)

    def transform_labels(self, labels, transformation=None, training=True):
        return ops.zeros_like(labels)

    def get_config(self):
        return super().get_config()


class RandomApplyTest(TestCase):
    @parameterized.parameters([-0.5, 1.7])
    def test_raises_error_on_invalid_rate_parameter(self, invalid_rate):
        with self.assertRaises(ValueError):
            layers.RandomApply(rate=invalid_rate, layer=ZeroOut())

    def test_works_with_batched_input(self):
        batch_size = 32
        dummy_inputs = random.uniform(shape=(batch_size, 224, 224, 3))
        layer = layers.RandomApply(rate=0.5, layer=ZeroOut(), seed=1234)

        outputs = layer(dummy_inputs)
        num_zero_inputs = layers.RandomApply._num_zero_batches(dummy_inputs)
        num_zero_outputs = layers.RandomApply._num_zero_batches(outputs)

        self.assertEqual(num_zero_inputs, 0)
        self.assertLess(num_zero_outputs, batch_size)
        self.assertGreater(num_zero_outputs, 0)

    def test_works_with_batchwise_layers(self):
        batch_size = 32
        dummy_inputs = random.uniform(shape=(batch_size, 224, 224, 3))
        random_flip_layer = layers.RandomFlip(
            "vertical", data_format="channels_last", seed=42
        )
        layer = layers.RandomApply(random_flip_layer, rate=0.5, batchwise=True)
        outputs = layer(dummy_inputs)
        self.assertEqual(outputs.shape, dummy_inputs.shape)

    def test_inputs_unchanged_with_zero_rate(self):
        dummy_inputs = random.uniform(shape=(32, 224, 224, 3))
        layer = layers.RandomApply(rate=0.0, layer=ZeroOut())

        outputs = layer(dummy_inputs)
        self.assertAllClose(outputs, dummy_inputs)

    def test_all_inputs_changed_with_rate_equal_to_one(self):
        dummy_inputs = random.uniform(shape=(32, 224, 224, 3))
        layer = layers.RandomApply(rate=1.0, layer=ZeroOut())
        outputs = layer(dummy_inputs)
        self.assertTrue(
            ops.all(ops.equal(outputs, ops.zeros_like(dummy_inputs)))
        )

    def test_works_with_single_image(self):
        dummy_inputs = random.uniform(shape=(224, 224, 3))
        layer = layers.RandomApply(rate=1.0, layer=ZeroOut())
        outputs = layer(dummy_inputs)
        self.assertTrue(
            ops.all(ops.equal(outputs, ops.zeros_like(dummy_inputs)))
        )

    def test_can_modify_label(self):
        dummy_inputs = random.uniform(shape=(32, 224, 224, 3))
        dummy_labels = ops.ones(shape=(32, 2))
        layer = layers.RandomApply(rate=1.0, layer=ZeroOut())
        outputs = layer({"images": dummy_inputs, "labels": dummy_labels})
        self.assertTrue(
            ops.all(ops.equal(outputs["labels"], ops.zeros_like(dummy_labels)))
        )

    @pytest.mark.skipif(
        ops.backend.backend() != "tensorflow",
        reason="XLA compilation is only supported with TensorFlow backend",
    )
    def test_works_with_xla(self):
        dummy_inputs = random.uniform(shape=(32, 224, 224, 3))
        layer = layers.RandomApply(
            rate=0.5, layer=ZeroOut(), auto_vectorize=False
        )

        def apply(x):
            return layer(x)

        outputs = apply(dummy_inputs)
        apply(outputs)
