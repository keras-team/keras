# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for np_utils."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras.testing_infra import test_combinations
from keras.utils import np_utils

NUM_CLASSES = 5


class TestNPUtils(test_combinations.TestCase):
    @parameterized.parameters(
        [
            ((1,), (1, NUM_CLASSES)),
            ((3,), (3, NUM_CLASSES)),
            ((4, 3), (4, 3, NUM_CLASSES)),
            ((5, 4, 3), (5, 4, 3, NUM_CLASSES)),
            ((3, 1), (3, NUM_CLASSES)),
            ((3, 2, 1), (3, 2, NUM_CLASSES)),
        ]
    )
    def test_to_categorical(self, shape, expected_shape):
        label = np.random.randint(0, NUM_CLASSES, shape)
        one_hot = np_utils.to_categorical(label, NUM_CLASSES)
        # Check shape
        self.assertEqual(one_hot.shape, expected_shape)
        # Make sure there is only one 1 in a row
        self.assertTrue(np.all(one_hot.sum(axis=-1) == 1))
        # Get original labels back from one hots
        self.assertTrue(
            np.all(np.argmax(one_hot, -1).reshape(label.shape) == label)
        )

    def test_to_categorial_without_num_classes(self):
        label = [0, 2, 5]
        one_hot = np_utils.to_categorical(label)
        self.assertEqual(one_hot.shape, (3, 5 + 1))

    @parameterized.parameters(
        [
            ((1,), (1, NUM_CLASSES - 1)),
            ((3,), (3, NUM_CLASSES - 1)),
            ((4, 3), (4, 3, NUM_CLASSES - 1)),
            ((5, 4, 3), (5, 4, 3, NUM_CLASSES - 1)),
            ((3, 1), (3, NUM_CLASSES - 1)),
            ((3, 2, 1), (3, 2, NUM_CLASSES - 1)),
        ]
    )
    def test_to_ordinal(self, shape, expected_shape):
        label = np.random.randint(0, NUM_CLASSES, shape)
        ordinal = np_utils.to_ordinal(label, NUM_CLASSES)
        # Check shape
        self.assertEqual(ordinal.shape, expected_shape)
        # Make sure all the values are either 0 or 1
        self.assertTrue(np.all(np.logical_or(ordinal == 0, ordinal == 1)))
        # Get original labels back from ordinal matrix
        self.assertTrue(
            np.all(ordinal.cumprod(-1).sum(-1).reshape(label.shape) == label)
        )

    def test_to_ordinal_without_num_classes(self):
        label = [0, 2, 5]
        one_hot = np_utils.to_ordinal(label)
        self.assertEqual(one_hot.shape, (3, 5))


if __name__ == "__main__":
    tf.test.main()
