# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for image_utils."""

from absl.testing import parameterized
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import image_utils
import numpy as np
import tensorflow.compat.v2 as tf


@test_utils.run_v2_only
class TestImageUtils(test_combinations.TestCase):

  def test_smart_resize(self):
    test_input = np.random.random((20, 40, 3))
    output = image_utils.smart_resize(test_input, size=(50, 50))
    self.assertIsInstance(output, np.ndarray)
    self.assertListEqual(list(output.shape), [50, 50, 3])
    output = image_utils.smart_resize(test_input, size=(10, 10))
    self.assertListEqual(list(output.shape), [10, 10, 3])
    output = image_utils.smart_resize(test_input, size=(100, 50))
    self.assertListEqual(list(output.shape), [100, 50, 3])
    output = image_utils.smart_resize(test_input, size=(5, 15))
    self.assertListEqual(list(output.shape), [5, 15, 3])

  @parameterized.named_parameters(('size1', (50, 50)), ('size2', (10, 10)),
                                  ('size3', (100, 50)), ('size4', (5, 15)))
  def test_smart_resize_tf_dataset(self, size):
    test_input_np = np.random.random((2, 20, 40, 3))
    test_ds = tf.data.Dataset.from_tensor_slices(test_input_np)

    resize = lambda img: image_utils.smart_resize(img, size=size)
    test_ds = test_ds.map(resize)
    for sample in test_ds.as_numpy_iterator():
      self.assertIsInstance(sample, np.ndarray)
      self.assertListEqual(list(sample.shape), [size[0], size[1], 3])

  def test_smart_resize_batch(self):
    img = np.random.random((2, 20, 40, 3))
    out = image_utils.smart_resize(img, size=(20, 20))
    self.assertListEqual(list(out.shape), [2, 20, 20, 3])
    self.assertAllClose(out, img[:, :, 10:-10, :])

  def test_smart_resize_errors(self):
    with self.assertRaisesRegex(ValueError, 'a tuple of 2 integers'):
      image_utils.smart_resize(np.random.random((20, 20, 2)), size=(10, 5, 3))
    with self.assertRaisesRegex(ValueError, 'incorrect rank'):
      image_utils.smart_resize(np.random.random((2, 4)), size=(10, 5))
    with self.assertRaisesRegex(ValueError, 'incorrect rank'):
      image_utils.smart_resize(np.random.random((2, 4, 4, 5, 3)), size=(10, 5))


if __name__ == '__main__':
  tf.test.main()
