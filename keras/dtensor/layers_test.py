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
"""Tests for layers."""

from keras import backend
from keras import layers
from keras.dtensor import dtensor_api as dtensor
from keras.utils import tf_utils
import numpy as np
import tensorflow.compat.v2 as tf

from keras.dtensor.tests import test_util


class LayersTest(test_util.DTensorBaseTest):

  def setUp(self):
    super(LayersTest, self).setUp()
    backend.enable_tf_random_generator()
    tf_utils.set_random_seed(1337)
    global_ids = test_util.create_device_ids_array((2, 2))
    local_device_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        'CPU':
            dtensor.Mesh(['X', 'Y'], global_ids,
                         local_device_ids,
                         test_util.create_device_list((2, 2), 'CPU'))
    }
    self.mesh = self.configTestMesh(mesh_dict)
    self.layout_4d = dtensor.Layout.replicated(self.mesh, rank=4)
    self.layout_3d = dtensor.Layout.replicated(self.mesh, rank=3)
    self.layout_2d = dtensor.Layout.replicated(self.mesh, rank=2)
    self.layout_1d = dtensor.Layout.replicated(self.mesh, rank=1)

  def test_dense_layer_with_layout(self):
    dense = layers.Dense(10,
                         kernel_layout=self.layout_2d,
                         bias_layout=self.layout_1d)
    inputs = np.random.randint(size=[32, 8], low=0, high=4)
    inputs = tf.constant(inputs, dtype=tf.float32)
    d_inputs = dtensor.copy_to_mesh(
        inputs, dtensor.Layout.replicated(self.mesh, rank=2))

    output = dense(d_inputs)
    self.assertIsInstance(dense.kernel, dtensor.DVariable)
    self.assertIsInstance(dense.bias, dtensor.DVariable)
    expected_layout = dtensor.Layout(
        [dtensor.UNSHARDED, dtensor.UNSHARDED], self.mesh)
    self.assertEqual(dtensor.fetch_layout(output), expected_layout)

    # Make sure to produce same output when layout is not used
    tf_utils.set_random_seed(1337)
    dense_2 = layers.Dense(10)
    output_2 = dense_2(inputs)
    self.assertAllClose(output, output_2)

  def test_dense_layer_without_layout(self):
    # Make sure the layer works as normal with tf.Tensor
    dense = layers.Dense(10)
    inputs = np.random.randint(size=[32, 8], low=0, high=4)
    inputs = tf.constant(inputs, dtype=tf.float32)
    dense(inputs)

    self.assertNotIsInstance(dense.kernel, dtensor.DVariable)
    self.assertNotIsInstance(dense.bias, dtensor.DVariable)

  def test_conv2d_layer_with_layout(self):
    conv = layers.Conv2D(32, kernel_size=(3, 3),
                         kernel_layout=self.layout_4d,
                         bias_layout=self.layout_1d)
    inputs = np.random.randint(size=[10, 28, 28, 1], low=0, high=4)
    inputs = tf.constant(inputs, dtype=tf.float32)
    d_inputs = dtensor.copy_to_mesh(inputs, self.layout_4d)
    output = conv(d_inputs)
    self.assertIsInstance(conv.kernel, dtensor.DVariable)
    self.assertIsInstance(conv.bias, dtensor.DVariable)
    self.assertEqual(dtensor.fetch_layout(output), self.layout_4d)

    # Make sure to produce same output when layout is not used
    tf_utils.set_random_seed(1337)
    conv2 = layers.Conv2D(32, kernel_size=(3, 3))
    output_2 = conv2(inputs)
    self.assertAllClose(output, output_2)

if __name__ == '__main__':
  tf.test.main()
