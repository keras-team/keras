# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
#,============================================================================
"""Tests for functional_utils."""

from keras import keras_parameterized
from keras import layers
from keras import models
from keras.engine import functional_utils
from keras.engine import input_layer as input_layer_lib

import numpy as np
import tensorflow.compat.v2 as tf


class FunctionalModelSlideTest(keras_parameterized.TestCase):

  def testfind_nodes_by_inputs_and_outputs(self):
    inputs = input_layer_lib.Input((10,))
    unconnected_inputs = input_layer_lib.Input((10,))
    x = layers.Dense(8)(inputs)
    y = layers.Dense(6)(x)
    output = layers.Dense(4)(y)

    nodes_in_graph = functional_utils.find_nodes_by_inputs_and_outputs(
        x, output)
    self.assertLen(nodes_in_graph, 2)
    expected_nodes = [output.node, y.node]
    self.assertCountEqual(nodes_in_graph, expected_nodes)

    # Make sure we raise error if we specify invalid input/output pair
    with self.assertRaisesRegex(
        ValueError, 'Found input tensor cannot be reached'):
      functional_utils.find_nodes_by_inputs_and_outputs(output, x)

    with self.assertRaisesRegex(
        ValueError, 'Found input tensor cannot be reached'):
      functional_utils.find_nodes_by_inputs_and_outputs(unconnected_inputs,
                                                        output)

    with self.assertRaisesRegex(
        ValueError, 'Found unvisited input tensors that are disconnected'):
      functional_utils.find_nodes_by_inputs_and_outputs(
          [inputs, unconnected_inputs], output)

  def testfind_nodes_by_inputs_and_outputs_with_complicated_network(self):
    input1 = input_layer_lib.Input((10,))
    input2 = input_layer_lib.Input((10,))
    input3 = input_layer_lib.Input((10,))
    unconnected_input = input_layer_lib.Input((10,))

    dense1 = layers.Dense(4, name='dense1')
    dense2 = layers.Dense(4, name='dense2')
    # dense1 are shared between input1 and input2
    a = dense1(input1)
    b = dense1(input2)

    c = layers.Add()([a, b])
    d = dense2(input3)
    e = layers.Add()([c, d])
    # There are 5 nodes (invoke of __call__) in the graph.

    nodes = functional_utils.find_nodes_by_inputs_and_outputs(input1, a)
    self.assertCountEqual(nodes, [a.node])

    nodes = functional_utils.find_nodes_by_inputs_and_outputs(input2, b)
    self.assertCountEqual(nodes, [b.node])

    nodes = functional_utils.find_nodes_by_inputs_and_outputs([input2, input1],
                                                              c)
    # This should contains 2 dense call and 1 add
    self.assertCountEqual(nodes, [a.node, b.node, c.node])

    # Missing input3
    with self.assertRaisesRegex(
        ValueError, 'Found input tensor cannot be reached'):
      functional_utils.find_nodes_by_inputs_and_outputs([input1, input2], e)

    nodes = functional_utils.find_nodes_by_inputs_and_outputs(
        [input1, input2, input3], e)
    self.assertCountEqual(nodes, [a.node, b.node, c.node, d.node, e.node])

    # Make sure we can create from intermediate tensors
    nodes = functional_utils.find_nodes_by_inputs_and_outputs([a, b, input3], e)
    self.assertCountEqual(nodes, [c.node, d.node, e.node])
    # Also make sure we can add intermediate outputs
    nodes = functional_utils.find_nodes_by_inputs_and_outputs([a, b, input3],
                                                              [d, e])
    self.assertCountEqual(nodes, [c.node, d.node, e.node])

    # input1 and 2 are not needed for computing d
    with self.assertRaisesRegex(
        ValueError, 'Found unvisited input tensors that are disconnected'):
      functional_utils.find_nodes_by_inputs_and_outputs(
          [input1, input2, input3], d)

    with self.assertRaisesRegex(
        ValueError, 'Found unvisited input tensors that are disconnected'):
      functional_utils.find_nodes_by_inputs_and_outputs(
          [a, b, input3, unconnected_input], [e, d, c])

  def test_build_model_from_intermediate_tensor(self):
    batch_size = 4
    inputs = input_layer_lib.Input(shape=(8,))
    layer1 = layers.Dense(32)
    layer2 = layers.Dense(16)
    x = layer1(inputs)
    y = layer2(x)
    cloned_inputs, cloned_outputs = functional_utils.clone_graph_nodes(x, y)
    # Make sure the inputs and outputs are cloned.
    self.assertIsNot(x, cloned_inputs)
    self.assertIsNot(y, cloned_outputs)
    # Make sure a new node is attached to layer2, which mimic y = layer2(x)
    self.assertLen(layer2.inbound_nodes, 2)

    model = models.Model(cloned_inputs, cloned_outputs)
    self.assertIsInstance(model, models.Model)

    model.compile('rmsprop', 'mse')
    model.fit(np.random.randn(batch_size, 32), np.random.randn(batch_size, 16))


if __name__ == '__main__':
  tf.test.main()
