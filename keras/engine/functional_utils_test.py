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

import collections
import os

from keras import keras_parameterized
from keras import layers
from keras import models
from keras.engine import functional_utils
from keras.engine import input_layer as input_layer_lib

import numpy as np
import tensorflow.compat.v2 as tf


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class FunctionalModelSlideTest(keras_parameterized.TestCase):

  def test_find_nodes_by_inputs_and_outputs(self):
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

  def test_find_nodes_by_inputs_and_outputs_with_complicated_network(self):
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
    model = models.Model(x, y)
    # Make sure a new node is attached to layer2, which mimic y = layer2(x)
    self.assertLen(layer2.inbound_nodes, 2)

    self.assertIsInstance(model, models.Model)
    # The model only contains 1 dense layer and 1 input layer.
    self.assertLen(model.layers, 2)
    self.assertIs(model.layers[1], layer2)

    model.compile('rmsprop', 'mse')
    model.fit(np.random.randn(batch_size, 32), np.random.randn(batch_size, 16))
    # Test for model saving
    output_path = os.path.join(self.get_temp_dir(), 'tf_keras_saved_model')
    model.save(output_path, save_format='tf')
    loaded_model = models.load_model(output_path)
    self.assertEqual(model.summary(), loaded_model.summary())

    # Also make sure the orignal inputs and y can still be used to build model
    new_model = models.Model(inputs, y)
    # Make sure no new node is attached to layer2
    self.assertLen(layer2.inbound_nodes, 2)

    self.assertLen(new_model.layers, 3)
    self.assertIs(new_model.layers[1], layer1)
    self.assertIs(new_model.layers[2], layer2)

  def test_build_model_from_intermediate_tensor_with_complicated_model(self):
    # The topology is like below:
    # input1 -> dense1 -> a
    #                     + -> c - + --> d - + --> output
    # input2 -> dense1 -> b -------^         ^
    # input3 -> dense2 -> e -----------------|
    batch_size = 8
    input1 = input_layer_lib.Input((2,))
    input2 = input_layer_lib.Input((2,))
    input3 = input_layer_lib.Input((8,))

    dense1 = layers.Dense(8, name='dense1')
    dense2 = layers.Dense(8, name='dense2')

    # dense1 are shared between input1 and input2
    a = dense1(input1)
    b = dense1(input2)

    c = layers.Add()([a, b])
    # d has a residual connection from b.
    d = layers.Add()([b, c])
    e = dense2(input3)
    output = layers.Add()([d, e])

    # We skip the input2 here and use b instead.
    model = models.Model([input1, b, input3], output)
    # Make sure we have 8 layers, 3 for inputs, 2 for dense and 3 for Add.
    # Note that dense1 is still in use by input1.
    self.assertLen(model.layers, 8)
    # Since the layers are not ordered, let's check class of the layers to make
    # sure it match the expectation.
    class_count = collections.Counter([l.__class__ for l in model.layers])
    self.assertEqual(class_count[input_layer_lib.InputLayer], 3)
    self.assertEqual(class_count[layers.Dense], 2)
    self.assertEqual(class_count[layers.Add], 3)

    model.compile('rmsprop', 'mse')
    model.fit([np.random.randn(batch_size, 2),
               np.random.randn(batch_size, 8),  # The shape of b is (batch, 8)
               np.random.randn(batch_size, 8)],
              np.random.randn(batch_size, 8))
    output_path = os.path.join(self.get_temp_dir(), 'tf_keras_saved_model')
    model.save(output_path, save_format='tf')
    loaded_model = models.load_model(output_path)
    self.assertEqual(model.summary(), loaded_model.summary())

    model2 = models.Model([a, b], d)
    # 2 input layers and 2 Add layer.
    self.assertLen(model2.layers, 4)
    class_count = collections.Counter([l.__class__ for l in model2.layers])
    self.assertEqual(class_count[input_layer_lib.InputLayer], 2)
    self.assertEqual(class_count[layers.Add], 2)

    model2.compile('rmsprop', 'mse')
    model2.fit([np.random.randn(batch_size, 8),
                np.random.randn(batch_size, 8)],
               np.random.randn(batch_size, 8))


if __name__ == '__main__':
  tf.test.main()
