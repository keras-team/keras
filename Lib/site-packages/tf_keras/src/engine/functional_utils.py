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
# ==============================================================================
"""Utilities for keras functional model."""

import tensorflow.compat.v2 as tf

from tf_keras.src import backend
from tf_keras.src.engine import input_layer as input_layer_module
from tf_keras.src.engine import keras_tensor
from tf_keras.src.engine import node as node_module

_KERAS_TENSOR_TYPE_CHECK_ERROR_MSG = (
    "Found unexpected instance while processing input tensors for keras "
    "functional model. Expecting KerasTensor which is from tf.keras.Input() "
    "or output from keras layer call(). Got: {}"
)


def is_input_keras_tensor(tensor):
    """Check if tensor is directly generated from `tf.keras.Input`.

    This check is useful when constructing the functional model, since we will
    need to clone Nodes and KerasTensors if the model is building from non input
    tensor.

    Args:
      tensor: A `KerasTensor` as inputs to the functional model.

    Returns:
      bool. Whether the tensor is directly generated from `tf.keras.Input`.

    Raises:
      ValueError: if the tensor is not a KerasTensor instance.
    """
    if not node_module.is_keras_tensor(tensor):
        raise ValueError(_KERAS_TENSOR_TYPE_CHECK_ERROR_MSG.format(tensor))
    return tensor.node.is_input


def find_nodes_by_inputs_and_outputs(inputs, outputs):
    """Fetch all Nodes in the graph defined by "inputs" and "outputs".

    This method is used to find and then clone Nodes when creating a new
    sub-model from an existing functional model.

    Args:
      inputs: A nested structure of KerasTensor to use as model inputs.
      outputs: A nested structure of KerasTensor to use as model outputs.

    Returns:
      A list of Nodes that are connected to the inputs and outputs.

    Raises:
      ValueError: when inputs and outputs are disconnected or in case of
        unexpected objects in the inputs/outputs.
    """
    # We walk the graph bottom up, starting from output nodes, and keep tracing
    # the upstream node, until we find all the inputs nodes. We don't use top
    # down search here since we don't know whether a certain node is in the
    # graph between inputs and outputs, e.g. a functional graph could have
    # multiple outputs, and the user could choose a subset of them to build the
    # model. The bottom up approach will ensure all the nodes we visit are
    # actually in use. If we reach the top and didn't find the nodes in the
    # `inputs`, that's an error, since the user didn't specify the correct
    # inputs.
    start_keras_tensors = tf.nest.flatten(outputs)
    end_keras_tensors = tf.nest.flatten(inputs)

    for t in start_keras_tensors + end_keras_tensors:
        if not node_module.is_keras_tensor(t):
            raise ValueError(_KERAS_TENSOR_TYPE_CHECK_ERROR_MSG.format(t))
    end_ids = set([id(kt) for kt in end_keras_tensors])
    # Track all the end tensors we found so far, if we didn't reach all the
    # user-specified keras inputs after we finish the search, then that's an
    # error since the inputs are disconnected from the outputs.
    end_ids_found = set()

    nodes_to_visit = []
    nodes_in_graph = []
    node_id_visited = set()
    for t in start_keras_tensors:
        nodes_to_visit.append(t.node)

    while nodes_to_visit:
        node = nodes_to_visit.pop(0)
        if id(node) in node_id_visited:
            continue
        node_id_visited.add(id(node))
        nodes_in_graph.append(node)
        # Any input keras_tensor that produce the current node.
        for kt in node.keras_inputs:
            if id(kt) in end_ids:
                # We found the inputs of the model, stop tracing upstream nodes
                end_ids_found.add(id(kt))
                continue

            inbound_node = kt.node
            # In case this is the tf.keras.Input node, we have reached the end
            # of the tracing of upstream nodes. Any further tracing will just be
            # an infinite loop. we should raise an error here since we didn't
            # find the input in the user-specified inputs.
            if inbound_node.is_input:
                raise ValueError(
                    "Found input tensor cannot be reached given provided "
                    "output tensors. Please make sure the tensor {} is "
                    "included in the model inputs when building "
                    "functional model.".format(kt)
                )
            nodes_to_visit.append(inbound_node)

    # Do a final check and make sure we have reached all the user-specified
    # inputs
    if end_ids != end_ids_found:
        unvisited_inputs = [
            kt for kt in end_keras_tensors if id(kt) not in end_ids_found
        ]
        raise ValueError(
            "Found unvisited input tensors that are disconnected from "
            "the outputs: {}".format(unvisited_inputs)
        )
    return nodes_in_graph


def clone_graph_nodes(inputs, outputs):
    """Clone the `Node` between the inputs and output tensors.

    This function is used to create a new functional model from any intermediate
    keras tensors. The clone of the nodes mimic the behavior of reconstructing
    the functional graph network by re-executing all the __call__ methods. The
    cloned nodes will be appended to the layers.

    Note that a new tf.keras.Inputs will be created for any items in the
    `inputs`

    Args:
      inputs: A nested structure of keras_tensors.
      outputs: A nested structure of keras_tensors.

    Returns:
      A pair of inputs and outputs, with cloned keras_tensors. They can be used
      to create a new functional model.
    """
    nodes_to_clone = find_nodes_by_inputs_and_outputs(inputs, outputs)
    cloned_inputs = []
    cloned_outputs = []
    # We not only need to create copies of Nodes (mimic the calls), also need to
    # clone keras_tensors to avoid the override of _keras_history attached on
    # the keras_tensor. The following dict is used to track any keras tensor we
    # cloned The key is the string ID of the original keras tensor, and value is
    # the cloned keras_tensor instance.
    kt_id_mapping = {}

    for kt_input in tf.nest.flatten(inputs):
        if kt_input.node.is_input:
            # For any existing keras_tensor from tf.keras.Input, we leave them
            # as is.
            cloned_inputs.append(kt_input)
            kt_id_mapping[id(kt_input)] = kt_input
        else:
            # We need to create a new tf.keras.Input for any intermediate
            # keras_tensor
            cpy = _clone_keras_tensor(kt_input)
            cloned_input = input_layer_module.Input(tensor=cpy)
            cloned_inputs.append(cloned_input)
            kt_id_mapping[id(kt_input)] = cloned_input
    cloned_inputs = tf.nest.pack_sequence_as(inputs, cloned_inputs)

    for kt_output in tf.nest.flatten(outputs):
        cpy = _clone_keras_tensor(kt_output)
        # We reuse the _keras_history here, which contains the old information.
        # It is used in the Node constructor to check if the tensor
        # "is_keras_tensor()" The history will be override by the Node
        # constructor anyway for the corresponding layer output anyway.
        cpy._keras_history = kt_output._keras_history
        cloned_outputs.append(cpy)
        kt_id_mapping[id(kt_output)] = cpy
    cloned_outputs = tf.nest.pack_sequence_as(outputs, cloned_outputs)

    for node in nodes_to_clone:
        # Clone any keras_tensors to avoid override of _keras_history
        # Or reuse an existing keras_tensor if it has already been cloned.
        output_copy = clone_keras_tensors(node.output_tensors, kt_id_mapping)
        call_args_copy = clone_keras_tensors(node.call_args, kt_id_mapping)
        call_kwargs_copy = clone_keras_tensors(node.call_kwargs, kt_id_mapping)
        # Creating new nodes based on the existing node information.  Node wires
        # itself to inbound and outbound layers.  The Node constructor actually
        # updates this layer's self._inbound_nodes, sets _keras_history on the
        # outputs, and adds itself to the `_outbound_nodes` of the layers that
        # produced the inputs to this layer call.
        node_module.Node(
            node.layer,
            call_args=call_args_copy,
            call_kwargs=call_kwargs_copy,
            outputs=output_copy,
        )
    return cloned_inputs, cloned_outputs


def clone_keras_tensors(args, keras_tensor_mapping):
    """Clone the keras tensors from the inputs.

    For any KerasTensor instance in the `args`, a new copy of KerasTensor will
    be created if it has not been cloned yet (by checking the
    `keras_tensor_mapping`). For any other types, the instance will be
    unchanged. This function is useful for cloning the Nodes since KerasTensor
    can't be reused across the models.

    Args:
      args: A nested structure of objects, which could contain KerasTensor.
      keras_tensor_mapping: A dict contains the ID of original KerasTensor, and
        the cloned KerasTensor instance. The dict will be updated with newly
        copied KerasTensor instances within this method.
    Returns:
      Same structure as inputs, with KerasTensor cloned.
    """
    result = []
    for obj in tf.nest.flatten(args):
        if node_module.is_keras_tensor(obj):
            if id(obj) in keras_tensor_mapping:
                cpy = keras_tensor_mapping[id(obj)]
            else:
                # Create copy of keras_tensor if we haven't done it before
                cpy = _clone_keras_tensor(obj)
                cpy._keras_history = obj._keras_history
                keras_tensor_mapping[id(obj)] = cpy
            result.append(cpy)
        else:
            result.append(obj)
    return tf.nest.pack_sequence_as(args, result)


def _clone_keras_tensor(kt):
    """Create an identical keras_tensor based on the input.

    We use keras_tensor_to_placeholder and keras_tensor_from_tensor to make sure
    inferred shape are not lost during the copy.

    Args:
      kt: the input KerasTensor.

    Returns:
      An identical copy of the input KerasTensor.
    """
    # Create a scratch graph since we don't intend to use the placeholders.
    with backend._scratch_graph() as scratch_graph:
        with scratch_graph.as_default():
            placeholder = keras_tensor.keras_tensor_to_placeholder(kt)
            return keras_tensor.keras_tensor_from_tensor(placeholder)

