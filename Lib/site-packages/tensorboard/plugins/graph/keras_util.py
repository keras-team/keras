# -*- coding: utf-8 -*-
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for handling Keras model in graph plugin.

Two canonical types of Keras model are Functional and Sequential.
A model can be serialized as JSON and deserialized to reconstruct a model.
This utility helps with dealing with the serialized Keras model.

They have distinct structures to the configurations in shapes below:
Functional:
  config
    name: Name of the model. If not specified, it is 'model' with
          an optional suffix if there are more than one instance.
    input_layers: Keras.layers.Inputs in the model.
    output_layers: Layer names that are outputs of the model.
    layers: list of layer configurations.
      layer: [*]
        inbound_nodes: inputs to this layer.

Sequential:
  config
    name: Name of the model. If not specified, it is 'sequential' with
          an optional suffix if there are more than one instance.
    layers: list of layer configurations.
      layer: [*]

[*]: Note that a model can be a layer.
Please refer to https://github.com/tensorflow/tfjs-layers/blob/master/src/keras_format/model_serialization.ts
for more complete definition.
"""
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.tensorflow_stub import dtypes
from tensorboard.util import tb_logging


logger = tb_logging.get_logger()


def _walk_layers(keras_layer):
    """Walks the nested keras layer configuration in preorder.

    Args:
      keras_layer: Keras configuration from model.to_json.

    Yields:
      A tuple of (name_scope, layer_config).
      name_scope: a string representing a scope name, similar to that of tf.name_scope.
      layer_config: a dict representing a Keras layer configuration.
    """
    yield ("", keras_layer)
    if keras_layer.get("config").get("layers"):
        name_scope = keras_layer.get("config").get("name")
        for layer in keras_layer.get("config").get("layers"):
            for sub_name_scope, sublayer in _walk_layers(layer):
                sub_name_scope = (
                    "%s/%s" % (name_scope, sub_name_scope)
                    if sub_name_scope
                    else name_scope
                )
                yield (sub_name_scope, sublayer)


def _scoped_name(name_scope, node_name):
    """Returns scoped name for a node as a string in the form '<scope>/<node
    name>'.

    Args:
      name_scope: a string representing a scope name, similar to that of tf.name_scope.
      node_name: a string representing the current node name.

    Returns
      A string representing a scoped name.
    """
    if name_scope:
        return "%s/%s" % (name_scope, node_name)
    return node_name


def _is_model(layer):
    """Returns True if layer is a model.

    Args:
      layer: a dict representing a Keras model configuration.

    Returns:
      bool: True if layer is a model.
    """
    return layer.get("config").get("layers") is not None


def _norm_to_list_of_layers(maybe_layers):
    """Normalizes to a list of layers.

    Args:
      maybe_layers: A list of data[1] or a list of list of data.

    Returns:
      List of list of data.

    [1]: A Functional model has fields 'inbound_nodes' and 'output_layers' which can
    look like below:
    - ['in_layer_name', 0, 0]
    - [['in_layer_is_model', 1, 0], ['in_layer_is_model', 1, 1]]
    The data inside the list seems to describe [name, size, index].
    """
    return (
        maybe_layers if isinstance(maybe_layers[0], (list,)) else [maybe_layers]
    )


def _get_inbound_nodes(layer):
    """Returns a list of [name, size, index] for all inbound nodes of the given layer."""
    inbound_nodes = []
    if layer.get("inbound_nodes") is not None:
        for maybe_inbound_node in layer.get("inbound_nodes", []):
            if not isinstance(maybe_inbound_node, dict):
                # Note that the inbound node parsing is not backward compatible with
                # Keras 2. If given a Keras 2 model, the input nodes will be missing
                # in the final graph.
                continue
            for inbound_node_args in maybe_inbound_node.get("args", []):
                # Sometimes this field is a list when there are multiple inbound nodes
                # for the given layer.
                if not isinstance(inbound_node_args, list):
                    inbound_node_args = [inbound_node_args]
                for arg in inbound_node_args:
                    history = arg.get("config", {}).get("keras_history", [])
                    if len(history) < 3:
                        continue
                    inbound_nodes.append(history[:3])
    return inbound_nodes


def _update_dicts(
    name_scope,
    model_layer,
    input_to_in_layer,
    model_name_to_output,
    prev_node_name,
):
    """Updates input_to_in_layer, model_name_to_output, and prev_node_name
    based on the model_layer.

    Args:
      name_scope: a string representing a scope name, similar to that of tf.name_scope.
      model_layer: a dict representing a Keras model configuration.
      input_to_in_layer: a dict mapping Keras.layers.Input to inbound layer.
      model_name_to_output: a dict mapping Keras Model name to output layer of the model.
      prev_node_name: a string representing a previous, in sequential model layout,
                      node name.

    Returns:
      A tuple of (input_to_in_layer, model_name_to_output, prev_node_name).
      input_to_in_layer: a dict mapping Keras.layers.Input to inbound layer.
      model_name_to_output: a dict mapping Keras Model name to output layer of the model.
      prev_node_name: a string representing a previous, in sequential model layout,
                      node name.
    """
    layer_config = model_layer.get("config")
    if not layer_config.get("layers"):
        raise ValueError("layer is not a model.")

    node_name = _scoped_name(name_scope, layer_config.get("name"))
    input_layers = layer_config.get("input_layers")
    output_layers = layer_config.get("output_layers")
    inbound_nodes = _get_inbound_nodes(model_layer)

    is_functional_model = bool(input_layers and output_layers)
    # In case of [1] and the parent model is functional, current layer
    # will have the 'inbound_nodes' property.
    is_parent_functional_model = bool(inbound_nodes)

    if is_parent_functional_model and is_functional_model:
        for input_layer, inbound_node in zip(input_layers, inbound_nodes):
            input_layer_name = _scoped_name(node_name, input_layer)
            inbound_node_name = _scoped_name(name_scope, inbound_node[0])
            input_to_in_layer[input_layer_name] = inbound_node_name
    elif is_parent_functional_model and not is_functional_model:
        # Sequential model can take only one input. Make sure inbound to the
        # model is linked to the first layer in the Sequential model.
        prev_node_name = _scoped_name(name_scope, inbound_nodes[0][0])
    elif (
        not is_parent_functional_model
        and prev_node_name
        and is_functional_model
    ):
        assert len(input_layers) == 1, (
            "Cannot have multi-input Functional model when parent model "
            "is not Functional. Number of input layers: %d" % len(input_layer)
        )
        input_layer = input_layers[0]
        input_layer_name = _scoped_name(node_name, input_layer)
        input_to_in_layer[input_layer_name] = prev_node_name

    if is_functional_model and output_layers:
        layers = _norm_to_list_of_layers(output_layers)
        layer_names = [_scoped_name(node_name, layer[0]) for layer in layers]
        model_name_to_output[node_name] = layer_names
    else:
        last_layer = layer_config.get("layers")[-1]
        last_layer_name = last_layer.get("config").get("name")
        output_node = _scoped_name(node_name, last_layer_name)
        model_name_to_output[node_name] = [output_node]
    return (input_to_in_layer, model_name_to_output, prev_node_name)


def keras_model_to_graph_def(keras_layer):
    """Returns a GraphDef representation of the Keras model in a dict form.

    Note that it only supports models that implemented to_json().

    Args:
      keras_layer: A dict from Keras model.to_json().

    Returns:
      A GraphDef representation of the layers in the model.
    """
    input_to_layer = {}
    model_name_to_output = {}
    g = GraphDef()

    # Sequential model layers do not have a field "inbound_nodes" but
    # instead are defined implicitly via order of layers.
    prev_node_name = None

    for name_scope, layer in _walk_layers(keras_layer):
        if _is_model(layer):
            (
                input_to_layer,
                model_name_to_output,
                prev_node_name,
            ) = _update_dicts(
                name_scope,
                layer,
                input_to_layer,
                model_name_to_output,
                prev_node_name,
            )
            continue

        layer_config = layer.get("config")
        node_name = _scoped_name(name_scope, layer_config.get("name"))

        node_def = g.node.add()
        node_def.name = node_name

        if layer.get("class_name") is not None:
            keras_cls_name = layer.get("class_name").encode("ascii")
            node_def.attr["keras_class"].s = keras_cls_name

        dtype_or_policy = layer_config.get("dtype")
        dtype = None
        has_unsupported_value = False
        # If this is a dict, try and extract the dtype string from
        # `config.name`. Keras will export like this for non-input layers and
        # some other cases (e.g. tf/keras/mixed_precision/Policy, as described
        # in issue #5548).
        if isinstance(dtype_or_policy, dict) and "config" in dtype_or_policy:
            dtype = dtype_or_policy.get("config").get("name")
        elif dtype_or_policy is not None:
            dtype = dtype_or_policy

        if dtype is not None:
            try:
                tf_dtype = dtypes.as_dtype(dtype)
                node_def.attr["dtype"].type = tf_dtype.as_datatype_enum
            except TypeError:
                has_unsupported_value = True
        elif dtype_or_policy is not None:
            has_unsupported_value = True

        if has_unsupported_value:
            # There's at least one known case when this happens, which is when
            # mixed precision dtype policies are used, as described in issue
            # #5548. (See https://keras.io/api/mixed_precision/).
            # There might be a better way to handle this, but here we are.
            logger.warning(
                "Unsupported dtype value in graph model config (json):\n%s",
                dtype_or_policy,
            )
        if layer.get("inbound_nodes") is not None:
            for name, size, index in _get_inbound_nodes(layer):
                inbound_name = _scoped_name(name_scope, name)
                # An input to a layer can be output from a model. In that case, the name
                # of inbound_nodes to a layer is a name of a model. Remap the name of the
                # model to output layer of the model. Also, since there can be multiple
                # outputs in a model, make sure we pick the right output_layer from the model.
                inbound_node_names = model_name_to_output.get(
                    inbound_name, [inbound_name]
                )
                # There can be multiple inbound_nodes that reference the
                # same upstream layer. This causes issues when looking for
                # a particular index in that layer, since the indices
                # captured in `inbound_nodes` doesn't necessarily match the
                # number of entries in the `inbound_node_names` list. To
                # avoid IndexErrors, we just use the last element in the
                # `inbound_node_names` in this situation.
                # Note that this is a quick hack to avoid IndexErrors in
                # this situation, and might not be an appropriate solution
                # to this problem in general.
                input_name = (
                    inbound_node_names[index]
                    if index < len(inbound_node_names)
                    else inbound_node_names[-1]
                )
                node_def.input.append(input_name)
        elif prev_node_name is not None:
            node_def.input.append(prev_node_name)

        if node_name in input_to_layer:
            node_def.input.append(input_to_layer.get(node_name))

        prev_node_name = node_def.name

    return g
