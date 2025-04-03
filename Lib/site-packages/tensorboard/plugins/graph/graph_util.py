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
"""Utilities for graph plugin."""

from tensorboard.compat.proto import graph_pb2


def _prefixed_op_name(prefix, op_name):
    return "%s/%s" % (prefix, op_name)


def _prefixed_func_name(prefix, func_name):
    """Returns function name prefixed with `prefix`.

    For function libraries, which are often created out of autographed Python
    function, are factored out in the graph vis. They are grouped under a
    function name which often has a shape of
    `__inference_[py_func_name]_[numeric_suffix]`.

    While it does not have some unique information about which graph it is from,
    creating another wrapping structure with graph prefix and "/" is less than
    ideal so we join the prefix and func_name using underscore.

    TODO(stephanwlee): add business logic to strip "__inference_" for more user
    friendlier name
    """
    return "%s_%s" % (prefix, func_name)


def _add_with_prepended_names(prefix, graph_to_add, destination_graph):
    for node in graph_to_add.node:
        new_node = destination_graph.node.add()
        new_node.CopyFrom(node)
        new_node.name = _prefixed_op_name(prefix, node.name)
        new_node.input[:] = [
            _prefixed_op_name(prefix, input_name) for input_name in node.input
        ]

        # Remap tf.function method name in the PartitionedCall. 'f' is short for
        # function.
        if new_node.op == "PartitionedCall" and new_node.attr["f"]:

            new_node.attr["f"].func.name = _prefixed_func_name(
                prefix,
                new_node.attr["f"].func.name,
            )

    for func in graph_to_add.library.function:
        new_func = destination_graph.library.function.add()
        new_func.CopyFrom(func)
        new_func.signature.name = _prefixed_func_name(
            prefix, new_func.signature.name
        )

    for gradient in graph_to_add.library.gradient:
        new_gradient = destination_graph.library.gradient.add()
        new_gradient.CopyFrom(gradient)
        new_gradient.function_name = _prefixed_func_name(
            prefix,
            new_gradient.function_name,
        )
        new_gradient.gradient_func = _prefixed_func_name(
            prefix,
            new_gradient.gradient_func,
        )


def merge_graph_defs(graph_defs):
    """Merges GraphDefs by adding unique prefix, `graph_{ind}`, to names.

    All GraphDefs are expected to be of TensorBoard's.

    When collecting graphs using the `tf.summary.trace` API, node names are not
    guranteed to be unique.  When non-unique names are not considered, it can
    lead to graph visualization showing them as one which creates inaccurate
    depiction of the flow of the graph (e.g., if there are A -> B -> C and D ->
    B -> E, you may see {A, D} -> B -> E).  To prevent such graph, we checked
    for uniquenss while merging but it resulted in
    https://github.com/tensorflow/tensorboard/issues/1929.

    To remedy these issues, we simply "apply name scope" on each graph by
    prefixing it with unique name (with a chance of collision) to create
    unconnected group of graphs.

    In case there is only one graph def passed, it returns the original
    graph_def. In case no graph defs are passed, it returns an empty GraphDef.

    Args:
      graph_defs: TensorBoard GraphDefs to merge.

    Returns:
      TensorBoard GraphDef that merges all graph_defs with unique prefixes.

    Raises:
      ValueError in case GraphDef versions mismatch.
    """
    if len(graph_defs) == 1:
        return graph_defs[0]
    elif len(graph_defs) == 0:
        return graph_pb2.GraphDef()

    dst_graph_def = graph_pb2.GraphDef()

    if graph_defs[0].versions.producer:
        dst_graph_def.versions.CopyFrom(graph_defs[0].versions)

    for index, graph_def in enumerate(graph_defs):
        if dst_graph_def.versions.producer != graph_def.versions.producer:
            raise ValueError("Cannot combine GraphDefs of different versions.")

        _add_with_prepended_names(
            "graph_%d" % (index + 1),
            graph_def,
            dst_graph_def,
        )

    return dst_graph_def
