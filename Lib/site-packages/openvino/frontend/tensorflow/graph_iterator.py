# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import tensorflow as tf
from openvino.frontend.tensorflow.node_decoder import TFGraphNodeDecoder
from openvino.frontend.tensorflow.py_tensorflow_frontend import _FrontEndPyGraphIterator as GraphIterator


class GraphIteratorTFGraph(GraphIterator):
    def __init__(self, tf_graph: tf.Graph, share_weights: bool, inner_graph: bool = False,
                 input_names_map: dict = None, output_names_map: dict = None):
        GraphIterator.__init__(self)
        self.m_graph = tf_graph
        self.m_node_index = 0
        self.m_decoders = []
        self.m_inner_graph = inner_graph
        self.m_share_weights = share_weights
        self.m_input_names_map = input_names_map or {}
        self.m_output_names_map = output_names_map or {}
        self.m_vars = None
        if hasattr(tf_graph, "variables"):
            # This field is needed to keep the link to graph variables,
            # otherwise Python releases memory kept by variables when it is accessed from c++ bindings
            self.m_vars = tf_graph.variables

        for op in tf_graph.get_operations():
            self.m_decoders.append(TFGraphNodeDecoder(op, share_weights, inner_graph))

        self.m_iterators = {}
        for func_name, _ in self.m_graph._functions.items():
            self.m_iterators[func_name] = None

    def get_input_names(self) -> list:
        # returns a vector of input names in the original order
        # Note: used only for the library functions
        if not self.m_inner_graph:
            return []
        inp_ops = filter(lambda op: op.type == "Placeholder", self.m_graph.get_operations())
        inp_names = []
        if hasattr(self.m_graph, 'inputs') and self.m_graph.inputs:
            for inp in self.m_graph.inputs:
                inp_names.append(inp.name)
            return inp_names
        for inp in inp_ops:
            assert isinstance(inp, tf.Operation), "Unknown node type. Expected tf.Operation, got {}".format(type(inp))
            assert hasattr(inp, "node_def") and isinstance(inp.node_def, tf.compat.v1.NodeDef), \
                "Could not find node_def in node {}".format(inp.name)
            type_attr = inp.node_def.attr["dtype"].type

            # Placeholders with type "resource" have exact values in "variables" field,
            # so they are passed to TF FE as constants.
            # For this reason they are not listed as model inputs.
            if tf.dtypes.DType(type_attr).name != "resource" or self.m_inner_graph:
                inp_names.append(inp.name)
        return inp_names

    def get_output_names(self) -> list:
        # returns a vector of output names in the original order
        # Note: used only for the library functions
        if not self.m_inner_graph:
            return []

        if hasattr(self.m_graph, 'outputs') and self.m_graph.outputs:
            outputs = []
            for out in self.m_graph.outputs:
                outputs.append(out.name)
            return outputs
        # If graph has no 'outputs' field, find nodes without outputs and consider them graph outputs.
        # The order of outputs is important and wrong order may lead to conversion error.
        non_outputs = set()
        for op in self.m_graph.get_operations():
            assert isinstance(op, tf.Operation), "Unknown node type. Expected tf.Operation, got {}".format(type(op))
            for inp in op.inputs:
                non_outputs.add(inp.op.name)

        outputs = []
        for op in self.m_graph.get_operations():
            if op.name not in non_outputs:
                for output in op.outputs:
                    outputs = [output.name] + outputs
        return outputs

    def get_input_names_map(self) -> dict:
        # returns a map from (user-defined) external tensor name to internal name for inputs
        return self.m_input_names_map

    def get_output_names_map(self) -> dict:
        # returns a map from (user-defined) external tensor name to internal name for outputs
        return self.m_output_names_map

    def is_end(self) -> bool:
        return self.m_node_index >= len(self.m_decoders)

    def reset(self):
        self.m_node_index = 0

    def size(self) -> int:
        return len(self.m_decoders)

    def next_impl(self):
        self.m_node_index += 1

    def get_decoder(self):
        return self.m_decoders[self.m_node_index]

    def get_body_graph_iterator(self, func_name):
        if func_name not in self.m_iterators:
            return None
        if self.m_iterators[func_name] is None:
            self.m_iterators[func_name] = GraphIteratorTFGraph(self.m_graph._functions[func_name].graph,
                                                               self.m_share_weights,
                                                               True)
        return self.m_iterators[func_name]
