# SPDX-License-Identifier: Apache-2.0


"""q dq optimizer
   Pushes Quantize ops up and Dequantize ops down to maximize DQ -> op -> Q patterns for ORT
   Does not work for per-channel quantization yet
"""

from .optimizer_base import GraphOptimizerBase

# pylint: disable=logging-not-lazy,unused-argument,missing-docstring


class QDQOptimizer(GraphOptimizerBase):

    def __init__(self):  # pylint: disable=useless-super-delegation
        super(QDQOptimizer, self).__init__()

    def _optimize(self, graph):
        return self._apply_optimization(graph, self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, graph):
        graph_changed = True
        while graph_changed:
            graph_changed = False
            ops = graph.get_nodes()
            for op in ops:
                if op.type == "QuantizeLinear" and self._optimize_quantize(op, graph):
                    graph_changed = True
                    self.graph_been_opt = True
                elif op.type == "DequantizeLinear" and self._optimize_dequantize(op, graph):
                    graph_changed = True
                    self.graph_been_opt = True
        return graph

    def _optimize_quantize(self, quant_node, graph):
        if 'axis' in quant_node.attr:
            return False
        node = quant_node.inputs[0]
        if node.type == "DequantizeLinear":
            # Remove DQ -> Q
            if not self.has_same_quantization_params(quant_node, node):
                return False
            if quant_node.output[0] in graph.outputs or node.output[0] in graph.outputs:
                return False
            graph.replace_all_inputs(quant_node.output[0], node.input[0])
            if not graph.find_output_consumers(quant_node.output[0]):
                graph.remove_node(quant_node.name)
            if not graph.find_output_consumers(node.output[0]):
                graph.remove_node(node.name)
            return True

        # Push quantize nodes up
        tensor_idx = is_tensor_op(graph, node)
        if tensor_idx is None:
            return False
        inp_indices, out_indices = tensor_idx
        for i in out_indices:
            consumers = graph.find_output_consumers(node.output[i])
            if node.output[i] in graph.outputs:
                return False
            for c in consumers:
                if c.type != "QuantizeLinear":
                    return False
                if not self.has_same_quantization_params(c, quant_node):
                    return False
                if c.output[0] in graph.outputs:
                    return False
        # All outputs are quantized. Push quantization up to input.
        for i in inp_indices:
            inp_q = self.make_q_or_dq(graph, "QuantizeLinear", node.input[i], quant_node, node.name)
            graph.replace_input(node, node.input[i], inp_q.output[0], i)

        for i in out_indices:
            graph.copy_dtype(quant_node.output[0], node.output[i])
            consumers = graph.find_output_consumers(node.output[i])
            for c in consumers:
                graph.replace_all_inputs(c.output[0], node.output[i])

        return True

    def _optimize_dequantize(self, dequant_node, graph):
        if 'axis' in dequant_node.attr:
            return False
        # Push dequantize nodes down
        consumers = graph.find_output_consumers(dequant_node.output[0])
        for node in consumers:
            if self._optimize_dequantize_and_node(dequant_node, node, graph):
                return True
        return False

    def _optimize_dequantize_and_node(self, dequant_node, node, graph):
        tensor_idx = is_tensor_op(graph, node)
        if tensor_idx is None:
            return False
        inp_indices, out_indices = tensor_idx
        for i in inp_indices:
            inp = node.inputs[i]
            if inp.type != "DequantizeLinear":
                return False
            if not self.has_same_quantization_params(inp, dequant_node):
                return False
            if inp.output[0] in graph.outputs:
                return False
        for i in out_indices:
            if node.output[i] in graph.outputs:
                return False
        # All inputs are dequantized. Push dequantization down to output.
        for i in inp_indices:
            # Skip the dequantize on the input
            graph.replace_input(node, node.input[i], node.inputs[i].input[0], i)

        for i in out_indices:
            graph.copy_dtype(dequant_node.input[0], node.output[i])
            out_dq = self.make_q_or_dq(graph, "DequantizeLinear", node.output[i], dequant_node, node.name)
            graph.insert_node_on_output(out_dq, node.output[i])

        return True

    def has_same_quantization_params(self, node1, node2):
        if node1.get_attr_value("axis") != node2.get_attr_value("axis"):
            return False
        # Constant merging will ensure these are the same nodes if they are equal
        return node1.input[1:] == node2.input[1:]

    def make_q_or_dq(self, graph, op_type, inp, reference_node, name_scope):
        """Makes a QuantizeLinear or DequantizeLinear with quantization params copied from the reference_node"""
        axis = reference_node.get_attr_value("axis")
        if axis is None:
            attr = {}
        else:
            attr = {'axis': axis}
        return graph.make_node(op_type, [inp] + reference_node.input[1:], attr=attr, op_name_scope=name_scope)


def is_tensor_op(g, node):
    """Detects ops that reshape/shuffle tensor elements without computing/changing them (Transpose, Gather, etc.)
    Returns None or a tuple (inp_indices, out_indices) s.t. all corresponding outputs of the node depend only
    on elements of the corresponding inputs of the node and all other inputs/outputs are unchanged.
    WARNING: Transpose optimizer pushes tranpose down so be careful when swapping to avoid infinite loop."""
    if node.type in ["Identity", "Reshape", "Flatten", "Expand", "Transpose", "Squeeze", "Unsqueeze", "Slice"]:
        return ([0], [0])
    if node.type in ["Gather", "GatherND", "GatherElements"]:
        # Output depends on data if indices is unchanged
        return ([0], [0])
    if node.type in ["Scatter", "ScatterND", "ScatterElements"]:
        # Output depends on data and updates if indices is unchanged
        return ([0, 2], [0])
    if node.type == "Concat":
        return (list(range(len(node.input))), [0])
    if node.type == "Split":
        return ([0], list(range(len(node.output))))
    if node.type in ["Compress", "Tile", "ReverseSequence", "DepthToSpace"]:
        return ([0], [0])
    return None
