# SPDX-License-Identifier: Apache-2.0


"""const dequantize Optimizer.
   if a dequantize op's inputs are const we may be able to fold it through the next op
"""

from .optimizer_base import GraphOptimizerBase
from .const_fold_optimizer import ConstFoldOptimizer

# pylint: disable=logging-not-lazy,unused-argument,missing-docstring


class ConstDequantizeOptimizer(GraphOptimizerBase):

    def __init__(self):  # pylint: disable=useless-super-delegation
        super(ConstDequantizeOptimizer, self).__init__()

    def _optimize(self, graph):
        return self._apply_optimization(graph, self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, graph):
        graph_changed = True
        while graph_changed:
            graph_changed = False
            ops = graph.get_nodes()
            for op in ops:
                if self._fold_node(op, graph):
                    graph_changed = True
                    self.graph_been_opt = True
        return graph

    def _fold_node(self, node, graph):
        """ if a dequantize op's inputs are const and it is fed into a tensor reshaping op, we can apply the op
            directly to the quantized inputs.  Returns True if the graph is changed.
        """
        if node.type not in ["Transpose", "Reshape", "Unsqueeze"]:
            return False
        dequant_node = node.inputs[0]
        if dequant_node.type != "DequantizeLinear":
            return False
        if len(graph.find_output_consumers(dequant_node.output[0])) > 1:
            return False
        if not self._all_inputs_are_const(node.inputs[1:]) or self._is_graph_output(node, graph):
            return False
        if not self._all_inputs_are_const(dequant_node.inputs):
            return False
        if len(dequant_node.inputs[1].get_tensor_value(as_list=False).flatten()) != 1:
            # If using per-channel quantization, we must compute the new axis
            old_axis = dequant_node.get_attr_value("axis")
            input_shape = dequant_node.inputs[0].get_tensor_value(as_list=False).shape
            new_axis = self.compute_new_axis(node, graph, old_axis, input_shape)
            if new_axis is None:
                return False
            dequant_node.set_attr("axis", new_axis)
        graph.replace_input(node, node.input[0], dequant_node.input[0], 0)
        const_outputs = ConstFoldOptimizer.compute_const_folding(node, graph)
        graph.replace_all_inputs(node.output[0], dequant_node.output[0])
        graph.remove_node(node.name)
        dequant_const = dequant_node.inputs[0]
        if len(graph.find_output_consumers(dequant_const.output[0])) > 1:
            dequant_const = graph.copy_const(dequant_const)
            graph.replace_input(dequant_node, dequant_node.input[0], dequant_const.output[0], 0)
        dequant_const.set_tensor_value(const_outputs[0])
        return True

    @staticmethod
    def _all_inputs_are_const(nodes):
        return all(node.is_const() for node in nodes if node)

    @staticmethod
    def _is_graph_output(node, graph):
        node_out_set = set(node.output)
        graph_out_set = set(graph.outputs)
        return node_out_set.intersection(graph_out_set)

    @staticmethod
    def compute_new_axis(node, graph, old_axis, input_shape):
        if old_axis < 0:
            old_axis += len(input_shape)
        if node.type == "Transpose":
            perm = node.get_attr_value("perm")
            if perm is None:
                return None
            return perm.index(old_axis)
        if node.type == "Reshape":
            prod = 1
            for d in input_shape[:old_axis+1]:
                prod *= d
            new_shape = node.inputs[1].get_tensor_value(as_list=True)
            new_prod = 1
            for i, d in enumerate(new_shape):
                new_prod *= d
                if new_prod == prod:
                    if new_shape[i] == input_shape[old_axis]:
                        return i
                    return None
            return None
        if node.type == "Unsqueeze":
            if graph.opset >= 13:
                axes = node.inputs[1].get_tensor_value(as_list=True)
            else:
                axes = node.get_attr_value("axes")
            new_rank = len(input_shape) + len(axes)
            axes = [axis if axis >= 0 else axis + new_rank for axis in axes]
            for i in range(new_rank):
                if i not in axes:
                    if old_axis == 0:
                        return i
                    old_axis -= 1
            return None
        return None
