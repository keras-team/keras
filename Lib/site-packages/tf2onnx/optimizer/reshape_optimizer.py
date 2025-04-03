# SPDX-License-Identifier: Apache-2.0


"""reshape optimizer
   Finds reshape ops with computed shapes and attempts to replace them with constant shapes.
   Even if some of the dimensions are non-constant, they can often be replaced with -1 or 0.
   Specifically, optimizes the pattern:  node -> shape -> [computation] -> reshape
                                             `-----------------------------^
"""

from collections import Counter
import numpy as np
from tf2onnx import utils
from tf2onnx.graph_builder import GraphBuilder
from tf2onnx.symbolic_executor import SymbolicExecutor, SymbolicTensorElement, SymbolicExecutionException
from .optimizer_base import GraphOptimizerBase

# pylint: disable=logging-not-lazy,unused-argument,missing-docstring


class ReshapeOptimizer(GraphOptimizerBase):

    def __init__(self):  # pylint: disable=useless-super-delegation
        super(ReshapeOptimizer, self).__init__()

    def _optimize(self, graph):
        return self._apply_optimization(graph, self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, graph):
        graph_changed = True
        while graph_changed:
            graph_changed = False
            ops = graph.get_nodes()
            for op in ops:
                if op.type == "Reshape" and self._optimize_reshape(op, graph):
                    graph_changed = True
                    self.graph_been_opt = True
        return graph

    def _optimize_reshape(self, node, graph):
        if node.inputs[1].is_const():
            return False
        inp_shape = graph.get_shape(node.input[0])
        if inp_shape is None:
            # The rank must be known
            return False
        feed_dict = {}
        for n in graph.find_output_consumers(node.input[0]):
            if n.type == "Shape":
                symbolic_shape = []
                for i, d in enumerate(inp_shape):
                    if d == -1:
                        # Make a variable representing each unknown dim
                        symbolic_shape.append(SymbolicTensorElement.from_variable(i))
                    else:
                        symbolic_shape.append(SymbolicTensorElement.from_const(d))
                feed_dict[n.output[0]] = np.array(symbolic_shape, object)
        try:
            symbolic_res = SymbolicExecutor(graph).compute_outputs([node.input[1]], feed_dict)
        except SymbolicExecutionException:
            return False
        utils.make_sure(len(symbolic_res[0].shape) == 1, "Shape must have rank 1")
        symbolic_shape = symbolic_res[0].tolist()
        product_cnt = len([val for val in symbolic_shape if val.has_multiple_terms()])
        idx_cnt = len([val for val in symbolic_shape if val.is_single_var()])
        if product_cnt > 1:
            # The -1 lets us handle at most one dim with multiple terms
            return False
        if idx_cnt + product_cnt <= 1:
            # Only 1 non-const dim. Use -1 and consts for the rest.
            new_shape = [v.constant if v.is_const() else -1 for v in symbolic_shape]
            shift = 0
        else:
            # We will need to use some 0s. We can shift using squeeze/unsqueeze to line up equal dims
            def get_shift(val, i):
                if not val.is_single_var():
                    return None
                return val.terms[0] - i
            shifts = [get_shift(val, i) for i, val in enumerate(symbolic_shape)]
            # Find the most popular shift
            most_common = Counter(s for s in shifts if s is not None).most_common(1)
            shift = most_common[0][0] if most_common else 0
            def get_reshape_dim(val, i, shift):
                if val.is_const():
                    return val.constant
                if get_shift(val, i) == shift:
                    return 0
                # Use -1 only as a last resort
                return -1
            new_shape = [get_reshape_dim(v, i, shift) for i, v in enumerate(symbolic_shape)]
        if new_shape.count(-1) > 1:
            return False

        new_reshape_shape = None
        if shift > 0:
            new_shape = [1] * shift + new_shape
            squeeze_node = GraphBuilder(graph).make_squeeze(
                {'data': node.output[0], 'axes': list(range(shift))},
                return_node=True, shapes=node.output_shapes, dtypes=node.output_dtypes)
            new_reshape_shape = [1] * shift + graph.get_shape(node.output[0])
            graph.insert_node_on_output(squeeze_node, node.output[0])
        const_shape = graph.make_const(utils.make_name(node.name + "_shape"), np.array(new_shape, np.int64)).output[0]
        if inp_shape == [-1] and len(new_shape) > 1:
            # This is a mismatch.
            return False
        if new_reshape_shape is not None:
            graph.set_shape(node.output[0], new_reshape_shape)
        graph.replace_inputs(node, [node.input[0], const_shape])
        if shift < 0:
            unsqueeze_node = GraphBuilder(graph).make_unsqueeze({'data': node.input[0], 'axes': list(range(-shift))})
            graph.replace_inputs(node, [unsqueeze_node, const_shape])

        return True
