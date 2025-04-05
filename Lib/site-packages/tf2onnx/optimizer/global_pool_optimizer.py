# SPDX-License-Identifier: Apache-2.0


"""global pool optimizer
   Replaces ReduceMean and ReduceMax patterns with GlobalAveragePool and GlobalMaxPool
"""

from onnx import TensorProto
from tf2onnx.graph_builder import GraphBuilder
from .optimizer_base import GraphOptimizerBase

# pylint: disable=logging-not-lazy,unused-argument,missing-docstring


class GlobalPoolOptimizer(GraphOptimizerBase):

    def __init__(self):  # pylint: disable=useless-super-delegation
        super(GlobalPoolOptimizer, self).__init__()

    def _optimize(self, graph):
        return self._apply_optimization(graph, self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, graph):
        graph_changed = True
        while graph_changed:
            graph_changed = False
            ops = graph.get_nodes()
            for op in ops:
                if op.type in ["ReduceMean", "ReduceMax"] and self._optimize_reduce(op, graph):
                    graph_changed = True
                    self.graph_been_opt = True
        return graph

    def _optimize_reduce(self, node, graph):
        if graph.get_dtype(node.output[0]) not in [TensorProto.FLOAT, TensorProto.DOUBLE]:
            return False
        if node.output[0] in graph.outputs:
            # Replacement is unsafe
            return False
        axes = node.get_attr_value('axes')
        inp_rank = graph.get_rank(node.input[0])
        if inp_rank is None:
            return False
        if axes != list(range(2, inp_rank)):
            return False
        op_map = {"ReduceMean": "GlobalAveragePool", "ReduceMax": "GlobalMaxPool"}
        node.type = op_map[node.type]
        del node.attr['axes']
        if not node.get_attr_value('keepdims', True):
            out_shapes = node.output_shapes
            out_dtypes = node.output_dtypes
            new_out_shape = graph.get_shape(node.input[0])[:2] + [1] * len(axes)
            graph.set_shape(node.output[0], new_out_shape)
            squeeze_node = GraphBuilder(graph).make_squeeze(
                {'data': node.output[0], 'axes': axes}, shapes=out_shapes, dtypes=out_dtypes,
                return_node=True, op_name_scope=node.name)
            graph.insert_node_on_output(squeeze_node, node.output[0])
        if 'keepdims' in node.attr:
            del node.attr['keepdims']
        return True
