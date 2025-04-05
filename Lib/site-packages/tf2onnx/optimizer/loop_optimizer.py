# SPDX-License-Identifier: Apache-2.0


"""Loop Optimizer.
   some op in loop's body graph can be moved out of the loop
"""

from tf2onnx.utils import make_name, make_sure
from .optimizer_base import GraphOptimizerBase


# pylint: disable=logging-not-lazy,unused-argument,missing-docstring,unused-variable,arguments-differ


class LoopOptimizer(GraphOptimizerBase):
    """Loop Optimizer."""

    # a lot of terms used here come from loop's onnx spec
    # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Loop
    def __init__(self):  # pylint: disable=useless-super-delegation
        super(LoopOptimizer, self).__init__()

    def _optimize(self, graph):
        return self._apply_optimization(graph, self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, g):
        has_update = True
        while has_update:
            has_update = False
            nodes = [n for n in g.get_nodes() if n.type == "Loop"]
            for n in nodes:
                has_update_tmp = self._try_move_transpose_out_of_body_graph(n)
                if has_update_tmp:
                    has_update = True
                    self.graph_been_opt = True
        return g

    @staticmethod
    def num_consumers(graph, node):
        make_sure(len(node.output) == 1, "only consider node with only one output")
        res = len(graph.find_output_consumers(node.output[0]))
        # This is an optimizer so we cannot rely on outputs having Identity nodes
        res += graph.outputs.count(node.output[0])
        return res

    def _try_move_transpose_out_of_body_graph(self, loop_node):
        # output node of body graph can be loop-carried-dependent, if so it can't be moved out of the body graph
        # return True if moving some nodes successfully
        # for now, we only consider moving transpose
        body_graph = loop_node.get_body_graphs()["body"]
        parent_graph = loop_node.graph
        scan_nodes_name_in_body, scan_node_in_parent = self._scan_outputs(loop_node)
        scan_nodes = [body_graph.get_node_by_output(name) for name in scan_nodes_name_in_body]
        graph_is_changed = False
        for node, name_in_parent in zip(scan_nodes, scan_node_in_parent):
            # 1 delete node in body graph if possible
            # only consider two case: trans is output, or transpose > identity > output
            need_process = False
            if node.type == "Transpose" and self.num_consumers(body_graph, node) == 1:
                trans = node
                new_output = node.input[0]
                body_graph.remove_node(node.name)
                need_process = True
            elif node.type == "Identity" and node.inputs[0].type == "Transpose" \
                    and self.num_consumers(body_graph, node) == 1\
                    and self.num_consumers(body_graph, node.inputs[0]) == 1:
                trans = node.inputs[0]
                new_output = node.inputs[0].input[0]
                body_graph.remove_node(node.inputs[0].name)
                body_graph.remove_node(node.name)
                need_process = True

            if need_process:
                # 2 correct body graph's output
                body_outputs = body_graph.outputs
                body_outputs[body_outputs.index(node.output[0])] = new_output
                # 3 insert new node in parent graph
                ori_perm = list(trans.get_attr("perm").ints)
                new_perm = [0] + [i + 1 for i in ori_perm]  # body output's rank is m > rank of loop's output is m+1
                name = make_name("trans_moved_from_loop_body")
                _ = parent_graph.insert_new_node_on_output("Transpose", name_in_parent, name, perm=new_perm)
                graph_is_changed = True

        return graph_is_changed

    @classmethod
    def _scan_outputs(cls, loop):
        # loop has 2+N inputs; loop has N+K outputs;
        # loop's body graph has 1+N+K outputs
        loop_carried = len(loop.input) - 2
        body_graph = loop.get_body_graphs()["body"]
        return body_graph.outputs[loop_carried + 1:], loop.output[loop_carried:]
