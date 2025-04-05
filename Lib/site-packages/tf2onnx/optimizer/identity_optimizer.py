# SPDX-License-Identifier: Apache-2.0


"""Identity Optimizer.
   Remove useless Identity node in graphs including subgraphs, but does not hurt model output names.
"""

from .optimizer_base import GraphOptimizerBase


# pylint: disable=logging-not-lazy,unused-argument,missing-docstring,unused-variable,arguments-differ


class IdentityOptimizer(GraphOptimizerBase):
    """Identity Optimizer."""

    def __init__(self):  # pylint: disable=useless-super-delegation
        super(IdentityOptimizer, self).__init__()

    def _optimize(self, graph):
        return self._apply_optimization(graph, self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, g):
        has_update = True
        while has_update:
            has_update = False
            nodes = [n for n in g.get_nodes() if n.type == "Identity"]
            for n in nodes:
                if n.graph is None:
                    self.logger.debug("node has been removed from this graph, skip")
                    continue

                graph_outputs = set(n.output).intersection(g.outputs)
                ret = False
                if graph_outputs:
                    ret = self._handle_graph_output_identity(g, n, graph_outputs)
                else:
                    ret = self._handle_non_graph_output_identity(g, n)
                has_update = ret
                if ret:
                    self.graph_been_opt = True
        return g

    @staticmethod
    def _handle_non_graph_output_identity(graph, identity):
        old_name = identity.output[0]
        new_name = identity.input[0]
        graph.replace_all_inputs(old_name, new_name, ops=graph.get_nodes())
        graph.remove_node(identity.name)
        return True

    def _handle_graph_output_identity(self, graph, identity, graph_outputs):
        input_id = identity.input[0]
        input_node = identity.inputs[0]

        if input_node.graph != graph:
            # If input node is in parent graph, we don't handle it now
            self.logger.debug("input node in parent graph, skip")
            return False

        if input_node.is_graph_input():
            # Identity between input and output should not be removed.
            self.logger.debug("skip identity between input and output")
            return False

        output_id = identity.output[0]
        output_shape = graph.get_shape(output_id)
        output_dtype = graph.get_dtype(output_id)
        if input_id in graph.outputs:
            # input id already be graph output, so we cannot make that be another graph output.
            # this Identity must be kept.
            self.logger.debug("identity input already be graph output")
            return False

        graph.remove_node(identity.name)
        new_output = [output_id if o == input_id else o for o in input_node.output]
        input_node.output = new_output

        graph.set_shape(output_id, output_shape)
        graph.set_dtype(output_id, output_dtype)

        graph.replace_all_inputs(input_id, output_id, ops=graph.get_nodes())
        return True
