# SPDX-License-Identifier: Apache-2.0


"""Graph Optimizer Base"""

import copy

from .. import logging, utils


class GraphOptimizerBase(object):
    """optimizer graph to improve performance
    """

    def __init__(self):
        self._logger = logging.getLogger('.'.join(__name__.split('.')[:-1] + [self.__class__.__name__]))
        self._graph_been_opt = False
        self.opt_iteration = 0

    @property
    def logger(self):
        return self._logger

    @property
    def is_debug_mode(self):
        return utils.is_debug_mode()

    @property
    def graph_been_opt(self):
        return self._graph_been_opt

    @graph_been_opt.setter
    def graph_been_opt(self, value):
        self._graph_been_opt = value

    def optimize(self, graph, iteration):
        """ Optimize graph, return optimized graph. """
        before = graph.dump_node_statistics()

        self.opt_iteration = iteration
        graph = self._optimize(graph)
        graph.update_proto()
        graph.delete_unused_nodes(graph.outputs)

        after = graph.dump_node_statistics()
        self._print_stat_diff(before, after)
        return graph

    def _optimize(self, graph):
        """ Derived class should override this function. """
        raise NotImplementedError

    @staticmethod
    def _apply_optimization(graph, optimize_func):
        """
        optimize graph
        will also optimize graph of nodes'
        Args:
            graph: the top level graph to be optimized
            optimize_func: function to optimize graph
        """
        graph = optimize_func(graph)
        for node in graph.get_nodes():
            body_graphs = node.get_body_graphs()
            if body_graphs:
                for attr, b_g in body_graphs.items():
                    b_g = GraphOptimizerBase._apply_optimization(b_g, optimize_func)
                    node.set_body_graph_as_attr(attr, b_g)
        return graph

    def _print_stat_diff(self, before, after):
        diff = copy.deepcopy(after)
        diff.subtract(before)
        diff = ["{} {} ({}->{})".format(k, str(v) if v < 0 else '+' + str(v), before.get(k, 0), after.get(k, 0))
                for k, v in sorted(diff.items()) if v != 0]
        self.logger.verbose(', '.join(diff) if diff else "no change")
