# SPDX-License-Identifier: Apache-2.0

"""Resize Optimizer.
    Replace resize operations with all ones in scale with Identity nodes
"""

import numpy as np

from .optimizer_base import GraphOptimizerBase

# pylint: disable=logging-not-lazy,unused-argument,missing-docstring,unused-variable,arguments-differ


class UpsampleOptimizer(GraphOptimizerBase):
    """Upsample Optimizer."""

    def __init__(self):  # pylint: disable=useless-super-delegation
        super(UpsampleOptimizer, self).__init__()
        self._g = None

    def _optimize(self, graph):
        return self._apply_optimization(
            graph,
            self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, graph):
        self._g = graph
        # replace upsample node with all ones in scale with identity node
        for n in self._g.get_nodes():
            if n.type == "Upsample":
                node_changed = False
                # upsample in opset <=8 has scales in attributes
                if self._g.opset <= 8:
                    scales = n.get_attr_value("scales")
                    if scales and all([float(s) == 1. for s in scales]):
                        n.type = "Identity"
                        node_changed = True
                # upsample in opset >= 9 has scales in input[1]
                if self._g.opset >= 9 and len(n.input) == 2:
                    scales_input = n.inputs[1]

                    if scales_input.is_const() and \
                            np.all(scales_input.get_tensor_value(as_list=False) == 1.):
                        n.type = "Identity"
                        n.input = [n.input[0]]
                        node_changed = True
                if node_changed:
                    self.logger.debug("replacing " + n.name +
                                      " with Identity operation ")

        return self._g
