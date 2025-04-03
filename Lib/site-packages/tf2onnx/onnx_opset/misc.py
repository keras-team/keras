# SPDX-License-Identifier: Apache-2.0


"""
misc
"""

import logging

from tf2onnx.handler import tf_op


logger = logging.getLogger(__name__)

# pylint: disable=unused-argument,missing-docstring

@tf_op(["CheckNumerics", "StopGradient"])
class MoveToIdent:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.type = "Identity"
        if node.inputs[0].is_const():
            # should not remove the identity node if it is output of the graph
            if node.output[0] in ctx.outputs:
                return
            # if identity has a const as input, remove it
            input_name = node.input[0]
            output_name = node.output[0]
            ctx.replace_all_inputs(output_name, input_name)  # ops=ctx.get_nodes()
            ctx.remove_node(node.name)


@tf_op(["Placeholder", "PlaceholderV2", "PlaceholderWithDefault"])
class DirectOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass


@tf_op("NoOp")
class NukeNode:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        ctx.remove_node(node.name)
