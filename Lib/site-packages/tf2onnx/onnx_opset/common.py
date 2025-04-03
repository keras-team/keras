# SPDX-License-Identifier: Apache-2.0


"""
common
"""

import logging

from tf2onnx import constants


logger = logging.getLogger(__name__)

# pylint: disable=unused-argument,missing-docstring

class BroadcastOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        """Elementwise Ops with broadcast flag."""
        if node.type == "AddV2":
            node.type = "Add"
        shape0 = ctx.get_shape(node.input[0])
        shape1 = ctx.get_shape(node.input[1])
        if shape0 != shape1:
            node.set_attr("broadcast", 1)
            # this works around shortcomings in the broadcasting code
            # of caffe2 and winml/rs4.
            if ctx.is_target(constants.TARGET_RS4):
                # in rs4 mul and add do not support scalar correctly
                if not shape0:
                    if node.inputs[0].is_const():
                        shape0 = node.inputs[0].scalar_to_dim1()
                if not shape1:
                    if node.inputs[1].is_const():
                        shape1 = node.inputs[1].scalar_to_dim1()
            if shape0 and shape1 and len(shape0) < len(shape1) and node.type in ["Mul", "Add"]:
                tmp = node.input[0]
                ctx.replace_input(node, node.input[0], node.input[1], 0)
                ctx.replace_input(node, node.input[1], tmp, 1)
        else:
            node.set_attr("broadcast", 0)

    @classmethod
    def version_6(cls, ctx, node, **kwargs):
        """Elementwise Ops with broadcast flag."""
        if node.type == "AddV2":
            node.type = "Add"
        shape0 = ctx.get_shape(node.input[0])
        shape1 = ctx.get_shape(node.input[1])
        if shape0 != shape1:
            # this works around shortcomings in the broadcasting code
            # of caffe2 and winml/rs4.
            if ctx.is_target(constants.TARGET_RS4):
                # in rs4 mul and add do not support scalar correctly
                if not shape0:
                    if node.inputs[0].is_const():
                        shape0 = node.inputs[0].scalar_to_dim1()
                if not shape1:
                    if node.inputs[1].is_const():
                        shape1 = node.inputs[1].scalar_to_dim1()
            if shape0 and shape1 and len(shape0) < len(shape1) and node.type in ["Mul", "Add"]:
                tmp = node.input[0]
                ctx.replace_input(node, node.input[0], node.input[1], 0)
                ctx.replace_input(node, node.input[1], tmp, 1)
