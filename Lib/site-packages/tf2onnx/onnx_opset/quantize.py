# SPDX-License-Identifier: Apache-2.0


"""
tensor
"""

import logging

import numpy as np
from onnx.onnx_pb import TensorProto

from tf2onnx import utils
from tf2onnx.handler import tf_op
from tf2onnx.utils import make_sure

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring,unused-variable,pointless-string-statement,invalid-name


@tf_op(["FakeQuantWithMinMaxArgs", "FakeQuantWithMinMaxVars"])
class FakeQuantWithMinMaxArgs:
    # see https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/fake-quant-with-min-max-args
    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        # hack to make up for the missing onnx pack op
        if node.type == "FakeQuantWithMinMaxVars":
            utils.make_sure(node.inputs[1].is_scalar(), "%s node %s requires const scalar value for min",
                            node.type, node.name)
            utils.make_sure(node.inputs[2].is_scalar(), "%s node %s requires const scalar value for max",
                            node.type, node.name)
            amin = node.inputs[1].get_tensor_value()
            amax = node.inputs[2].get_tensor_value()
        else:
            amin = node.get_attr("min").f
            amax = node.get_attr("max").f
        narrow_range = node.get_attr("narrow_range").i
        num_bits = node.get_attr("num_bits").i

        make_sure(
            not narrow_range,
            "Unable to convert node FakeQuantWithMinMaxArgs with narrow_range=%r",
            narrow_range)
        make_sure(
            num_bits == 8,
            "Unable to convert node FakeQuantWithMinMaxArgs with "
            "num_bits=%r", num_bits)

        scale = (amax - amin) / (2 ** num_bits - 1)
        min_adj = np.around(amin / scale)

        dtype = ctx.get_dtype(node.input[0])
        shape = ctx.get_shape(node.input[0])
        axis = 1
        idtype = TensorProto.UINT8

        pb_scale = ctx.make_const(
            utils.make_name("{}_scaley".format(node.name)),
            np.array(scale, dtype=np.float32))
        zero = np.array(-min_adj, dtype=np.uint8)
        make_sure(
            zero == -min_adj,
            "Cannot convert %s node %s with "
            "min=%r max=%r numbits=%r because zero_scale=%r "
            "is outside uint8 boundary",
            node.type, node.name, amin, amax, num_bits, -min_adj)
        zero_point = ctx.make_const(
            utils.make_name("{}_zpy".format(node.name)), zero)

        new_node = ctx.make_node(
            "QuantizeLinear", [node.input[0], pb_scale.name, zero_point.name],
            op_name_scope=node.name, attr={"axis": axis},
            shapes=[shape], dtypes=[idtype])
        output_name = new_node.output[0]
        ctx.replace_input(node, node.input[0], output_name, 0)

        ctx.remove_node(node.name)

        last_node = ctx.make_node(
            "DequantizeLinear", [new_node.output[0], pb_scale.name, zero_point.name],
            op_name_scope=node.name, attr={"axis": axis},
            shapes=[shape], dtypes=[dtype])
        ctx.replace_all_inputs(node.output[0], last_node.output[0])  # ops=ctx.get_nodes()
