# SPDX-License-Identifier: Apache-2.0


"""
logical
"""

import logging

from onnx import TensorProto
from tf2onnx import utils
from tf2onnx.handler import tf_op
from tf2onnx.onnx_opset import common


logger = logging.getLogger(__name__)

# pylint: disable=unused-argument,missing-docstring

def _add_cast_to_inputs(graph, node, supported_dtypes, target_dtype):
    is_support = True
    for inp in node.input:
        if graph.get_dtype(inp) not in supported_dtypes:
            is_support = False
            break
    if not is_support:
        for inp in node.input:
            inp_cast = graph.insert_new_node_on_input(node, "Cast", inp, to=target_dtype)
            graph.copy_shape(inp, inp_cast.output[0])
            graph.set_dtype(inp_cast.output[0], target_dtype)

def _add_cast_to_same_type_to_inputs(graph, node, supported_dtypes, target_dtype):
    common_dtype = graph.get_dtype(node.input[0])
    if common_dtype not in supported_dtypes:
        common_dtype = target_dtype

    for inp in node.input:
        if graph.get_dtype(inp) != common_dtype:
            inp_cast = graph.insert_new_node_on_input(node, "Cast", inp, to=common_dtype)
            graph.copy_shape(inp, inp_cast.output[0])
            graph.set_dtype(inp_cast.output[0], common_dtype)
            if graph.is_const(inp) and graph.get_tensor_value(inp) == '':
                # Convert '' string constant to -1 int
                # https://github.com/tensorflow/tensorflow/blob/4e7f0185c70faf35e12acbfe381a729d1e6cc38c/tensorflow/python/feature_column/feature_column.py#L2286
                const_node = graph.get_node_by_output(inp)
                const_node.set_tensor_value(utils.np.array(-1))


@tf_op("LogicalNot", onnx_op="Not")
class DirectOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass


@tf_op("LogicalAnd", onnx_op="And")
@tf_op("LogicalOr", onnx_op="Or")
class BroadcastOp(common.BroadcastOp):
    pass


@tf_op(["Equal", "NotEqual"])
class Equal:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        need_not = node.type == "NotEqual"
        common.BroadcastOp.version_1(ctx, node, **kwargs)
        if need_not:
            node.type = "Equal"
            output_name = node.output[0]
            not_node = ctx.insert_new_node_on_output("Not", output_name, name=utils.make_name(node.name))
            ctx.copy_shape(output_name, not_node.output[0])
            ctx.copy_dtype(output_name, not_node.output[0])

    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # T2 output = Equal(T1, x, T1 y), T1 \in {bool, int32, int64}
        need_not = node.type == "NotEqual"
        if need_not and node.input[0] == node.input[1]:
            # The only value not equal to itself is NaN
            node.type = "IsNaN"
            ctx.replace_inputs(node, [node.input[0]])
            return
        supported_dtypes = [
            TensorProto.BOOL,
            TensorProto.INT32,
            TensorProto.INT64
        ]
        # FIXME: casting is not the same as equal
        target_dtype = TensorProto.INT32
        _add_cast_to_inputs(ctx, node, supported_dtypes, target_dtype)
        if need_not:
            node.type = "Equal"
            output_name = node.output[0]
            not_node = ctx.insert_new_node_on_output("Not", output_name, name=utils.make_name(node.name))
            ctx.copy_shape(output_name, not_node.output[0])
            ctx.copy_dtype(output_name, not_node.output[0])

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # starting with opset-11, equal supports all numerical types (but both operands must be of the same type)
        # string type is not supported
        supported_dtypes = [
            TensorProto.BOOL,
            TensorProto.DOUBLE,
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.INT8,
            TensorProto.INT16,
            TensorProto.INT32,
            TensorProto.INT64,
            TensorProto.UINT8,
            TensorProto.UINT16,
            TensorProto.UINT32,
            TensorProto.UINT64
        ]
        target_dtype = TensorProto.INT32
        _add_cast_to_same_type_to_inputs(ctx, node, supported_dtypes, target_dtype)
        need_not = node.type == "NotEqual"
        if need_not:
            node.type = "Equal"
            output_name = node.output[0]
            not_node = ctx.insert_new_node_on_output("Not", output_name, name=utils.make_name(node.name))
            ctx.copy_shape(output_name, not_node.output[0])
            ctx.copy_dtype(output_name, not_node.output[0])


@tf_op(["Greater", "Less"])
class GreaterLess:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        common.BroadcastOp.version_1(ctx, node, **kwargs)

    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # T2 output = Greater(T1 x, T1 y), T2=tensor(bool)
        # T2 output = Less(T1 x, T1 y), T2=tensor(bool)
        # Great/Less in opset7 only supports limited types, insert Cast if needed
        supported_dtypes = [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.DOUBLE
        ]
        target_dtype = TensorProto.FLOAT
        _add_cast_to_inputs(ctx, node, supported_dtypes, target_dtype)

@tf_op(["GreaterEqual", "LessEqual"])
class GreaterLessEqual:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        GreaterLess.version_7(ctx, node, **kwargs)
        output_name = node.output[0]
        node.op.op_type = "Less" if node.op.op_type == "GreaterEqual" else "Greater"
        new_node = ctx.insert_new_node_on_output("Not", output_name, name=utils.make_name(node.name))
        ctx.copy_shape(output_name, new_node.output[0])
        ctx.set_dtype(new_node.output[0], ctx.get_dtype(output_name))

    @classmethod
    def version_12(cls, ctx, node, **kwargs):
        node.op.op_type = "GreaterOrEqual" if node.op.op_type == "GreaterEqual" else "LessOrEqual"
