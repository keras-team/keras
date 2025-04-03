# SPDX-License-Identifier: Apache-2.0


"""
tfl_math
"""

import logging
import numpy as np
from onnx.onnx_pb import TensorProto
from tf2onnx.handler import tfl_op
from tf2onnx import utils

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring,unused-variable,pointless-string-statement,invalid-name


def separate_fused_activation_function(ctx, node):
    activation_fn = node.attr['fused_activation_function'].s
    del node.attr['fused_activation_function']
    if activation_fn == b'RELU':
        ctx.insert_new_node_on_output("Relu", node.output[0])
    elif activation_fn == b'RELU6':
        # This is a TF op. We will convert it on the 2nd pass.
        shape = ctx.get_shape(node.output[0])
        dtype = ctx.get_dtype(node.output[0])
        new_node = ctx.make_node("Relu6", [node.output[0]], skip_conversion=False, shapes=[shape], dtypes=[dtype])
        ctx.insert_node_on_output(new_node, node.output[0])
    elif activation_fn == b'TANH':
        ctx.insert_new_node_on_output("Tanh", node.output[0])
    else:
        # TODO: SIGN_BIT and RELU_N1_TO_1 not supported yet
        utils.make_sure(activation_fn == b'NONE', "Unsupported fused activation function %s on node %s",
                        activation_fn, node.name)

@tfl_op(["TFL_ADD"], tf_op="Add")
class TflAdd:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        separate_fused_activation_function(ctx, node)

@tfl_op(["TFL_SUB"], tf_op="Sub")
class TflSub:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        separate_fused_activation_function(ctx, node)

@tfl_op(["TFL_MUL"], tf_op="Mul")
class TflMul:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        separate_fused_activation_function(ctx, node)

@tfl_op(["TFL_DIV"], tf_op="Div")
class TflDiv:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        separate_fused_activation_function(ctx, node)

@tfl_op(["TFL_LOGISTIC"], tf_op="Sigmoid")
class TflLogistic:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        pass

@tfl_op(["TFL_REDUCE_MAX"], tf_op="Max")
@tfl_op(["TFL_REDUCE_MIN"], tf_op="Min")
@tfl_op(["TFL_REDUCE_ANY"], tf_op="Any")
@tfl_op(["TFL_REDUCE_ALL"], tf_op="All")
@tfl_op(["TFL_REDUCE_PROD"], tf_op="Prod")
class TflReduceOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        pass

@tfl_op(["TFL_LOCAL_RESPONSE_NORMALIZATION"], tf_op="LRN")
class TFlLocalResponseNormalizationOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.attr["depth_radius"] = node.attr["radius"]
        del node.attr["radius"]

@tfl_op(["TFL_RANGE"], tf_op="Range")
class TflRangeOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.set_attr("Tidx", ctx.get_dtype(node.output[0]))

@tfl_op(["TFL_QUANTIZE"], onnx_op="QuantizeLinear")
class TflQuantizeOp:
    @classmethod
    def version_1(cls, ctx, node, dequantize=False, **kwargs):
        # We could just let the TFL_QUANTIZE fall through as an unconverted op, but they are added programmatically
        # so that might be confusing.
        raise ValueError("Opset 10 is required for quantization. Consider using the --dequantize flag or --opset 10.")

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        scale = node.get_attr_value('scale')
        zero_point = node.get_attr_value('zero_point')
        axis = node.get_attr_value('quantized_dimension')
        np_q_type = utils.map_onnx_to_numpy_type(ctx.get_dtype(node.output[0]))
        if len(scale) > 1 or len(zero_point) > 1:
            utils.make_sure(ctx.opset >= 13, "Opset 13 is required for per-axis quantization for node %s", node.name)
            node.set_attr("axis", axis)
        scale_node = ctx.make_const(utils.make_name("scale"), np.array(scale[0], dtype=np.float32))
        zero_point_node = ctx.make_const(utils.make_name("zero_point"), np.array(zero_point[0], dtype=np_q_type))
        ctx.replace_inputs(node, [node.input[0], scale_node.output[0], zero_point_node.output[0]])
        del node.attr["scale"]
        del node.attr["zero_point"]
        del node.attr["quantized_dimension"]
        if "min" in node.attr:
            del node.attr["min"]
        if "max" in node.attr:
            del node.attr["max"]

@tfl_op(["TFL_DEQUANTIZE"], onnx_op="DequantizeLinear")
class TflDequantizeOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        if 'scale' not in node.attr:
            # Somtimes tflite uses a Dequantize to go from fp16 to fp32
            node.type = "Cast"
            node.set_attr('to', ctx.get_dtype(node.output[0]))
            return
        scale = np.array(node.get_attr_value('scale'), dtype=np.float32)
        zero_point = np.array(node.get_attr_value('zero_point'), dtype=np.float32)
        axis = node.get_attr_value('quantized_dimension')
        in_rank = ctx.get_rank(node.input[0])
        def expand_tensor(t):
            if t.shape == (1,):
                return t[0]
            utils.make_sure(in_rank is not None, "Cannot dequantize node %s with unknown input rank", node.name)
            new_shape = [1] * in_rank
            new_shape[axis] = t.shape[0]
            return t.reshape(new_shape)
        scale = expand_tensor(scale)
        zero_point = expand_tensor(zero_point)
        if node.inputs[0].is_const():
            x_val = node.inputs[0].get_tensor_value(as_list=False).astype(np.float32)
            new_val = (x_val - zero_point) * scale
            dequant_const = ctx.make_const(utils.make_name(node.name), new_val)
            ctx.replace_all_inputs(node.output[0], dequant_const.output[0])
            ctx.remove_node(node.name)
        else:
            scale_const = ctx.make_const(utils.make_name(node.name + "_scale"), scale).output[0]
            zero_point_const = ctx.make_const(utils.make_name(node.name + "_zero_point"), zero_point).output[0]
            cast_node = ctx.make_node("Cast", [node.input[0]], attr={'to': TensorProto.FLOAT},
                                      op_name_scope=node.name).output[0]
            sub_node = ctx.make_node("Sub", [cast_node, zero_point_const], op_name_scope=node.name).output[0]
            mul_node = ctx.make_node("Mul", [sub_node, scale_const], op_name_scope=node.name).output[0]
            ctx.replace_all_inputs(node.output[0], mul_node)
            ctx.remove_node(node.name)

    @classmethod
    def version_10(cls, ctx, node, dequantize=False, **kwargs):
        if dequantize or 'scale' not in node.attr:
            cls.version_1(ctx, node, dequantize=True, **kwargs)
            return
        scale = node.get_attr_value('scale')
        zero_point = node.get_attr_value('zero_point')
        axis = node.get_attr_value('quantized_dimension')
        np_q_type = utils.map_onnx_to_numpy_type(ctx.get_dtype(node.input[0]))
        if len(scale) > 1 or len(zero_point) > 1:
            utils.make_sure(ctx.opset >= 13, "Opset 13 is required for per-axis quantization for node %s", node.name)
            node.set_attr("axis", axis)
            scale_node = ctx.make_const(utils.make_name("scale"), np.array(scale, dtype=np.float32))
            zero_point_node = ctx.make_const(utils.make_name("zero_point"), np.array(zero_point, dtype=np_q_type))
        else:
            scale_node = ctx.make_const(utils.make_name("scale"), np.array(scale[0], dtype=np.float32))
            zero_point_node = ctx.make_const(utils.make_name("zero_point"), np.array(zero_point[0], dtype=np_q_type))
        ctx.replace_inputs(node, [node.input[0], scale_node.output[0], zero_point_node.output[0]])
        del node.attr["scale"]
        del node.attr["zero_point"]
        del node.attr["quantized_dimension"]
        if "min" in node.attr:
            del node.attr["min"]
        if "max" in node.attr:
            del node.attr["max"]

def dynamic_quantize_inputs(ctx, node):
    if ctx.opset < 11:
        logger.warning("Opset 11 is required for asymmetric_quantize_inputs of node %s", node.name)
        return
    for i in range(len(node.input)):
        # Don't quantize inputs that are already quantized
        if node.inputs[i].type in ["DequantizeLinear", "TFL_DEQUANTIZE"]:
            continue
        dyn_quant = ctx.make_node("DynamicQuantizeLinear", [node.input[i]], output_count=3, op_name_scope=node.name)
        dyn_quant.skip_conversion = True
        dequant = ctx.make_node("DequantizeLinear", dyn_quant.output, op_name_scope=node.name)
        dequant.skip_conversion = True
        ctx.replace_input(node, node.input[i], dequant.output[0], input_index=i)

@tfl_op(["TFL_FULLY_CONNECTED"])
class TflFullyConnectedOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        separate_fused_activation_function(ctx, node)
        utils.make_sure(node.attr['weights_format'].s == b'DEFAULT',
                        "Only default weights format supported for fully connected op")
        utils.make_sure(node.attr['keep_num_dims'].i == 0,
                        "Only keep_num_dims=False supported for fully connected op")
        if node.attr['asymmetric_quantize_inputs'].i == 1:
            dynamic_quantize_inputs(ctx, node)

        if ctx.get_rank(node.input[0]) != 2:
            # When a fullyconnected node has keep_num_dims=0 and input[0] rank > 2, the extra dims must be compressed
            utils.make_sure(ctx.get_rank(node.input[1]) == 2, "weights for FullyConnected must have rank 2")
            weights_shape = ctx.get_shape(node.input[1])[1]
            utils.make_sure(weights_shape != -1, "weights for FullyConnected must have known shape")
            shape_const = ctx.make_const(utils.make_name("reshape_shape"), np.array([-1, weights_shape], np.int64))
            reshape_node = ctx.make_node("Reshape", [node.input[0], shape_const.output[0]])
            reshape_node.skip_conversion = True
            ctx.replace_inputs(node, [reshape_node.output[0], node.input[1]])

        transpose_node = ctx.insert_new_node_on_input(node, "Transpose", node.input[1],
                                                      name=None, input_index=1, perm=[1, 0])
        transpose_node.skip_conversion = True
        node.set_attr("transpose_a", 0)
        node.set_attr("transpose_b", 0)
        node.type = "MatMul"

        if len(node.input) == 3:
            # FIXME: Add a test for this
            bias_inp = node.input[2]
            ctx.replace_inputs(node, node.input[:2])
            add_node = ctx.insert_new_node_on_output("Add", node.output[0], inputs=[node.output[0], bias_inp])
            add_node.skip_conversion = True

        del node.attr["weights_format"]
        del node.attr["keep_num_dims"]
        del node.attr["asymmetric_quantize_inputs"]

@tfl_op(["TFL_SOFTMAX"], tf_op="Softmax")
class TFlSoftmaxOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        beta = node.get_attr_value("beta")
        if beta != 1:
            beta_node = ctx.make_const(utils.make_name("beta"), np.array(beta, dtype=np.float32))
            mul_node = ctx.insert_new_node_on_output("Mul", node.output[0], name=utils.make_name(node.name))
            ctx.replace_inputs(mul_node, [node.output[0], beta_node.output[0]])

@tfl_op(["TFL_PRELU"], onnx_op="PRelu")
class TflPreluOp:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        pass

@tfl_op(["TFL_UNSORTED_SEGMENT_MAX"], tf_op="UnsortedSegmentMax")
class TflUnsortedSegmentMax:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        pass

@tfl_op(["TFL_UNSORTED_SEGMENT_MIN"], tf_op="UnsortedSegmentMin")
class TflUnsortedSegmentMin:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        pass

@tfl_op(["TFL_UNSORTED_SEGMENT_PROD"], tf_op="UnsortedSegmentProd")
class TflUnsortedSegmentProd:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        pass

@tfl_op(["TFL_UNSORTED_SEGMENT_SUM"], tf_op="UnsortedSegmentSum")
class TflUnsortedSegmentSum:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        pass
