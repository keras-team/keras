# SPDX-License-Identifier: Apache-2.0

""" tf2onnx mapping functions for ms domain. """

import numpy as np

from onnx import onnx_pb
from onnx.onnx_pb import TensorProto
from tf2onnx import constants, utils
from tf2onnx.handler import tf_op
from tf2onnx.onnx_opset import controlflow
from tf2onnx.onnx_opset.nn import conv_convert_inputs, conv_dims_attr


# pylint: disable=unused-argument,missing-docstring

def make_range(ctx, start, limit, delta, output, scope_name, shape, dtype):
    if all(ctx.get_node_by_output(n).is_const() for n in [start, limit, delta]) is True:
        controlflow.make_range_const(ctx, start, limit, delta, output, scope_name, shape, dtype)
    else:
        _make_range_non_const(ctx, start, limit, delta, output, scope_name, shape, dtype)


def _make_range_non_const(ctx, start, limit, delta, output, scope_name, shape, dtype):
    utils.make_sure(
        dtype in [TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.INT16,
                  TensorProto.INT32, TensorProto.INT64,
                  TensorProto.COMPLEX64, TensorProto.COMPLEX128],
        "dtype %s is not supported", dtype)
    ctx.make_node("Range", [start, limit, delta], outputs=[output], name=scope_name, shapes=[shape], dtypes=[dtype],
                  domain=constants.MICROSOFT_DOMAIN)


@tf_op("Range", domain=constants.MICROSOFT_DOMAIN)
class Range:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        """Range."""
        # T range = Range(T start, T limit, T delta)
        dtype = node.get_attr_int("Tidx")
        shape = node.output_shapes[0]
        utils.make_sure(dtype is not None, "Tidx of %s is None", node.name)
        ctx.remove_node(node.name)
        make_range(ctx, node.input[0], node.input[1], node.input[2], node.output[0], node.name, shape, dtype)


@tf_op("Conv2DBackpropInput", domain=constants.MICROSOFT_DOMAIN, onnx_op="ConvTransposeWithDynamicPads")
class ConvTransposeWithDynamicPads:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T output = Conv2DBackpropInput(int32 input_sizes, T filter, T out_backprop,
        #    @list(int) strides, @bool use_cudnn_on_gpu, @string padding, @string data_format, @list(int) dilations)
        # T Y = ConvTranspose(T X, T W, T B, T pads, @STRING auto_pad, @INTS dilations,
        #    @INT group, @INTS kernel_shape, @INTS output_shape, @INTS strides)

        # tf uses "output_shape" while onnx uses "pads", the equation to calculate pads is:
        # total_padding[i] = stride[i] * (input_shape[i] - 1)+ kernel_shape[i] - output_shape[i]
        # pads[i_begin] = total_padding[i]/2
        # pads[i_end] = total_padding[i] - (total_padding[i]/2)
        # output dtype of onnx "shape" is int64 while in tf dtype could be specified
        utils.make_sure(node.is_nhwc(), "only support NHWC for now")
        node.domain = constants.MICROSOFT_DOMAIN
        input_shape = ctx.make_node("Shape", [node.input[2]])
        hw_indices = ctx.make_const(utils.make_name("hw_indices"), np.array([1, 2]).astype(np.int64))
        input_shape_hw = ctx.make_node("Gather", [input_shape.output[0], hw_indices.output[0]])
        output_shape = node.input[0]
        if ctx.get_dtype(output_shape) != onnx_pb.TensorProto.INT64:
            output_shape = ctx.make_node("Cast", [output_shape], attr={"to": onnx_pb.TensorProto.INT64}).output[0]
        output_shape_hw = ctx.make_node("Gather", [output_shape, hw_indices.output[0]])
        kernel_shape_hw = list(ctx.get_shape(node.input[1]))[0:2]
        kernel_shape = ctx.make_const(utils.make_name("const_convtrans"), np.array(kernel_shape_hw).astype(np.int64))
        strides = conv_dims_attr(node, "strides")
        utils.make_sure(len(strides) == 2, "only stride of H and W needed")

        stride_node = ctx.make_const(utils.make_name("const_convtrans"), np.array(strides).astype(np.int64))
        const_one = ctx.make_const(utils.make_name("cosnt_one"), np.array([1]).astype(np.int64))
        const_two = ctx.make_const(utils.make_name("cosnt_two"), np.array([2]).astype(np.int64))

        tmp0 = ctx.make_node("Sub", [input_shape_hw.output[0], const_one.output[0]])
        tmp1 = ctx.make_node("Mul", [stride_node.output[0], tmp0.output[0]])
        tmp2 = ctx.make_node("Add", [tmp1.output[0], kernel_shape.output[0]])
        total_pads = ctx.make_node("Sub", [tmp2.output[0], output_shape_hw.output[0]],
                                   dtypes=[onnx_pb.TensorProto.INT64])
        pads_beg = ctx.make_node("Div", [total_pads.output[0], const_two.output[0]], dtypes=[onnx_pb.TensorProto.INT64])
        pads_end = ctx.make_node("Sub", [total_pads.output[0], pads_beg.output[0]])
        pads = ctx.make_node("Concat", [pads_beg.output[0], pads_end.output[0]], attr={"axis": 0})
        # set node's attrs, Note: output_padding, group are left default.
        conv_dims_attr(node, "dilations")
        # set node's inputs from (output_shape, filter, input_tensor) to (input_tensor, filter, pads, Bias)
        ctx.replace_input(node, node.input[0], node.input[2], 0)
        ctx.replace_input(node, node.input[2], pads.output[0], 2)
        conv_convert_inputs(ctx, node, with_kernel=True)
        node.attr.pop("data_format")
        node.attr.pop("padding")
        if "explicit_paddings" in node.attr:
            node.attr.pop("explicit_paddings")

@tf_op("CropAndResize", domain=constants.MICROSOFT_DOMAIN)
class CropAndResize:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        """ utilize contrib cropandresize """
        node.attr['method'].name = 'mode'
        node.domain = constants.MICROSOFT_DOMAIN
        ctx.insert_new_node_on_input(node, "Transpose", node.input[0], perm=constants.NHWC_TO_NCHW)
        ctx.insert_new_node_on_output("Transpose", node.output[0], node.name + '_transposed',
                                      None, perm=constants.NCHW_TO_NHWC)

@tf_op("MatrixInverse", domain=constants.MICROSOFT_DOMAIN, onnx_op="Inverse")
class Inverse:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        utils.make_sure(node.get_attr('adjoint').i == 0, "adjoint must be false")
        del node.attr["adjoint"]
        node.domain = constants.MICROSOFT_DOMAIN
