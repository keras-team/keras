# SPDX-License-Identifier: Apache-2.0


"""
nn
"""

import logging

import numpy as np
from onnx import onnx_pb, helper
from onnx.onnx_pb import TensorProto
from tf2onnx import constants, utils
from tf2onnx.graph_builder import GraphBuilder
from tf2onnx.handler import tf_op
from tf2onnx.onnx_opset import common, controlflow, tensor

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring,unused-variable

def spatial_map(shape, perm):
    new_shape = shape[:]
    for i in perm:
        new_shape[i] = shape[perm[i]]
    return new_shape


def is_channels_last(node):
    """Returns whether node is channels last, so (N, ..., C)."""

    return not node.data_format.startswith("NC")


def make_shape_channels_first(shape):
    """Makes a (N, ..., C) shape into (N, C, ...)."""

    return shape[:1] + shape[-1:] + shape[1:-1]


def make_shape_channels_last(shape):
    """Makes a (N, C, ...) shape into (N, ..., C)."""

    return shape[:1] + shape[1:-1] + shape[1:2]


def get_channels_first_permutation(spatial):
    """Returns a permutation to make a (N, ..., C) array into (N, C, ...)."""

    return [0, spatial + 1] + list(range(1, spatial + 1))


def get_channels_last_permutation(spatial):
    """Returns a permutation to make a (N, C, ...) array into (N, ..., C)."""

    return [0] + list(range(2, spatial + 2)) + [1]


def conv_convert_inputs(ctx, node, with_kernel=False, new_kernel_shape=None,
                        input_indices=None, output_indices=None, spatial=2,
                        quantization_axis=0):
    """Convert input and kernel from tensorflow to onnx. This may be required to
        insert transpose ops for input, kernel, and output unless they are constants
        and we can transpose the constant.
        We transpose inputs if they are in NHWC. We always transpose the kernel from
        HWNC to NCHW. Outputs are transposed if the format is NHWC.
        Some convolutions like depthwise_conv2d require a reshape of the kernel.

    Args:
        ctx: The parent graph.
        node: Node of the convolution op.
        with_kernel: Transpose the kernel.
        new_kernel_shape: Pass to reshape the kernel.
        input_indices: Indices that define the inputs.
        output_indices: Indices that define the outputs.
        quantization_axis: Axis for the inserted QDQ nodes
    """

    if input_indices is None:
        input_indices = [0]
    if output_indices is None:
        output_indices = [0]

    # Transpose inputs if needed.
    if is_channels_last(node):
        # Ge channels first permutation.
        permutation = get_channels_first_permutation(spatial)

        # Transpose input if needed, no need to record shapes on input
        for idx in input_indices:
            # If input is a constant, transpose that one if we are the only consumer.
            input_node = node.inputs[idx]
            input_name = node.input[idx]

            if input_node.is_const() and len(ctx.find_output_consumers(input_name)) == 1:
                # Transpose constant to make it channels first.
                val = input_node.get_tensor_value(as_list=False)
                val = np.transpose(val, permutation)

                input_node.set_tensor_value(val)
            else:
                # Insert transpose op.
                transpose = ctx.insert_new_node_on_input(node, "Transpose", input_name)
                transpose.set_attr("perm", permutation)
                transpose.skip_conversion = True

                shape = ctx.get_shape(input_name)
                if shape is not None:
                    new_shape = make_shape_channels_first(shape)

                    ctx.set_shape(transpose.output[0], new_shape)

    # Transpose kernel if needed.
    if with_kernel:
        # Some ONNX convolution ops require to reshape the kernel (ie. depthwise_conv2d).
        if new_kernel_shape:
            kernel_name = node.input[1]
            if ctx.opset < 5:
                # Old reshape takes new shape as attribute.
                reshape = ctx.insert_new_node_on_input(node, "Reshape", kernel_name)
                reshape.set_attr("shape", new_kernel_shape)
                reshape.skip_conversion = True
            else:
                # New reshape takes new shape as input[1].
                shape_name = utils.make_name(node.name)
                ctx.make_const(shape_name, np.array(new_kernel_shape, dtype=np.int64))

                reshape = ctx.make_node("Reshape", [kernel_name, shape_name])
                ctx.replace_input(node, kernel_name, reshape.output[0], 1)

                reshape.skip_conversion = True
            ctx.set_shape(reshape.output[0], new_kernel_shape)

        # Get kernel (may have be changed to a reshape above).
        kernel_node = node.inputs[1]
        kernel_name = node.input[1]

        # Transpose kernel from (..., C_in, C_out) to (C_out, C_in, ...)
        permutation = [spatial + 1, spatial] + list(range(spatial))

        # If kernel is a constant, transpose that one if we are the only consumer.
        need_transpose = True
        if (kernel_node.is_const() or kernel_node.op.op_type == "DequantizeLinear") \
            and len(ctx.find_output_consumers(kernel_name)) == 1:
            if kernel_node.op.op_type == 'DequantizeLinear':
                # Assuming the model was trained in NHWC in TF,
                # the weights would be in [fH, fW, C_in, C_out].
                # orig_conv_weights -> Q -> DQ -> new_conv_weights -> conv
                weights_node = kernel_node.inputs[0].inputs[0]
                val = weights_node.get_tensor_value(as_list=False)
                val = np.transpose(val, permutation)
                weights_node.set_tensor_value(val)
                need_transpose = False
                # Change the quantization axis for Q and DQ node accordingly
                kernel_node.set_attr("axis", quantization_axis) # DQ node
                kernel_node.inputs[0].set_attr("axis", quantization_axis) # Q node
            else:
                val = kernel_node.get_tensor_value(as_list=False)
                val = np.transpose(val, permutation)
                kernel_node.set_tensor_value(val)
                need_transpose = False

        if need_transpose:
            transpose = ctx.insert_new_node_on_input(node, "Transpose", kernel_name)
            transpose.set_attr("perm", permutation)
            transpose.skip_conversion = True

            new_shape = spatial_map(ctx.get_shape(kernel_name), permutation)
            ctx.set_shape(transpose.output[0], new_shape)

    # Transpose outputs back if needed.
    if is_channels_last(node):
        for idx in output_indices:
            # Make output channels last again by transposing.
            output_name = node.output[idx]
            output_shape = ctx.get_shape(node.output[idx])

            permutation = get_channels_last_permutation(spatial)

            op_name = utils.make_name(node.name)
            transpose = ctx.insert_new_node_on_output("Transpose", output_name, name=op_name)

            transpose.set_attr("perm", permutation)
            transpose.skip_conversion = True

            # Set tensorflow channels last shape as the transpose node shape.
            ctx.set_shape(transpose.output[0], output_shape)

            # Make the current ONNX convolution output shape channels first.
            ctx.set_shape(output_name, make_shape_channels_first(output_shape))

        # NOTE: Not strictly correct as it can also be NCW or NCDHW for example.
        # NOTE: Generally speaking it's channels first.
        node.data_format = "NCHW"


def add_padding(ctx, node, kernel_shape, strides, dilations=None, spatial=2):
    padding = node.get_attr("padding")
    if not padding:
        return

    if dilations is None:
        dilations = [1] * spatial

    padding = padding.s.decode("utf-8")
    if padding == "SAME":
        # Initialize with all zeros.
        # Paddings are in (x_begin, y_begin, ..., x_end, y_end, ...) order.
        pads = [0] * (spatial * 2)

        # Get shapes and check whether valid.
        input_shape = ctx.get_shape(node.input[0])
        output_shape = ctx.get_shape(node.output[0])

        if len(input_shape) != spatial + 2:
            raise ValueError(
                "node {} output needs to be rank {}, is {}".format(
                    node.name, spatial + 2, len(input_shape)
                )
            )

        if len(output_shape) != spatial + 2:
            raise ValueError(
                "node {} output needs to be rank {}, is {}".format(
                    node.name, spatial + 2, len(output_shape)
                )
            )

        # Transpose to channels first if not so.
        if is_channels_last(node):
            input_shape = make_shape_channels_first(input_shape)
            output_shape = make_shape_channels_first(output_shape)

        # Check for unknown input/output dimensions. Fall back to auto padding if so.
        if any(input_shape[i + 2] == -1 or output_shape[i + 2] == -1 for i in range(spatial)):
            logger.debug(
                "node %s has unknown dim for pads calculation, fallback to auto_pad: "
                "input_shape=%s, output_shape=%s",
                node.name,
                input_shape,
                output_shape,
            )

            node.set_attr("auto_pad", "SAME_UPPER")
            return

        # Calculate paddings.
        for i in range(spatial):
            pad = (
                (output_shape[i + 2] - 1) * strides[i]
                + dilations[i] * (kernel_shape[i] - 1) + 1
                - input_shape[i + 2]
            )
            pad = max(pad, 0)

            pads[i] = pad // 2
            pads[i + spatial] = pad - pad // 2

        node.set_attr("pads", pads)
    elif padding == "VALID":
        pass
    elif padding == "EXPLICIT":
        pads = node.get_attr_value("explicit_paddings")
        start_pads = []
        end_pads = []
        d = 1 if is_channels_last(node) else 2
        for i in range(spatial):
            start_pads.append(pads[(d + i) * 2])
            end_pads.append(pads[(d + i) * 2 + 1])
        node.set_attr("pads", start_pads + end_pads)
    else:
        raise ValueError("invalid padding value: {}".format(padding))

def parse_dims_attr(node, dims, spatial):
    if is_channels_last(node):
        # We have (N, ..., C) or (...).
        if len(dims) != spatial:
            dims = dims[1:-1]
    else:
        # We have (N, C, ...) or (...).
        if len(dims) != spatial:
            dims = dims[2:]
    return dims

def conv_dims_attr(node, name, new_name=None, spatial=2):
    # Fetch attribute.
    if new_name is None:
        new_name = name

    dims = node.get_attr(name)
    if not dims:
        return None

    # Get spatial part.
    dims = dims.ints
    dims = parse_dims_attr(node, dims, spatial)

    # Set new value and return it.
    node.set_attr(new_name, dims)

    return dims


def conv_kernel_shape(ctx, node, input_idx, spatial=2):
    # Kernel shape is (..., C_in, C_out).
    kernel_shape = ctx.get_shape(node.input[input_idx])
    if len(kernel_shape) != spatial + 2:
        raise ValueError("kernel rank must be spatial+2")

    # Get spatial part.
    kernel_shape = kernel_shape[:spatial]

    # Set attribute value only if all dimensions are known.
    if all(d > 0 for d in kernel_shape):
        node.set_attr("kernel_shape", kernel_shape)

    return kernel_shape


def build_dynamic_target_size(ctx, transposed_intput, target_hw):
    """
    Build the target tensor shape for the Resize op.

    Args:
        - ctx: the graph context
        - transposed_intput: A tensor of rank 4 of shape [n c h w]
        - target_hw: tensor of rank 2 containing the target size for a resize: [nh nw]

    Returns:
        A tensor of rank 2 containing [n c nh nw]
    """
    # We get the first half [n c] of the target shape
    shape_of_transposed_input = ctx.make_node("Shape", [transposed_intput])
    first_half_of_shape = GraphBuilder(ctx).make_slice(
        {"data": shape_of_transposed_input.output[0], "ends": [2], "starts": [0]})
    if ctx.get_dtype(target_hw) != TensorProto.INT64:
        target_hw = ctx.make_node("Cast", [target_hw], attr={'to': TensorProto.INT64}).output[0]
    # We build a tensor containing [n c nh nw]
    final_target_size = ctx.make_node("Concat", [first_half_of_shape, target_hw], {'axis': 0})
    return final_target_size


@tf_op(["Conv1D", "Conv2D", "Conv3D"])
class ConvOp:
    @classmethod
    def any_version(cls, opset, ctx, node, **kwargs):
        # ONNX specification:
        #
        # T output = Conv2D(T input, T filter, @list(int) strides, @bool use_cudnn_on_gpu,
        #                       @string padding, @string data_format)
        #
        # T Y = Conv(T X, T W, T B, @AttrType.STRING auto_pad, @AttrType.INTS dilations, @AttrType.INT group,
        #                       @AttrType.INTS kernel_shape, @AttrType.INTS pads, @AttrType.INTS strides)
        #

        # Determine number of spatial dimensions.
        spatial = int(node.type[-2])

        # Make it a convolution node.
        node.type = "Conv"

        # Determine kernel spatial shape, strides and dilations.
        kernel_shape = conv_kernel_shape(ctx, node, 1, spatial=spatial)
        strides = conv_dims_attr(node, "strides", spatial=spatial)
        dilations = conv_dims_attr(node, "dilations", spatial=spatial)

        # prefix with batch dim of [1] to satisfy rank requirements
        input_shape = ctx.get_shape(node.input[0])
        if input_shape is not None and len(input_shape) == spatial + 1:
            gb = GraphBuilder(ctx)
            usq_node = gb.make_unsqueeze({"axes": [0], 'data': node.input[0]}, return_node=True)
            ctx.replace_inputs(node, [usq_node.output[0]] + node.input[1:])

        # Set padding.
        add_padding(
            ctx, node, kernel_shape, strides, dilations=dilations, spatial=spatial
        )
        groups = int(1)
        data_format = str(node.attr["data_format"].s, encoding="utf8")
        shape_dim = -1
        if data_format == "NHWC":
            shape_dim = ctx.get_shape(node.input[0])[-1]
        elif data_format == "NCHW":
            shape_dim = ctx.get_shape(node.input[0])[1]
        if shape_dim != -1:
            filter_in_channels = ctx.get_shape(node.input[1])[-2]
            if filter_in_channels != -1:
                groups = shape_dim // filter_in_channels

        node.set_attr("group", groups)

        # Convert input and filters.
        conv_convert_inputs(ctx, node, with_kernel=True, spatial=spatial)

    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        cls.any_version(1, ctx, node, **kwargs)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # No change.
        cls.any_version(11, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Signature change for operator Unsqueeze.
        cls.any_version(13, ctx, node, **kwargs)


def get_shape_from_const_or_concat(ctx, node):
    if node.is_const():
        return node.get_tensor_value()
    if node.type == 'Concat':
        # Sometimes the shape is formed by concating a bunch of consts together
        res = []
        if any(ctx.get_shape(inp) != [1] for inp in node.input):
            return None
        for i, inp in enumerate(node.inputs):
            # The concat is converted from a Pack. Conversion adds an unsqueeze to the inputs.
            if node.inputs[i].type == 'Unsqueeze' and node.inputs[i].inputs[0].is_scalar():
                res.append(node.inputs[i].inputs[0].get_tensor_value())
            else:
                if i == 0:
                    # For the batch dimension we don't care if it is unknown
                    res.append(-1)
                else:
                    return None
        return res
    return None

@tf_op(["Conv2DBackpropInput", "Conv3DBackpropInputV2"])
class ConvTranspose:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T output = Conv2DBackpropInput(int32 input_sizes, T filter, T out_backprop,
        #    @list(int) strides, @bool use_cudnn_on_gpu, @string padding, @string data_format, @list(int) dilations)
        # T Y = ConvTranspose(T X, T W, T B, @STRING auto_pad, @INTS dilations,
        #    @INT group, @INTS kernel_shape, @INTS output_shape, @INTS pads, @INTS strides)

        if node.type == "Conv3DBackpropInputV2":
            spatial = 3
        else:
            spatial = 2
        node.type = "ConvTranspose"
        # Note: inputs are reversed from what one would expect.
        conv_kernel_shape(ctx, node, 1, spatial=spatial)
        input_shape = ctx.get_shape(node.input[2])
        input_batch_dim = input_shape[0]
        output_c_dim = ctx.get_shape(node.input[1])[-2]
        if is_channels_last(node):
            input_dims = input_shape[1:1+spatial]
        else:
            input_dims = input_shape[2:2+spatial]
        output_shape_orig = node.output_shapes

        # output_shape is explicitly specified here and then converted to explicit pads.
        output_shape = get_shape_from_const_or_concat(ctx, node.inputs[0])
        if output_shape is not None:
            if is_channels_last(node):
                new_output_shape = [output_shape[1], output_shape[2]]
                if spatial == 3:
                    new_output_shape.append(output_shape[3])
            else:
                new_output_shape = [output_shape[2], output_shape[3]]
                if spatial == 3:
                    new_output_shape.append(output_shape[4])

            utils.make_sure(new_output_shape.count(-1) <= 0, "output dims need to be known")
            utils.make_sure(all(new_output_shape[i] >= input_dims[i] for i in range(spatial)),
                            "output dims cannot be smaller than input dims.")
            if -1 in input_dims:
                node.set_attr("output_shape", new_output_shape)
            else:
                if "strides" in node.attr:
                    strides = parse_dims_attr(node, node.get_attr("strides").ints, spatial)
                else:
                    strides = [1] * spatial
                if "dilations" in node.attr:
                    dilations = parse_dims_attr(node, node.get_attr("dilations").ints, spatial)
                else:
                    dilations = [1] * spatial
                kernel_shape = parse_dims_attr(node, node.get_attr("kernel_shape").ints, spatial)
                total_padding = [-1] * spatial
                pads = [1] * (spatial * 2)
                for i in range(spatial):
                    total_padding[i] = (strides[i] * (input_dims[i] - 1)
                                        + ((kernel_shape[i] - 1) * dilations[i] + 1)
                                        - new_output_shape[i])
                    start_i = i
                    end_i = i + spatial
                    pads[start_i] = int(total_padding[i] / 2)
                    pads[end_i] = total_padding[i] - pads[start_i]
                node.set_attr("pads", pads)
                node.set_attr("auto_pad", "NOTSET")
        else:
            utils.make_sure(ctx.opset >= 10, "Opset 10 needed for Conv Backprop Input with non-constant shape")
            strides = parse_dims_attr(node, node.get_attr('strides').ints, spatial)
            if 'dilations' in node.attr:
                dilations = parse_dims_attr(node, node.get_attr('dilations').ints, spatial)
            else:
                dilations = [1] * spatial
            kernel_shape = parse_dims_attr(node, node.get_attr('kernel_shape').ints, spatial)
            new_dims = [-1] * spatial
            for i in range(spatial):
                new_dims[i] = strides[i] * (input_dims[i] - 1) + ((kernel_shape[i] - 1) * dilations[i] + 1)
            if is_channels_last(node):
                new_shape = [input_batch_dim] + new_dims + [output_c_dim]
            else:
                new_shape = [input_batch_dim, output_c_dim] + new_dims
            ctx.set_shape(node.output[0], new_shape)

            use_strides_workaround = any(d > 1 for d in strides)
            if use_strides_workaround and ctx.opset < 12:
                # When strides > 1, ONNX and TF have an implementation difference in ConvTranspose. ONNX outputs a
                # slightly smaller tensor which must be padded with a row of 0s. Pad with dynamic shape requires
                # opset >= 11 and Max of int64 needs opset >= 12.  Depending on the output_shape, this row of 0s might
                # be shaved off, in which case TF and ONNX agree.  When output_shape is dynamic it is impossible to
                # know at conversion time whether this is the case and the workaround is needed.
                logger.warning("Conv Backprop Input with strides > 1 and non-constant shape has known bug. "
                               "Workaround requires opset 12.")
                use_strides_workaround = False
            input_shape = ctx.make_node("Cast", [node.input[0]], attr={'to': TensorProto.INT64})
            output_shape = ctx.make_node("Shape", [node.output[0]])
            output_h = GraphBuilder(ctx).make_slice(
                {"data": output_shape.output[0], "ends": [2], "starts": [1], "axes": [0]})
            output_w = GraphBuilder(ctx).make_slice(
                {"data": output_shape.output[0], "ends": [3], "starts": [2], "axes": [0]})
            expect_h = GraphBuilder(ctx).make_slice(
                {"data": input_shape.output[0], "ends": [2], "starts": [1], "axes": [0]})
            expect_w = GraphBuilder(ctx).make_slice(
                {"data": input_shape.output[0], "ends": [3], "starts": [2], "axes": [0]})
            diff_h = ctx.make_node("Sub", [output_h, expect_h])
            diff_w = ctx.make_node("Sub", [output_w, expect_w])
            nonneg_diff_h = diff_h
            nonneg_diff_w = diff_w

            if use_strides_workaround:
                const_zero = ctx.make_const(utils.make_name(node.name + "_const_zero"), np.array([0], dtype=np.int64))
                nonneg_diff_h = ctx.make_node("Max", [diff_h.output[0], const_zero.output[0]])
                nonneg_diff_w = ctx.make_node("Max", [diff_w.output[0], const_zero.output[0]])

            const_two = ctx.make_const(utils.make_name(node.name + "_const_two"), np.array([2], dtype=np.int64))
            start_h = ctx.make_node("Div", [nonneg_diff_h.output[0], const_two.output[0]])
            start_w = ctx.make_node("Div", [nonneg_diff_w.output[0], const_two.output[0]])
            end_h = ctx.make_node("Add", [start_h.output[0], expect_h])
            end_w = ctx.make_node("Add", [start_w.output[0], expect_w])
            if spatial == 3:
                output_d = GraphBuilder(ctx).make_slice(
                    {"data": output_shape.output[0], "ends": [4], "starts": [3], "axes": [0]})
                expect_d = GraphBuilder(ctx).make_slice(
                    {"data": input_shape.output[0], "ends": [4], "starts": [3], "axes": [0]})
                diff_d = ctx.make_node("Sub", [output_d, expect_d])
                nonneg_diff_d = diff_d
                if use_strides_workaround:
                    nonneg_diff_d = ctx.make_node("Max", [diff_d.output[0], const_zero.output[0]])
                start_d = ctx.make_node("Div", [nonneg_diff_d.output[0], const_two.output[0]])
                end_d = ctx.make_node("Add", [start_d.output[0], expect_d])

                starts = ctx.make_node("Concat", [start_h.output[0], start_w.output[0], start_d.output[0]],
                                       attr={"axis": 0})
                ends = ctx.make_node("Concat", [end_h.output[0], end_w.output[0], end_d.output[0]], attr={"axis": 0})
                slice_axes = ctx.make_const(utils.make_name(node.name + "_const_slice_axes"),
                                            np.array([1, 2, 3], dtype=np.int64))
            else:
                starts = ctx.make_node("Concat", [start_h.output[0], start_w.output[0]], attr={"axis": 0})
                ends = ctx.make_node("Concat", [end_h.output[0], end_w.output[0]], attr={"axis": 0})
                slice_axes = ctx.make_const(utils.make_name(node.name + "_const_slice_axes"),
                                            np.array([1, 2], dtype=np.int64))

            slice_node = ctx.make_node("Slice",
                                       [node.output[0], starts.output[0], ends.output[0], slice_axes.output[0]],
                                       shapes=output_shape_orig)

            final_node = slice_node

            if use_strides_workaround:
                cz = const_zero.output[0]

                neg_diff_h = ctx.make_node("Neg", [diff_h.output[0]])
                shrink_h_by = ctx.make_node("Max", [neg_diff_h.output[0], const_zero.output[0]])
                shb = shrink_h_by.output[0]

                neg_diff_w = ctx.make_node("Neg", [diff_w.output[0]])
                shrink_w_by = ctx.make_node("Max", [neg_diff_w.output[0], const_zero.output[0]])
                swb = shrink_w_by.output[0]

                if spatial == 3:
                    neg_diff_d = ctx.make_node("Neg", [diff_d.output[0]])
                    shrink_d_by = ctx.make_node("Max", [neg_diff_d.output[0], const_zero.output[0]])
                    sdb = shrink_d_by.output[0]
                    pads = ctx.make_node("Concat", [cz, cz, cz, cz, cz, cz, shb, swb, sdb, cz], attr={"axis": 0})
                    padded_node = ctx.make_node("Pad", [slice_node.output[0], pads.output[0]])
                else:
                    pads = ctx.make_node("Concat", [cz, cz, cz, cz, cz, shb, swb, cz], attr={"axis": 0})
                    padded_node = ctx.make_node("Pad", [slice_node.output[0], pads.output[0]])

                final_node = padded_node

            downstream_nodes = ctx.find_output_consumers(node.output[0])
            downstream_nodes.remove(output_shape)
            downstream_nodes.remove(slice_node)
            ctx.replace_all_inputs(node.output[0], final_node.output[0], ops=downstream_nodes)

        conv_dims_attr(node, "strides", spatial=spatial)
        conv_dims_attr(node, "dilations", spatial=spatial)

        # remove output_shapes input
        ctx.remove_input(node, node.input[0], 0)
        # swap data and kernel
        t = node.input[0]
        ctx.replace_input(node, node.input[0], node.input[1], 0)
        ctx.replace_input(node, node.input[1], t, 1)

        conv_convert_inputs(ctx, node, with_kernel=True, spatial=spatial, quantization_axis=1)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        cls.version_1(ctx, node, **kwargs)


@tf_op(["DepthwiseConv2d", "DepthwiseConv2dNative"])
class DepthwiseConv2d:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T output = DepthwiseConv2dNative(T input, T filter, @list(int) strides, @string padding, @string data_format)
        # T Y = ConvTranspose(T X, T W, T B, @AttrType.STRING auto_pad, @AttrType.INTS dilations, @AttrType.INT group,
        #        @AttrType.INTS kernel_shape, @AttrType.INTS output_shape, @AttrType.INTS pads, @AttrType.INTS strides)
        #
        # this is not documented well in onnx, the hint comes from pytorch documentation:
        # http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        #   The configuration when groups == in_channels and out_channels = K * in_channels
        #   where K is a positive integer is termed in literature as depthwise convolution.
        #   In other words, for an input of size (N,Cin,Hin,Win),
        #   if you want a depthwise convolution with a depthwise multiplier K,
        #   then you use the constructor arguments (in_channels=Cin,out_channels=Cin*K,...,groups=Cin)
        #
        node.type = "Conv"
        input_shape = ctx.get_shape(node.input[0])
        if len(input_shape) != 4:
            raise ValueError("only Conv2D is supported")

        kernel_shape = ctx.get_shape(node.input[1])
        if len(kernel_shape) != 4:
            raise ValueError("only Conv2D is supported")
        k_h, k_w, k_input_channels, k_channel_multiplier = kernel_shape
        if "depth_multiplier" in node.attr:
            depth_multiplier = node.get_attr_int("depth_multiplier")
            k_input_channels //= depth_multiplier
            k_channel_multiplier *= depth_multiplier
        if k_input_channels < 1:
            raise ValueError("input channel must be positive")
        k_output_channels = k_input_channels * k_channel_multiplier

        if k_h > 0 and k_w > 0:
            node.set_attr("kernel_shape", [k_h, k_w])
        strides = conv_dims_attr(node, "strides")
        dilations = conv_dims_attr(node, "dilations")
        node.set_attr("group", k_input_channels)
        add_padding(ctx, node, kernel_shape, strides, dilations)

        new_kernel_shape = [k_h, k_w, 1, k_output_channels]
        conv_convert_inputs(ctx, node, with_kernel=True, new_kernel_shape=new_kernel_shape)


@tf_op(["AvgPool", "AvgPool3D"], onnx_op="AveragePool")
@tf_op(["MaxPool", "MaxPoolV2", "MaxPool3D"], onnx_op="MaxPool")
class PoolOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        cls._convert(ctx, node, **kwargs)

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        cls._convert(ctx, node, **kwargs)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # no change
        cls._convert(ctx, node, **kwargs)

    @classmethod
    def _convert(cls, ctx, node, **kwargs):
        # T output = MaxPool(T input, @list(int) ksize, @list(int) strides, @string padding, @string data_format)
        # T Y = MaxPool(T X, @AttrType.STRING auto_pad, @AttrType.INTS kernel_shape, @AttrType.INTS pads,
        #               @AttrType.INTS strides)
        # above seems wrong - input[1] is ksize, input[2] is strides
        # stride and ksize in tf is not always NHWC, so watch out when converting into onnx's NCHW
        if kwargs["tf_op"] in ["AvgPool3D", "MaxPool3D"]:
            spatial = 3
        else:
            spatial = 2

        origin_dtype = ctx.get_dtype(node.output[0])
        if origin_dtype not in [onnx_pb.TensorProto.FLOAT16, onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.DOUBLE]:
            # the onnx spec doesn't allow int types for pool ops
            input_shapes = [ctx.get_shape(node.input[0])]
            output_shapes = [ctx.get_shape(node.output[0])]
            cast_node = ctx.make_node("Cast", [node.input[0]], dtypes=[onnx_pb.TensorProto.FLOAT], shapes=input_shapes,
                                      name=node.name + "_cast", attr={"to": onnx_pb.TensorProto.FLOAT})
            _ = ctx.insert_node_on_output(cast_node, node.inputs[0].output[0])
            cast_back_node = ctx.make_node("Cast", [node.output[0]], dtypes=[origin_dtype], shapes=output_shapes,
                                           name=node.name + "_castback", attr={"to": origin_dtype})
            _ = ctx.insert_node_on_output(cast_back_node, node.output[0])
            ctx.set_dtype(node.output[0], onnx_pb.TensorProto.FLOAT)

        if len(node.input) < 3:
            kernel_shape_tf = node.get_attr("ksize").ints
            strides_tf = node.get_attr("strides").ints
        else:
            kernel_shape_tf = node.inputs[1].get_tensor_value()
            strides_tf = node.inputs[2].get_tensor_value()
            ctx.remove_input(node, node.input[2], 2)
            ctx.remove_input(node, node.input[1], 1)

        kernel_shape_hw = parse_dims_attr(node, kernel_shape_tf, spatial)
        strides_hw = parse_dims_attr(node, strides_tf, spatial)

        node.set_attr("kernel_shape", kernel_shape_hw)
        node.set_attr("strides", strides_hw)
        dilations = conv_dims_attr(node, "dilations", spatial=spatial)
        add_padding(ctx, node, kernel_shape_hw, strides_hw, dilations=dilations, spatial=spatial)
        conv_convert_inputs(ctx, node, with_kernel=False, spatial=spatial)


@tf_op(["MaxPoolWithArgmax"], onnx_op="MaxPool")
class MaxPoolWithArgmaxOp:
    @classmethod
    def version_8(cls, ctx, node, **kwargs):
        # T output = MaxPool(T input, @list(int) ksize, @list(int) strides, @string padding, @string data_format)

        # Set kernel_shape attribute
        kernel_shape = node.get_attr("ksize").ints
        kernel_shape = [kernel_shape[1], kernel_shape[2]]
        node.set_attr("kernel_shape", kernel_shape)

        # Set strides attribute
        strides = node.get_attr("strides").ints
        strides = [strides[1], strides[2]]
        node.set_attr("strides", strides)

        # The input data_format is NHWC for TF MaxPoolWithArgmax
        node.set_attr("data_format", "NHWC")

        # Convert indices from NCHW to NHWC format
        input_shape = ctx.make_node("Shape", [node.input[0]]).output[0]
        input_shape_guess = ctx.get_shape(node.input[0])
        n, h, w, c = ctx.make_node("Split", [input_shape], attr={'axis': 0, 'num_outputs': 4}, output_count=4).output
        hw = ctx.make_node("Mul", [h, w]).output[0]
        chw = ctx.make_node("Mul", [hw, c]).output[0]
        consumers = ctx.find_output_consumers(node.output[1])
        if ctx.opset >= 10:
            xy = ctx.make_node("Mod", [node.output[1], hw]).output[0]
        else:
            xy_div = ctx.make_node("Div", [node.output[1], hw]).output[0]
            xy_mul = ctx.make_node("Mul", [xy_div, hw]).output[0]
            xy = ctx.make_node("Sub", [node.output[1], xy_mul]).output[0]
        xy_scale_c = ctx.make_node("Mul", [xy, c]).output[0]
        const_zero = ctx.make_const(utils.make_name("const_zero"), np.array(0, np.int64)).output[0]
        const_one = ctx.make_const(utils.make_name("const_one"), np.array(1, np.int64)).output[0]
        if input_shape_guess is not None and input_shape_guess[3] > 0:
            c_range_np = np.arange(input_shape_guess[3], dtype=np.int64)
            c_range = ctx.make_const(utils.make_name("c_range"), c_range_np).output[0]
        else:
            utils.make_sure(ctx.opset >= 11, "opset 11 required for MaxPoolWithArgmax with non-const num channels")
            c_sq = GraphBuilder(ctx).make_squeeze({'data': c, 'axes': [0]})
            c_range = ctx.make_node("Range", [const_zero, c_sq, const_one]).output[0]
        xyc = ctx.make_node("Add", [xy_scale_c, c_range]).output[0]
        single_batch = input_shape_guess is not None and input_shape_guess[0] == 1
        # Documentation says include_batch_in_index has default False, but tf 1.13 excludes it and assumes True
        if node.get_attr_value('include_batch_in_index', True) and not single_batch:
            utils.make_sure(ctx.opset >= 11, "opset 11 required for MaxPoolWithArgmax with include_batch_in_index")
            n_sq = GraphBuilder(ctx).make_squeeze({'data': n, 'axes': [0]})
            n_range = ctx.make_node("Range", [const_zero, n_sq, const_one]).output[0]
            n_range_unsq = GraphBuilder(ctx).make_unsqueeze({'data': n_range, 'axes': [1, 2, 3]})
            n_range_scale = ctx.make_node("Mul", [n_range_unsq, chw]).output[0]
            result = ctx.make_node("Add", [xyc, n_range_scale]).output[0]
        else:
            result = xyc
        ctx.replace_all_inputs(node.output[1], result, ops=consumers)

        add_padding(ctx, node, kernel_shape, strides)
        conv_convert_inputs(ctx, node, with_kernel=False, input_indices=[0], output_indices=[0, 1])


@tf_op(["BiasAdd", "BiasAddV1"])
class BiasAdd:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T output = BiasAdd(T value, T bias, @string data_format)
        # T output = BiasAddV1(T value, T bias)
        # TODO: for now use add. We may need to convert to NCHW.
        node.type = "Add"
        common.BroadcastOp.version_1(ctx, node, **kwargs)

    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # T output = BiasAdd(T value, T bias, @string data_format)
        # T output = BiasAddV1(T value, T bias)
        # According TF bias_add definition, the input dim is always only 1.
        node.type = "Add"
        common.BroadcastOp.version_6(ctx, node, **kwargs)

        # on NHWC, bias will broadcast from largest dim, which is default onnx Add op broadcast behavior.
        if not node.is_nhwc():
            # however, in NCHW, bias should be at 2nd dim, which by default onnx Add op has no way to know,
            # so it needs being reshaped into 3-dim tensor before add
            shape0 = ctx.get_shape(node.input[0])
            shape1 = ctx.get_shape(node.input[1])
            if node.inputs[1].type == 'Const' and len(shape1) == 1:
                new_broadcast_shape = [shape1[0]] + [1] * (len(shape0) - 2)
                shape_name = utils.make_name(node.name)
                ctx.make_const(shape_name, np.array(new_broadcast_shape, dtype=np.int64))
                op_name = node.input[1]
                reshape_node = ctx.make_node("Reshape", [op_name, shape_name])
                ctx.replace_input(node, op_name, reshape_node.output[0], 1)
                ctx.set_shape(reshape_node.output[0], new_broadcast_shape)


@tf_op(["Pad", "PadV2", "MirrorPad"], onnx_op="Pad")
class Pad:

    @classmethod
    def convert_symmetric_pads(cls, ctx, node):
        """Currently there isn't a symmetric padding mode in ONNX so we add a dummy row then use the reflect mode
        and remove the dummy row with compress. Ex: 1234 -> 012340 -> 2101234043 -> 21123443. Only do this to
        dims with non-zero pads (if pads are constant)"""
        rank = ctx.get_rank(node.input[0])
        utils.make_sure(rank is not None, "Cannot convert pad with symmetric mode and unknown rank")
        utils.make_sure(ctx.opset >= 9, "opset 9 required for symmetric padding mode")
        node.set_attr("mode", "reflect")
        const_pads = None
        consumers = ctx.find_output_consumers(node.output[0])
        output_shape = ctx.get_shape(node.output[0])
        if ctx.opset < 11:
            const_pads = node.get_attr_value("pads")
        elif node.inputs[1].is_const():
            const_pads = node.inputs[1].get_tensor_value()
        non_zero_axes = list(range(rank))
        if const_pads is not None:
            non_zero_axes = []
            for i in range(rank):
                if const_pads[i] != 0 or const_pads[i + rank] != 0:
                    non_zero_axes.append(i)

        inc_pads = [0] * (rank * 2)
        for a in non_zero_axes:
            inc_pads[a] = 1
            inc_pads[a + rank] = 1

        if ctx.opset < 11:
            padded_inp = ctx.make_node("Pad", [node.input[0]], attr={'mode': 'constant', 'pads': inc_pads}).output[0]
        else:
            pad1_pads_const = ctx.make_const(utils.make_name("pad1_pads"), np.array(inc_pads, np.int64)).output[0]
            padded_inp = ctx.make_node("Pad", [node.input[0], pad1_pads_const], attr={'mode': 'constant'}).output[0]
        ctx.replace_input(node, node.input[0], padded_inp, 0)
        ctx.update_node_shape_dtype(node, override=True)

        output = node.output[0]
        shape = ctx.make_node("Shape", [output]).output[0]
        dims = ctx.make_node("Split", [shape], attr={'num_outputs': rank}, output_count=rank).output
        two_false = ctx.make_const(utils.make_name("two_false"), np.array([False, False], bool)).output[0]
        inv_second = ctx.make_const(utils.make_name("inv_second"), np.array([1, -1], np.int64)).output[0]
        dec_second = ctx.make_const(utils.make_name("dec_second"), np.array([0, 1], np.int64)).output[0]
        for a in non_zero_axes:
            one_tensor = helper.make_tensor("value", onnx_pb.TensorProto.BOOL, dims=[1], vals=[1])
            ones_of_shape = ctx.make_node("ConstantOfShape", [dims[a]], attr={'value': one_tensor}).output[0]
            if const_pads is not None:
                to_remove_val = [const_pads[a], -1 - const_pads[a + rank]]
                to_remove = ctx.make_const(utils.make_name("to_remove"), np.array(to_remove_val, np.int64)).output[0]
            else:
                pads_idx = ctx.make_const(utils.make_name("pads_idx"), np.array([a, a + rank], np.int64)).output[0]
                pads_vals = ctx.make_node("Gather", [node.input[1], pads_idx]).output[0]
                pads_inv_second = ctx.make_node("Mul", [pads_vals, inv_second]).output[0]
                to_remove = ctx.make_node("Sub", [pads_inv_second, dec_second]).output[0]
            scatter_op = "ScatterElements" if ctx.opset >= 11 else "Scatter"
            dims_to_keep = ctx.make_node(scatter_op, [ones_of_shape, to_remove, two_false]).output[0]
            compress = ctx.make_node("Compress", [output, dims_to_keep], attr={'axis': a})
            output = compress.output[0]
        ctx.replace_all_inputs(node.output[0], output, consumers)
        ctx.set_shape(output, output_shape)

    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.type = "Pad"
        # T output = Pad(T input, int32 paddings, @type Tpaddings), CONST model using default value
        #  or PadV2(T input, int32 paddings, T constant_value, @type Tpaddings), CONST mode - default value specified
        #  or MirrorPad(T input, int32 paddings, @type Tpaddings, @STRING mode), other mode.
        # T output = Pad(T data, @STRING mode, @INTS pads, @FLOAT value)
        paddings = np.array(node.inputs[1].get_tensor_value()).transpose().flatten()
        mode = node.get_attr("mode")
        if mode:
            mode = mode.s.decode("utf-8").lower()
            node.set_attr("mode", mode)
        if mode not in [None, "symmetric", "constant", "reflect"]:
            raise ValueError(mode + " pad mode is not supported")

        if mode in [None, "constant"] and len(node.input) == 3:
            const_val = node.inputs[2].get_tensor_value()
            node.set_attr("value", const_val)
            ctx.remove_input(node, node.input[2], 2)

        ctx.remove_input(node, node.input[1], 1)
        node.set_attr("pads", paddings)

        origin_dtype = ctx.get_dtype(node.output[0])
        if origin_dtype not in [onnx_pb.TensorProto.FLOAT16, onnx_pb.TensorProto.FLOAT,
                                onnx_pb.TensorProto.DOUBLE]:
            cast_node = ctx.insert_new_node_on_input(node, "Cast", node.input[0], to=onnx_pb.TensorProto.FLOAT)
            ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.FLOAT)
            ctx.copy_shape(node.name, cast_node.output[0])

            cast_back_node = ctx.insert_new_node_on_output("Cast", node.output[0],
                                                           name=utils.make_name(node.name) + "_castback",
                                                           to=origin_dtype)
            ctx.set_dtype(cast_back_node.output[0], origin_dtype)
            ctx.copy_shape(node.name, cast_back_node.output[0])

        if mode == "symmetric":
            cls.convert_symmetric_pads(ctx, node)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        mode = node.get_attr("mode")
        if mode:
            mode = mode.s.decode("utf-8").lower()
            node.set_attr("mode", mode)
        if mode not in [None, "symmetric", "constant", "reflect"]:
            raise ValueError(mode + " pad mode is not supported")

        if not node.inputs[1].is_const():
            # pads must be int64.
            if ctx.get_dtype(node.input[1]) != onnx_pb.TensorProto.INT64:
                ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=onnx_pb.TensorProto.INT64)
            ctx.insert_new_node_on_input(node, "Transpose", node.input[1])
            shape_const = ctx.make_const(utils.make_name(node.name), np.array([-1]).astype(np.int64))
            ctx.insert_new_node_on_input(node, "Reshape", [node.input[1], shape_const.name])
        else:
            paddings = node.inputs[1].get_tensor_value(as_list=False).astype(np.int64).transpose().flatten()
            pad_const = ctx.make_const(utils.make_name("pad_const"), paddings)
            ctx.replace_input(node, node.input[1], pad_const.output[0], 1)

        origin_dtype = ctx.get_dtype(node.output[0])
        if origin_dtype not in [TensorProto.FLOAT, TensorProto.DOUBLE,
                                TensorProto.INT32, TensorProto.INT64]:
            cast_node = ctx.insert_new_node_on_input(node, "Cast", node.input[0], to=TensorProto.FLOAT)
            ctx.set_dtype(cast_node.output[0], TensorProto.FLOAT)
            ctx.copy_shape(node.name, cast_node.output[0])

            cast_back_node = ctx.insert_new_node_on_output("Cast", node.output[0],
                                                           name=utils.make_name(node.name) + "_castback",
                                                           to=origin_dtype)
            ctx.set_dtype(cast_back_node.output[0], origin_dtype)
            ctx.copy_shape(node.name, cast_back_node.output[0])

        if mode == "symmetric":
            cls.convert_symmetric_pads(ctx, node)


@tf_op(["FusedBatchNorm", "FusedBatchNormV2", "FusedBatchNormV3"])
class BatchNorm:
    @classmethod
    def version_6(cls, ctx, node, **kwargs):
        tf_type = node.type
        input_rank = len(ctx.get_shape(node.input[0]))
        if input_rank == 4:
            spatial = 2
        elif input_rank == 5:
            spatial = 3
        else:
            raise ValueError("node input must be 4 or 5-dimensional, is {} now".format(input_rank))

        node.type = "BatchNormalization"
        # tf inputs: x, scale, bias, mean, variance
        # tf outputs: y, batch_mean, batch_var
        # a: data_format, epsilon, is_training
        # onnx inputs: X, scale, B, mean, variance, attributes: epsilon, momentum=0.9, spatial : 1
        # output: y, mean, var, savedmean, savedvar,
        # detach unused outputs. While we could let the unused outputs dangle,
        # some runtimes like pytorch/caffe2 do complain about it.

        # onnx batchnorm requires same T for all inputs
        mean_type = ctx.get_dtype(node.input[3])
        x_dtype = ctx.get_dtype(node.input[0])
        if x_dtype != mean_type:
            # TODO: this works but more efficient would be to flip the other inputs. We'd need to check
            # TODO: first if this works with the onnx implementation so its a later for now
            ctx.insert_new_node_on_input(node, "Cast", node.input[0], to=mean_type)
            # casting the input[0] will change the output dtype of bn so we need to cast back
            cast_back_node = ctx.insert_new_node_on_output("Cast", node.output[0],
                                                           name=utils.make_name(node.name) + "_castback",
                                                           to=x_dtype)
            ctx.set_dtype(cast_back_node.output[0], x_dtype)
            ctx.copy_shape(node.name, cast_back_node.output[0])
            ctx.set_dtype(node.output[0], mean_type)

        consumers = [ctx.find_output_consumers(output_name) for output_name in node.output[1:]]
        if not any(consumers):
            new_output = [node.output[0]]
            # the setter makes a copy of new_output
            node.output = new_output

        conv_convert_inputs(ctx, node, with_kernel=False, spatial=spatial)

        inp_shape = ctx.get_shape(node.input[0])
        inp_rank = len(inp_shape) if inp_shape is not None else None
        scale_shape = ctx.get_shape(node.input[1])
        mean_shape = ctx.get_shape(node.input[3])
        var_shape = ctx.get_shape(node.input[4])
        val_type = utils.map_onnx_to_numpy_type(ctx.get_dtype(node.input[1]))
        is_training = node.get_attr_value('is_training', True)

        if is_training and node.get_attr_value('exponential_avg_factor', 1.0) == 1.0:
            # Sometimes TF uses a BatchNorm op with training = True and exponential_avg_factor = 1.0
            # to perform layer mean/variance normalization. In such cases, the mean/var are computed from the input.
            # TF allows mean/variance to be excluded only if is_training and exponential_avg_factor == 1.0
            utils.make_sure(inp_rank is not None, "Cannot convert node %s of type %s with input of unknown rank.",
                            node.name, tf_type)
            dims = [0] + list(range(2, inp_rank))
            avg = GraphBuilder(ctx).make_reduce_mean({"data": node.input[0], "axes": dims, "keepdims": True})
            avg_squeezed = GraphBuilder(ctx).make_squeeze({"data": avg, "axes": dims})
            sub = ctx.make_node("Sub", [node.input[0], avg]).output[0]
            var_squeezed = GraphBuilder(ctx).make_reduce_sum_square({"data": sub, "axes": dims, "keepdims": False})

            inp_shape = ctx.make_node("Shape", [node.input[0]]).output[0]
            dims_const = ctx.make_const(utils.make_name("axes_const"), np.array(dims, dtype=np.int64)).output[0]
            reduce_dims = ctx.make_node("Gather", [inp_shape, dims_const]).output[0]
            dims_product = GraphBuilder(ctx).make_reduce_prod({"data": reduce_dims, "axes": [0], "keepdims": False})
            cnt_float = ctx.make_node("Cast", [dims_product], attr={'to': ctx.get_dtype(node.input[0])})

            pop_var_squeezed = ctx.make_node("Div", [var_squeezed, cnt_float.output[0]]).output[0]
            ctx.replace_inputs(node, node.input[:3] + [avg_squeezed, pop_var_squeezed])
        elif is_training:
            logger.warning("Node %s of type %s has is_training set to true, which is not supperted. "
                           "Please re-save the model with training set to false.",
                           node.name, tf_type)
            # As long as the mean/variance estimates are provided, we should be OK
            is_training = False

        if not is_training and mean_shape != scale_shape and all(d >= 0 for d in scale_shape):
            new_mean_value = np.array(np.resize(node.inputs[3].get_tensor_value(as_list=False), scale_shape),
                                      dtype=val_type)
            new_mean_node_name = utils.make_name(node.name)
            ctx.make_const(new_mean_node_name, new_mean_value)
            ctx.replace_input(node, node.input[3], new_mean_node_name, 3)

        if not is_training and var_shape != scale_shape and all(d >= 0 for d in scale_shape):
            new_var_value = np.array(np.resize(node.inputs[4].get_tensor_value(as_list=False), scale_shape),
                                     dtype=val_type)
            new_val_node_name = utils.make_name(node.name)
            ctx.make_const(new_val_node_name, new_var_value)
            ctx.replace_input(node, node.input[4], new_val_node_name, 4)

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        # is_test was removed - no change for us
        cls.version_6(ctx, node, **kwargs)


@tf_op(["SpaceToDepth"])
class SpaceToDepth:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        block_size = node.get_attr("block_size")
        node.set_attr("blocksize", block_size.i)
        conv_convert_inputs(ctx, node, with_kernel=False)


@tf_op(["DepthToSpace"])
class DepthToSpace:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        block_size = node.get_attr("block_size")
        node.set_attr("blocksize", block_size.i)
        conv_convert_inputs(ctx, node, with_kernel=False)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Onnx-11 CRD mode added. No change for tf2onnx
        cls.version_1(ctx, node, **kwargs)


@tf_op(["SampleDistortedBoundingBox", "SampleDistortedBoundingBoxV2"])
class SampleDistortedBoundingBox:
    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        # See tensorflow sample_distorted_bounding_box_op.cc
        image_size, bounding_boxes, min_object_covered = node.input

        seed = node.get_attr_value("seed", 0)
        seed2 = node.get_attr_value("seed2", 0)
        onnx_seed = utils.combine_seeds(seed, seed2)
        rand_attr = {}
        if onnx_seed is not None:
            rand_attr['seed'] = onnx_seed

        min_aspect_ratio, max_aspect_ratio = node.get_attr_value("aspect_ratio_range", [0.75, 1.33])
        ratio_range = max_aspect_ratio - min_aspect_ratio
        min_area, max_area = node.get_attr_value("area_range", [0.05, 1.0])
        max_attempts = node.get_attr_value("max_attempts", 100)
        use_image_if_no_bounding_boxes = node.get_attr_value("use_image_if_no_bounding_boxes", 0)

        min_area_node = ctx.make_const(utils.make_name("min_area"), np.array(min_area, np.float32)).output[0]
        max_area_node = ctx.make_const(utils.make_name("max_area"), np.array(max_area, np.float32)).output[0]
        min_ratio_node = ctx.make_const(utils.make_name("min_ratio"), np.array(min_aspect_ratio, np.float32)).output[0]
        ratio_range_node = ctx.make_const(utils.make_name("max_ratio"), np.array(ratio_range, np.float32)).output[0]

        boxes_tensor_shape = ctx.get_shape(bounding_boxes)
        if boxes_tensor_shape is not None and 0 in boxes_tensor_shape and use_image_if_no_bounding_boxes:
            no_boxes = True
            min_area_node = ctx.make_node("Max", [min_object_covered, min_area_const]).output[0]
        else:
            no_boxes = False

        rand_attr['shape'] = [max_attempts, 4]
        random_nums = ctx.make_node("RandomUniform", [], attr=rand_attr, op_name_scope=node.name).output[0]
        r1, r2, r3, r4 = ctx.make_node("Split", [random_nums], attr={'axis': 1, 'num_outputs': 4},
                                       output_count=4).output

        # Use r1 to sample the aspect ratio
        scaled_r1 = ctx.make_node("Mul", [r1, ratio_range_node]).output[0]
        aspect_ratio = ctx.make_node("Add", [scaled_r1, min_ratio_node]).output[0]

        image_size_float = ctx.make_node("Cast", [image_size], attr={'to': TensorProto.FLOAT}).output[0]
        img_height = GraphBuilder(ctx).make_slice({"data": image_size_float, "starts": [0], "ends": [1], "axes": [0]})
        img_width = GraphBuilder(ctx).make_slice({"data": image_size_float, "starts": [1], "ends": [2], "axes": [0]})
        img_aspect_ratio = ctx.make_node("Div", [img_width, img_height]).output[0]
        adjusted_aspect_ratio = ctx.make_node("Div", [aspect_ratio, img_aspect_ratio]).output[0]

        # Use r2 to sample height
        const_one = ctx.make_const(utils.make_name("const_one"), np.array(1, np.float32)).output[0]
        min_height_squared = ctx.make_node("Div", [min_area_node, adjusted_aspect_ratio]).output[0]
        max_height_squared = ctx.make_node("Div", [max_area_node, adjusted_aspect_ratio]).output[0]
        min_height = ctx.make_node("Sqrt", [min_height_squared]).output[0]
        max_height = ctx.make_node("Sqrt", [max_height_squared]).output[0]
        max_allowed_height = ctx.make_node("Div", [img_aspect_ratio, aspect_ratio]).output[0]
        max_allowed_height2 = ctx.make_node("Min", [max_allowed_height, const_one]).output[0]
        max_height2 = ctx.make_node("Min", [max_height, max_allowed_height2]).output[0]
        min_height2 = ctx.make_node("Min", [min_height, max_height2]).output[0]
        height_range = ctx.make_node("Sub", [max_height2, min_height2]).output[0]
        scaled_r2 = ctx.make_node("Mul", [r2, height_range]).output[0]
        box_height = ctx.make_node("Add", [scaled_r2, min_height2]).output[0]
        box_width = ctx.make_node("Mul", [box_height, adjusted_aspect_ratio]).output[0]

        # Use r3 and r4 to get x and y pos
        max_shift_x = ctx.make_node("Sub", [const_one, box_width]).output[0]
        max_shift_y = ctx.make_node("Sub", [const_one, box_height]).output[0]
        x1 = ctx.make_node("Mul", [r3, max_shift_x]).output[0]
        y1 = ctx.make_node("Mul", [r4, max_shift_y]).output[0]

        x2 = ctx.make_node("Add", [x1, box_width]).output[0]
        y2 = ctx.make_node("Add", [y1, box_height]).output[0]

        all_boxes = ctx.make_node("Concat", [y1, x1, y2, x2], attr={'axis': 1}).output[0]

        area = ctx.make_node("Mul", [box_height, box_width]).output[0]
        area_too_large = ctx.make_node("Greater", [area, max_area_node]).output[0]
        area_too_small = ctx.make_node("Less", [area, min_area_node]).output[0]
        area_out_of_bounds = ctx.make_node("Or", [area_too_large, area_too_small]).output[0]
        area_in_bounds = ctx.make_node("Not", [area_out_of_bounds]).output[0]
        acceptable = area_in_bounds

        if not no_boxes:
            boxes_shape = ctx.make_const(utils.make_name("reshape_const"), np.array([-1, 4, 1], np.int64)).output[0]
            bounding_boxes_flat = ctx.make_node("Reshape", [bounding_boxes, boxes_shape]).output[0]

            box_y1, box_x1, box_y2, box_x2 = \
                ctx.make_node("Split", [bounding_boxes_flat], attr={'axis': 1, 'num_outputs': 4}, output_count=4).output

            combined_max_y = ctx.make_node("Min", [y2, box_y2]).output[0]
            combined_max_x = ctx.make_node("Min", [x2, box_x2]).output[0]
            combined_min_y = ctx.make_node("Max", [y1, box_y1]).output[0]
            combined_min_x = ctx.make_node("Max", [x1, box_x1]).output[0]

            box_height = ctx.make_node("Sub", [box_y2, box_y1]).output[0]
            box_width = ctx.make_node("Sub", [box_x2, box_x1]).output[0]
            box_area = ctx.make_node("Mul", [box_height, box_width]).output[0]

            const_zero = ctx.make_const(utils.make_name("const_zero"), np.array(0, np.float32)).output[0]
            overlap_height = ctx.make_node("Sub", [combined_max_y, combined_min_y]).output[0]
            non_neg_height = ctx.make_node("Max", [overlap_height, const_zero]).output[0]
            overlap_width = ctx.make_node("Sub", [combined_max_x, combined_min_x]).output[0]
            non_neg_width = ctx.make_node("Max", [overlap_width, const_zero]).output[0]

            overlap_area = ctx.make_node("Mul", [non_neg_height, non_neg_width]).output[0]
            overlap_ratio = ctx.make_node("Div", [overlap_area, box_area]).output[0]
            overlap_bad = ctx.make_node("Less", [overlap_ratio, min_object_covered]).output[0]
            overlap_ok = ctx.make_node("Not", [overlap_bad]).output[0]
            overlap_ok_fp = ctx.make_node("Cast", [overlap_ok], attr={'to': TensorProto.FLOAT}).output[0]
            num_ok = GraphBuilder(ctx).make_reduce_sum(
                {"data": overlap_ok_fp, "axes": [0], "keepdims": 0, "noop_with_empty_axes": 1})
            any_ok = ctx.make_node("Greater", [num_ok, const_zero]).output[0]
            acceptable = ctx.make_node("And", [acceptable, any_ok]).output[0]

        acceptable_sq = GraphBuilder(ctx).make_squeeze({'data': acceptable, 'axes': [1]})
        boxes_shape = ctx.get_shape(all_boxes)
        filtered_shape = [-1] + boxes_shape[1:] if boxes_shape is not None else None
        filtered = ctx.make_node("Compress", [all_boxes, acceptable_sq], attr={'axis': 0},
                                 dtypes=[ctx.get_dtype(all_boxes)], shapes=[filtered_shape]).output[0]
        default_box = np.array([0.0, 0.0, 1.0, 1.0], np.float32).reshape([1, 4])
        const_default_box = ctx.make_const(utils.make_name("default_box"), default_box).output[0]
        filtered_non_empty = ctx.make_node("Concat", [filtered, const_default_box], attr={'axis': 0}).output[0]

        first_valid_box = GraphBuilder(ctx).make_slice(
            {"data": filtered_non_empty, "starts": [0], "ends": [1], "axes": [0]})
        first_valid_box_sq = GraphBuilder(ctx).make_squeeze({'data': first_valid_box, 'axes': [0]})

        int_dtype = ctx.get_dtype(image_size)
        np_int_dtype = utils.map_onnx_to_numpy_type(int_dtype)
        scale = ctx.make_node("Concat", [img_height, img_width, img_height, img_width], attr={'axis': 0}).output[0]
        box_scaled = ctx.make_node("Mul", [first_valid_box_sq, scale]).output[0]
        if ctx.opset >= 11:
            box_rounded = ctx.make_node("Round", [box_scaled]).output[0]
        else:
            box_rounded = box_scaled   # Close enough
        box_cast = ctx.make_node("Cast", [box_rounded], attr={'to': int_dtype}).output[0]
        bb_begin, bb_end = ctx.make_node("Split", [box_cast], attr={'axis': 0, 'num_outputs': 2}, output_count=2).output
        bb_size = ctx.make_node("Sub", [bb_end, bb_begin]).output[0]

        const_zero_int = ctx.make_const(utils.make_name("const_zero"), np.array([0], np_int_dtype)).output[0]
        const_neg_one_int = ctx.make_const(utils.make_name("const_neg_one"), np.array([-1], np_int_dtype)).output[0]

        begin = ctx.make_node("Concat", [bb_begin, const_zero_int], attr={'axis': 0},
                              op_name_scope=node.name).output[0]
        size = ctx.make_node("Concat", [bb_size, const_neg_one_int], attr={'axis': 0},
                             op_name_scope=node.name).output[0]
        bboxes = GraphBuilder(ctx).make_unsqueeze({'data': first_valid_box, 'axes': [0]})

        ctx.replace_all_inputs(node.output[0], begin)
        ctx.replace_all_inputs(node.output[1], size)
        ctx.replace_all_inputs(node.output[2], bboxes)
        ctx.remove_node(node.name)


@tf_op(["CropAndResize"])
class CropAndResize:
    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        utils.make_sure(node.inputs[1].type == "Const", "boxes input must be a Const")
        utils.make_sure(node.inputs[3].type == "Const", "boxes input must be a Const")
        name = node.name
        output_height = node.inputs[3].get_tensor_value()[0]
        output_width = node.inputs[3].get_tensor_value()[1]
        rois = node.inputs[1].get_tensor_value()
        rois_shape = ctx.get_shape(node.input[1])
        img_shape = ctx.get_shape(node.input[0])
        transform_rois = np.zeros(list(rois_shape), dtype=np.float32)
        for i in range(rois_shape[0]):
            y1, x1, y2, x2 = rois[i]
            y1 = y1 * (img_shape[1] - 1)
            y2 = y2 * (img_shape[1] - 1)
            x1 = x1 * (img_shape[2] - 1)
            x2 = x2 * (img_shape[2] - 1)
            spacing_h = (y2 - y1)
            spacing_w = (x2 - x1)
            b1 = y1 - 0.5 * spacing_h / (output_height - 1)
            a1 = x1 - 0.5 * spacing_w / (output_width - 1)
            b2 = y2 + 0.5 * spacing_h / (output_height - 1)
            a2 = x2 + 0.5 * spacing_w / (output_width - 1)
            transform_rois[i][0] = a1
            transform_rois[i][1] = b1
            transform_rois[i][2] = a2
            transform_rois[i][3] = b2
        cast_node = ctx.make_node("Cast", [node.input[2]], attr={"to": onnx_pb.TensorProto.INT64})
        bbox_node = ctx.make_const(utils.make_name("bbox"), transform_rois)
        dtypes = [ctx.get_dtype(node.output[0])]
        shapes = [ctx.get_shape(node.output[0])]
        input_nchw = ctx.make_node("Transpose", [node.input[0]], {"perm": [0, 3, 1, 2]},
                                   name=utils.make_name(node.name))
        crop_and_resize = ctx.make_node("RoiAlign", inputs=[input_nchw.output[0], bbox_node.output[0],
                                                            cast_node.output[0]],
                                        attr={"output_height": output_height, "output_width": output_width,
                                              "spatial_scale": 1.0, "sampling_ratio": 1},
                                        name=utils.make_name(node.name))
        ctx.remove_node(name)
        ctx.make_node("Transpose", crop_and_resize.output, {"perm": [0, 2, 3, 1]},
                      name=name, outputs=node.output, shapes=shapes, dtypes=dtypes)

    @classmethod
    def any_version_after11(cls, opset, ctx, node, **kwargs):
        # create loop of resize to cater to tensorflow CropAndResize, one box one iteration
        mode = "nearest" if node.get_attr("method") is not None and node.get_attr(
            "method").s == b"nearest" else "linear"
        extrapolation_value = float(node.get_attr("extrapolation_value", "0").f)
        input_x = node.input[0]
        x_shape = ctx.make_node("Shape", [input_x]).output[0]
        num_channels = GraphBuilder(ctx).make_slice({"data": x_shape, "starts": [3], "ends": [4], "axes": [0]})
        boxes = node.input[1]
        box_ind = node.input[2]
        crop_size = node.input[3]
        if ctx.get_dtype(crop_size) != TensorProto.INT64:
            crop_size = ctx.make_node("Cast", [crop_size], attr={'to': TensorProto.INT64}).output[0]
        trip_name = utils.make_name(node.name + "_i")
        cond_name = utils.make_name(node.name + "_cond")
        cond_out_name = utils.make_name(node.name + "cond_out")
        g = ctx.create_new_graph_with_same_config()
        g.add_graph_input(trip_name, TensorProto.INT64, [1])
        g.add_graph_input(cond_name, TensorProto.BOOL, [])
        g.parent_graph = ctx
        const_zero = g.make_const(utils.make_name(node.name + "_const_zero"), np.array([0], dtype=np.int32))
        const_zero_long = g.make_const(utils.make_name(node.name + "_const_zero_long"), np.array([0], dtype=np.int64))
        const_one = g.make_const(utils.make_name(node.name + "_const_one"), np.array([1], dtype=np.int32))
        const_one_long = g.make_const(utils.make_name(node.name + "_const_one_long"), np.array([1], dtype=np.int64))
        index_end = g.make_node("Add", [trip_name, const_one_long.output[0]])
        box_index_from = g.make_node("Slice", [box_ind, trip_name, index_end.output[0]], name="Slice_a")
        box_index_to = g.make_node("Add", [box_index_from.output[0], const_one.output[0]])
        target_x = g.make_node("Slice", [input_x, box_index_from.output[0], box_index_to.output[0],
                                         const_zero.output[0]], name="Slice_b")
        transposed_x = g.make_node("Transpose", [target_x.output[0]], attr={'perm': constants.NHWC_TO_NCHW})
        const_zero_zero = g.make_const(utils.make_name(node.name + "_const_zero_zero"),
                                       np.array([0, 0], dtype=np.float32))
        const_one_one = g.make_const(utils.make_name(node.name + "_const_one_one"),
                                     np.array([1, 1], dtype=np.float32))
        const_four = g.make_const(utils.make_name(node.name + "_const_four"), np.array([4], dtype=np.int64))
        const_empty_float = g.make_const(utils.make_name("const_empty_float"), np.array([], dtype=np.float32))
        box = g.make_node("Slice", [boxes, trip_name, index_end.output[0], const_zero_long.output[0]],
                          name="Slice_c")
        roi_raw = g.make_node("Reshape", [box.output[0], const_four.output[0]])
        roi_raw_first_half = GraphBuilder(g).make_slice({"data": roi_raw.output[0], "ends": [2], "starts": [0]})
        roi_raw_second_half = GraphBuilder(g).make_slice({"data": roi_raw.output[0], "ends": [4], "starts": [2]})
        roi_concat_1 = g.make_node("Concat", [const_zero_zero.output[0], roi_raw_first_half], attr={'axis': 0})
        roi_concat_2 = g.make_node("Concat", [const_one_one.output[0], roi_raw_second_half], attr={'axis': 0})
        final_roi = g.make_node("Concat", [roi_concat_1.output[0], roi_concat_2.output[0]], attr={'axis': 0})
        final_crop_size = build_dynamic_target_size(g, transposed_x.output[0], crop_size)
        resized_x = g.make_node("Resize", [transposed_x.output[0], final_roi.output[0], const_empty_float.output[0],
                                           final_crop_size.output[0]],
                                attr={"mode": mode, "extrapolation_value": extrapolation_value,
                                      "coordinate_transformation_mode": "tf_crop_and_resize"})
        recovered_x = g.make_node("Transpose", [resized_x.output[0]], attr={'perm': constants.NCHW_TO_NHWC})
        squeeze_x = GraphBuilder(g).make_squeeze({'data': recovered_x.output[0], 'axes': [0]}, return_node=True)
        g.make_node("Identity", [cond_name], outputs=[cond_out_name])
        g.add_graph_output(cond_out_name, TensorProto.BOOL, [])
        g.add_graph_output(squeeze_x.output[0], ctx.get_dtype(node.input[0]), [-1, -1, -1])
        trip_node = ctx.make_node("Size", [box_ind])
        cond_const = ctx.make_const(utils.make_name("cond"), np.ones((), dtype=bool))
        ctx.remove_node(node.name)
        branches = {"body": g}
        inner_loop = ctx.make_node("Loop", [trip_node.output[0], cond_const.output[0]], name=node.name,
                                   outputs=node.output, branches=branches)
        const_neg_one = ctx.make_const(utils.make_name("const_neg_one"), np.array([-1], np.int64)).output[0]
        final_shape = ctx.make_node("Concat", [const_neg_one, crop_size, num_channels], attr={'axis': 0}).output[0]
        # This reshape fixes the case when there are no iterations and the scan output is empty.
        ctx.insert_new_node_on_output("Reshape", inner_loop.output[0], inputs=[inner_loop.output[0], final_shape])

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        cls.any_version_after11(11, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Signature of operator Squeeze changed.
        cls.any_version_after11(13, ctx, node, **kwargs)


@tf_op(["ResizeBilinear", "ResizeNearestNeighbor", "ResizeBicubic", "ResizeArea"])
class Resize:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        utils.make_sure(node.type != "ResizeBicubic", "Opset 11 is required for bicubic interpolation for node %s",
                        node.name)
        mode = "linear" if node.type == "ResizeBilinear" else "nearest"
        node.type = "Upsample"
        shape = ctx.get_shape(node.input[0])
        target_shape = node.inputs[1].get_tensor_value()
        # https://www.tensorflow.org/api_docs/python/tf/image/resize_nearest_neighbor
        # wants the input to be NHWC - adjust target_shape to this.
        n, h, w, c = shape
        nh, nw = target_shape
        utils.make_sure(all(i != -1 for i in [nh, nw]), "h and w need to be known")
        # scaler is nchw
        scaler = [1., 1., float(nh) / h, float(nw) / w]
        node.set_attr("scales", scaler)
        node.set_attr("mode", mode)
        ctx.remove_input(node, node.input[1], 1)
        node.data_format = "NHWC"
        conv_convert_inputs(ctx, node, with_kernel=False)

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        cls._convert_since_9(ctx, node, op_type="Upsample")

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        cls._convert_since_9(ctx, node, op_type="Resize")

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        cubic_coeff_a = None
        exclude_outside = False
        if node.type == "ResizeBilinear":
            mode = "linear"
        elif node.type == "ResizeBicubic":
            mode = "cubic"
            cubic_coeff_a = -0.5
            exclude_outside = True
        else:
            mode = "nearest"
        roi = ctx.make_const(utils.make_name("roi"), np.array([]).astype(np.float32))
        input_nchw = ctx.make_node("Transpose", [node.input[0]], {"perm": constants.NHWC_TO_NCHW})
        shape = ctx.get_shape(node.input[0])
        if shape and shape[2] != -1 and shape[1] != -1 and node.inputs[1].is_const():
            target_shape = node.inputs[1].get_tensor_value()
            n, h, w, c = shape
            nh, nw = target_shape
            if "sizes" in node.attr:
                sizes_val = np.array([1.0, 1.0, nh, nw]).astype(np.int64)
                resize_params = ctx.make_const(utils.make_name("sizes"), sizes_val, raw=False)
            else:  # scales
                scale_val = np.array([1.0, 1.0, float(nh) / h, float(nw) / w]).astype(np.float32)
                resize_params = ctx.make_const(utils.make_name("scales"), scale_val, raw=False)
            resize_inputs = [
                input_nchw.output[0],
                roi.output[0],
                resize_params.output[0]
            ]
        else:
            const_zero = ctx.make_const(utils.make_name("const_zero"), np.array([0]).astype(np.int64))
            const_two = ctx.make_const(utils.make_name("const_two"), np.array([2]).astype(np.int64))
            const_empty_float = ctx.make_const(utils.make_name("const_empty_float"), np.array([]).astype(np.float32))
            shape_input = ctx.make_node("Shape", [input_nchw.output[0]])
            sliced_shape = ctx.make_node("Slice", [shape_input.output[0], const_zero.output[0], const_two.output[0]])
            size_int64 = ctx.make_node("Cast", [node.input[1]], attr={"to": onnx_pb.TensorProto.INT64})
            concat_shape = ctx.make_node("Concat", [sliced_shape.output[0], size_int64.output[0]], {'axis': 0})
            resize_inputs = [
                input_nchw.output[0],
                roi.output[0],
                const_empty_float.output[0],
                concat_shape.output[0]
            ]
        transformation_mode = "asymmetric"
        nearest_mode = "floor"
        if "align_corners" in node.attr and node.attr["align_corners"].i:
            transformation_mode = "align_corners"
            nearest_mode = "round_prefer_ceil"
        if "half_pixel_centers" in node.attr and node.attr["half_pixel_centers"].i:
            if node.type == "ResizeNearestNeighbor" and not ctx.is_target(constants.TARGET_TENSORRT):
                # TensorRT only supports nearest_mode = "floor" for mode = "nearest"
                transformation_mode = "tf_half_pixel_for_nn"
            else:
                transformation_mode = "half_pixel"
        attr = {"mode": mode, "nearest_mode": nearest_mode, "coordinate_transformation_mode": transformation_mode,
                "exclude_outside": exclude_outside}
        if cubic_coeff_a is not None:
            attr["cubic_coeff_a"] = cubic_coeff_a
        resize = ctx.make_node("Resize", resize_inputs, attr=attr)
        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        ctx.make_node("Transpose", resize.output, {"perm": constants.NCHW_TO_NHWC},
                      name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)

    @classmethod
    def _convert_since_9(cls, ctx, node, op_type, use_target_size=False):

        # float32 out = ResizeBilinear/ResizeNearestNeighbor(T images, int size)
        # https://www.tensorflow.org/api_docs/python/tf/image/resize_nearest_neighbor
        # wants the input to be NHWC - adjust target_shape to this.
        utils.make_sure(node.type != "ResizeBicubic", "Opset 11 is required for bicubic interpolation for node %s",
                        node.name)
        if "align_corners" in node.attr:
            utils.make_sure(not node.attr["align_corners"].i,
                            "Opset 11 is required for align_corners=True for node %s", node.name)
        if "half_pixel_centers" in node.attr:
            utils.make_sure(not node.attr["half_pixel_centers"].i,
                            "Opset 11 is required for half_pixel_centers=True for node %s", node.name)
        mode = "linear" if node.type == "ResizeBilinear" else "nearest"

        # because onnxruntime only supports to scale the last two dims so transpose is inserted
        input_nchw = ctx.make_node("Transpose", [node.input[0]], {"perm": constants.NHWC_TO_NCHW})
        if use_target_size:
            final_target_size = build_dynamic_target_size(ctx, input_nchw.output[0], node.input[1])
            roi = ctx.make_const(utils.make_name("roi"), np.array([]).astype(np.float32))
            const_empty_float = ctx.make_const(utils.make_name("const_empty_float"), np.array([], dtype=np.float32))
            resize_inputs = [
                input_nchw.output[0],
                roi.output[0],
                const_empty_float.output[0],
                final_target_size.output[0]
            ]
            upsample = ctx.make_node("Resize", resize_inputs,
                                     attr={"mode": mode, "nearest_mode": "floor",
                                           "coordinate_transformation_mode": "asymmetric"})
        else:
            # first create "scales" info for onnx upsample
            # if shape of input and output known then  "scale" is calculated statically and set as a const node
            shape = ctx.get_shape(node.input[0])
            if shape and shape[2] != -1 and shape[1] != -1 and node.inputs[1].is_const():
                target_shape = node.inputs[1].get_tensor_value()
                n, h, w, c = shape
                nh, nw = target_shape
                # scales is nchw
                # the reason not storing data at raw field is because of the bug:
                # https://github.com/onnx/onnx/issues/1852
                scale_val = np.array([1.0, 1.0, float(nh) / h, float(nw) / w]).astype(np.float32)
                scales = ctx.make_const(utils.make_name("scales"), scale_val, raw=False)
            else:
                ori_shape = ctx.make_node("Shape", [node.input[0]])
                attr = {"axes": [0], "starts": [1], "ends": [3]}
                inputs_map = {"data": ori_shape.output[0], **attr}
                ori_shape_hw = GraphBuilder(ctx).make_slice(inputs_map)
                ori_shape_hw_float = ctx.make_node("Cast", [ori_shape_hw], attr={"to": onnx_pb.TensorProto.FLOAT})

                target_hw = node.inputs[1]
                target_hw_float = ctx.make_node("Cast", target_hw.output, attr={"to": onnx_pb.TensorProto.FLOAT})

                scales_hw = ctx.make_node("Div", [target_hw_float.output[0], ori_shape_hw_float.output[0]])

                const_one_array = ctx.make_const(utils.make_name("one"), np.array([1.0, 1.0]).astype(np.float32))
                # scales is nchw
                scales = ctx.make_node("Concat", [const_one_array.output[0], scales_hw.output[0]], {"axis": 0})
            upsample = ctx.make_node(op_type, [input_nchw.output[0], scales.output[0]], attr={"mode": mode})

        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        ctx.make_node("Transpose", upsample.output, {"perm": constants.NCHW_TO_NHWC},
                      name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)


@tf_op("AdjustContrastv2")
class AdjustContrastv2:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        images, contrast_factor = node.input
        dtype = ctx.get_dtype(images)
        if ctx.get_dtype(contrast_factor) != dtype:
            contrast_factor = ctx.make_node("Cast", [dtype], attr={'to': dtype}).output[0]
        rank = ctx.get_rank(images)
        utils.make_sure(rank is not None, "AdjustContrastv2 requires input of known rank")
        # Reduce height and width only
        axes_to_reduce = list(range(rank))[-3:-1]
        mean = GraphBuilder(ctx).make_reduce_mean({"data": images, "axes": axes_to_reduce, "keepdims": True},
                                                  op_name_scope=node.name)
        diff = ctx.make_node("Sub", [images, mean], op_name_scope=node.name).output[0]
        scaled = ctx.make_node("Mul", [diff, contrast_factor], op_name_scope=node.name).output[0]
        result = ctx.make_node("Add", [scaled, mean], op_name_scope=node.name).output[0]
        ctx.replace_all_inputs(node.output[0], result)
        ctx.remove_node(node.name)


@tf_op("AdjustSaturation")
class AdjustSaturation:
    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        images, factor = node.input
        dtype = ctx.get_dtype(images)
        np_dtype = utils.map_onnx_to_numpy_type(dtype)
        k = ctx.make_const(utils.make_name("three"), np.array([3], np.int64)).output[0]
        ordered, indices = ctx.make_node("TopK", [images, k], attr={'axis': -1}, output_count=2).output
        # Sorted and separated into channels
        max_c, mid_c, min_c = ctx.make_node("Split", [ordered], attr={'axis': -1, 'num_outputs': 3},
                                            output_count=3).output
        delta = ctx.make_node("Sub", [max_c, min_c]).output[0]
        scaled_delta = ctx.make_node("Mul", [delta, factor], op_name_scope=node.name).output[0]
        new_delta = ctx.make_node("Min", [scaled_delta, max_c]).output[0]
        new_min = ctx.make_node("Sub", [max_c, new_delta]).output[0]
        delta2 = ctx.make_node("Sub", [mid_c, min_c]).output[0]
        const_zero = ctx.make_const(utils.make_name("zero"), np.array(0, np_dtype)).output[0]
        delta_z = ctx.make_node("Equal", [delta, const_zero]).output[0]
        delta_z_cast = ctx.make_node("Cast", [delta_z], attr={'to': dtype}).output[0]
        delta_nz = ctx.make_node("Add", [delta, delta_z_cast]).output[0]
        delta2_scale = ctx.make_node("Div", [new_delta, delta_nz]).output[0]
        new_delta2 = ctx.make_node("Mul", [delta2, delta2_scale], op_name_scope=node.name).output[0]
        new_mid = ctx.make_node("Add", [new_min, new_delta2]).output[0]
        new_ordered = ctx.make_node("Concat", [max_c, new_mid, new_min], attr={'axis': -1}).output[0]
        # Now put it back in order
        result = ctx.make_node("GatherElements", [new_ordered, indices], attr={'axis': -1}).output[0]
        ctx.replace_all_inputs(node.output[0], result)
        ctx.remove_node(node.name)


@tf_op("AdjustHue")
class AdjustHue:
    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        images, angle_delta = node.input
        dtype = ctx.get_dtype(images)
        np_dtype = utils.map_onnx_to_numpy_type(dtype)

        const_one = ctx.make_const(utils.make_name("const_one"), np.array(1.0, np_dtype)).output[0]
        const_six = ctx.make_const(utils.make_name("const_six"), np.array(6.0, np_dtype)).output[0]
        const_half = ctx.make_const(utils.make_name("const_half"), np.array(0.5, np_dtype)).output[0]
        const_zero = ctx.make_const(utils.make_name("zero"), np.array(0, np_dtype)).output[0]

        k = ctx.make_const(utils.make_name("three"), np.array([3], np.int64)).output[0]
        ordered, indices = ctx.make_node("TopK", [images, k], attr={'axis': -1},
                                         output_count=2, op_name_scope=node.name).output
        # Sorted and separated into channels
        max_c, mid_c, min_c = ctx.make_node("Split", [ordered], attr={'axis': -1, 'num_outputs': 3},
                                            output_count=3).output
        delta = ctx.make_node("Sub", [max_c, min_c]).output[0]
        delta2 = ctx.make_node("Sub", [mid_c, min_c]).output[0]
        delta_z = ctx.make_node("Equal", [delta, const_zero]).output[0]
        delta_z_cast = ctx.make_node("Cast", [delta_z], attr={'to': dtype}).output[0]
        delta_nz = ctx.make_node("Add", [delta, delta_z_cast]).output[0]
        progress_within_sector = ctx.make_node("Div", [delta2, delta_nz]).output[0]

        # Compute HSV angle
        sector_lookup = np.zeros((3, 3, 3, 1), np_dtype)
        parity_lookup = np.zeros((3, 3, 3, 1), np_dtype)

        sector_lookup[0, 1, 2, 0] = 0.0
        sector_lookup[1, 0, 2, 0] = 2 / 6
        sector_lookup[1, 2, 0, 0] = 2 / 6
        sector_lookup[2, 1, 0, 0] = 4 / 6
        sector_lookup[2, 0, 1, 0] = 4 / 6
        sector_lookup[0, 2, 1, 0] = 1.0

        parity_lookup[0, 1, 2, 0] = 1 / 6
        parity_lookup[1, 0, 2, 0] = -1 / 6
        parity_lookup[1, 2, 0, 0] = 1 / 6
        parity_lookup[2, 1, 0, 0] = -1 / 6
        parity_lookup[2, 0, 1, 0] = 1 / 6
        parity_lookup[0, 2, 1, 0] = -1 / 6

        sector_lookup_const = ctx.make_const(utils.make_name("hue_sector_lookup"), sector_lookup).output[0]
        parity_lookup_const = ctx.make_const(utils.make_name("hue_parity_lookup"), parity_lookup).output[0]

        sectors = ctx.make_node("GatherND", [sector_lookup_const, indices]).output[0]
        parities = ctx.make_node("GatherND", [parity_lookup_const, indices]).output[0]

        angle_offset = ctx.make_node("Mul", [progress_within_sector, parities]).output[0]
        initial_angle = ctx.make_node("Add", [angle_offset, sectors]).output[0]

        # Add angle delta
        angle_delta_pos = ctx.make_node("Add", [angle_delta, const_one]).output[0]
        new_angle = ctx.make_node("Add", [initial_angle, angle_delta_pos]).output[0]

        # Convert to RGB
        sector_to_rgb = ctx.make_const(utils.make_name("sector_to_rgb_const"), np.array([0, 4/6, 2/6], np_dtype))
        add_node = ctx.make_node("Add", [new_angle, sector_to_rgb.output[0]]).output[0]
        mod_node = ctx.make_node("Mod", [add_node, const_one], attr={'fmod': 1}).output[0]
        sub_node = ctx.make_node("Sub", [mod_node, const_half]).output[0]
        abs_node = ctx.make_node("Abs", [sub_node]).output[0]
        mul_node = ctx.make_node("Mul", [abs_node, const_six]).output[0]
        sub_node_2 = ctx.make_node("Sub", [mul_node, const_one]).output[0]
        clip_node = ctx.make_node("Clip", [sub_node_2, const_zero, const_one]).output[0]
        scaled_node = ctx.make_node("Mul", [clip_node, delta], op_name_scope=node.name).output[0]
        offset_node = ctx.make_node("Add", [scaled_node, min_c], op_name_scope=node.name).output[0]

        ctx.replace_all_inputs(node.output[0], offset_node)
        ctx.remove_node(node.name)


@tf_op("MatrixBandPart")
class MatrixBandPart:
    @classmethod
    def _apply_mask_and_transform(cls, ctx, node, mask):
        shapes = node.output_shapes
        dtypes = node.output_dtypes
        dtype = ctx.get_dtype(node.input[0])
        data = node.input[0]
        if dtype == TensorProto.BOOL:
            # bool is not supported for 'Mul', so convert mask and input supported dtype
            mask = ctx.make_node("Cast", inputs=mask.output, attr={'to': TensorProto.FLOAT}).output[0]
            data = ctx.make_node("Cast", [data], attr={'to': TensorProto.FLOAT}).output[0]
            result = ctx.make_node(op_type="Mul", inputs=[mask, data], shapes=shapes, dtypes=[TensorProto.FLOAT])
            ctx.remove_node(node.name)
            ctx.make_node("Cast", inputs=result.output, attr={'to': dtype},
                          name=node.name, outputs=node.output, dtypes=dtypes)
        else:
            mask = ctx.make_node(op_type="Cast", inputs=mask.output, attr={"to": dtype}).output[0]
            ctx.remove_node(node.name)
            ctx.make_node(op_type="Mul", inputs=[mask, data],
                          name=node.name, outputs=node.output, shapes=shapes,
                          dtypes=dtypes)

    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # T output = MatrixBandPart(T input, int num_lower, int num_upper)
        # data-flow: first generate mask matrix and then use element-wise mul op
        input_rank = len(ctx.get_shape(node.input[0]))
        utils.make_sure(input_rank == 2, error_msg="MatrixBandPart op: only rank 2 is supported")
        bandpart = [node.inputs[ind].get_tensor_value() for ind in [1, 2]]
        utils.make_sure(bandpart in [[-1, 0], [0, -1]], "only support Lower/Upper triangular for opset < 11")
        # methods to generate mask matrix: if lower triangular is needed, then generate column one by one
        # otherwise row is generated one by one.
        axis, counter_axis, squeeze_axis = (1, 0, 2) if bandpart == [-1, 0] else (0, 1, 1)
        # 1: subgraph to implement tf.onelike(input[:, 0]),
        # no need to worry about the dtype, because bool type is needed as Xor only support bool
        node_name = utils.make_name("const_zero")
        const_zero = ctx.make_const(name=node_name, np_val=np.array([0]).astype(np.int32))
        first_col_or_row = ctx.make_node(op_type="Gather", inputs=[node.input[0], const_zero.output[0]],
                                         attr={"axis": axis})
        first_col_or_row_casted = ctx.make_node(op_type="Cast", inputs=first_col_or_row.output,
                                                attr={"to": onnx_pb.TensorProto.BOOL})
        # line means one col or one row
        zero_line = ctx.make_node(op_type="Xor", inputs=first_col_or_row_casted.output * 2)
        one_line = ctx.make_node(op_type="Not", inputs=zero_line.output)

        # 2: "loop" to generate mask matrix: generate col or row of matrix one by one
        g = ctx.create_new_graph_with_same_config()
        node_name = utils.make_name("const_zero_bool")
        const_zero_bool = g.make_const(name=node_name, np_val=np.array([[0]]).astype(bool))
        g.set_dtype(const_zero_bool.output[0], onnx_pb.TensorProto.BOOL)

        g.add_graph_input("trip", onnx_pb.TensorProto.INT64, [])
        g.add_graph_input("cond", onnx_pb.TensorProto.BOOL, [])
        g.add_graph_input("line", onnx_pb.TensorProto.BOOL, [-1, -1])

        # shift right the line and add zero at the left.
        new_line = g.make_node(op_type="Concat", inputs=[const_zero_bool.output[0], "line"],
                               attr={"axis": counter_axis},
                               dtypes=[onnx_pb.TensorProto.BOOL])
        attr = {"axes": [counter_axis], "starts": [0], "ends": [-1]}
        inputs_map = {"data": new_line.output[0], **attr}
        slice_node = GraphBuilder(g).make_slice(inputs_map)

        g.make_node("Identity", ["cond"], outputs=["cond_out"])
        g.make_node("Identity", ["line"], outputs=["res"])
        g.make_node("Identity", [slice_node], outputs=["line_out"])

        g.add_graph_output("cond_out", onnx_pb.TensorProto.BOOL, [])
        g.add_graph_output("line_out", onnx_pb.TensorProto.BOOL, [-1, -1])
        g.add_graph_output("res", onnx_pb.TensorProto.BOOL, [-1, -1])

        # initial value of body vars
        shape = ctx.make_node(op_type="Shape", inputs=[node.input[0]])  # dtype of result is int64
        node_name = utils.make_name("line_num_index")
        col_or_row_num_index = ctx.make_const(name=node_name, np_val=np.array(axis).astype(np.int32))
        line_num = ctx.make_node(op_type="Gather", inputs=[shape.output[0], col_or_row_num_index.output[0]])
        trip_cnt = line_num.output[0]
        node_name = utils.make_name("true")
        cond = ctx.make_const(name=node_name, np_val=np.array(1).astype(bool))
        col_init = one_line.output[0]

        branches = {"body": g}
        loop_node = ctx.make_node(op_type="Loop", inputs=[trip_cnt, cond.output[0], col_init],
                                  output_count=2, branches=branches)
        # convert generated mask matrix from bool to right shape and data type
        squeeze = GraphBuilder(ctx).make_squeeze(
            {'data': loop_node.output[1], 'axes': [squeeze_axis]}, return_node=True)
        cast1 = ctx.make_node(op_type="Cast", inputs=squeeze.output, attr={"to": onnx_pb.TensorProto.FLOAT})
        if axis == 1:
            mask_matrix = ctx.make_node(op_type="Transpose", inputs=cast1.output)
        else:
            mask_matrix = squeeze
        cls._apply_mask_and_transform(ctx, node, mask_matrix)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        num_lower_const = node.inputs[1].get_tensor_value() if node.inputs[1].is_const() else None
        num_upper_const = node.inputs[2].get_tensor_value() if node.inputs[2].is_const() else None
        data, num_lower, num_upper = node.input
        rank = ctx.get_rank(data)
        int_max_val = utils.get_max_value(np.int64)
        dtype = ctx.get_dtype(data)
        if rank == 2:
            shape = ctx.make_node("Shape", [data]).output[0]
        else:
            whole_shape = ctx.make_node("Shape", [data]).output[0]
            shape = GraphBuilder(ctx).make_slice(
                {'data': whole_shape, 'starts': [-2], 'ends': [int_max_val], 'axes': [0]})
        if num_lower_const == 0 and num_upper_const == 0:
            if rank == 2:
                identity_node = ctx.make_node("EyeLike", [data])
            else:
                zero_tensor = helper.make_tensor("value", dtype, dims=[1], vals=[0])
                const_of_shape = ctx.make_node("ConstantOfShape", [shape], attr={'value': zero_tensor}).output[0]
                identity_node = ctx.make_node("EyeLike", [const_of_shape])
            cls._apply_mask_and_transform(ctx, node, identity_node)
            return
        zero_const = ctx.make_const(utils.make_name("zero"), np.array(0, np.int64)).output[0]
        one_const = ctx.make_const(utils.make_name("one"), np.array(1, np.int64)).output[0]
        conditions = []
        row_cnt = GraphBuilder(ctx).make_slice({'data': shape, 'axes': [0], 'starts': [0], 'ends': [1]})
        col_cnt = GraphBuilder(ctx).make_slice({'data': shape, 'axes': [0], 'starts': [1], 'ends': [2]})
        limit = ctx.make_node("Mul", [row_cnt, col_cnt]).output[0]
        # idx_cnt = ctx.make_node("Range", [zero_const, limit, one_const]).output[0]

        ones_of_shape = ctx.make_node("Expand", [one_const, limit]).output[0]
        idx_cnt = ctx.make_node("CumSum", [ones_of_shape, zero_const], attr={'exclusive': True}).output[0]

        idx_reshape = ctx.make_node("Reshape", [idx_cnt, shape]).output[0]
        row_idx = ctx.make_node("Div", [idx_reshape, col_cnt]).output[0]
        col_idx = ctx.make_node("Mod", [idx_reshape, col_cnt]).output[0]
        idx_diff = ctx.make_node("Sub", [col_idx, row_idx]).output[0]

        if num_upper_const is None or num_upper_const >= 0:
            if ctx.get_dtype(num_upper) != TensorProto.INT64:
                num_upper = ctx.make_node("Cast", [num_upper], attr={'to': TensorProto.INT64}).output[0]
            greater = ctx.make_node("Greater", [idx_diff, num_upper]).output[0]
            less_or_equal = ctx.make_node("Not", [greater])
            conditions.append(less_or_equal)
        if num_lower_const is None or num_lower_const >= 0:
            if ctx.get_dtype(num_lower) != TensorProto.INT64:
                num_lower = ctx.make_node("Cast", [num_lower], attr={'to': TensorProto.INT64}).output[0]
            num_lower_neg = ctx.make_node("Neg", [num_lower]).output[0]
            greater = ctx.make_node("Greater", [num_lower_neg, idx_diff]).output[0]
            less_or_equal = ctx.make_node("Not", [greater])
            conditions.append(less_or_equal)
        if len(conditions) == 0:
            node.type = "Identity"
            ctx.replace_inputs(node, [data])
            return
        if len(conditions) == 1:
            cond = conditions[0]
        if len(conditions) == 2:
            cond = ctx.make_node("And", inputs=[c.output[0] for c in conditions])
        cls._apply_mask_and_transform(ctx, node, cond)


def _make_softmax_cross_entropy_with_logits(ctx, label, logit, tf_ori_node):
    label_dtype = ctx.get_dtype(label.output[0])
    logit_dtype = ctx.get_dtype(logit.output[0])
    utils.make_sure(label_dtype == logit_dtype, "the following logic only works on same dtype of label and logit")

    log_softmax = ctx.make_node(op_type="LogSoftmax", inputs=logit.output)
    # implement tf.multiply(-1, tf.reduce_sum(tf.multiply(label, log_softmax), axis=1))
    mul1 = ctx.make_node(op_type="Mul", inputs=[label.output[0], log_softmax.output[0]])
    reduce_sum_output = GraphBuilder(ctx).make_reduce_sum(
        {"data": mul1.output[0], "axes": [-1], "keepdims": 1, "noop_with_empty_axes": 1})
    const_negative_one = ctx.make_const(name=utils.make_name("const_negative_one"),
                                        np_val=np.array(-1).astype(utils.ONNX_TO_NUMPY_DTYPE[logit_dtype]))
    mul2 = ctx.make_node(op_type="Mul", inputs=[const_negative_one.output[0], reduce_sum_output])
    shapes = tf_ori_node.output_shapes
    dtypes = tf_ori_node.output_dtypes
    ctx.remove_node(tf_ori_node.name)
    GraphBuilder(ctx).make_squeeze({'axes': [1], 'data': mul2.output[0], 'outputs': [tf_ori_node.output[0]]},
                                   shapes=[shapes[0]], dtypes=[dtypes[0]])


def sparse_softmax_cross_entropy_with_logits_op_by_gathernd(ctx, node, **kwargs):
    # make subgraph to implement one_hot, idea comes from onehot_op
    indices_name = node.input[1]
    indices_shape = ctx.get_shape(indices_name)
    if len(indices_shape) != 1:
        # TODO: this works for rank=1 but tensorflow supports more than this.
        # Same principle should work but we need to implement our own eye.
        raise ValueError("onehot op: only rank1 is supported")
    logit_name = node.input[0]
    logit_dtype = ctx.get_dtype(logit_name)
    logit_shape = ctx.get_shape(logit_name)
    utils.make_sure(logit_dtype, "Dtype of {} is None".format(logit_name))
    indices_dtype = ctx.get_dtype(indices_name)
    if indices_dtype != TensorProto.INT64:
        indices_cast = ctx.make_node("Cast", [indices_name], attr={"to": TensorProto.INT64})
        indices_name = indices_cast.output[0]
    indices_size = ctx.make_node("Size", [indices_name])
    gb = GraphBuilder(ctx)
    indices_unsqueeze = gb.make_unsqueeze({'data': indices_name, "axes": [1]}, return_node=True)
    zero_const = ctx.make_const(utils.make_name("zero"), np.array(0, dtype=np.int64))
    one_const = ctx.make_const(utils.make_name("one"), np.array(1, dtype=np.int64))
    id_name = utils.make_name("sparse_softmax_id")
    id_output = utils.port_name(id_name)
    controlflow.make_range(ctx, zero_const.output[0], indices_size.output[0], one_const.output[0],
                           id_output, id_name, shape=[-1], dtype=TensorProto.INT64)
    id_unsqueeze = gb.make_unsqueeze({'data': id_output, "axes": [1]}, return_node=True)
    indices_with_id = ctx.make_node("Concat",
                                    [id_unsqueeze.output[0], indices_unsqueeze.output[0]],
                                    attr={"axis": 1})
    log_softmax = ctx.make_node(op_type="LogSoftmax",
                                inputs=[logit_name], dtypes=[logit_dtype], shapes=[logit_shape])
    gathernd_name = utils.make_name("sparse_softmax_gathernd")
    gathernd_output = utils.port_name(gathernd_name)
    tensor.make_gathernd(ctx, log_softmax.output[0], indices_with_id.output[0], gathernd_output,
                         gathernd_name, logit_dtype, [logit_shape], [logit_dtype])
    const_name = utils.make_name("const_negative_one")
    const_negative_one = ctx.make_const(const_name, np.array(-1).astype(utils.map_onnx_to_numpy_type(logit_dtype)))
    mul2 = ctx.make_node(op_type="Mul", inputs=[const_negative_one.output[0], gathernd_output])
    shapes = node.output_shapes
    dtypes = node.output_dtypes
    ctx.remove_node(node.name)
    gb = GraphBuilder(ctx)
    gb.make_squeeze({'data': mul2.output[0], 'outputs': [node.output[0]], "axes": [1]},
                    shapes=[shapes[0]], dtypes=[dtypes[0]])


@tf_op("SoftmaxCrossEntropyWithLogits")
class SoftmaxCrossEntropyWithLogits:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        logits = node.inputs[0]
        logit_dtype = ctx.get_dtype(logits.output[0])
        labels = node.inputs[1]
        label_dtype = ctx.get_dtype(labels.output[0])
        if label_dtype != logit_dtype:
            labels = ctx.make_node("Cast", labels.output, attr={"to": logit_dtype}, dtypes=[logit_dtype])

        _make_softmax_cross_entropy_with_logits(ctx, labels, logits, node)


def _make_sparse_softmax_cross_entropy_with_logits(ctx, label, logit, tf_ori_node):
    logit = logit.output[0]
    label = label.output[0]
    label_dtype = ctx.get_dtype(label)
    logit_dtype = ctx.get_dtype(logit)
    utils.make_sure(label_dtype == logit_dtype, "the following logic only works on same dtype of label and logit")

    # when label is onehot, logic "tf.multiply(-1, tf.reduce_sum(tf.multiply(label, log_softmax), axis=1))" is equal to
    # "-log(q_i)" where i is the selected index specified by label, q_i = logic_i/sum, the detail process is as follows:
    # logit_exp=exp(logit) >> sum = tf.reduce_sum(logit_exp, axis = -1), masked_sum = reduce_sum(mul(logit_exp, mul))
    # >> -log(masked_sum/sum)
    logit_max = GraphBuilder(ctx).make_reduce_max({"data": logit, "axes": [-1], "keepdims": 1})
    logit_norm = ctx.make_node(op_type="Sub", inputs=[logit, logit_max]).output[0]
    logit_exp = ctx.make_node(op_type="Exp", inputs=[logit_norm]).output[0]
    logit_exp_sum = GraphBuilder(ctx).make_reduce_sum(
        {"data": logit_exp, "axes": [-1], "keepdims": 0, "noop_with_empty_axes": 1})
    masked = ctx.make_node(op_type="Mul", inputs=[label, logit_exp]).output[0]
    masked_sum = GraphBuilder(ctx).make_reduce_sum(
        {"data": masked, "axes": [-1], "keepdims": 0, "noop_with_empty_axes": 1})
    probability = ctx.make_node(op_type="Div", inputs=[masked_sum, logit_exp_sum]).output[0]
    log_prob = ctx.make_node(op_type="Log", inputs=[probability]).output[0]
    const_negative_one = ctx.make_const(name=utils.make_name("const_negative_one"),
                                        np_val=np.array(-1).astype(utils.ONNX_TO_NUMPY_DTYPE[logit_dtype])).output[0]

    shapes = tf_ori_node.output_shapes
    dtypes = tf_ori_node.output_dtypes
    ctx.remove_node(tf_ori_node.name)
    ctx.make_node(op_type="Mul", inputs=[log_prob, const_negative_one],
                  outputs=[tf_ori_node.output[0]], shapes=[shapes[0]], dtypes=[dtypes[0]])


@tf_op("SparseSoftmaxCrossEntropyWithLogits")
class SparseSoftmaxCrossEntropyWithLogits:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # make subgraph to implement one_hot, idea comes from onehot_op
        indices_name = node.input[1]
        indices_shape = ctx.get_shape(indices_name)
        if len(indices_shape) != 1:
            # TODO: this works for rank=1 but tensorflow supports more than this.
            # Same principle should work but we need to implement our own eye.
            raise ValueError("onehot op: only rank1 is supported")
        logit_name = node.input[0]
        depth = ctx.get_shape(logit_name)[-1]
        # if number of classes is unknown or too large
        if depth == utils.ONNX_UNKNOWN_DIMENSION or depth > 20000:
            sparse_softmax_cross_entropy_with_logits_op_by_gathernd(ctx, node, **kwargs)
            return
        logit_dtype = ctx.get_dtype(logit_name)
        utils.make_sure(logit_dtype, "Dtype of {} is None".format(logit_name))

        dtype = utils.map_onnx_to_numpy_type(logit_dtype)
        eye = np.eye(depth).astype(dtype)
        const_name = utils.make_name("const_eye")
        const_eye = ctx.make_const(name=const_name, np_val=eye)
        onehot = ctx.make_node(op_type="Gather", inputs=[const_eye.output[0], indices_name], attr={"axis": 0})
        log_softmax = ctx.make_node(op_type="LogSoftmax", inputs=[logit_name])
        # implement tf.multiply(np.float32(-1.0), tf.reduce_sum(tf.multiply(one_hot, log_softmax), axis=1))
        mul1 = ctx.make_node(op_type="Mul", inputs=[onehot.output[0], log_softmax.output[0]])
        reduce_sum_output = GraphBuilder(ctx).make_reduce_sum(
            {"data": mul1.output[0], "axes": [1], "keepdims": 1, "noop_with_empty_axes": 1})
        const_name = utils.make_name("const_negative_one")
        const_negative_one = ctx.make_const(name=const_name, np_val=np.array(-1).astype(dtype))
        mul2 = ctx.make_node(op_type="Mul", inputs=[const_negative_one.output[0], reduce_sum_output])

        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        ctx.make_node(op_type="Squeeze", inputs=[mul2.output[0]], outputs=[node.output[0]], attr={"axes": [1]},
                      shapes=[shapes[0]], dtypes=[dtypes[0]])

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        # float32/64 output = SparseSoftmaxCrossEntropyWithLogits(float32/64 features, int32/64 labels)
        # the detail math process of this op is: a = onehot(labels), b = logsoftmax(features), reduce_sum(mul(a, b))
        logit_node = node.inputs[0]
        logit_shape = ctx.get_shape(node.input[0])
        logit_dtype = ctx.get_dtype(node.input[0])

        label_name = node.input[1]

        if logit_shape is not None and logit_shape[-1] != -1:
            num_class = logit_shape[-1]
            node_nme = utils.make_name("onehot_depth")
            depth_node = ctx.make_const(node_nme, np.array([num_class]).astype(np.int64)).output[0]
        else:
            logit_shape = ctx.make_node("Shape", [node.input[0]]).output[0]
            slice_args = {"data": logit_shape,
                          "starts": [-1], "ends": [int(utils.get_max_value(np.int32))]}
            num_class = GraphBuilder(ctx).make_slice(kwargs=slice_args)
            depth_node = num_class
        values_node = ctx.make_const(utils.make_name("onehot_values"), np.array([0, 1]).astype(np.int64)).output[0]
        label_dtype = ctx.get_dtype(label_name)
        if label_dtype != TensorProto.INT64:
            onehot_indice = ctx.make_node("Cast", [label_name], attr={"to": TensorProto.INT64}).output[0]
        else:
            onehot_indice = label_name
        if ctx.opset < 11:
            label_node = ctx.make_node(op_type="OneHot",
                                       inputs=[onehot_indice, depth_node, values_node])
        else:
            # OneHot is very slow but this workaround requires opset 11
            index_unsq = GraphBuilder(ctx).make_unsqueeze({'data': onehot_indice, 'axes': [-1]})
            depth_sq = GraphBuilder(ctx).make_squeeze({'data': depth_node, 'axes': [0]})
            zero_const = ctx.make_const(utils.make_name("const_zero"), np.array(0, np.int64)).output[0]
            one_const = ctx.make_const(utils.make_name("const_one"), np.array(1, np.int64)).output[0]
            dp_range = ctx.make_node("Range", [zero_const, depth_sq, one_const]).output[0]
            label_node = ctx.make_node("Equal", [index_unsq, dp_range])
        # the above logic makes output dtype of label_node now always int64
        # make sure label has same dtype as logit
        if logit_dtype != ctx.get_dtype(label_node.output[0]):
            label_node = ctx.make_node("Cast", label_node.output, attr={"to": logit_dtype}, dtypes=[logit_dtype])

        _make_sparse_softmax_cross_entropy_with_logits(ctx, label_node, logit_node, node)


@tf_op("CTCGreedyDecoder")
class CTCGreedyDecoder:
    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # shape = [max_time, batch_size, num_classes]
        inp = node.input[0]
        # shape = [batch_size]
        seq_lens = node.input[1]
        seq_lens_int64 = ctx.make_node("Cast", [seq_lens], attr={"to": TensorProto.INT64}).output[0]
        # shape = [1, batch_size, 1]
        seq_lens_unsq = GraphBuilder(ctx).make_unsqueeze({"data": seq_lens_int64, "axes": [0, 2]})

        merge_repeated = node.get_attr_value("merge_repeated", False)

        inp_shape = ctx.make_node("Shape", [inp]).output[0]
        max_time_unsq, num_batch_unsq, num_classes_unsq = ctx.make_node("Split", [inp_shape], attr={'num_outputs': 3},
                                                                        output_count=3).output
        max_time = GraphBuilder(ctx).make_squeeze({"data": max_time_unsq, "axes": [0]})
        num_batch = GraphBuilder(ctx).make_squeeze({"data": num_batch_unsq, "axes": [0]})
        num_classes = GraphBuilder(ctx).make_squeeze({"data": num_classes_unsq, "axes": [0]})
        const_one = ctx.make_const(utils.make_name("const_one"), np.array(1, np.int64)).output[0]
        const_one_unsq = ctx.make_const(utils.make_name("const_one"), np.array([1], np.int64)).output[0]
        const_zero = ctx.make_const(utils.make_name("const_zero"), np.array(0, np.int64)).output[0]
        blank_label = ctx.make_node("Sub", [num_classes, const_one]).output[0]
        time = ctx.make_node("Range", [const_zero, max_time, const_one]).output[0]
        batch = ctx.make_node("Range", [const_zero, num_batch, const_one]).output[0]
        # shape = [max_time, 1, 1]
        time_unsq = GraphBuilder(ctx).make_unsqueeze({"data": time, "axes": [1, 2]})
        valid_elts = ctx.make_node("Less", [time_unsq, seq_lens_unsq]).output[0]
        # shape = [max_time, batch_size, 1]
        valid_mask = ctx.make_node("Cast", [valid_elts], attr={"to": TensorProto.FLOAT}).output[0]
        # shape = [max_time, batch_size, num_classes]
        valid_inp = ctx.make_node("Mul", [inp, valid_mask]).output[0]

        # shape = [max_time, batch_size, 1]
        max_val, max_idx = ctx.make_node("TopK", [valid_inp, const_one_unsq], attr={"axis": 2},
                                         output_count=2, op_name_scope=node.name).output
        # shape = [batch_size, 1]
        sum_max = GraphBuilder(ctx).make_reduce_sum({"data": max_val, "axes": [0], "keepdims": False})
        sum_max_neg = ctx.make_node("Neg", [sum_max]).output[0]

        valid_elts_sq = GraphBuilder(ctx).make_squeeze({"data": valid_elts, "axes": [2]})
        max_idx_sq = GraphBuilder(ctx).make_squeeze({"data": max_idx, "axes": [2]})
        # shape = [batch_size, max_time]
        max_idx_trans = ctx.make_node("Transpose", [max_idx_sq], attr={"perm": [1, 0]}).output[0]
        valid_elts_trans = ctx.make_node("Transpose", [valid_elts_sq], attr={"perm": [1, 0]}).output[0]

        # value = [batch_size, max_time]
        idx_shape = ctx.make_node("Shape", [max_idx_trans]).output[0]
        keep_idx = ctx.make_node("Less", [max_idx_trans, blank_label]).output[0]
        keep_idx = ctx.make_node("And", [keep_idx, valid_elts_trans]).output[0]

        if merge_repeated:
            # val = [batch_size, 1]
            shift_row_shape = ctx.make_node("Concat", [num_batch_unsq, const_one_unsq], attr={'axis': 0}).output[0]
            neg_one_tensor = helper.make_tensor("value", onnx_pb.TensorProto.INT64, dims=[1], vals=[-1])
            # shape = [batch_size, 1]
            neg_ones = ctx.make_node("ConstantOfShape", [shift_row_shape], {'value': neg_one_tensor}).output[0]
            max_idx_cut = GraphBuilder(ctx).make_slice(
                {"data": max_idx_trans, "starts": [0], "ends": [-1], "axes": [1]})
            # shape = [batch_size, max_time]
            max_idx_shift = ctx.make_node("Concat", [neg_ones, max_idx_cut], attr={"axis": 1}).output[0]
            repeat_elts = ctx.make_node("Equal", [max_idx_shift, max_idx_trans]).output[0]
            not_repeat = ctx.make_node("Not", [repeat_elts]).output[0]
            keep_idx = ctx.make_node("And", [keep_idx, not_repeat]).output[0]

        batch_unsq = GraphBuilder(ctx).make_unsqueeze({"data": batch, "axes": [1]})
        batch_expand = ctx.make_node("Expand", [batch_unsq, idx_shape]).output[0]
        keep_idx_int = ctx.make_node("Cast", [keep_idx], attr={"to": TensorProto.INT64}).output[0]
        filtered_time = ctx.make_node("CumSum", [keep_idx_int, const_one], attr={"exclusive": True}).output[0]

        flat_shape = ctx.make_const(utils.make_name("const_neg_one"), np.array([-1], np.int64)).output[0]
        flat_shape2 = ctx.make_const(utils.make_name("const_shape"), np.array([-1, 1], np.int64)).output[0]
        idx_flat = ctx.make_node("Reshape", [max_idx_trans, flat_shape]).output[0]
        keep_idx_flat = ctx.make_node("Reshape", [keep_idx, flat_shape]).output[0]
        time_flat = ctx.make_node("Reshape", [filtered_time, flat_shape2]).output[0]
        batch_flat = ctx.make_node("Reshape", [batch_expand, flat_shape2]).output[0]
        sparse_idx = ctx.make_node("Concat", [batch_flat, time_flat], attr={'axis': 1}).output[0]
        idx_compress = ctx.make_node("Compress", [idx_flat, keep_idx_flat], attr={'axis': 0}, shapes=[[-1]],
                                     op_name_scope=node.name).output[0]
        sparse_idx_compress = ctx.make_node("Compress", [sparse_idx, keep_idx_flat], attr={'axis': 0},
                                            shapes=[[-1, 2]], op_name_scope=node.name).output[0]
        max_sparse_idx = GraphBuilder(ctx).make_reduce_max({"data": sparse_idx_compress, "axes": [0],
                                                            "keepdims": False})
        max_time = GraphBuilder(ctx).make_slice(
            {"data": max_sparse_idx, "starts": [1], "ends": [2], "axes": [0]})
        max_time_inc = ctx.make_node("Add", [max_time, const_one]).output[0]
        sparse_shape = ctx.make_node("Concat", [num_batch_unsq, max_time_inc], attr={'axis': 0}).output[0]

        ctx.replace_all_inputs(node.output[0], sparse_idx_compress)
        ctx.replace_all_inputs(node.output[1], idx_compress)
        ctx.replace_all_inputs(node.output[2], sparse_shape)
        ctx.replace_all_inputs(node.output[3], sum_max_neg)

        ctx.remove_node(node.name)
