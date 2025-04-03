# SPDX-License-Identifier: Apache-2.0


"""
tensor
"""

import logging
import sys

import numpy as np
from onnx import onnx_pb, helper
from onnx.onnx_pb import TensorProto

from tf2onnx import constants, utils
from tf2onnx.graph_builder import GraphBuilder
from tf2onnx.handler import tf_op
from tf2onnx.onnx_opset import nn, math
from tf2onnx.constants import NCHW_TO_NHWC, NHWC_TO_NCHW

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring,unused-variable,pointless-string-statement,invalid-name


def _convert_shapenode_to_int64(ctx, node, input_number):
    """cast int32 shape into int64 shape."""
    name = node.input[input_number]

    cast_node = ctx.insert_new_node_on_input(node, "Cast", name, to=onnx_pb.TensorProto.INT64)
    ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.INT64)
    ctx.copy_shape(name, cast_node.output[0])


def _wrap_concat_with_cast(ctx, node):
    """wrap concat in casts for opset < 8 since it only supports."""
    supported_types = [onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.FLOAT16]
    dtype = ctx.get_dtype(node.output[0])
    need_casting = dtype not in supported_types
    if need_casting:
        output_name = node.output[0]
        # cast each inputs to float
        for i, inp in enumerate(node.inputs):
            input_cast = ctx.insert_new_node_on_input(node, "Cast", node.input[i],
                                                      to=onnx_pb.TensorProto.FLOAT)
            ctx.set_dtype(input_cast.output[0], onnx_pb.TensorProto.FLOAT)
        next_nodes = ctx.find_output_consumers(node.output[0])
        # cast output back to dtype unless the next op is a cast
        if next_nodes[0].type != "Cast":
            output_cast = ctx.insert_new_node_on_output("Cast", output_name, name=node.child_name(),
                                                        to=dtype)
            ctx.set_dtype(output_cast.output[0], dtype)
            ctx.copy_shape(output_name, output_cast.output[0])
        ctx.set_dtype(node.output[0], onnx_pb.TensorProto.FLOAT)


@tf_op("Size")
class Size:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        output_name = node.output[0]
        dtype = ctx.get_dtype(output_name)
        # TF size can output int32 or int64 but onnx only does int 64
        if dtype != onnx_pb.TensorProto.INT64:
            ctx.set_dtype(output_name, onnx_pb.TensorProto.INT64)
            output_cast = ctx.insert_new_node_on_output("Cast", output_name, name=node.child_name(),
                                                        to=dtype)
            ctx.set_dtype(output_cast.output[0], dtype)
            ctx.copy_shape(output_name, output_cast.output[0])


@tf_op("Flatten")
class Flatten:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        # no change for us
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # no change
        cls.version_1(ctx, node, **kwargs)


@tf_op("Dropout")
class Dropout:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass

    @classmethod
    def version_6(cls, ctx, node, **kwargs):
        pass

    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        pass

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        pass

    @classmethod
    def version_12(cls, ctx, node, **kwargs):
        pass


@tf_op("Identity")
class Identity:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        if node.inputs[0].is_const():
            # should not remove the identity node if it is output of the graph
            if node.output[0] in ctx.outputs:
                return
            # if identity has a const as input, remove it
            input_name = node.input[0]
            output_name = node.output[0]
            ctx.replace_all_inputs(output_name, input_name)  # ops=ctx.get_nodes()
            ctx.remove_node(node.name)


@tf_op("IdentityN")
class IdentityN:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        ctx.remove_node(node.name)
        for input_name, output_name in zip(node.input, node.output):
            ctx.replace_all_inputs(output_name, input_name)  # ops=ctx.get_nodes()


@tf_op("EnsureShape")
class EnsureShape:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.type = "Identity"


@tf_op("Reshape")
class Reshape:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T output = Reshape(T tensor, Tshape shape, @type Tshape)
        # T reshaped = Reshape(T data, @INTS shape) - but takes a optional 2nd input for shape
        shape_node = node.inputs[1]
        shape = shape_node.get_tensor_value()
        if shape is None:
            logger.error("Reshape on node %s does not have a const shape", node.name)
            return
        ctx.remove_input(node, node.input[1], 1)
        node.set_attr("shape", shape)
        ctx.set_shape(node.output[0], shape)

    @classmethod
    def version_5(cls, ctx, node, **kwargs):
        dtype = ctx.get_dtype(node.output[0])
        if node.inputs[1].is_const():
            target_shape = node.inputs[1].get_tensor_value(as_list=True)
            inp_shape = ctx.get_shape(node.input[0])
            if inp_shape is not None and inp_shape == target_shape:
                # Remove useless Reshape
                node.type = "Identity"
                ctx.replace_inputs(node, [node.input[0]])
                return
        need_casting = dtype in [onnx_pb.TensorProto.INT32,
                                 onnx_pb.TensorProto.INT16,
                                 onnx_pb.TensorProto.INT64]
        # onnx wants reshape.input[1] to have the value be int64 which is not the case for tensorflow.
        _convert_shapenode_to_int64(ctx, node, 1)
        if ctx.opset >= 8 or not need_casting:
            # onnx reshape can handle the type - done
            return

        # onnx < opset 8 does not know reshape for other types than float*, wrap the reshape in casts
        ctx.insert_new_node_on_input(node, "Cast", node.input[0], to=onnx_pb.TensorProto.FLOAT)

        # if the next node is already a cast we don't need to insert another one
        next_nodes = ctx.find_output_consumers(node.output[0])
        if len(next_nodes) != 1 or next_nodes[0].type != "Cast":
            output_cast = ctx.insert_new_node_on_output("Cast", node.output[0], name=node.child_name(),
                                                        to=dtype)
            ctx.set_dtype(output_cast.output[0], dtype)
            ctx.copy_shape(node.output[0], output_cast.output[0])
        ctx.set_dtype(node.output[0], onnx_pb.TensorProto.FLOAT)


@tf_op("Squeeze")
class Squeeze:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T output = Squeeze(T input, @list(int) squeeze_dims)
        # T squeezed = Squeeze(T data, @AttrType.INTS axes), axes are list of positive integers.
        axes = node.get_attr_value("squeeze_dims")
        if axes is None:
            axes = []
        else:
            del node.attr["squeeze_dims"]

        # TF uses empty axes to indicate that all 1 dims should be squeezed
        if len(axes) > 0:
            neg_axis = any([val < 0 for val in axes])
            if neg_axis and ctx.opset < 11:
                shape = ctx.get_shape(node.input[0])
                utils.make_sure(shape is not None, "squeeze with negative axes and unknown rank requires opset >= 11")
                shape_len = len(shape)
                axes = [a + shape_len if a < 0 else a for a in axes]
            if ctx.opset < 13:
                node.set_attr("axes", axes)
            else:
                axes_const = ctx.make_const(utils.make_name("axes_const"), np.array(axes, dtype=np.int64))
                ctx.replace_inputs(node, [node.input[0], axes_const.output[0]])

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Opset 13: parameters moved to inputs
        cls.version_1(ctx, node, **kwargs)


@tf_op("Transpose")
class Transpose:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T y = Transpose(T x, Tperm perm, @type Tperm)
        # T transposed = Transpose(T data, @INTS perm)
        if len(node.input) > 1:
            perm = node.inputs[1]
            if perm.is_const():
                # perms is passed as const
                dims = perm.get_tensor_value()
                ctx.remove_input(node, node.input[1], 1)
                node.set_attr("perm", dims)
            else:
                utils.make_sure(False, "perm can't be dynamic in ONNX")
        else:
            # graph rewrite moved perm to attribute
            pass


@tf_op("Concat")
class Concat:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # old concat op has axis as input[0]
        node.type = "Concat"
        axis_node = node.inputs[0]
        axis_val = axis_node.get_tensor_value()
        ctx.remove_input(node, node.input[0], 0)

        if axis_val < 0:  # onnxruntime does not support -1 axis, but TF supports.
            input_shape = ctx.get_shape(node.input[0])
            axis_val = len(input_shape) + axis_val
        node.set_attr("axis", axis_val)

        if ctx.opset < 8:
            # opset < 8: might need to wrap concat in casts since only float is supported
            _wrap_concat_with_cast(ctx, node)
            return

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.version_1(ctx, node, **kwargs)


@tf_op("ConcatV2")
class ConcatV2:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T output = ConcatV2(T values, Tidx axis, @int N, @type Tidx)
        # T concat_result = Concat(T inputs, @INT axis)
        # if any input is empty, remove the input and concat the others
        # NOTE: workaround for https://github.com/Microsoft/onnxruntime/issues/681
        node.type = "Concat"
        removed_indices = []
        for i, inp in enumerate(node.inputs):
            if inp.is_const() and inp.get_tensor_value(as_list=False).size == 0:
                removed_indices.append(i)
        for i in reversed(removed_indices):
            ctx.remove_input(node, node.input[i], i)
        # all inputs are deleted
        if not node.input:
            raise RuntimeError("all inputs of {} are empty".format(node.name))

        axis_node = node.inputs[-1]
        utils.make_sure(axis_node.is_const(), "{} needs to be const".format(axis_node.name))
        axis_val = axis_node.get_tensor_value()
        ctx.remove_input(node, node.input[-1], len(node.input) - 1)

        if axis_val < 0:  # onnxruntime does not support -1 axis, but TF supports.
            input_shape = None
            for node_input in node.input:
                input_shape = ctx.get_shape(node_input)
                if input_shape is not None:
                    break
            utils.make_sure(input_shape is not None,
                            "the shapes of the following inputs are None: {}".format(', '.join(node.input)))
            axis_val = len(input_shape) + axis_val
        node.set_attr("axis", axis_val)

        if ctx.opset < 8:
            # opset < 8: might need to wrap concat in casts since only float is supported
            _wrap_concat_with_cast(ctx, node)
            return


@tf_op("Slice")
class Slice:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T output = Slice(T input, Index begin, Index size)
        # T output = Slice(T input, Tind starts, Tind ends, Tind axes, Tind steps)
        # "ends" are exclusive, "axes" and "steps" are optional, their default val are [0, ...] and 1
        input_tensor = node.input[0]
        starts = node.input[1]
        size = node.input[2]
        # in tf, size can be -1 which means all elem are taken, so size can't be added starts directly.
        # the way to make sure size are not less than 0: set "sizes"'s elem to be int_max if elem val is -1
        size_dtype = ctx.get_dtype(size)
        size_np_dtype = utils.map_onnx_to_numpy_type(size_dtype)
        if ctx.get_node_by_output(size).is_const() and ctx.get_node_by_output(starts).is_const():
            starts = ctx.get_node_by_output(starts).get_tensor_value()
            sizes = ctx.get_node_by_output(size).get_tensor_value()
            ends = []
            for start, size in zip(starts, sizes):
                # get all elements
                if size == -1:
                    dtype = ctx.get_dtype(node.input[1])
                    utils.make_sure(dtype, "dtype of {} is None".format(node.input[1]))
                    utils.make_sure(dtype, "dtype of {} is None".format(node.input[1]))
                    ends.append(np.iinfo(dtype).max)
                else:
                    ends.append(start + size)

        else:
            neg_one_val = np.array([-1]).astype(size_np_dtype)
            neg_one = ctx.make_const(utils.make_name("const"), neg_one_val).output[0]

            int_max_val = np.array([utils.get_max_value(size_np_dtype)]).astype(size_np_dtype)
            int_max = ctx.make_const(utils.make_name("largest_int_val"), int_max_val).output[0]

            size_are_neg_one_flag = ctx.make_node("Equal", [neg_one, size]).output[0]
            size_are_neg_one_flag = ctx.make_node("Cast", [size_are_neg_one_flag], attr={"to": size_dtype}).output[0]
            value_to_add = ctx.make_node("Mul", [int_max, size_are_neg_one_flag]).output[0]
            size_processed = ctx.make_node("Add", [size, value_to_add]).output[0]
            ends = ctx.make_node("Add", [starts, size_processed]).output[0]

        ctx.remove_node(node.name)
        inputs_map = {"data": input_tensor, "starts": starts, "ends": ends}
        kwargs = {**inputs_map, "outputs": node.output}
        _ = GraphBuilder(ctx).make_slice(kwargs, name=node.name)

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        cls.version_1(ctx, node, **kwargs)


@tf_op("Roll")
class Roll:
    @classmethod
    def any_version(cls, opset, ctx, node, **kwargs):
        utils.make_sure(node.inputs[2].is_const(), "Can only convert Roll is axis is const")
        axes = node.inputs[2].get_tensor_value()
        if not isinstance(axes, list):
            axes = [axes]
        rank = ctx.get_rank(node.input[0])
        axes = [a if a >= 0 else a + rank for a in axes]

        shifts_dtype = ctx.get_dtype(node.input[1])
        if shifts_dtype != TensorProto.INT64:
            shifts_casted = ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=TensorProto.INT64).output[0]
        else:
            shifts_casted = node.input[1]

        if len(axes) == 1:
            unsqueeze_node = GraphBuilder(ctx).make_unsqueeze(
                {'data': shifts_casted, "axes": [0]}, op_name_scope=node.name, return_node=True)
            shifts_split = [unsqueeze_node.output[0]]
        else:
            shifts_split = ctx.make_node("Split", [shifts_casted], attr={'axis': 0, 'num_outputs': len(axes)},
                                         output_count=len(axes), op_name_scope=node.name).output

        zero_const = ctx.make_const(utils.make_name("zeros_const"), np.array([0], np.int64)).output[0]
        shape_node = ctx.make_node("Shape", [node.input[0]], op_name_scope=node.name)

        data = node.input[0]

        for axis, shift in zip(axes, shifts_split):
            len_along_axis = GraphBuilder(ctx).make_slice(
                {"data": shape_node.output[0], "ends": [axis + 1], "starts": [axis]})
            shift_mod = ctx.make_node("Mod", [shift, len_along_axis]).output[0]
            remaining_len = ctx.make_node("Sub", [len_along_axis, shift_mod], op_name_scope=node.name).output[0]
            axes_const = ctx.make_const(utils.make_name("axes_const"), np.array([axis], np.int64)).output[0]
            slice_one = ctx.make_node("Slice", [data, zero_const, remaining_len, axes_const], op_name_scope=node.name)
            slice_two = ctx.make_node("Slice", [data, remaining_len, len_along_axis, axes_const],
                                      op_name_scope=node.name)
            concat_node = ctx.make_node("Concat", [slice_two.output[0], slice_one.output[0]],
                                        attr={'axis': axis}, op_name_scope=node.name)
            data = concat_node.output[0]

        ctx.replace_all_inputs(node.output[0], data)
        ctx.remove_node(node.name)

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        cls.any_version(10, ctx, node, **kwargs)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        cls.any_version(11, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Parameters moved to inputs for operator Squeeze, Unsqueeze.
        cls.any_version(13, ctx, node, **kwargs)


@tf_op("Gather")
class Gather:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.type = "Gather"

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # no change
        cls.version_1(ctx, node, **kwargs)


@tf_op("GatherV2")
class GatherV2:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # for GatherV2 axis come as input
        err_msg = "Opset 12 required for batch_dims attribute of GatherV2"
        utils.make_sure(node.get_attr_value("batch_dims", 0) == 0, err_msg)
        node.type = "Gather"
        utils.make_sure(node.inputs[2].is_const(), "Axis of GatherV2 node must be constant")
        axis = node.inputs[2].get_tensor_value()
        ctx.remove_input(node, node.input[2], 2)
        node.set_attr("axis", axis)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # no change
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_12(cls, ctx, node, **kwargs):
        batch_dims = node.get_attr_value("batch_dims", 0)
        if batch_dims == 0:
            cls.version_1(ctx, node, **kwargs)
            return
        # If batch_dims is not zero, use GatherND to simulate Gather with batch dims.
        data_inp, indices_inp, axis_inp = node.input
        utils.make_sure(node.inputs[2].is_const(), "Axis of GatherV2 node must be constant")
        axis = node.inputs[2].get_tensor_value()
        ctx.remove_input(node, axis_inp, 2)
        if ctx.get_dtype(indices_inp) != TensorProto.INT64:
            indices_inp = ctx.make_node("Cast", [indices_inp], attr={'to': TensorProto.INT64}).output[0]
        unperm = None
        # GatherND doesn't take an axis so we have to transpose stuff around
        if axis != batch_dims:
            data_rank = ctx.get_rank(data_inp)
            indices_rank = ctx.get_rank(indices_inp)
            result_rank = data_rank + indices_rank - 1 - batch_dims
            shift_amt = axis - batch_dims
            err_msg = "Cannot convert GatherV2 with batch dims since inputs have unknown ranks."
            utils.make_sure(data_rank is not None and indices_rank is not None, err_msg)
            perm = list(range(data_rank))
            perm = perm[:batch_dims] + perm[axis:axis+1] + perm[batch_dims:axis] + perm[axis+1:]
            data_inp = ctx.make_node("Transpose", [data_inp], attr={'perm': perm}).output[0]
            ctx.replace_input(node, node.input[0], data_inp, 0)
            unperm = list(range(result_rank))
            j = indices_rank+shift_amt
            unperm = unperm[:batch_dims] + unperm[indices_rank:j] + unperm[batch_dims:indices_rank] + unperm[j:]
        node.type = "GatherND"
        unsqueeze_node = GraphBuilder(ctx).make_unsqueeze({'data': indices_inp, 'axes': [-1]})
        ctx.replace_input(node, node.input[1], unsqueeze_node, 1)
        if unperm is not None:
            ctx.update_node_shape_dtype(node, override=True)
            ctx.insert_new_node_on_output("Transpose", node.output[0], perm=unperm)


def _make_gathernd_inner_loop(ctx, params, index, dtype):
    """create the inner loop for GatherNd."""
    # gather_cur = params
    # for (int i = 0; i < size(index); i++)
    #   gather_res = gather(gather_cur, index[i])
    scope_name = utils.make_name("gathernd_inner_loop")
    trip_node = ctx.make_node("Size", [index.output[0]])
    cond_const = ctx.make_const(utils.make_name("cond"), np.ones((), dtype=bool))
    trip_name = utils.make_name("i")
    cond_name = utils.make_name("cond")
    cond_out_name = utils.make_name("cond_out")
    cur_name = utils.make_name("gather_cur")
    result_name = utils.make_name("res")

    # body graph creation
    g = ctx.create_new_graph_with_same_config()
    g.add_graph_input(trip_name, TensorProto.INT64, [1])
    g.add_graph_input(cond_name, TensorProto.BOOL, [])
    g.add_graph_input(cur_name, dtype, [])
    g.parent_graph = ctx

    index_i = g.make_node("Gather", [index.output[0], trip_name], attr={"axis": 0})
    gather = g.make_node("Gather", [cur_name, index_i.output[0]], attr={"axis": 0})
    GraphBuilder(g).make_squeeze(
        {'data': gather.output[0], "axes": [0], 'outputs': [result_name]})
    g.make_node("Identity", [cond_name], outputs=[cond_out_name])

    g.add_graph_output(cond_out_name, TensorProto.BOOL, [])
    g.add_graph_output(result_name, dtype, [])

    branches = {"body": g}
    inner_loop = ctx.make_node("Loop",
                               [trip_node.output[0], cond_const.output[0], params],
                               op_name_scope=scope_name, skip_conversion=False, branches=branches)
    return inner_loop


def make_gathernd(ctx, params, indices, output, scope_name, t_params, shapes, dtypes):
    """make GatherNd op."""
    # Tparams output = GatherNd(Tparams params, Tidx indices)
    scope_name = utils.make_name(scope_name)
    # reshape indices into [sum(indices[:-1]), indices[-1]]
    indices_shape = ctx.make_node("Shape", [indices], dtypes=[TensorProto.INT64])
    indices_size = ctx.make_node("Size", [indices])
    attr = {"axes": [0], "ends": [sys.maxsize], "starts": [-1]}
    inputs_map = {"data": indices_shape.output[0], **attr}
    inner_shape = GraphBuilder(ctx).make_slice(inputs_map, dtypes=[TensorProto.INT64])
    outter_shape = ctx.make_node("Div",
                                 [indices_size.output[0], inner_shape],
                                 dtypes=[TensorProto.INT64])
    flatten_shape = ctx.make_node("Concat",
                                  [outter_shape.output[0], inner_shape],
                                  attr={"axis": 0},
                                  dtypes=[TensorProto.INT64])
    flatten_indices = ctx.make_node("Reshape", [indices, flatten_shape.output[0]])

    # outter loop for each index
    # for (int i=0; i<outter_shape; i++) inner_loop(params, flatten_indices[i])
    cond_const = ctx.make_const(utils.make_name("cond"), np.ones((), dtype=bool))
    ctx.make_const(utils.make_name("dummy"), np.ones((), dtype=np.int64))

    # body graph creation
    g = ctx.create_new_graph_with_same_config()
    trip_name = utils.make_name("i")
    cond_name = utils.make_name("cond")
    cond_out_name = utils.make_name("cond_out")
    dummy_name = utils.make_name("dummy")
    dummy_out_name = utils.make_name("dummy_out")
    result_name = utils.make_name("res")

    g.add_graph_input(trip_name, TensorProto.INT64, [1])
    g.add_graph_input(cond_name, TensorProto.BOOL, [])
    g.add_graph_input(dummy_name, t_params, [])
    g.parent_graph = ctx

    index = g.make_node("Gather", [flatten_indices.output[0], trip_name], attr={"axis": 0})
    index_squeeze = GraphBuilder(g).make_squeeze(
        {'data': index.output[0], "axes": [0]}, return_node=True)
    # inner loop to gather result
    inner_loop = _make_gathernd_inner_loop(g, params, index_squeeze, t_params)
    g.make_node("Identity", [cond_name], outputs=[cond_out_name])
    g.make_node("Identity", [dummy_name], outputs=[dummy_out_name])
    g.make_node("Identity", [inner_loop.output[0]], outputs=[result_name])

    g.add_graph_output(cond_out_name, TensorProto.BOOL, [])
    g.add_graph_output(dummy_out_name, t_params, [])
    g.add_graph_output(result_name, t_params, [])

    branches = {"body": g}
    gathernd_loop = ctx.make_node("Loop",
                                  [outter_shape.output[0], cond_const.output[0], params],
                                  output_count=2,
                                  op_name_scope=scope_name, skip_conversion=False, branches=branches)

    # reshape to target shape
    # output shape of gathernd: indices.shape[:-1] + gathernd_output.shape[1:]
    inner_loop_shape = ctx.make_node("Shape", [gathernd_loop.output[1]], dtypes=[TensorProto.INT64])
    # workaround in case gathernd_loop is 1-dimensional
    one_const = ctx.make_const(utils.make_name("one"), np.array([1], dtype=np.int64))
    inner_loop_shape_ = ctx.make_node("Concat",
                                      [inner_loop_shape.output[0], one_const.output[0]],
                                      attr={"axis": 0},
                                      dtypes=[TensorProto.INT64])
    attr = {"axes": [0], "ends": [sys.maxsize], "starts": [1]}
    inputs_map = {"data": inner_loop_shape_.output[0], **attr}
    output_inner_shape = GraphBuilder(ctx).make_slice(inputs_map, dtypes=[TensorProto.INT64])
    attr = {"axes": [0], "ends": [-1], "starts": [0]}
    inputs_map = {"data": indices_shape.output[0], **attr}
    indices_outter_shape = GraphBuilder(ctx).make_slice(inputs_map, dtypes=[TensorProto.INT64])
    output_shape_ = ctx.make_node("Concat",
                                  [indices_outter_shape, output_inner_shape],
                                  attr={"axis": 0},
                                  dtypes=[TensorProto.INT64])
    attr = {"axes": [0], "ends": [-1], "starts": [0]}
    inputs_map = {"data": output_shape_.output[0], **attr}
    output_shape = GraphBuilder(ctx).make_slice(inputs_map, dtypes=[TensorProto.INT64])
    ctx.make_node("Reshape",
                  [gathernd_loop.output[1], output_shape],
                  outputs=[output],
                  shapes=shapes,
                  dtypes=dtypes)


@tf_op("GatherNd", onnx_op="GatherND")
class GatherND:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # Tparams output = GatherNd(Tparams params, Tidx indices)
        params = node.input[0]
        indices = node.input[1]
        output = node.output[0]
        # same as the attr Tparams
        t_params = ctx.get_dtype(params)
        utils.make_sure(t_params, "Dtype of {} is None".format(indices))
        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        make_gathernd(ctx, params, indices, output, node.name, t_params, shapes, dtypes)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # indicies input
        input1 = node.input[1]
        target_dtype = TensorProto.INT64
        if ctx.get_dtype(input1) != TensorProto.INT64:
            inp_cast = ctx.insert_new_node_on_input(node, "Cast", input1, to=target_dtype)
            ctx.copy_shape(input1, inp_cast.output[0])
            ctx.set_dtype(inp_cast.output[0], target_dtype)


@tf_op("ScatterNd", onnx_op="ScatterND")
class ScatterND:
    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        onnxdtype = ctx.get_dtype(node.input[1])
        zero_tensor = helper.make_tensor("value", onnxdtype, dims=[1], vals=[0])
        const_of_shape = ctx.make_node("ConstantOfShape", [node.input[2]], attr={'value': zero_tensor},
                                       shapes=node.output_shapes, dtypes=[onnxdtype])
        ctx.replace_input(node, node.input[2], const_of_shape.output[0], 2)
        ctx.insert_new_node_on_input(const_of_shape, "Cast", const_of_shape.input[0], to=TensorProto.INT64)
        ctx.insert_new_node_on_input(node, "Cast", node.input[0], to=TensorProto.INT64)
        # reorder inputs to match onnx
        ctx.replace_inputs(node, [node.input[2], node.input[0], node.input[1]])


@tf_op("TensorScatterAdd", onnx_op="ScatterND")
class TensorScatterAdd:
    @classmethod
    def version_16(cls, ctx, node, **kwargs):
        # indicies input must be int64 in ONNX.
        if ctx.get_dtype(node.input[1]) != TensorProto.INT64:
            ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=TensorProto.INT64)
        node.set_attr("reduction", 'add')


@tf_op("TensorScatterMax", onnx_op="ScatterND")
class TensorScatterMax:
    @classmethod
    def version_16(cls, ctx, node, **kwargs):
        # indices input must be int64 in ONNX.
        if ctx.get_dtype(node.input[1]) != TensorProto.INT64:
            ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=TensorProto.INT64)
        node.set_attr("reduction", 'max')


@tf_op("TensorScatterMin", onnx_op="ScatterND")
class TensorScatterMin:
    @classmethod
    def version_16(cls, ctx, node, **kwargs):
        # indices input must be int64 in ONNX.
        if ctx.get_dtype(node.input[1]) != TensorProto.INT64:
            ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=TensorProto.INT64)
        node.set_attr("reduction", 'min')


@tf_op("TensorScatterSub", onnx_op="ScatterND")
class TensorScatterSub:
    @classmethod
    def version_16(cls, ctx, node, **kwargs):
        # indices input must be int64 in ONNX.
        if ctx.get_dtype(node.input[1]) != TensorProto.INT64:
            ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=TensorProto.INT64)
        ctx.insert_new_node_on_input(node, "Neg", node.input[2])
        node.set_attr("reduction", 'add')


@tf_op("TensorScatterUpdate", onnx_op="ScatterND")
class TensorScatterUpdate:
    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        if ctx.get_dtype(node.input[1]) != TensorProto.INT64:
            ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=TensorProto.INT64)


@tf_op("Split")
class Split:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T output = Split(int32 split_dim, T value, @int num_split)
        # T outputs = Split(T input, @INT axis, @INTS split)
        split_dims = node.inputs[0].get_tensor_value()
        ctx.remove_input(node, node.input[0], 0)
        node.set_attr("num_outputs", node.get_attr_int("num_split"))
        node.set_attr("axis", split_dims)

    @classmethod
    def version_2(cls, ctx, node, **kwargs):
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # no change
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Default axis is not -1 but doesn't matter since we always set it.
        cls.version_1(ctx, node, **kwargs)


@tf_op("SplitV")
class SplitV:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T output = SplitV(T value, Tlen size_splits, int32 split_dim, @int num_split, @type Tlen)
        # T outputs = Split(T input, @INT axis, @INTS split)
        node.type = "Split"
        split = node.inputs[1].get_tensor_value()
        split_dims = node.inputs[2].get_tensor_value()
        if -1 in split:
            # negative split = use the remaining size
            shape = ctx.get_shape(node.input[0])
            final_sum = shape[split_dims]
            sums = sum([i for i in split if i >= 0])
            for i, v in enumerate(split):
                if v == -1:
                    split[i] = final_sum - sums
        ctx.remove_input(node, node.input[2], 2)
        ctx.remove_input(node, node.input[1], 1)
        node.set_attr("split", split)
        node.set_attr("axis", split_dims)

    @classmethod
    def version_2(cls, ctx, node, **kwargs):
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Split now supports dynamic split lengths
        if node.inputs[1].is_const():
            # Call version 1 to deal with -1 cases
            cls.version_1(ctx, node, **kwargs)
            # Convert attr to input
            split_val = node.get_attr_value("split")
            split_const = ctx.make_const(utils.make_name("split"), np.array(split_val, np.int64))
            ctx.replace_inputs(node, [node.input[0], split_const.output[0]])
            del node.attr["split"]
        else:
            # Technically incorrect if any of the splits are -1
            node.type = "Split"
            split_dims = node.inputs[2].get_tensor_value()
            ctx.remove_input(node, node.input[2], 2)
            node.set_attr("axis", split_dims)
            if ctx.get_dtype(node.input[1]) != TensorProto.INT64:
                ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=TensorProto.INT64)


@tf_op("ExpandDims")
class ExpandDims:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        shape = ctx.get_shape(node.output[0])
        dim_node = node.inputs[1]

        utils.make_sure(dim_node.is_const(), "ExpandDims with non-const axes requires opset 13")
        node.type = "Unsqueeze"
        # tf.expanddims() wants a scalar per doc but quietly accepts any single-element tensor
        axis = dim_node.get_tensor_value(as_list=False).flatten()[0]

        if axis < 0 and ctx.opset < 11:
            utils.make_sure(shape is not None, "ExpandDims with negative axes and unknown rank requires opset >= 11")
            out_rank = len(shape)
            axis += out_rank
        node.set_attr("axes", [axis])
        ctx.remove_input(node, node.input[1], 1)

    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Parameters moved to inputs for operator Squeeze, Unsqueeze.
        if ctx.get_dtype(node.input[1]) != onnx_pb.TensorProto.INT64:
            ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=onnx_pb.TensorProto.INT64)
        if ctx.get_shape(node.input[1]) != [1]:
            const_newshape = ctx.make_const(utils.make_name("reshape_const"), np.array([1], dtype=np.int64))
            reshape_node = ctx.make_node("Reshape", [node.input[1], const_newshape.output[0]])
            ctx.replace_inputs(node, [node.input[0], reshape_node.output[0]])
        node.type = "Unsqueeze"


@tf_op("StridedSlice")
class StridedSlice:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # for now we implement common cases. Things like strides!=1 are not mappable to onnx.
        not_supported_attr = ["new_axis_mask"]
        for attr_name in not_supported_attr:
            attr = node.get_attr(attr_name)
            if attr is not None and attr.i != 0:
                raise ValueError("StridedSlice: attribute " + attr_name + " not supported")

        onnx_dtype = ctx.get_dtype(node.input[1])
        np_dtype = utils.ONNX_TO_NUMPY_DTYPE[onnx_dtype]
        max_size = np.iinfo(np_dtype).max
        begin = node.inputs[1].get_tensor_value()
        end = node.inputs[2].get_tensor_value()
        strides = node.inputs[3].get_tensor_value()
        end_mask = node.get_attr("end_mask")
        end_mask = end_mask.i if end_mask is not None else 0
        begin_mask = node.get_attr("begin_mask")
        begin_mask = begin_mask.i if begin_mask is not None else 0
        shrink_axis_mask = node.get_attr("shrink_axis_mask")
        shrink_axis_mask = shrink_axis_mask.i if shrink_axis_mask is not None else 0
        ellipsis_mask = node.get_attr("ellipsis_mask")
        ellipsis_mask = ellipsis_mask.i if ellipsis_mask is not None else 0
        new_begin = []
        new_end = []
        axes = []
        # onnx slice op can't remove a axis, track axis and add a squeeze op if needed
        needs_squeeze = []
        # ellipsis: one bit at most can be 1. An ellipsis implicitly creates as many range specifications as
        # necessary to fully specify the sliced range for every dimension.
        # For example for a 4-dimensional tensor foo the slice foo[2, ..., 5:8] implies foo[2, :, :, 5:8]
        # NOTE: we ignore those axes denoted by ellipsis using `axes` attribute
        ellipsis_gap = 0
        for idx, begin_item in enumerate(begin):
            if strides[idx] != 1:
                raise ValueError("StridedSlice: only strides=1 is supported")
            if (ellipsis_mask >> idx) & 1:
                input_shape = ctx.get_shape(node.input[0])
                utils.make_sure(
                    input_shape is not None,
                    "StridedSlice op {} requires the shape of input".format(node.name)
                )
                ellipsis_gap = len(input_shape) - len(begin)
                continue

            # ignore ellipsis axes
            axes.append(idx + ellipsis_gap)
            end_item = end[idx]

            # an implicit condition is stride == 1 (checked in above)
            if begin_item < 0 and end_item == 0:
                end_item = max_size

            mask = (shrink_axis_mask >> idx) & 1
            if mask != 0:
                new_begin.append(begin_item)
                end_item = begin_item + 1 if begin_item != -1 else max_size
                new_end.append(end_item)
                needs_squeeze.append(idx + ellipsis_gap)
                continue

            mask = (begin_mask >> idx) & 1
            if mask != 0:
                new_begin.append(0)
            else:
                new_begin.append(begin_item)

            mask = (end_mask >> idx) & 1
            if mask != 0:
                new_end.append(max_size)
            else:
                new_end.append(end_item)

        out_dtypes = [ctx.get_dtype(node.output[0])]
        out_shapes = [ctx.get_shape(node.output[0])]
        ctx.remove_node(node.name)

        attr = {"starts": new_begin, "ends": new_end, "axes": axes}
        inputs_map = {"data": node.input[0], **attr}
        kwargs = {**inputs_map, "outputs": node.output}
        if len(axes) > 0:
            node = GraphBuilder(ctx).make_slice(
                kwargs, name=node.name, dtypes=out_dtypes, shapes=out_shapes, return_node=True)
        else:
            node = ctx.make_node("Identity", [node.input[0]], name=node.name, outputs=node.output,
                                 dtypes=out_dtypes, shapes=out_shapes)
        nodes = [node]
        if needs_squeeze:
            # insert_new_node_on_output(self, op_type, output_name=None, name=None, inputs=None, domain=None, **kwargs)
            # ctx.insert_new_node_on_output("Squeeze", node.output[0], name)
            name = utils.make_name(node.name)
            shape = ctx.get_shape(node.output[0])
            dtype = ctx.get_dtype(node.output[0])
            squeeze_node = GraphBuilder(ctx).make_squeeze(
                {"axes": needs_squeeze, 'data': node.output[0]}, name=name,
                dtypes=[dtype], shapes=[shape], return_node=True)
            ctx.insert_node_on_output(squeeze_node)

            nodes.append(squeeze_node)
            ctx.update_node_shape_dtype(node, override=True)

        # onnx slice as of opset 7 does only take float tensors ... cast if needed
        input_dtype = ctx.get_dtype(node.input[0])
        if ctx.opset < 9:
            if input_dtype != onnx_pb.TensorProto.FLOAT:
                if node.inputs[0].type == "Cast" and len(ctx.find_output_consumers(node.inputs[0].output[0])) == 1:
                    # override the previous cast
                    cast_node = node.inputs[0]
                    cast_node.set_attr("to", onnx_pb.TensorProto.FLOAT)
                else:
                    cast_node = ctx.insert_new_node_on_input(node, "Cast", node.input[0],
                                                             to=onnx_pb.TensorProto.FLOAT)
                    nodes.insert(0, cast_node)
                ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.FLOAT)
                ctx.copy_shape(node.input[0], cast_node.output[0])
                # undo the cast afer slice
                name = utils.make_name(node.name)
                cast_node = ctx.insert_new_node_on_output("Cast", nodes[-1].output[0], name,
                                                          to=input_dtype)
                ctx.set_dtype(cast_node.output[0], input_dtype)
                ctx.copy_shape(node.output[0], cast_node.output[0])
                nodes.append(cast_node)
            ctx.set_dtype(node.output[0], onnx_pb.TensorProto.FLOAT)

    @classmethod
    def any_version_after10(cls, opset, ctx, node, **kwargs):
        # T output = Slice(T input, Index begin, Index end, Index strides
        #                 @int begin_mask, @int end_mask, @int ellipsis_mask
        #                 @int shrink_axis_mask, @int new_axis_mask)
        # T output = Slice(T input, Tind starts, Tind ends, Tind axes, Tind steps)
        # "ends" are exclusive, "axes" and "steps" are optional, their default val are [0, ...] and 1
        input_x = node.input[0]
        begin = node.input[1]
        end = node.input[2]
        strides = node.input[3]
        new_axis_mask = node.get_attr("new_axis_mask")
        new_axis_mask = new_axis_mask.i if new_axis_mask is not None else 0

        if ctx.is_const(begin) and ctx.is_const(end) and ctx.is_const(strides) \
                and all(val == 1 for val in ctx.get_tensor_value(strides)) \
                and new_axis_mask == 0:
            cls.version_1(ctx, node, **kwargs)
            return

        onnx_dtype = ctx.get_dtype(node.input[1])
        np_dtype = utils.ONNX_TO_NUMPY_DTYPE[onnx_dtype]

        # NOTE: Max op only supports float32, deal with overflow when cast back to int32
        # enable it after Max supports int32 and int64
        # max_size = utils.get_max_value(np_dtype)
        # min_size = utils.get_min_value(np_dtype)
        max_size = 1e9
        min_size = -1e9

        end_mask = node.get_attr("end_mask")
        end_mask = end_mask.i if end_mask is not None else 0
        begin_mask = node.get_attr("begin_mask")
        begin_mask = begin_mask.i if begin_mask is not None else 0
        ellipsis_mask = node.get_attr("ellipsis_mask")
        ellipsis_mask = ellipsis_mask.i if ellipsis_mask is not None else 0
        shrink_axis_mask = node.get_attr("shrink_axis_mask")
        shrink_axis_mask = shrink_axis_mask.i if shrink_axis_mask is not None else 0

        param_shape = ctx.get_shape(node.input[1]) or \
                      ctx.get_shape(node.input[2]) or \
                      ctx.get_shape(node.input[3])
        utils.make_sure(
            param_shape is not None,
            "StridedSlice op {} requires the shape of begin/end/strides".format(node.name)
        )
        param_rank = param_shape[0]

        if new_axis_mask != 0:
            unqueeze_at = []
            ellipsis_gap = 0
            num_new = 0
            for bit in range(32):
                if (new_axis_mask >> bit) & 1 == 1:
                    num_new += 1
                if (ellipsis_mask >> bit) & 1:
                    input_shape = ctx.get_shape(input_x)
                    # calculate what rank for ellipsis: input rank - (being rank - all new_axis - 1)
                    ellipsis_gap = len(input_shape) - param_rank + num_new + 1
                if (new_axis_mask >> bit) & 1 == 1:
                    unqueeze_at.append(bit + ellipsis_gap)
                    begin_mask |= 1 << bit
                    end_mask |= 1 << bit

            if ellipsis_mask:
                unqueeze_at = []
                ellipsis_gap = 0
                num_new = 0
                end_mask = node.get_attr("end_mask")
                end_mask = end_mask.i if end_mask is not None else 0
                begin_mask = node.get_attr("begin_mask")
                begin_mask = begin_mask.i if begin_mask is not None else 0

                for bit in range(32):
                    new_axis_flag = (new_axis_mask >> bit) & 1
                    ellipsis_flag = (ellipsis_mask >> bit) & 1
                    num_new += not ellipsis_flag and new_axis_flag

                for bit in range(32):
                    if (ellipsis_mask >> bit) & 1:
                        ellipsis_gap = len(ctx.get_shape(input_x)) - param_rank + num_new + 1
                    elif (new_axis_mask >> bit) & 1:
                        effective_bit = bit if not ellipsis_gap else bit + ellipsis_gap - 1
                        unqueeze_at.append(effective_bit)
                        begin_mask |= 1 << bit
                        end_mask |= 1 << bit

            input_x = GraphBuilder(ctx).make_unsqueeze(
                {'data': input_x, 'axes': unqueeze_at})


        # use in onnx graph to mask begin
        new_begin_mask = [1] * param_rank
        # use in onnx graph to mask end
        new_end_mask = [min_size] * param_rank
        # for shrink mask, if shrink mask is 1, set stride to be max_size
        shrink_strided_mask = [min_size] * param_rank
        axes = []
        # onnx slice op can't remove a axis, track axis and add a squeeze op if needed
        needs_squeeze = []
        ellipsis_gap = 0
        for idx in range(param_rank):
            if (ellipsis_mask >> idx) & 1:
                input_shape = ctx.get_shape(input_x)
                utils.make_sure(
                    input_shape is not None,
                    "StridedSlice op {} requires the shape of input".format(node.name)
                )
                ellipsis_gap = len(input_shape) - param_rank
                # handle the redundant param
                new_begin_mask[idx] = 0
                new_end_mask[idx] = max_size
                axes.append(idx)
                continue

            # ignore ellipsis axes
            axes.append(idx + ellipsis_gap)

            mask = (shrink_axis_mask >> idx) & 1
            if mask != 0:
                shrink_strided_mask[idx] = max_size
                new_end_mask[idx] = max_size
                needs_squeeze.append(idx + ellipsis_gap)
                continue

            mask = (begin_mask >> idx) & 1
            if mask != 0:
                new_begin_mask[idx] = 0

            mask = (end_mask >> idx) & 1
            if mask != 0:
                new_end_mask[idx] = max_size

        out_dtypes = [ctx.get_dtype(node.output[0])]
        out_shapes = [ctx.get_shape(node.output[0])]
        ctx.remove_node(node.name)

        # mask begin
        new_begin_mask = np.array(new_begin_mask, dtype=np_dtype)
        if not np.all(new_begin_mask == 1):
            if ctx.is_const(begin) and ctx.is_const(strides):
                new_begin_vals = np.copy(ctx.get_tensor_value(begin, as_list=False))
                strides_vals = ctx.get_tensor_value(strides, as_list=False)
                idx1 = np.where(new_begin_mask == 0)
                idx2 = np.where(strides_vals < 0)
                idx3 = np.intersect1d(idx1, idx2)
                new_begin_vals[idx3] = max_size
                begin = ctx.make_const(utils.make_name("begin_masked"), new_begin_vals).output[0]
            else:
                begin_mask_const = ctx.make_const(utils.make_name("begin_mask"), np.equal(new_begin_mask, 0))
                zero_const = ctx.make_const(utils.make_name("zero_const"), np.zeros(1, dtype=np_dtype))
                max_const = ctx.make_const(utils.make_name("max_const"), np.array(max_size, dtype=np_dtype))
                op1 = ctx.make_node("Less", [strides, zero_const.output[0]], op_name_scope=node.name)
                op2 = ctx.make_node("And", [op1.output[0], begin_mask_const.output[0]], op_name_scope=node.name)
                begin = ctx.make_node("Where", [op2.output[0], max_const.output[0], begin],
                                      op_name_scope=node.name).output[0]

        # mask end
        new_end_mask = np.array(new_end_mask, dtype=np_dtype)
        if not np.all(new_end_mask == min_size):
            if ctx.is_const(end) and ctx.is_const(strides):
                new_end_mask = np.maximum(ctx.get_tensor_value(end, as_list=False), new_end_mask)
                idx = np.where(new_end_mask == max_size)
                sign = np.sign(ctx.get_tensor_value(strides, as_list=False))[idx]
                new_end_mask[idx] = new_end_mask[idx] * sign
                end = ctx.make_const(utils.make_name("end_masked"), new_end_mask).output[0]
            else:
                # Overlay new_end_mask with specified end values.
                # Adjust max_size to min_size if steps are < 0
                max_const = ctx.make_const(utils.make_name("max_const"), np.array(max_size, dtype=np_dtype))
                min_const = ctx.make_const(utils.make_name("min_const"), np.array(min_size, dtype=np_dtype))
                zero_const = ctx.make_const(utils.make_name("zero_const"), np.zeros(1, dtype=np_dtype))
                end_mask_const = ctx.make_const(utils.make_name("end_mask"), np.array(new_end_mask, dtype=np_dtype))
                outputname = utils.make_name("{}__newendmask".format(node.name))
                new_end_mask = math.make_min_or_max_op(ctx, "Max", [end, end_mask_const.output[0]],
                                                       [outputname])
                op1 = ctx.make_node("Less", [strides, zero_const.output[0]], op_name_scope=node.name)
                op2 = ctx.make_node("Equal", [new_end_mask.output[0], max_const.output[0]], op_name_scope=node.name)
                op3 = ctx.make_node("And", [op2.output[0], op1.output[0]], op_name_scope=node.name)
                end = ctx.make_node("Where", [op3.output[0], min_const.output[0], new_end_mask.output[0]],
                                    op_name_scope=node.name).output[0]

        # mask strides for shrink
        shrink_strided_mask = np.array(shrink_strided_mask, dtype=np_dtype)
        if not np.all(shrink_strided_mask == min_size):
            if ctx.is_const(strides):
                strides = ctx.make_const(
                    utils.make_name("strides_masked"),
                    np.maximum(ctx.get_tensor_value(strides, as_list=False), shrink_strided_mask)
                ).output[0]
            else:
                shrink_strided_mask_const = ctx.make_const(
                    utils.make_name("strides_mask"),
                    np.array(shrink_strided_mask, dtype=np_dtype)
                )
                strides_output = utils.make_name("{}__strides".format(node.name))
                math.make_min_or_max_op(
                    ctx, "Max",
                    [strides, shrink_strided_mask_const.output[0]],
                    [strides_output]
                )
                strides = strides_output
        # create axes input
        axes_const = ctx.make_const(
            utils.make_name("slice_axes"),
            np.array(axes, dtype=np_dtype)
        )
        axes_output = axes_const.output[0]

        inputs_map = {
            "data": input_x,
            "starts": begin,
            "ends": end,
            "steps": strides,
            "axes": axes_output
        }
        kwargs = {**inputs_map, "outputs": node.output}
        node = GraphBuilder(ctx).make_slice(kwargs, name=node.name, dtypes=out_dtypes, shapes=out_shapes)
        node = ctx.get_node_by_output(node)
        if needs_squeeze:
            squeeze_node = GraphBuilder(ctx).make_squeeze(
                {"axes": needs_squeeze, "data": node.output[0]}, name=node.child_name(), return_node=True)
            ctx.insert_node_on_output(squeeze_node, node.output[0])
            input_dtype = ctx.get_dtype(node.output[0])
            ctx.set_dtype(squeeze_node.output[0], input_dtype)
            ctx.copy_shape(node.output[0], squeeze_node.output[0])

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        cls.any_version_after10(10, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Parameters moved to inputs for operator Squeeze, Unsqueeze.
        cls.any_version_after10(13, ctx, node, **kwargs)


@tf_op("Cast")
class Cast:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # DstT y = Cast(SrcT x, @type SrcT, @type DstT)
        # T2 output = Cast(T1 input, @STRING to)
        dst = node.get_attr("to")
        dst = utils.ONNX_DTYPE_NAMES[dst]
        node.set_attr("to", dst)

    @classmethod
    def version_6(cls, ctx, node, **kwargs):
        pass

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        pass


@tf_op("TopKV2", onnx_op="TopK")
class TopKV2:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T values, int32 indices = TopKV2(T input, int32 k, @bool sorted=true, @realnumbertype T)
        # T values, I indices = TopK(T x, @int axis=-1, @int k). I: int64
        topk_node_name = node.name
        topk_output1 = node.output[0]
        topk_output2 = node.output[1]

        shapes = node.output_shapes
        dtypes = node.output_dtypes
        k = node.inputs[1].get_tensor_value()
        ctx.remove_node(topk_node_name)
        new_topk_name = utils.make_name(topk_node_name)
        new_topk_node = ctx.make_node("TopK", [node.input[0]],
                                      outputs=[topk_output1, utils.port_name(new_topk_name, 1)],
                                      name=new_topk_name, attr={"k": k},
                                      shapes=shapes, dtypes=[dtypes[0], onnx_pb.TensorProto.INT64])

        if dtypes[0] != onnx_pb.TensorProto.FLOAT:
            # opset-1 only supports float dtypes
            ctx.insert_new_node_on_output("Cast", new_topk_node.input[0], to=onnx_pb.TensorProto.FLOAT)
            ctx.insert_new_node_on_output("Cast", new_topk_node.output[0], to=dtypes[0])
        new_cast_name = utils.make_name(topk_node_name)
        ctx.make_node("Cast", [new_topk_node.output[1]], outputs=[topk_output2],
                      name=new_cast_name, attr={"to": onnx_pb.TensorProto.INT32},
                      shapes=[shapes[1]], dtypes=[onnx_pb.TensorProto.INT32])

    @classmethod
    def any_version_after10(cls, opset, ctx, node, **kwargs):
        # onnx only supports input K as a 1D tesor with dtype int64
        # while in tf, K is a 0D tensor with dtype int32
        dtypes = node.output_dtypes
        k_0d = node.input[1]
        cast = ctx.make_node("Cast", [k_0d], attr={"to": onnx_pb.TensorProto.INT64})
        k_1d = GraphBuilder(ctx).make_unsqueeze({'data': cast.output[0], "axes": [0]}, return_node=True)
        ctx.replace_input(node, k_0d, k_1d.output[0], 1)
        # cast X if needed
        if dtypes[0] != onnx_pb.TensorProto.FLOAT:
            # opset-10 supports types other than float but onnxruntime does not
            ctx.insert_new_node_on_input(node, "Cast", node.input[0], input_index=0, to=onnx_pb.TensorProto.FLOAT)
            ctx.insert_new_node_on_output("Cast", node.output[0], to=dtypes[0])
        # cast the index output to int32
        cast_out = ctx.insert_new_node_on_output("Cast", node.output[1], name=utils.make_name(node.name), to=dtypes[1])
        ctx.set_dtype(cast_out.output[0], dtypes[1])
        ctx.copy_shape(node.output[1], cast_out.output[0])
        ctx.set_dtype(node.output[1], onnx_pb.TensorProto.INT64)

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        cls.any_version_after10(10, ctx, node, **kwargs)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # opset 11 supports negative axis, and new attrs 'largest' and 'sorted'
        # the core logic doesn't change, using defaults for new attrs
        cls.any_version_after10(11, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Parameters moved to inputs for operator Squeeze, Unsqueeze.
        cls.any_version_after10(13, ctx, node, **kwargs)


@tf_op("Tile")
class Tile:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # onnx wants shape input to be int64
        _convert_shapenode_to_int64(ctx, node, 1)


@tf_op("Pack")
class Pack:
    @classmethod
    def any_version(cls, opset, ctx, node, **kwargs):
        # hack to make up for the missing onnx pack op
        axis = node.get_attr("axis").i
        if axis < 0:
            axis += len(ctx.get_shape(node.input[0])) + 1

        inputs = []
        dtype = None
        gb = GraphBuilder(ctx)
        # insert Unsqueeze on each input
        for i, n in enumerate(node.inputs):
            dtype = ctx.get_dtype(node.input[i])
            shape = ctx.get_shape(node.input[i])
            if shape is not None:
                shape = shape.copy()
                shape.insert(axis, 1)
            new_node = gb.make_unsqueeze(
                {'data': node.input[i], 'axes': [axis]},
                op_name_scope=node.name, shapes=[shape], dtypes=[dtype], return_node=True)
            output_name = new_node.output[0]
            ctx.replace_input(node, node.input[i], output_name, i)
            inputs.append(output_name)

        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        # concat all unqueezes
        concat = ctx.make_node("Concat", inputs, op_name_scope=node.name, attr={"axis": axis},
                               shapes=shapes, dtypes=dtypes)
        ctx.replace_all_inputs(node.output[0], concat.output[0])  # ops=ctx.get_nodes()

    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        cls.any_version(1, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Parameters moved to inputs for operator Squeeze, Unsqueeze.
        cls.any_version(13, ctx, node, **kwargs)


@tf_op("Unpack")
class Unpack:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # hack to make up for the missing onnx unpack op
        # squeeze does not support negative axis
        axis = node.get_attr("axis").i
        if axis < 0:
            shape = ctx.get_shape(node.input[0])
            utils.make_sure(shape is not None, "shape of unpack input is None: {}".format(node.input[0]))
            axis += len(shape)
        num_outputs = node.get_attr_value("num")
        # split the tensor into n outputs
        node.type = "Split"
        node.set_attr("num_outputs", num_outputs)
        node.set_attr("axis", axis)

        # for each output we need to squeeze axis
        for n in node.output:
            op_name = utils.make_name(node.name)
            shape = ctx.get_shape(n)
            dtype = ctx.get_dtype(n)
            squeeze_node = GraphBuilder(ctx).make_squeeze(
                {'data': n, 'axes': [axis]}, name=op_name, return_node=True, shapes=[shape], dtypes=[dtype])
            ctx.insert_node_on_output(squeeze_node, n)

        # split node is 1 rank higher than squeeze nodes
        for out in node.output:
            output_shape = ctx.get_shape(out)
            if output_shape:
                split_shape = output_shape[:axis] + [1] + output_shape[axis:]
                ctx.set_shape(out, split_shape)


@tf_op("OneHot")
class OneHot:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # until there is no onehot op in onnx, a workaround using gather from eye
        indices_name = node.input[0]
        indices_shape = ctx.get_shape(indices_name)
        if len(indices_shape) != 1:
            # TODO: this works for rank=1 but tensorflow supports more than this.
            # Same principle should work but we need to implemtn our own eye.
            raise ValueError("onehot op: only rank1 is supported")
        axis = node.get_attr("axis")
        # axis becomes axis for gather
        node.set_attr("axis", 0)
        depth = node.inputs[1].get_tensor_value()
        on_val = node.inputs[2].get_tensor_value(as_list=False)
        on = on_val.tolist()
        off = node.inputs[3].get_tensor_value()
        eye = np.eye(depth, dtype=on_val.dtype)
        if on != 0:
            eye[eye == 1] = on
            eye[eye == 0] = off
        else:
            eye[eye == 0] = off
            eye[eye == 1] = on

        const_name = utils.make_name(node.name)
        ctx.make_const(const_name, eye)
        # setup gather inputs
        ctx.replace_inputs(node, [const_name, indices_name])
        node.type = "Gather"
        if axis.i == 0:
            # TODO: revisit for rank > 1
            name = utils.make_name(node.name)
            shape = ctx.get_shape(node.output[0])
            transpose_node = ctx.make_node("Transpose", [node.output[0]], name=name, shapes=[shape])
            ctx.insert_node_on_output(transpose_node, node.output[0])
            if shape is not None:
                ctx.set_shape(node.output[0], shape[::-1])

    @classmethod
    def any_version_after9(cls, opset, ctx, node, **kwargs):
        # T output = OneHot(uint8/int32/int64 input, T depth, T on-value, T off-value, @int axis, @dtype)
        # tf requires that dtype is same as on-value's and off-value's dtype
        # in ONNX, op's schema is (input, depth, value, @int axis), meaning of "value" is [off-value, on-value]
        # onnxruntime only supports int64
        output_dtype = ctx.get_dtype(node.input[2])
        supported_dtypes = [onnx_pb.TensorProto.FLOAT]
        if ctx.is_target(constants.TARGET_RS6):
            supported_dtypes = [onnx_pb.TensorProto.INT64, onnx_pb.TensorProto.INT32]
        if output_dtype not in supported_dtypes:
            logger.warning("unsupported dtype in target runtime, OneHot op can't be used directly")
            cls.version_1(ctx, node, **kwargs)
            return

        depth = GraphBuilder(ctx).make_unsqueeze({'data': node.input[1], 'axes': [0]})
        on_value = node.input[2]
        off_value = node.input[3]
        on_value = GraphBuilder(ctx).make_unsqueeze({'data': on_value, 'axes': [0]})
        off_value = GraphBuilder(ctx).make_unsqueeze({'data': off_value, 'axes': [0]})
        off_on_value = ctx.make_node("Concat", [off_value, on_value], attr={"axis": 0}).output[0]

        indices = node.input[0]
        indices_rank = ctx.get_rank(indices)

        # Add a special support for 0-rank indices, to do so we have to expand the dimension to 1
        # before the one hot encoding and remove it after.
        if indices_rank == 0:
            dims = ctx.make_const(name=utils.make_name('dims'), np_val=np.array([1], dtype=np.int64))
            indices = ctx.make_node("Expand", [indices, dims.name]).output[0]

            # Axis 0 is supported by TensorFlow for the one-hot encoding of a 0-rank tensor. It should behave
            # as if axis has been set to -1 so we artificially set it as is here.
            if node.get_attr('axis').i == 0:
                node.set_attr('axis', -1)

        if ctx.is_target(constants.TARGET_RS6) \
                and ctx.get_dtype(indices) != onnx_pb.TensorProto.INT64:
            indices = ctx.make_node("Cast", [indices], attr={"to": onnx_pb.TensorProto.INT64}).output[0]
        ctx.replace_input(node, node.input[0], indices, 0)

        if ctx.is_target(constants.TARGET_RS6) \
                and ctx.get_dtype(depth) != onnx_pb.TensorProto.INT64:
            depth = ctx.make_node("Cast", [depth], attr={"to": onnx_pb.TensorProto.INT64}).output[0]
        ctx.replace_input(node, node.input[1], depth, 1)

        if ctx.is_target(constants.TARGET_RS6) \
                and output_dtype != onnx_pb.TensorProto.INT64:
            off_on_value = ctx.make_node("Cast", [off_on_value], attr={"to": onnx_pb.TensorProto.INT64}).output[0]
        ctx.replace_input(node, node.input[2], off_on_value, 2)
        ctx.remove_input(node, node.input[3], 3)

        if ctx.is_target(constants.TARGET_RS6) \
                and output_dtype != onnx_pb.TensorProto.INT64:
            new_node_name = utils.make_name("onehot_output")
            new_node = ctx.insert_new_node_on_output("Cast", node.output[0], new_node_name, to=output_dtype)
            ctx.set_dtype(new_node.output[0], output_dtype)
            ctx.set_shape(new_node.output[0], ctx.get_shape(node.output[0]))

        # Remove the dimension artificially added in order to support 0-rank indices
        if indices_rank == 0:
            nodes = [node]
            name = utils.make_name(node.name)
            shape = ctx.get_shape(node.output[0])
            dtype = ctx.get_dtype(node.output[0])
            squeeze_node = GraphBuilder(ctx).make_squeeze(
                {
                    "axes": [0],
                    'data': node.output[0]
                },
                name=name,
                dtypes=[dtype],
                shapes=[shape],
                return_node=True)
            ctx.insert_node_on_output(squeeze_node)

            nodes.append(squeeze_node)
            ctx.update_node_shape_dtype(node, override=True)

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        cls.any_version_after9(9, ctx, node, **kwargs)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.any_version_after9(11, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Parameters moved to inputs for operator Squeeze, Unsqueeze.
        cls.any_version_after9(13, ctx, node, **kwargs)


@tf_op("Shape")
class Shape:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # out_type output = Shape(T input, @int32|int64 out_type), out_type by default int32
        # int64 output = Shape(T input)
        dtype = ctx.get_dtype(node.output[0])
        if dtype == onnx_pb.TensorProto.INT64:
            return
        op_name = utils.make_name(node.name)
        output_cast = ctx.insert_new_node_on_output("Cast", node.output[0], name=op_name, to=dtype)
        ctx.set_dtype(output_cast.output[0], dtype)
        ctx.set_dtype(node.output[0], onnx_pb.TensorProto.INT64)
        ctx.copy_shape(node.output[0], output_cast.output[0])


@tf_op("IsNan", onnx_op="IsNaN")
class IsNan:
    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        pass


@tf_op("BatchToSpaceND", onnx_op="DepthToSpace")
class BatchToSpace:
    @classmethod
    def any_version(cls, opset, ctx, node, **kwargs):
        # block_shape impacts Transpose 'perm' attribute values.
        # must be available at compile time
        utils.make_sure(node.inputs[1].is_const(), 'only support constant block_shape value.')

        block_shape = node.inputs[1].get_tensor_value(False)
        blocklen = len(block_shape)
        xlen = len(ctx.get_shape(node.input[0]))

        # if 3d or 4d tensor & square 2d block_shape , can optimize
        cond1 = xlen in [3, 4]
        cond2 = blocklen == 2 and block_shape[0] == block_shape[1]
        if cond1 and cond2:
            # https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/batch-to-space-n-d.html
            # the above link says the data format of input tensor should be (batch, spatial_shape, remaining_shape)
            # and we only support 3D and 4D here, and the data format is NHC and NHWC
            # onnx op "DepthToSpace" does the same work on input tensor except that it works on "C",
            # and it only supports NCHW
            # T out = BatchToSpaceND(T input, int32 block_shape, int32 crops)
            input_tensor = node.inputs[0]
            input_shape = ctx.get_shape(input_tensor.output[0])

            if len(input_shape) == 3:
                # insert automatically an Unsqueeze op if the input is 3d
                unsqz1 = GraphBuilder(ctx).make_unsqueeze(
                    {"axes": [3], "data": input_tensor.output[0]}, return_node=True)
                # NHWC TO CNHW, so onnx op will work on "N" which is the same as tensorflow
                trans1 = ctx.make_node("Transpose", unsqz1.output, {"perm": [3, 0, 1, 2]})
            else:
                # Add explicit NHWC_TO_NCHW transpose before and NCHW_TO_NHWC transpose after subgraph.
                # That enables more optimizations in TransposeOptimizer.
                trans_nchw = ctx.make_node("Transpose", input_tensor.output, {"perm": NHWC_TO_NCHW})
                # NCHW TO CNHW
                trans1 = ctx.make_node("Transpose", trans_nchw.output, {"perm": [1, 0, 2, 3]})
            reorganize_node = ctx.make_node(node.type, trans1.output, attr={"blocksize": block_shape[0]})

            # implement crop logic, the data format is NCHW
            slice_axis = [2, 3]
            if node.inputs[2].is_const():
                crops = node.inputs[2].get_tensor_value()
                top, bottom = crops[0]
                left, right = crops[1]
                starts = [top, left]
                ends = []
                for end in [bottom, right]:
                    if end != 0:
                        ends.append(-end)
                    else:
                        ends.append(np.iinfo(np.int32).max)
                attr = {"axes": slice_axis, "ends": ends, "starts": starts}
            else:
                shape = ctx.make_const(name=utils.make_name("shape"), np_val=np.array([-1], dtype=np.int64))
                reshape = ctx.make_node("Cast",
                                        ctx.make_node("Reshape", inputs=[node.input[2], shape.output[0]]).output,
                                        attr={"to": utils.map_numpy_to_onnx_dtype(np.int64)})
                crops = ctx.make_node("Split", inputs=reshape.output, attr={'num_outputs': 4}, output_count=4).output
                zero = ctx.make_const(name=utils.make_name("zero"), np_val=np.array([0], dtype=np.int64)).output[0]
                int32_max = ctx.make_const(name=utils.make_name("int32_max"),
                                           np_val=np.array([np.iinfo(np.int32).max], dtype=np.int64)).output[0]
                def crop_to_end(crop):
                    eq = ctx.make_node("Equal", [crop, zero])
                    not_eq = ctx.make_node("Not", eq.output)
                    cast_eq = ctx.make_node("Cast", eq.output, attr={"to": utils.map_numpy_to_onnx_dtype(np.int64)})
                    cast_not_eq = ctx.make_node("Cast", not_eq.output,
                                                attr={"to": utils.map_numpy_to_onnx_dtype(np.int64)})
                    neg = ctx.make_node("Neg", cast_not_eq.output)
                    add = ctx.make_node("Add",
                                        [
                                            ctx.make_node("Mul", [crop, neg.output[0]]).output[0],
                                            ctx.make_node("Mul", [int32_max, cast_eq.output[0]]).output[0],
                                        ])
                    return add.output[0]

                starts = ctx.make_node("Concat", [crops[0], crops[2]], {'axis': 0})
                ends = ctx.make_node("Concat", [crop_to_end(crops[1]), crop_to_end(crops[3])], {'axis': 0})
                axes = ctx.make_const(name=utils.make_name("axes"), np_val=np.array(slice_axis, dtype=np.int64))
                attr = {"axes": axes.output[0], "ends": ends.output[0], "starts": starts.output[0]}
            inputs_map = {"data": reorganize_node.output[0], **attr}
            dtypes = node.output_dtypes
            shapes = node.output_shapes

            ctx.remove_node(node.name)
            if len(input_shape) == 3:
                # add a squeeze op to convert output into 3d
                kwargs = {**inputs_map}
                node_slice = GraphBuilder(ctx).make_slice(kwargs)
                # CNHW TO NHWC
                trans2 = ctx.make_node("Transpose", [node_slice], {"perm": [1, 2, 3, 0]})
                GraphBuilder(ctx).make_squeeze({"axes": [3], "data": trans2.output[0], "outputs": node.output},
                                               name=node.name, shapes=shapes, dtypes=dtypes)
            else:
                node_slice = GraphBuilder(ctx).make_slice(inputs_map)
                # CNHW TO NCHW
                trans2 = ctx.make_node("Transpose", [node_slice], {"perm": [1, 0, 2, 3]})
                ctx.make_node("Transpose", trans2.output, {"perm": NCHW_TO_NHWC},
                              name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)
        else:
            def mknode(optype, inputs, attrs=None):
                nodename = utils.make_name(node.name + '_' + optype.lower())
                if opset >= 13 and optype == 'Squeeze':
                    return GraphBuilder(ctx).make_squeeze(
                        {"axes": attrs['axes'], "data": inputs[0]}, name=nodename, return_node=True)
                return ctx.make_node(optype, inputs, attrs, name=nodename)


            # support non 3D/4D tensors and dynamic crop vals
            # dynamic slice starts at opset 10
            utils.make_sure(ctx.opset >= 11, 'non-4D tensor or non-const crops require opset 11')

            input0 = node.input[0]
            input2 = node.input[2]

            # const vals
            int_max_const, one_const, minus1_const, blocklen_resize_const, \
            blocklenplus1_const, block_shape_const = \
                [n.output[0] for n in ctx.make_consts([[utils.get_max_value(np.int64)], [1], [-1],\
                                                       [-1, blocklen], [blocklen + 1], block_shape])]

            x_shape = ctx.insert_new_node_on_input(node, 'Shape', node.input[0])

            # get the spatial and depth (i.e remaining) dimensions
            # compute target spatial dimensions by multiplying block_shape
            spatial = mknode('Slice', [x_shape.output[0], one_const, blocklenplus1_const])
            depth = mknode('Slice', [x_shape.output[0], blocklenplus1_const, int_max_const])
            target_spatial = mknode('Mul', [spatial.output[0], block_shape_const])

            # shape to use before shuffle  (part 1)
            ccat1 = mknode('Concat', [spatial.output[0], block_shape_const], {'axis': 0})
            re1 = mknode('Reshape', [ccat1.output[0], blocklen_resize_const])
            tr1 = mknode('Transpose', [re1.output[0]])
            interleave = mknode('Reshape', [tr1.output[0], minus1_const])
            shape1 = mknode('Concat', [minus1_const, interleave.output[0], depth.output[0]], {'axis': 0})

            # shape to use before shuffle (part 2)
            g1 = list(range(2, 2 * blocklen + 1, 2))
            g2 = list(range(1, 2 * blocklen + 1, 2))
            g = g1 + [0] + g2 + list(range(0, xlen + blocklen)[1 + 2 * blocklen:])

            # permutation values for shuffling
            p = np.asarray(range(0, xlen + blocklen))
            p[0] = blocklen
            p[1] = blocklen + 1
            p[2] = 0
            for i in range(3, blocklen * 2 + 1):
                p[i] = p[i - 2] + 1

            # reshape to create moving blocks, shuffle, and reshape to target_spatial
            indices = ctx.make_consts([list(g)])[0].output[0]
            gather = mknode('Gather', [shape1.output[0], indices])
            x2 = mknode('Reshape', [input0, gather.output[0]])
            tr2 = mknode('Transpose', [x2.output[0]], {'perm': np.array(p)})
            shape2 = mknode('Concat', [minus1_const, target_spatial.output[0], depth.output[0]], {'axis': 0})
            x3 = mknode('Reshape', [tr2.output[0], shape2.output[0]])

            # crop axes
            slice_starts_const1, slice_starts_const2, slice_ends_const1, \
            slice_ends_const2, axes_const = \
                [n.output[0] for n in ctx.make_consts([[0, 0], [1, utils.get_max_value(np.int64)], [1, 0],\
                                                       [2, utils.get_max_value(np.int64)], range(1, blocklen + 1)])]

            crop = mknode('Cast', [input2], {'to': TensorProto.INT64})
            crop_transposed = mknode('Transpose', [crop.output[0]])
            crop_starts = mknode('Slice', [crop_transposed.output[0], slice_starts_const1, slice_starts_const2])
            crop_ends = mknode('Slice', [crop_transposed.output[0], slice_ends_const1, slice_ends_const2])
            crop_starts_squeeze = mknode('Squeeze', [crop_starts.output[0]], {'axes': [0]})
            crop_ends_squeeze = mknode('Squeeze', [crop_ends.output[0]], {'axes': [0]})
            end_range = mknode('Sub', [target_spatial.output[0], crop_ends_squeeze.output[0]])
            orig_shape = node.output_shapes
            orig_dtypes = node.output_dtypes
            ctx.remove_node(node.name)
            ctx.make_node('Slice', [x3.output[0], crop_starts_squeeze.output[0], end_range.output[0], axes_const],
                          name=node.name, outputs=node.output, shapes=orig_shape, dtypes=orig_dtypes)

    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        cls.any_version(1, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Parameters moved to inputs for operator Squeeze, Unsqueeze.
        cls.any_version(13, ctx, node, **kwargs)


@tf_op("SpaceToBatchND", onnx_op="SpaceToDepth")
class SpaceToBatch:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # block_shape impacts Transpose 'perm' attribute values.
        # must be available at compile time
        utils.make_sure(node.inputs[1].is_const(), 'only support constant block_shape value.')

        block_shape = node.inputs[1].get_tensor_value(False)
        blocklen = len(block_shape)
        xlen = len(ctx.get_shape(node.input[0]))

        # if 3d or 4d tensor & square 2d block_shape , can optimize
        cond1 = xlen in [3, 4]
        # with opset 11 (or above), we can deal with non-const pads
        # by creating a subgraph with Split and Concat and pass its output
        # to Pad's second input
        cond2 = node.inputs[2].is_const() or ctx.opset >= 11
        cond3 = blocklen == 2 and block_shape[0] == block_shape[1]
        if cond1 and cond2 and cond3:
            # https://www.tensorflow.org/api_docs/python/tf/space_to_batch_nd
            # the above link says the data format of input tensor should be (batch, spatial_shape, remaining_shape)
            # and we only support 4D here, so the data format is NHWC
            # onnx op "SpaceToDepth" does the same work on input tensor except that it works on "C",
            # and it only supports NCHW
            # T out = SpaceToBatchND(T input, int32 block_shape, int32 crops)
            input_tensor = node.inputs[0]
            input_shape = ctx.get_shape(input_tensor.output[0])
            shapes = [ctx.get_shape(node.output[0])]
            dtypes = [ctx.get_dtype(node.output[0])]

            if len(input_shape) == 3:
                # insert automatically an Unsqueeze op if the input is 3d
                unsqz1 = GraphBuilder(ctx).make_unsqueeze(
                    {"axes": [3], "data": input_tensor.output[0]}, return_node=True)
                # NHWC TO CNHW
                trans1 = ctx.make_node("Transpose", unsqz1.output, {"perm": [3, 0, 1, 2]})
            else:
                # Add explicit NHWC_TO_NCHW transpose before and NCHW_TO_NHWC transpose after subgraph.
                # That enables more optimizations in TransposeOptimizer.
                trans_nchw = ctx.make_node("Transpose", input_tensor.output, {"perm": NHWC_TO_NCHW})
                # NCHW TO CNHW
                trans1 = ctx.make_node("Transpose", trans_nchw.output, {"perm": [1, 0, 2, 3]})
            # implement pads logic, the data format is NCHW
            if ctx.opset <= 10 or node.inputs[2].is_const():
                paddings = node.inputs[2].get_tensor_value()
                top, bottom = paddings[0]
                left, right = paddings[1]
                pads = [0, 0, top, left,
                        0, 0, bottom, right]
                if ctx.opset <= 10:
                    pad_op = ctx.make_node("Pad", trans1.output, attr={"pads": pads})
                else:
                    new_pads = ctx.make_const(name=utils.make_name("pads"), np_val=np.array(pads, dtype=np.int64))
                    pad_op = ctx.make_node("Pad", [trans1.output[0], new_pads.output[0]])
            else:
                # TODO: we should be able to support dynamic input here.
                shape = ctx.make_const(name=utils.make_name("shape"), np_val=np.array([-1], dtype=np.int64))
                reshape = ctx.make_node("Reshape", inputs=[node.input[2], shape.output[0]])
                cast = ctx.make_node("Cast", reshape.output, attr={'to': utils.map_numpy_to_onnx_dtype(np.int64)})
                split = ctx.make_node("Split", inputs=cast.output, attr={'num_outputs': 4}, output_count=4)
                pads = split.output
                zero = ctx.make_const(name=utils.make_name("zero"), np_val=np.array([0], dtype=np.int64)).output[0]
                new_pads = ctx.make_node("Concat", [zero, zero, pads[0], pads[2], zero, zero, pads[1], pads[3]],
                                         {'axis': 0})
                pad_op = ctx.make_node("Pad", [trans1.output[0], new_pads.output[0]])

            reorganize_node = ctx.make_node(node.type, pad_op.output, attr={"blocksize": block_shape[0]})

            ctx.remove_node(node.name)
            if len(input_shape) == 3:
                # CNHW TO NHWC
                trans2 = ctx.make_node("Transpose", reorganize_node.output, {"perm": [1, 2, 3, 0]})
                # add a squeeze op to convert output into 3d
                GraphBuilder(ctx).make_squeeze({"axes": [3], "data": trans2.output[0], "outputs": node.output},
                                               name=node.name, shapes=shapes, dtypes=dtypes)
            else:
                # CNHW TO NCHW
                trans2 = ctx.make_node("Transpose", reorganize_node.output, {"perm": [1, 0, 2, 3]})
                ctx.make_node("Transpose", trans2.output, {"perm": NCHW_TO_NHWC},
                              name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)
        else:
            def mknode(optype, inputs, attrs=None):
                nodename = utils.make_name(node.name + '_' + optype.lower())
                return ctx.make_node(optype, inputs, attrs, name=nodename)

            # support non 3D/4D tensors and dynamic pad vals
            # dynamic slice starts at opset 10
            utils.make_sure(ctx.opset >= 11, 'non-4D tensor or non-const pads require opset 11')

            input0 = node.input[0]
            input2 = node.input[2]

            # const vals
            int_max_const, zero_const, one_const, minus1_const, blocklen_resize_const, \
            blocklenplus1_const, filltop_const, fillbottom_const, block_shape_const = \
                [n.output[0] for n in ctx.make_consts([[utils.get_max_value(np.int64)], [0], [1],\
                                                       [-1], [-1, blocklen], [blocklen + 1],\
                                                       [1, 0, 0, 0], [0, 0, 1, 0], block_shape])]

            x_shape = ctx.insert_new_node_on_input(node, 'Shape', node.input[0])
            x_rank = mknode('Size', [x_shape.output[0]])

            # pad x prior to compute
            pad = mknode('Cast', [input2], {'to': TensorProto.INT64})
            pad_shape = mknode('Shape', [pad.output[0]])
            pad_rank = mknode('Slice', [pad_shape.output[0], zero_const, one_const])
            pad_gap = mknode('Sub', [x_rank.output[0], pad_rank.output[0]])
            gapminus1 = mknode('Sub', [pad_gap.output[0], one_const])
            gapminus1fillbot = mknode('Mul', [fillbottom_const, gapminus1.output[0]])
            padfilltop = mknode('Pad', [pad.output[0], filltop_const])
            padfilltopbottom = mknode('Pad', [padfilltop.output[0], gapminus1fillbot.output[0]])
            pad_t = mknode('Transpose', [padfilltopbottom.output[0]])
            pad1d = mknode('Reshape', [pad_t.output[0], minus1_const])

            # get the spatial and depth (i.e remaining) dimensions
            # compute reduced spatial dimensions by dividing block_shape
            x1 = mknode('Pad', [input0, pad1d.output[0]])
            x1_shape = mknode('Shape', [x1.output[0]])
            spatial = mknode('Slice', [x1_shape.output[0], one_const, blocklenplus1_const])
            depth = mknode('Slice', [x1_shape.output[0], blocklenplus1_const, int_max_const])
            reduced = mknode('Div', [spatial.output[0], block_shape_const])

            # reshape x into smaller blocks before shuffle
            ccat1 = mknode('Concat', [reduced.output[0], block_shape_const], {'axis': 0})
            reshape1 = mknode('Reshape', [ccat1.output[0], blocklen_resize_const])
            tr1 = mknode('Transpose', [reshape1.output[0]])
            interleave = mknode('Reshape', [tr1.output[0], minus1_const])
            shape1 = mknode('Concat', [minus1_const, interleave.output[0], depth.output[0]], {'axis': 0})
            x2 = mknode('Reshape', [x1.output[0], shape1.output[0]])

            # permutation values for shuffling
            p1 = list(range(2, 2 * blocklen + 1, 2))
            p2 = list(range(1, 2 * blocklen + 1, 2))
            perm = p1 + [0] + p2 + list(range(0, xlen + blocklen)[1 + 2 * blocklen:])

            tr2 = mknode('Transpose', [x2.output[0]], {'perm': perm})
            shape2 = mknode('Concat', [minus1_const, reduced.output[0], depth.output[0]], {'axis': 0})
            orig_shape = node.output_shapes
            orig_dtypes = node.output_dtypes
            ctx.remove_node(node.name)
            ctx.make_node('Reshape', [tr2.output[0], shape2.output[0]],
                          name=node.name, outputs=node.output, shapes=orig_shape,
                          dtypes=orig_dtypes)


@tf_op("IsInf", onnx_op="IsInf")
class IsInf:
    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        node_dtype = ctx.get_dtype(node.input[0])
        utils.make_sure(node_dtype, "Dtype of {} is None".format(node.name))
        if node_dtype not in [onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.DOUBLE]:
            raise ValueError("dtype " + str(node_dtype) + " is not supported in onnx for now")


@tf_op(["NonMaxSuppressionV2", "NonMaxSuppressionV3", "NonMaxSuppressionV4", "NonMaxSuppressionV5"])
class NonMaxSuppression:
    @classmethod
    def any_version(cls, opset, ctx, node, **kwargs):
        # int32 = NonMaxSuppressionV2(T boxes, T scores, int32 max_output_size, T iou_threshold, T score_threshold)
        # int64 = NonMaxSuppression(T boxes, T scores, int64 max_output_size, T iou_threshold, T score_threshold),
        # T means float32 here, the last 3 params are optional
        # tf boxes is 2D ([boxes_num, 4]) while onnx is 3D ([num_batches, boxes_num, 4])
        # tf scores is 1D ([boxes_num])while onnx is 2D ([num_batches, num_classes, boxes_num])
        # onnx output is [num_selected_boxes, 3], the meaning of last dim is [batch_index, class_index, box_index]
        # while tf's output is [num_selected_boxes]

        # NonMaxSuppressionV2, NonMaxSuppressionV3 return selected_indices
        # NonMaxSuppressionV4 returns selected_indices, valid_outputs
        # NonMaxSuppressionV5 returns selected_indices, selected_scores, valid_outputs

        needs_padding = "pad_to_max_output_size" in node.attr and node.attr["pad_to_max_output_size"].i == 1
        gb = GraphBuilder(ctx)
        input_score0 = gb.make_unsqueeze({'data': node.input[0], 'axes': [0]}, return_node=True)
        input_score1 = gb.make_unsqueeze({'data': node.input[1], 'axes': [0, 1]}, return_node=True)
        ctx.replace_input(node, node.input[0], input_score0.output[0], 0)
        ctx.replace_input(node, node.input[1], input_score1.output[0], 1)
        input_score = input_score1

        ctx.insert_new_node_on_input(node, "Cast", node.input[2], to=onnx_pb.TensorProto.INT64)
        # replace original node with nonmaxsurppress + slice + squeeze + cast
        dtypes = [[ctx.get_dtype(output)] for output in node.output]
        shapes = [[ctx.get_shape(output)] for output in node.output]
        max_output_size = node.input[2]
        utils.make_sure(len(node.inputs) <= 5 or int(node.inputs[5].get_tensor_value(False)) == 0,
                        "soft_nms_sigma must be 0")
        ctx.remove_node(node.name)
        new_nonmaxsurppress = ctx.make_node("NonMaxSuppression", node.input[: 5]).output[0]
        slice_op = GraphBuilder(ctx).make_slice({"data": new_nonmaxsurppress,
                                                 "axes": [1], "ends": [3], "starts": [2]})
        nms_output = GraphBuilder(ctx).make_squeeze({'data': slice_op, "axes": [1]}, return_node=True)
        original_nms_output = nms_output
        if node.type in ["NonMaxSuppressionV4", "NonMaxSuppressionV5"]:
            # add valid_outputs count
            output_idx = 2 if node.type in ["NonMaxSuppressionV5"] else 1
            shape_op = ctx.make_node("Shape", inputs=[nms_output.output[0]])
            reduce_op = GraphBuilder(ctx).make_reduce_sum(
                {"data": shape_op.output[0], "axes": [0], "keepdims": 0, "noop_with_empty_axes": 1})
            ctx.make_node("Cast", inputs=[reduce_op], attr={"to": onnx_pb.TensorProto.INT32},
                          outputs=[node.output[output_idx]], dtypes=dtypes[output_idx], shapes=shapes[output_idx],
                          op_name_scope=node.name)

        pad_amt = None
        if needs_padding:
            # pad_amt might be shared between selected_indices, selected_scores
            sub_op = ctx.make_node("Sub", inputs=[max_output_size, shape_op.output[0]])
            raw_pad_float = ctx.make_node("Cast", inputs=[sub_op.output[0]], attr={"to": onnx_pb.TensorProto.FLOAT})
            relu_op = ctx.make_node("Relu", inputs=[raw_pad_float.output[0]])
            pad_amt = ctx.make_node("Cast", inputs=[relu_op.output[0]], attr={"to": onnx_pb.TensorProto.INT64})
            #
            # pad selected_indices
            #
            if ctx.opset <= 10:  # Dynamic padding not supported before opset 11
                zero_tensor = helper.make_tensor("value", onnx_pb.TensorProto.INT64, dims=[1], vals=[0])
                padding = ctx.make_node("ConstantOfShape", inputs=[pad_amt.output[0]], attr={"value": zero_tensor})
                pad_op = ctx.make_node("Concat", inputs=[nms_output.output[0], padding.output[0]], attr={'axis': 0})
            else:
                const_zero = ctx.make_const(utils.make_name("const_zero"), np.array([0], dtype=np.int64))
                pad_val = ctx.make_node("Concat", inputs=[const_zero.output[0], pad_amt.output[0]], attr={'axis': 0})
                pad_op = ctx.make_node("Pad", inputs=[nms_output.output[0], pad_val.output[0]])
            nms_output = pad_op

        if node.type in ["NonMaxSuppressionV5"]:
            if needs_padding:
                # add selected_scores with padding
                gather_op = ctx.make_node("Gather", inputs=[input_score.input[0], original_nms_output.output[0]],
                                          dtypes=dtypes[1], shapes=shapes[1])
                if ctx.opset <= 10:  # Dynamic padding not supported before opset 11
                    zero_tensor = helper.make_tensor("value", dtypes[1], dims=[1], vals=[0])
                    padding = ctx.make_node("ConstantOfShape", inputs=[pad_amt.output[0]], attr={"value": zero_tensor})
                    pad_op = ctx.make_node("Concat", inputs=[gather_op.output[0], padding.output[0]],
                                           outputs=[node.output[1]], dtypes=dtypes[1], shapes=shapes[1],
                                           attr={'axis': 0})
                else:
                    const_zero = ctx.make_const(utils.make_name("const_zero"), np.array([0], dtype=np.int64))
                    pad_val = ctx.make_node("Concat", inputs=[const_zero.output[0], pad_amt.output[0]],
                                            attr={'axis': 0})
                    pad_op = ctx.make_node("Pad", inputs=[gather_op.output[0], pad_val.output[0]],
                                           outputs=[node.output[1]], dtypes=dtypes[1], shapes=shapes[1])
            else:
                # add selected_scores without padding
                ctx.make_node("Gather", inputs=[input_score.input[0], nms_output.output[0]],
                              outputs=[node.output[1]], dtypes=dtypes[1], shapes=shapes[1])

        # cast selected_indices back to int32
        ctx.make_node("Cast", inputs=nms_output.output, attr={"to": onnx_pb.TensorProto.INT32},
                      outputs=[node.output[0]], dtypes=dtypes[0], shapes=shapes[0])

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        cls.any_version(10, ctx, node, **kwargs)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # no change
        cls.any_version(11, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Parameters moved to inputs for operator Squeeze, Unsqueeze.
        cls.any_version(13, ctx, node, **kwargs)


@tf_op(["CombinedNonMaxSuppression"])
class CombinedNonMaxSuppression:
    @classmethod
    def version_12(cls, ctx, node, **kwargs):
        # boxes.shape = [batch_size, num_boxes, (1 OR num_classes), 4]
        # scores.shape = [batch_size, num_boxes, num_classes]
        boxes, scores, max_per_class, max_total_size, iou_threshold, score_threshold = node.input

        max_per_class = ctx.make_node("Cast", [max_per_class], attr={'to': TensorProto.INT64}).output[0]
        max_total_size = ctx.make_node("Cast", [max_total_size], attr={'to': TensorProto.INT64}).output[0]

        pad_per_class = node.get_attr_value("pad_per_class", False)
        clip_boxes = node.get_attr_value("clip_boxes", True)
        shape = ctx.get_shape(boxes)
        share_boxes_across_classes = shape is not None and shape[2] == 1
        utils.make_sure(share_boxes_across_classes,
                        "CombinedNonMaxSuppression only currently implemented for boxes shared across classes.")

        scores_shape = ctx.make_node("Shape", [scores]).output[0]
        # value: [batch_size]
        batch_size = GraphBuilder(ctx).make_slice({'data': scores_shape, 'starts': [0], 'ends': [1], 'axes': [0]})

        num_classes = GraphBuilder(ctx).make_slice({'data': scores_shape, 'starts': [2], 'ends': [3], 'axes': [0]})
        max_per_class_times_classes = ctx.make_node("Mul", [max_per_class, num_classes]).output[0]

        const_zero_float = ctx.make_const(utils.make_name("const_zero"), np.array(0, np.float32)).output[0]
        const_one_float = ctx.make_const(utils.make_name("const_one"), np.array(1, np.float32)).output[0]
        const_zero = ctx.make_const(utils.make_name("const_zero"), np.array(0, np.int64)).output[0]
        const_neg_one = ctx.make_const(utils.make_name("const_neg_one"), np.array(-1, np.int64)).output[0]
        const_one = ctx.make_const(utils.make_name("const_one"), np.array(1, np.int64)).output[0]

        boxes_sq = GraphBuilder(ctx).make_squeeze({'data': boxes, 'axes': [2]})
        # scores_trans.shape = [batch_size, num_classes, num_boxes]
        scores_trans = ctx.make_node("Transpose", [scores], attr={'perm': [0, 2, 1]}).output[0]
        # shape: [num_selected, 3], elts of format [batch_index, class_index, box_index]
        selected_indices = ctx.make_node(
            "NonMaxSuppression", [boxes_sq, scores_trans, max_per_class, iou_threshold, score_threshold],
            op_name_scope=node.name).output[0]
        selected_classes_unsq = GraphBuilder(ctx).make_slice(
            {'data': selected_indices, 'starts': [1], 'ends': [2], 'axes': [1]})
        selected_classes = GraphBuilder(ctx).make_squeeze({'data': selected_classes_unsq, 'axes': [1]})
        # shape: [num_selected]
        selected_scores = ctx.make_node("GatherND", [scores_trans, selected_indices], op_name_scope=node.name).output[0]
        # shape: [num_selected, 1]
        selected_batch_idx = GraphBuilder(ctx).make_slice(
            {'data': selected_indices, 'starts': [0], 'ends': [1], 'axes': [1]})
        selected_box_num = GraphBuilder(ctx).make_slice(
            {'data': selected_indices, 'starts': [2], 'ends': [3], 'axes': [1]})
        combined_box_idx = ctx.make_node("Concat", [selected_batch_idx, selected_box_num], attr={'axis': 1}).output[0]
        selected_boxes_unsq = ctx.make_node("GatherND", [boxes, combined_box_idx], op_name_scope=node.name).output[0]
        # shape: [num_selected, 4]
        selected_boxes = GraphBuilder(ctx).make_squeeze({'data': selected_boxes_unsq, 'axes': [1]})

        clipped_boxes = selected_boxes
        if clip_boxes:
            clipped_boxes = ctx.make_node('Max', [clipped_boxes, const_zero_float]).output[0]
            clipped_boxes = ctx.make_node('Min', [clipped_boxes, const_one_float]).output[0]

        # shape: [num_selected]
        batch_idx_sq = GraphBuilder(ctx).make_squeeze({'data': selected_batch_idx, 'axes': [1]})
        # value: [num_selected]
        num_selected = ctx.make_node("Shape", [selected_scores]).output[0]
        num_selected_sq = GraphBuilder(ctx).make_squeeze({'data': num_selected, 'axes': [0]})
        # shape: [num_selected]
        selected_range = ctx.make_node("Range", [const_zero, num_selected_sq, const_one]).output[0]


        id_shape = ctx.make_node("Concat", [batch_size, batch_size], attr={'axis': 0}).output[0]
        zero_tensor = helper.make_tensor("value", TensorProto.INT64, dims=[1], vals=[0])
        zeros_of_shape = ctx.make_node("ConstantOfShape", [id_shape], attr={"value": zero_tensor}).output[0]
        # shape: [batch_size, batch_size]
        id_matrix = ctx.make_node("EyeLike", [zeros_of_shape]).output[0]
        # shape: [num_selected, batch_size]
        one_hot_batch_idx = ctx.make_node("Gather", [id_matrix, batch_idx_sq], attr={'axis': 0}).output[0]
        cum_batch_idx = ctx.make_node("CumSum", [one_hot_batch_idx, const_zero], {'exclusive': True}).output[0]
        # shape: [num_selected]
        idx_within_batch = ctx.make_node("GatherND", [cum_batch_idx, selected_batch_idx], attr={'batch_dims': 1},
                                         op_name_scope=node.name).output[0]
        idx_within_batch_unsq = GraphBuilder(ctx).make_unsqueeze({'data': idx_within_batch, 'axes': [1]})
        combined_idx = ctx.make_node("Concat", [selected_batch_idx, idx_within_batch_unsq], attr={'axis': 1}).output[0]

        zero_tensor_float = helper.make_tensor("value", TensorProto.FLOAT, dims=[1], vals=[0])
        neg_one_tensor_float = helper.make_tensor("value", TensorProto.INT64, dims=[1], vals=[-1])
        # value: [batch_size, max_per_class_times_classes]
        results_grid_shape = ctx.make_node(
            "Concat", [batch_size, max_per_class_times_classes], attr={'axis': 0}).output[0]
        scores_by_batch_empty = ctx.make_node(
            "ConstantOfShape", [results_grid_shape], attr={"value": zero_tensor_float}).output[0]
        idx_by_batch_empty = ctx.make_node(
            "ConstantOfShape", [results_grid_shape], attr={"value": neg_one_tensor_float}).output[0]

        scores_by_batch = ctx.make_node("ScatterND", [scores_by_batch_empty, combined_idx, selected_scores]).output[0]
        idx_by_batch = ctx.make_node("ScatterND", [idx_by_batch_empty, combined_idx, selected_range]).output[0]

        k_val = ctx.make_node("Min", [max_total_size, max_per_class_times_classes]).output[0]

        # shape: [batch_size, k_val]
        top_k_vals, top_k_indices = \
            ctx.make_node("TopK", [scores_by_batch, k_val], attr={'axis': 1}, output_count=2).output

        top_k_selected_indices = ctx.make_node("GatherElements", [idx_by_batch, top_k_indices], attr={'axis': 1},
                                               op_name_scope=node.name).output[0]

        target_size = max_total_size
        if pad_per_class:
            target_size = k_val

        pad_amt = ctx.make_node("Sub", [target_size, k_val]).output[0]
        pads_const = ctx.make_const(utils.make_name("pad_const"), np.array([0, 0, 0], np.int64)).output[0]
        pads = ctx.make_node("Concat", [pads_const, pad_amt], attr={'axis': 0}).output[0]

        top_scores_pad = ctx.make_node("Pad", [top_k_vals, pads, const_zero_float]).output[0]
        top_indices_pad = ctx.make_node("Pad", [top_k_selected_indices, pads, const_neg_one]).output[0]
        top_indices_increment = ctx.make_node("Add", [top_indices_pad, const_one]).output[0]

        valid_indices = ctx.make_node("Greater", [top_k_selected_indices, const_neg_one]).output[0]
        valid_indices_int = ctx.make_node("Cast", [valid_indices], attr={'to': TensorProto.INT32}).output[0]
        # shape: [batch_size]
        valid_indices_cnt = GraphBuilder(ctx).make_reduce_sum(
            {"data": valid_indices_int, "axes": [-1], "keepdims": 0, "noop_with_empty_axes": 1})

        box_pads = ctx.make_const(utils.make_name("pad_const"), np.array([1, 0, 0, 0], np.int64)).output[0]
        class_pads = ctx.make_const(utils.make_name("pad_const"), np.array([1, 0], np.int64)).output[0]
        clipped_boxes_pad = ctx.make_node("Pad", [clipped_boxes, box_pads, const_zero_float]).output[0]
        selected_classes_pad = ctx.make_node("Pad", [selected_classes, class_pads, const_zero]).output[0]
        nmsed_boxes = ctx.make_node("Gather", [clipped_boxes_pad, top_indices_increment], attr={'axis': 0},
                                    op_name_scope=node.name).output[0]
        nmsed_classes = ctx.make_node("Gather", [selected_classes_pad, top_indices_increment], attr={'axis': 0},
                                      op_name_scope=node.name).output[0]
        nmsed_classes_float = ctx.make_node("Cast", [nmsed_classes], attr={'to': TensorProto.FLOAT}).output[0]

        ctx.replace_all_inputs(node.output[0], nmsed_boxes)
        ctx.replace_all_inputs(node.output[1], top_scores_pad)
        ctx.replace_all_inputs(node.output[2], nmsed_classes_float)
        ctx.replace_all_inputs(node.output[3], valid_indices_cnt)
        ctx.remove_node(node.name)


@tf_op("ReverseSequence")
class ReverseSequence:
    @classmethod
    def version_8(cls, ctx, node, **kwargs):
        # T output = ReverseSequence(T input, int32|int64 seq_lengths, @int seq_dim, @int batch_dim)
        # T output = Scan(int64 sequence_lens, variadic initial_state_and_scan_inputs, @graph body,
        #                 @ints directions,@int num_scan_inputs)
        seq_dim = node.get_attr("seq_dim")
        batch_dim = node.get_attr("batch_dim")
        batch_major = seq_dim.i == 1 and (batch_dim or batch_dim.i == 0)
        time_major = batch_dim.i == 1 and (seq_dim or seq_dim.i == 0)
        perm_val = None

        if not batch_major and not time_major:
            error_msg = "unsupported attributes, seq_dim:{}, batch_dim:{}".format(seq_dim, batch_dim)
            raise ValueError(error_msg)

        if time_major:
            old_shape = ctx.get_shape(node.input[0])
            old_dtype = ctx.get_dtype(node.input[0])
            perm_val = [1, 0]
            rank = len(old_shape)
            utils.make_sure(rank >= 2, "rank of reverse_sequence input {} is at least 2".format(node.input[0]))
            perm_val += list(range(2, rank))
            trans_node = ctx.insert_new_node_on_input(node, "Transpose", node.input[0], perm=perm_val)
            new_shape = nn.spatial_map(old_shape, perm_val)
            ctx.set_shape(trans_node.output[0], new_shape)
            ctx.set_dtype(trans_node.output[0], old_dtype)

        # handle batch_major input
        node.type = "Scan"
        node.set_attr("num_scan_inputs", 1)
        input_dtype = ctx.get_dtype(node.input[0])
        input_shape = ctx.get_shape(node.input[0])

        g = ctx.create_new_graph_with_same_config()
        g.parent_graph = ctx
        g.add_graph_input('X', input_dtype, input_shape[2:])
        g.make_node('Identity', ['X'], outputs=['Y'])
        g.add_graph_output('Y', input_dtype, input_shape[2:])

        node.set_body_graph_as_attr("body", g)
        node.set_attr("directions", [1])  # reverse the scan input

        seq_len_dtype = ctx.get_dtype(node.input[1])
        if seq_len_dtype != onnx_pb.TensorProto.INT64:
            cast_node = ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=onnx_pb.TensorProto.INT64)
            ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.INT64)
            ctx.copy_shape(node.input[1], cast_node.output[0])

        if time_major:
            # get back to time_major
            op_name = utils.make_name(node.name)
            curr_shape = ctx.get_shape(node.output[0])
            if curr_shape is not None:
                new_shape = [curr_shape[perm_val.index(i)] for i in range(len(perm_val))]
                ctx.set_shape(node.output[0], new_shape)
            trans_back_node = ctx.insert_new_node_on_output("Transpose", node.output[0],
                                                            name=op_name, perm=perm_val)
            ctx.copy_dtype(node.output[0], trans_back_node.output[0])

        tmp = node.input[0]
        ctx.replace_input(node, node.input[0], node.input[1], 0)
        ctx.replace_input(node, node.input[1], tmp, 1)

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        # T output = ReverseSequence(T input, int32|int64 seq_lengths, @int seq_dim, @int batch_dim)
        # we cannot easily construct reverse_sequence equivalence in opset 9, so we will not support it
        # here. Actually using loops to do that is kind of meaningless since there will be performance
        # issue there for sure.
        raise NotImplementedError("ReverseSequence is not supported to convert in OPSET 9,"
                                  " if possible please try using OPSET 8, or OPSET >=10 instead.")

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        # T output = ReverseSequence(T input, int32|int64 seq_lengths, @int seq_dim, @int batch_dim)
        # T output = ReverseSequence(T input, int64 sequence_lens, @int time_axis, @int batch_axis)
        seq_dim = node.get_attr("seq_dim")
        utils.make_sure(seq_dim is not None, "sequence dim must be given in {}".format(node.name))
        seq_dim = seq_dim.i
        batch_dim = node.get_attr_value("batch_dim", 0)

        ctx.remove_node(node.name)
        shape = ctx.get_shape(node.input[0])
        dtype = ctx.get_dtype(node.input[0])
        node = ctx.make_node(
            "ReverseSequence",
            node.input,
            outputs=node.output,
            attr={"batch_axis": batch_dim, "time_axis": seq_dim}, shapes=[shape], dtypes=[dtype])

        seq_len_dtype = ctx.get_dtype(node.input[1])
        utils.make_sure(seq_len_dtype is not None, "dtype of {} is None".format(node.input[1]))
        target_dtype = TensorProto.INT64
        if seq_len_dtype != target_dtype:
            ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=target_dtype)


@tf_op("ReverseV2")
class ReverseV2:
    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        # T output = ReverseV2(T input, int32|int64 seq_lengths, @int seq_dim, @int batch_dim)
        # Implement tensorflow ReverseV2 op using multiple ReverseSequence (for each axis)
        # and Transpose ops. We sort the axis vector (if non-empty) at the start. Each axis can
        # be reversed only once (in tf) and so we can compute the transpose for each axis
        # (other than 0), feed the tensor to a ReverseSequence node and finally transpose again
        # to get back the original shape.

        axes_node = node.inputs[1]
        axes = axes_node.get_tensor_value(as_list=False)
        # Current support is for when axis is a 1D tensor.
        utils.make_sure(len(axes.shape) == 1,
                        "Currently no support for reverseV2 tensor axis")

        axes = axes.tolist()
        len_axes = len(axes)

        input_rank = ctx.get_rank(node.input[0])
        utils.make_sure(input_rank is not None, "rank of {} is unknown".format(node.input[0]))
        needs_squeeze = False
        if input_rank == 1 and len_axes != 0:
            # ReverseSequence node requires rank >= 2
            utils.make_sure(axes in [[-1], [0]], "Invalid value %s for axes of ReverseV2 of 1d tensor", axes)
            axes = [1]
            new_inp = GraphBuilder(ctx).make_unsqueeze({'data': node.input[0], 'axes': [0]})
            ctx.replace_input(node, node.input[0], new_inp, 0)
            input_rank = 2
            needs_squeeze = True

        # Store input and output parameters of the ReverseV2 node.
        rv2_in_names = [node.input[0]]

        input_shape_node = ctx.make_node("Shape", [node.input[0]], op_name_scope=node.name)

        rv2_node_name = node.name
        # ReverseV2 has a single output.
        rv2_output_dtypes = node.output_dtypes
        rv2_output_shapes = node.output_shapes
        if needs_squeeze and rv2_output_shapes is not None:
            rv2_output_shapes[0] = [1] + rv2_output_shapes[0]

        # Remove ReverseV2 node from graph.
        ctx.remove_node(rv2_node_name)

        # Variable to store input names for the next node.
        inputs = rv2_in_names

        new_node = None

        # Empty axis vector.
        if len_axes == 0:
            # Replace ReverseV2 with an identity block.
            ctx.make_node(
                "Identity",
                inputs=inputs,
                outputs=node.output,
                shapes=rv2_output_shapes,
                dtypes=rv2_output_dtypes,
                op_name_scope=rv2_node_name,
            )

        else:
            # For negative indices use the positive counterpart.
            for i, ax in enumerate(axes):
                if ax < 0:
                    axes[i] += input_rank

            axes = sorted(axes)

            orig_perm = list(range(input_rank))
            curr_perm = []

            # Add ReverseSequence nodes for each element of axis.
            for i in range(len_axes):

                axis = axes[i]

                curr_perm = orig_perm.copy()
                # Permutation indices relative to original tensor.
                curr_perm[axis], curr_perm[0] = curr_perm[0], curr_perm[axis]

                # Add a Transpose node if the axis != 0 (finish first due to sort).
                if axis != 0:
                    # Permutation indices for the transpose node relative to IN tensor shape.
                    new_node = ctx.make_node(
                        "Transpose",
                        inputs=inputs,
                        op_name_scope=rv2_node_name,
                        dtypes=rv2_output_dtypes,
                        attr={"perm": curr_perm}
                    )

                    inputs = [new_node.output[0]]

                const_one_name = utils.make_name('const_one')
                const_one = ctx.make_const(name=const_one_name, np_val=np.array([1], dtype=np.int64))
                const_axis_name = utils.make_name(f'const_{axis}')
                const_axis = ctx.make_const(name=const_axis_name, np_val=np.array([axis], dtype=np.int64))

                # Add a Constant node (seq_len) for ReverseSequence.
                # Index 1 for the shape should not return 0, since rank(input) >=2
                input_shape = ctx.make_node("Shape", [inputs[-1]], op_name_scope=rv2_node_name)
                batch_size = ctx.make_node("Gather", [input_shape.output[0], const_one.output[0]],
                                           op_name_scope=rv2_node_name)
                axis_dim = ctx.make_node("Gather", [input_shape_node.output[0], const_axis.output[0]],
                                         op_name_scope=rv2_node_name)
                seq_array = ctx.make_node("Expand", [axis_dim.output[0], batch_size.output[0]])
                inputs.append(seq_array.output[0])

                # Add a ReverseSequence node.

                # If processing for the final axis and the tensor shape permutation is
                # original then the output is fed to the output of the ReverseV2 node.
                #
                # Else a new output is created which is fed to a Transpose node.
                rs_out_name = node.output if \
                    ((i == len_axes - 1) and (curr_perm == orig_perm)) \
                    else None

                if rs_out_name is not None:
                    rs_out_shapes = rv2_output_shapes
                else:
                    rs_out_shapes = [ctx.get_shape(inputs[0])]

                new_node = ctx.make_node(
                    "ReverseSequence",
                    inputs=inputs,
                    op_name_scope=rv2_node_name,
                    outputs=rs_out_name,
                    shapes=rs_out_shapes,
                    dtypes=rv2_output_dtypes,
                    attr={"batch_axis": 1, "time_axis": 0}
                )

                inputs = [new_node.output[0]]

            # Additional transpose block is required if the current
            # permutation list is not the original one.
            if curr_perm != orig_perm:

                # Compute the required permutation list.
                if len_axes != 1:
                    for i, ax in enumerate(axes[::-1][1:]):
                        curr_perm[0], curr_perm[ax] = \
                            curr_perm[ax], curr_perm[0]

                # Add a Transpose node to restore shape.
                ctx.make_node(
                    "Transpose",
                    inputs=inputs,
                    op_name_scope=rv2_node_name,
                    outputs=node.output,
                    shapes=rv2_output_shapes,
                    dtypes=rv2_output_dtypes,
                    attr={"perm": curr_perm}
                )

            if needs_squeeze:
                sq_node = GraphBuilder(ctx).make_squeeze({"data": node.output[0], "axes": [0]}, return_node=True)
                ctx.insert_node_on_output(sq_node)


@tf_op("Unique", onnx_op="Unique")
@tf_op("UniqueWithCounts", onnx_op="Unique")
class Unique:
    int_cast = [TensorProto.BOOL, TensorProto.INT32, TensorProto.INT16, TensorProto.UINT8,
                TensorProto.UINT16, TensorProto.UINT32, TensorProto.UINT64]
    dtype_map = {k: TensorProto.INT64 for k in int_cast}
    dtype_map[TensorProto.DOUBLE] = TensorProto.FLOAT

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # opset 11 supports explicitly
        node_name = node.name
        node_inputs = node.input
        node_outputs = node.output
        inp_dtype = ctx.get_dtype(node.input[0])

        ctx.remove_node(node_name)

        # due to ORT missing implementations we need to cast INT inputs to INT64 and FLOAT inputs to FLOAT32
        if inp_dtype in cls.dtype_map:
            inp_cast = ctx.make_node("Cast", [node_inputs[0]], attr={'to': cls.dtype_map[inp_dtype]}).output[0]
            node_inputs[0] = inp_cast

        new_node = ctx.make_node("Unique", node_inputs, name=node_name, attr={'sorted': 0},
                                 outputs=[utils.make_name("y"), utils.make_name("idx_first"),
                                          utils.make_name("idx"), utils.make_name("counts")])
        ctx.replace_all_inputs(node_outputs[0], new_node.output[0])
        ctx.replace_all_inputs(node_outputs[1], new_node.output[2])
        if len(node_outputs) == 3: # we need counts too (UniqueWithCounts)
            ctx.replace_all_inputs(node_outputs[2], new_node.output[3])
        if ctx.get_dtype(new_node.output[0]) != inp_dtype:
            ctx.insert_new_node_on_output("Cast", new_node.output[0], to=inp_dtype,
                                          name=utils.make_name(node.name) + "_cast")

        # cast idx and counts if needed
        out_dtype = node.get_attr_value('out_idx')
        if out_dtype != TensorProto.INT64:
            for i in range(1, len(node_outputs)):
                cast_node = ctx.insert_new_node_on_output("Cast", new_node.output[i+1], to=out_dtype,
                                                          name=utils.make_name(node.name) + "_cast")


@tf_op(["Bincount", "DenseBincount"])
class Bincount:
    @classmethod
    def any_version(cls, opset, ctx, node, **kwargs):
        # arr, size are int32
        arr_inp, size_inp, weights_inp = node.input
        binary_output = node.get_attr_value("binary_output", False)
        arr_int64 = ctx.make_node("Cast", [arr_inp], attr={'to': TensorProto.INT64}).output[0]
        size_int64 = ctx.make_node("Cast", [size_inp], attr={'to': TensorProto.INT64}).output[0]

        weights_shape = ctx.get_shape(weights_inp)
        res_dtype = ctx.get_dtype(weights_inp)
        weights_is_zero = weights_shape is not None and 0 in weights_shape
        utils.make_sure(weights_is_zero, "Non-empty weights not yet supported for bincount")

        if ctx.get_rank(arr_inp) == 2:
            zero_const = ctx.make_const(utils.make_name("zero_const"), np.array(0, np.int64)).output[0]
            one_const = ctx.make_const(utils.make_name("one_const"), np.array(1, np.int64)).output[0]
            inp_shape = ctx.make_node("Shape", [arr_inp]).output[0]
            num_rows = GraphBuilder(ctx).make_slice({"data": inp_shape, "starts": [0], "ends": [1], "axes": [0]})
            num_rows_sq = GraphBuilder(ctx).make_squeeze({"data": num_rows, "axes": [0]})
            row_idx = ctx.make_node("Range", [zero_const, num_rows_sq, one_const]).output[0]
            row_idx_unsq = GraphBuilder(ctx).make_unsqueeze({"data": row_idx, "axes": [1]})
            row_idx_expand = ctx.make_node("Expand", [row_idx_unsq, inp_shape]).output[0]
            arr_int64_unsq = GraphBuilder(ctx).make_unsqueeze({"data": arr_int64, "axes": [2]})
            row_idx_expand_unsq = GraphBuilder(ctx).make_unsqueeze({"data": row_idx_expand, "axes": [2]})
            concat = ctx.make_node("Concat", [row_idx_expand_unsq, arr_int64_unsq], {"axis": 2}).output[0]
            reshape_const = ctx.make_const(utils.make_name("reshape_const"), np.array([-1, 2], np.int64)).output[0]
            reshaped = ctx.make_node("Reshape", [concat, reshape_const]).output[0]
            values, _, _, counts = ctx.make_node("Unique", [reshaped], attr={'sorted': 1, 'axis': 0}, output_count=4,
                                                 op_name_scope=node.name).output
            values_to_check_unsq = GraphBuilder(ctx).make_slice(
                {"data": values, "starts": [1], "ends": [2], "axes": [1]})
            values_to_check = GraphBuilder(ctx).make_squeeze({"data": values_to_check_unsq, "axes": [1]})
            size_unsq = GraphBuilder(ctx).make_unsqueeze({'data': size_int64, "axes": [0]})
            output_shape = ctx.make_node("Concat", [num_rows, size_unsq], attr={"axis": 0}).output[0]
        else:
            values, _, _, counts = ctx.make_node("Unique", [arr_int64], attr={'sorted': 1}, output_count=4,
                                                 op_name_scope=node.name).output
            values_to_check = values
            output_shape = GraphBuilder(ctx).make_unsqueeze({'data': size_int64, "axes": [0]})

        neg_one_const = ctx.make_const(utils.make_name("neg_one_const"), np.array(-1, np.int64)).output[0]
        non_neg_val_locs = ctx.make_node("Greater", [values_to_check, neg_one_const]).output[0]
        small_val_locs = ctx.make_node("Less", [values_to_check, size_int64]).output[0]
        valid_val_locs = ctx.make_node("And", [non_neg_val_locs, small_val_locs]).output[0]

        valid_values = ctx.make_node("Compress", [values, valid_val_locs], attr={'axis': 0}).output[0]
        if binary_output:
            counts_shape = ctx.make_node("Shape", [valid_values]).output[0]
            counts_shape_1d = GraphBuilder(ctx).make_slice(
                {"data": counts_shape, "starts": [0], "ends": [1], "axes": [0]})
            ones_tensor = helper.make_tensor("value", TensorProto.INT64, dims=[1], vals=[1])
            valid_counts = ctx.make_node("ConstantOfShape", [counts_shape_1d], attr={'value': ones_tensor}).output[0]
        else:
            valid_counts = ctx.make_node("Compress", [counts, valid_val_locs], attr={'axis': 0}).output[0]

        zero_tensor = helper.make_tensor("value", TensorProto.INT64, dims=[1], vals=[0])
        zeros = ctx.make_node("ConstantOfShape", [output_shape], attr={'value': zero_tensor}).output[0]

        if ctx.get_rank(arr_inp) == 2:
            result = ctx.make_node("ScatterND", [zeros, valid_values, valid_counts]).output[0]
        else:
            result = ctx.make_node("ScatterElements", [zeros, valid_values, valid_counts], attr={'axis': 0}).output[0]
        result_cast = result
        if res_dtype != TensorProto.INT64:
            result_cast = ctx.make_node("Cast", [result], attr={'to': res_dtype}).output[0]

        ctx.replace_all_inputs(node.output[0], result_cast)
        ctx.remove_node(node.name)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        cls.any_version(11, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Parameters moved to inputs for operator Squeeze, Unsqueeze.
        cls.any_version(13, ctx, node, **kwargs)


@tf_op("SparseToDense")
class SparseToDense:
    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        sparse_indices, out_shape, sparse_vals, default_val = node.input
        idx_shape = ctx.get_shape(sparse_indices)
        val_shape = ctx.get_shape(sparse_vals)
        val_is_scalar = val_shape is not None and val_shape[0] == 1
        idx_is_scalar = idx_shape is not None and idx_shape[0] == 1
        utils.make_sure(not val_is_scalar or idx_is_scalar, "SparseToDense not implemented yet for scalar values")

        expand_node = ctx.make_node("Expand", [default_val, out_shape])
        node.type = "ScatterND"
        ctx.replace_inputs(node, [expand_node.output[0], sparse_indices, sparse_vals])


def ragged_lengths_to_sparse_indices(ctx, ragged_lens):
    const_zero_int64 = ctx.make_const(utils.make_name("const_zero"), np.array(0, dtype=np.int64)).output[0]
    num_cols = GraphBuilder(ctx).make_reduce_max({"data": ragged_lens, "axes": [0], "keepdims": True})
    num_rows = ctx.make_node("Shape", [ragged_lens]).output[0]
    range_len = ctx.make_node("Mul", [num_cols, num_rows]).output[0]

    # ORT seems to have a shape inference bug for the Range node. Use CumSum instead.
    one_tensor = helper.make_tensor("value", TensorProto.INT64, dims=[1], vals=[1])
    ones_of_shape = ctx.make_node("ConstantOfShape", [range_len], attr={"value": one_tensor}).output[0]
    range_node = ctx.make_node("CumSum", [ones_of_shape, const_zero_int64], attr={'exclusive': True}).output[0]
    #const_one_int64 = ctx.make_const(utils.make_name("const_one"), np.array(1, dtype=np.int64)).output[0]
    #range_node = ctx.make_node("Range", [const_zero_int64, range_len, const_one_int64]).output[0]

    col_indices_dense = ctx.make_node("Mod", [range_node, num_cols]).output[0]
    row_indices_dense = ctx.make_node("Div", [range_node, num_cols]).output[0]
    row_lens_dense = ctx.make_node("Gather", [ragged_lens, row_indices_dense]).output[0]
    indices_to_keep = ctx.make_node("Less", [col_indices_dense, row_lens_dense]).output[0]
    col_indices = ctx.make_node("Compress", [col_indices_dense, indices_to_keep]).output[0]
    row_indices = ctx.make_node("Compress", [row_indices_dense, indices_to_keep]).output[0]
    return num_rows, num_cols, row_indices, col_indices


def ragged_row_ids_to_sparse_indices(ctx, row_ids):
    _, indices, _, counts = ctx.make_node("Unique", [row_ids], attr={'axis': 0}, output_count=4).output
    num_cols = GraphBuilder(ctx).make_reduce_max({"data": counts, "axes": [0], "keepdims": True})
    const_one = ctx.make_const(utils.make_name("const_one"), np.array(1, np.int64)).output[0]
    const_zero_unsq = ctx.make_const(utils.make_name("const_zero"), np.array([0], np.int64)).output[0]
    const_zero = ctx.make_const(utils.make_name("const_zero"), np.array(0, np.int64)).output[0]
    const_neg_one_unsq = ctx.make_const(utils.make_name("const_neg_one"), np.array([-1], np.int64)).output[0]
    one_minus_cnt = ctx.make_node("Sub", [const_one, counts]).output[0]
    cnts_prefixed = ctx.make_node("Concat", [const_zero_unsq, one_minus_cnt], attr={'axis': 0}).output[0]
    cnts_shifted = GraphBuilder(ctx).make_slice(
        {'data': cnts_prefixed, 'starts': const_zero_unsq, 'ends': const_neg_one_unsq, 'axes': [0]})
    ids_shape = ctx.make_node("Shape", [row_ids]).output[0]
    one_tensor = helper.make_tensor("value", onnx_pb.TensorProto.INT64, dims=[1], vals=[1])
    ones_of_shape = ctx.make_node("ConstantOfShape", [ids_shape], attr={'value': one_tensor}).output[0]
    deltas = ctx.make_node("ScatterElements", [ones_of_shape, indices, cnts_shifted], attr={'axis': 0}).output[0]
    col_indices = ctx.make_node("CumSum", [deltas, const_zero]).output[0]
    return num_cols, col_indices


def ragged_nested_splits_to_sparse_indices(ctx, nested_splits, op_name_scope):
    sparse_indices = None
    dense_shape_dims = []
    for split in nested_splits:
        if ctx.get_dtype(split) != TensorProto.INT64:
            split = ctx.make_node("Cast", [split], attr={'to': TensorProto.INT64}).output[0]
        max_int64 = int(utils.get_max_value(np.int64))
        slice1 = GraphBuilder(ctx).make_slice(
            {"data": split, "ends": [max_int64], "starts": [1], "axes": [0]})
        slice2 = GraphBuilder(ctx).make_slice(
            {"data": split, "ends": [-1], "starts": [0], "axes": [0]})
        ragged_lens = ctx.make_node("Sub", [slice1, slice2]).output[0]
        num_rows, num_cols, row_indices, col_indices = ragged_lengths_to_sparse_indices(ctx, ragged_lens)
        if not dense_shape_dims:
            dense_shape_dims.append(num_rows)
        dense_shape_dims.append(num_cols)
        if sparse_indices is None:
            row_indices = GraphBuilder(ctx).make_unsqueeze({"data": row_indices, "axes": [1]})
        else:
            row_indices = ctx.make_node("Gather", [sparse_indices, row_indices]).output[0]
        col_indices = GraphBuilder(ctx).make_unsqueeze({"data": col_indices, "axes": [1]})
        sparse_indices = ctx.make_node("Concat", [row_indices, col_indices], attr={'axis': 1},
                                       op_name_scope=op_name_scope).output[0]
    dense_shape = ctx.make_node("Concat", dense_shape_dims, attr={'axis': 0}, op_name_scope=op_name_scope).output[0]
    return sparse_indices, dense_shape


def ragged_nested_row_ids_to_sparse_indices(ctx, num_rows, nested_row_ids, op_name_scope):
    sparse_indices = None
    if ctx.get_dtype(num_rows) != TensorProto.INT64:
        num_rows = ctx.make_node("Cast", [num_rows], attr={'to': TensorProto.INT64}).output[0]
    num_rows = GraphBuilder(ctx).make_unsqueeze({"data": num_rows, "axes": [0]})
    dense_shape_dims = [num_rows]
    for row_ids in nested_row_ids:
        if ctx.get_dtype(row_ids) != TensorProto.INT64:
            row_ids = ctx.make_node("Cast", [row_ids], attr={'to': TensorProto.INT64}).output[0]
        num_cols, col_indices = ragged_row_ids_to_sparse_indices(ctx, row_ids)
        dense_shape_dims.append(num_cols)
        if sparse_indices is None:
            row_indices = GraphBuilder(ctx).make_unsqueeze({"data": row_ids, "axes": [1]})
        else:
            row_indices = ctx.make_node("Gather", [sparse_indices, row_ids]).output[0]
        col_indices = GraphBuilder(ctx).make_unsqueeze({"data": col_indices, "axes": [1]})
        sparse_indices = ctx.make_node("Concat", [row_indices, col_indices], attr={'axis': 1},
                                       op_name_scope=op_name_scope).output[0]
    dense_shape = ctx.make_node("Concat", dense_shape_dims, attr={'axis': 0}, op_name_scope=op_name_scope).output[0]
    return sparse_indices, dense_shape


@tf_op("RaggedTensorToSparse")
class RaggedTensorToSparse:
    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # https://www.tensorflow.org/guide/ragged_tensor#multiple_ragged_dimensions
        dense_values = node.input[-1]
        nested_splits = node.input[:-1]
        err_msg2 = "RaggedTensorToSparse conversion only supports tensors with no dense dimensions"
        utils.make_sure(ctx.get_rank(dense_values) in [None, 1], err_msg2)
        sparse_indices, dense_shape = ragged_nested_splits_to_sparse_indices(ctx, nested_splits, node.name)
        ctx.replace_all_inputs(node.output[0], sparse_indices)
        ctx.replace_all_inputs(node.output[1], dense_values)
        ctx.replace_all_inputs(node.output[2], dense_shape)
        ctx.remove_node(node.name)


@tf_op("RaggedTensorToTensor")
class RaggedTensorToTensor:
    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        shape, values, default_value, *row_partition_tensors = node.input
        has_uniform_dims = ctx.get_rank(values) != 1
        partition_types = node.get_attr_value("row_partition_types")
        layout_type = None
        if len(partition_types) >= 2 and partition_types[0] == b'FIRST_DIM_SIZE' and \
            all(t == b'VALUE_ROWIDS' for t in partition_types[1:]):
            layout_type = 'VALUE_ROWIDS'
        elif all(t == b'ROW_SPLITS' for t in partition_types):
            layout_type = 'ROW_SPLITS'
        error_msg = "Only ROW_SPLITS partition and VALUE_ROWIDS types supported for RaggedTensorToTensor. types: %r"

        if layout_type == 'ROW_SPLITS':
            nested_splits = row_partition_tensors
            n_dims = len(nested_splits) + 1
            sparse_indices, dense_shape = ragged_nested_splits_to_sparse_indices(ctx, nested_splits, node.name)
        else:
            utils.make_sure(layout_type == 'VALUE_ROWIDS', error_msg, partition_types)
            first_dim = row_partition_tensors[0]
            row_ids = row_partition_tensors[1:]
            n_dims = len(row_ids) + 1
            sparse_indices, dense_shape = ragged_nested_row_ids_to_sparse_indices(ctx, first_dim, row_ids, node.name)

        # A shape of rank 0 means the natural shape should be used.
        if ctx.get_rank(shape) != 0:
            if ctx.get_dtype(shape) != TensorProto.INT64:
                shape = ctx.make_node("Cast", [shape], attr={'to': TensorProto.INT64}).output[0]
            if has_uniform_dims:
                const_zero_unsq = ctx.make_const(utils.make_name("const_zero"), np.array([0], dtype=np.int64)).output[0]
                const_n_unsq = ctx.make_const(utils.make_name("const_num_dims"),
                                              np.array([n_dims], dtype=np.int64)).output[0]
                shape = GraphBuilder(ctx).make_slice(
                    {'data': shape, 'starts': const_zero_unsq, 'ends': const_n_unsq, 'axes': [0]})
            const_zero_int64 = ctx.make_const(utils.make_name("const_zero"), np.array(0, dtype=np.int64)).output[0]
            unspec_dims = ctx.make_node("Less", [shape, const_zero_int64]).output[0]
            out_shape = ctx.make_node("Where", [unspec_dims, dense_shape, shape]).output[0]
            out_shape_unsq = GraphBuilder(ctx).make_unsqueeze({'data': out_shape, 'axes': [0]})
            amt_idx_in_bounds = ctx.make_node("Sub", [out_shape_unsq, sparse_indices]).output[0]
            amt_in_bounds_flat = GraphBuilder(ctx).make_reduce_min({"data": amt_idx_in_bounds, "axes": [1],
                                                                    "keepdims": False})
            idx_in_bounds = ctx.make_node("Greater", [amt_in_bounds_flat, const_zero_int64]).output[0]
            sparse_indices = ctx.make_node("Compress", [sparse_indices, idx_in_bounds], attr={'axis': 0}).output[0]
            values = ctx.make_node("Compress", [values, idx_in_bounds], attr={'axis': 0}).output[0]
        else:
            out_shape = dense_shape

        if has_uniform_dims:
            values_shape = ctx.make_node("Shape", [values]).output[0]
            const_one_unsq = ctx.make_const(utils.make_name("const_one"), np.array([1], dtype=np.int64)).output[0]
            max_int64 = np.array([utils.get_max_value(np.int64)], dtype=np.int64)
            const_max_val_unsq = ctx.make_const(utils.make_name("max_int"), max_int64).output[0]
            uniform_dims = GraphBuilder(ctx).make_slice(
                {'data': values_shape, 'starts': const_one_unsq, 'ends': const_max_val_unsq, 'axes': [0]})
            out_shape = ctx.make_node("Concat", [out_shape, uniform_dims], attr={'axis': 0}).output[0]

        expand_node = ctx.make_node("Expand", [default_value, out_shape])
        node.type = "ScatterND"
        ctx.replace_inputs(node, [expand_node.output[0], sparse_indices, values])


@tf_op("RaggedRange")
class RaggedRange:
    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        starts, limits, deltas = node.input
        data_dtype = ctx.get_dtype(starts)
        data_np_dtype = utils.map_onnx_to_numpy_type(data_dtype)
        data_is_float = np.dtype(data_np_dtype).kind == 'f'

        if data_is_float:
            sub_node = ctx.make_node("Sub", [limits, starts]).output[0]
            div_node = ctx.make_node("Div", [sub_node, deltas]).output[0]
            ceil_node = ctx.make_node("Ceil", [div_node]).output[0]
            row_lens = ctx.make_node("Cast", [ceil_node], attr={'to': TensorProto.INT64}).output[0]

        else:
            # compute ceil(a/b) with ints
            starts_cast = ctx.make_node("Cast", [starts], attr={'to': TensorProto.INT64}).output[0]
            limits_cast = ctx.make_node("Cast", [limits], attr={'to': TensorProto.INT64}).output[0]
            deltas_cast = ctx.make_node("Cast", [deltas], attr={'to': TensorProto.INT64}).output[0]
            sub_node = ctx.make_node("Sub", [limits_cast, starts_cast]).output[0]
            div_node = ctx.make_node("Div", [sub_node, deltas_cast]).output[0]
            mul_node = ctx.make_node("Mul", [div_node, deltas_cast]).output[0]
            eq_node = ctx.make_node("Equal", [mul_node, sub_node]).output[0]
            ne_node = ctx.make_node("Not", [eq_node]).output[0]
            # we want to round up if it isn't evenly divisible
            offset = ctx.make_node("Cast", [ne_node], attr={'to': TensorProto.INT64}).output[0]
            row_lens = ctx.make_node("Add", [div_node, offset]).output[0]

        const_zero_int64 = ctx.make_const(utils.make_name("const_zero"), np.array(0, dtype=np.int64)).output[0]
        if ctx.opset <= 11:
            const_zero_double = ctx.make_const(utils.make_name("const_zero"), np.array(0, dtype=np.float64)).output[0]
            row_lens = ctx.make_node("Cast", [row_lens], attr={'to': TensorProto.DOUBLE}).output[0]
            row_lens = ctx.make_node("Max", [row_lens, const_zero_double]).output[0]
            row_lens = ctx.make_node("Cast", [row_lens], attr={'to': TensorProto.INT64}).output[0]
        else:
            row_lens = ctx.make_node("Max", [row_lens, const_zero_int64]).output[0]

        const_zero_list = ctx.make_const(utils.make_name("const_zero_list"), np.array([0], dtype=np.int64)).output[0]

        num_rows, _, row_indices, col_indices = ragged_lengths_to_sparse_indices(ctx, row_lens)

        split_ends = ctx.make_node("CumSum", [row_lens, const_zero_int64]).output[0]
        splits_out = ctx.make_node("Concat", [const_zero_list, split_ends], attr={'axis': 0}).output[0]
        col_indices_cast = ctx.make_node("Cast", [col_indices], attr={'to': data_dtype}).output[0]

        if ctx.get_rank(starts) != 1:
            starts = ctx.make_node("Expand", [starts, num_rows]).output[0]

        if ctx.get_rank(deltas) != 1:
            deltas = ctx.make_node("Expand", [deltas, num_rows]).output[0]

        gather_starts = ctx.make_node("Gather", [starts, row_indices]).output[0]
        gather_deltas = ctx.make_node("Gather", [deltas, row_indices]).output[0]

        mul_node = ctx.make_node("Mul", [col_indices_cast, gather_deltas], op_name_scope=node.name).output[0]
        dense_vals_out = ctx.make_node("Add", [gather_starts, mul_node], op_name_scope=node.name).output[0]

        ctx.replace_all_inputs(node.output[0], splits_out)
        ctx.replace_all_inputs(node.output[1], dense_vals_out)
        ctx.remove_node(node.name)


@tf_op("RaggedGather")
class RaggedGather:
    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        *params_nested_splits, params_dense_values, indices = node.input
        inp_ragged_rank = node.get_attr_value("PARAMS_RAGGED_RANK")
        out_ragged_rank = node.get_attr_value("OUTPUT_RAGGED_RANK")
        err_msg = "RaggedGather conversion only supports ragged rank of 1"
        utils.make_sure(inp_ragged_rank == 1 and out_ragged_rank == 1 and len(params_nested_splits) == 1, err_msg)
        splits = params_nested_splits[0]
        err_msg2 = "RaggedGather conversion only supports tensors with no dense dimensions"
        utils.make_sure(ctx.get_rank(splits) in [None, 1] and ctx.get_rank(params_dense_values) in [None, 1], err_msg2)
        splits_dtype = ctx.get_dtype(splits)

        if splits_dtype != TensorProto.INT64:
            splits_64 = ctx.make_node("Cast", [splits], attr={'to': TensorProto.INT64}).output[0]
        else:
            splits_64 = splits

        max_int64 = int(utils.get_max_value(np.int64))
        slice1 = GraphBuilder(ctx).make_slice(
            {"data": splits_64, "ends": [max_int64], "starts": [1], "axes": [0]})
        slice2 = GraphBuilder(ctx).make_slice(
            {"data": splits_64, "ends": [-1], "starts": [0], "axes": [0]})
        ragged_lens = ctx.make_node("Sub", [slice1, slice2]).output[0]

        gathered_lens = ctx.make_node("Gather", [ragged_lens, indices], op_name_scope=node.name).output[0]

        const_zero_unsq = ctx.make_const(utils.make_name("const_zero"), np.array([0], dtype=np.int64)).output[0]
        const_one_unsq = ctx.make_const(utils.make_name("const_one"), np.array([1], dtype=np.int64)).output[0]
        gathered_lens_w_zero = ctx.make_node("Concat", [const_zero_unsq, gathered_lens], attr={'axis': 0}).output[0]

        const_zero_int64 = ctx.make_const(utils.make_name("const_zero"), np.array(0, dtype=np.int64)).output[0]
        const_one_int64 = ctx.make_const(utils.make_name("const_one"), np.array(1, dtype=np.int64)).output[0]

        gathered_splits = ctx.make_node("CumSum", [gathered_lens_w_zero, const_zero_int64]).output[0]
        if splits_dtype != TensorProto.INT64:
            output_splits = ctx.make_node("Cast", [gathered_splits], attr={'to': splits_dtype}).output[0]
        else:
            output_splits = gathered_splits

        # Now that we have the splits, we just need to make the list of values.
        total_length = GraphBuilder(ctx).make_slice(
            {"data": gathered_splits, "ends": [max_int64], "starts": [-1], "axes": [0]})
        gathered_starts = ctx.make_node("Gather", [splits_64, indices], op_name_scope=node.name).output[0]
        # We disregard any length 0 segments
        non_zero_pos = ctx.make_node("Greater", [gathered_lens, const_zero_int64]).output[0]
        non_zero_lens = ctx.make_node("Compress", [gathered_lens, non_zero_pos]).output[0]
        non_zero_lens_shifted = ctx.make_node("Concat", [const_zero_unsq, non_zero_lens], attr={'axis': 0}).output[0]
        non_zero_prev_lens = GraphBuilder(ctx).make_slice(
            {"data": non_zero_lens_shifted, "ends": [-1], "starts": [0], "axes": [0]})
        non_zero_starts = ctx.make_node("Compress", [gathered_starts, non_zero_pos]).output[0]
        non_zero_splits = ctx.make_node("Compress", [gathered_splits, non_zero_pos]).output[0]

        prev_starts = GraphBuilder(ctx).make_slice(
            {"data": non_zero_starts, "ends": [-1], "starts": [0], "axes": [0]})
        prev_starts_concat = ctx.make_node("Concat", [const_one_unsq, prev_starts], attr={'axis': 0}).output[0]
        deltas = ctx.make_node("Sub", [non_zero_starts, prev_starts_concat]).output[0]
        deltas2 = ctx.make_node("Sub", [deltas, non_zero_prev_lens]).output[0]
        deltas3 = ctx.make_node("Add", [deltas2, const_one_int64]).output[0]
        one_tensor = helper.make_tensor("value", TensorProto.INT64, dims=[1], vals=[1])
        ones_of_shape = ctx.make_node("ConstantOfShape", [total_length], attr={"value": one_tensor}).output[0]
        full_deltas = ctx.make_node("ScatterElements", [ones_of_shape, non_zero_splits, deltas3], attr={'axis': 0})
        full_indices = ctx.make_node("CumSum", [full_deltas.output[0], const_zero_int64]).output[0]
        output_values = ctx.make_node("Gather", [params_dense_values, full_indices], op_name_scope=node.name).output[0]

        ctx.replace_all_inputs(node.output[0], output_splits)
        ctx.replace_all_inputs(node.output[1], output_values)
        ctx.remove_node(node.name)


@tf_op("RaggedTensorFromVariant")
class RaggedTensorFromVariant:
    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        inp = node.inputs[0]
        if inp.is_while():
            row_lengths = inp.ragged_scan_output_to_len.get(node.input[0])
            utils.make_sure(row_lengths is not None, "Couldn't find lengths for %s node %s" % (node.type, node.name))
            dense_values = ctx.make_node("ConcatFromSequence", [node.input[0]], attr={'axis': 0}).output[0]
            const_zero = ctx.make_const(utils.make_name("const_zero"), np.array(0, np.int64)).output[0]
            const_zero_unsq = ctx.make_const(utils.make_name("const_zero"), np.array([0], np.int64)).output[0]
            row_splits = ctx.make_node("CumSum", [row_lengths, const_zero]).output[0]
            row_splits_w_zero = ctx.make_node("Concat", [const_zero_unsq, row_splits], attr={'axis': 0}).output[0]
            idx_dtype = ctx.get_dtype(node.output[0])
            if idx_dtype != TensorProto.INT64:
                row_splits_w_zero = ctx.make_node("Cast", [row_splits_w_zero], attr={'to': idx_dtype}).output[0]
            ctx.replace_all_inputs(node.output[0], row_splits_w_zero)
            ctx.replace_all_inputs(node.output[1], dense_values)
            ctx.remove_node(node.name)
            return

        utils.make_sure(inp.type == "Gather", "RaggedTensorFromVariant only supported after TensorListGetItem")
        variant = inp.inputs[0]
        err_msg = "RaggedTensorFromVariant only supported if variant is a graph input"
        # Variant input will be found during loop conversion
        utils.make_sure(variant.type == "Placeholder", err_msg)
        ctx.ragged_variant_list_reads.append(node)


@tf_op("RaggedTensorToVariant")
class RaggedTensorToVariant:
    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        cons = ctx.find_output_consumers(node.output[0])
        err_msg = "RaggedTensorToVariant only supported as input/output to loops"
        utils.make_sure(len(cons) == 1, err_msg)
        if cons[0].type == "TensorListFromTensor":
            # Will be delt with in loop
            cons = ctx.find_output_consumers(cons[0].output[0])
            utils.make_sure(all(n.is_while() for n in cons), err_msg)
            return
        utils.make_sure(cons[0].type == "TensorListSetItem", err_msg)
        tensor_set_item = cons[0]
        list_output = tensor_set_item.output[0]
        cons = ctx.find_output_consumers(list_output)
        while len(cons) == 1 and cons[0].type == "Identity":
            list_output = cons[0].output[0]
            cons = ctx.find_output_consumers(list_output)
        utils.make_sure(not cons, err_msg)
        utils.make_sure(list_output in ctx.outputs, err_msg)
        err_msg2 = "RaggedTensorToVariant within loop requires RAGGED_RANK=0"
        err_msg3 = "RaggedTensorToVariant within loop requires batched_input=False"
        utils.make_sure(node.get_attr_value("RAGGED_RANK") == 0, err_msg2)
        utils.make_sure(not node.get_attr_value("batched_input"), err_msg3)
        ctx.ragged_variant_list_writes.append((ctx.outputs.index(list_output), node))


@tf_op("SparseReshape")
class SparseReshape:
    @classmethod
    def any_version(cls, opset, ctx, node, **kwargs):
        indices_inp, shape_inp, new_shape_inp = node.input

        product_curr_dims = GraphBuilder(ctx).make_reduce_prod({"data": shape_inp, "axes": [0], "keepdims": 1})
        product_new_dims = GraphBuilder(ctx).make_reduce_prod({"data": new_shape_inp, "axes": [0], "keepdims": 1})
        neg_missing_dims = ctx.make_node("Div", [product_curr_dims, product_new_dims]).output[0]
        pos_missing_dims = ctx.make_node("Neg", [neg_missing_dims]).output[0]
        zero_const = ctx.make_const(utils.make_name("cosnt_zero"), np.array(0, dtype=np.int64)).output[0]
        one_const = ctx.make_const(utils.make_name("cosnt_one"), np.array(1, dtype=np.int64)).output[0]
        unknown_dim_loc = ctx.make_node("Less", [new_shape_inp, zero_const]).output[0]

        new_shape = ctx.make_node("Where", [unknown_dim_loc, pos_missing_dims, new_shape_inp]).output[0]

        zero_tensor = helper.make_tensor("value", TensorProto.INT64, dims=[1], vals=[0])

        def cum_prod_of_vector(vector):
            shape = ctx.get_shape(vector)
            rank = shape[0] if shape is not None else -1
            if rank != -1:
                lower_tri = np.tri(rank, rank, dtype=bool)
                lower_triangular_bool = ctx.make_const(utils.make_name("lower_tri_const"), lower_tri).output[0]
            else:
                rank = ctx.make_node("Shape", [vector]).output[0]
                rank_sq = ctx.make_node("Concat", [rank, rank], attr={'axis': 0}).output[0]
                square_of_rank = ctx.make_node("ConstantOfShape", [rank_sq], attr={'value': zero_tensor}).output[0]
                identity_matrix = ctx.make_node("EyeLike", [square_of_rank]).output[0]
                lower_triangular = ctx.make_node("CumSum", [identity_matrix, zero_const]).output[0]
                lower_triangular_bool = ctx.make_node("Cast", [lower_triangular],
                                                      attr={'to': TensorProto.BOOL}).output[0]
            terms = ctx.make_node("Where", [lower_triangular_bool, one_const, vector]).output[0]
            return GraphBuilder(ctx).make_reduce_prod({"data": terms, "axes": [1], "keepdims": 0})

        cum_prod_curr_shape = cum_prod_of_vector(shape_inp)
        cum_prod_new_shape = cum_prod_of_vector(new_shape)
        cum_prod_new_concat = ctx.make_node("Concat", [product_curr_dims, cum_prod_new_shape],
                                            attr={'axis': 0}).output[0]
        pads = ctx.make_const(utils.make_name("pad_const"), np.array([0, -1], dtype=np.int64)).output[0]
        cum_prod_new_inc = ctx.make_node("Pad", [cum_prod_new_concat, pads]).output[0]

        flat_indices = ctx.make_node("MatMul", [indices_inp, cum_prod_curr_shape]).output[0]
        indices_unsqueeze = GraphBuilder(ctx).make_unsqueeze({'data': flat_indices, "axes": [1]})
        mod_indices = ctx.make_node("Mod", [indices_unsqueeze, cum_prod_new_inc], op_name_scope=node.name).output[0]
        new_indices = ctx.make_node("Div", [mod_indices, cum_prod_new_shape], op_name_scope=node.name).output[0]

        ctx.replace_all_inputs(node.output[0], new_indices)
        ctx.replace_all_inputs(node.output[1], new_shape)
        ctx.remove_node(node.name)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        cls.any_version(11, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Parameters moved to inputs for operator Squeeze, Unsqueeze.
        cls.any_version(13, ctx, node, **kwargs)


@tf_op("SparseFillEmptyRows")
class SparseFillEmptyRows:
    @classmethod
    def any_version(cls, opset, ctx, node, **kwargs):
        sparse_indices, sparse_vals, dense_shape, default_val = node.input
        utils.make_sure(len(ctx.find_output_consumers(node.output[3])) == 0,
                        "reverse_index_map output of SparseFillEmptyRows not implemented")
        axis_0_indices = GraphBuilder(ctx).make_slice({"data": sparse_indices, "ends": [1], "starts": [0], "axes": [1]})
        unique_indices = ctx.make_node("Unique", [axis_0_indices], op_name_scope=node.name).output[0]
        axis_0_len = GraphBuilder(ctx).make_slice({"data": dense_shape, "ends": [1], "starts": [0], "axes": [0]})

        true_tensor = helper.make_tensor("value", TensorProto.BOOL, dims=[1], vals=[True])
        true_of_shape = ctx.make_node("ConstantOfShape", inputs=[axis_0_len], attr={"value": true_tensor},
                                      op_name_scope=node.name).output[0]
        unique_shape = ctx.make_node("Shape", [unique_indices], op_name_scope=node.name).output[0]
        false_tensor = helper.make_tensor("value", TensorProto.BOOL, dims=[1], vals=[False])
        false_of_shape = ctx.make_node("ConstantOfShape", inputs=[unique_shape], attr={"value": false_tensor},
                                       op_name_scope=node.name).output[0]

        indicators = ctx.make_node("ScatterElements", [true_of_shape, unique_indices, false_of_shape],
                                   op_name_scope=node.name).output[0]
        zero_const = ctx.make_const(utils.make_name("zero_const"), np.array(0, dtype=np.int64)).output[0]
        one_const = ctx.make_const(utils.make_name("one_const"), np.array(1, dtype=np.int64)).output[0]

        scalar_len = GraphBuilder(ctx).make_squeeze({'data': axis_0_len, "axes": [0]}, op_name_scope=node.name)
        idx_range = ctx.make_node("Range", [zero_const, scalar_len, one_const], op_name_scope=node.name).output[0]
        new_indices = ctx.make_node("Compress", [idx_range, indicators], op_name_scope=node.name).output[0]
        new_indices_unsqueeze = GraphBuilder(ctx).make_unsqueeze(
            {'data': new_indices, 'axes': [1]}, op_name_scope=node.name)
        num_empty_rows = ctx.make_node("Shape", [new_indices], op_name_scope=node.name).output[0]
        new_values = ctx.make_node("Expand", [default_val, num_empty_rows], op_name_scope=node.name).output[0]
        indices_shape = ctx.make_node("Shape", [sparse_indices], op_name_scope=node.name).output[0]
        idx_shape = GraphBuilder(ctx).make_slice({"data": indices_shape, "ends": [2], "starts": [1], "axes": [0]})
        idx_shape_min_1 = ctx.make_node("Sub", [idx_shape, one_const], op_name_scope=node.name).output[0]

        triple_0 = ctx.make_const(utils.make_name("triple_0"), np.array([0, 0, 0], dtype=np.int64)).output[0]
        new_indices_pads = ctx.make_node("Concat", [triple_0, idx_shape_min_1], attr={"axis": 0},
                                         op_name_scope=node.name).output[0]
        new_indices_2d = ctx.make_node("Pad", [new_indices_unsqueeze, new_indices_pads],
                                       op_name_scope=node.name).output[0]

        combined_indices = ctx.make_node("Concat", [sparse_indices, new_indices_2d], attr={"axis": 0},
                                         op_name_scope=node.name).output[0]
        combined_vals = ctx.make_node("Concat", [sparse_vals, new_values], attr={"axis": 0},
                                      op_name_scope=node.name).output[0]

        # The indices will not be sorted (violates a TF requirement), but conversions for subsequent ops
        # (like SparseToDense) don't care and will work fine.  Add a TopK to sort in the future if needed.
        ctx.replace_all_inputs(node.output[0], combined_indices)
        ctx.replace_all_inputs(node.output[1], combined_vals)
        ctx.replace_all_inputs(node.output[2], indicators)

        ctx.remove_node(node.name)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        cls.any_version(11, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Parameters moved to inputs for operator Squeeze, Unsqueeze.
        cls.any_version(13, ctx, node, **kwargs)

@tf_op("DenseToDenseSetOperation")
class DenseToDenseSetOperation:
    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        inp_a, inp_b = node.input
        dtype = ctx.get_dtype(node.output[1])
        if dtype != TensorProto.INT64:
            inp_a = ctx.make_node("Cast", [inp_a], attr={'to': TensorProto.INT64}).output[0]
            inp_b = ctx.make_node("Cast", [inp_b], attr={'to': TensorProto.INT64}).output[0]
        set_op = node.get_attr_value('set_operation')
        if set_op == b'b-a':
            set_op = b'a-b'
            inp_a, inp_b = inp_b, inp_a

        one_tensor = helper.make_tensor("value", TensorProto.INT64, dims=[1], vals=[1])
        const_one = ctx.make_const(utils.make_name("const_one"), np.array(1, np.int64)).output[0]
        const_zero = ctx.make_const(utils.make_name("const_zero"), np.array(0, np.int64)).output[0]
        const_zero_unsq = ctx.make_const(utils.make_name("const_zero"), np.array([0], np.int64)).output[0]
        const_neg_one_unsq = ctx.make_const(utils.make_name("const_neg_one"), np.array([-1], np.int64)).output[0]
        max_int64 = int(utils.get_max_value(np.int64))
        const_two = ctx.make_const(utils.make_name("const_two"), np.array(2, np.int64)).output[0]

        def concat_indices(tensor):
            shape = ctx.make_node("Shape", [tensor]).output[0]
            tensor_flat = ctx.make_node("Reshape", [tensor, const_neg_one_unsq]).output[0]
            tensor_flat_unsq = GraphBuilder(ctx).make_unsqueeze({'data': tensor_flat, 'axes': [1]})
            ones_of_shape = ctx.make_node("ConstantOfShape", [shape], attr={'value': one_tensor}).output[0]
            indices = ctx.make_node("NonZero", [ones_of_shape]).output[0]
            sliced_indices = GraphBuilder(ctx).make_slice({'data': indices, 'starts': [0], 'ends': [-1], 'axes': [0]})
            sliced_indices_trans = ctx.make_node("Transpose", [sliced_indices], attr={'perm': [1, 0]}).output[0]
            return ctx.make_node("Concat", [sliced_indices_trans, tensor_flat_unsq], attr={'axis': 1}).output[0]

        if set_op == b'union':
            combined = ctx.make_node("Concat", [inp_a, inp_b], attr={'axis': -1}).output[0]
            shape = ctx.make_node("Shape", [combined]).output[0]
            shape_prefix = GraphBuilder(ctx).make_slice({'data': shape, 'starts': [0], 'ends': [-1], 'axes': [0]})
            indices_and_vals = concat_indices(combined)
            res_idx_and_vals = ctx.make_node("Unique", [indices_and_vals], attr={'axis': 0}).output[0]
        else:
            shape = ctx.make_node("Shape", [inp_a]).output[0]
            shape_prefix = GraphBuilder(ctx).make_slice({'data': shape, 'starts': [0], 'ends': [-1], 'axes': [0]})
            a_idx_and_vals = concat_indices(inp_a)
            b_idx_and_vals = concat_indices(inp_b)
            a_unique = ctx.make_node("Unique", [a_idx_and_vals], attr={'axis': 0}).output[0]
            b_unique = ctx.make_node("Unique", [b_idx_and_vals], attr={'axis': 0}).output[0]
            if set_op == b'intersection':
                combined = ctx.make_node("Concat", [a_unique, b_unique], attr={'axis': 0}).output[0]
                desired_cnt = const_two
            else:
                utils.make_sure(set_op == b'a-b', "Unsupported set operation: %s", set_op)
                combined = ctx.make_node("Concat", [a_unique, b_unique, b_unique], attr={'axis': 0}).output[0]
                # cnt will be 1 if and only if element is in only set A
                desired_cnt = const_one
            unique_rows, _, _, row_cnts = ctx.make_node("Unique", [combined], attr={'axis': 0}, output_count=4).output
            keep = ctx.make_node("Equal", [row_cnts, desired_cnt]).output[0]
            compress_shape = None
            rows_shape = ctx.get_shape(unique_rows)
            if rows_shape is not None:
                compress_shape = rows_shape.copy()
                compress_shape[0] = -1
            res_idx_and_vals = ctx.make_node("Compress", [unique_rows, keep], attr={'axis': 0},
                                             shapes=[compress_shape]).output[0]

        merged_indices = GraphBuilder(ctx).make_slice(
            {'data': res_idx_and_vals, 'starts': [0], 'ends': [-1], 'axes': [1]})
        merged_values = GraphBuilder(ctx).make_slice(
            {'data': res_idx_and_vals, 'starts': [-1], 'ends': [max_int64], 'axes': [1]})
        merged_values_sq = GraphBuilder(ctx).make_squeeze({'data': merged_values, 'axes': [1]})
        merged_values_sq_cast = ctx.make_node("Cast", [merged_values_sq], attr={'to': dtype}).output[0]

        _, idx_loc, _, idx_cnts, = ctx.make_node("Unique", [merged_indices], attr={'axis': 0},
                                                 output_count=4, op_name_scope=node.name).output

        max_cnt = GraphBuilder(ctx).make_reduce_max({"data": idx_cnts, "axes": [0], "keepdims": True})
        final_shape = ctx.make_node("Concat", [shape_prefix, max_cnt], attr={'axis': 0}).output[0]
        one_minus_cnts = ctx.make_node("Sub", [const_one, idx_cnts]).output[0]
        cnts_sliced = GraphBuilder(ctx).make_slice(
            {"data": one_minus_cnts, "starts": [0], "ends": [-1], "axes": [0]})
        cnts_shifted = ctx.make_node("Concat", [const_zero_unsq, cnts_sliced], attr={'axis': 0}).output[0]
        values_shape = ctx.make_node("Shape", [merged_values_sq_cast]).output[0]
        ones_of_shape = ctx.make_node("ConstantOfShape", [values_shape], attr={'value': one_tensor}).output[0]
        idx_deltas = ctx.make_node("ScatterElements", [ones_of_shape, idx_loc, cnts_shifted]).output[0]
        last_dim_idx = ctx.make_node("CumSum", [idx_deltas, const_zero]).output[0]
        last_dim_idx_unsq = GraphBuilder(ctx).make_unsqueeze({"data": last_dim_idx, "axes": [1]})
        full_indices = ctx.make_node("Concat", [merged_indices, last_dim_idx_unsq], attr={'axis': 1}).output[0]

        ctx.replace_all_inputs(node.output[0], full_indices)
        ctx.replace_all_inputs(node.output[1], merged_values_sq_cast)
        ctx.replace_all_inputs(node.output[2], final_shape)

@tf_op("DynamicPartition")
class DynamicPartition:
    @classmethod
    def any_version(cls, opset, ctx, node, **kwargs):
        # For desired behavior, see diagram: https://www.tensorflow.org/api_docs/python/tf/raw_ops/DynamicPartition
        data_inp = node.input[0]
        partition_inp = node.input[1]
        partition_shape = ctx.get_shape(partition_inp)
        num_partitions = node.get_attr_value('num_partitions')
        utils.make_sure(partition_shape is not None, "DynamicPartition requires known rank")
        utils.make_sure(len(partition_shape) == 1, "DynamicPartition only implemented for partitions of rank 1")
        # Put partitions into OneHot format
        range_val = np.arange(num_partitions, dtype=np.int32).reshape([num_partitions, 1])
        range_const = ctx.make_const(utils.make_name('range_const'), range_val)
        equal_node = ctx.make_node("Equal", [partition_inp, range_const.output[0]])
        # Cast bool to int since ORT doesn't implement Split on bool.
        equal_int32 = ctx.make_node("Cast", [equal_node.output[0]], attr={"to": TensorProto.INT32})
        split_node = ctx.make_node("Split", [equal_int32.output[0]], output_count=num_partitions,
                                   attr={'axis': 0, 'num_outputs': num_partitions})
        for i in range(num_partitions):
            cond_bools = ctx.make_node("Cast", [split_node.output[i]], attr={"to": TensorProto.BOOL})
            squeeze_node = GraphBuilder(ctx).make_squeeze({'data': cond_bools.output[0], "axes": [0]}, return_node=True)
            compress_node = ctx.make_node("Compress", [data_inp, squeeze_node.output[0]], attr={'axis': 0})
            ctx.replace_all_inputs(node.output[i], compress_node.output[0])
            ctx.copy_dtype(node.output[i], compress_node.output[0])
            ctx.copy_shape(node.output[i], compress_node.output[0])
        ctx.remove_node(node.name)

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        cls.any_version(9, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Parameters moved to inputs for operator Squeeze, Unsqueeze.
        cls.any_version(13, ctx, node, **kwargs)


@tf_op(["DynamicStitch", "ParallelDynamicStitch"])
class DynamicStitch:
    @classmethod
    def any_version(cls, opset, ctx, node, **kwargs):
        num_partitions = len(node.input) // 2
        index_inputs = node.input[:num_partitions]
        data_inputs = node.input[num_partitions:]
        index_shapes = [ctx.get_shape(inp) for inp in index_inputs]
        data_shapes = [ctx.get_shape(inp) for inp in data_inputs]
        utils.make_sure(all(s is not None and len(s) == 1 for s in index_shapes),
                        "DynamicStitch only implemented for index tensors of rank 1")
        utils.make_sure(all(s is not None for s in data_shapes), "DynamicStitch requires data tensors of known rank")
        data_rank = len(data_shapes[0])
        dtype = ctx.get_dtype(node.output[0])
        concat_indices = ctx.make_node("Concat", index_inputs, attr={'axis': 0})
        concat_indices_int64 = ctx.make_node("Cast", [concat_indices.output[0]], attr={"to": TensorProto.INT64})

        concat_data = ctx.make_node("Concat", data_inputs, attr={'axis': 0})

        data_shape = ctx.make_node("Shape", [concat_data.output[0]])
        unsqueezed_indices = concat_indices_int64
        if data_rank > 1:
            unsqueeze_axes = list(range(1, data_rank))
            unsqueezed_indices = GraphBuilder(ctx).make_unsqueeze(
                {'data': concat_indices_int64.output[0], "axes": unsqueeze_axes}, return_node=True)
        expanded_indices = ctx.make_node("Expand", [unsqueezed_indices.output[0], data_shape.output[0]])

        zero_tensor = helper.make_tensor("value", dtype, dims=[1], vals=[0])
        zeros_of_shape = ctx.make_node("ConstantOfShape", [data_shape.output[0]], attr={"value": zero_tensor})

        name = node.name
        outputs = node.output
        ctx.remove_node(node.name)
        ctx.make_node("ScatterElements",
                      [zeros_of_shape.output[0], expanded_indices.output[0], concat_data.output[0]],
                      name=name,
                      outputs=outputs,
                      attr={'axis': 0})

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        cls.any_version(10, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Parameters moved to inputs for operator Squeeze, Unsqueeze.
        cls.any_version(13, ctx, node, **kwargs)


@tf_op("MatrixDiagPart")
class MatrixDiagPart:
    @classmethod
    def any_version(cls, opset, ctx, node, **kwargs):
        # MatrixDiagPart by slice and gather
        minus_two_one, minus_two, minus_one, zeo, zeo_zeo, one, two, two_one = \
            [n.output[0] for n in ctx.make_consts([[-2, -1], [-2], [-1], [0], [0, 0], [1], [2], [2, 1]])]
        zeo_, one_ = [n.output[0] for n in ctx.make_consts([0, 1])]

        input_shape = ctx.make_node('Shape', [node.input[0]])
        input_shape_size = ctx.make_node('Shape', [input_shape.output[0]])
        matrice_shape = ctx.make_node('Slice',
                                      [input_shape.output[0], minus_two, input_shape_size.output[0]])
        matrice_shape_float = ctx.make_node('Cast', [matrice_shape.output[0]], attr={'to': TensorProto.FLOAT})
        matrice_shape_float_x = ctx.make_node('Slice', [matrice_shape_float.output[0], zeo, one])
        matrice_shape_float_y = ctx.make_node('Slice',
                                              [matrice_shape_float.output[0], one, two])
        min_matrice_dim_float = ctx.make_node('Min', [matrice_shape_float_x.output[0], matrice_shape_float_y.output[0]])
        min_matrice_dim = ctx.make_node('Cast', [min_matrice_dim_float.output[0]], attr={'to': TensorProto.INT64})
        double_matrice_dim = ctx.make_node('Concat', [min_matrice_dim.output[0], min_matrice_dim.output[0]],
                                           attr={'axis': -1})
        sliced_input = ctx.make_node('Slice', [node.input[0], zeo_zeo, double_matrice_dim.output[0], two_one])
        sliced_input_shape = ctx.make_node('Shape', [sliced_input.output[0]])
        sliced_input_shape_half = ctx.make_node('Slice', [sliced_input_shape.output[0], zeo,
                                                          minus_one])
        sliced_input_shape_new = ctx.make_node('Concat', [sliced_input_shape_half.output[0], one],
                                               attr={'axis': -1})
        gb = GraphBuilder(ctx)
        min_matrice_dim_ = gb.make_squeeze(
            {'data': min_matrice_dim.output[0], "axes": [0]}, return_node=True)
        matrice_range = ctx.make_node('Range', [zeo_, min_matrice_dim_.output[0], one_])
        unsqueezed_matrice_range = gb.make_unsqueeze(
            {'data': matrice_range.output[0], "axes": [-1]}, return_node=True)
        expanded_range = ctx.make_node('Expand', [unsqueezed_matrice_range.output[0], sliced_input_shape_new.output[0]])
        gathered_result = ctx.make_node('GatherElements', [sliced_input.output[0], expanded_range.output[0]],
                                        attr={'axis': -1})
        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        gb.make_squeeze(
            {'data': gathered_result.output[0], "axes": [-1], 'outputs': node.output}, return_node=True,
            name=node.name, shapes=shapes, dtypes=dtypes)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        cls.any_version(11, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Parameters moved to inputs for operator Squeeze, Unsqueeze.
        cls.any_version(13, ctx, node, **kwargs)


@tf_op(["MatrixDiagPartV2", "MatrixDiagPartV3"])
class MatrixDiagPartV2V3:
    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # assemble MatrixDiagPart V2&V3 by looping k diagonals with proper pads
        minus_two, minus_one, zeo, one, two = \
            [n.output[0] for n in ctx.make_consts([[-2], [-1], [0], [1], [2]])]

        def normalize():
            raw_k = ctx.make_node('Cast', [node.input[1]], attr={'to': TensorProto.INT64}).output[0]
            return ctx.make_node('Reshape', [raw_k, minus_one]).output[0]

        input_tensor = node.input[0]
        k = normalize()
        padding = node.input[2]
        align = 'LEFT_LEFT'
        if node.op.op_type == 'MatrixDiagPartV3':
            align = node.get_attr_str('align') if 'align' in node.attr else 'LEFT_RIGHT'
        input_rank = len(ctx.get_shape(input_tensor))
        raw_input_shape = [-1] * input_rank
        per_loop_shape = raw_input_shape[:-1]
        raw_output_shape = raw_input_shape[:-2] + [-1]
        loop_output_shape = raw_output_shape + [-1]
        ctx.set_shape(node.output[0], raw_output_shape)
        for out in ctx.find_output_consumers(node.output[0]):
            if out.op.op_type == 'Identity':
                ctx.set_shape(out.output[0], raw_output_shape)

        # prepare new_shape of input
        input_shape = ctx.make_node('Shape', [input_tensor])
        shape_input_shape = ctx.make_node('Shape', [input_shape.output[0]])
        matrix_shape = ctx.make_node('Slice',
                                     [input_shape.output[0], minus_two, shape_input_shape.output[0]])
        min_dim = ctx.make_node('ReduceMin', [matrix_shape.output[0]])
        input_depth = ctx.make_node('Slice', [matrix_shape.output[0], minus_two, minus_one])
        input_width = ctx.make_node('Slice', [matrix_shape.output[0], minus_one, two])
        temp_shape = ctx.make_node('Concat', [minus_one, matrix_shape.output[0]], attr={'axis': 0})
        temp_input = ctx.make_node('Reshape', [input_tensor, temp_shape.output[0]])
        temp_transposed = ctx.make_node('Transpose', [temp_input.output[0]], attr={'perm': [0, 2, 1]})
        half_shape = ctx.make_node('Slice', [input_shape.output[0], zeo, minus_two])
        new_shape = ctx.make_node('Concat', [half_shape.output[0], input_width.output[0], input_depth.output[0]],
                                  attr={'axis': 0})
        # define body graph for main loop
        k_shape = ctx.make_node('Shape', [k])
        k_start = ctx.make_node('Slice', [k, zeo, one])
        k_end = ctx.make_node('Slice', [k, minus_one, k_shape.output[0]])
        raw_total_k = ctx.make_node('Sub', [k_end.output[0], k_start.output[0]])
        total_k = ctx.make_node('Add', [raw_total_k.output[0], one])
        trip_name = utils.make_name(node.name + "_i")
        cond_name = utils.make_name(node.name + "_cond")
        body_graph = ctx.create_new_graph_with_same_config()
        body_graph.add_graph_input(trip_name, TensorProto.INT64, [1])
        body_graph.add_graph_input(cond_name, TensorProto.BOOL, [])
        body_graph.parent_graph = ctx
        # identity of input
        identity_input_graph = body_graph.create_new_graph_with_same_config()
        identity_input_graph.parent_graph = body_graph
        identity_input = identity_input_graph.make_node('Identity', [input_tensor])
        identity_input_graph.add_graph_output(identity_input.output[0], ctx.get_dtype(node.input[0]), raw_input_shape)
        # transposed input
        transposed_input_graph = body_graph.create_new_graph_with_same_config()
        transposed_input_graph.parent_graph = body_graph
        next_shape = transposed_input_graph.make_node('Concat', [half_shape.output[0], input_width.output[0],
                                                                 input_depth.output[0]], attr={'axis': 0})
        transposed_input = transposed_input_graph.make_node('Reshape',
                                                            [temp_transposed.output[0], next_shape.output[0]])
        transposed_input_graph.add_graph_output(transposed_input.output[0], ctx.get_dtype(node.input[0]),
                                                raw_input_shape)
        # compute current k of the loop
        current_k = body_graph.make_node('Sub', [k_end.output[0], trip_name])
        is_k_noneg = body_graph.make_node('Greater', [current_k.output[0], minus_one])
        branches = {'then_branch': identity_input_graph, 'else_branch': transposed_input_graph}
        processed_input = body_graph.make_node('If', [is_k_noneg.output[0]], branches=branches)
        processed_shape = body_graph.make_node('Shape', [processed_input.output[0]])
        shape_processed_shape = body_graph.make_node('Shape', [processed_shape.output[0]])
        new_depth = body_graph.make_node('Slice',
                                         [processed_shape.output[0], minus_two, minus_one])
        new_width = body_graph.make_node('Slice', [processed_shape.output[0], minus_one,
                                                   shape_processed_shape.output[0]])
        abs_k = body_graph.make_node('Abs', [current_k.output[0]])

        range_k = body_graph.make_node('Range', [abs_k.output[0], new_width.output[0], one],
                                       domain="com.microsoft", dtypes=[TensorProto.INT64])
        sliced_range = body_graph.make_node('Slice', [range_k.output[0], zeo, new_depth.output[0]])
        sliced_shape = body_graph.make_node('Shape', [sliced_range.output[0]])
        pad_length = body_graph.make_node('Sub', [new_depth.output[0], sliced_shape.output[0]])
        pad_length_2 = body_graph.make_node('Concat', [zeo, pad_length.output[0]], attr={'axis': 0})
        padded_range = body_graph.make_node('Pad', [sliced_range.output[0], pad_length_2.output[0]])
        # opset == 11, no need to change unsqueeze
        unsqueezed_range = GraphBuilder(body_graph).make_unsqueeze(
            {'data': padded_range.output[0], 'axes': [1]}, return_node=True)
        half_shape_x = body_graph.make_node('Slice',
                                            [new_shape.output[0], zeo, minus_two])
        shape_range = body_graph.make_node('Shape', [unsqueezed_range.output[0]])
        full_shape = body_graph.make_node('Concat', [half_shape_x.output[0], shape_range.output[0]], attr={'axis': 0})
        expanded_range = body_graph.make_node('Expand', [unsqueezed_range.output[0], full_shape.output[0]])
        gathered_input = body_graph.make_node('GatherElements', [processed_input.output[0], expanded_range.output[0]],
                                              attr={'axis': -1})
        squeezed_input = GraphBuilder(body_graph).make_squeeze(
            {'data': gathered_input.output[0], 'axes': [-1]}, return_node=True)
        left_width = body_graph.make_node('Sub', [new_width.output[0], abs_k.output[0]])
        dims = body_graph.make_node('Concat', [left_width.output[0], new_depth.output[0]], attr={'axis': 0})
        valid_dim = body_graph.make_node('ReduceMin', [dims.output[0]])
        raw_output = body_graph.make_node('Slice', [squeezed_input.output[0], zeo, valid_dim.output[0],
                                                    minus_one])
        gap_output = body_graph.make_node('Sub', [min_dim.output[0], valid_dim.output[0]])
        gaps = body_graph.make_node('Concat', [zeo, gap_output.output[0]], attr={'axis': 0})
        processed_gap = body_graph.make_node('ReduceMax', [gaps.output[0]])
        pad_zero = body_graph.make_node('Mul', [new_shape.output[0], zeo])
        sliced_zero = body_graph.make_node('Slice', [pad_zero.output[0], zeo, minus_two])
        # gap_pos_k_graph
        gap_pos_k_graph = body_graph.create_new_graph_with_same_config()
        gap_pos_k_graph.parent_graph = body_graph
        gap_pos_k = gap_pos_k_graph.make_node('Concat', [zeo,
                                                         processed_gap.output[0]],
                                              attr={'axis': 0}) \
            if align.startswith('LEFT') \
            else gap_pos_k_graph.make_node('Concat', [processed_gap.output[0],
                                                      zeo],
                                           attr={'axis': 0})
        gap_pos_k_graph.add_graph_output(gap_pos_k.output[0], TensorProto.INT64, [-1])
        # gap_neg_k_graph
        gap_neg_k_graph = body_graph.create_new_graph_with_same_config()
        gap_neg_k_graph.parent_graph = body_graph
        gap_neg_k = gap_neg_k_graph.make_node('Concat', [zeo,
                                                         processed_gap.output[0]],
                                              attr={'axis': 0}) \
            if align.endswith('LEFT') \
            else gap_neg_k_graph.make_node('Concat', [processed_gap.output[0],
                                                      zeo],
                                           attr={'axis': 0})
        gap_neg_k_graph.add_graph_output(gap_neg_k.output[0], TensorProto.INT64, [-1])
        # pad output with gap
        branches = {"then_branch": gap_pos_k_graph, "else_branch": gap_neg_k_graph}
        gap_k = body_graph.make_node('If', [is_k_noneg.output[0]], branches=branches)
        gap_left = body_graph.make_node('Slice', [gap_k.output[0], zeo, one])
        gap_right = body_graph.make_node('Slice', [gap_k.output[0], one, two])
        gap_all = body_graph.make_node('Concat', [sliced_zero.output[0], gap_left.output[0], sliced_zero.output[0],
                                                  gap_right.output[0]], attr={'axis': 0})
        padded_output = body_graph.make_node('Pad', [raw_output.output[0], gap_all.output[0], padding])
        cond_output = body_graph.make_node('Identity', [cond_name])
        body_graph.add_graph_output(cond_output.output[0], TensorProto.BOOL, [])
        body_graph.add_graph_output(padded_output.output[0], ctx.get_dtype(node.input[0]), per_loop_shape)
        body_graph.add_graph_output(gap_k.output[0], TensorProto.INT64, [-1])
        # make loop
        cond_const = ctx.make_const(utils.make_name("cond"), np.ones((), dtype=bool))
        branches = {"body": body_graph}
        main_loop = ctx.make_node('Loop', [total_k.output[0], cond_const.output[0]], output_count=2, branches=branches)
        # reshape output
        next_padded_shape = ctx.make_node('Concat', [total_k.output[0], minus_one, min_dim.output[0]],
                                          attr={'axis': 0})
        reshaped_padded = ctx.make_node('Reshape', [main_loop.output[0], next_padded_shape.output[0]])
        transposed_padded = ctx.make_node('Transpose', [reshaped_padded.output[0]], attr={'perm': [1, 0, 2]})
        output_shape = ctx.make_node('Concat', [half_shape.output[0], total_k.output[0], minus_one],
                                     attr={'axis': 0})
        reshaped_output = ctx.make_node('Reshape', [transposed_padded.output[0], output_shape.output[0]])
        # compute pads
        left_pads = ctx.make_node('Slice', [main_loop.output[1], minus_two, minus_one,
                                            minus_one])
        flattened_left_pads = ctx.make_node('Reshape', [left_pads.output[0], minus_one])
        min_left_pads = ctx.make_node('ReduceMin', [flattened_left_pads.output[0]])
        right_pads = ctx.make_node('Slice', [main_loop.output[1], minus_one, two,
                                             minus_one])
        flattened_right_pads = ctx.make_node('Reshape', [right_pads.output[0], minus_one])
        min_right_pads = ctx.make_node('ReduceMin', [flattened_right_pads.output[0]])
        # trim left pads
        identity_left_sliced_graph = ctx.create_new_graph_with_same_config()
        identity_left_sliced_graph.parent_graph = ctx
        identity_left_sliced = identity_left_sliced_graph.make_node('Identity', [reshaped_output.output[0]])
        identity_left_sliced_graph.add_graph_output(identity_left_sliced.output[0], ctx.get_dtype(node.input[0]),
                                                    loop_output_shape)
        output_left_sliced_graph = ctx.create_new_graph_with_same_config()
        output_left_sliced_graph.parent_graph = ctx
        output_left_sliced = output_left_sliced_graph.make_node('Slice',
                                                                [reshaped_output.output[0], min_left_pads.output[0],
                                                                 min_dim.output[0], minus_one])
        output_left_sliced_graph.add_graph_output(output_left_sliced.output[0], ctx.get_dtype(node.input[0]),
                                                  loop_output_shape)
        left_pads_greater_than_zero = ctx.make_node('Greater', [min_left_pads.output[0], zeo])
        branches = {"then_branch": output_left_sliced_graph, "else_branch": identity_left_sliced_graph}
        final_output_left_sliced = ctx.make_node('If', [left_pads_greater_than_zero.output[0]], branches=branches)
        # trim right pads
        valid_right_dim = ctx.make_node('Sub', [min_dim.output[0], min_right_pads.output[0]])
        identity_right_sliced_graph = ctx.create_new_graph_with_same_config()
        identity_right_sliced_graph.parent_graph = ctx
        identity_right_sliced = identity_right_sliced_graph.make_node('Identity', [final_output_left_sliced.output[0]])
        identity_right_sliced_graph.add_graph_output(identity_right_sliced.output[0], ctx.get_dtype(node.input[0]),
                                                     loop_output_shape)
        output_right_sliced_graph = ctx.create_new_graph_with_same_config()
        output_right_sliced_graph.parent_graph = ctx
        output_right_sliced = output_right_sliced_graph.make_node('Slice', [final_output_left_sliced.output[0],
                                                                            zeo,
                                                                            valid_right_dim.output[0],
                                                                            minus_one])
        output_right_sliced_graph.add_graph_output(output_right_sliced.output[0], ctx.get_dtype(node.input[0]),
                                                   loop_output_shape)
        right_dim_greater_than_valid = ctx.make_node('Greater', [min_dim.output[0], valid_right_dim.output[0]])
        branches = {"then_branch": output_right_sliced_graph, "else_branch": identity_right_sliced_graph}
        final_output_right_sliced = ctx.make_node('If', [right_dim_greater_than_valid.output[0]], branches=branches)
        # squeeze output
        latest_shape = ctx.make_node('Shape', [final_output_right_sliced.output[0]])
        latest_depth = ctx.make_node('Slice',
                                     [latest_shape.output[0], minus_two, minus_one])
        need_squeeze = ctx.make_node('Equal', [latest_depth.output[0], one])
        identity_sliced_graph = ctx.create_new_graph_with_same_config()
        identity_sliced_graph.parent_graph = ctx
        identity_sliced = identity_sliced_graph.make_node('Identity', [final_output_right_sliced.output[0]])
        identity_sliced_graph.add_graph_output(identity_sliced.output[0], ctx.get_dtype(node.input[0]),
                                               raw_output_shape + [-1])
        squeeze_sliced_graph = ctx.create_new_graph_with_same_config()
        squeeze_sliced_graph.parent_graph = ctx
        squeeze_sliced = GraphBuilder(squeeze_sliced_graph).make_squeeze(
            {'data': final_output_right_sliced.output[0], 'axes': [-2]}, return_node=True)
        squeeze_sliced_graph.add_graph_output(squeeze_sliced.output[0], ctx.get_dtype(node.input[0]), raw_output_shape)
        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        branches = {"then_branch": squeeze_sliced_graph, "else_branch": identity_sliced_graph}
        squeeze_if = ctx.make_node('If', [need_squeeze.output[0]], name=node.name, outputs=node.output, shapes=shapes,
                                   dtypes=dtypes, branches=branches)

    @classmethod
    def any_version_after12(cls, opset, ctx, node, **kwargs):

        # assemble MatrixDiagPart V2&V3
        m = node.input[0]
        m_shape = ctx.get_shape(m)
        m_rank = len(m_shape)
        pads = np.zeros(2 * m_rank, dtype=np.int64)
        pads[-2:] = [1, 1]
        utils.make_sure(m_rank > 1, 'Input data should be at least 2D %s', str(m_shape))

        align = 'LEFT_LEFT'
        if node.op.op_type == 'MatrixDiagPartV3':
            align = node.get_attr_str('align') if 'align' in node.attr else 'LEFT_RIGHT'
        xalign, yalign = align.split('_')

        # consts
        const_zero_float, const_neg_one_float = [n.output[0] for n in ctx.make_consts([0, -1], np.float32)]
        const_zero, const_one, const_neg_one, const_neg_two, const_pad_vals, const_t = \
            [n.output[0] for n in ctx.make_consts([[0], [1], [-1], [-2], pads, [-1, 1]])]
        const_zero_scalar, const_one_scalar, const_neg_one_scalar = \
            [n.output[0] for n in ctx.make_consts([0, 1, -1])]

        m_shape = ctx.make_node('Shape', [node.input[0]]).output[0]
        xlen = ctx.make_node('Gather', [m_shape, const_neg_one]).output[0]
        ylen = ctx.make_node('Gather', [m_shape, const_neg_two]).output[0]
        xlenp = ctx.make_node('Add', [xlen, const_one]).output[0]
        stride = ctx.make_node('Add', [xlenp, const_one]).output[0]
        minxy_0 = ctx.make_node('Concat', [xlen, ylen], attr={'axis': 0}).output[0]
        minxy = ctx.make_node('ReduceMin', [minxy_0]).output[0]
        minxy_float = ctx.make_node('Cast', [minxy], attr={'to': TensorProto.FLOAT}).output[0]
        xmax_0 = ctx.make_node('Mul', [xlen, xlenp]).output[0]
        xmax_1 = ctx.make_node('Add', [xmax_0, xlenp]).output[0]
        xmax = ctx.make_node('Add', [xmax_1, const_neg_one]).output[0]
        ymax_0 = ctx.make_node('Mul', [xlenp, ylen]).output[0]
        ymax = ctx.make_node('Add', [ymax_0, const_neg_one]).output[0]
        ymax_float = ctx.make_node('Cast', [ymax], attr={'to': TensorProto.FLOAT}).output[0]
        partial_shape = ctx.make_node('Slice', [m_shape, const_zero, const_neg_two]).output[0]
        m2_shape = ctx.make_node('Concat', [partial_shape, const_neg_one], attr={'axis': 0}).output[0]
        gather_shape = ctx.make_node('Concat', [partial_shape, const_one], attr={'axis': 0}).output[0]

        def normalize():
            raw_input1 = ctx.make_node('Cast', [node.input[1]], attr={'to': TensorProto.INT64}).output[0]
            return ctx.make_node('Reshape', [raw_input1, const_neg_one])

        # get k0, k1 values. diags to be extracted
        input1 = normalize()
        k0 = ctx.make_node('ReduceMin', [input1.output[0]]).output[0]
        k1 = ctx.make_node('ReduceMax', [input1.output[0]]).output[0]
        k0_scalar = ctx.make_node('Squeeze', [k0]).output[0]
        k1_scalar = ctx.make_node('Squeeze', [k1]).output[0]
        m_padded = ctx.make_node('Pad', [m, const_pad_vals, node.input[2]])

        # starting indexes for super diagonals
        xstart_0 = ctx.make_node('Cast', [k0_scalar], attr={'to': TensorProto.FLOAT})
        xstart_1 = ctx.make_node('Max', [const_zero_float, xstart_0.output[0]])
        xstart_2 = ctx.make_node('Cast', [xstart_1.output[0]], attr={'to': TensorProto.INT64})
        xstart_3 = ctx.make_node('Add', [xstart_2.output[0], const_neg_one_scalar])
        xstart_4 = ctx.make_node('Range', [k1_scalar, xstart_3.output[0], const_neg_one_scalar])
        xstart = ctx.make_node('Reshape', [xstart_4.output[0], const_t])

        # starting indexes for sub diagonals
        ystart_0 = ctx.make_node('Cast', [k1_scalar], attr={'to': TensorProto.FLOAT})
        ystart_1 = ctx.make_node('Min', [const_neg_one_float, ystart_0.output[0]])
        ystart_2 = ctx.make_node('Cast', [ystart_1.output[0]], attr={'to': TensorProto.INT64})
        ystart_3 = ctx.make_node('Add', [k0_scalar, const_neg_one_scalar])
        ystart_4 = ctx.make_node('Range', [ystart_2.output[0], ystart_3.output[0], const_neg_one_scalar])
        ystart = ctx.make_node('Reshape', [ystart_4.output[0], const_t])

        xmax_0 = ctx.make_node('Mul', [xstart.output[0], xlenp])
        xmax = ctx.make_node('Sub', [xmax, xmax_0.output[0]])
        xmax_float = ctx.make_node('Cast', [xmax.output[0]], attr={'to': TensorProto.FLOAT})

        # lengths of super/sub diags to extract
        xsize_0 = ctx.make_node('Sub', [xlen, xstart.output[0]])
        xsize_1 = ctx.make_node('Cast', [xsize_0.output[0]], attr={'to': TensorProto.FLOAT})
        xsize_2 = ctx.make_node('Min', [xsize_1.output[0], minxy_float])
        xsize = ctx.make_node('Cast', [xsize_2.output[0]], attr={'to': TensorProto.INT64})
        ysize_0 = ctx.make_node('Add', [ylen, ystart.output[0]])
        ysize_1 = ctx.make_node('Cast', [ysize_0.output[0]], attr={'to': TensorProto.FLOAT})
        ysize_2 = ctx.make_node('Min', [ysize_1.output[0], minxy_float])
        ysize = ctx.make_node('Cast', [ysize_2.output[0]], attr={'to': TensorProto.INT64})
        diagsize = ctx.make_node('Concat', [xsize.output[0], ysize.output[0]], attr={'axis': 0})
        maxsize = ctx.make_node('ReduceMax', [diagsize.output[0]], attr={'keep_dims': 0})
        maxsize_0 = ctx.make_node('Reshape', [maxsize.output[0], const_neg_one])
        maxsize_scalar = ctx.make_node('Squeeze', [maxsize.output[0]])

        diagdistances_0 = ctx.make_node('Range', [const_zero_scalar, maxsize_scalar.output[0], const_one_scalar])
        diagdistances = ctx.make_node('Mul', [diagdistances_0.output[0], stride])

        def right_align(sizes, indices, starts, maxval):
            op1 = ctx.make_node('Sub', [maxsize.output[0], sizes.output[0]])
            op2 = ctx.make_node('Mul', [op1.output[0], stride])
            op3 = ctx.make_node('Sub', [indices.output[0], op2.output[0]])
            op4 = ctx.make_node('Less', [op3.output[0], starts.output[0]])
            op5 = ctx.make_node('Where', [op4.output[0], maxval, op3.output[0]])
            return op5

        # xdiags, ydiags contain indices of diagonal elements
        xdiags_0 = ctx.make_node('Add', [xstart.output[0], diagdistances.output[0]])
        xdiags_1 = ctx.make_node('Cast', [xdiags_0.output[0]], attr={'to': TensorProto.FLOAT})
        if xalign == 'RIGHT':
            xdiags = right_align(xsize, xdiags_0, xstart, ymax)
        else:
            xdiags_2 = ctx.make_node('Min', [xdiags_1.output[0], xmax_float.output[0]])
            xdiags = ctx.make_node('Cast', [xdiags_2.output[0]], attr={'to': TensorProto.INT64})

        ydiags_0_ = ctx.make_node('Abs', [ystart.output[0]])
        ydiags_1 = ctx.make_node('Mul', [ydiags_0_.output[0], xlenp])
        ydiags_2 = ctx.make_node('Add', [ydiags_1.output[0], diagdistances.output[0]])
        ydiags_3 = ctx.make_node('Cast', [ydiags_2.output[0]], attr={'to': TensorProto.FLOAT})
        if yalign == 'RIGHT':
            ydiags = right_align(ysize, ydiags_2, ydiags_1, ymax)
        else:
            ydiags_4 = ctx.make_node('Min', [ydiags_3.output[0], ymax_float])
            ydiags = ctx.make_node('Cast', [ydiags_4.output[0]], attr={'to': TensorProto.INT64})

        # flatten last dimension of matrix
        m2 = ctx.make_node('Reshape', [m_padded.output[0], m2_shape])

        diags_0 = ctx.make_node('Concat', [xdiags.output[0], ydiags.output[0]], attr={'axis': 0})
        diags_1 = ctx.make_node('Reshape', [diags_0.output[0], const_neg_one])
        diags_2 = ctx.make_node('Expand', [diags_1.output[0], gather_shape])
        diags = ctx.make_node('GatherElements', [m2.output[0], diags_2.output[0]], attr={'axis': -1})

        def compute_out_shape(k0_k1_same=False):
            g = ctx.create_new_graph_with_same_config()
            g.parent_graph = ctx
            if k0_k1_same:
                dims = [partial_shape, maxsize_0.output[0]]
            else:
                dims = [partial_shape, const_neg_one, maxsize_0.output[0]]
            outshape = g.make_node('Concat', dims, attr={'axis': 0})
            g.add_graph_output(outshape.output[0], TensorProto.INT64, [-1])
            return g

        # if k0=k1, rank of output matrix is 1 less than usual
        # hence, need 'If' to compute right output matrix shape
        k0_k1_same = ctx.make_node('Equal', [k1, k0])
        branches = {'then_branch': compute_out_shape(True), 'else_branch': compute_out_shape(False)}
        if_node = ctx.make_node('If', [k0_k1_same.output[0]], branches=branches)

        shapes = ctx.get_shape(node.output[0])
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        ctx.make_node('Reshape', [diags.output[0], if_node.output[0]], name=node.name, outputs=node.output,
                      shapes=[shapes], dtypes=dtypes)

        for consumer in ctx.find_output_consumers(node.output[0]):
            if consumer.type == 'Identity':
                ctx.set_shape(consumer.output[0], shapes)

    @classmethod
    def version_12(cls, ctx, node, **kwargs):
        cls.any_version_after12(12, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Parameters moved to inputs for operator Squeeze, Unsqueeze.
        cls.any_version_after12(13, ctx, node, **kwargs)


@tf_op(["MatrixDiag", "MatrixDiagV2", "MatrixDiagV3"])
class MatrixDiag:
    @classmethod
    def version_12(cls, ctx, node, **kwargs):
        # Assemble MatrixDiagV3 by ReverseSequence
        argc = len(node.input)

        if ctx.opset >= 13:
            squeeze_axes0 = ctx.make_const(utils.make_name("const_axes"), np.array([0], dtype=np.int64)).output[0]
            squeeze_axes_1 = ctx.make_const(utils.make_name("const_axes"), np.array([-1], dtype=np.int64)).output[0]
            squeeze_axes_2 = ctx.make_const(utils.make_name("const_axes"), np.array([-2], dtype=np.int64)).output[0]

        minus_two, minus_one, zeo, one, two = \
            [n.output[0] for n in ctx.make_consts([[-2], [-1], [0], [1], [2]])]

        def mknode(op, args, **kwargs):
            return ctx.make_node(op, args, **kwargs).output[0]

        def mknode2(g, op, args, **kwargs):
            return g.make_node(op, args, **kwargs).output[0]

        def normalize(name):
            # normalize arguments
            casted = mknode("Cast", [name], attr={'to': TensorProto.INT64})
            reshaped = mknode("Reshape", [casted, minus_one])
            return reshaped

        def cast(name):
            return mknode("Cast", [name], attr={"to": ctx.get_dtype(node.input[0])})

        def processdiag():
            # unsqueeze diag if necessary
            diag = node.input[0]
            shape = ctx.get_shape(diag)
            if len(shape) == 1:
                if ctx.opset < 13:
                    diag = mknode("Unsqueeze", [diag], attr={"axes": [0]})
                else:
                    diag = mknode("Unsqueeze", [diag, squeeze_axes0])
                shape = [1] + shape
                ctx.set_shape(diag, shape)

            diag_shape = mknode("Shape", [diag])
            diag_depth = mknode("Slice", [diag_shape, minus_two, minus_one])
            k = normalize(node.input[1]) if argc > 1 else zeo
            k_min = GraphBuilder(ctx).make_reduce_min({"data": k})
            k_max = GraphBuilder(ctx).make_reduce_max({"data": k})
            k_max_nxt = mknode("Add", [k_max, one])
            k_depth = mknode("Sub", [k_max_nxt, k_min])
            equal = mknode("Equal", [k_depth, diag_depth])

            def id_diag():
                g = ctx.create_new_graph_with_same_config()
                g.parent_graph = ctx
                idt = mknode2(g, "Identity", [diag])
                g.add_graph_output(idt, ctx.get_dtype(node.input[0]), ctx.get_shape(diag))
                return g

            def ex_diag():
                g = ctx.create_new_graph_with_same_config()
                g.parent_graph = ctx
                if ctx.opset < 13:
                    ex = mknode2(g, "Unsqueeze", [diag], attr={"axes": [-2]})
                else:
                    ex = mknode2(g, "Unsqueeze", [diag, squeeze_axes_2])
                rank = len(ctx.get_shape(diag)) + 1
                g.add_graph_output(ex, ctx.get_dtype(node.input[0]), [-1] * rank)
                return g

            branches = {"then_branch": id_diag(), "else_branch": ex_diag()}
            expand_diag = ctx.make_node("If", [equal], branches=branches)
            return expand_diag.output[0], k, k_min, k_max, k_max_nxt

        def squeeze_12(name):
            return ctx.make_node("Squeeze", [name], attr={"axis": -1}).output[0]

        def squeeze_13(name):
            return ctx.make_node("Squeeze", [name, squeeze_axes_1]).output[0]

        squeeze = squeeze_12 if ctx.opset < 13 else squeeze_13

        # gather inputs
        diag, k, k_min, k_max, k_max_nxt = processdiag()
        row, col, pad, align = normalize(node.input[2]) if argc > 2 else minus_one, \
                               normalize(node.input[3]) if argc > 3 else minus_one, \
                               node.input[4] if argc > 4 else cast(zeo), \
                               node.get_attr_str("align") if "align" in node.attr else "LEFT_LEFT"

        diag_shape = mknode("Shape", [diag])
        diag_rank = mknode("Shape", [diag_shape])
        head_shape = mknode("Slice", [diag_shape, zeo, minus_two])
        tail_shape = mknode("Slice", [diag_shape, minus_two, diag_rank])
        diag_width = mknode("Slice", [diag_shape, minus_one, diag_rank])
        diag_depth = mknode("Slice", [diag_shape, minus_two, minus_one])
        k_range = mknode("Range", [squeeze(k_min), squeeze(k_max_nxt), squeeze(one)])
        abs_k_range = mknode("Abs", [k_range])
        min_k2zeo = GraphBuilder(ctx).make_reduce_min({"data": abs_k_range})
        max_diag_len = mknode("Add", [min_k2zeo, diag_width])

        def outrowcol():
            # get output matrix shape
            row_set = mknode("Greater", [row, zeo])
            col_set = mknode("Greater", [col, zeo])

            def rowset():
                # if row is set
                g = ctx.create_new_graph_with_same_config()
                g.parent_graph = ctx

                def rowsetcolset():
                    # if col is set
                    gg = g.create_new_graph_with_same_config()
                    gg.parent_graph = g
                    id_row = mknode2(gg, "Identity", [row])
                    id_col = mknode2(gg, "Identity", [col])
                    shape = mknode2(gg, "Concat", [id_row, id_col], attr={"axis": -1})
                    gg.add_graph_output(shape, TensorProto.INT64, [-1])
                    return gg

                def rowsetcolnotset():
                    # if col is not set
                    gg = g.create_new_graph_with_same_config()
                    gg.parent_graph = g
                    id_row = mknode2(gg, "Identity", [row])
                    id_diag_width = mknode2(gg, "Identity", [diag_width])
                    shape = mknode2(gg, "Concat", [id_row, id_diag_width], attr={"axis": -1})
                    gg.add_graph_output(shape, TensorProto.INT64, [-1])
                    return gg

                branches = {"then_branch": rowsetcolset(), "else_branch": rowsetcolnotset()}
                if_col_set = g.make_node("If", [col_set], branches=branches)
                g.add_graph_output(if_col_set.output[0], TensorProto.INT64, [-1])
                return g

            def rownotset():
                # if row is not set
                g = ctx.create_new_graph_with_same_config()
                g.parent_graph = ctx

                def rownotsetcolset():
                    # if col is set
                    gg = g.create_new_graph_with_same_config()
                    gg.parent_graph = g
                    id_diag_width = gg.make_node("Identity", [diag_width]).output[0]
                    id_col = gg.make_node("Identity", [col]).output[0]
                    shape = gg.make_node("Concat", [id_diag_width, id_col], attr={"axis": -1}).output[0]
                    gg.add_graph_output(shape, TensorProto.INT64, [-1])
                    return gg

                def rownotsetcolnotset():
                    # if col is not set
                    gg = g.create_new_graph_with_same_config()
                    gg.parent_graph = g
                    id_max_diag_len = gg.make_node("Identity", [max_diag_len]).output[0]
                    shape = gg.make_node("Concat", [id_max_diag_len, id_max_diag_len], attr={"axis": -1}).output[0]
                    gg.add_graph_output(shape, TensorProto.INT64, [-1])
                    return gg

                branches = {"then_branch": rownotsetcolset(), "else_branch": rownotsetcolnotset()}
                if_col_set = g.make_node("If", [col_set], branches=branches)
                g.add_graph_output(if_col_set.output[0], TensorProto.INT64, [-1])
                return g

            branches = {"then_branch": rowset(), "else_branch": rownotset()}
            if_row_set = ctx.make_node("If", [row_set], branches=branches)
            return if_row_set.output[0]

        out_shape = outrowcol()
        out_row = mknode("Slice", [out_shape, zeo, one])
        out_col = mknode("Slice", [out_shape, one, two])
        k_btm = mknode("Sub", [one, out_row]) # lowest possible k

        def getklens():
            # return diag len of all ks
            rwcl_min = mknode("Min", [out_row, out_col])
            rwcl_gap = mknode("Sub", [out_row, out_col])
            absl_gap = mknode("Abs", [rwcl_gap])
            left_btm = mknode("Range", [squeeze(one), squeeze(rwcl_min), squeeze(one)])
            riht_top = mknode("Abs", [mknode("Sub", [left_btm, rwcl_min])])
            klen_mid = mknode("Expand", [rwcl_min, mknode("Add", [absl_gap, one])])
            return mknode("Concat", [left_btm, klen_mid, riht_top], attr={"axis": -1})

        k_lens = getklens()

        def reverseseq(args):
            return mknode("ReverseSequence", args, attr={"batch_axis": 0, "time_axis": 1})

        def reverse1d(name):
            # reverse an array
            shape = mknode("Shape", [name])
            temp_shape = mknode("Concat", [minus_one, shape], attr={"axis": -1})
            reshaped = mknode("Reshape", [name, temp_shape])
            rev = reverseseq([reshaped, shape])
            return mknode("Reshape", [rev, shape])

        def sortdiag():
            # sort diag to "LEFT_RIGHT" so each col form a line of the out matrix
            k_sup_stt = mknode("Sub", [mknode("Max", [zeo, k_min]), k_btm])
            k_sup_end = mknode("Sub", [k_max_nxt, k_btm])
            k_sup_len = mknode("Max", [zeo, mknode("Sub", [k_sup_end, k_sup_stt])])
            k_sub_stt = mknode("Sub", [k_min, k_btm])
            k_sub_end = mknode("Sub", [mknode("Min", [zeo, k_max_nxt]), k_btm])
            k_sub_len = mknode("Max", [zeo, mknode("Sub", [k_sub_end, k_sub_stt])])
            sup_k_lens = mknode("Slice", [k_lens, k_sup_stt, k_sup_end])
            sub_k_lens = mknode("Slice", [k_lens, k_sub_stt, k_sub_end])
            all_k_lens = mknode("Concat", [sub_k_lens, sup_k_lens], attr={"axis": -1})
            max_k_len = mknode("ReduceMax", [all_k_lens])
            top_k_len = mknode("Slice", [all_k_lens, minus_one, diag_depth])
            btm_k_len = mknode("Slice", [all_k_lens, zeo, one])
            diag_rev_shap = mknode("Concat", [minus_one, diag_width], attr={"axis": -1})
            reshaped_diag = mknode("Reshape", [diag, diag_rev_shap])
            rev_shape = mknode("Slice", [diag_shape, zeo, minus_one])

            sup_rev_len_1 = mknode("Expand", [one, k_sup_len]) if align.startswith("LEFT") else mknode("Expand",
                                                                                                       [diag_width,
                                                                                                        k_sup_len])
            sub_rev_len_1 = mknode("Expand", [one, k_sub_len]) if align.endswith("RIGHT") else sub_k_lens
            cnt_rev_len_1 = mknode("Concat", [sub_rev_len_1, sup_rev_len_1], attr={"axis": -1})
            exp_rev_len_1 = mknode("Expand", [reverse1d(cnt_rev_len_1), rev_shape])

            sup_rev_len_2 = mknode("Expand", [one, k_sup_len]) if align.startswith("LEFT") else sup_k_lens
            sub_rev_len_2 = mknode("Expand", [one, k_sub_len]) if align.endswith("RIGHT") else mknode("Expand",
                                                                                                      [diag_width,
                                                                                                       k_sub_len])
            cnt_rev_len_2 = mknode("Concat", [sub_rev_len_2, sup_rev_len_2], attr={"axis": -1})
            exp_rev_len_2 = mknode("Expand", [reverse1d(cnt_rev_len_2), rev_shape])

            reversed_diag_1 = reverseseq([reshaped_diag, mknode("Reshape", [exp_rev_len_1, minus_one])])
            reversed_diag_2 = reverseseq([reversed_diag_1, mknode("Reshape", [exp_rev_len_2, minus_one])])

            return mknode("Reshape", [reversed_diag_2, diag_shape]), \
                   mknode("Sub", [max_k_len, top_k_len]), \
                   mknode("Sub", [max_k_len, btm_k_len])

        sorted_diag, top_pad, btm_pad = sortdiag()

        def trandiag():
            # transpose last two dim of diag
            temp_shape = mknode("Concat", [minus_one, tail_shape], attr={"axis": -1})
            reshaped = mknode("Reshape", [sorted_diag, temp_shape])
            transposed = mknode("Transpose", [reshaped], attr={"perm": [0, 2, 1]})
            out_shape = mknode("Concat", [head_shape, reverse1d(tail_shape)], attr={"axis": -1})
            return mknode("Reshape", [transposed, out_shape])

        tran_diag = trandiag()

        def relu1(name):
            # all return values >= 1
            minusd = mknode("Sub", [name, one])
            casted = mknode("Cast", [minusd], attr={"to": TensorProto.FLOAT})
            relued = mknode("Relu", [casted])
            casted = mknode("Cast", [relued], attr={"to": TensorProto.INT64})
            return mknode("Add", [casted, one])

        def makediagonal():
            # padding with required value and move lines so they form diagonals
            shape = mknode("Shape", [tran_diag])
            rank = mknode("Shape", [shape])
            width = mknode("Slice", [shape, minus_one, rank])
            temp_shape = mknode("Concat", [minus_one, width], attr={"axis": -1})
            reshaped = mknode("Reshape", [tran_diag, temp_shape])
            left_pad, riht_pad = top_pad, mknode("Add", [btm_pad, diag_width])
            full_pad = mknode("Concat", [zeo, left_pad, zeo, riht_pad], attr={"axis": -1})
            diag_pad = mknode("Pad", [reshaped, full_pad, pad])
            diag_pad_shape = mknode("Shape", [diag_pad])
            diag_pad_width = mknode("Slice", [diag_pad_shape, one, two])
            exp_shape = mknode("Concat", [head_shape, diag_width], attr={"axis": -1})

            def padleft():
                # set pads from left
                fm = mknode("Add", [left_pad, left_pad])
                to = mknode("Sub", [fm, diag_width])
                rg = reverse1d(relu1(mknode("Range", [squeeze(fm), squeeze(to), squeeze(minus_one)])))
                expanded_range = mknode("Expand", [rg, exp_shape])
                reshaped_range = mknode("Reshape", [expanded_range, minus_one])
                pad_left = mknode("ReverseSequence", [diag_pad, reshaped_range], attr={"batch_axis": 0, "time_axis": 1})
                return mknode("Slice", [pad_left, left_pad, diag_pad_width, one])

            pad_left = padleft()

            def padright():
                # set pads from right
                pad_left_shape = mknode("Shape", [pad_left])
                pad_left_depth = mknode("Slice", [pad_left_shape, zeo, one])
                pad_left_width = mknode("Slice", [pad_left_shape, one, two])
                pad_full_lenth = mknode("Expand", [pad_left_width, pad_left_depth])
                rev = mknode("ReverseSequence", [pad_left, pad_full_lenth], attr={"batch_axis": 0, "time_axis": 1})
                fm = mknode("Add", [riht_pad, btm_pad])
                to = mknode("Sub", [fm, diag_width])
                rg = mknode("Range", [squeeze(fm), squeeze(to), squeeze(minus_one)])
                expanded_range = mknode("Expand", [rg, exp_shape])
                reshaped_range = mknode("Reshape", [expanded_range, minus_one])
                raw_pad_right = mknode("ReverseSequence", [rev, reshaped_range],
                                       attr={"batch_axis": 0, "time_axis": 1})
                shape = mknode("Shape", [raw_pad_right])
                width = mknode("Slice", [shape, one, two])
                sliced = mknode("Slice", [raw_pad_right, btm_pad, width, one])
                all_width = mknode("Expand", [mknode("Sub", [width, btm_pad]), mknode("Shape", [reshaped_range])])
                return mknode("ReverseSequence", [sliced, all_width], attr={"batch_axis": 0, "time_axis": 1})

            pad_right = padright()

            def diagonize():
                # move lines to right to form diagonals
                fm = mknode("Sub", [diag_depth, btm_pad])
                to = mknode("Add", [fm, diag_width])
                rg = mknode("Range", [squeeze(fm), squeeze(to), squeeze(one)])
                expanded_range = mknode("Expand", [rg, exp_shape])
                reshaped_range = mknode("Reshape", [expanded_range, minus_one])
                rev = mknode("ReverseSequence", [pad_right, reshaped_range],
                             attr={"batch_axis": 0, "time_axis": 1})
                k_max_idx = mknode("Sub", [k_max, k_btm])
                k_max_idx_nxt = mknode("Add", [k_max_idx, one])
                k_max_len = mknode("Slice", [k_lens, k_max_idx, k_max_idx_nxt])
                k_gap = mknode("Sub", [mknode("Abs", [k_max]), min_k2zeo])
                width = mknode("Add", [k_max_len, k_gap])
                return mknode("Slice", [rev, zeo, width, one]), width

            diag, width = diagonize()
            shape = mknode("Concat", [head_shape, diag_width, minus_one], attr={"axis": -1})
            return mknode("Reshape", [diag, shape]), diag_width, width

        new_diag, new_depth, new_width = makediagonal()

        def paddiag():
            # pad to output shape
            pad_row, pad_col = mknode("Sub", [out_row, new_depth]), mknode("Sub", [out_col, new_width])
            pad_top = mknode("Max", [zeo, mknode("Sub", [zeo, k_max])])
            pad_lft = mknode("Max", [zeo, mknode("Sub", [k_min, zeo])])
            pad_btm = mknode("Sub", [pad_row, pad_top])
            pad_rht = mknode("Sub", [pad_col, pad_lft])
            pad_hlf = mknode("Mul", [zeo, head_shape])
            pad_ful = mknode("Concat", [pad_hlf, pad_top, pad_lft, pad_hlf, pad_btm, pad_rht], attr={"axis": -1})
            return mknode("Pad", [new_diag, pad_ful, pad])

        padded = paddiag()
        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        ctx.make_node("Identity", [padded], name=node.name,
                      outputs=node.output, shapes=shapes, dtypes=dtypes)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Parameters moved to inputs for operator Squeeze, Unsqueeze.
        cls.version_12(ctx, node, **kwargs)


@tf_op("MatrixSetDiagV3")
class MatrixSetDiagV3:
    @classmethod
    def version_12(cls, ctx, node, **kwargs):
        # Assemble MatrixSetDiagV3 by MatrixDiagPartV3 and MatrixDiagV3

        minus_two, minus_one, zeo, one = \
            [n.output[0] for n in ctx.make_consts([[-2], [-1], [0], [1]])]

        def mknode(op, args, **kwargs):
            return ctx.make_node(op, args, **kwargs).output[0]

        def integer(name):
            return mknode("Cast", [name], attr={"to": TensorProto.INT64})

        def cast(name):
            return mknode("Cast", [name], attr={"to": ctx.get_dtype(node.input[0])})

        def normalize():
            k = node.input[2]
            casted = mknode("Cast", [k], attr={"to": TensorProto.INT64})
            return mknode("Reshape", [casted, minus_one])

        x = node.input[0]
        diag = node.input[1]
        k = normalize()
        attr = {"align": node.get_attr_str("align")}

        shape = mknode("Shape", [x])
        rank = mknode("Shape", [shape])
        row = mknode("Slice", [shape, minus_two, minus_one])
        col = mknode("Slice", [shape, minus_one, rank])

        # ones of x shape
        zeos = mknode("Mul", [integer(x), zeo])
        ones = mknode("Add", [zeos, one])

        # make diag of 1s
        ones_diag = ctx.make_node("MatrixDiagPartV3", [ones, k, zeo], attr,
                                  shapes=[ctx.get_shape(x)], dtypes=[ctx.get_dtype(x)])
        MatrixDiagPartV2V3.version_11(ctx, ones_diag)
        # MatrixDiagPartV2V3.version_12(ctx, ones_diag) # todo: fix exception

        # make matrix of bool
        ctx.set_dtype(ones_diag.output[0], TensorProto.INT64)
        ones_matrix = ctx.make_node("MatrixDiagV3", [ones_diag.output[0], k, row, col, zeo], attr,
                                    shapes=[ctx.get_shape(x)], dtypes=[ctx.get_dtype(x)])
        MatrixDiag.version_12(ctx, ones_matrix)
        ones_bool = mknode("Equal", [ones_matrix.output[0], one])

        # make matrix out of diag
        diag_matrix = ctx.make_node("MatrixDiagV3", [diag, k, row, col, cast(zeo)], attr,
                                    shapes=[ctx.get_shape(x)], dtypes=[ctx.get_dtype(x)])
        MatrixDiag.version_12(ctx, diag_matrix)

        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        mknode("Where", [ones_bool, diag_matrix.output[0], x],
               name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)


@tf_op("BroadcastTo")
class BroadcastTo:
    @classmethod
    def version_8(cls, ctx, node, **kwargs):
        # broadcast by expanding
        node.type = "Expand"
        ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=TensorProto.INT64)


@tf_op("AsString")
class AsString:
    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        if (node.get_attr_value("precision") or node.get_attr_value("scientific") or node.get_attr_value("fill")):
            logger.warning(
                "ONNX does not support precision, scientific and fill attributes for AsString")
        shapes = node.output_shapes
        ctx.remove_node(node.name)
        _ = ctx.make_node("Cast", node.input, name=node.name,
                          dtypes=[TensorProto.STRING], shapes=shapes, attr={"to": TensorProto.STRING})


@tf_op("StringToNumber")
class StringToNumber:
    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        _ = ctx.make_node("Cast", node.input, name=node.name,
                          dtypes=dtypes, shapes=shapes, attr={"to": dtypes[0]})
