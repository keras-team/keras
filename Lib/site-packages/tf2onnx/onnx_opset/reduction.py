# SPDX-License-Identifier: Apache-2.0


"""
reduction
"""

import logging

import numpy as np
from onnx import onnx_pb, helper

from tf2onnx import utils
from tf2onnx.handler import tf_op
from tf2onnx.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring,protected-access

@tf_op("Max", onnx_op="ReduceMax")
@tf_op("Mean", onnx_op="ReduceMean")
@tf_op("Min", onnx_op="ReduceMin")
@tf_op("Prod", onnx_op="ReduceProd")
@tf_op("Sum", onnx_op="ReduceSum")
class ReduceOpBase:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        axes_node = node.inputs[1]
        axes = axes_node.get_tensor_value()
        if np.isscalar(axes):
            axes = [axes]
        input_shape = ctx.get_shape(node.input[0])
        if input_shape is None:
            if any([val < 0 for val in axes]) and ctx.opset < 11:
                raise ValueError("reduce_op: cannot have negative axis if opset < 11 because we don't know input rank")
        else:
            input_rank = len(ctx.get_shape(node.input[0]))
            axes = [val + input_rank if val < 0 else val for val in axes]

        node.set_attr("axes", axes)
        ctx.remove_input(node, node.input[1], 1)
        keep_dims = node.get_attr_value("keep_dims", 0)
        node.set_attr("keepdims", keep_dims)
        del node.attr['keep_dims']

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        if node.type == "ReduceSum":
            keep_dims = node.get_attr_value("keep_dims", 0)
            node.set_attr("keepdims", keep_dims)
            del node.attr['keep_dims']
            node.set_attr("noop_with_empty_axes", 1)
            if ctx.get_dtype(node.input[1]) != onnx_pb.TensorProto.INT64:
                ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=onnx_pb.TensorProto.INT64)
            input_shape = ctx.get_shape(node.input[1])
            input_rank = len(input_shape) if input_shape is not None else None
            if input_rank != 1:
                new_shape = ctx.make_const(utils.make_name("reshape_const"), np.array([-1], np.int64))
                ctx.insert_new_node_on_input(node, "Reshape", [node.input[1], new_shape.output[0]])
        else:
            cls.version_11(ctx, node, **kwargs)

    @classmethod
    def version_18(cls, ctx, node, **kwargs):
        keep_dims = node.get_attr_value("keep_dims", 0)
        node.set_attr("keepdims", keep_dims)
        del node.attr['keep_dims']
        node.set_attr("noop_with_empty_axes", 1)
        if ctx.get_dtype(node.input[1]) != onnx_pb.TensorProto.INT64:
            ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=onnx_pb.TensorProto.INT64)
        input_shape = ctx.get_shape(node.input[1])
        input_rank = len(input_shape) if input_shape is not None else None
        if input_rank != 1:
            new_shape = ctx.make_const(utils.make_name("reshape_const"), np.array([-1], np.int64))
            ctx.insert_new_node_on_input(node, "Reshape", [node.input[1], new_shape.output[0]])

@tf_op(["ArgMax", "ArgMin"])
class ArgMax:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # output_type output = ArgMin(T input, Tidx dimension, @type Tidx, @type output_type)
        # tensor(int32) reduced = ArgMin(T data, @INT axis, @INT keepdims)
        axis_node = node.inputs[1]
        axis = axis_node.get_tensor_value()
        if axis < 0:
            # ArgMax|ArgMin in onnx don't necessary support negative axis(not in doc explicitly)
            input_shape = ctx.get_shape(node.input[0])
            dim_count = len(input_shape) if input_shape else 0
            axis = dim_count + axis

        # TF ArgMin/ArgMax may return int32 or int64
        # Onnx ArgMin/ArgMax only supports int64 output, add cast if needed
        if node.get_attr_int("output_type") == onnx_pb.TensorProto.INT32:
            # current node will return int64 after conversion, which differs from previous dtype got from tf
            ctx.set_dtype(node.output[0], onnx_pb.TensorProto.INT64)
            op_name = utils.make_name("Cast")
            cast_node = ctx.insert_new_node_on_output("Cast", node.output[0], name=op_name,
                                                      to=onnx_pb.TensorProto.INT32)
            ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.INT32)
            ctx.copy_shape(node.output[0], cast_node.output[0])

        node.set_attr("axis", axis)
        node.set_attr("keepdims", 0)
        ctx.remove_input(node, node.input[1], 1)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic same
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_12(cls, ctx, node, **kwargs):
        # Opset 12 adds extra attribute 'select_last_index'
        # No changes needed
        cls.version_1(ctx, node, **kwargs)

@tf_op(["All", "Any"])
class AllAny:
    @classmethod
    def version_6(cls, ctx, node, **kwargs):
        # T output = All(T x, list(int) reduce_indices, @bool keepdims)
        # T output = Any(T x, list(int) reduce_indices, @bool keepdims)
        reduce_dim = node.inputs[1].get_tensor_value()

        # for Any, the reduce_indices can be scalar as observed.
        if np.isscalar(reduce_dim):
            reduce_dim = [reduce_dim]

        if ctx.opset < 11:
            inp_rank = ctx.get_rank(node.input[0])
            if inp_rank is not None:
                reduce_dim = [d + inp_rank if d < 0 else d for d in reduce_dim]
            utils.make_sure(all(i >= 0 for i in reduce_dim), "negative reduce axis is not supported in onnx for now")

        cast = ctx.make_node(op_type="Cast", inputs=[node.input[0]], attr={"to": onnx_pb.TensorProto.FLOAT})
        keepdims = helper.get_attribute_value(node.get_attr("keep_dims"))
        op_type = "ReduceMin" if node.type == "All" else "ReduceSum"

        if op_type == "ReduceSum":
            reduce_node_output = GraphBuilder(ctx).make_reduce_sum(
                {"data": cast.output[0], "axes": reduce_dim, "keepdims": keepdims, "noop_with_empty_axes": 1})
        else:
            reduce_node_output = ctx.make_node(op_type=op_type, inputs=cast.output,
                                               attr={"axes": reduce_dim, "keepdims": keepdims}).output[0]

        zero_node = ctx.make_const(utils.make_name("zero_reduce"), np.array(0, dtype=np.float32))

        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        ctx.make_node(op_type="Greater", inputs=[reduce_node_output, zero_node.output[0]],
                      name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        keepdims = node.get_attr_value('keep_dims')
        reduce_input = node.input[0]
        if node.type == "All":
            reduce_input = ctx.make_node("Not", [reduce_input]).output[0]
        cast = ctx.make_node("Cast", inputs=[reduce_input], attr={"to": onnx_pb.TensorProto.FLOAT}).output[0]
        axes_cast = node.input[1]
        if ctx.get_rank(axes_cast) == 0:
            # Unsqueeze scalar axes
            axes_cast = GraphBuilder(ctx).make_unsqueeze({'data': axes_cast, 'axes': [0]})
        if ctx.get_dtype(axes_cast) != onnx_pb.TensorProto.INT64:
            axes_cast = ctx.make_node("Cast", inputs=[axes_cast], attr={"to": onnx_pb.TensorProto.INT64}).output[0]
        reduce_node_output = GraphBuilder(ctx).make_reduce_sum(
            {"data": cast, "axes": axes_cast, "keepdims": keepdims, "noop_with_empty_axes": 1},
            shapes=node.output_shapes, op_name_scope=node.name)
        zero_node = ctx.make_const(utils.make_name("zero_reduce"), np.array(0, dtype=np.float32))
        greater_node = ctx.make_node(op_type="Greater", inputs=[reduce_node_output, zero_node.output[0]])
        result = greater_node.output[0]
        if node.type == "All":
            result = ctx.make_node("Not", [result]).output[0]
        ctx.replace_all_inputs(node.output[0], result)


@tf_op("AddN")
class AddN():
    @classmethod
    def version_6(cls, ctx, node, **kwargs):
        node.type = "Sum"


@tf_op(["SegmentSum", "SegmentProd", "SegmentMax", "SegmentMin", "SegmentMean",
        "SparseSegmentSum", "SparseSegmentMean", "SparseSegmentSqrtN",
        "SparseSegmentSumWithNumSegments", "SparseSegmentMeanWithNumSegments", "SparseSegmentSqrtNWithNumSegments",
        "UnsortedSegmentSum", "UnsortedSegmentProd", "UnsortedSegmentMax", "UnsortedSegmentMin"])
class SegmentSum():
    @classmethod
    def any_version(cls, opset, ctx, node, **kwargs):
        node_inputs = node.input.copy()
        num_segments_specified = False
        num_segs_const = None
        if node.type.endswith("WithNumSegments") or node.type.startswith("Unsorted"):
            num_segments_specified = True
            num_segments = node_inputs.pop()
            node.type = node.type.replace("WithNumSegments", "")
            node.type = node.type.replace("Unsorted", "")
            if node.inputs[-1].is_const():
                num_segs_const = node.inputs[-1].get_tensor_value(as_list=True)
        if node.type.startswith("Sparse"):
            data_inp, indices_inp, segment_inp = node_inputs
            gather_node = ctx.make_node("Gather", [data_inp, indices_inp], attr={'axis': 0})
            data_inp = gather_node.output[0]
            node.type = node.type.replace("Sparse", "")
        else:
            data_inp, segment_inp = node_inputs

        # Data has shape [n, a, b, ..., c]
        data_shape = ctx.get_shape(data_inp)
        data_rank = len(data_shape) if data_shape is not None else None
        data_dtype = ctx.get_dtype(data_inp)
        seg_rank = ctx.get_rank(segment_inp)

        if ctx.get_dtype(segment_inp) != onnx_pb.TensorProto.INT64:
            segment_inp = ctx.make_node("Cast", [segment_inp], attr={"to": onnx_pb.TensorProto.INT64}).output[0]

        utils.make_sure(seg_rank == 1, "Segment ops only supported for segments of rank 1, not %s", seg_rank)
        data_np_dtype = utils.map_onnx_to_numpy_type(data_dtype)
        seg_np_dtype = utils.map_onnx_to_numpy_type(ctx.get_dtype(segment_inp))

        if num_segments_specified and ctx.get_dtype(segment_inp) != ctx.get_dtype(num_segments):
            num_segments = ctx.make_node("Cast", [num_segments], attr={"to": ctx.get_dtype(segment_inp)}).output[0]

        data_is_float = np.dtype(data_np_dtype).kind == 'f'
        data_is_int = np.dtype(data_np_dtype).kind == 'i'
        utils.make_sure(data_is_float or data_is_int, "dtype for Segment ops must be float or int")

        if node.type in ["SegmentSum", "SegmentMean", "SegmentSqrtN"]:
            onnx_op = "ReduceSum"
            identity_value = np.array(0, dtype=data_np_dtype)
        elif node.type == "SegmentProd":
            onnx_op = "ReduceProd"
            identity_value = np.array(1, dtype=data_np_dtype)
        elif node.type == "SegmentMax":
            onnx_op = "ReduceMax"
            if data_is_float:
                identity_value = np.array('-inf', dtype=data_np_dtype)
            else:
                identity_value = np.iinfo(data_np_dtype).min
        elif node.type == "SegmentMin":
            onnx_op = "ReduceMin"
            if data_is_float:
                identity_value = np.array('inf', dtype=data_np_dtype)
            else:
                identity_value = np.iinfo(data_np_dtype).max

        if not num_segments_specified:
            max_segment = GraphBuilder(ctx).make_reduce_max({"data": segment_inp, "axes": [0], "keepdims": 0})
            one_const = ctx.make_const(utils.make_name("const_one"), np.array(1, dtype=seg_np_dtype))
            num_segments = ctx.make_node("Add", [max_segment, one_const.output[0]]).output[0]
        num_segments_unsq = GraphBuilder(ctx).make_unsqueeze({'data': num_segments, 'axes': [0]})

        seg_shape = ctx.make_node("Shape", [segment_inp]).output[0]
        seg_shape_sq = GraphBuilder(ctx).make_squeeze({"data": seg_shape, "axes": [0]})
        segs_sorted, indices = ctx.make_node(
            "TopK", [segment_inp, seg_shape], attr={'axis': 0, 'largest': False, 'sorted': True},
            output_count=2, op_name_scope=node.name).output
        seg_unique_node = ctx.make_node(
            "Unique", [segs_sorted], attr={'axis': 0, 'sorted': True}, output_count=4, op_name_scope=node.name)
        seg_unique_node.output[1] = ""
        seg_values, _, inv_indices, seg_cnts_sorted = seg_unique_node.output

        max_cnt = GraphBuilder(ctx).make_reduce_max({"data": seg_cnts_sorted, "axes": [0], "keepdims": True})

        if node.type in ["SegmentMean", "SegmentSqrtN"]:
            zero_tensor = helper.make_tensor("value", onnx_pb.TensorProto.INT64, dims=[1], vals=[0])
            if num_segs_const is not None:
                zeros = ctx.make_const(utils.make_name("zeros"), np.zeros([num_segs_const], np.int64)).output[0]
            else:
                zeros = ctx.make_node("ConstantOfShape", [num_segments_unsq], attr={'value': zero_tensor}).output[0]
            seg_cnts = ctx.make_node("ScatterElements", [zeros, seg_values, seg_cnts_sorted],
                                     attr={'axis': 0}).output[0]
            seg_cnts_float = ctx.make_node("Cast", [seg_cnts], attr={'to': data_dtype}).output[0]
        if node.type == "SegmentMean":
            scaling_amt = seg_cnts_float
        elif node.type == "SegmentSqrtN":
            scaling_amt = ctx.make_node("Sqrt", [seg_cnts_float]).output[0]
        else:
            scaling_amt = None

        if scaling_amt is not None and num_segments_specified:
            # If empty segments are possible, we must avoid division by zero
            const_one_float = ctx.make_const(utils.make_name("const_one_float"), np.array(1, dtype=np.float32))
            scaling_amt = ctx.make_node("Max", [scaling_amt, const_one_float.output[0]]).output[0]


        zero_const_int64 = ctx.make_const(utils.make_name("const_zero"), np.array(0, dtype=np.int64)).output[0]
        one_const_int64 = ctx.make_const(utils.make_name("const_one"), np.array(1, dtype=np.int64)).output[0]
        seg_range = ctx.make_node("Range", [zero_const_int64, seg_shape_sq, one_const_int64]).output[0]

        id_to_cnt = ctx.make_node("Gather", [seg_cnts_sorted, inv_indices]).output[0]
        range_mod = ctx.make_node("Mod", [seg_range, id_to_cnt]).output[0]

        idx_grid_shape = ctx.make_node("Concat", [num_segments_unsq, max_cnt], {'axis': 0}).output[0]
        neg_one_tensor = helper.make_tensor("value", onnx_pb.TensorProto.INT64, dims=[1], vals=[-1])
        idx_grid = ctx.make_node("ConstantOfShape", [idx_grid_shape], {'value': neg_one_tensor}).output[0]

        segs_sorted_unsq = GraphBuilder(ctx).make_unsqueeze({'data': segs_sorted, 'axes': [-1]})
        range_mod_unsq = GraphBuilder(ctx).make_unsqueeze({'data': range_mod, 'axes': [-1]})
        scatter_indices = ctx.make_node("Concat", [segs_sorted_unsq, range_mod_unsq], attr={'axis': 1}).output[0]
        scatted_grid = ctx.make_node("ScatterND", [idx_grid, scatter_indices, indices]).output[0]

        data_shape = ctx.make_node("Shape", [data_inp]).output[0]

        max_int64 = int(utils.get_max_value(np.int64))
        identity_shape = GraphBuilder(ctx).make_slice(
            {'data': data_shape, 'starts': [1], 'ends': [max_int64], 'axes': [0]})
        id_tensor = helper.make_tensor("value", ctx.get_dtype(data_inp), dims=[1], vals=[identity_value.tolist()])
        identity = ctx.make_node("ConstantOfShape", [identity_shape], {'value': id_tensor}).output[0]
        id_unsq = GraphBuilder(ctx).make_unsqueeze({'data': identity, 'axes': [0]})
        data_with_id = ctx.make_node("Concat", [data_inp, id_unsq], attr={'axis': 0}).output[0]
        data_grid = ctx.make_node("Gather", [data_with_id, scatted_grid]).output[0]
        if onnx_op == "ReduceSum":
            reduction_result = GraphBuilder(ctx).make_reduce_sum(
                {"data": data_grid, "axes": [1], "keepdims": False}, op_name_scope=node.name)
        else:
            reduction_result = GraphBuilder(ctx)._make_reduce_op(
                onnx_op, 18, {"data": data_grid, "axes": [1], "keepdims": False}, op_name_scope=node.name)
        if scaling_amt is not None:
            if data_rank is None:
                # Left pad scale to match data rank
                data_slice_rank = ctx.make_node("Shape", [identity_shape]).output[0]
                one_tensor = helper.make_tensor("value", onnx_pb.TensorProto.INT64, dims=[1], vals=[1])
                ones_of_shape = ctx.make_node("ConstantOfShape", [data_slice_rank], {'value': one_tensor}).output[0]
                zero_unsq = ctx.make_const(utils.make_name('const_zero'), np.array([0], np.int64)).output[0]
                scaling_shape = ctx.make_node("Concat", [zero_unsq, ones_of_shape], attr={'axis': 0}).output[0]
                scaling_amt = ctx.make_node("Reshape", [scaling_amt, scaling_shape]).output[0]
            elif data_rank != 1:
                scaling_amt = GraphBuilder(ctx).make_unsqueeze(
                    {'data': scaling_amt, 'axes': list(range(1, data_rank))})
            reduction_result = ctx.make_node("Div", [reduction_result, scaling_amt]).output[0]

        ctx.replace_all_inputs(node.output[0], reduction_result)
        ctx.remove_node(node.name)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        cls.any_version(11, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        cls.any_version(13, ctx, node, **kwargs)
