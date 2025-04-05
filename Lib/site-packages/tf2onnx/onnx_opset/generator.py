# SPDX-License-Identifier: Apache-2.0


"""
generator
"""

import logging

import numpy as np
from onnx import onnx_pb, numpy_helper, helper
from tf2onnx import utils
from tf2onnx.handler import tf_op
from tf2onnx.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring

@tf_op(["Const", "ConstV2"])
class DirectOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass


@tf_op(["RandomNormal", "RandomUniform", "RandomUniformInt", "RandomStandardNormal"])
class RandomOp:
    @classmethod
    def randuniform_int(cls, ctx, rand_node, rand_out, min_inp, max_inp):
        dtype = ctx.get_dtype(rand_out)
        min_node = ctx.get_node_by_output(min_inp)
        max_node = ctx.get_node_by_output(max_inp)
        ctx.set_dtype(rand_node.output[0], onnx_pb.TensorProto.FLOAT)
        ctx.set_dtype(rand_out, onnx_pb.TensorProto.FLOAT)
        if min_node.is_const() and max_node.is_const():
            rand_node.set_attr('low', float(min_node.get_tensor_value()))
            rand_node.set_attr('high', float(max_node.get_tensor_value()))
            out = rand_out
        elif min_node.is_const() and min_node.get_tensor_value() == 0:
            max_float = ctx.make_node("Cast", [max_inp], attr={'to': onnx_pb.TensorProto.FLOAT}).output[0]
            mul_node = ctx.insert_new_node_on_output("Mul", rand_out, inputs=[rand_out, max_float])
            out = mul_node.output[0]
        else:
            min_float = ctx.make_node("Cast", [min_inp], attr={'to': onnx_pb.TensorProto.FLOAT}).output[0]
            max_float = ctx.make_node("Cast", [max_inp], attr={'to': onnx_pb.TensorProto.FLOAT}).output[0]
            diff = ctx.make_node("Sub", [max_float, min_float]).output[0]
            diff_float = ctx.make_node("Cast", [diff], attr={'to': onnx_pb.TensorProto.FLOAT}).output[0]
            mul_node = ctx.insert_new_node_on_output("Mul", rand_out, inputs=[rand_out, diff_float])
            mul = mul_node.output[0]
            add_node = ctx.insert_new_node_on_output("Add", mul, inputs=[mul, min_float])
            out = add_node.output[0]
        floor_node = ctx.insert_new_node_on_output("Floor", out)
        ctx.insert_new_node_on_output("Cast", floor_node.output[0], to=dtype)

    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # in tf-2.0 grappler optimizes the graph pretty well and our matching logic
        # in the rewriter does not trigger. grappler will send the random uniform
        # with shape as input so we need to pickup the input here and if the shape is
        # const we make it an attribute.
        seed = node.get_attr("seed2")
        node.set_attr("seed", float(seed.i))
        utils.make_sure(node.inputs[0].is_const(), "%s node with non-const shape requires opset >= 9", node.type)
        shape = node.inputs[0].get_tensor_value()
        ctx.remove_input(node, node.input[0], 0)
        if len(shape) == 0:
            # ORT can't take an empty shape (scalar)
            node.set_attr("shape", [1])
            ctx.set_shape(node.output[0], [1])
            squeeze_node = GraphBuilder(ctx).make_squeeze({'data': node.output[0], 'axes': [0]}, return_node=True)
            ctx.insert_node_on_output(squeeze_node, node.output[0])
            rand_out = squeeze_node.output[0]
        else:
            node.set_attr("shape", shape)
            ctx.set_shape(node.output[0], shape)
            rand_out = node.output[0]
        if node.type == "RandomUniformInt":
            cls.randuniform_int(ctx, node, rand_out, node.input[0], node.input[1])
            node.type = "RandomUniform"
            ctx.replace_inputs(node, [])
        elif node.type == "RandomStandardNormal":
            node.type = "RandomNormal"

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        if node.inputs[0].is_const():
            cls.version_1(ctx, node, **kwargs)
        else:
            seed = node.get_attr("seed2")
            node.set_attr("seed", float(seed.i))
            cast_node = ctx.make_node("Cast", [node.input[0]], attr={'to': onnx_pb.TensorProto.INT64})
            const_node = ctx.make_node("ConstantOfShape", cast_node.output)
            inputs = node.input.copy()
            ctx.replace_inputs(node, const_node.output.copy())
            if node.type == "RandomUniformInt":
                cls.randuniform_int(ctx, node, node.output[0], inputs[1], inputs[2])
                node.type = "RandomUniformLike"
            elif node.type == "RandomStandardNormal":
                node.type = "RandomNormalLike"
            else:
                node.type = node.type + 'Like'


@tf_op(["RandomNormalLike", "RandomUniformLike"])
class PassThroughOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass

@tf_op(["RandomShuffle"])
class RandomShuffleOp:
    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        inp_shape = ctx.make_node("Shape", [node.input[0]]).output[0]
        dim_0 = GraphBuilder(ctx).make_slice({'data': inp_shape, 'starts': [0], 'ends': [1], 'axes': [0]})
        zeros = ctx.make_node("ConstantOfShape", [dim_0], shapes=[[-1]]).output[0]

        seed = node.get_attr_value("seed", 0)
        seed2 = node.get_attr_value("seed2", 0)
        onnx_seed = utils.combine_seeds(seed, seed2)
        rand_attr = {'dtype': onnx_pb.TensorProto.FLOAT}
        if onnx_seed is not None:
            rand_attr['seed'] = onnx_seed

        random_floats = ctx.make_node("RandomUniformLike", [zeros], op_name_scope=node.name, shapes=[[-1]],
                                      attr=rand_attr).output[0]
        # Use indices of the TopK to get a random ordering
        _, random_ordering = ctx.make_node("TopK", [random_floats, dim_0], output_count=2, attr={'axis': -1}).output
        shuffled_res = ctx.make_node("Gather", [node.input[0], random_ordering]).output[0]
        ctx.replace_all_inputs(node.output[0], shuffled_res)

@tf_op("Fill")
class Fill:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # T output = Fill(int32 dims, T value, @int32 index_type)
        # T outputs = Tile(T value, int64 repeats (e.g. dims))
        fill_shape = ctx.get_shape(node.input[0])
        utils.make_sure(fill_shape is not None, "shape of {} is None".format(node.input[0]))
        fill_shape_dims = fill_shape[0]
        utils.make_sure(fill_shape_dims > 0, "opset 7 requires fill shape length > 0, or please try opset > 7")
        val_dtype = ctx.get_dtype(node.input[1])
        val_shape = ctx.get_shape(node.input[1])

        need_cast = val_dtype != onnx_pb.TensorProto.FLOAT and ctx.opset < 9
        new_dtype = val_dtype
        if need_cast:
            new_dtype = onnx_pb.TensorProto.FLOAT
            attr = {"to": new_dtype}
            cast_to_float = ctx.insert_new_node_on_input(node, "Cast", node.input[1], name=None, **attr)
            ctx.set_dtype(cast_to_float.output[0], new_dtype)
            ctx.set_shape(cast_to_float.output[0], val_shape)

        for _ in range(fill_shape_dims):
            attr = {"axes": [0]}
            shape = ctx.get_shape(node.input[1])
            unsqueeze_node = ctx.insert_new_node_on_input(node, "Unsqueeze", node.input[1], name=None, **attr)
            ctx.set_dtype(unsqueeze_node.output[0], new_dtype)
            if shape:
                shape = [1] + shape
            else:
                shape = [1]
            ctx.set_shape(unsqueeze_node.output[0], shape)

        # Tile's repeats must be INT64
        attr = {"to": onnx_pb.TensorProto.INT64}
        tile_shape_int64 = ctx.insert_new_node_on_input(node, "Cast", node.input[0], name=None, **attr)
        ctx.set_dtype(tile_shape_int64.output[0], onnx_pb.TensorProto.INT64)
        ctx.set_shape(tile_shape_int64.output[0], fill_shape)

        tmp = node.input[0]
        ctx.replace_input(node, node.input[0], node.input[1], 0)
        ctx.replace_input(node, node.input[1], tmp, 1)
        node.type = "Tile"
        ctx.set_dtype(node.output[0], new_dtype)

        if need_cast:
            attr = {"to": val_dtype}
            op_name = utils.make_name(node.name + "/cast_back")
            cast_back = ctx.insert_new_node_on_output("Cast", node.output[0], name=op_name, **attr)
            ctx.set_dtype(cast_back.output[0], val_dtype)

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        node.type = "ConstantOfShape"
        # both shape and value in tensorflow are passed as tensor.
        # In onnx the value is an attribute so we need to fetch the value as const which
        # sooner or later will be a problem for tensorflow-onnx.
        # ConstantOfShape in onnxruntime only support int64, so insert cast op
        input_dtype_is_int64 = utils.map_onnx_to_numpy_type(ctx.get_dtype(node.input[0])) == np.int64
        if not input_dtype_is_int64:
            ctx.insert_new_node_on_input(node, "Cast", node.input[0], to=onnx_pb.TensorProto.INT64)
        dtype = ctx.get_dtype(node.output[0])
        value = np.array([node.inputs[1].get_tensor_value()]).astype(utils.map_onnx_to_numpy_type(dtype))
        value_proto = numpy_helper.from_array(value)
        node.set_attr("value", value_proto)
        ctx.remove_input(node, node.input[1], 1)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # cls.version_7(ctx, node, **kwargs)
        node.type = "Expand"
        ctx.replace_inputs(node, [node.input[1], node.input[0]])
        # cast shape to int64 if needed
        if ctx.get_dtype(node.input[1]) != onnx_pb.TensorProto.INT64:
            ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=onnx_pb.TensorProto.INT64)


@tf_op("Multinomial")
class Multinomial:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # output_dtype output = Multinomial(T logits, int32 num_samples, @int seed, @int seed2, @type output_dtype)
        sample_size = node.inputs[1].get_tensor_value()
        seed = node.get_attr("seed")
        if seed:
            node.set_attr("seed", float(seed.i))
        output_dtype = node.get_attr("output_dtype")
        if output_dtype:
            output_dtype = output_dtype.i
        else:
            output_dtype = onnx_pb.TensorProto.INT32
        node.set_attr("dtype", output_dtype)
        node.set_attr("sample_size", sample_size)
        ctx.remove_input(node, node.input[1], 1)


def _const_like_version_1(ctx, node, value):
    shapes = node.output_shapes
    dtypes = node.output_dtypes
    ctx.remove_node(node.name)
    casted_input = ctx.make_node("Cast", node.input, attr={'to': onnx_pb.TensorProto.INT64})
    const_value = ctx.make_const(utils.make_name("value"), np.array(value).astype(np.int64))
    mul_node = ctx.make_node('Mul', inputs=[casted_input.output[0], const_value.output[0]])
    ctx.make_node("Cast", inputs=[mul_node.output[0]],
                  attr={'to': dtypes[0]},
                  name=node.name, outputs=node.output,
                  shapes=shapes, dtypes=dtypes)


def _const_like_version_9(ctx, node, value):
    dtypes = node.output_dtypes
    ctx.remove_node(node.name)
    shape = ctx.make_node("Shape", node.input).output[0]
    value_tensor = helper.make_tensor("value", dtypes[0], [1], vals=[value])
    ctx.make_node("ConstantOfShape", inputs=[shape],
                  attr={'value': value_tensor},
                  name=node.name, outputs=node.output,
                  dtypes=dtypes)


@tf_op("ZerosLike")
class ZerosLike:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        _const_like_version_1(ctx, node, 0)

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        _const_like_version_9(ctx, node, 0)


@tf_op("OnesLike")
class OnesLike:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        _const_like_version_1(ctx, node, 1)

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        _const_like_version_9(ctx, node, 1)


@tf_op(["IteratorV2", "FIFOQueueV2"])
class Iterator:
    @classmethod
    def version_8(cls, ctx, node, **kwargs):
        ctx.remove_node(node.name)


@tf_op(["IteratorGetNext", "QueueDequeueV2"])
class IteratorGetNext:
    @classmethod
    def version_8(cls, ctx, node, **kwargs):
        output_names = node.output.copy()  # to make sure remove_node
                                           # does not alter the list
        type_0 = ctx.get_dtype(output_names[0])
        type_1 = ctx.get_dtype(output_names[1])
        shape_0 = ctx.get_shape(output_names[0])
        shape_1 = ctx.get_shape(output_names[1])
        ctx.remove_node(node.name)
        ctx.add_graph_input(output_names[0], type_0, shape_0)
        ctx.add_graph_input(output_names[1], type_1, shape_1)


@tf_op(["QueueDequeueManyV2", "QueueDequeueUpToV2"])
class QueueDequeueManyV2:
    @classmethod
    def version_8(cls, ctx, node, **kwargs):
        outputs = node.output.copy()  # copy to make remove_node
                                      # does not alter the list
        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        for i, output in enumerate(outputs):
            ctx.add_graph_input(output, dtypes[i], shapes[i])
