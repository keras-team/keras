# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.shape_inference - shape inference function for tf2onnx
"""

import logging
from collections import defaultdict
import numpy as np
from packaging.version import Version
from tf2onnx import utils
from tf2onnx.tf_utils import get_tf_tensor_shape, get_tf_const_value, get_tf_shape_attr, get_tf_version
from tf2onnx.tf_loader import tf_reload_graph

# pylint: disable=logging-not-lazy,missing-docstring,consider-swap-variables


logger = logging.getLogger(__name__)


def infer_shape(tf_graph, shape_override):
    """Infer shape for TF graph with shape_override set first."""
    if shape_override:
        logger.info("Apply shape override:")
        for name, shape in shape_override.items():
            logger.info("\tSet %s shape to %s", name, shape)
            tf_graph.get_tensor_by_name(name).set_shape(shape)
        tf_graph = tf_reload_graph(tf_graph)

    tf_graph = infer_shape_for_graph(tf_graph)

    op_outputs_with_none_shape = check_shape_for_tf_graph(tf_graph)
    if op_outputs_with_none_shape:
        if get_tf_version() > Version("1.5.0"):
            for op, outs in op_outputs_with_none_shape.items():
                logger.warning(
                    "Cannot infer shape for %s: %s",
                    op, ",".join(outs)
                )
        tf_graph = infer_shape_for_graph_legacy(tf_graph)

    return tf_graph


def check_shape_for_tf_graph(tf_graph):
    """
    Check whether TF graph misses any shape,
    and return all ops with None shape outputs for TF graph.
    """
    skip_list = {'FusedBatchNormV3': 5}
    op_outputs_mapping_none_shape = defaultdict(list)
    for op in tf_graph.get_operations():
        for i, out in enumerate(op.outputs):
            if op.type in skip_list:
                if skip_list[op.type] == i:
                    continue
            if get_tf_tensor_shape(out) is None:
                op_outputs_mapping_none_shape[op.name].append(out.name)
    return op_outputs_mapping_none_shape


def infer_shape_for_graph(tf_graph):
    """
    Infer shape for Tensorflow ops.
    Tensorflow explicitly sets shape for some ops in python code, such as Switch, Merge and TensorArrayGather.
    These shapes may be lost after freezing TF graph to graph_def without add_shapes=True.
    To bring these shapes back, we implement our own shape inference for these control flow ops based on one assumption:
    **outputs of Merge op have the same shape (at least the same rank) of its inputs**.
    With this assumption, our shape inference can handle:
        1. in tf.cond, outputs of two branches have the same rank.
        2. in tf.while_loop, loop variables don't change their rank.
    """
    shape_updated = True
    while shape_updated:
        shape_updated = False
        for o in tf_graph.get_operations():
            updated = infer_shape_for_op(o)
            if updated:
                shape_updated = True
        if shape_updated:
            tf_graph = tf_reload_graph(tf_graph)
    return tf_graph


def infer_shape_for_op(op):
    has_unknown_output_shape = any(get_tf_tensor_shape(out) is None for out in op.outputs)

    if not has_unknown_output_shape:
        return False

    if op.type == "Placeholder":
        # if placeholder shape is not found, try to get it from "shape" attribute.
        attr_shape = get_tf_shape_attr(op)
        if attr_shape is not None:
            new_shape = list(attr_shape)
            op.outputs[0].set_shape(new_shape)
            logger.debug("set placeholder op [%s] with new shape %s", op.outputs[0].name, new_shape)
            return True
        logger.warning("Shape of placeholder '%s' is unknown, treated it as a scalar. Please use the --inputs flag "
                       "and append the shape to the input name if this input is not a scalar.", op.name)
        op.outputs[0].set_shape([])
        return True

    if op.type == "Merge":
        s1 = get_tf_tensor_shape(op.inputs[0])
        s2 = get_tf_tensor_shape(op.inputs[1])
        new_shape = None
        if s1 is None and s2 is None:
            return False
        if s1 is None and s2 is not None:
            new_shape = s2
        if s1 is not None and s2 is None:
            new_shape = s1

        if new_shape is not None:
            op.inputs[0].set_shape(new_shape)
            op.inputs[1].set_shape(new_shape)
            op.outputs[0].set_shape(new_shape)
            logger.debug("set [%s] with new shape %s", op.outputs[0].name, new_shape)
            return True

        # inputs' shapes both exist
        if s1 != s2:
            if len(s1) != len(s2):
                logger.warning("Shapes of Merge %s have different ranks: %s, %s", op.name, len(s1), len(s2))
                return False

            logger.debug("Inputs of Merge %s have different shapes: %s, %s, but the same rank", op.name, s1, s2)
            new_shape = _merge_shapes_for_tf(s1, s2)
            op.outputs[0].set_shape(new_shape)
            logger.debug("set [%s] with new shape %s", op.outputs[0].name, new_shape)
        else:
            new_shape = s1
            op.outputs[0].set_shape(new_shape)
            logger.debug("set [%s] with new shape %s", op.outputs[0].name, new_shape)

        return True

    if op.type == "Switch":
        new_shape = get_tf_tensor_shape(op.inputs[0])
        if new_shape is not None:
            op.outputs[0].set_shape(new_shape)
            op.outputs[1].set_shape(new_shape)
            logger.debug("set [%s] with new shape %s", op.outputs[0].name, new_shape)
            logger.debug("set [%s] with new shape %s", op.outputs[1].name, new_shape)
            return True
        return False

    if op.type == "Enter":
        new_shape = get_tf_tensor_shape(op.inputs[0])
        if new_shape is not None:
            op.outputs[0].set_shape(new_shape)
            logger.debug("set [%s] with new shape %s", op.outputs[0].name, new_shape)
            return True
        return False

    if op.type == "TensorArrayGatherV3":
        # TensorArrayGatherV3's output: all of the elem in the TensorArray,
        # concatenated along a new axis (the new dimension 0), so shape of TensorArray should be found first.
        # And TensorArrayWrite will write elem to TensorArray, so shape of TensorArray can be got from TensorArrayWrite
        # so the process is: first find TensorArrayWrite and then get TensorArray's shape,
        # and finally add one dim to the shape is shape of TensorArrayGather

        handle_op = op.inputs[0].op
        if handle_op.type != "TensorArrayV3":
            return False

        # find TensorArrayWrite
        tensor_array_write_op = _find_tensorarray_write(handle_op)
        if not tensor_array_write_op:
            return False
        # get TensorArray shape from input tensor of the found TensorArrayWrite op
        shape = get_tf_tensor_shape(tensor_array_write_op.inputs[2])
        # update TensorArray's shape info
        if shape is not None:
            new_shape = [None] + shape
            op.outputs[0].set_shape(new_shape)
            logger.debug("set [%s] with new shape %s", op.outputs[0].name, new_shape)
            return True
        return False

    if op.type == "TensorArrayReadV3":
        # TensorArrayRead reads an element from the TensorArray into output value.
        # The TensorArray's shape can be got from TensorArrayScatter.
        # So the process is: first find TensorArrayScatter's shape and then TensorArray's
        # and finally take its last n-1 dim.
        flow_in_op = op.inputs[2].op
        if flow_in_op.type != "Enter":
            return False

        scatter_op = flow_in_op.inputs[0].op
        if scatter_op.type != "TensorArrayScatterV3":
            return False

        value_shape_before_scatter = get_tf_tensor_shape(scatter_op.inputs[2])
        if value_shape_before_scatter is None:
            return False

        new_shape = value_shape_before_scatter[1:]
        if new_shape is not None:
            op.outputs[0].set_shape(new_shape)
            logger.debug("set [%s] with new shape %s", op.outputs[0].name, new_shape)
            return True
        return False

    return False


def _find_tensorarray_write(op):
    utils.make_sure(op.type == "TensorArrayV3", "op should be tensorarray")

    tensor_array_consumers = op.outputs[0].consumers()
    for i in tensor_array_consumers:
        if i.type == "Enter":
            consumer_ops = i.outputs[0].consumers()
            for j in consumer_ops:
                if j.type == "TensorArrayWriteV3":
                    return j
    return None


def _merge_shapes_for_tf(shape1, shape2):
    """
    Merge 2 shapes, return merged shape, set unknown for dims with different values.
    Raise exception for mismatch.
    """
    if shape1 is None:
        return shape2
    if shape2 is None:
        return shape1

    utils.make_sure(utils.is_list_or_tuple(shape1), "invalid type for shape1")
    utils.make_sure(utils.is_list_or_tuple(shape2), "invalid type for shape2")
    utils.make_sure(len(shape1) == len(shape2), "shapes rank mismatch: shape1=%s, shape2=%s", shape1, shape2)

    merged = []
    for d1, d2 in zip(shape1, shape2):
        d = d1
        if d1 is None:
            d = d2
        elif d2 is not None:
            # None means unknown in tensorflow
            d = None
        merged.append(d)
    return merged


######################################################################
####   Below is our old tf shape inference as a supplementary     ####
####            and a subtitute for TF 1.5.0                      ####
######################################################################

direct_ops = [
    "Cast",
    "Exit",
    "Floor",
    "Identity",
    "LogicalNot",
    "ReverseSequence",
    "Relu6",
    "Sigmoid",
    "Square",
    "Tanh"
]
broadcast_ops = [
    "Add",
    "Greater",
    "GreaterEqual",
    "Less",
    "LessEqual",
    "LogicalAnd",
    "LogicalOr",
    "Mul",
    "RealDiv",
    "Sub"
]


def infer_shape_for_graph_legacy(tf_graph):
    shape_updated = True
    while shape_updated:
        shape_updated = False
        for op in tf_graph.get_operations():
            updated = infer_shape_for_op_legacy(op)
            if updated:
                shape_updated = True

    return tf_graph


def infer_shape_for_op_legacy(op):
    # invoke tf shape inference first
    infer_shape_for_op(op)

    has_unknown_input_shape = any(get_tf_tensor_shape(inp) is None for inp in op.inputs)
    has_unknown_output_shape = any(get_tf_tensor_shape(out) is None for out in op.outputs)

    # an input shape may be inferred from op output or other input shapes
    # try to infer it first
    if has_unknown_input_shape:
        if infer_input_shapes(op):
            return True

    if not has_unknown_output_shape:
        return False

    # for those ops, we don't expect all input shapes available to infer output shapes.
    ret = infer_output_shapes_with_partial_inputs(op)
    if ret is not None:
        return ret

    # for ops, we need all input shapes ready to infer output shapes.
    are_all_input_shape_ready = True
    no_shape = []
    for i in op.inputs:
        if get_tf_tensor_shape(i) is None:
            are_all_input_shape_ready = False
            no_shape.append(i.name)

    if not are_all_input_shape_ready:
        logger.debug("op %s has inputs don't have shape specified, they are: %s", op.name, no_shape)
        return False

    if op.type in direct_ops:
        return set_shape_from_input(op.inputs[0], op.outputs[0])

    if op.type in broadcast_ops:
        return set_shape_from_inputs_broadcast(op.inputs, op.outputs[0])

    if op.type == "RandomUniform":
        shape_op = op.inputs[0].op
        if not shape_op or shape_op.type != "Shape":
            return False
        return set_shape_from_input(shape_op.inputs[0], op.outputs[0])

    if op.type == "Gather":
        # uses the follwing link to know how to infer shape of output
        # https://www.tensorflow.org/api_docs/python/tf/gather
        shape_params = get_tf_tensor_shape(op.inputs[0])
        shape_indices = get_tf_tensor_shape(op.inputs[1])
        # gather can only have 2 inputs
        # https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/gather.html
        if len(op.inputs) == 3:
            axis_op = op.inputs[2].op
            if not utils.is_tf_const_op(axis_op):
                return False
            axis = get_tf_const_value(axis_op)
        else:
            axis = 0

        shape = shape_params[:axis] + shape_indices + shape_params[axis + 1:]
        op.outputs[0].set_shape(shape)
        return True

    if op.type in ["All", "Any", "Max", "Min"]:
        axis_op = op.inputs[1].op
        if not utils.is_tf_const_op(axis_op):
            return False
        axis = get_tf_const_value(axis_op)
        if not isinstance(axis, list):
            axis = [axis]
        keep_dims = op.get_attr("keep_dims")
        shape = get_tf_tensor_shape(op.inputs[0])
        for i, _ in enumerate(axis):
            if axis[i] < 0:
                axis[i] += len(shape)

        new_shape = []
        for i, _ in enumerate(shape):
            if i in axis:
                if keep_dims:
                    new_shape.append(1)
            else:
                new_shape.append(shape[i])

        op.outputs[0].set_shape(new_shape)
        logger.debug("set %s op [%s] with new shape %s", op.type, op.outputs[0].name, new_shape)
        return True

    if op.type == "ExpandDims":
        # https://www.tensorflow.org/api_docs/python/tf/expand_dims
        input_shape = get_tf_tensor_shape(op.inputs[0])
        dim_op = op.inputs[1].op
        if input_shape is None or not utils.is_tf_const_op(dim_op):
            return False

        dim = get_tf_const_value(dim_op)
        if dim < 0:
            dim = dim + len(input_shape) + 1

        new_shape = input_shape[:dim] + [1] + input_shape[dim:]
        op.outputs[0].set_shape(new_shape)
        logger.debug("set [%s] with new shape %s", op.outputs[0].name, new_shape)
        return True

    if op.type == "Unpack":
        input_shape = get_tf_tensor_shape(op.inputs[0])
        if input_shape is None:
            return False

        axis = op.get_attr("axis")
        axis = axis if axis >= 0 else axis + len(input_shape)
        # the link below says that the rank of output is "rank(input) -1",
        # from this statement "num" must equal to input_shape[axis], and if not tf will throw a runtime error
        # https://www.tensorflow.org/api_docs/python/tf/unstack
        new_shape = input_shape[:axis] + input_shape[axis + 1:]
        for output in op.outputs:
            output.set_shape(new_shape)
            logger.debug("set %s op [%s] with new shape %s", op.type, output.name, new_shape)
        return True

    if op.type in ["Minimum", "Maximum"]:
        # ops that are elementwise and support broadcasting
        input_shapes = [get_tf_tensor_shape(op) for op in op.inputs]
        new_shape = broadcast_shape_inference(*input_shapes)
        op.outputs[0].set_shape(new_shape)
        return True

    return False


def infer_input_shapes(op):
    if op.type in ["Select", "SelectV2"]:
        shape_t = get_tf_tensor_shape(op.inputs[1])
        shape_e = get_tf_tensor_shape(op.inputs[2])
        # copy shape if t OR e does not have a shape, no update if t AND e both have shapes
        if shape_t is None or shape_e is None:
            new_shape = shape_t or shape_e
            if new_shape is not None:
                op.inputs[1].set_shape(new_shape)
                op.inputs[2].set_shape(new_shape)
                logger.debug("set [%s, %s] with new shape %s", op.inputs[1].name, op.inputs[2].name, new_shape)
                return True
    return False


def infer_output_shapes_with_partial_inputs(op):
    # output shape of concat op: only the dim val of concatenated dim will be changed
    # so only partial(at least one) input shapes need to be known to infer output shape of concat op
    if utils.is_tf_concat_op(op):
        data_inputs = op.inputs[:-1]
        input_shapes = [get_tf_tensor_shape(inp) for inp in data_inputs]
        input_shapes = [shape for shape in input_shapes if shape is not None]
        if not input_shapes:
            logger.debug("all input shapes of concat op %s are None, can't infer its output shape", op.name)
            return False

        new_shape = input_shapes[0]
        axis_op = op.inputs[-1]
        rank = len(new_shape)
        if not utils.is_tf_const_op(axis_op):
            op.outputs[0].set_shape([-1] * rank)
            return True

        axis = get_tf_const_value(axis_op)
        axis = axis if axis >= 0 else axis + rank
        new_shape[axis] = -1
        if len(input_shapes) == len(data_inputs):  # all input shapes are known
            concat_dim_vals = list(np.array(input_shapes)[:, axis])
            # only when inputs' shape are known, then val of concat dim can be calculated
            if concat_dim_vals.count(-1) == 0:
                new_shape[axis] = sum(concat_dim_vals)

        op.outputs[0].set_shape(new_shape)
        logger.debug("set Concat op [%s] with new shape %s", op.outputs[0].name, new_shape)
        return True

    if op.type in ["Select", "SelectV2"]:
        new_shape = get_tf_tensor_shape(op.inputs[1])
        if new_shape is None:
            new_shape = get_tf_tensor_shape(op.inputs[2])
        if new_shape is not None:
            op.outputs[0].set_shape(new_shape)
            op.inputs[1].set_shape(new_shape)
            op.inputs[2].set_shape(new_shape)
            logger.debug("set [%s] with new shape %s", op.outputs[0].name, new_shape)
            return True
        return False

    if op.type == "Pack":
        axis = op.get_attr("axis")
        input_shape = None
        for i in op.inputs:
            s = get_tf_tensor_shape(i)
            if s is not None:
                input_shape = s
                break
        if input_shape is None:
            return False
        if axis < 0:
            axis += len(input_shape)
        for i in op.inputs:
            if not get_tf_tensor_shape(i):
                i.set_shape(input_shape)
                logger.debug("set [%s] with new shape %s", i.name, input_shape)
        new_shape = input_shape[:axis] + [len(op.inputs)] + input_shape[axis:]
        op.outputs[0].set_shape(new_shape)
        logger.debug("set Pack op [%s] with new shape %s", op.outputs[0].name, new_shape)
        return True

    if op.type == "Pow":
        # https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/pow
        new_shape = get_tf_tensor_shape(op.inputs[0])
        if new_shape is None:
            new_shape = get_tf_tensor_shape(op.inputs[1])
        if new_shape is not None:
            op.outputs[0].set_shape(new_shape)
            logger.debug("set [%s] with new shape %s", op.outputs[0].name, new_shape)
            return True
        return False

    return None


def set_shape_from_input(input_tensor, output_tensor):
    new_shape = get_tf_tensor_shape(input_tensor)
    if new_shape is not None:
        output_tensor.set_shape(new_shape)
        logger.debug("set [%s] with new shape %s", output_tensor.name, new_shape)
        return True
    return False


def set_shape_from_inputs_broadcast(input_tensors, output_tensor):
    s1 = get_tf_tensor_shape(input_tensors[0])
    s2 = get_tf_tensor_shape(input_tensors[1])
    new_shape = broadcast_shape_inference(s1, s2)
    if new_shape is not None:
        output_tensor.set_shape(new_shape)
        logger.debug("set [%s] with new shape %s", output_tensor.name, new_shape)
        return True
    return False


def broadcast_shape_inference(shape_0, shape_1):
    if shape_0 is None:
        return shape_1
    if shape_1 is None:
        return shape_0

    # two dimensions are compatible when they are equal, or one of them is 1
    # compare from last dim
    if len(shape_0) > len(shape_1):
        tmp = shape_0
        shape_0 = shape_1
        shape_1 = tmp

    new_shape = shape_1
    l = len(shape_0)
    if l == 0:
        return new_shape

    i = l - 1
    while i >= 0:
        if shape_0[i] == shape_1[i]:
            # do nothing
            pass
        elif shape_0[i] == 1:
            # do nothing
            pass
        elif shape_1[i] == 1:
            new_shape[i] = shape_0[i]
        # maybe one of them is -1, we can use the other one as real shape.
        elif shape_0[i] == -1:
            pass
        elif shape_1[i] == -1:
            new_shape[i] = shape_0[i]
        else:
            logger.warning("two shapes not possible to broadcast, %s, %s", shape_0, shape_1)
            return None
        i -= 1
    return new_shape
