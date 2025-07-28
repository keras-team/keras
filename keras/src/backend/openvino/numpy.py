import numpy as np
import openvino.runtime.opset14 as ov_opset
from openvino import Type

from keras.src.backend import config
from keras.src.backend.common import dtypes
from keras.src.backend.common.variables import standardize_dtype
from keras.src.backend.openvino.core import DTYPES_MAX
from keras.src.backend.openvino.core import DTYPES_MIN
from keras.src.backend.openvino.core import OPENVINO_DTYPES
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import (
    align_operand_types as _align_operand_types,
)
from keras.src.backend.openvino.core import convert_to_tensor
from keras.src.backend.openvino.core import get_ov_output
from keras.src.backend.openvino.core import ov_to_keras_type


def add(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "add()")
    return OpenVINOKerasTensor(ov_opset.add(x1, x2).output(0))


def einsum(subscripts, *operands, **kwargs):
    inputs = []
    for operand in operands:
        operand = get_ov_output(operand)
        inputs.append(operand)
    return OpenVINOKerasTensor(ov_opset.einsum(inputs, subscripts).output(0))


def subtract(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "subtract()")
    return OpenVINOKerasTensor(ov_opset.subtract(x1, x2).output(0))


def matmul(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "matmul()")
    return OpenVINOKerasTensor(ov_opset.matmul(x1, x2, False, False).output(0))


def multiply(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "multiply()")
    return OpenVINOKerasTensor(ov_opset.multiply(x1, x2).output(0))


def mean(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    if axis is None:
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        x = ov_opset.reshape(x, flatten_shape, False).output(0)
        axis = 0
    axis_const = ov_opset.constant(axis, dtype=Type.i32).output(0)
    mean_ops = ov_opset.reduce_mean(x, axis_const, keepdims)
    return OpenVINOKerasTensor(mean_ops.output(0))


def max(x, axis=None, keepdims=False, initial=None):
    assert initial is None, (
        "`max` with not None initial is not supported by openvino backend"
    )
    x = get_ov_output(x)
    reduce_axis = ov_opset.constant(axis, Type.i32).output(0)
    return OpenVINOKerasTensor(
        ov_opset.reduce_max(x, reduce_axis, keepdims).output(0)
    )


def ones(shape, dtype=None):
    dtype = standardize_dtype(dtype) or config.floatx()
    ov_type = OPENVINO_DTYPES[dtype]
    const_one = ov_opset.constant(1, ov_type).output(0)
    if isinstance(shape, tuple):
        shape = list(shape)
    elif isinstance(shape, int):
        shape = [shape]
    output_shape = ov_opset.constant(shape, dtype=Type.i32).output(0)
    ones = ov_opset.broadcast(const_one, output_shape)
    return OpenVINOKerasTensor(ones.output(0))


def zeros(shape, dtype=None):
    dtype = standardize_dtype(dtype) or config.floatx()
    ov_type = OPENVINO_DTYPES[dtype]
    const_zero = ov_opset.constant(0, dtype=ov_type).output(0)
    if isinstance(shape, tuple):
        shape = list(shape)
    elif isinstance(shape, int):
        shape = [shape]
    output_shape = ov_opset.constant(shape, dtype=Type.i32).output(0)
    zeros = ov_opset.broadcast(const_zero, output_shape)
    return OpenVINOKerasTensor(zeros.output(0))


def absolute(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.absolute(x).output(0))


def abs(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.absolute(x).output(0))


def all(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    if axis is None:
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        x = ov_opset.reshape(x, flatten_shape, False).output(0)
        axis = 0
    axis = ov_opset.constant(axis, Type.i32).output(0)
    return OpenVINOKerasTensor(
        ov_opset.reduce_logical_and(x, axis, keepdims).output(0)
    )


def angle(x):
    raise NotImplementedError("`angle` is not supported with openvino backend")


def any(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    if axis is None:
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        x = ov_opset.reshape(x, flatten_shape, False).output(0)
        axis = 0
    axis = ov_opset.constant(axis, Type.i32).output(0)
    return OpenVINOKerasTensor(
        ov_opset.reduce_logical_or(x, axis, keepdims).output(0)
    )


def amax(x, axis=None, keepdims=False):
    if axis == () or axis == []:
        return x
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if axis is None:
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        x = ov_opset.reshape(x, flatten_shape, False).output(0)
        axis = 0
    if isinstance(axis, tuple):
        axis = list(axis)
    axis = ov_opset.constant(axis, Type.i32).output(0)
    if x_type == Type.boolean:
        return OpenVINOKerasTensor(
            ov_opset.reduce_logical_or(x, axis, keepdims).output(0)
        )
    return OpenVINOKerasTensor(ov_opset.reduce_max(x, axis, keepdims).output(0))


def amin(x, axis=None, keepdims=False):
    if axis == () or axis == []:
        return x
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if axis is None:
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        x = ov_opset.reshape(x, flatten_shape, False).output(0)
        axis = 0
    if isinstance(axis, tuple):
        axis = list(axis)
    axis = ov_opset.constant(axis, Type.i32).output(0)
    if x_type == Type.boolean:
        return OpenVINOKerasTensor(
            ov_opset.reduce_logical_and(x, axis, keepdims).output(0)
        )
    return OpenVINOKerasTensor(ov_opset.reduce_min(x, axis, keepdims).output(0))


def append(x1, x2, axis=None):
    x1, x2 = get_ov_output(x1), get_ov_output(x2)
    x1, x2 = _align_operand_types(x1, x2, "append()")
    if axis is None:
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        x1 = ov_opset.reshape(x1, flatten_shape, False).output(0)
        x2 = ov_opset.reshape(x2, flatten_shape, False).output(0)
        axis = 0
    return OpenVINOKerasTensor(ov_opset.concat([x1, x2], axis).output(0))


def arange(start, stop=None, step=1, dtype=None):
    if stop is None:
        start, stop = get_ov_output(0), get_ov_output(start)
    else:
        start, stop = get_ov_output(start), get_ov_output(stop)

    step = get_ov_output(1) if step is None else get_ov_output(step)

    ov_type = None
    if dtype is not None:
        ov_type = OPENVINO_DTYPES[standardize_dtype(dtype)]
    else:
        ov_type = OPENVINO_DTYPES[
            dtypes.result_type(
                ov_to_keras_type(start.get_element_type()),
                ov_to_keras_type(stop.get_element_type()),
                ov_to_keras_type(step.get_element_type()),
                "int32",
            )
        ]

    start_node = ov_opset.convert(start, ov_type)
    stop_node = ov_opset.convert(stop, ov_type)
    step_node = ov_opset.convert(step, ov_type)

    return OpenVINOKerasTensor(
        ov_opset.range(start_node, stop_node, step_node, ov_type).output(0)
    )


def arccos(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, ov_type)
    return OpenVINOKerasTensor(ov_opset.acos(x).output(0))


def arccosh(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, ov_type)
    return OpenVINOKerasTensor(ov_opset.acosh(x).output(0))


def arcsin(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, ov_type)
    return OpenVINOKerasTensor(ov_opset.asin(x).output(0))


def arcsinh(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, ov_type)
    return OpenVINOKerasTensor(ov_opset.asinh(x).output(0))


def arctan(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, ov_type)
    return OpenVINOKerasTensor(ov_opset.atan(x).output(0))


def arctan2(x1, x2):
    x1 = get_ov_output(x1)
    x2 = get_ov_output(x2)

    x1_type = ov_to_keras_type(x1.get_element_type())
    x2_type = ov_to_keras_type(x2.get_element_type())
    result_type = dtypes.result_type(x1_type, x2_type, float)
    result_type = OPENVINO_DTYPES[result_type]
    x1 = ov_opset.convert(x1, result_type)
    x2 = ov_opset.convert(x2, result_type)

    x = ov_opset.divide(x1, x2)
    y = ov_opset.atan(x)

    ov_type = x1.get_element_type()
    pi = ov_opset.constant(float(np.pi), ov_type)
    half_pi = ov_opset.constant(float(np.pi / 2), ov_type)
    neg_half_pi = ov_opset.constant(-float(np.pi / 2), ov_type)
    zero_const = ov_opset.constant(0.0, ov_type)

    cond_x2_gt0 = ov_opset.greater(x2, zero_const).output(0)
    cond_x2_lt0 = ov_opset.less(x2, zero_const).output(0)

    cond_x1_ge0 = ov_opset.greater_equal(x1, zero_const).output(0)
    cond_x1_gt0 = ov_opset.greater(x1, zero_const).output(0)
    cond_x1_eq0 = ov_opset.equal(x1, zero_const).output(0)

    out_x2_lt0 = ov_opset.select(
        cond_x1_ge0,
        ov_opset.add(y, pi),
        ov_opset.subtract(y, pi),
    )

    out_x1_zero = ov_opset.select(cond_x1_eq0, zero_const, neg_half_pi)
    out_x2_zero = ov_opset.select(cond_x1_gt0, half_pi, out_x1_zero)

    out_not_pos = ov_opset.select(cond_x2_lt0, out_x2_lt0, out_x2_zero)

    final_out = ov_opset.select(cond_x2_gt0, y, out_not_pos)
    return OpenVINOKerasTensor(final_out.output(0))


def arctanh(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, ov_type)
    return OpenVINOKerasTensor(ov_opset.atanh(x).output(0))


def argmax(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    x_shape = x.get_partial_shape()
    rank = x_shape.rank.get_length()
    if rank == 0:
        return OpenVINOKerasTensor(ov_opset.constant([0], Type.i32).output(0))
    if axis is None:
        flatten_shape = ov_opset.constant(
            [-1] + [1] * (rank - 1), Type.i32
        ).output(0)
        x = ov_opset.reshape(x, flatten_shape, False).output(0)
        axis = 0
        k = ov_opset.constant(1, Type.i32).output(0)
    else:
        if axis < 0:
            axis = rank + axis
        k = ov_opset.constant(1, Type.i32).output(0)
    topk_outputs = ov_opset.topk(
        x,
        k=k,
        axis=axis,
        mode="max",
        sort="value",
        stable=True,
        index_element_type=Type.i32,
    )
    topk_indices = topk_outputs.output(1)
    if not keepdims:
        topk_indices = ov_opset.squeeze(topk_indices, [axis]).output(0)
    return OpenVINOKerasTensor(topk_indices)


def argmin(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    x_shape = x.get_partial_shape()
    rank = x_shape.rank.get_length()
    if rank == 0:
        return OpenVINOKerasTensor(ov_opset.constant([0], Type.i32).output(0))
    if axis is None:
        flatten_shape = ov_opset.constant(
            [-1] + [1] * (rank - 1), Type.i32
        ).output(0)
        x = ov_opset.reshape(x, flatten_shape, False).output(0)
        axis = 0
        k = ov_opset.constant(1, Type.i32).output(0)
    else:
        if axis < 0:
            axis = rank + axis
        k = ov_opset.constant(1, Type.i32).output(0)
    topk_outputs = ov_opset.topk(
        x,
        k=k,
        axis=axis,
        mode="min",
        sort="value",
        stable=True,
        index_element_type=Type.i32,
    )
    topk_indices = topk_outputs.output(1)
    if not keepdims:
        topk_indices = ov_opset.squeeze(topk_indices, [axis]).output(0)
    return OpenVINOKerasTensor(topk_indices)


def argsort(x, axis=-1):
    x = get_ov_output(x)
    x_shape = x.get_partial_shape()
    rank = x_shape.rank.get_length()
    if rank == 0:
        return OpenVINOKerasTensor(ov_opset.constant([0], Type.i32).output(0))
    if axis is None:
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        x = ov_opset.reshape(x, flatten_shape, False).output(0)
        x_shape_tensor = ov_opset.shape_of(x, Type.i32).output(0)
        k = ov_opset.reduce_prod(
            x_shape_tensor, ov_opset.constant([0], Type.i32), keep_dims=False
        )
        axis = 0
    else:
        if axis < 0:
            axis = rank + axis
        x_shape_tensor = ov_opset.shape_of(x, Type.i32).output(0)
        k = ov_opset.gather(
            x_shape_tensor,
            ov_opset.constant(axis, Type.i32).output(0),
            ov_opset.constant(0, Type.i32).output(0),
        ).output(0)
    sorted_indices = ov_opset.topk(
        x,
        k=k,
        axis=axis,
        mode="min",
        sort="value",
    ).output(1)
    return OpenVINOKerasTensor(sorted_indices)


def array(x, dtype=None):
    if dtype is not None:
        return np.array(x, dtype=dtype)
    return np.array(x)


def average(x, axis=None, weights=None):
    x = get_ov_output(x)
    if weights is not None:
        weights = get_ov_output(weights)
    if axis is None:
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        x = ov_opset.reshape(x, flatten_shape, False).output(0)
        if weights is not None:
            weights = ov_opset.reshape(weights, flatten_shape, False).output(0)
        axis = 0

    if weights is not None:
        x_type = x.get_element_type()
        weights_type = weights.get_element_type()
        if (weights_type.is_integral() or weights_type == Type.boolean) and (
            x_type.is_integral() or x_type == Type.boolean
        ):
            x = ov_opset.convert(x, Type.f32).output(0)
            weights = ov_opset.convert(weights, Type.f32).output(0)
        x, weights = _align_operand_types(x, weights, "multiply()")
        x = ov_opset.multiply(x, weights)

    if isinstance(axis, tuple):
        axis = list(axis)
    if axis == []:
        return OpenVINOKerasTensor(x)

    axis_const = ov_opset.constant(axis, dtype=Type.i32).output(0)
    mean_ops = ov_opset.reduce_mean(x, axis_const, False)
    return OpenVINOKerasTensor(mean_ops.output(0))


def bartlett(x):
    raise NotImplementedError(
        "`bartlett` is not supported with openvino backend"
    )


def hamming(x):
    raise NotImplementedError(
        "`hamming` is not supported with openvino backend"
    )


def heaviside(x1, x2):
    raise NotImplementedError(
        "`heaviside` is not supported with openvino backend"
    )


def kaiser(x, beta):
    raise NotImplementedError("`kaiser` is not supported with openvino backend")


def bincount(x, weights=None, minlength=0, sparse=False):
    if x is None:
        raise ValueError("input x is None")
    if sparse:
        raise ValueError("Unsupported value `sparse=True`")
    x = get_ov_output(x)
    x_type = x.get_element_type()
    shape_x = ov_opset.shape_of(x, "i64").output(0)
    rank_x = ov_opset.shape_of(shape_x, "i64").output(0)
    rank_x = ov_opset.convert(rank_x, x_type).output(0)
    scalar_shape = ov_opset.constant([], x_type).output(0)
    rank_x = ov_opset.reshape(rank_x, scalar_shape, False).output(0)
    const_minus_one = ov_opset.constant(-1, x_type).output(0)
    rank_minus_one = ov_opset.add(rank_x, const_minus_one).output(0)
    minlength = get_ov_output(minlength)
    minlength = ov_opset.convert(minlength, x_type).output(0)
    const_one = ov_opset.constant(1, x_type).output(0)
    const_zero = ov_opset.constant(0, x_type).output(0)
    max_element = ov_opset.reduce_max(x, const_zero, keep_dims=False).output(0)
    depth = ov_opset.add(max_element, const_one).output(0)
    depth = ov_opset.maximum(depth, minlength).output(0)
    depth_scalar = ov_opset.reduce_max(
        depth, const_zero, keep_dims=False
    ).output(0)
    one_hot = ov_opset.one_hot(
        x, depth_scalar, const_one, const_zero, axis=-1
    ).output(0)
    if weights is not None:
        weights = get_ov_output(weights)
        weights_type = weights.get_element_type()
        weights_new = ov_opset.reshape(weights, [-1, 1], False).output(0)
        one_hot = ov_opset.convert(one_hot, weights_type).output(0)
        final_one_hot = ov_opset.multiply(one_hot, weights_new).output(0)
        final_output = ov_opset.reduce_sum(
            final_one_hot, rank_minus_one, keep_dims=False
        ).output(0)
        return OpenVINOKerasTensor(final_output)
    else:
        final_output = ov_opset.reduce_sum(
            one_hot, rank_minus_one, keep_dims=False
        ).output(0)
        final_output = ov_opset.convert(final_output, Type.i32).output(0)
        return OpenVINOKerasTensor(final_output)


def blackman(x):
    raise NotImplementedError(
        "`blackman` is not supported with openvino backend"
    )


def broadcast_to(x, shape):
    assert isinstance(shape, (tuple, list)), (
        "`broadcast_to` is supported only for tuple and list `shape`"
    )
    target_shape = ov_opset.constant(list(shape), Type.i32).output(0)
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.broadcast(x, target_shape).output(0))


def cbrt(x):
    raise NotImplementedError("`cbrt` is not supported with openvino backend")


def ceil(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.ceil(x).output(0))


def clip(x, x_min, x_max):
    x = get_ov_output(x)
    x_min = get_ov_output(x_min, x.get_element_type())
    x_max = get_ov_output(x_max, x.get_element_type())
    clip_by_min = ov_opset.maximum(x, x_min).output(0)
    clip_by_max = ov_opset.minimum(clip_by_min, x_max).output(0)
    return OpenVINOKerasTensor(clip_by_max)


def concatenate(xs, axis=0):
    assert isinstance(xs, list), "`concatenate` is supported only for `x` list"
    elems = []
    for elem in xs:
        elem = get_ov_output(elem)
        elems.append(elem)
    res = ov_opset.concat(elems, axis).output(0)
    return OpenVINOKerasTensor(res)


def conjugate(x):
    raise NotImplementedError(
        "`conjugate` is not supported with openvino backend"
    )


def conj(x):
    raise NotImplementedError("`conj` is not supported with openvino backend")


def copy(x):
    return x


def cos(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, ov_type)
    return OpenVINOKerasTensor(ov_opset.cos(x).output(0))


def cosh(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, ov_type)
    return OpenVINOKerasTensor(ov_opset.cosh(x).output(0))


def count_nonzero(x, axis=None):
    x = get_ov_output(x)
    zero_constant = ov_opset.constant(0, dtype=Type.i32).output(0)
    zero_constant = ov_opset.convert_like(zero_constant, x)
    x = ov_opset.not_equal(x, zero_constant).output(0)
    x = ov_opset.convert(x, Type.i32).output(0)
    if axis is None:
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        x = ov_opset.reshape(x, flatten_shape, False).output(0)
        axis = 0
    if isinstance(axis, tuple):
        axis = list(axis)
    if axis == []:
        return OpenVINOKerasTensor(x)
    axis = ov_opset.constant(axis, Type.i32).output(0)
    return OpenVINOKerasTensor(ov_opset.reduce_sum(x, axis, False).output(0))


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    raise NotImplementedError("`cross` is not supported with openvino backend")


def cumprod(x, axis=None, dtype=None):
    raise NotImplementedError(
        "`cumprod` is not supported with openvino backend"
    )


def cumsum(x, axis=None, dtype=None):
    x = get_ov_output(x)
    if dtype is not None:
        ov_type = OPENVINO_DTYPES[standardize_dtype(dtype)]
        x = ov_opset.convert(x, ov_type).output(0)
    if axis is None:
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        x = ov_opset.reshape(x, flatten_shape, False).output(0)
        axis = 0
    axis = ov_opset.constant(axis, Type.i32).output(0)
    return OpenVINOKerasTensor(ov_opset.cumsum(x, axis).output(0))


def deg2rad(x):
    raise NotImplementedError(
        "`deg2rad` is not supported with openvino backend"
    )


def diag(x, k=0):
    raise NotImplementedError("`diag` is not supported with openvino backend")


def diagonal(x, offset=0, axis1=0, axis2=1):
    raise NotImplementedError(
        "`diagonal` is not supported with openvino backend"
    )


def diff(a, n=1, axis=-1):
    if n == 0:
        return OpenVINOKerasTensor(get_ov_output(a))
    if n < 0:
        raise ValueError("order must be non-negative but got " + repr(n))
    a = get_ov_output(a)
    a_type = a.get_element_type()
    if isinstance(a, np.ndarray):
        rank = a.ndim
    else:
        rank = a.get_partial_shape().rank.get_length()
    if axis < 0:
        axis = axis + rank
    result = a
    for _ in range(n):
        rank = result.get_partial_shape().rank.get_length()
        strides = ov_opset.constant(
            np.array([1] * rank, dtype=np.int64), Type.i64
        ).output(0)

        begin_upper_list = [0] * rank
        begin_upper_list[axis] = 1
        begin_upper = ov_opset.constant(
            np.array(begin_upper_list, dtype=np.int64), Type.i64
        ).output(0)
        end_upper = ov_opset.constant(
            np.array([0] * rank, dtype=np.int64), Type.i64
        ).output(0)
        begin_mask_upper = [1] * rank
        begin_mask_upper[axis] = 0
        end_mask_upper = [1] * rank
        upper = ov_opset.strided_slice(
            data=result,
            begin=begin_upper,
            end=end_upper,
            strides=strides,
            begin_mask=begin_mask_upper,
            end_mask=end_mask_upper,
            new_axis_mask=[],
            shrink_axis_mask=[],
            ellipsis_mask=[],
        ).output(0)

        begin_lower = ov_opset.constant(
            np.array([0] * rank, dtype=np.int64), Type.i64
        ).output(0)
        end_lower_list = [0] * rank
        end_lower_list[axis] = -1
        end_lower = ov_opset.constant(
            np.array(end_lower_list, dtype=np.int64), Type.i64
        ).output(0)
        begin_mask_lower = [1] * rank
        end_mask_lower = [1] * rank
        end_mask_lower[axis] = 0
        lower = ov_opset.strided_slice(
            data=result,
            begin=begin_lower,
            end=end_lower,
            strides=strides,
            begin_mask=begin_mask_lower,
            end_mask=end_mask_lower,
            new_axis_mask=[],
            shrink_axis_mask=[],
            ellipsis_mask=[],
        ).output(0)

        if a_type == Type.boolean:
            result = ov_opset.not_equal(upper, lower).output(0)
        else:
            result = ov_opset.subtract(upper, lower).output(0)
    return OpenVINOKerasTensor(result)


def digitize(x, bins):
    raise NotImplementedError(
        "`digitize` is not supported with openvino backend"
    )


def dot(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "dot()")
    if x1.get_partial_shape().rank == 0 or x2.get_partial_shape().rank == 0:
        return OpenVINOKerasTensor(ov_opset.multiply(x1, x2).output(0))
    return OpenVINOKerasTensor(ov_opset.matmul(x1, x2, False, False).output(0))


def empty(shape, dtype=None):
    dtype = standardize_dtype(dtype) or config.floatx()
    ov_type = OPENVINO_DTYPES[dtype]
    if isinstance(shape, tuple):
        shape = list(shape)
    elif isinstance(shape, int):
        shape = [shape]
    shape_node = ov_opset.constant(shape, Type.i32).output(0)
    const_zero = ov_opset.constant(0, dtype=ov_type).output(0)
    empty_tensor = ov_opset.broadcast(const_zero, shape_node).output(0)
    return OpenVINOKerasTensor(empty_tensor)


def equal(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "equal()")
    return OpenVINOKerasTensor(ov_opset.equal(x1, x2).output(0))


def exp(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, ov_type)
    return OpenVINOKerasTensor(ov_opset.exp(x).output(0))


def expand_dims(x, axis):
    x = get_ov_output(x)
    if isinstance(axis, tuple):
        axis = list(axis)
    axis = ov_opset.constant(axis, Type.i32).output(0)
    return OpenVINOKerasTensor(ov_opset.unsqueeze(x, axis).output(0))


def expm1(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, ov_type)
    exp_x = ov_opset.exp(x).output(0)
    const_one = ov_opset.constant(1, exp_x.get_element_type())
    result = ov_opset.subtract(exp_x, const_one).output(0)
    return OpenVINOKerasTensor(result)


def flip(x, axis=None):
    raise NotImplementedError("`flip` is not supported with openvino backend")


def floor(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.floor(x).output(0))


def full(shape, fill_value, dtype=None):
    dtype = standardize_dtype(dtype) or config.floatx()
    ov_type = OPENVINO_DTYPES[dtype]
    fill_value = get_ov_output(fill_value, ov_type)
    if isinstance(shape, tuple):
        shape = list(shape)
    target_shape = ov_opset.constant(shape, Type.i32)
    return OpenVINOKerasTensor(
        ov_opset.broadcast(fill_value, target_shape).output(0)
    )


def full_like(x, fill_value, dtype=None):
    x = get_ov_output(x)
    shape_x = ov_opset.shape_of(x)
    if dtype is not None:
        ov_type = OPENVINO_DTYPES[standardize_dtype(dtype)]
    else:
        ov_type = x.get_element_type()
    const_value = ov_opset.constant(fill_value, ov_type).output(0)
    res = ov_opset.broadcast(const_value, shape_x).output(0)
    return OpenVINOKerasTensor(res)


def greater(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "greater()")
    return OpenVINOKerasTensor(ov_opset.greater(x1, x2).output(0))


def greater_equal(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "greater_equal()")
    return OpenVINOKerasTensor(ov_opset.greater_equal(x1, x2).output(0))


def hstack(xs):
    if not isinstance(xs, (list, tuple)):
        xs = (xs,)
    elems = [convert_to_tensor(elem) for elem in xs]
    element_type = elems[0].output.get_element_type()
    elems = [get_ov_output(elem, element_type) for elem in elems]
    is_1d = elems and len(elems[0].get_partial_shape().to_shape()) == 1
    axis = 0 if is_1d else 1
    for i in range(1, len(elems)):
        elems[0], elems[i] = _align_operand_types(
            elems[0], elems[i], "hstack()"
        )
    return OpenVINOKerasTensor(ov_opset.concat(elems, axis).output(0))


def identity(n, dtype=None):
    n = get_ov_output(n)
    dtype = Type.f32 if dtype is None else dtype
    if isinstance(dtype, str):
        ov_dtype = OPENVINO_DTYPES[dtype]
    else:
        ov_dtype = dtype
    n32 = ov_opset.convert(n, Type.i32).output(0)
    identity_matrix = ov_opset.eye(
        num_rows=n32, num_columns=n32, diagonal_index=0, output_type=ov_dtype
    )
    return OpenVINOKerasTensor(identity_matrix.output(0))


def imag(x):
    raise NotImplementedError("`imag` is not supported with openvino backend")


def isclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    dtype = OPENVINO_DTYPES[config.floatx()]

    x1 = ov_opset.convert(get_ov_output(x1), dtype)
    x2 = ov_opset.convert(get_ov_output(x2), dtype)
    rtol = ov_opset.convert(get_ov_output(rtol), dtype)
    atol = ov_opset.convert(get_ov_output(atol), dtype)

    abs_diff = ov_opset.abs(x1 - x2)
    abs_x2 = ov_opset.abs(x2)
    total_tolerance = atol + rtol * abs_x2
    is_close = ov_opset.less_equal(abs_diff, total_tolerance)
    if equal_nan:
        both_nan = ov_opset.logical_and(ov_opset.isnan(x1), ov_opset.isnan(x2))
        is_close = ov_opset.logical_or(is_close, both_nan)

    return OpenVINOKerasTensor(is_close.output(0))


def isfinite(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.is_finite(x).output(0))


def isinf(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.is_inf(x).output(0))


def isnan(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.is_nan(x).output(0))


def less(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "less()")
    return OpenVINOKerasTensor(ov_opset.less(x1, x2).output(0))


def less_equal(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "less_equal()")
    return OpenVINOKerasTensor(ov_opset.less_equal(x1, x2).output(0))


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    raise NotImplementedError(
        "`linspace` is not supported with openvino backend"
    )


def log(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        x_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, x_type)
    return OpenVINOKerasTensor(ov_opset.log(x).output(0))


def log10(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        x_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, x_type)
    log_x = ov_opset.log(x).output(0)
    const_10 = ov_opset.constant(10, x_type).output(0)
    log_10 = ov_opset.log(const_10).output(0)
    result = ov_opset.divide(log_x, log_10).output(0)
    return OpenVINOKerasTensor(result)


def log1p(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()

    if x_type.is_integral():
        x_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, x_type)

    one_const = ov_opset.constant(1, x_type).output(0)
    added = ov_opset.add(x, one_const).output(0)
    result = ov_opset.log(added).output(0)
    return OpenVINOKerasTensor(result)


def log2(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        x_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, x_type)
    log_x = ov_opset.log(x).output(0)
    const_2 = ov_opset.constant(2, x_type).output(0)
    log_2 = ov_opset.log(const_2).output(0)
    result = ov_opset.divide(log_x, log_2).output(0)
    return OpenVINOKerasTensor(result)


def logaddexp(x1, x2):
    raise NotImplementedError(
        "`logaddexp` is not supported with openvino backend"
    )


def logical_and(x1, x2):
    x1 = get_ov_output(x1)
    x2 = get_ov_output(x2)
    x1 = ov_opset.convert(x1, Type.boolean).output(0)
    x2 = ov_opset.convert(x2, Type.boolean).output(0)
    return OpenVINOKerasTensor(ov_opset.logical_and(x1, x2).output(0))


def logical_not(x):
    x = get_ov_output(x)
    x = ov_opset.convert(x, Type.boolean).output(0)
    return OpenVINOKerasTensor(ov_opset.logical_not(x).output(0))


def logical_or(x1, x2):
    x1 = get_ov_output(x1)
    x2 = get_ov_output(x2)
    x1 = ov_opset.convert(x1, Type.boolean).output(0)
    x2 = ov_opset.convert(x2, Type.boolean).output(0)
    return OpenVINOKerasTensor(ov_opset.logical_or(x1, x2).output(0))


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    raise NotImplementedError(
        "`logspace` is not supported with openvino backend"
    )


def maximum(x1, x2):
    x1 = get_ov_output(x1)
    x2 = get_ov_output(x2)
    x1, x2 = _align_operand_types(x1, x2, "maximum()")
    return OpenVINOKerasTensor(ov_opset.maximum(x1, x2).output(0))


def median(x, axis=None, keepdims=False):
    raise NotImplementedError("`median` is not supported with openvino backend")


def meshgrid(*x, indexing="xy"):
    raise NotImplementedError(
        "`meshgrid` is not supported with openvino backend"
    )


def min(x, axis=None, keepdims=False, initial=None):
    x = get_ov_output(x)
    original_type = x.get_element_type()
    x_type = original_type
    x_shape = x.get_partial_shape().to_shape()

    is_bool = x_type == Type.boolean
    if is_bool:
        x = ov_opset.convert(x, Type.i32).output(0)
        x_type = Type.i32

    if isinstance(axis, tuple) and len(axis) == 0:
        return OpenVINOKerasTensor(x)

    if axis is None:
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        x = ov_opset.reshape(x, flatten_shape, False).output(0)
        axis = 0

    if isinstance(axis, tuple):
        axis = list(axis)

    axis_const = ov_opset.constant(axis, Type.i32).output(0)
    min_result = ov_opset.reduce_min(x, axis_const, keepdims).output(0)

    if initial is not None:
        initial_tensor = ov_opset.constant(initial, x_type).output(0)
        min_result = ov_opset.minimum(min_result, initial_tensor).output(0)

    if keepdims:
        result_shape = [1] * len(x_shape)
        min_result = ov_opset.reshape(
            min_result,
            ov_opset.constant(result_shape, Type.i32).output(0),
            False,
        ).output(0)

    if is_bool:
        min_result = ov_opset.convert(min_result, Type.boolean).output(0)

    return OpenVINOKerasTensor(min_result)


def minimum(x1, x2):
    x1 = get_ov_output(x1)
    x2 = get_ov_output(x2)
    x1, x2 = _align_operand_types(x1, x2, "minimum()")
    return OpenVINOKerasTensor(ov_opset.minimum(x1, x2).output(0))


def mod(x1, x2):
    x1 = get_ov_output(x1)
    x2 = get_ov_output(x2)
    x1, x2 = _align_operand_types(x1, x2, "mod()")
    return OpenVINOKerasTensor(ov_opset.floor_mod(x1, x2).output(0))


def moveaxis(x, source, destination):
    x = get_ov_output(x)
    if isinstance(source, int):
        source = [source]
    if isinstance(destination, int):
        destination = [destination]

    ndim = x.get_partial_shape().rank.get_length()
    source = [axis if axis >= 0 else axis + ndim for axis in source]
    destination = [axis if axis >= 0 else axis + ndim for axis in destination]

    axes = list(range(ndim))
    for src, dst in zip(source, destination):
        axes.remove(src)
        axes.insert(dst, src)

    axes_const = ov_opset.constant(axes, Type.i32).output(0)
    return OpenVINOKerasTensor(ov_opset.transpose(x, axes_const).output(0))


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    x = get_ov_output(x)
    dtype = x.get_element_type()
    if dtype.is_integral():
        return OpenVINOKerasTensor(x)
    isfloat64 = True if dtype == Type.f64 else False
    if isfloat64:  # conversion to f32 due to https://github.com/openvinotoolkit/openvino/issues/30264
        x = ov_opset.convert(x, Type.f32).output(0)
        dtype = Type.f32
    nan_val = ov_opset.constant(nan, dtype).output(0)
    posinf_val = ov_opset.constant(
        posinf if posinf is not None else DTYPES_MAX[dtype], dtype
    ).output(0)
    neginf_val = ov_opset.constant(
        neginf if neginf is not None else DTYPES_MIN[dtype], dtype
    ).output(0)
    posinf_mask = ov_opset.is_inf(
        x,
        {"detect_positive": True, "detect_negative": False},
    ).output(0)
    neginf_mask = ov_opset.is_inf(
        x,
        {"detect_positive": False, "detect_negative": True},
    ).output(0)
    nan_mask = ov_opset.is_nan(x).output(0)
    x = ov_opset.select(nan_mask, nan_val, x).output(0)
    x = ov_opset.select(posinf_mask, posinf_val, x).output(0)
    x = ov_opset.select(neginf_mask, neginf_val, x).output(0)
    if isfloat64:
        x = ov_opset.convert(x, Type.f64).output(0)
    return OpenVINOKerasTensor(x)


def ndim(x):
    x = get_ov_output(x)
    shape_tensor = ov_opset.shape_of(x, Type.i64).output(0)
    rank_tensor = ov_opset.shape_of(shape_tensor, Type.i64).output(0)
    return OpenVINOKerasTensor(rank_tensor)


def nonzero(x):
    x = get_ov_output(x)
    res = ov_opset.non_zero(data=x, output_type="i32").output(0)
    return OpenVINOKerasTensor(res)


def not_equal(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "not_equal()")
    return OpenVINOKerasTensor(ov_opset.not_equal(x1, x2).output(0))


def zeros_like(x, dtype=None):
    x = get_ov_output(x)
    shape_x = ov_opset.shape_of(x)
    if dtype is not None:
        ov_type = OPENVINO_DTYPES[standardize_dtype(dtype)]
        const_zero = ov_opset.constant(0, ov_type).output(0)
    else:
        const_zero = ov_opset.constant(0, x.get_element_type()).output(0)
    res = ov_opset.broadcast(const_zero, shape_x).output(0)
    return OpenVINOKerasTensor(res)


def ones_like(x, dtype=None):
    x = get_ov_output(x)
    shape_x = ov_opset.shape_of(x)
    if dtype is not None:
        ov_type = OPENVINO_DTYPES[standardize_dtype(dtype)]
        const_one = ov_opset.constant(1, ov_type).output(0)
    else:
        const_one = ov_opset.constant(1, x.get_element_type()).output(0)
    res = ov_opset.broadcast(const_one, shape_x).output(0)
    return OpenVINOKerasTensor(res)


def outer(x1, x2):
    x1 = get_ov_output(x1)
    x2 = get_ov_output(x2)

    x1, x2 = _align_operand_types(x1, x2, "outer()")

    new_shape_x1 = ov_opset.constant([-1, 1], Type.i32).output(0)
    new_shape_x2 = ov_opset.constant([1, -1], Type.i32).output(0)

    # Reshape directly from original tensors
    x1_reshaped = ov_opset.reshape(x1, new_shape_x1, False).output(0)
    x2_reshaped = ov_opset.reshape(x2, new_shape_x2, False).output(0)

    result = ov_opset.multiply(x1_reshaped, x2_reshaped).output(0)

    return OpenVINOKerasTensor(result)


def pad(x, pad_width, mode="constant", constant_values=None):
    x = get_ov_output(x)
    pad_value = None
    if constant_values is not None:
        if mode != "constant":
            raise ValueError(
                "Argument `constant_values` can only be "
                "provided when `mode == 'constant'`. "
                f"Received: mode={mode}"
            )
        assert isinstance(constant_values, int), (
            "`pad` operation supports only scalar pad value "
            "in constant mode by openvino backend"
        )
        pad_value = constant_values

    # split pad_width into two tensors pads_begin and pads_end
    pads_begin = []
    pads_end = []
    for pads_pair in pad_width:
        pads_begin.append(pads_pair[0])
        pads_end.append(pads_pair[1])
    pads_begin = ov_opset.constant(pads_begin, Type.i32).output(0)
    pads_end = ov_opset.constant(pads_end, Type.i32).output(0)
    return OpenVINOKerasTensor(
        ov_opset.pad(x, pads_begin, pads_end, mode, pad_value).output(0)
    )


def prod(x, axis=None, keepdims=False, dtype=None):
    raise NotImplementedError("`prod` is not supported with openvino backend")


def quantile(x, q, axis=None, method="linear", keepdims=False):
    raise NotImplementedError(
        "`quantile` is not supported with openvino backend"
    )


def ravel(x):
    x = get_ov_output(x)
    target_shape = ov_opset.constant([-1], dtype=Type.i32).output(0)
    return OpenVINOKerasTensor(
        ov_opset.reshape(x, target_shape, special_zero=False).output(0)
    )


def real(x):
    raise NotImplementedError("`real` is not supported with openvino backend")


def reciprocal(x):
    x = get_ov_output(x)
    one_constant = ov_opset.constant(1, dtype=x.get_element_type()).output(0)
    x = ov_opset.divide(one_constant, x).output(0)
    return OpenVINOKerasTensor(x)


def repeat(x, repeats, axis=None):
    raise NotImplementedError("`repeat` is not supported with openvino backend")


def reshape(x, newshape):
    x = get_ov_output(x)
    if isinstance(newshape, tuple):
        newshape = list(newshape)
    newshape = ov_opset.constant(newshape, Type.i32).output(0)
    return OpenVINOKerasTensor(ov_opset.reshape(x, newshape, False).output(0))


def roll(x, shift, axis=None):
    raise NotImplementedError("`roll` is not supported with openvino backend")


def sign(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.sign(x).output(0))


def signbit(x):
    raise NotImplementedError(
        "`signbit` is not supported with openvino backend"
    )


def sin(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, ov_type)
    return OpenVINOKerasTensor(ov_opset.sin(x).output(0))


def sinh(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, ov_type)
    return OpenVINOKerasTensor(ov_opset.sinh(x).output(0))


def size(x):
    raise NotImplementedError("`size` is not supported with openvino backend")


def sort(x, axis=-1):
    raise NotImplementedError("`sort` is not supported with openvino backend")


def split(x, indices_or_sections, axis=0):
    x = get_ov_output(x)
    axis_tensor = ov_opset.constant(axis, dtype=Type.i32).output(0)

    shape_tensor = ov_opset.shape_of(x)
    axis_i32 = ov_opset.constant([axis], dtype=Type.i32)
    dim_at_axis_tensor = ov_opset.gather(
        shape_tensor, axis_i32, ov_opset.constant(0, dtype=Type.i32)
    )

    if isinstance(indices_or_sections, int):
        num_splits = indices_or_sections
        splits = ov_opset.split(x, axis_tensor, num_splits=num_splits)
        result = []
        for i in range(num_splits):
            result.append(OpenVINOKerasTensor(splits.output(i)))
        return result

    if isinstance(indices_or_sections, (list, tuple, np.ndarray)):
        indices = list(indices_or_sections)
        split_lengths = []
        split_lengths.append(indices[0])
        for i in range(1, len(indices)):
            split_lengths.append(indices[i] - indices[i - 1])

        last_index_tensor = ov_opset.constant(indices[-1], dtype=Type.i64)
        remaining_length_tensor = ov_opset.subtract(
            dim_at_axis_tensor, last_index_tensor
        )

        length_parts = []
        length_parts.append(ov_opset.constant(split_lengths, dtype=Type.i64))
        length_parts.append(remaining_length_tensor)
        length_tensor = ov_opset.concat(length_parts, axis=0)

        splits = ov_opset.variadic_split(x, axis_tensor, length_tensor)
        result = []
        for i in range(len(split_lengths) + 1):
            result.append(OpenVINOKerasTensor(splits.output(i)))
        return result

    raise TypeError(
        f"unsupported type of indices_or_sections: {type(indices_or_sections)}"
    )


def stack(x, axis=0):
    if isinstance(x, tuple):
        x = list(x)
    assert isinstance(x, list), "`stack` supports only `x` as list or tuple"
    elems = [get_ov_output(e) for e in x]
    ref = elems[0]
    for i in range(1, len(elems)):
        ref, elems[i] = _align_operand_types(ref, elems[i], "stack()")
    elems[0] = ref
    const_axis = ov_opset.constant(axis, Type.i32).output(0)
    elems = [ov_opset.unsqueeze(e, const_axis).output(0) for e in elems]
    res = ov_opset.concat(elems, axis).output(0)
    return OpenVINOKerasTensor(res)


def std(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    if axis is None:
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        x = ov_opset.reshape(x, flatten_shape, False).output(0)
        axis = 0
    axis = ov_opset.constant(axis, Type.i32).output(0)
    # The variance is computed using $Var = E[|x|^2] - |E[x]|^2$, It is faster
    # but less numerically stable.
    mean = ov_opset.reduce_mean(x, axis, keepdims).output(0)
    const_two = ov_opset.constant(2, x.get_element_type()).output(0)
    squared_x = ov_opset.power(x, const_two).output(0)
    squared_mean = ov_opset.power(mean, const_two).output(0)
    squared_x_mean = ov_opset.reduce_mean(squared_x, axis, keepdims)
    variance = ov_opset.subtract(squared_x_mean, squared_mean).output(0)
    std_var = OpenVINOKerasTensor(ov_opset.sqrt(variance).output(0))
    return std_var


def swapaxes(x, axis1, axis2):
    raise NotImplementedError(
        "`swapaxes` is not supported with openvino backend"
    )


def take(x, indices, axis=None):
    x = get_ov_output(x)
    indices = get_ov_output(indices)
    if axis is None:
        target_shape = ov_opset.constant([-1], dtype=Type.i32).output(0)
        x = ov_opset.reshape(x, target_shape, False).output(0)
        axis = ov_opset.constant(0, dtype=Type.i32).output(0)
    else:
        axis = ov_opset.constant(axis, dtype=Type.i32).output(0)
    return OpenVINOKerasTensor(ov_opset.gather(x, indices, axis).output(0))


def take_along_axis(x, indices, axis=None):
    x = get_ov_output(x)
    indices = get_ov_output(indices)

    if axis is None:
        target_shape = ov_opset.constant([-1], dtype=Type.i32).output(0)
        x_flat = ov_opset.reshape(x, target_shape, False).output(0)
        indices_flat = ov_opset.reshape(indices, target_shape, False).output(0)
        result = ov_opset.gather_elements(x_flat, indices_flat, 0).output(0)
        return OpenVINOKerasTensor(result)

    x_rank = len(x.get_partial_shape())
    if axis < 0:
        axis += x_rank

    x_shape = ov_opset.shape_of(x, Type.i32).output(0)
    indices_shape = ov_opset.shape_of(indices, Type.i32).output(0)

    zero_const = ov_opset.constant(0, dtype=Type.i32).output(0)
    axis_index = ov_opset.constant([axis], dtype=Type.i32).output(0)

    # Fix negative indices
    dim_size = ov_opset.squeeze(
        ov_opset.gather(x_shape, axis_index, zero_const).output(0), zero_const
    ).output(0)
    zero_scalar = ov_opset.constant(0, indices.get_element_type()).output(0)
    is_neg = ov_opset.less(indices, zero_scalar).output(0)
    dim_size_cast = ov_opset.convert(
        dim_size, indices.get_element_type()
    ).output(0)
    indices = ov_opset.select(
        is_neg, ov_opset.add(indices, dim_size_cast).output(0), indices
    ).output(0)
    indices = ov_opset.convert(indices, Type.i32).output(0)

    x_target_parts, indices_target_parts = [], []

    for i in range(x_rank):
        dim_idx = ov_opset.constant([i], dtype=Type.i32).output(0)
        x_dim = ov_opset.gather(x_shape, dim_idx, zero_const).output(0)
        indices_dim = ov_opset.gather(
            indices_shape, dim_idx, zero_const
        ).output(0)

        if i == axis:
            # For axis dimension: keep original dimensions
            x_target_parts.append(x_dim)
            indices_target_parts.append(indices_dim)
        else:
            # For other dimensions: use maximum for broadcasting
            max_dim = ov_opset.maximum(x_dim, indices_dim).output(0)
            x_target_parts.append(max_dim)
            indices_target_parts.append(max_dim)

    x_target_shape = ov_opset.concat(x_target_parts, axis=0).output(0)
    indices_target_shape = ov_opset.concat(indices_target_parts, axis=0).output(
        0
    )

    # Broadcast to target shapes and gather elements
    x_broadcasted = ov_opset.broadcast(x, x_target_shape).output(0)
    indices_broadcasted = ov_opset.broadcast(
        indices, indices_target_shape
    ).output(0)
    result = ov_opset.gather_elements(
        x_broadcasted, indices_broadcasted, axis
    ).output(0)

    return OpenVINOKerasTensor(result)


def tan(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, ov_type)
    return OpenVINOKerasTensor(ov_opset.tan(x).output(0))


def tanh(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, ov_type)
    return OpenVINOKerasTensor(ov_opset.tanh(x).output(0))


def tensordot(x1, x2, axes=2):
    raise NotImplementedError(
        "`tensordot` is not supported with openvino backend"
    )


def round(x, decimals=0):
    raise NotImplementedError("`round` is not supported with openvino backend")


def tile(x, repeats):
    raise NotImplementedError("`tile` is not supported with openvino backend")


def trace(x, offset=0, axis1=0, axis2=1):
    raise NotImplementedError("`trace` is not supported with openvino backend")


def tri(N, M=None, k=0, dtype=None):
    if M is None:
        M = N
    if dtype is None:
        dtype = "float32"

    ov_dtype = OPENVINO_DTYPES[dtype]

    def ensure_constant(value, default_type=Type.i32):
        if isinstance(value, (int, float)):
            return ov_opset.constant(value, default_type)
        elif hasattr(value, "get_element_type"):
            if value.get_element_type() != Type.i32:
                value = ov_opset.convert(value, Type.i32)
            return ov_opset.squeeze(value, ov_opset.constant([0], Type.i32))
        else:
            return ov_opset.constant(value, default_type)

    N_const = ensure_constant(N)
    M_const = ensure_constant(M)
    k_const = ensure_constant(k)

    # Create row and column indices
    row_range = ov_opset.range(
        ov_opset.constant(0, Type.i32),
        N_const,
        ov_opset.constant(1, Type.i32),
        output_type=Type.i32,
    )
    col_range = ov_opset.range(
        ov_opset.constant(0, Type.i32),
        M_const,
        ov_opset.constant(1, Type.i32),
        output_type=Type.i32,
    )

    # Reshape indices for broadcasting
    row_idx = ov_opset.unsqueeze(row_range, ov_opset.constant([1], Type.i32))
    col_idx = ov_opset.unsqueeze(col_range, ov_opset.constant([0], Type.i32))

    mask = ov_opset.less_equal(col_idx, ov_opset.add(row_idx, k_const))

    if ov_dtype == Type.boolean:
        result = mask
    else:
        result = ov_opset.convert(mask, ov_dtype)

    return OpenVINOKerasTensor(result.output(0))


def tril(x, k=0):
    x = get_ov_output(x)
    ov_type = x.get_element_type()
    shape = ov_opset.shape_of(x, Type.i32)
    zero_const = ov_opset.constant(0, Type.i32)
    minus2 = ov_opset.constant([-2], Type.i32)
    minus1 = ov_opset.constant([-1], Type.i32)
    M = ov_opset.squeeze(ov_opset.gather(shape, minus2, zero_const), zero_const)
    N = ov_opset.squeeze(ov_opset.gather(shape, minus1, zero_const), zero_const)
    tri_mask = tri(M, N, k=k, dtype="bool").output
    mask = ov_opset.convert(tri_mask, ov_type)
    if ov_type == Type.boolean:
        out = ov_opset.logical_and(x, mask)
    else:
        out = ov_opset.multiply(x, mask)
    return OpenVINOKerasTensor(out.output(0))


def triu(x, k=0):
    x = get_ov_output(x)
    ov_type = x.get_element_type()
    shape = ov_opset.shape_of(x, Type.i32)
    zero_const = ov_opset.constant(0, Type.i32)
    minus2 = ov_opset.constant([-2], Type.i32)
    minus1 = ov_opset.constant([-1], Type.i32)
    M = ov_opset.squeeze(ov_opset.gather(shape, minus2, zero_const), zero_const)
    N = ov_opset.squeeze(ov_opset.gather(shape, minus1, zero_const), zero_const)
    tri_mask = tri(M, N, k=k - 1, dtype="bool").output
    if ov_type == Type.boolean:
        mask = ov_opset.logical_not(tri_mask)
    else:
        const_one = ov_opset.constant(1, ov_type)
        converted_mask = ov_opset.convert(tri_mask, ov_type)
        mask = ov_opset.subtract(const_one, converted_mask)
    if ov_type == Type.boolean:
        out = ov_opset.logical_and(x, mask)
    else:
        out = ov_opset.multiply(x, mask)
    return OpenVINOKerasTensor(out.output(0))


def vdot(x1, x2):
    raise NotImplementedError("`vdot` is not supported with openvino backend")


def vstack(xs):
    raise NotImplementedError("`vstack` is not supported with openvino backend")


def vectorize(pyfunc, *, excluded=None, signature=None):
    raise NotImplementedError(
        "`vectorize` is not supported with openvino backend"
    )


def where(condition, x1=None, x2=None):
    condition = get_ov_output(condition)
    if x1 is None and x2 is None:
        nonzero_indices = ov_opset.non_zero(condition)
        return OpenVINOKerasTensor(nonzero_indices.output(0))
    if x1 is None:
        return OpenVINOKerasTensor(condition)
    if x2 is None:
        raise ValueError("x2 must be provided if x1 is specified.")

    def cast_literal_like_tensor(literal, x):
        ov_type = get_ov_output(x).get_element_type()
        is_bool = ov_type == Type.boolean
        is_float_to_int = isinstance(literal, float) and ov_type.is_integral()
        if is_bool or is_float_to_int:
            return get_ov_output(literal), get_ov_output(x)
        return get_ov_output(literal, ov_type), get_ov_output(x)

    if isinstance(x1, (int, float)):
        x1, x2 = cast_literal_like_tensor(x1, x2)
    elif isinstance(x2, (int, float)):
        x2, x1 = cast_literal_like_tensor(x2, x1)
    else:
        x1 = get_ov_output(x1)
        x2 = get_ov_output(x2)
    x1, x2 = _align_operand_types(x1, x2, "select()")
    return OpenVINOKerasTensor(ov_opset.select(condition, x1, x2).output(0))


def divide(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1_type = ov_to_keras_type(x1.get_element_type())
    x2_type = ov_to_keras_type(x2.get_element_type())
    result_type = dtypes.result_type(x1_type, x2_type, float)
    result_type = OPENVINO_DTYPES[result_type]
    x1 = ov_opset.convert(x1, result_type).output(0)
    x2 = ov_opset.convert(x2, result_type).output(0)
    return OpenVINOKerasTensor(ov_opset.divide(x1, x2).output(0))


def divide_no_nan(x1, x2):
    raise NotImplementedError(
        "`divide_no_nan` is not supported with openvino backend"
    )


def true_divide(x1, x2):
    return divide(x1, x2)


def power(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "power()")
    return OpenVINOKerasTensor(ov_opset.power(x1, x2).output(0))


def negative(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.negative(x).output(0))


def square(x):
    x = get_ov_output(x)
    const_two = ov_opset.constant(2, x.get_element_type()).output(0)
    return OpenVINOKerasTensor(ov_opset.power(x, const_two).output(0))


def sqrt(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.sqrt(x).output(0))


def squeeze(x, axis=None):
    x = get_ov_output(x)
    if axis is None:
        axis = []
        for idx, dim in enumerate(x.get_partial_shape()):
            if dim == 1:
                axis.append(idx)
    axis = ov_opset.constant(axis, Type.i32).output(0)
    return OpenVINOKerasTensor(ov_opset.squeeze(x, axis).output(0))


def transpose(x, axes=None):
    x = get_ov_output(x)
    if axes is None:
        # generate reverse permutation vector
        shape_x = ov_opset.shape_of(x, "i64").output(0)
        rank_x = ov_opset.shape_of(shape_x, "i64").output(0)
        scalar_shape = ov_opset.constant([], Type.i32).output(0)
        rank_x = ov_opset.reshape(rank_x, scalar_shape, False).output(0)
        const_minus_one = ov_opset.constant(-1, Type.i64).output(0)
        rank_minus_one = ov_opset.add(rank_x, const_minus_one).output(0)
        axes = ov_opset.range(
            rank_minus_one, const_minus_one, const_minus_one, "i64"
        ).output(0)
    else:
        if isinstance(axes, tuple):
            axes = list(axes)
        axes = ov_opset.constant(axes, Type.i32).output(0)
    return OpenVINOKerasTensor(ov_opset.transpose(x, axes).output(0))


def var(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    if axis is None:
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        x = ov_opset.reshape(x, flatten_shape, False).output(0)
        axis = 0
    axis = ov_opset.constant(axis, Type.i32).output(0)
    # The variance is computed using $Var = E[|x|^2] - |E[x]|^2$, It is faster
    # but less numerically stable.
    mean = ov_opset.reduce_mean(x, axis, keepdims).output(0)
    const_two = ov_opset.constant(2, x.get_element_type()).output(0)
    squared_x = ov_opset.power(x, const_two).output(0)
    squared_mean = ov_opset.power(mean, const_two).output(0)
    squared_x_mean = ov_opset.reduce_mean(squared_x, axis, keepdims)
    variance = OpenVINOKerasTensor(
        ov_opset.subtract(squared_x_mean, squared_mean).output(0)
    )
    return variance


def sum(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    if axis is None:
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        x = ov_opset.reshape(x, flatten_shape, False).output(0)
        axis = 0
    axis = ov_opset.constant(axis, Type.i32).output(0)
    return OpenVINOKerasTensor(ov_opset.reduce_sum(x, axis, keepdims).output(0))


def eye(N, M=None, k=0, dtype=None):
    raise NotImplementedError("`eye` is not supported with openvino backend")


def floor_divide(x1, x2):
    raise NotImplementedError(
        "`floor_divide` is not supported with openvino backend"
    )


def logical_xor(x1, x2):
    x1 = get_ov_output(x1)
    x2 = get_ov_output(x2)
    x1 = ov_opset.convert(x1, Type.boolean).output(0)
    x2 = ov_opset.convert(x2, Type.boolean).output(0)
    return OpenVINOKerasTensor(ov_opset.logical_xor(x1, x2).output(0))


def corrcoef(x):
    raise NotImplementedError(
        "`corrcoef` is not supported with openvino backend"
    )


def correlate(x1, x2, mode="valid"):
    raise NotImplementedError(
        "`correlate` is not supported with openvino backend"
    )


def select(condlist, choicelist, default=0):
    raise NotImplementedError("`select` is not supported with openvino backend")


def slogdet(x):
    raise NotImplementedError(
        "`slogdet` is not supported with openvino backend"
    )


def argpartition(x, kth, axis=-1):
    raise NotImplementedError(
        "`argpartition` is not supported with openvino backend"
    )
