import numpy as np
import openvino.opset14 as ov_opset
from openvino import Type

from keras.src.backend import config
from keras.src.backend.common import dtypes
from keras.src.backend.common.backend_utils import canonicalize_axis
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
    if x1.get_element_type() == Type.boolean:
        return OpenVINOKerasTensor(ov_opset.logical_xor(x1, x2).output(0))
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
    x_ov = get_ov_output(x)
    x_type = x_ov.get_element_type()

    was_axis_none = axis is None
    x_resolved, axis_resolved = _resolve_axis(x_ov, axis)

    if axis_resolved is None:
        return OpenVINOKerasTensor(x_ov)

    if x_type.is_integral():
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x_resolved = ov_opset.convert(x_resolved, ov_type).output(0)

    result = ov_opset.reduce_mean(x_resolved, axis_resolved, keepdims).output(0)

    if keepdims and was_axis_none:
        rank = x.get_partial_shape().rank.get_length()
        result_shape = [1] * rank
        result = ov_opset.reshape(
            result,
            ov_opset.constant(result_shape, Type.i32).output(0),
            False,
        ).output(0)

    return OpenVINOKerasTensor(result)


def max(x, axis=None, keepdims=False, initial=None):
    return _compute_extrema(x, "max", axis, keepdims, initial)


def _compute_extrema(x, operation, axis=None, keepdims=False, initial=None):
    if operation == "min":
        reduction_op = ov_opset.reduce_min
        elementwise_op = ov_opset.minimum
    elif operation == "max":
        reduction_op = ov_opset.reduce_max
        elementwise_op = ov_opset.maximum
    else:
        raise ValueError(
            f"Operation must be 'min' or 'max', received {operation}"
        )

    x = get_ov_output(x)
    x_type = x.get_element_type()
    x_for_rank = x

    is_bool = x_type == Type.boolean
    if is_bool:
        x = ov_opset.convert(x, Type.i32).output(0)
        x_type = Type.i32

    if isinstance(axis, tuple) and len(axis) == 0:
        return OpenVINOKerasTensor(x)

    was_axis_none = axis is None
    x, axis = _resolve_axis(x, axis)

    result = reduction_op(x, axis, keepdims).output(0)

    if initial is not None:
        initial_tensor = ov_opset.constant(initial, x_type).output(0)
        result = elementwise_op(result, initial_tensor).output(0)

    if keepdims and was_axis_none:
        orig_shape = ov_opset.shape_of(x_for_rank, Type.i32).output(0)
        orig_rank_shape = ov_opset.shape_of(orig_shape, Type.i32).output(0)
        one = ov_opset.constant(1, Type.i32).output(0)
        result_shape = ov_opset.broadcast(one, orig_rank_shape).output(0)
        result = ov_opset.reshape(result, result_shape, False).output(0)

    if is_bool:
        result = ov_opset.convert(result, Type.boolean).output(0)

    return OpenVINOKerasTensor(result)


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
    x_type = x.get_element_type()
    if x_type == Type.boolean:
        return OpenVINOKerasTensor(x)
    return OpenVINOKerasTensor(ov_opset.absolute(x).output(0))


def abs(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.absolute(x).output(0))


def all(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    x, axis = _resolve_axis(x, axis)
    if axis is None:
        return OpenVINOKerasTensor(x)
    x = ov_opset.convert(x, Type.boolean).output(0)
    return OpenVINOKerasTensor(
        ov_opset.reduce_logical_and(x, axis, keepdims).output(0)
    )


def angle(x):
    raise NotImplementedError("`angle` is not supported with openvino backend")


def any(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    x, axis = _resolve_axis(x, axis)
    if axis is None:
        return OpenVINOKerasTensor(x)
    x = ov_opset.convert(x, Type.boolean).output(0)
    return OpenVINOKerasTensor(
        ov_opset.reduce_logical_or(x, axis, keepdims).output(0)
    )


def amax(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    x, axis = _resolve_axis(x, axis)
    if axis is None:
        return OpenVINOKerasTensor(x)
    if x_type == Type.boolean:
        return OpenVINOKerasTensor(
            ov_opset.reduce_logical_or(x, axis, keepdims).output(0)
        )
    return OpenVINOKerasTensor(ov_opset.reduce_max(x, axis, keepdims).output(0))


def amin(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    x, axis = _resolve_axis(x, axis)
    if axis is None:
        return OpenVINOKerasTensor(x)
    if x_type == Type.boolean:
        return OpenVINOKerasTensor(
            ov_opset.reduce_logical_and(x, axis, keepdims).output(0)
        )
    return OpenVINOKerasTensor(ov_opset.reduce_min(x, axis, keepdims).output(0))


def _resolve_axis(x, axis):
    if axis == () or axis == []:
        return x, None
    if axis is None:
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        x = ov_opset.reshape(x, flatten_shape, False).output(0)
        axis = 0
    if isinstance(axis, tuple):
        axis = list(axis)
    axis = ov_opset.constant(axis, Type.i32).output(0)
    return x, axis


def _upcast_type_if_needed(x):
    x_type = x.get_element_type()
    if x_type == Type.boolean:
        x = ov_opset.convert(x, Type.i32).output(0)
    elif x_type in (Type.i8, Type.i16):
        x = ov_opset.convert(x, Type.i32).output(0)
    elif x_type in (Type.u8, Type.u16):
        x = ov_opset.convert(x, Type.u32).output(0)
    return x


def append(x1, x2, axis=None):
    x1, x2 = get_ov_output(x1), get_ov_output(x2)
    x1, x2 = _align_operand_types(x1, x2, "append()")
    if axis is None:
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        x1 = ov_opset.reshape(x1, flatten_shape, False).output(0)
        x2 = ov_opset.reshape(x2, flatten_shape, False).output(0)
        axis = 0
    return OpenVINOKerasTensor(ov_opset.concat([x1, x2], axis).output(0))


def arange(start, stop=None, step=None, dtype=None):
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


def view(x, dtype=None):
    raise NotImplementedError("`view` is not supported with openvino backend")


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
    x = get_ov_output(x)
    zero_const = ov_opset.constant(0, Type.i64)
    one_const = ov_opset.constant(1, Type.i64)
    two_const = ov_opset.constant(2, Type.i64)
    two_const_f64 = ov_opset.constant(2.0, Type.f64)
    if x.get_element_type() != Type.i64:
        x = ov_opset.convert(x, Type.i64)
    half = ov_opset.convert(
        ov_opset.divide(ov_opset.subtract(x, one_const), two_const), Type.f64
    )
    n = ov_opset.range(zero_const, x, one_const, Type.f64)
    condition = ov_opset.less_equal(n, half)
    first_half = ov_opset.divide(
        ov_opset.multiply(two_const_f64, n),
        ov_opset.convert(ov_opset.subtract(x, one_const), Type.f64),
    )
    second_half = ov_opset.subtract(two_const_f64, first_half)
    window = ov_opset.select(condition, first_half, second_half)
    window = ov_opset.convert(window, OPENVINO_DTYPES[config.floatx()]).output(
        0
    )
    return OpenVINOKerasTensor(window)


def hamming(x):
    m = get_ov_output(x)

    m_i64 = (
        m if m.get_element_type() == Type.i64 else ov_opset.convert(m, Type.i64)
    )

    start = ov_opset.constant(0, Type.i64)
    step = ov_opset.constant(1, Type.i64)
    n = ov_opset.range(start, m_i64, step, Type.f64)

    one_i64 = ov_opset.constant(1, Type.i64)
    denom_i64 = ov_opset.subtract(m_i64, one_i64)
    denom = ov_opset.convert(denom_i64, Type.f64)

    two_pi = ov_opset.constant(2.0 * np.pi, Type.f64)
    two_pi_over_m_minus_1 = ov_opset.divide(two_pi, denom)

    x = ov_opset.multiply(two_pi_over_m_minus_1, n)
    c = ov_opset.cos(x)

    # 0.54 - 0.46 * cos(...)
    a = ov_opset.constant(0.54, Type.f64)
    b = ov_opset.constant(0.46, Type.f64)
    hamming_window = ov_opset.subtract(a, ov_opset.multiply(b, c))
    hamming_window = ov_opset.convert(
        hamming_window, OPENVINO_DTYPES[config.floatx()]
    )

    return OpenVINOKerasTensor(hamming_window.output(0))


def heaviside(x1, x2):
    x1 = get_ov_output(x1)
    x_type = x1.get_element_type()
    x2 = get_ov_output(x2, x_type)

    zero_scalar = ov_opset.constant(0, x_type).output(0)
    one_scalar = ov_opset.constant(1, x_type).output(0)

    neg = ov_opset.less(x1, zero_scalar).output(0)
    pos = ov_opset.greater(x1, zero_scalar).output(0)
    eq = ov_opset.equal(x1, zero_scalar).output(0)

    x = ov_opset.select(neg, zero_scalar, x1).output(0)
    x = ov_opset.select(pos, one_scalar, x).output(0)
    x = ov_opset.select(eq, x2, x).output(0)
    return OpenVINOKerasTensor(x)


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
    x = get_ov_output(x)
    zero_const = ov_opset.constant(0, Type.i64)
    one_const = ov_opset.constant(1, Type.i64)
    two_pi = ov_opset.constant(2.0 * np.pi, Type.f64)
    term_1 = ov_opset.constant(0.42, Type.f64)
    term_2 = ov_opset.constant(0.5, Type.f64)
    term_3 = ov_opset.constant(0.08, Type.f64)
    if x.get_element_type() != Type.i64:
        x = ov_opset.convert(x, Type.i64)
    n = ov_opset.range(zero_const, x, one_const, Type.f64)
    n_minus_1 = ov_opset.subtract(
        ov_opset.convert(x, Type.f64), ov_opset.constant(1.0, Type.f64)
    ).output(0)
    angle_2pi = ov_opset.divide(ov_opset.multiply(two_pi, n), n_minus_1)
    angle_4pi = ov_opset.multiply(angle_2pi, ov_opset.constant(2.0, Type.f64))
    cos_2pi = ov_opset.cos(angle_2pi)
    cos_4pi = ov_opset.cos(angle_4pi)
    term_2_final = ov_opset.multiply(term_2, cos_2pi)
    term_3_final = ov_opset.multiply(term_3, cos_4pi)
    window = ov_opset.add(ov_opset.subtract(term_1, term_2_final), term_3_final)
    window = ov_opset.convert(window, OPENVINO_DTYPES[config.floatx()]).output(
        0
    )
    return OpenVINOKerasTensor(window)


def broadcast_to(x, shape):
    assert isinstance(shape, (tuple, list)), (
        "`broadcast_to` is supported only for tuple and list `shape`"
    )
    target_shape = ov_opset.constant(list(shape), Type.i32).output(0)
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.broadcast(x, target_shape).output(0))


def cbrt(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral() or x_type == Type.boolean:
        x = ov_opset.convert(x, OPENVINO_DTYPES[config.floatx()]).output(0)
    sign_x = ov_opset.sign(x)
    abs_x = ov_opset.absolute(x)
    one_third = ov_opset.constant(1.0 / 3.0, x.get_element_type())
    root_abs = ov_opset.power(abs_x, one_third)
    res = ov_opset.multiply(sign_x, root_abs)
    return OpenVINOKerasTensor(res.output(0))


def ceil(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        x = ov_opset.convert(x, OPENVINO_DTYPES[config.floatx()]).output(0)
    ceiling = ov_opset.ceil(x).output(0)
    return OpenVINOKerasTensor(ceiling)


def clip(x, x_min, x_max):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type == Type.boolean:
        x = ov_opset.convert(x, Type.i32).output(0)
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
    x, axis = _resolve_axis(x, axis)
    if not axis:
        return OpenVINOKerasTensor(x)
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
    x, axis = _resolve_axis(x, axis)
    if x.get_element_type() == Type.boolean:
        x = ov_opset.convert(x, Type.i32).output(0)
    return OpenVINOKerasTensor(ov_opset.cumsum(x, axis).output(0))


def deg2rad(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    pi_over_180 = np.pi / 180.0

    if x_type == Type.i64:
        output_type = Type.f64
    elif x_type.is_integral():
        output_type = OPENVINO_DTYPES[config.floatx()]
    else:
        output_type = x_type

    if x_type != output_type:
        x = ov_opset.convert(x, output_type)

    const_pi_over_180 = ov_opset.constant(pi_over_180, output_type).output(0)
    result = ov_opset.multiply(x, const_pi_over_180).output(0)

    return OpenVINOKerasTensor(result)


def diag(x, k=0):
    x = get_ov_output(x)
    x_shape = x.get_partial_shape()
    rank = x_shape.rank.get_length()

    if rank == 1:
        N_dim = x_shape[0]
        if not N_dim.is_static:
            raise ValueError(
                "diag requires input with static shape for 1D input."
            )
        N = N_dim.get_length()
        output_size = N + np.abs(k)
        out_shape = ov_opset.constant(
            [output_size, output_size], dtype=Type.i32
        ).output(0)
        zeros_const = ov_opset.constant(0, x.get_element_type()).output(0)
        diag_matrix = ov_opset.broadcast(zeros_const, out_shape)

        indices = []
        if k >= 0:
            for i in range(N):
                indices.append([i, i + k])
        else:
            for i in range(N):
                indices.append([i - k, i])

        indices = np.array(indices, dtype=np.int32)
        indices_const = ov_opset.constant(indices, dtype=Type.i32).output(0)
        updated = ov_opset.scatter_nd_update(diag_matrix, indices_const, x)
        return OpenVINOKerasTensor(updated.output(0))

    elif rank == 2:
        M_dim = x_shape[0]
        N_dim = x_shape[1]
        if not M_dim.is_static or not N_dim.is_static:
            raise ValueError(
                "diag requires input with static shape for 2D input."
            )
        M = M_dim.get_length()
        N = N_dim.get_length()

        if k >= 0:
            L = np.minimum(M, N - k) if (N - k) > 0 else 0
            indices = [[i, i + k] for i in range(L)]
        else:
            L = np.minimum(M + k, N) if (M + k) > 0 else 0
            indices = [[i - k, i] for i in range(L)]

        if L <= 0:
            keras_dtype = ov_to_keras_type(x.get_element_type())
            np_dtype = np.dtype(keras_dtype)
            empty_np = np.empty((0,), dtype=np_dtype)
            empty_const = ov_opset.constant(
                empty_np, x.get_element_type()
            ).output(0)
            return OpenVINOKerasTensor(empty_const)

        indices = np.array(indices, dtype=np.int32)
        indices_const = ov_opset.constant(indices, dtype=Type.i32).output(0)
        diag_vec = ov_opset.gather_nd(x, indices_const)
        return OpenVINOKerasTensor(diag_vec.output(0))

    else:
        raise ValueError("diag supports only 1D or 2D tensors")


def diagonal(x, offset=0, axis1=0, axis2=1):
    x = get_ov_output(x)
    shape = x.get_partial_shape()
    rank = x.get_partial_shape().rank.get_length()
    if rank is None:
        raise ValueError("`diagonal` requires input tensor with static rank.")
    if rank < 2:
        raise ValueError(
            f"diagonal requires input tensor with rank >= 2.Given rank: {rank}"
        )
    axis1 = canonicalize_axis(axis1, rank)
    axis2 = canonicalize_axis(axis2, rank)
    if axis1 == axis2:
        raise ValueError("`axis1` and `axis2` cannot be the same.")

    perm_order = [axis1, axis2] + [
        i for i in range(rank) if i != axis1 and i != axis2
    ]
    perm_const = ov_opset.constant(perm_order, dtype=Type.i32).output(0)
    x_transposed = ov_opset.transpose(x, perm_const)

    N_dim = shape[axis1]
    M_dim = shape[axis2]
    if not N_dim.is_static or not M_dim.is_static:
        raise ValueError(
            "`diagonal` requires input tensor with static shape for axes "
            f"`axis1` ({axis1}) and `axis2` ({axis2})."
        )
    N = N_dim.get_length()
    M = M_dim.get_length()
    if offset >= 0:
        L = np.minimum(N, M - offset) if (M - offset) > 0 else 0
        indices = [[i, i + offset] for i in range(L)]
    else:
        L = np.minimum(N + offset, M) if (N + offset) > 0 else 0
        indices = [[i - offset, i] for i in range(L)]

    indices = np.array(indices, dtype=np.int32).reshape(L, 2)
    indices_const = ov_opset.constant(indices, dtype=Type.i32).output(0)

    diag_gathered = ov_opset.gather_nd(x_transposed, indices_const)

    out_rank = rank - 1
    out_perm_order = list(range(1, out_rank)) + [0]
    out_perm_const = ov_opset.constant(out_perm_order, dtype=Type.i32).output(0)

    final_output = ov_opset.transpose(diag_gathered, out_perm_const)
    return OpenVINOKerasTensor(final_output.output(0))


def diff(a, n=1, axis=-1):
    if n == 0:
        return OpenVINOKerasTensor(get_ov_output(a))
    if n < 0:
        raise ValueError(f"order must be non-negative but got {repr(n)}")
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
    x_node = get_ov_output(x)

    if isinstance(bins, OpenVINOKerasTensor):
        bins_node = get_ov_output(bins)
    else:
        bins_np = np.asarray(bins)
        if bins_np.ndim != 1:
            raise ValueError("`bins` must be 1-D array-like")
        bins_node = ov_opset.constant(bins_np).output(0)

    x_node, bins_node = _align_operand_types(x_node, bins_node, "digitize()")

    if x_node.get_element_type() == Type.boolean:
        x_node = ov_opset.convert(x_node, Type.f32).output(0)
        bins_node = ov_opset.convert(bins_node, Type.f32).output(0)

    result = ov_opset.bucketize(
        x_node,
        bins_node,
        output_type=Type.i32,
        with_right_bound=False,
    ).output(0)

    return OpenVINOKerasTensor(result)


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


def empty_like(x, dtype=None):
    return zeros_like(x, dtype=dtype)


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
    x_node = get_ov_output(x)

    # Using OpenVINO tensor shape
    ndim = len(x_node.get_partial_shape())
    if ndim is None:
        raise ValueError(
            "The `flip` operation does not support tensors with dynamic rank "
            "for the OpenVINO backend."
        )

    if axis is None:
        axis = list(range(ndim))
    elif isinstance(axis, int):
        axis = [axis]

    axis = [a + ndim if a < 0 else a for a in axis]

    begin = [0] * ndim
    end = [0] * ndim
    strides = [1] * ndim
    for a in axis:
        strides[a] = -1

    all_ones_mask = [1] * ndim
    result = ov_opset.strided_slice(
        data=x_node,
        begin=begin,
        end=end,
        strides=strides,
        begin_mask=all_ones_mask,
        end_mask=all_ones_mask,
    )
    return OpenVINOKerasTensor(result.output(0))


def rot90(array, k=1, axes=(0, 1)):
    """Rotate an array by 90 degrees in the plane specified by axes."""
    array = get_ov_output(array)

    if not isinstance(axes, (tuple, list)) or len(axes) != 2:
        raise ValueError("axes must be a tuple of length 2")

    shape = array.get_partial_shape()
    ndim = shape.rank.get_length()
    if ndim is None:
        raise ValueError(
            "`rot90` does not support tensors with dynamic rank "
            "for the OpenVINO backend."
        )

    axis1 = canonicalize_axis(axes[0], ndim)
    axis2 = canonicalize_axis(axes[1], ndim)

    if axis1 == axis2:
        raise ValueError("axes must be different")

    k = k % 4
    if k == 0:
        return OpenVINOKerasTensor(array)

    result = array

    for _ in range(k):
        # 1️ Transpose axis1 <-> axis2
        perm = list(range(ndim))
        perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
        perm_const = ov_opset.constant(perm, Type.i32).output(0)
        result = ov_opset.transpose(result, perm_const).output(0)

        # 2️ Reverse along axis1 using StridedSlice
        begin = [0] * ndim
        end = [0] * ndim
        strides = [1] * ndim
        strides[axis1] = -1

        begin_mask = [1] * ndim
        end_mask = [1] * ndim

        result = ov_opset.strided_slice(
            data=result,
            begin=begin,
            end=end,
            strides=strides,
            begin_mask=begin_mask,
            end_mask=end_mask,
        ).output(0)

    return OpenVINOKerasTensor(result)


def floor(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        x = ov_opset.convert(x, OPENVINO_DTYPES[config.floatx()])
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


def gcd(x1, x2):
    raise NotImplementedError("`gcd` is not supported with openvino backend")


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


def hypot(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "hypot()")
    x_type = x1.get_element_type()
    if x_type.is_integral() or x_type == Type.boolean:
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x1 = ov_opset.convert(x1, ov_type)
        x2 = ov_opset.convert(x2, ov_type)
    x1_abs = ov_opset.absolute(x1)
    x2_abs = ov_opset.absolute(x2)
    max_val = ov_opset.maximum(x1_abs, x2_abs)
    min_val = ov_opset.minimum(x1_abs, x2_abs)
    one = ov_opset.constant(1, max_val.get_element_type())
    is_zero_mask = ov_opset.equal(
        max_val, ov_opset.constant(0, max_val.get_element_type())
    )
    safe_divisor = ov_opset.select(is_zero_mask, one, max_val)
    ratio = ov_opset.divide(min_val, safe_divisor)
    result = ov_opset.multiply(
        max_val,
        ov_opset.sqrt(ov_opset.add(one, ov_opset.multiply(ratio, ratio))),
    )
    return OpenVINOKerasTensor(result.output(0))


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
    # NOTE: openvino has an is_finite operation but it does not properly
    # catch np.inf and -np.inf as not finite values. Hence we bootstrap here. If
    # that ever changes, we could simplify this to just call that operation.
    inf_values = get_ov_output(isinf(x))
    nan_values = get_ov_output(isnan(x))
    all_non_finite_values = ov_opset.logical_or(inf_values, nan_values).output(
        0
    )
    is_finite = ov_opset.logical_not(all_non_finite_values).output(0)
    return OpenVINOKerasTensor(is_finite)


def isin(x1, x2, assume_unique=False, invert=False):
    x1 = get_ov_output(x1)
    x2 = get_ov_output(x2)
    output_shape = ov_opset.shape_of(x1).output(0)
    x1, x2 = _align_operand_types(x1, x2, "isin()")

    minus_one = ov_opset.constant([-1], dtype=Type.i64)
    x1 = ov_opset.reshape(x1, minus_one, special_zero=False).output(0)
    x2 = ov_opset.reshape(x2, minus_one, special_zero=False).output(0)
    if not assume_unique:
        x2 = ov_opset.unique(x2).output(0)
    x1 = ov_opset.unsqueeze(x1, 1).output(0)
    x2 = ov_opset.unsqueeze(x2, 0).output(0)
    cmp = ov_opset.equal(x1, x2).output(0)
    result_flat = ov_opset.reduce_logical_or(cmp, 1).output(0)

    if invert:
        result_flat = ov_opset.logical_not(result_flat).output(0)
    result = ov_opset.reshape(result_flat, output_shape, False).output(0)
    return OpenVINOKerasTensor(result)


def isinf(x):
    pos_inf = get_ov_output(isposinf(x))
    neg_inf = get_ov_output(isneginf(x))
    inf = ov_opset.logical_or(pos_inf, neg_inf).output(0)
    return OpenVINOKerasTensor(inf)


def isnan(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        x = ov_opset.convert(x, OPENVINO_DTYPES[config.floatx()])
    return OpenVINOKerasTensor(ov_opset.is_nan(x).output(0))


def isneginf(x):
    return _is_inf(x, pos=False)


def isposinf(x):
    return _is_inf(x)


def _is_inf(x, pos=True):
    # NOTE: there is an ov_opset.is_inf but it does not catch
    # numpy infinite values like np.inf and -np.inf, hence why we have this
    # if this ever changes in OpenVINO, we can do this instead:
    # ov_opset.is_inf(x, {"detect_positive": pos, "detect_negative": not pos})
    # for each infinite sign
    inf_value = np.inf if pos else -np.inf
    x = get_ov_output(x)
    x_type = x.get_element_type()

    if x_type.is_integral() or x_type == Type.boolean:
        shape = ov_opset.shape_of(x, "i32").output(0)
        false_const = ov_opset.constant(False, Type.boolean).output(0)
        return OpenVINOKerasTensor(
            ov_opset.broadcast(false_const, shape).output(0)
        )

    if x_type == Type.bf16:
        x_f32 = ov_opset.convert(x, Type.f32).output(0)
        inf = ov_opset.constant(inf_value, Type.f32).output(0)
        is_inf = ov_opset.equal(x_f32, inf).output(0)
    else:
        if x_type == Type.f16:
            inf = ov_opset.constant(inf_value, Type.f16).output(0)
        elif x_type == Type.f32:
            inf = ov_opset.constant(inf_value, Type.f32).output(0)
        elif x_type == Type.f64:
            inf = ov_opset.constant(inf_value, Type.f64).output(0)
        else:
            inf = ov_opset.constant(inf_value, Type.f32).output(0)
        is_inf = ov_opset.equal(x, inf).output(0)
    return OpenVINOKerasTensor(is_inf)


def isreal(x):
    raise NotImplementedError("`isreal` is not supported with openvino backend")


def kron(x1, x2):
    x1 = get_ov_output(x1)
    x2 = get_ov_output(x2)
    x1, x2 = _align_operand_types(x1, x2, "kron()")
    x1_shape = x1.get_partial_shape()
    x2_shape = x2.get_partial_shape()
    if x1_shape.rank.is_dynamic or x2_shape.rank.is_dynamic:
        raise ValueError(
            "`kron` does not support tensors with dynamic rank for "
            "the OpenVINO backend."
        )
    ndim1 = x1_shape.rank.get_length()
    ndim2 = x2_shape.rank.get_length()
    if ndim1 < ndim2:
        axes = ov_opset.range(
            ov_opset.constant(0, Type.i32),
            ov_opset.constant(ndim2 - ndim1, Type.i32),
            ov_opset.constant(1, Type.i32),
        )
        x1 = ov_opset.unsqueeze(x1, axes)
        ndim1 = ndim2
    elif ndim2 < ndim1:
        axes = ov_opset.range(
            ov_opset.constant(0, Type.i32),
            ov_opset.constant(ndim1 - ndim2, Type.i32),
            ov_opset.constant(1, Type.i32),
        )
        x2 = ov_opset.unsqueeze(x2, axes)
        ndim2 = ndim1
    shape1 = ov_opset.shape_of(x1, Type.i32)
    shape2 = ov_opset.shape_of(x2, Type.i32)
    ones = ov_opset.broadcast(
        ov_opset.constant(1, Type.i32), ov_opset.constant([ndim1], Type.i32)
    )
    axis = ov_opset.constant(1, Type.i32)
    flatten = ov_opset.constant([-1], Type.i32)
    unsqueezed_ones = ov_opset.unsqueeze(ones, axis)
    x1_new_shape = ov_opset.reshape(
        ov_opset.concat(
            [ov_opset.unsqueeze(shape1, axis), unsqueezed_ones],
            axis=1,
        ),
        flatten,
        False,
    )
    x2_new_shape = ov_opset.reshape(
        ov_opset.concat(
            [unsqueezed_ones, ov_opset.unsqueeze(shape2, axis)],
            axis=1,
        ),
        flatten,
        False,
    )
    result = ov_opset.multiply(
        ov_opset.reshape(x1, x1_new_shape, False),
        ov_opset.reshape(x2, x2_new_shape, False),
    )
    result = ov_opset.reshape(
        result, ov_opset.multiply(shape1, shape2), False
    ).output(0)
    return OpenVINOKerasTensor(result)


def lcm(x1, x2):
    raise NotImplementedError("`lcm` is not supported with openvino backend")


def ldexp(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "ldexp()")

    float_dtype = OPENVINO_DTYPES[config.floatx()]
    if x1.get_element_type().is_integral():
        x1 = ov_opset.convert(x1, float_dtype)
    if x2.get_element_type().is_integral():
        x2 = ov_opset.convert(x2, float_dtype)

    const_two = ov_opset.constant(2, x2.get_element_type())
    result = ov_opset.multiply(x1, ov_opset.power(const_two, x2))

    return OpenVINOKerasTensor(result.output(0))


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
    """Return evenly spaced numbers over a specified interval.

    Supports axis=0 (prepend) and axis=-1 (append). Intermediate axis values are
    treated as axis=-1.

    If `retstep` is True, also returns the step size between values.

    """

    start = get_ov_output(start)
    stop = get_ov_output(stop)

    if hasattr(num, "output") or isinstance(num, OpenVINOKerasTensor):
        num_tensor = get_ov_output(num)
        try:
            if num_tensor.get_node().get_type_name() == "Constant":
                num_value = num_tensor.get_node().get_vector()[0]
                num = int(num_value)
            else:
                raise NotImplementedError(
                    "Dynamic num values not fully supported"
                )
        except Exception as e:
            raise NotImplementedError(
                "Could not extract num value from tensor"
            ) from e
    else:
        num = int(num)

    if dtype is None:
        output_type = OPENVINO_DTYPES[config.floatx()]
    else:
        output_type = OPENVINO_DTYPES[dtype]

    start = ov_opset.convert(start, output_type).output(0)
    stop = ov_opset.convert(stop, output_type).output(0)

    if num < 0:
        raise ValueError("Number of samples, `num`, must be non-negative.")

    if num == 0:
        empty_shape = ov_opset.constant([0], Type.i32).output(0)
        result = ov_opset.broadcast(
            ov_opset.constant(0.0, output_type).output(0), empty_shape
        ).output(0)
        if retstep:
            nan_step = ov_opset.constant(np.nan, output_type).output(0)
            return OpenVINOKerasTensor(result), OpenVINOKerasTensor(nan_step)
        return OpenVINOKerasTensor(result)

    if num == 1:
        result_val = start
        axis_const = ov_opset.constant([axis], Type.i32).output(0)
        result = ov_opset.unsqueeze(result_val, axis_const).output(0)
        if retstep:
            if endpoint:
                step = ov_opset.constant(np.nan, output_type).output(0)
            else:
                step = ov_opset.subtract(stop, start).output(0)
            return OpenVINOKerasTensor(result), OpenVINOKerasTensor(step)
    zero_i32 = ov_opset.constant(0, Type.i32).output(0)
    one_i32 = ov_opset.constant(1, Type.i32).output(0)
    one_i32_array = ov_opset.constant([1], Type.i32).output(0)

    num_const = ov_opset.constant(num, output_type).output(0)

    if endpoint:
        divisor = ov_opset.subtract(
            num_const, ov_opset.constant(1, output_type).output(0)
        ).output(0)
    else:
        divisor = num_const

    step = ov_opset.divide(
        ov_opset.subtract(stop, start).output(0), divisor
    ).output(0)

    indices = ov_opset.range(
        zero_i32,
        ov_opset.constant(num, Type.i32).output(0),
        one_i32,
        output_type,
    ).output(0)

    start_shape = ov_opset.convert(
        ov_opset.shape_of(start).output(0), Type.i32
    ).output(0)
    indices_shape = ov_opset.convert(
        ov_opset.shape_of(indices).output(0), Type.i32
    ).output(0)

    start_rank = ov_opset.shape_of(start_shape).output(0)
    ones_for_start = ov_opset.broadcast(one_i32, start_rank).output(0)

    if axis == 0:
        indices_target_shape = ov_opset.concat(
            [indices_shape, ones_for_start], 0
        ).output(0)
        start_target_shape = ov_opset.concat(
            [one_i32_array, start_shape], 0
        ).output(0)
    else:
        indices_target_shape = ov_opset.concat(
            [ones_for_start, indices_shape], 0
        ).output(0)
        start_target_shape = ov_opset.concat(
            [start_shape, one_i32_array], 0
        ).output(0)

    indices_reshaped = ov_opset.reshape(
        indices, indices_target_shape, False
    ).output(0)
    start_reshaped = ov_opset.reshape(start, start_target_shape, False).output(
        0
    )
    step_reshaped = ov_opset.reshape(step, start_target_shape, False).output(0)

    scaled_indices = ov_opset.multiply(indices_reshaped, step_reshaped).output(
        0
    )
    result = ov_opset.add(start_reshaped, scaled_indices).output(0)

    if retstep:
        return OpenVINOKerasTensor(result), OpenVINOKerasTensor(step)
    return OpenVINOKerasTensor(result)


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
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "logaddexp()")

    if x1.element_type.is_integral() or x2.element_type.is_integral():
        float_dtype = OPENVINO_DTYPES[config.floatx()]
        if x1.element_type.is_integral():
            x1 = ov_opset.convert(x1, float_dtype)
        if x2.element_type.is_integral():
            x2 = ov_opset.convert(x2, float_dtype)

    # Get the output nodes properly
    max_val_node = ov_opset.maximum(x1, x2)
    max_val = max_val_node.output(0)

    # Compute absolute difference
    sub_node = ov_opset.subtract(x1, x2)
    abs_diff_node = ov_opset.abs(sub_node.output(0))
    abs_diff = abs_diff_node.output(0)

    # Compute negative absolute difference and its exponential
    neg_abs_diff_node = ov_opset.negative(abs_diff)
    neg_abs_diff = neg_abs_diff_node.output(0)
    exp_neg_abs_node = ov_opset.exp(neg_abs_diff)
    exp_neg_abs = exp_neg_abs_node.output(0)

    # Get the element type from the node, not the output
    element_type = exp_neg_abs_node.get_element_type()
    one_node = ov_opset.constant(1, element_type)
    one = one_node.output(0)

    # Compute log term
    one_plus_exp_node = ov_opset.add(one, exp_neg_abs)
    one_plus_exp = one_plus_exp_node.output(0)
    log_term_node = ov_opset.log(one_plus_exp)
    log_term = log_term_node.output(0)

    # Final result
    result_node = ov_opset.add(max_val, log_term)
    result = result_node.output(0)

    return OpenVINOKerasTensor(result)


def logaddexp2(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "logaddexp2()")

    if x1.element_type.is_integral() or x2.element_type.is_integral():
        float_dtype = OPENVINO_DTYPES[config.floatx()]
        if x1.get_element_type().is_integral():
            x1 = ov_opset.convert(x1, float_dtype)
        if x2.get_element_type().is_integral():
            x2 = ov_opset.convert(x2, float_dtype)

    max_val = ov_opset.maximum(x1, x2)

    sub = ov_opset.subtract(x1, x2)
    abs_diff = ov_opset.abs(sub)

    neg_abs_diff = ov_opset.negative(abs_diff)

    element_type = neg_abs_diff.get_element_type()

    two = ov_opset.constant(2, dtype=element_type)

    power_of_2 = ov_opset.power(two, neg_abs_diff)

    one_plus_power = ov_opset.add(
        ov_opset.constant(1, dtype=element_type), power_of_2
    )
    log2_term = ov_opset.divide(ov_opset.log(one_plus_power), ov_opset.log(two))
    result = ov_opset.add(max_val, log2_term).output(0)

    return OpenVINOKerasTensor(result)


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
    linear_samples = linspace(
        start=start,
        stop=stop,
        num=num,
        endpoint=endpoint,
        retstep=False,
        dtype=dtype,
        axis=axis,
    )

    if dtype is None:
        output_type = OPENVINO_DTYPES[config.floatx()]
    else:
        output_type = OPENVINO_DTYPES[dtype]

    linear_output = get_ov_output(linear_samples)
    base_tensor = get_ov_output(base)

    base_tensor = ov_opset.convert(base_tensor, output_type).output(0)

    result = ov_opset.power(base_tensor, linear_output).output(0)

    return OpenVINOKerasTensor(result)


def maximum(x1, x2):
    x1 = get_ov_output(x1)
    x2 = get_ov_output(x2)
    x1, x2 = _align_operand_types(x1, x2, "maximum()")
    return OpenVINOKerasTensor(ov_opset.maximum(x1, x2).output(0))


def median(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    x_shape = x.get_partial_shape()
    rank = x_shape.rank.get_length()

    if rank == 0:
        return OpenVINOKerasTensor(x)

    # Handle axis=None by flattening the input
    flattened_all = False
    if axis is None:
        x = ov_opset.reshape(x, [-1], False).output(0)
        axis = 0
        original_rank = rank
        rank = 1
        flattened_all = True
    else:
        # Handle tuple axis - for median, we only support single axis
        if isinstance(axis, (tuple, list)):
            if len(axis) != 1:
                raise ValueError("median only supports single axis reduction")
            axis = axis[0]

        # Handle negative axis
        if axis < 0:
            axis = rank + axis
        original_rank = rank

    # Get the size of the dimension to sort
    shape_tensor = ov_opset.shape_of(x, output_type=Type.i32).output(0)
    k = ov_opset.gather(
        shape_tensor,
        ov_opset.constant([axis], Type.i32).output(0),
        ov_opset.constant(0, Type.i32).output(0),
    ).output(0)

    # Convert k to a scalar value
    k_scalar = ov_opset.squeeze(k, [0]).output(0)

    # Use topk with k=size_of_axis to get all elements sorted
    topk_outputs = ov_opset.topk(
        x, k=k_scalar, axis=axis, mode="min", sort="value", stable=True
    )

    # Get the sorted values
    sorted_values = topk_outputs.output(0)

    # Convert to float for median calculation
    x1_type = ov_to_keras_type(sorted_values.get_element_type())
    result_type = dtypes.result_type(x1_type, float)
    result_type = OPENVINO_DTYPES[result_type]
    sorted_values = ov_opset.convert(sorted_values, result_type).output(0)

    # Calculate median indices
    # For odd length: median_idx = (k-1) // 2
    # For even length: we need indices (k//2 - 1) and k//2, then average

    k_minus_1 = ov_opset.subtract(
        k_scalar, ov_opset.constant(1, Type.i32).output(0)
    ).output(0)
    k_div_2 = ov_opset.divide(
        k_scalar, ov_opset.constant(2, Type.i32).output(0)
    ).output(0)
    k_minus_1_div_2 = ov_opset.divide(
        k_minus_1, ov_opset.constant(2, Type.i32).output(0)
    ).output(0)

    # Check if k is odd
    k_mod_2 = ov_opset.mod(
        k_scalar, ov_opset.constant(2, Type.i32).output(0)
    ).output(0)
    is_odd = ov_opset.equal(
        k_mod_2, ov_opset.constant(1, Type.i32).output(0)
    ).output(0)

    # For odd case: take the middle element
    odd_idx = k_minus_1_div_2

    # For even case: take average of two middle elements
    even_idx1 = ov_opset.subtract(
        k_div_2, ov_opset.constant(1, Type.i32).output(0)
    ).output(0)
    even_idx2 = k_div_2

    # Gather elements for both cases
    # Create gather indices tensor for the axis
    gather_indices_odd = ov_opset.unsqueeze(odd_idx, [0]).output(0)
    gather_indices_even1 = ov_opset.unsqueeze(even_idx1, [0]).output(0)
    gather_indices_even2 = ov_opset.unsqueeze(even_idx2, [0]).output(0)

    # Gather the median elements
    odd_result = ov_opset.gather(
        sorted_values,
        gather_indices_odd,
        ov_opset.constant(axis, Type.i32).output(0),
    ).output(0)
    even_result1 = ov_opset.gather(
        sorted_values,
        gather_indices_even1,
        ov_opset.constant(axis, Type.i32).output(0),
    ).output(0)
    even_result2 = ov_opset.gather(
        sorted_values,
        gather_indices_even2,
        ov_opset.constant(axis, Type.i32).output(0),
    ).output(0)

    # Average the two middle elements for even case
    even_sum = ov_opset.add(even_result1, even_result2).output(0)
    even_result = ov_opset.divide(
        even_sum, ov_opset.constant(2.0, result_type).output(0)
    ).output(0)

    # Select between odd and even results
    median_result = ov_opset.select(is_odd, odd_result, even_result).output(0)

    # Remove the gathered dimension (squeeze)
    median_result = ov_opset.squeeze(median_result, [axis]).output(0)

    # Handle keepdims
    if keepdims:
        if flattened_all:
            # When axis=None, keepdims should restore all dimensions as 1
            ones_shape = ov_opset.constant(
                [1] * original_rank, Type.i32
            ).output(0)
            median_result = ov_opset.reshape(
                median_result, ones_shape, False
            ).output(0)
        else:
            median_result = ov_opset.unsqueeze(median_result, [axis]).output(0)

    return OpenVINOKerasTensor(median_result)


def meshgrid(*x, indexing="xy"):
    if len(x) < 2:
        raise ValueError(
            "meshgrid requires at least 2 input arrays. "
            f"Received: {len(x)} input array(s)."
        )
    if indexing not in ("xy", "ij"):
        raise ValueError("indexing must be either 'xy' or 'ij'")

    tensors = [get_ov_output(xi) for xi in x]
    n = len(tensors)

    shapes = [
        ov_opset.shape_of(t, Type.i64).output(0) for t in tensors
    ]  # each is [Ni]
    one = ov_opset.constant([1], Type.i64).output(0)

    if indexing == "xy":
        shape_list = [shapes[1], shapes[0]] + shapes[2:]
        out_shape = ov_opset.concat(shape_list, axis=0).output(0)
    else:
        out_shape = ov_opset.concat(shapes, axis=0).output(0)

    outputs = []
    for i, t in enumerate(tensors):
        reshape_parts = [one] * n
        if indexing == "xy":
            if i == 0:
                reshape_parts[1] = shapes[0]
            elif i == 1:
                reshape_parts[0] = shapes[1]
            else:
                reshape_parts[i] = shapes[i]
        else:
            reshape_parts[i] = shapes[i]

        reshape_shape = ov_opset.concat(reshape_parts, axis=0).output(0)
        reshaped = ov_opset.reshape(t, reshape_shape, False).output(0)
        broadcasted = ov_opset.broadcast(reshaped, out_shape).output(0)
        outputs.append(OpenVINOKerasTensor(broadcasted))

    return outputs


def min(x, axis=None, keepdims=False, initial=None):
    return _compute_extrema(x, "min", axis, keepdims, initial)


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


def nanmax(x, axis=None, keepdims=False):
    raise NotImplementedError("`nanmax` is not supported with openvino backend")


def nanmean(x, axis=None, keepdims=False):
    raise NotImplementedError(
        "`nanmean` is not supported with openvino backend"
    )


def nanmin(x, axis=None, keepdims=False):
    raise NotImplementedError("`nanmin` is not supported with openvino backend")


def nansum(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    x_type = x.get_element_type()

    if not x_type.is_integral() and x_type != Type.boolean:
        nan_mask = ov_opset.is_nan(x)
        zero = ov_opset.constant(0, x_type)
        x = ov_opset.select(nan_mask, zero, x).output(0)

    x, axis = _resolve_axis(x, axis)
    if axis is None:
        return OpenVINOKerasTensor(x)

    x = _upcast_type_if_needed(x)
    result = ov_opset.reduce_sum(x, axis, keepdims).output(0)

    return OpenVINOKerasTensor(result)


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
        pad_value = ov_opset.constant(
            constant_values, x.get_element_type()
        ).output(0)

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
    x = get_ov_output(x)

    # If a specific dtype is requested, cast the input to that dtype.
    if dtype is not None:
        ov_dtype = OPENVINO_DTYPES[standardize_dtype(dtype)]
        x = ov_opset.convert(x, ov_dtype).output(0)
    # Otherwise, apply dtype promotion rules before reduction.
    else:
        x = _upcast_type_if_needed(x)
    x, axis = _resolve_axis(x, axis)
    if axis is None:
        return OpenVINOKerasTensor(x)
    # Compute the product
    result = ov_opset.reduce_prod(x, axis, keepdims).output(0)

    return OpenVINOKerasTensor(result)


def ptp(x, axis=None, keepdims=False):
    if axis == ():
        return zeros_like(x)
    x = get_ov_output(x)

    x_resolved, resolved_axis = _resolve_axis(x, axis)

    max_val = ov_opset.reduce_max(x_resolved, resolved_axis, keepdims)
    min_val = ov_opset.reduce_min(x_resolved, resolved_axis, keepdims)

    return OpenVINOKerasTensor(ov_opset.subtract(max_val, min_val).output(0))


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
    x = get_ov_output(x)
    const_0 = ov_opset.constant(0, Type.i32)
    const_1 = ov_opset.constant(1, Type.i32)
    const_neg_1 = ov_opset.constant([-1], Type.i32)

    if axis is not None and axis < 0:
        axis += len(x.get_partial_shape())

    if axis is None:
        x = ov_opset.reshape(x, const_neg_1, special_zero=False)
        axis = 0

    if isinstance(repeats, (int, np.integer)) or (
        isinstance(repeats, np.ndarray)
        and repeats.ndim == 1
        and repeats.size == 1
    ):
        repeats_val = (
            int(repeats)
            if isinstance(repeats, (np.integer, np.ndarray))
            else repeats
        )
        dim_len = ov_opset.gather(
            ov_opset.shape_of(x, Type.i32),
            ov_opset.constant([axis], Type.i32),
            const_0,
        )
        dim_len = ov_opset.squeeze(dim_len, ov_opset.constant([0], Type.i32))
        idx_range = ov_opset.range(
            const_0, dim_len, const_1, output_type=Type.i32
        )
        idx_range = ov_opset.unsqueeze(idx_range, const_1)
        tiled = ov_opset.tile(
            idx_range, ov_opset.constant([1, repeats_val], Type.i32)
        )
        idx = ov_opset.reshape(tiled, const_neg_1, special_zero=False)
        result = ov_opset.gather(x, idx, ov_opset.constant(axis, Type.i32))
        return OpenVINOKerasTensor(result.output(0))
    repeats_tensor = get_ov_output(repeats)
    cumsum = ov_opset.cumsum(repeats_tensor, const_0)
    total = ov_opset.reduce_sum(
        repeats_tensor, ov_opset.constant([0], Type.i32), keep_dims=False
    )
    total = ov_opset.convert(total, Type.i32)
    out_indices = ov_opset.range(const_0, total, const_1, output_type=Type.i32)
    cumsum_unsq = ov_opset.unsqueeze(cumsum, const_0)
    out_indices_unsq = ov_opset.unsqueeze(out_indices, const_1)
    cumsum_unsq = ov_opset.convert(cumsum_unsq, Type.i32)
    mask = ov_opset.greater_equal(out_indices_unsq, cumsum_unsq)
    gather_indices = ov_opset.reduce_sum(
        ov_opset.convert(mask, Type.i32), ov_opset.constant([1], Type.i32)
    )
    result = ov_opset.gather(
        x, gather_indices, ov_opset.constant(axis, Type.i32)
    )
    return OpenVINOKerasTensor(result.output(0))


def reshape(x, newshape):
    x = get_ov_output(x)
    if isinstance(newshape, tuple):
        newshape = list(newshape)
    newshape = ov_opset.constant(newshape, Type.i32).output(0)
    return OpenVINOKerasTensor(ov_opset.reshape(x, newshape, False).output(0))


def roll(x, shift, axis=None):
    x = get_ov_output(x)
    if axis is not None:
        result = ov_opset.roll(x, shift, axis).output(0)
    else:
        output_shape = ov_opset.shape_of(x).output(0)
        flattened = ov_opset.reshape(
            x, ov_opset.constant([-1], Type.i32), False
        ).output(0)
        result = ov_opset.roll(flattened, shift, 0).output(0)
        result = ov_opset.reshape(result, output_shape, False).output(0)
    return OpenVINOKerasTensor(result)


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
    x = get_ov_output(x)
    shape_tensor = ov_opset.shape_of(x, output_type=Type.i64)
    final_size = ov_opset.reduce_prod(
        shape_tensor,
        ov_opset.constant([0], Type.i64),
        keep_dims=False,
    )
    return OpenVINOKerasTensor(final_size.output(0))


def sort(x, axis=-1):
    x = get_ov_output(x)
    x_shape = x.get_partial_shape()
    rank = x_shape.rank.get_length()

    if rank == 0:
        return OpenVINOKerasTensor(x)

    # Handle axis=None by flattening the input
    if axis is None:
        x = ov_opset.reshape(
            x, ov_opset.constant([-1], Type.i32), False
        ).output(0)
        axis = 0
    # Handle negative axis
    elif axis < 0:
        axis = rank + axis

    # Get the size of the dimension to sort
    shape_tensor = ov_opset.shape_of(x, output_type=Type.i32).output(0)
    k = ov_opset.gather(
        shape_tensor,
        ov_opset.constant([axis], Type.i32).output(0),
        ov_opset.constant(0, Type.i32).output(0),
    ).output(0)

    # Convert k to a scalar value
    k_scalar = ov_opset.squeeze(k, ov_opset.constant([0], Type.i32)).output(0)

    # Use topk with k=size_of_axis to get all elements sorted
    topk_outputs = ov_opset.topk(
        x, k=k_scalar, axis=axis, mode="min", sort="value", stable=True
    )

    # Get the sorted values
    sorted_values = topk_outputs.output(0)

    return OpenVINOKerasTensor(sorted_values)


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


def array_split(x, indices_or_sections, axis=0):
    original_shape = x.shape
    x = get_ov_output(x)

    num_splits_val = indices_or_sections
    total_size = original_shape[axis]
    if total_size is None:
        raise ValueError(
            f"Cannot use array_split with static Python logic on dynamic axis. "
            f"Axis {axis} has unknown dimension for shape {original_shape}."
        )

    base_size = total_size // num_splits_val
    remainder = total_size % num_splits_val

    split_lengths = [base_size + 1] * remainder + [base_size] * (
        num_splits_val - remainder
    )
    split_lengths_tensor = ov_opset.constant(
        split_lengths, dtype=Type.i64
    ).output(0)

    axis_tensor = ov_opset.constant(axis, dtype=Type.i32).output(0)
    splits = ov_opset.variadic_split(x, axis_tensor, split_lengths_tensor)

    result = []
    for i in range(num_splits_val):
        result.append(OpenVINOKerasTensor(splits.output(i)))
    return result


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
    var_x = var(x, axis, keepdims)
    std_dev = ov_opset.sqrt(var_x).output(0)
    return OpenVINOKerasTensor(std_dev)


def swapaxes(x, axis1, axis2):
    x = get_ov_output(x)
    x_shape = x.get_partial_shape()
    if x_shape.rank.is_dynamic:
        raise ValueError(
            "`swapaxes` does not support tensors with dynamic rank for the "
            "OpenVINO backend."
        )
    rank = x_shape.rank.get_length()
    axis1 = canonicalize_axis(axis1, rank)
    axis2 = canonicalize_axis(axis2, rank)
    axes = list(range(rank))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
    result = ov_opset.transpose(x, ov_opset.constant(axes, Type.i32))
    return OpenVINOKerasTensor(result.output(0))


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
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral() or x_type == Type.boolean:
        x = ov_opset.convert(x, OPENVINO_DTYPES[config.floatx()])

    if decimals == 0:
        result = ov_opset.round(x, "half_to_even")
    else:
        factor = ov_opset.constant(10.0**decimals, x.get_element_type())
        scaled = ov_opset.multiply(x, factor)
        rounded = ov_opset.round(scaled, "half_to_even")
        result = ov_opset.divide(rounded, factor)

    if x_type.is_integral():
        result = ov_opset.convert(result, x_type)

    return OpenVINOKerasTensor(result.output(0))


def tile(x, repeats):
    x = get_ov_output(x)

    if isinstance(repeats, int):
        repeats = [repeats]
    repeats = get_ov_output(repeats)

    if repeats.get_element_type() != Type.i64:
        repeats = ov_opset.convert(repeats, Type.i64)

    if len(repeats.get_partial_shape()) != 1:
        repeats = ov_opset.reshape(repeats, [-1], False)

    shape_x = ov_opset.shape_of(x, Type.i64)
    rank_x = ov_opset.shape_of(shape_x, Type.i64)
    rank_r = ov_opset.shape_of(repeats, Type.i64)

    one = ov_opset.constant(1, Type.i64)
    zero = ov_opset.constant(0, Type.i64)

    pad_x = ov_opset.maximum(ov_opset.subtract(rank_r, rank_x), zero)
    new_x_shape = ov_opset.concat(
        [ov_opset.broadcast(one, pad_x).output(0), shape_x], 0
    )
    x = ov_opset.reshape(x, new_x_shape, False)

    pad_r = ov_opset.maximum(ov_opset.subtract(rank_x, rank_r), zero)
    repeats = ov_opset.concat(
        [ov_opset.broadcast(one, pad_r).output(0), repeats], 0
    )

    return OpenVINOKerasTensor(ov_opset.tile(x, repeats).output(0))


def trace(x, offset=0, axis1=0, axis2=1):
    x = diagonal(x, offset=offset, axis1=axis1, axis2=axis2)
    return sum(x, axis=-1)


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
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "vdot()")
    if x1.get_partial_shape().rank == 0 or x2.get_partial_shape().rank == 0:
        return OpenVINOKerasTensor(ov_opset.multiply(x1, x2).output(0))
    flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
    x1 = ov_opset.reshape(x1, flatten_shape, False).output(0)
    x2 = ov_opset.reshape(x2, flatten_shape, False).output(0)
    return OpenVINOKerasTensor(ov_opset.matmul(x1, x2, False, False).output(0))


def vstack(xs):
    if not isinstance(xs, (list, tuple)):
        xs = (xs,)
    elems = [convert_to_tensor(elem) for elem in xs]
    element_type = elems[0].output.get_element_type()
    elems = [get_ov_output(elem, element_type) for elem in elems]
    axis = 0
    for i in range(1, len(elems)):
        elems[0], elems[i] = _align_operand_types(
            elems[0], elems[i], "vstack()"
        )
    return OpenVINOKerasTensor(ov_opset.concat(elems, axis).output(0))


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
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "divide_no_nan()")

    zero = ov_opset.constant(0, x2.get_element_type())
    div = ov_opset.divide(x1, x2)
    is_zero = ov_opset.equal(x2, zero)
    result = ov_opset.select(is_zero, zero, div)
    return OpenVINOKerasTensor(result.output(0))


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


def nextafter(x1, x2):
    raise NotImplementedError(
        "`nextafter` is not supported with openvino backend"
    )


def square(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type == Type.boolean:
        x = ov_opset.convert(x, Type.i32).output(0)
    const_two = ov_opset.constant(2, x.get_element_type()).output(0)
    return OpenVINOKerasTensor(ov_opset.power(x, const_two).output(0))


def sqrt(x):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    if x_type.is_integral():
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, ov_type).output(0)
    return OpenVINOKerasTensor(ov_opset.sqrt(x).output(0))


def squeeze(x, axis=None):
    x = get_ov_output(x)
    if axis is None:
        axis = []
        for idx, dim in enumerate(x.get_partial_shape()):
            if dim == 1:
                axis.append(idx)
    if isinstance(axis, tuple):
        axis = list(axis)
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


def _helper_trapezoid(y, axis):
    rank = y.get_partial_shape().rank.get_length()
    strides = ov_opset.constant([1] * rank, dtype=Type.i64).output(0)

    # y[:-1]
    begin1 = ov_opset.constant([0] * rank, dtype=Type.i64).output(0)
    end1_list = [0] * rank
    end1_list[axis] = -1
    end1 = ov_opset.constant(end1_list, dtype=Type.i64).output(0)
    begin_mask1 = [1] * rank
    begin_mask1[axis] = 0
    end_mask1 = [1] * rank
    end_mask1[axis] = 0
    y1 = ov_opset.strided_slice(
        y, begin1, end1, strides, begin_mask1, end_mask1
    ).output(0)

    # y[1:]
    begin2_list = [0] * rank
    begin2_list[axis] = 1
    begin2 = ov_opset.constant(begin2_list, dtype=Type.i64).output(0)
    end2 = ov_opset.constant([0] * rank, dtype=Type.i64).output(0)
    begin_mask2 = [1] * rank
    begin_mask2[axis] = 0
    end_mask2 = [1] * rank
    y2 = ov_opset.strided_slice(
        y, begin2, end2, strides, begin_mask2, end_mask2
    ).output(0)

    return y1, y2


def trapezoid(y, x=None, dx=1.0, axis=-1):
    y = get_ov_output(y)
    y_type = y.get_element_type()

    if y_type.is_integral():
        y_type = OPENVINO_DTYPES[config.floatx()]
        y = ov_opset.convert(y, y_type).output(0)

    y1, y2 = _helper_trapezoid(y, axis)
    y_final = ov_opset.add(y1, y2).output(0)
    const_two = ov_opset.constant(2, dtype=y_type).output(0)
    y_final = ov_opset.divide(y_final, const_two).output(0)

    if x is not None:
        x = get_ov_output(x)
        x_type = x.get_element_type()
        if x_type.is_integral():
            x_type = OPENVINO_DTYPES[config.floatx()]
            x = ov_opset.convert(x, x_type).output(0)

        x1, x2 = _helper_trapezoid(x, axis)
        x_final = ov_opset.subtract(x2, x1).output(0)

    else:
        x_final = ov_opset.constant(dx, dtype=y_type).output(0)

    result = ov_opset.multiply(y_final, x_final).output(0)
    const_axis = ov_opset.constant([axis], Type.i64).output(0)
    result = ov_opset.reduce_sum(result, const_axis, False).output(0)

    return OpenVINOKerasTensor(result)


def vander(x, N=None, increasing=False):
    x = get_ov_output(x)
    x_type = x.get_element_type()

    shape_x = ov_opset.shape_of(x, Type.i64).output(0)

    const_zero_1D = ov_opset.constant([0], dtype=Type.i64).output(0)
    const_zero = ov_opset.constant(0, dtype=Type.i64).output(0)
    const_one = ov_opset.constant(1, dtype=Type.i64).output(0)
    const_mone = ov_opset.constant(-1, dtype=Type.i64).output(0)

    if N is None:
        const_N = ov_opset.squeeze(shape_x, const_zero_1D).output(0)
        const_N_1D = shape_x
    else:
        const_N = ov_opset.constant(N, Type.i64).output(0)
        const_N_1D = ov_opset.constant([N], Type.i64).output(0)

    const_N_minus_one = ov_opset.subtract(const_N, const_one).output(0)
    if increasing:
        powers = ov_opset.range(const_zero, const_N, const_one, x_type).output(
            0
        )
    else:
        powers = ov_opset.range(
            const_N_minus_one, const_mone, const_mone, x_type
        ).output(0)

    target_shape = ov_opset.concat([shape_x, const_N_1D], 0).output(0)

    const_one_1D = ov_opset.constant([1], dtype=Type.i64).output(0)

    powers = ov_opset.unsqueeze(powers, const_zero_1D).output(0)
    x = ov_opset.unsqueeze(x, const_one_1D).output(0)

    result = ov_opset.broadcast(x, target_shape).output(0)

    result = ov_opset.power(result, powers).output(0)

    return OpenVINOKerasTensor(result)


def var(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    x_type = x.get_element_type()
    x, axis = _resolve_axis(x, axis)

    work_dtype = Type.f64 if x_type.is_integral() else x.get_element_type()
    if x_type.is_integral():
        x = ov_opset.convert(x, work_dtype).output(0)
    if axis is None:
        const_zero = ov_opset.constant(0, dtype=work_dtype).output(0)
        return OpenVINOKerasTensor(
            ov_opset.broadcast(const_zero, ov_opset.shape_of(x)).output(0)
        )
    # The variance is computed using $Var = E[|x|^2] - |E[x]|^2$, It is faster
    # but less numerically stable.
    mean = ov_opset.reduce_mean(x, axis, keepdims).output(0)
    const_two = ov_opset.constant(2, work_dtype).output(0)

    squared_x = ov_opset.power(x, const_two).output(0)
    squared_mean = ov_opset.power(mean, const_two).output(0)

    squared_x_mean = ov_opset.reduce_mean(squared_x, axis, keepdims).output(0)
    variance = OpenVINOKerasTensor(
        ov_opset.subtract(squared_x_mean, squared_mean).output(0)
    )
    return variance


def sum(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    x, axis = _resolve_axis(x, axis)
    if axis is None:
        return OpenVINOKerasTensor(x)
    x = _upcast_type_if_needed(x)
    summed_value = ov_opset.reduce_sum(x, axis, keepdims).output(0)
    return OpenVINOKerasTensor(summed_value)


def eye(N, M=None, k=0, dtype=None):
    dtype = standardize_dtype(dtype) or config.floatx()
    ov_type = OPENVINO_DTYPES[dtype]
    if M is None:
        M = N
    return OpenVINOKerasTensor(
        ov_opset.eye(
            ov_opset.constant(N, Type.i32),
            ov_opset.constant(M, Type.i32),
            ov_opset.constant(k, Type.i32),
            output_type=ov_type,
        ).output(0)
    )


def floor_divide(x1, x2):
    x1_output = get_ov_output(x1)
    x2_output = get_ov_output(x2)
    if x1_output.get_element_type() == Type.boolean:
        x1_output = ov_opset.convert(x1_output, Type.i32).output(0)
    if isinstance(x2, (int, float)):
        if x1_output.get_element_type().is_integral() and isinstance(x2, float):
            ov_type = OPENVINO_DTYPES[config.floatx()]
        else:
            ov_type = x1_output.get_element_type()
        x1 = ov_opset.convert(x1_output, ov_type).output(0)
        x2 = ov_opset.convert(x2_output, ov_type).output(0)
    else:
        x1, x2 = _align_operand_types(x1_output, x2_output, "floor_divide()")
    div = ov_opset.divide(x1, x2).output(0)
    floored_div = ov_opset.floor(div).output(0)
    return OpenVINOKerasTensor(floored_div)


def logical_xor(x1, x2):
    x1 = get_ov_output(x1)
    x2 = get_ov_output(x2)
    x1 = ov_opset.convert(x1, Type.boolean).output(0)
    x2 = ov_opset.convert(x2, Type.boolean).output(0)
    return OpenVINOKerasTensor(ov_opset.logical_xor(x1, x2).output(0))


def corrcoef(x):
    x_ov = get_ov_output(x)
    x_type = x_ov.get_element_type()
    ov_type = x_type

    if x_type.is_integral():
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x_ov = ov_opset.convert(x_ov, ov_type).output(0)

    const_one = ov_opset.constant(1, dtype=Type.i64).output(0)
    const_two = ov_opset.constant(2, dtype=ov_type).output(0)

    mean = ov_opset.reduce_mean(x_ov, const_one, True).output(0)
    x_ov = ov_opset.subtract(x_ov, mean).output(0)

    cov = ov_opset.matmul(x_ov, x_ov, False, True).output(0)
    xsqr = ov_opset.power(x_ov, const_two).output(0)
    xvar = ov_opset.reduce_sum(xsqr, const_one, True).output(0)
    xstd = ov_opset.sqrt(xvar).output(0)

    den = ov_opset.matmul(xstd, xstd, False, True).output(0)

    result = ov_opset.divide(cov, den).output(0)

    return OpenVINOKerasTensor(result)


def correlate(x1, x2, mode="valid"):
    x1 = get_ov_output(x1)
    x2 = get_ov_output(x2)
    x1_type = x1.get_element_type()
    x2_type = x2.get_element_type()
    x1_type = ov_to_keras_type(x1_type)
    x2_type = ov_to_keras_type(x2_type)
    result_type = dtypes.result_type(x1_type, x2_type, float)

    result_type = OPENVINO_DTYPES[result_type]
    x1 = ov_opset.convert(x1, result_type).output(0)
    x2 = ov_opset.convert(x2, result_type).output(0)

    shape_filter = ov_opset.shape_of(x2, Type.i64).output(0)
    const_two = ov_opset.constant(2, Type.f64).output(0)
    const_one = ov_opset.constant(1, Type.i64).output(0)
    const_zero = ov_opset.constant(0, result_type).output(0)
    shape_filter_minus_one = ov_opset.subtract(shape_filter, const_one).output(
        0
    )

    # padding x1
    if mode == "valid":
        pass

    elif mode == "same":
        shape_minus_one_float = ov_opset.convert(
            shape_filter_minus_one, Type.f64
        ).output(0)

        right = ov_opset.divide(shape_minus_one_float, const_two).output(0)
        left = ov_opset.ceil(right).output(0)
        right = ov_opset.floor(right).output(0)
        left = ov_opset.convert(left, Type.i64).output(0)
        right = ov_opset.convert(right, Type.i64).output(0)
        x1 = ov_opset.pad(x1, left, right, "constant", const_zero).output(0)

    elif mode == "full":
        pad = shape_filter_minus_one
        x1 = ov_opset.pad(x1, pad, pad, "constant", const_zero).output(0)

    else:
        raise ValueError(
            f"mode: {mode} not available chose from valid, same, full."
        )

    axes = ov_opset.constant([0, 1], dtype=Type.i64).output(0)
    x2 = ov_opset.unsqueeze(x2, axes).output(0)
    x1 = ov_opset.unsqueeze(x1, axes).output(0)

    result = ov_opset.convolution(x1, x2, [1], [0], [0], [1]).output(0)

    result = ov_opset.squeeze(result, axes).output(0)

    return OpenVINOKerasTensor(result)


def select(condlist, choicelist, default=0):
    if len(condlist) != len(choicelist):
        raise ValueError(
            "select(): condlist and choicelist must have the same length"
        )
    conds = [get_ov_output(c) for c in condlist]
    choices = [get_ov_output(v) for v in choicelist]

    result = get_ov_output(default)
    for cond_idx in reversed(range(len(conds))):
        cond = conds[cond_idx]
        choice = choices[cond_idx]
        choice, result = _align_operand_types(choice, result, "select()")
        result = ov_opset.select(cond, choice, result).output(0)
    return OpenVINOKerasTensor(result)


def slogdet(x):
    raise NotImplementedError(
        "`slogdet` is not supported with openvino backend"
    )


def argpartition(x, kth, axis=-1):
    x = get_ov_output(x)
    x_shape = x.get_partial_shape()
    rank = x_shape.rank.get_length()
    axis = canonicalize_axis(axis, rank)
    axes = list(range(rank))
    axes[axis], axes[-1] = axes[-1], axes[axis]
    x = ov_opset.transpose(x, ov_opset.constant(axes))
    x_shape_tensor = ov_opset.shape_of(x)
    n = ov_opset.gather(
        x_shape_tensor,
        ov_opset.constant(-1),
        ov_opset.constant(0),
    )
    if isinstance(kth, int) and kth < 0:
        kth_tensor = ov_opset.add(
            n,
            ov_opset.constant(kth, n.get_element_type()),
        )
    else:
        kth_tensor = ov_opset.constant(kth, n.get_element_type())
    one = ov_opset.constant(1, kth_tensor.get_element_type())
    k_val = ov_opset.add(kth_tensor, one)
    bottom_ind = ov_opset.topk(
        ov_opset.negative(x),
        k=k_val,
        axis=-1,
        mode="max",
        sort="value",
    ).output(1)
    one_hot_mask = ov_opset.one_hot(
        bottom_ind,
        n,
        ov_opset.constant(1),
        ov_opset.constant(0),
        axis=-1,
    )
    mask = ov_opset.reduce_sum(
        one_hot_mask,
        ov_opset.constant([-2]),
        keep_dims=False,
    )
    ones = ov_opset.broadcast(
        ov_opset.constant(1),
        x_shape_tensor,
    )
    proxy = ov_opset.subtract(ones, mask)
    remaining_k = ov_opset.subtract(n, k_val)
    top_ind = ov_opset.topk(
        proxy,
        k=remaining_k,
        axis=-1,
        mode="max",
        sort="value",
    ).output(1)
    result = ov_opset.concat([bottom_ind, top_ind], axis=-1)
    inv_axes = [0] * rank
    for i, a in enumerate(axes):
        inv_axes[a] = i
    result = ov_opset.transpose(
        result,
        ov_opset.constant(inv_axes),
    ).output(0)
    return OpenVINOKerasTensor(result)
