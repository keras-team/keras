import builtins
import contextlib
import warnings

import ml_dtypes
import numpy as np
import openvino as ov
import openvino.opset15 as ov_opset
from openvino import Model
from openvino import Tensor
from openvino import Type
from openvino import compile_model

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import dtypes
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import slice_along_axis
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope

SUPPORTS_SPARSE_TENSORS = False
SUPPORTS_RAGGED_TENSORS = False
SUPPORTS_COMPLEX_DTYPES = False
IS_THREAD_SAFE = True

OPENVINO_DTYPES = {
    "float16": ov.Type.f16,
    "float32": ov.Type.f32,
    "float64": ov.Type.f64,
    "uint8": ov.Type.u8,
    "uint16": ov.Type.u16,
    "uint32": ov.Type.u32,
    "uint64": ov.Type.u64,
    "int8": ov.Type.i8,
    "int16": ov.Type.i16,
    "int32": ov.Type.i32,
    "int64": ov.Type.i64,
    "bfloat16": ov.Type.bf16,
    "bool": ov.Type.boolean,
    "float8_e4m3fn": ov.Type.f8e4m3,
    "float8_e5m2": ov.Type.f8e5m2,
    "string": ov.Type.string,
}

DTYPES_MAX = {
    ov.Type.bf16: 3.38953139e38,
    ov.Type.f16: np.finfo(np.float16).max,
    ov.Type.f32: np.finfo(np.float32).max,
    ov.Type.f64: np.finfo(np.float64).max,
    ov.Type.u8: np.iinfo(np.uint8).max,
    ov.Type.u16: np.iinfo(np.uint16).max,
    ov.Type.u32: np.iinfo(np.uint32).max,
    ov.Type.u64: np.iinfo(np.uint64).max,
    ov.Type.i8: np.iinfo(np.int8).max,
    ov.Type.i16: np.iinfo(np.int16).max,
    ov.Type.i32: np.iinfo(np.int32).max,
    ov.Type.i64: np.iinfo(np.int64).max,
    ov.Type.boolean: 1,
}

DTYPES_MIN = {
    ov.Type.bf16: -3.38953139e38,
    ov.Type.f16: np.finfo(np.float16).min,
    ov.Type.f32: np.finfo(np.float32).min,
    ov.Type.f64: np.finfo(np.float64).min,
    ov.Type.u8: np.iinfo(np.uint8).min,
    ov.Type.u16: np.iinfo(np.uint16).min,
    ov.Type.u32: np.iinfo(np.uint32).min,
    ov.Type.u64: np.iinfo(np.uint64).min,
    ov.Type.i8: np.iinfo(np.int8).min,
    ov.Type.i16: np.iinfo(np.int16).min,
    ov.Type.i32: np.iinfo(np.int32).min,
    ov.Type.i64: np.iinfo(np.int64).min,
    ov.Type.boolean: 0,
}


def align_operand_types(x1, x2, op_name, force_float=False):
    x1_type = x1.element_type
    x2_type = x2.element_type
    if x1_type.is_dynamic() or x2_type.is_dynamic():
        raise ValueError(
            f"'{op_name}' operation is not supported for dynamic operand type "
            "with openvino backend"
        )
    x1_type = ov_to_keras_type(x1_type)
    x2_type = ov_to_keras_type(x2_type)
    if force_float:
        result_type = dtypes.result_type(x1_type, x2_type, float)
    else:
        result_type = dtypes.result_type(x1_type, x2_type)
    result_type = OPENVINO_DTYPES[result_type]
    if x1_type != result_type:
        x1 = ov_opset.convert(x1, result_type).output(0)
    if x2_type != result_type:
        x2 = ov_opset.convert(x2, result_type).output(0)
    return x1, x2


# create ov.Output (symbolic OpenVINO tensor)
# for different input `x`
def get_ov_output(x, ov_type=None, context_dtype=None):
    if (
        isinstance(x, (float, int))
        and ov_type is None
        and context_dtype is not None
    ):
        ov_type = OPENVINO_DTYPES[dtypes.result_type(context_dtype, type(x))]
    if isinstance(x, float):
        if ov_type is None:
            ov_type = Type.f32
        if ov_type == Type.bf16:
            x = ov_opset.constant(x, Type.f32).output(0)
            x = ov_opset.convert(x, Type.bf16).output(0)
        else:
            x = ov_opset.constant(x, ov_type).output(0)
    elif isinstance(x, int):
        if ov_type is None:
            ov_type = Type.i32
        if ov_type == Type.bf16:
            x = ov_opset.constant(float(x), Type.f32).output(0)
            x = ov_opset.convert(x, Type.bf16).output(0)
        else:
            x = ov_opset.constant(x, ov_type).output(0)
    elif isinstance(x, np.ndarray):
        if x.dtype == "bfloat16":
            x = ov_opset.constant(x, OPENVINO_DTYPES["bfloat16"]).output(0)
        else:
            x = ov_opset.constant(x).output(0)
    elif isinstance(x, (list, tuple)):
        if isinstance(x, tuple):
            x = list(x)
        if ov_type is None:
            x = ov_opset.constant(x).output(0)
        else:
            x = ov_opset.constant(x, ov_type).output(0)
    elif np.isscalar(x):
        x = ov_opset.constant(x).output(0)
    elif isinstance(x, KerasVariable):
        if isinstance(x.value, OpenVINOKerasTensor):
            return x.value.output
        x = ov_opset.constant(x.value.data).output(0)
    elif isinstance(x, OpenVINOKerasTensor):
        x = x.output
    elif isinstance(x, ov.Output):
        return x
    elif isinstance(x, Tensor):
        x = ov_opset.constant(x.data).output(0)
    else:
        raise ValueError(
            "unsupported type of `x` to create ov.Output: {}".format(type(x))
        )
    return x


def shape_to_ov_output(shape):
    """Convert a shape tuple/list to an i32 ov.Output.

    Unlike get_ov_output, handles mixed shapes where some dims are
    OpenVINOKerasTensor scalars (from ops.shape() on dynamic tensors).
    None dims (from tensor.shape) are not supported — use ops.shape(x) instead.
    """
    if not isinstance(shape, (list, tuple)):
        raise ValueError(f"shape must be a list or tuple, got {type(shape)}")
    if any(e is None for e in shape):
        raise ValueError(
            "Shape contains None (dynamic) dimensions. Use ops.shape(x) "
            "instead of x.shape to get a runtime-resolved shape."
        )
    if not any(isinstance(e, (OpenVINOKerasTensor, ov.Output)) for e in shape):
        return ov_opset.constant(list(shape), Type.i32).output(0)
    parts = []
    for e in shape:
        if isinstance(e, OpenVINOKerasTensor):
            elem = e.output
        elif isinstance(e, ov.Output):
            elem = e
        else:
            elem = ov_opset.constant([e], Type.i32).output(0)
        if elem.get_element_type() != Type.i32:
            elem = ov_opset.convert(elem, Type.i32).output(0)
        # Scalar dims need to be reshaped to [1] for concat
        ps = elem.get_partial_shape()
        if ps.rank.is_static and ps.rank.get_length() == 0:
            elem = ov_opset.reshape(
                elem, ov_opset.constant([1], Type.i32).output(0), False
            ).output(0)
        parts.append(elem)
    return ov_opset.concat(parts, 0).output(0)


# wrapper for OpenVINO symbolic tensor ov.Output
# that provides interface similar to KerasTensor
# with dtype and shape members
class OpenVINOKerasTensor:
    def __init__(self, x, data=None):
        x_shape = x.get_partial_shape()
        if x_shape.rank.is_dynamic:
            x_keras_shape = None
        else:
            x_keras_shape = [
                None if dim.is_dynamic else dim.get_length()
                for dim in list(x_shape)
            ]
        x_type = x.get_element_type()
        x_keras_type = ov_to_keras_type(x_type)
        self.output = x
        if x_keras_shape is not None:
            self.shape = tuple(x_keras_shape)
        else:
            self.shape = None
        self.dtype = x_keras_type
        self.ndim = None
        self.data = data
        if x.get_partial_shape().rank.is_static:
            self.ndim = x.get_partial_shape().rank.get_length()

    def __add__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__add__"
        )
        return OpenVINOKerasTensor(ov_opset.add(first, other).output(0))

    def __radd__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__radd__"
        )
        return OpenVINOKerasTensor(ov_opset.add(first, other).output(0))

    def __sub__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__sub__"
        )
        if first.get_element_type() == Type.boolean:
            return OpenVINOKerasTensor(
                ov_opset.logical_xor(first, other).output(0)
            )
        return OpenVINOKerasTensor(ov_opset.subtract(first, other).output(0))

    def __rsub__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__rsub__"
        )
        return OpenVINOKerasTensor(ov_opset.subtract(other, first).output(0))

    def __mul__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__mul__"
        )
        if first.get_element_type() == Type.boolean:
            return OpenVINOKerasTensor(
                ov_opset.logical_and(first, other).output(0)
            )
        return OpenVINOKerasTensor(ov_opset.multiply(first, other).output(0))

    def __rmul__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__rmul__"
        )
        if first.get_element_type() == Type.boolean:
            return OpenVINOKerasTensor(
                ov_opset.logical_and(first, other).output(0)
            )
        return OpenVINOKerasTensor(ov_opset.multiply(first, other).output(0))

    def __truediv__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__truediv__", force_float=True
        )
        return OpenVINOKerasTensor(ov_opset.divide(first, other).output(0))

    def __rtruediv__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__rtruediv__", force_float=True
        )
        return OpenVINOKerasTensor(ov_opset.divide(other, first).output(0))

    def __floordiv__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__floordiv__"
        )
        div = ov_opset.divide(first, other).output(0)
        div_type = div.get_element_type()
        if div_type.is_integral():
            div = ov_opset.convert(div, Type.f32).output(0)
            div = ov_opset.floor(div).output(0)
            div = ov_opset.convert(div, div_type).output(0)
            return OpenVINOKerasTensor(div)
        return OpenVINOKerasTensor(ov_opset.floor(div).output(0))

    def __rfloordiv__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__rfloordiv__"
        )
        div = ov_opset.divide(other, first).output(0)
        div_type = div.get_element_type()
        if div_type.is_integral():
            div = ov_opset.convert(div, Type.f32).output(0)
            div = ov_opset.floor(div).output(0)
            div = ov_opset.convert(div, div_type).output(0)
            return OpenVINOKerasTensor(div)
        return OpenVINOKerasTensor(ov_opset.floor(div).output(0))

    def __neg__(self):
        first = self.output
        return OpenVINOKerasTensor(ov_opset.negative(first).output(0))

    def __abs__(self):
        first = self.output
        return OpenVINOKerasTensor(ov_opset.absolute(first).output(0))

    def __invert__(self):
        first = self.output
        return OpenVINOKerasTensor(ov_opset.logical_not(first).output(0))

    def __pow__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__pow__"
        )
        return OpenVINOKerasTensor(ov_opset.power(first, other).output(0))

    def __rpow__(self, other):
        other = get_ov_output(other, context_dtype=self.dtype)
        first = self.output
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__rpow__"
        )
        return OpenVINOKerasTensor(ov_opset.power(other, first).output(0))

    def __lt__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__lt__"
        )
        return OpenVINOKerasTensor(ov_opset.less(first, other).output(0))

    def __gt__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__gt__"
        )
        return OpenVINOKerasTensor(ov_opset.greater(first, other).output(0))

    def __le__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__le__"
        )
        return OpenVINOKerasTensor(ov_opset.less_equal(first, other).output(0))

    def __ge__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__ge__"
        )
        return OpenVINOKerasTensor(
            ov_opset.greater_equal(first, other).output(0)
        )

    def __eq__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__eq__"
        )
        return OpenVINOKerasTensor(ov_opset.equal(first, other).output(0))

    def __ne__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__ne__"
        )
        return OpenVINOKerasTensor(ov_opset.not_equal(first, other).output(0))

    def __getitem__(self, indices):
        data = self.output
        rank = len(data.get_partial_shape())
        axes, gather_indices_nodes = [], []
        slice_axes, slice_starts, slice_ends, slice_steps = [], [], [], []
        unsqueeze_axes = []

        if not isinstance(indices, tuple):
            indices = (indices,)

        if any(i is Ellipsis for i in indices):
            ellipsis_pos = indices.index(Ellipsis)
            num_specified = sum(
                i is not Ellipsis and i is not None for i in indices
            )
            num_missing = rank - num_specified
            indices = (
                indices[:ellipsis_pos]
                + (builtins.slice(None),) * num_missing
                + indices[ellipsis_pos + 1 :]
            )

        def count_unsqueeze_before(dim):
            return sum(1 for i in range(dim) if indices[i] is None)

        partial_shape = ov_opset.shape_of(data, Type.i32)
        zero_const = ov_opset.constant(0, Type.i32)

        for dim, index in enumerate(indices):
            if isinstance(index, bool):
                raise ValueError(
                    "OpenVINO backend does not support boolean indexing"
                )
            elif isinstance(index, (int, np.integer, np.ndarray)):
                if isinstance(index, (np.ndarray, np.integer)):
                    if isinstance(index, np.ndarray) and len(index.shape) != 0:
                        raise ValueError(
                            "OpenVINO backend does not support"
                            "multi-dimensional indexing"
                        )
                    index = int(index)
                actual_dim = dim - count_unsqueeze_before(dim)
                if not (0 <= actual_dim < rank):
                    raise IndexError(
                        f"Index {index} is out of bounds for "
                        f"axis {dim} with rank {rank}"
                    )
                length = ov_opset.gather(
                    partial_shape,
                    ov_opset.constant([actual_dim], Type.i32),
                    zero_const,
                )
                if index >= 0:
                    idx_value = ov_opset.constant([index], Type.i32)
                else:
                    idx_value = ov_opset.add(
                        ov_opset.constant([index], Type.i32), length
                    )
                axes.append(dim)
                gather_indices_nodes.append(idx_value.output(0))
            elif isinstance(index, builtins.slice):
                if (
                    index.start is None
                    and index.stop is None
                    and index.step is None
                ):
                    continue
                if (
                    index.step is not None
                    and not isinstance(index.step, OpenVINOKerasTensor)
                    and index.step < 0
                ):
                    raise ValueError("OpenVINO doesn't support negative steps")
                slice_axes.append(dim)
                slice_starts.append(0 if index.start is None else index.start)
                slice_ends.append(
                    2**31 - 1 if index.stop is None else index.stop
                )
                slice_steps.append(1 if index.step is None else index.step)
            elif index is None:
                unsqueeze_axes.append(dim)
            elif isinstance(index, OpenVINOKerasTensor):
                index = get_ov_output(index)
                index_type = index.get_element_type()
                index_shape = index.get_partial_shape()
                if index_type == Type.boolean or not index_type.is_integral():
                    raise ValueError(
                        "OpenVINO backend does not "
                        f"support {index_type} indexing"
                    )
                axes.append(dim)
                if len(index_shape) > 1:
                    raise ValueError(
                        "OpenVINO backend does not "
                        "support multi-dimensional indexing"
                    )
                if len(index_shape) == 0:
                    index = ov_opset.unsqueeze(index, zero_const).output(0)
                if index_type != Type.i32:
                    index = ov_opset.convert(index, Type.i32).output(0)
                shape_tensor = ov_opset.shape_of(data, Type.i32)
                axis_i32 = ov_opset.constant([dim], dtype=Type.i32)
                dim_size = ov_opset.gather(shape_tensor, axis_i32, zero_const)
                is_negative = ov_opset.less(index, zero_const)
                adjusted_index = ov_opset.add(index, dim_size)
                index = ov_opset.select(
                    is_negative, adjusted_index, index
                ).output(0)
                gather_indices_nodes.append(index)
            else:
                raise ValueError(
                    f"Unsupported index type {type(index)} "
                    "in OpenVINOKerasTensor.__getitem__"
                )

        if slice_axes:

            def _to_slice_bound(values, dtype=Type.i32):
                nodes = []
                for v in values:
                    if isinstance(v, OpenVINOKerasTensor):
                        node = v.output
                    else:
                        node = ov_opset.constant([v], dtype).output(0)
                    if node.get_element_type() != dtype:
                        node = ov_opset.convert(node, dtype).output(0)
                    ps = node.get_partial_shape()
                    if len(ps) == 0:
                        node = ov_opset.unsqueeze(
                            node, ov_opset.constant(0, Type.i32)
                        ).output(0)
                    nodes.append(node)
                if len(nodes) == 1:
                    return nodes[0]
                return ov_opset.concat(nodes, axis=0).output(0)

            step = _to_slice_bound(slice_steps)
            start = _to_slice_bound(slice_starts)
            stop = _to_slice_bound(slice_ends)
            adjusted_slice_axes = [
                ax - sum(1 for unsq in unsqueeze_axes if unsq <= ax)
                for ax in slice_axes
            ]
            axes_const = ov_opset.constant(
                adjusted_slice_axes, Type.i32
            ).output(0)
            data = ov_opset.slice(data, start, stop, step, axes_const).output(0)

        if axes:
            gather_indices_const = (
                gather_indices_nodes[0]
                if len(gather_indices_nodes) == 1
                else ov_opset.concat(gather_indices_nodes, axis=0).output(0)
            )
            adjusted_axes = [
                ax - sum(1 for unsq in unsqueeze_axes if unsq <= ax)
                for ax in axes
            ]
            if len(axes) == 1:
                data = ov_opset.gather(
                    data, gather_indices_const, adjusted_axes[0]
                ).output(0)
                data = ov_opset.squeeze(data, adjusted_axes[0]).output(0)
            else:
                rank = len(data.get_partial_shape())
                remaining_axes = [
                    i for i in range(rank) if i not in adjusted_axes
                ]
                perm = ov_opset.constant(
                    adjusted_axes + remaining_axes, Type.i32
                )
                data = ov_opset.transpose(data, perm).output(0)
                data = ov_opset.gather_nd(data, gather_indices_const).output(0)

        if unsqueeze_axes:
            adjusted_unsqueeze = []
            for ax in unsqueeze_axes:
                ax -= sum(1 for s in axes if s < ax)
                ax -= sum(1 for s in slice_axes if s < ax)
                adjusted_unsqueeze.append(ax)
            unsqueeze_const = ov_opset.constant(
                adjusted_unsqueeze, Type.i32
            ).output(0)
            data = ov_opset.unsqueeze(data, unsqueeze_const).output(0)

        return OpenVINOKerasTensor(data)

    def __len__(self):
        ov_output = self.output
        ov_shape = ov_output.get_partial_shape()
        if not (ov_shape.rank.is_static and ov_shape.rank.get_length() > 0):
            raise ValueError(
                "Rank must be static and greater than zero to compute `len()`. "
                f"rank={ov_shape.rank}"
            )
        if not ov_shape[0].is_static:
            raise ValueError(
                "The first dimension must be static to compute `len()`. "
                f"shape={ov_shape}"
            )
        return ov_shape[0].get_length()

    def __iter__(self):
        if self.shape is None or len(self.shape) == 0:
            raise TypeError("iteration over a 0-d tensor")
        for i in range(self.shape[0]):
            yield self[i]

    def __bool__(self):
        return bool(self.numpy())

    def __mod__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__mod__"
        )
        return OpenVINOKerasTensor(ov_opset.mod(first, other).output(0))

    def __array__(self, dtype=None):
        try:
            tensor = cast(self, dtype=dtype) if dtype is not None else self
            return convert_to_numpy(tensor)
        except Exception as e:
            raise RuntimeError(
                "An OpenVINOKerasTensor is symbolic: it's a placeholder "
                "for a shape and a dtype.\n"
                "It doesn't have any actual numerical value.\n"
                "You cannot convert it to a NumPy array."
            ) from e

    def numpy(self):
        return self.__array__()

    def __rmod__(self, other):
        other = get_ov_output(other, context_dtype=self.dtype)
        first = self.output
        other, first = align_operand_types(
            other, first, "OpenVINOKerasTensor::__rmod__"
        )
        return OpenVINOKerasTensor(ov_opset.mod(other, first).output(0))

    def __matmul__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__matmul__"
        )
        return OpenVINOKerasTensor(
            ov_opset.matmul(first, other, False, False).output(0)
        )

    def __rmatmul__(self, other):
        other = get_ov_output(other, context_dtype=self.dtype)
        first = self.output
        other, first = align_operand_types(
            other, first, "OpenVINOKerasTensor::__rmatmul__"
        )
        return OpenVINOKerasTensor(
            ov_opset.matmul(other, first, False, False).output(0)
        )

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __and__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__and__"
        )
        return OpenVINOKerasTensor(ov_opset.logical_and(first, other).output(0))

    def __rand__(self, other):
        other = get_ov_output(other, context_dtype=self.dtype)
        first = self.output
        other, first = align_operand_types(
            other, first, "OpenVINOKerasTensor::__rand__"
        )
        return OpenVINOKerasTensor(ov_opset.logical_and(other, first).output(0))

    def __or__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__or__"
        )
        return OpenVINOKerasTensor(ov_opset.logical_or(first, other).output(0))

    def __ror__(self, other):
        other = get_ov_output(other, context_dtype=self.dtype)
        first = self.output
        other, first = align_operand_types(
            other, first, "OpenVINOKerasTensor::__ror__"
        )
        return OpenVINOKerasTensor(ov_opset.logical_or(other, first).output(0))

    def __xor__(self, other):
        first = self.output
        other = get_ov_output(other, context_dtype=self.dtype)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__xor__"
        )
        return OpenVINOKerasTensor(ov_opset.logical_xor(first, other).output(0))

    def __rxor__(self, other):
        other = get_ov_output(other, context_dtype=self.dtype)
        first = self.output
        other, first = align_operand_types(
            other, first, "OpenVINOKerasTensor::__rxor__"
        )
        return OpenVINOKerasTensor(ov_opset.logical_xor(other, first).output(0))

    def __int__(self):
        arr = convert_to_numpy(self)
        if arr.ndim > 0:
            raise TypeError(
                "Only scalar arrays can be converted to Python scalars. "
                f"Got: shape={arr.shape}"
            )
        return int(arr)

    def __float__(self):
        arr = convert_to_numpy(self)
        if arr.ndim > 0:
            raise TypeError(
                "Only scalar arrays can be converted to Python scalars. "
                f"Got: shape={arr.shape}"
            )
        return float(arr)

    def __repr__(self):
        return f"<OpenVINOKerasTensor shape={self.shape}, dtype={self.dtype}>"

    def __round__(self, ndigits=None):
        first = self.output
        decimals = ndigits or 0
        if decimals == 0:
            result = ov_opset.round(first, "half_to_even")
        else:
            factor = ov_opset.constant(10.0**decimals, first.get_element_type())
            scaled = ov_opset.multiply(first, factor)
            rounded = ov_opset.round(scaled, "half_to_even")
            result = ov_opset.divide(rounded, factor)
        return OpenVINOKerasTensor(result.output(0))

    def reshape(self, new_shape):
        first = self.output
        shape_const = get_ov_output(new_shape)
        return OpenVINOKerasTensor(
            ov_opset.reshape(first, shape_const, False).output(0)
        )

    def squeeze(self, axis=None):
        first = self.output
        if axis is not None:
            axes = get_ov_output([axis] if isinstance(axis, int) else axis)
        else:
            axes = get_ov_output(
                [i for i, d in enumerate(self.shape) if d == 1]
            )
        return OpenVINOKerasTensor(ov_opset.squeeze(first, axes).output(0))


def ov_to_keras_type(ov_type):
    for _keras_type, _ov_type in OPENVINO_DTYPES.items():
        if ov_type == _ov_type:
            return _keras_type
    raise ValueError(
        f"Requested OpenVINO type has no keras analogue '{ov_type.to_string()}'"
    )


@contextlib.contextmanager
def device_scope(device_name):
    yield


def get_device():
    return "CPU"


class Variable(KerasVariable):
    def _initialize(self, value):
        if isinstance(value, OpenVINOKerasTensor):
            self._value = value
        elif isinstance(value, Tensor):
            value_const = ov_opset.constant(
                value.data, dtype=OPENVINO_DTYPES[self._dtype]
            )
            self._value = OpenVINOKerasTensor(value_const.output(0))
        else:
            value_const = ov_opset.constant(
                value, dtype=OPENVINO_DTYPES[self._dtype]
            )
            self._value = OpenVINOKerasTensor(value_const.output(0))

    def _direct_assign(self, value):
        self._value = value

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    def __array__(self):
        return convert_to_numpy(self)

    def __getitem__(self, idx):
        arr = convert_to_numpy(self)
        return arr.__getitem__(idx)

    def __int__(self):
        arr = convert_to_numpy(self)
        if arr.ndim > 0:
            raise TypeError(
                "Only scalar arrays can be converted to Python scalars. "
                f"Got: shape={arr.shape}"
            )
        return int(arr)

    def __float__(self):
        arr = convert_to_numpy(self)
        if arr.ndim > 0:
            raise TypeError(
                "Only scalar arrays can be converted to Python scalars. "
                f"Got: shape={arr.shape}"
            )
        return float(arr)


def _is_scalar(elem):
    return not isinstance(elem, (list, tuple, set, dict))


def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    if sparse:
        raise ValueError("`sparse=True` is not supported with openvino backend")
    if ragged:
        raise ValueError("`ragged=True` is not supported with openvino backend")
    if dtype is not None:
        dtype = standardize_dtype(dtype)
    if isinstance(x, OpenVINOKerasTensor):
        if dtype and dtype != standardize_dtype(x.dtype):
            x = cast(x, dtype)
        return x
    elif isinstance(x, np.ndarray):
        if dtype is not None:
            ov_type = OPENVINO_DTYPES[dtype]
        else:
            ov_type = OPENVINO_DTYPES[standardize_dtype(x.dtype)]
        return OpenVINOKerasTensor(ov_opset.constant(x, ov_type).output(0))
    elif isinstance(x, (list, tuple)):
        if dtype is None:
            dtype = result_type(
                *[
                    getattr(item, "dtype", type(item))
                    for item in tree.flatten(x)
                ]
            )
        x = np.array(x, dtype=dtype)
        ov_type = OPENVINO_DTYPES[dtype]
        return OpenVINOKerasTensor(ov_opset.constant(x, ov_type).output(0), x)
    elif isinstance(x, (float, int, bool)):
        if dtype is None:
            dtype = standardize_dtype(type(x))
        ov_type = OPENVINO_DTYPES[dtype]
        return OpenVINOKerasTensor(ov_opset.constant(x, ov_type).output(0), x)
    elif isinstance(x, ov.Output):
        return OpenVINOKerasTensor(x)
    if isinstance(x, Variable):
        x = x.value
        if dtype and dtype != x.dtype:
            x = cast(x, dtype)
        return x
    original_type = type(x)
    try:
        if dtype is None:
            dtype = getattr(x, "dtype", original_type)
            ov_type = OPENVINO_DTYPES[standardize_dtype(dtype)]
        else:
            ov_type = OPENVINO_DTYPES[dtype]
        x = np.array(x)
        return OpenVINOKerasTensor(ov_opset.constant(x, ov_type).output(0))
    except Exception as e:
        raise TypeError(
            f"Cannot convert object of type {original_type} "
            f"to OpenVINOKerasTensor: {e}"
        )


def convert_to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (int, float)):
        return np.array(x)
    elif isinstance(x, (list, tuple)):
        x_new = []
        for elem in x:
            x_new.append(convert_to_numpy(elem))
        return np.array(x_new)
    elif np.isscalar(x):
        return x
    elif isinstance(x, ov.Tensor):
        return x.data
    elif x is None:
        return x
    elif isinstance(x, KerasVariable):
        if isinstance(x.value, OpenVINOKerasTensor):
            x = x.value
        else:
            return x.value.data
    if not isinstance(x, OpenVINOKerasTensor):
        raise ValueError(f"unsupported type {type(x)} for `convert_to_numpy`.")
    # if the tensor is backed by a Constant OV node, extract
    # its data array directly without compiling a model.
    try:
        node = x.output.get_node()
        if node.get_type_name() == "Constant":
            data = node.data
            # OpenVINO returns bf16 constant bytes as float16 (same width,
            # but wrong dtype) because numpy has no native bfloat16 type.
            # Re-interpret the raw bytes as ml_dtypes.bfloat16.
            if node.output(0).get_element_type() == Type.bf16:
                data = data.view(ml_dtypes.bfloat16)
            return np.array(data)
    except Exception:
        # fall back to the slow path.
        pass
    try:
        ov_result = x.output
        casted_from_bool = False
        if ov_result.get_element_type() == Type.boolean:
            ov_result = ov_opset.convert(ov_result, Type.i32).output(0)
            casted_from_bool = True
        ov_model = Model(results=[ov_result], parameters=[])
        ov_compiled_model = compile_model(
            ov_model,
            get_device(),
            config={"INFERENCE_PRECISION_HINT": "f32"},
        )
        result = ov_compiled_model({})[0]
        if casted_from_bool:
            result = result.astype(bool)
    except Exception as inner_exception:
        raise RuntimeError(
            "`convert_to_numpy` failed to convert the tensor."
        ) from inner_exception
    data = np.array(result)
    # Same byte-reinterpretation issue applies to inference results.
    if x.dtype == "bfloat16" and data.dtype != ml_dtypes.bfloat16:
        data = data.view(ml_dtypes.bfloat16)
    return data


def is_tensor(x):
    if isinstance(x, OpenVINOKerasTensor):
        return True
    if isinstance(x, ov.Tensor):
        return True
    return False


def shape(x):
    if not isinstance(x, OpenVINOKerasTensor):
        return tuple(x.shape)

    static_shape = x.shape
    if static_shape is None or None not in static_shape:
        return static_shape

    # For dynamic dims, return OpenVINOKerasTensor scalars obtained at runtime
    shape_node = ov_opset.shape_of(x.output, Type.i32).output(0)
    axis = ov_opset.constant(0, Type.i32).output(0)
    result = []
    for i, dim in enumerate(static_shape):
        if dim is None:
            idx = ov_opset.constant(i, Type.i32).output(0)
            dim_scalar = ov_opset.gather(shape_node, idx, axis).output(0)
            result.append(OpenVINOKerasTensor(dim_scalar))
        else:
            result.append(dim)
    return tuple(result)


def cast(x, dtype):
    dtype = standardize_dtype(dtype)
    ov_type = OPENVINO_DTYPES[dtype]
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.convert(x, ov_type).output(0))


def cond(pred, true_fn, false_fn):
    class _TrackingScope(StatelessScope):
        """StatelessScope that retains variable object references."""

        def __init__(self):
            super().__init__()
            self._var_objects = {}

        def add_update(self, update):
            variable, value = update
            super().add_update(update)
            self._var_objects[id(variable)] = variable

    # Run both branches in isolated scopes so variable assignments are
    # captured but not applied eagerly (same semantics as torch.cond).
    true_scope = _TrackingScope()
    with true_scope:
        true_val = true_fn()

    false_scope = _TrackingScope()
    with false_scope:
        false_val = false_fn()

    if isinstance(pred, bool):
        pred_ov = ov_opset.constant(pred, Type.boolean).output(0)
    else:
        pred_ov = get_ov_output(pred)
        if pred_ov.get_element_type() != Type.boolean:
            pred_ov = ov_opset.convert(pred_ov, Type.boolean).output(0)

    def _select(t, f):
        t_ov, f_ov = align_operand_types(
            get_ov_output(t), get_ov_output(f), "cond"
        )
        return OpenVINOKerasTensor(
            ov_opset.select(pred_ov, t_ov, f_ov).output(0)
        )

    # Apply selected variable updates: for each variable touched by either
    # branch, select between the true-branch value and the false-branch value
    # (defaulting to the pre-cond stored value for the branch that didn't
    # update it).
    all_var_ids = set(true_scope.state_mapping) | set(false_scope.state_mapping)
    for var_id in all_var_ids:
        if var_id in true_scope._var_objects:
            var = true_scope._var_objects[var_id]
        else:
            var = false_scope._var_objects[var_id]
        true_new = true_scope.state_mapping.get(var_id, var._value)
        false_new = false_scope.state_mapping.get(var_id, var._value)
        var._direct_assign(_select(true_new, false_new))

    if true_val is None:
        return None

    if isinstance(true_val, (list, tuple)):
        return type(true_val)(
            _select(t, f) for t, f in zip(true_val, false_val)
        )
    return _select(true_val, false_val)


def vectorized_map(function, elements):
    return map(function, elements)


# Shape / dtype inference util
def compute_output_spec(fn, *args, **kwargs):
    with StatelessScope():

        def convert_keras_tensor_to_openvino(x):
            if isinstance(x, KerasTensor):
                x_shape = list(x.shape)
                x_shape = [-1 if dim is None else dim for dim in x_shape]
                x_type = OPENVINO_DTYPES[x.dtype]
                param = ov_opset.parameter(shape=x_shape, dtype=x_type)
                return OpenVINOKerasTensor(param.output(0))
            return x

        args_1, kwargs_1 = tree.map_structure(
            lambda x: convert_keras_tensor_to_openvino(x),
            (args, kwargs),
        )
        outputs_1 = fn(*args_1, **kwargs_1)

        outputs = outputs_1

        def convert_openvino_to_keras_tensor(x):
            if is_tensor(x):
                x_type = x.dtype
                x_shape = x.shape
                return KerasTensor(x_shape, x_type)
            elif isinstance(x, OpenVINOKerasTensor):
                x_type = x.dtype
                x_shape = x.shape
                return KerasTensor(x_shape, x_type)
            return x

        output_spec = tree.map_structure(
            convert_openvino_to_keras_tensor, outputs
        )
    return output_spec


def map(f, xs):
    def g(_, x):
        return (), f(x)

    _, ys = scan(g, (), xs)
    return ys


def scan(f, init, xs=None, length=None, reverse=False, unroll=1):
    # Ref: jax.lax.scan
    if not callable(f):
        raise TypeError(f"`f` should be a callable. Received: f={f}")
    if not isinstance(unroll, bool):
        if not isinstance(unroll, int) or unroll < 1:
            raise ValueError(
                "`unroll` must be an positive integer or boolean. "
                f"Received: unroll={unroll}"
            )
    if xs is None and length is None:
        raise ValueError("Got no `xs` to scan over and `length` not provided.")

    input_is_sequence = tree.is_nested(xs)
    output_is_sequence = tree.is_nested(init)

    def pack_input(x):
        return tree.pack_sequence_as(xs, x) if input_is_sequence else x[0]

    def pack_output(x):
        return tree.pack_sequence_as(init, x) if output_is_sequence else x[0]

    if xs is None:
        xs_flat = []
        n = int(length)
    else:
        xs_flat = tree.flatten(xs)
        xs_flat = [convert_to_tensor(elem) for elem in xs_flat]
        n = (
            int(length)
            if length is not None
            else (shape(xs_flat[0])[0] if xs_flat else 0)
        )

    init_flat = tree.flatten(init)
    init_flat = [convert_to_tensor(i) for i in init_flat]
    init = pack_output(init_flat)

    dummy_y = []
    for i in init_flat:
        i_ov = get_ov_output(i)
        zero = ov_opset.constant(0, i_ov.get_element_type()).output(0)
        shape_node = ov_opset.shape_of(i_ov, Type.i32).output(0)
        dummy_y.append(
            OpenVINOKerasTensor(ov_opset.broadcast(zero, shape_node).output(0))
        )

    carry = init
    ys = []
    maybe_reversed = reversed if reverse else lambda x: x
    for i in maybe_reversed(range(n)):
        xs_slice = [x[i] for x in xs_flat]
        packed_xs = pack_input(xs_slice) if len(xs_slice) > 0 else None
        carry, y = f(carry, packed_xs)
        ys.append(y if y is not None else dummy_y)

    def _stack(tensors):
        elems = [get_ov_output(t) for t in tensors]
        const_axis = ov_opset.constant(0, Type.i32).output(0)
        elems = [ov_opset.unsqueeze(e, const_axis).output(0) for e in elems]
        return OpenVINOKerasTensor(ov_opset.concat(elems, 0).output(0))

    stacked_y = tree.map_structure(
        lambda *y: _stack(list(y)), *maybe_reversed(ys)
    )
    return carry, stacked_y


def associative_scan(f, elems, reverse=False, axis=0):
    # Ref: jax.lax.associative_scan
    if not callable(f):
        raise TypeError(f"`f` should be a callable. Received: f={f}")
    elems_flat = tree.flatten(elems)
    elems_flat = [convert_to_tensor(elem) for elem in elems_flat]

    def _flip(x, axis):
        x_ov = get_ov_output(x)
        ndim = len(x_ov.get_partial_shape())
        begin = [0] * ndim
        end = [0] * ndim
        strides = [1] * ndim
        strides[axis] = -1
        mask = [1] * ndim
        result = ov_opset.strided_slice(
            data=x_ov,
            begin=begin,
            end=end,
            strides=strides,
            begin_mask=mask,
            end_mask=mask,
        ).output(0)
        return OpenVINOKerasTensor(result)

    def _concat(tensors, axis):
        elems = [get_ov_output(t) for t in tensors]
        keras_types = [ov_to_keras_type(e.get_element_type()) for e in elems]
        if keras_types:
            target = OPENVINO_DTYPES[result_type(*keras_types)]
            elems = [
                ov_opset.convert(e, target).output(0)
                if e.get_element_type() != target
                else e
                for e in elems
            ]
        return OpenVINOKerasTensor(ov_opset.concat(elems, axis).output(0))

    def _unsqueeze(x, axis):
        x_ov = get_ov_output(x)
        const_axis = ov_opset.constant(axis, Type.i32).output(0)
        return OpenVINOKerasTensor(
            ov_opset.unsqueeze(x_ov, const_axis).output(0)
        )

    if reverse:
        elems_flat = [_flip(elem, axis) for elem in elems_flat]

    def _combine(a_flat, b_flat):
        a = tree.pack_sequence_as(elems, a_flat)
        b = tree.pack_sequence_as(elems, b_flat)
        c = f(a, b)
        return tree.flatten(c)

    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError(
            "Array inputs to associative_scan must have the same "
            "first dimension. (saw: {})".format(
                [elem.shape for elem in elems_flat]
            )
        )

    def _interleave(a, b, axis):
        n_a = a.shape[axis]
        n_b = b.shape[axis]

        a_common = slice_along_axis(a, 0, n_b, axis=axis)
        a_exp = _unsqueeze(a_common, axis + 1)
        b_exp = _unsqueeze(b, axis + 1)
        interleaved = _concat([a_exp, b_exp], axis + 1)

        interleaved_ov = get_ov_output(interleaved)
        orig_shape = ov_opset.shape_of(interleaved_ov, Type.i32).output(0)
        ndim = len(interleaved_ov.get_partial_shape())
        pre = ov_opset.slice(
            orig_shape,
            ov_opset.constant([0], Type.i32),
            ov_opset.constant([axis], Type.i32),
            ov_opset.constant([1], Type.i32),
        ).output(0)
        merged_dim = ov_opset.constant([n_b * 2], Type.i32).output(0)
        post = ov_opset.slice(
            orig_shape,
            ov_opset.constant([axis + 2], Type.i32),
            ov_opset.constant([ndim], Type.i32),
            ov_opset.constant([1], Type.i32),
        ).output(0)
        target_shape = ov_opset.concat([pre, merged_dim, post], 0).output(0)
        interleaved = OpenVINOKerasTensor(
            ov_opset.reshape(interleaved_ov, target_shape, False).output(0)
        )

        if n_a > n_b:
            last = slice_along_axis(a, n_b, n_b + 1, axis=axis)
            interleaved = _concat([interleaved, last], axis)

        return interleaved

    def _scan(elems):
        num_elems = elems[0].shape[axis]
        if num_elems < 2:
            return elems

        reduced_elems = _combine(
            [slice_along_axis(e, 0, -1, step=2, axis=axis) for e in elems],
            [slice_along_axis(e, 1, None, step=2, axis=axis) for e in elems],
        )
        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = _combine(
                [slice_along_axis(e, 0, -1, axis=axis) for e in odd_elems],
                [
                    slice_along_axis(e, 2, None, step=2, axis=axis)
                    for e in elems
                ],
            )
        else:
            even_elems = _combine(
                odd_elems,
                [
                    slice_along_axis(e, 2, None, step=2, axis=axis)
                    for e in elems
                ],
            )
        even_elems = [
            _concat(
                [slice_along_axis(elem, 0, 1, axis=axis), result],
                axis,
            )
            for elem, result in zip(elems, even_elems)
        ]
        return [_interleave(e, o, axis) for e, o in zip(even_elems, odd_elems)]

    scanned_elems = _scan(elems_flat)
    if reverse:
        scanned_elems = [_flip(elem, axis) for elem in scanned_elems]
    return tree.pack_sequence_as(elems, scanned_elems)


def scatter(indices, values, shape):
    indices = get_ov_output(indices)
    values = get_ov_output(values)

    # Create a zeros tensor of the target shape.
    shape = get_ov_output(shape)
    zero_const = ov_opset.constant(0, values.get_element_type())
    zeros = ov_opset.broadcast(zero_const, shape).output(0)

    return scatter_update(zeros, indices, values, "add")


def scatter_update(inputs, indices, updates, reduction=None):
    inputs = get_ov_output(inputs)
    indices = get_ov_output(indices)
    updates = get_ov_output(updates)

    inputs, updates = align_operand_types(inputs, updates, "scatter_update")

    # Map Keras reduction to OpenVINO ScatterNDUpdate reduction.
    # OpenVINO Opset 15 supports: "none", "sum", "sub", "prod", "min", "max".
    if reduction is None:
        ov_reduction = "none"
    elif reduction == "add":
        ov_reduction = "sum"
    elif reduction == "mul":
        ov_reduction = "prod"
    elif reduction in ("max", "min"):
        ov_reduction = reduction
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

    result = ov_opset.scatter_nd_update(
        inputs, indices, updates, reduction=ov_reduction
    ).output(0)
    return OpenVINOKerasTensor(result)


def slice(inputs, start_indices, shape):
    inputs = get_ov_output(inputs)
    if isinstance(start_indices, (list, np.ndarray)):
        start_indices = tuple(start_indices)
    if isinstance(shape, (list, np.ndarray)):
        shape = tuple(shape)
    if not isinstance(start_indices, tuple):
        raise ValueError(
            "`slice` operation requires tuple for `start_indices with the "
            f"openvino backend. Received: start_indices={start_indices}"
        )
    if not isinstance(shape, tuple):
        raise ValueError(
            "`slice` operation requires tuple for `shape` with the "
            f"openvino backend. Received: shape={shape}"
        )

    axes = []
    start = []
    stop = []

    def prepare_slice_index(val):
        val_type = val.get_element_type()
        if not val_type.is_integral():
            raise ValueError(
                "`slice` is not supported by OpenVINO backend "
                "for `start_indices` or `shape` with non-integer types"
            )
        if val_type != Type.i32:
            val = ov_opset.convert(val, Type.i32).output(0)
        if len(val.get_partial_shape()) == 0:
            val = ov_opset.unsqueeze(
                val, ov_opset.constant(0, Type.i32)
            ).output(0)
        return val

    for idx, length in enumerate(shape):
        if length is not None and length >= 0:
            axes.append(idx)
            start_val = prepare_slice_index(get_ov_output(start_indices[idx]))
            stop_val = prepare_slice_index(
                get_ov_output(start_indices[idx] + length)
            )
            start.append(start_val)
            stop.append(stop_val)

    if len(axes) == 0:
        return inputs

    step = [1] * len(start)
    step = ov_opset.constant(step, Type.i32).output(0)
    start = ov_opset.concat(start, axis=0).output(0)
    stop = ov_opset.concat(stop, axis=0).output(0)
    axes = ov_opset.constant(axes, Type.i32).output(0)
    result = ov_opset.slice(inputs, start, stop, step, axes).output(0)

    # Apply reshape to ensure output matches expected shape
    # Convert None (dynamic) dimensions to -1 for OpenVINO compatibility
    if all(dim is None or (isinstance(dim, int) and dim >= 0) for dim in shape):
        reshape_pattern = [(-1 if dim is None else dim) for dim in shape]
        target_shape = ov_opset.constant(reshape_pattern, Type.i32).output(0)
        result = ov_opset.reshape(result, target_shape, False).output(0)

    return OpenVINOKerasTensor(result)


def slice_update(inputs, start_indices, updates):
    inputs = get_ov_output(inputs)
    updates_tensor = get_ov_output(updates)

    if isinstance(start_indices, (list, np.ndarray)):
        start_indices = tuple(start_indices)
    if not isinstance(start_indices, tuple):
        raise ValueError(
            "`slice_update` is not supported by openvino backend"
            " for `start_indices` of type {}".format(type(start_indices))
        )

    zero_scalar = ov_opset.constant(0, Type.i32)
    one_scalar = ov_opset.constant(1, Type.i32)
    zero_tensor = ov_opset.constant([0], Type.i32)
    one_tensor = ov_opset.constant([1], Type.i32)

    processed_start_indices = []
    for idx in start_indices:
        val = get_ov_output(idx)
        if not val.get_element_type().is_integral():
            raise ValueError("`slice_update` requires integral start_indices")
        if val.get_element_type() != Type.i32:
            val = ov_opset.convert(val, Type.i32).output(0)
        if val.get_partial_shape().rank.get_length() == 0:
            val = ov_opset.unsqueeze(val, zero_scalar).output(0)
        processed_start_indices.append(val)

    updates_shape = ov_opset.shape_of(updates_tensor, Type.i32).output(0)
    rank = updates_tensor.get_partial_shape().rank.get_length()
    if rank == 0:
        # Handle scalar update
        start_tensor = ov_opset.concat(processed_start_indices, axis=0).output(
            0
        )
        # For scatter_nd_update,
        # indices should be of shape [num_updates, rank_of_inputs]
        # and updates should be of shape [num_updates]. Here num_updates is 1.
        absolute_indices = ov_opset.unsqueeze(start_tensor, zero_scalar).output(
            0
        )
        updates_flat = ov_opset.unsqueeze(updates_tensor, zero_scalar).output(0)
        result = ov_opset.scatter_nd_update(
            inputs, absolute_indices, updates_flat
        ).output(0)
        return OpenVINOKerasTensor(result)

    # Compute the total number of elements in the updates tensor.
    # Example:
    # if updates.shape = [2, 3], total_elements = 6.
    total_elements = ov_opset.reduce_prod(
        updates_shape, zero_tensor, keep_dims=False
    ).output(0)

    # Generate a flat range [0, 1, ..., total_elements-1].
    # This will be used to enumerate all positions in the updates tensor.
    flat_indices = ov_opset.range(
        zero_scalar, total_elements, one_scalar, output_type=Type.i32
    ).output(0)

    dim_sizes = []
    strides = []

    # For each dimension, compute its size and the stride.
    # (number of elements to skip to move to the next index in this dimension).
    # Example:
    # for shape [2, 3], strides = [3, 1].
    for dim in range(rank):
        dim_size = ov_opset.gather(
            updates_shape, ov_opset.constant([dim], Type.i32), zero_scalar
        ).output(0)
        dim_size_scalar = ov_opset.squeeze(dim_size, zero_tensor).output(0)
        dim_sizes.append(dim_size_scalar)

        # Strides to convert a flat index into a multi-dimensional index.
        # This allows us to map each element in the flattened updates tensor
        # to its correct N-dimensional position, so we can compute the absolute
        # index in the input tensor for the scatter update.
        # Stride for a dimension is the product of all dimensions after it.
        # For the last dimension, stride is 1.
        # Example:
        # For a 3D tensor with shape [2, 3, 4]:
        #   - stride for dim=0 (first axis) is 3*4=12
        #     (to move to the next "block" along axis 0)
        #   - stride for dim=1 is 4 (to move to the next row along axis 1)
        #   - stride for dim=2 is 1 (to move to the next element along axis 2)
        # This is equivalent to how numpy flattens multi-dimensional arrays.
        if dim < rank - 1:
            remaining_dims = ov_opset.slice(
                updates_shape,
                ov_opset.constant([dim + 1], Type.i32),
                ov_opset.constant([rank], Type.i32),
                one_tensor,
                zero_tensor,
            ).output(0)
            stride = ov_opset.reduce_prod(
                remaining_dims, zero_tensor, keep_dims=False
            ).output(0)
        else:
            stride = one_scalar
        strides.append(stride)

    coord_tensors = []
    # For each dimension, compute the coordinate for every flat index.
    # Example:
    # for shape [2, 3], flat index 4 -> coordinates [1, 1] (row 1, col 1).
    for dim in range(rank):
        coords = ov_opset.mod(
            ov_opset.divide(flat_indices, strides[dim]).output(0),
            dim_sizes[dim],
        ).output(0)
        coord_tensors.append(coords)

    coord_tensors_unsqueezed = []
    for coord in coord_tensors:
        # Unsqueeze to make each coordinate a column vector for concatenation.
        coord_unsqueezed = ov_opset.unsqueeze(coord, one_tensor).output(0)
        coord_tensors_unsqueezed.append(coord_unsqueezed)

    # Concatenate all coordinate columns to form [total_elements, rank] matrix.
    # Each row is a multi-dimensional index into the updates tensor.
    # Example:
    # for shape [2, 3], row 4 = [1, 1].
    indices_matrix = ov_opset.concat(coord_tensors_unsqueezed, axis=1).output(0)

    # Broadcast start indices to match the number of updates.
    # Example:
    # start_indices = (2, 3), indices_matrix = [[0,0],[0,1],...],
    # start_broadcast = [[2,3],[2,3],...]
    start_tensor = ov_opset.concat(processed_start_indices, axis=0).output(0)
    start_reshaped = ov_opset.reshape(
        start_tensor, ov_opset.constant([1, rank], Type.i32), special_zero=False
    ).output(0)

    broadcast_shape = ov_opset.concat(
        [
            ov_opset.unsqueeze(total_elements, zero_tensor).output(0),
            one_tensor,
        ],
        axis=0,
    ).output(0)

    start_broadcast = ov_opset.tile(start_reshaped, broadcast_shape).output(0)

    # Add the broadcasted start indices to the relative indices
    # to get absolute indices in the input tensor.
    # Example:
    # if start=(2,3), update index [1,1] -> absolute index [3,4].
    absolute_indices = ov_opset.add(indices_matrix, start_broadcast).output(0)

    # Flatten the updates tensor to match the flat indices.
    updates_flat = ov_opset.reshape(
        updates_tensor,
        ov_opset.unsqueeze(total_elements, zero_tensor).output(0),
        special_zero=False,
    ).output(0)

    # Perform the scatter update: for each absolute index,
    # set the corresponding value from updates_flat.
    result = ov_opset.scatter_nd_update(
        inputs, absolute_indices, updates_flat
    ).output(0)
    return OpenVINOKerasTensor(result)


def switch(index, branches, *operands):
    if len(branches) == 1:
        return branches[0](*operands)

    n = len(branches)
    index_ov = get_ov_output(convert_to_tensor(index, "int32"))
    index_ov = ov_opset.clamp(index_ov, 0, n - 1).output(0)
    operands_ov = [get_ov_output(op_val) for op_val in operands]

    def _trace_branch(branch_fn):
        params, wrapped = [], []
        for ov_out in operands_ov:
            p = ov_opset.parameter(
                ov_out.get_partial_shape(), ov_out.get_element_type()
            )
            params.append(p)
            wrapped.append(OpenVINOKerasTensor(p.output(0)))
        raw = branch_fn(*wrapped)
        if raw is None:
            flat = []
        elif isinstance(raw, (list, tuple)):
            flat = [get_ov_output(o) for o in raw]
        else:
            flat = [get_ov_output(raw)]
        return params, Model(flat, params), raw

    def _build(branch_idx):
        inner_outputs = None
        then_params, then_body, then_raw = _trace_branch(branches[branch_idx])
        if branch_idx == n - 2:
            else_params, else_body, _ = _trace_branch(branches[branch_idx + 1])
        else:
            inner_outputs, _ = _build(branch_idx + 1)
            else_params, pt_results = [], []
            for inner_out in inner_outputs:
                ep = ov_opset.parameter(
                    inner_out.get_partial_shape(),
                    inner_out.get_element_type(),
                )
                else_params.append(ep)
                pt_results.append(ep.output(0))
            else_body = Model(pt_results, else_params)

        cond = ov_opset.equal(
            index_ov,
            ov_opset.constant(branch_idx, Type.i32).output(0),
        ).output(0)
        if_node = ov_opset.if_op(cond)
        if_node.set_then_body(then_body)
        if_node.set_else_body(else_body)

        if inner_outputs is None:
            for ov_inp, tp, ep in zip(operands_ov, then_params, else_params):
                if_node.set_input(ov_inp, tp, ep)
        else:
            for ov_inp, tp in zip(operands_ov, then_params):
                if_node.set_input(ov_inp, tp, None)
            for inner_out, ep in zip(inner_outputs, else_params):
                if_node.set_input(inner_out, None, ep)

        outputs = [
            if_node.set_output(then_body.results[i], else_body.results[i])
            for i in range(len(then_body.results))
        ]
        return outputs, then_raw

    final_outputs, template_raw = _build(0)
    wrapped = [OpenVINOKerasTensor(o) for o in final_outputs]

    if template_raw is None:
        return None
    elif isinstance(template_raw, tuple):
        return tuple(wrapped)
    elif isinstance(template_raw, list):
        return list(wrapped)
    else:
        return wrapped[0]


def while_loop(
    cond,
    body,
    loop_vars,
    maximum_iterations=None,
):
    def flatten_structure(data):
        if isinstance(data, dict):
            return [v for k in sorted(data) for v in flatten_structure(data[k])]
        elif isinstance(data, (tuple, list)):
            return [v for item in data for v in flatten_structure(item)]
        else:
            return [data]

    def pack_structure(template, flat):
        if isinstance(template, dict):
            keys = sorted(template)
            packed = {}
            for k in keys:
                value, flat = pack_structure(template[k], flat)
                packed[k] = value
            return packed, flat
        elif isinstance(template, (tuple, list)):
            packed = []
            for item in template:
                value, flat = pack_structure(item, flat)
                packed.append(value)
            return (
                tuple(packed) if isinstance(template, tuple) else packed
            ), flat
        else:
            return flat[0], flat[1:]

    is_scalar_input = _is_scalar(loop_vars)

    if is_scalar_input:
        loop_vars = (loop_vars,)
    elif isinstance(loop_vars, (list, np.ndarray)):
        loop_vars = tuple(loop_vars)
    else:
        if not isinstance(loop_vars, (tuple, dict)):
            raise ValueError(
                "Expected tuple or dict for `loop_vars`, "
                f"Received: {type(loop_vars)}"
            )

    flat_loop_vars = flatten_structure(loop_vars)
    loop_vars_ov = [get_ov_output(var) for var in flat_loop_vars]

    maximum_iterations = (
        ov_opset.constant(-1, Type.i32).output(0)
        if maximum_iterations is None
        else get_ov_output(maximum_iterations)
    )

    trip_count = maximum_iterations
    execution_condition = ov_opset.constant(True, Type.boolean).output(0)
    loop = ov_opset.loop(trip_count, execution_condition)

    shapes = [var.get_partial_shape() for var in loop_vars_ov]
    types = [var.get_element_type() for var in loop_vars_ov]
    params = [
        ov_opset.parameter(shape, dtype) for shape, dtype in zip(shapes, types)
    ]
    param_tensors = [OpenVINOKerasTensor(p.output(0)) for p in params]

    packed_args, _ = pack_structure(loop_vars, param_tensors)
    if isinstance(packed_args, dict):
        body_out = body(packed_args)
    else:
        body_out = body(*packed_args)

    if not isinstance(body_out, (list, tuple, dict)):
        body_out = (body_out,)

    flat_body_out = flatten_structure(body_out)
    if isinstance(packed_args, dict):
        cond_output = get_ov_output(cond(body_out))
    else:
        cond_output = get_ov_output(cond(*body_out))

    if len(cond_output.get_partial_shape()) != 0:
        raise ValueError(
            "`cond` function must return a scalar boolean value, "
            "but got shape {}".format(cond_output.get_partial_shape())
        )

    for p, out in zip(params, flat_body_out):
        out_shape = get_ov_output(out).get_partial_shape()
        p.set_partial_shape(out_shape)

    results = [cond_output] + [get_ov_output(x) for x in flat_body_out]
    body_func = Model(results=results, parameters=params)
    loop.set_function(body_func)
    loop.set_special_body_ports([-1, 0])

    for param, init_val, next_val in zip(params, loop_vars_ov, flat_body_out):
        loop.set_merged_input(param, init_val, get_ov_output(next_val))

    outputs_flat = [
        OpenVINOKerasTensor(loop.get_iter_value(get_ov_output(val)))
        for val in flat_body_out
    ]
    final_output, _ = pack_structure(loop_vars, outputs_flat)

    if is_scalar_input:
        if isinstance(final_output, tuple):
            return final_output[0]
        else:
            return final_output
    else:
        return final_output


def fori_loop(lower, upper, body_fun, init_val):
    return while_loop(
        lambda i, val: i < upper,
        lambda i, val: (i + 1, body_fun(i, val)),
        (lower, init_val),
    )[1]


def stop_gradient(variable):
    return variable


def unstack(x, num=None, axis=0):
    x_ov = get_ov_output(x)
    axis_ov = get_ov_output(axis)

    if num is None:
        shape = x_ov.get_partial_shape()
        num = shape[axis].get_length()

    split_ov = ov_opset.split(x_ov, axis_ov, num)

    return [
        OpenVINOKerasTensor(ov_opset.squeeze(out, axis_ov).output(0))
        for out in split_ov.outputs()
    ]


def random_seed_dtype():
    # OpenVINO arithmetic promotes uint32 * int32 → int32 (Python ints are
    # i32 in get_ov_output), so the seed tensor from SeedGenerator.next()
    # ends up as int32. Returning int32 keeps the declared dtype consistent
    # with what the backend actually produces.
    return "int32"


class custom_gradient:
    """Decorator for custom gradients.

    OpenVINO is an inference-only backend, so this acts as a pass-through:
    it runs the forward pass and discards the gradient function.

    Arguments:
        fun: The forward pass function.
    """

    def __init__(self, fun):
        warnings.warn(
            "`custom_gradient` for the openvino backend acts as a "
            "pass-through to support the forward pass. No gradient "
            "computation or modification takes place."
        )
        self.fun = fun

    def __call__(self, *args, **kwargs):
        outputs, _ = self.fun(*args, **kwargs)
        return outputs


def remat(f):
    warnings.warn(
        "Rematerialization memory optimization is not supported by the "
        "OpenVino backend. Please switch to JAX, TensorFlow, or PyTorch to "
        "utilize this feature."
    )
    return f
