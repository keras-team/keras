import builtins
import contextlib
import warnings

import numpy as np
import openvino as ov
import openvino.opset14 as ov_opset
from openvino import Model
from openvino import Tensor
from openvino import Type
from openvino import compile_model

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import dtypes
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope

SUPPORTS_SPARSE_TENSORS = False
SUPPORTS_RAGGED_TENSORS = False
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


def align_operand_types(x1, x2, op_name):
    x1_type = x1.element_type
    x2_type = x2.element_type
    if x1_type.is_dynamic() or x2_type.is_dynamic():
        raise ValueError(
            f"'{op_name}' operation is not supported for dynamic operand type "
            "with openvino backend"
        )
    x1_type = ov_to_keras_type(x1_type)
    x2_type = ov_to_keras_type(x2_type)
    result_type = dtypes.result_type(x1_type, x2_type)
    result_type = OPENVINO_DTYPES[result_type]
    if x1_type != result_type:
        x1 = ov_opset.convert(x1, result_type).output(0)
    if x2_type != result_type:
        x2 = ov_opset.convert(x2, result_type).output(0)
    return x1, x2


# create ov.Output (symbolic OpenVINO tensor)
# for different input `x`
def get_ov_output(x, ov_type=None):
    if isinstance(x, float):
        if ov_type is None:
            ov_type = Type.f32
        x = ov_opset.constant(x, ov_type).output(0)
    elif isinstance(x, int):
        if ov_type is None:
            ov_type = Type.i32
        x = ov_opset.constant(x, ov_type).output(0)
    elif isinstance(x, np.ndarray):
        if x.dtype == np.dtype("bfloat16"):
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
    elif isinstance(x, Tensor):
        x = ov_opset.constant(x.data).output(0)
    else:
        raise ValueError(
            "unsupported type of `x` to create ov.Output: {}".format(type(x))
        )
    return x


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
        self.shape = tuple(x_keras_shape)
        self.dtype = x_keras_type
        self.ndim = None
        self.data = data
        if x.get_partial_shape().rank.is_static:
            self.ndim = x.get_partial_shape().rank.get_length()

    def __add__(self, other):
        first = self.output
        other = get_ov_output(other)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__add__"
        )
        return OpenVINOKerasTensor(ov_opset.add(first, other).output(0))

    def __radd__(self, other):
        first = self.output
        other = get_ov_output(other)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__radd__"
        )
        return OpenVINOKerasTensor(ov_opset.add(first, other).output(0))

    def __sub__(self, other):
        first = self.output
        other = get_ov_output(other)
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
        other = get_ov_output(other)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__rsub__"
        )
        return OpenVINOKerasTensor(ov_opset.subtract(other, first).output(0))

    def __mul__(self, other):
        first = self.output
        other = get_ov_output(other)
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
        other = get_ov_output(other)
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
        other = get_ov_output(other)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__truediv__"
        )
        return OpenVINOKerasTensor(ov_opset.divide(first, other).output(0))

    def __rtruediv__(self, other):
        first = self.output
        other = get_ov_output(other)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__rtruediv__"
        )
        return OpenVINOKerasTensor(ov_opset.divide(other, first).output(0))

    def __floordiv__(self, other):
        first = self.output
        other = get_ov_output(other)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__floordiv__"
        )
        return OpenVINOKerasTensor(ov_opset.divide(first, other).output(0))

    def __rfloordiv__(self, other):
        first = self.output
        other = get_ov_output(other)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__rfloordiv__"
        )
        return OpenVINOKerasTensor(ov_opset.divide(other, first).output(0))

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
        other = get_ov_output(other)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__pow__"
        )
        return OpenVINOKerasTensor(ov_opset.power(first, other).output(0))

    def __rpow__(self, other):
        other = get_ov_output(other)
        first = self.output
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__rpow__"
        )
        return OpenVINOKerasTensor(ov_opset.power(other, first).output(0))

    def __lt__(self, other):
        first = self.output
        other = get_ov_output(other)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__lt__"
        )
        return OpenVINOKerasTensor(ov_opset.less(first, other).output(0))

    def __gt__(self, other):
        first = self.output
        other = get_ov_output(other)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__gt__"
        )
        return OpenVINOKerasTensor(ov_opset.greater(first, other).output(0))

    def __le__(self, other):
        first = self.output
        other = get_ov_output(other)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__le__"
        )
        return OpenVINOKerasTensor(ov_opset.less_equal(first, other).output(0))

    def __ge__(self, other):
        first = self.output
        other = get_ov_output(other)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__ge__"
        )
        return OpenVINOKerasTensor(
            ov_opset.greater_equal(first, other).output(0)
        )

    def __eq__(self, other):
        first = self.output
        other = get_ov_output(other)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__eq__"
        )
        return OpenVINOKerasTensor(ov_opset.equal(first, other).output(0))

    def __ne__(self, other):
        first = self.output
        other = get_ov_output(other)
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
                if index == builtins.slice(None):
                    continue
                if index.step is not None and index.step < 0:
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
            step = ov_opset.constant(slice_steps, Type.i32).output(0)
            start = ov_opset.constant(slice_starts, Type.i32).output(0)
            stop = ov_opset.constant(slice_ends, Type.i32).output(0)
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
        assert ov_shape.rank.is_static and ov_shape.rank.get_length() > 0, (
            "rank must be static and greater than zero"
        )
        assert ov_shape[0].is_static, "the first dimension must be static"
        return ov_shape[0].get_length()

    def __mod__(self, other):
        first = self.output
        other = get_ov_output(other)
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


def ov_to_keras_type(ov_type):
    for _keras_type, _ov_type in OPENVINO_DTYPES.items():
        if ov_type == _ov_type:
            return _keras_type
    raise ValueError(
        f"Requested OpenVINO type has no keras analogue '{ov_type.to_string()}'"
    )


@contextlib.contextmanager
def device_scope(device_name):
    current_device = _parse_device_input(device_name)
    global_state.set_global_attribute("openvino_device", current_device)


def get_device():
    device = global_state.get_global_attribute("openvino_device", None)
    if device is None:
        return "CPU"
    return device


def _parse_device_input(device_name):
    if isinstance(device_name, str):
        # We support string value like "cpu:0", "gpu:1", and need to convert
        # "gpu" to "cuda"
        device_name = device_name.upper()
        device_type, _ = device_name.split(":")
        return device_type
    else:
        raise ValueError(
            "Invalid value for argument `device_name`. "
            "Expected a string like 'gpu:0' or 'cpu'. "
            f"Received: device_name='{device_name}'"
        )
    return device_name


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
        if isinstance(self.value, OpenVINOKerasTensor):
            return self.value.output.get_node().data
        return self.value.data

    def __getitem__(self, idx):
        if isinstance(self.value, OpenVINOKerasTensor):
            arr = self.value.output.get_node().data
            return arr.__getitem__(idx)
        return self.value.__getitem__(idx)

    def __int__(self):
        if isinstance(self.value, OpenVINOKerasTensor):
            arr = self.value.output.get_node().data
        else:
            arr = self.value.data
        if arr.ndim > 0:
            raise TypeError(
                "Only scalar arrays can be converted to Python scalars. "
                f"Got: shape={arr.shape}"
            )
        return int(arr)

    def __float__(self):
        if isinstance(self.value, OpenVINOKerasTensor):
            arr = self.value.output.get_node().data
        else:
            arr = self.value.data
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
    assert isinstance(x, OpenVINOKerasTensor), (
        "unsupported type {} for `convert_to_numpy` in openvino backend".format(
            type(x)
        )
    )
    try:
        ov_result = x.output
        ov_model = Model(results=[ov_result], parameters=[])
        ov_compiled_model = compile_model(ov_model, get_device())
        result = ov_compiled_model({})[0]
    except Exception as inner_exception:
        raise RuntimeError(
            "`convert_to_numpy` failed to convert the tensor."
        ) from inner_exception
    return result


def is_tensor(x):
    if isinstance(x, OpenVINOKerasTensor):
        return True
    if isinstance(x, ov.Tensor):
        return True
    return False


def shape(x):
    return tuple(x.shape)


def cast(x, dtype):
    dtype = standardize_dtype(dtype)
    ov_type = OPENVINO_DTYPES[dtype]
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.convert(x, ov_type).output(0))


def cond(pred, true_fn, false_fn):
    raise NotImplementedError("`cond` is not supported with openvino backend")


def vectorized_map(function, elements):
    raise NotImplementedError(
        "`vectorized_map` is not supported with openvino backend"
    )


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


def scan(f, init, xs=None, length=None, reverse=False, unroll=1):
    raise NotImplementedError("`scan` is not supported with openvino backend")


def scatter(indices, values, shape):
    raise NotImplementedError(
        "`scatter` is not supported with openvino backend"
    )


def scatter_update(inputs, indices, updates):
    raise NotImplementedError(
        "`scatter_update` is not supported with openvino backend"
    )


def slice(inputs, start_indices, shape):
    inputs = get_ov_output(inputs)
    if isinstance(start_indices, (list, np.ndarray)):
        start_indices = tuple(start_indices)
    if isinstance(shape, (list, np.ndarray)):
        shape = tuple(shape)
    assert isinstance(start_indices, tuple), (
        "`slice` is not supported by openvino backend"
        " for `start_indices` of type {}".format(type(start_indices))
    )
    assert isinstance(shape, tuple), (
        "`slice` is not supported by openvino backend"
        " for `shape` of type {}".format(type(shape))
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
        assert isinstance(loop_vars, (tuple, dict)), (
            f"Unsupported type {type(loop_vars)} for loop_vars"
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
    raise NotImplementedError(
        "`fori_loop` is not supported with openvino backend"
    )


def stop_gradient(variable):
    return variable


def unstack(x, num=None, axis=0):
    raise NotImplementedError(
        "`unstack` is not supported with openvino backend"
    )


def random_seed_dtype():
    return "uint32"


def custom_gradient(fun):
    """Decorator for custom gradients.

    Args:
        fun: Forward pass function.
    """

    def __init__(self, fun):
        warnings.warn(
            "`custom_gradient` for the openvino backend"
            " acts as a pass-through to "
            "support the forward pass."
            " No gradient computation or modification "
            "takes place."
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
