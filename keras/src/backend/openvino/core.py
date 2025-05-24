import contextlib
import warnings

import numpy as np
import openvino as ov
import openvino.runtime.opset14 as ov_opset
from openvino import Model
from openvino import Tensor
from openvino import compile_model
from openvino.runtime import Type

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
        return OpenVINOKerasTensor(ov_opset.multiply(first, other).output(0))

    def __rmul__(self, other):
        first = self.output
        other = get_ov_output(other)
        first, other = align_operand_types(
            first, other, "OpenVINOKerasTensor::__rmul__"
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

    def __getitem__(self, indices):
        # now it has limited functionaly
        # and supports only a case with one integer index in indices
        # other indices must be None
        data = self.output
        axis = []
        gather_index = None
        if isinstance(indices, int):
            indices = (indices,)
        assert isinstance(indices, tuple), "only tuple is supported"
        for dim, index in enumerate(indices):
            if isinstance(index, int):
                axis.append(dim)
                gather_index = ov_opset.constant(index, Type.i32)
            else:
                assert (
                    index.start is None
                    and index.stop is None
                    and index.step is None
                )
        assert len(axis) == 1, "axis must contain one element"
        axis = ov_opset.constant(axis, Type.i32)
        return OpenVINOKerasTensor(
            ov_opset.gather(data, gather_index, axis).output(0)
        )

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


def _get_first_element(x):
    if isinstance(x, (tuple, list)):
        for elem_in_x in x:
            elem = _get_first_element(elem_in_x)
            if elem is not None:
                return elem
    elif _is_scalar(x):
        return x
    return None


def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    if sparse:
        raise ValueError("`sparse=True` is not supported with openvino backend")
    if ragged:
        raise ValueError("`ragged=True` is not supported with openvino backend")
    if isinstance(x, OpenVINOKerasTensor):
        return x
    elif isinstance(x, np.ndarray):
        if dtype is not None:
            dtype = standardize_dtype(dtype)
            ov_type = OPENVINO_DTYPES[dtype]
            return OpenVINOKerasTensor(ov_opset.constant(x, ov_type).output(0))
        return OpenVINOKerasTensor(ov_opset.constant(x).output(0))
    elif isinstance(x, (list, tuple)):
        if dtype is not None:
            dtype = standardize_dtype(dtype)
        else:
            # try to properly deduce element type
            elem = _get_first_element(x)
            if isinstance(elem, float):
                dtype = "float32"
            elif isinstance(elem, int):
                dtype = "int32"
        x = np.array(x, dtype=dtype)
        return OpenVINOKerasTensor(ov_opset.constant(x).output(0), x)
    elif isinstance(x, (float, int)):
        dtype = standardize_dtype(dtype)
        ov_type = OPENVINO_DTYPES[dtype]
        return OpenVINOKerasTensor(ov_opset.constant(x, ov_type).output(0), x)
    if dtype is not None:
        dtype = standardize_dtype(dtype)
    if isinstance(x, Variable):
        if dtype and dtype != x.dtype:
            return x.value.astype(dtype)
        return x.value
    if not is_tensor(x) and standardize_dtype(dtype) == "bfloat16":
        return ov.Tensor(np.asarray(x).astype(dtype))
    if dtype is None:
        dtype = result_type(
            *[getattr(item, "dtype", type(item)) for item in tree.flatten(x)]
        )
    return ov.Tensor(np.array(x, dtype=dtype))


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
    except:
        raise "`convert_to_numpy` cannot convert to numpy"
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


def slice(inputs, start_indices, lengths):
    inputs = get_ov_output(inputs)
    assert isinstance(start_indices, tuple), (
        "`slice` is not supported by openvino backend"
        " for `start_indices` of type {}".format(type(lengths))
    )
    assert isinstance(lengths, tuple), (
        "`slice` is not supported by openvino backend"
        " for `lengths` of type {}".format(type(lengths))
    )

    axes = []
    start = []
    stop = []
    for idx, length in enumerate(lengths):
        if length is not None and length >= 0:
            axes.append(idx)
            start.append(start_indices[idx])
            stop.append(start_indices[idx] + length)

    if len(axes) == 0:
        return inputs

    step = [1] * len(start)
    step = ov_opset.constant(step, Type.i32).output(0)
    start = ov_opset.constant(start, Type.i32).output(0)
    stop = ov_opset.constant(stop, Type.i32).output(0)
    axes = ov_opset.constant(axes, Type.i32).output(0)
    return OpenVINOKerasTensor(
        ov_opset.slice(inputs, start, stop, step, axes).output(0)
    )


def slice_update(inputs, start_indices, updates):
    raise NotImplementedError(
        "`slice_update` is not supported with openvino backend"
    )


def while_loop(
    cond,
    body,
    loop_vars,
    maximum_iterations=None,
):
    raise NotImplementedError(
        "`while_loop` is not supported with openvino backend"
    )


def fori_loop(lower, upper, body_fun, init_val):
    raise NotImplementedError(
        "`fori_loop` is not supported with openvino backend"
    )


def stop_gradient(x):
    return x


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
