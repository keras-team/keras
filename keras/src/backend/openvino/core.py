import contextlib

import numpy as np

from keras.src.backend.common import global_state
from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
import openvino as ov

SUPPORTS_SPARSE_TENSORS = False

OPENVINO_DTYPES = {
    "float16": ov.Type.f16,
    "float32": ov.Type.f32,
    "float64": ov.Type.f64,
    "uint8": ov.Type.u8,
    "uint16": ov.Type.u16,
    "uint32": ov.Type.u32,
    "int8": ov.Type.i8,
    "int16": ov.Type.i16,
    "int32": ov.Type.i32,
    "int64": ov.Type.i64,
    "bfloat16": ov.Type.bf16,
    "bool": ov.Type.boolean,
    "float8_e4m3fn": ov.Type.f8e4m3,
    "float8_e5m2": ov.Type.f8e5m2,
}


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
        self._value = np.array(value, dtype=self._dtype)

    def _direct_assign(self, value):
        self._value = np.array(value, dtype=self._dtype)

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    # Overload native accessor.
    def __array__(self):
        return self.value


def convert_to_tensor(x, dtype=None, sparse=None):
    if sparse:
        raise ValueError("`sparse=True` is not supported with numpy backend")
    if dtype is not None:
        dtype = standardize_dtype(dtype)
    if isinstance(x, Variable):
        if dtype and dtype != x.dtype:
            return x.value.astype(dtype)
        return x.value
    if not is_tensor(x) and standardize_dtype(dtype) == "bfloat16":
        # Can't create bfloat16 arrays on the fly (e.g. from a h5 Dataset).
        # Instead we convert "as is" (to stored dtype) and cast.
        return np.asarray(x).astype(dtype)
    if dtype is None:
        dtype = result_type(
            *[getattr(item, "dtype", type(item)) for item in tree.flatten(x)]
        )
    return np.array(x, dtype=dtype)


def convert_to_numpy(x):
    return np.array(x)


def is_tensor(x):
    if isinstance(x, (np.generic, np.ndarray)):
        return True
    return False


def shape(x):
    return x.shape


def cast(x, dtype):
    raise NotImplementedError(
        "`cast` is not supported with openvino backend"
    )


def cond(pred, true_fn, false_fn):
    raise NotImplementedError(
        "`cond` is not supported with openvino backend"
    )


def vectorized_map(function, elements):
    raise NotImplementedError(
        "`vectorized_map` is not supported with openvino backend"
    )


# Shape / dtype inference util
def compute_output_spec(fn, *args, **kwargs):
    with StatelessScope():

        def has_none_shape(x):
            if isinstance(x, KerasTensor):
                return None in x.shape
            return False

        none_in_shape = any(map(has_none_shape, tree.flatten((args, kwargs))))

        def convert_keras_tensor_to_numpy(x, fill_value=None):
            if isinstance(x, KerasTensor):
                shape = list(x.shape)
                if fill_value:
                    for i, e in enumerate(shape):
                        if e is None:
                            shape[i] = fill_value
                return np.empty(
                    shape=shape,
                    dtype=x.dtype,
                )
            return x

        args_1, kwargs_1 = tree.map_structure(
            lambda x: convert_keras_tensor_to_numpy(x, fill_value=83),
            (args, kwargs),
        )
        outputs_1 = fn(*args_1, **kwargs_1)

        outputs = outputs_1

        if none_in_shape:
            args_2, kwargs_2 = tree.map_structure(
                lambda x: convert_keras_tensor_to_numpy(x, fill_value=89),
                (args, kwargs),
            )
            outputs_2 = fn(*args_2, **kwargs_2)

            flat_out_1 = tree.flatten(outputs_1)
            flat_out_2 = tree.flatten(outputs_2)

            flat_out = []
            for x1, x2 in zip(flat_out_1, flat_out_2):
                shape = list(x1.shape)
                for i, e in enumerate(x2.shape):
                    if e != shape[i]:
                        shape[i] = None
                flat_out.append(KerasTensor(shape, standardize_dtype(x1.dtype)))
            outputs = tree.pack_sequence_as(outputs_1, flat_out)

        def convert_numpy_to_keras_tensor(x):
            if is_tensor(x):
                return KerasTensor(x.shape, standardize_dtype(x.dtype))
            return x

        output_spec = tree.map_structure(convert_numpy_to_keras_tensor, outputs)
    return output_spec


def scan(f, init, xs=None, length=None, reverse=False, unroll=1):
    raise NotImplementedError(
        "`scan` is not supported with openvino backend"
    )


def scatter(indices, values, shape):
    raise NotImplementedError(
        "`scatter` is not supported with openvino backend"
    )


def scatter_update(inputs, indices, updates):
    raise NotImplementedError(
        "`scatter_update` is not supported with openvino backend"
    )


def slice(inputs, start_indices, lengths):
    raise NotImplementedError(
        "`slice` is not supported with openvino backend"
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


def custom_gradient(fun):
    raise NotImplementedError(
        "`custom_gradient` is not supported with numpy backend"
    )
