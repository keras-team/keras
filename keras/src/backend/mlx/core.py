import mlx.core as mx
import numpy as np

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope

SUPPORTS_SPARSE_TENSORS = False

MLX_DTYPES = {
    "float16": mx.float16,
    "float32": mx.float32,
    "float64": None,  # mlx does not support float64
    "uint8": mx.uint8,
    "uint16": mx.uint16,
    "uint32": mx.uint32,
    "uint64": mx.uint64,
    "int8": mx.int8,
    "int16": mx.int16,
    "int32": mx.int32,
    "int64": mx.int64,
    "bfloat16": mx.bfloat16,
    "bool": mx.bool_,
}


def to_mlx_dtype(dtype):
    print(f"Input dtype: {dtype}")
    # Check if the dtype is already an instance of mlx.core.Dtype
    if isinstance(dtype, mx.Dtype):
        print("dtype is an instance of mx.Dtype")
        return dtype

    print("Standardizing dtype...")
    # Convert numpy dtype string or Python type to a standard dtype string
    dtype_str = np.dtype(dtype).name if not isinstance(dtype, str) else dtype
    # Fetch the corresponding mlx.core.Dtype
    standardized_dtype = MLX_DTYPES.get(dtype_str, None)

    print(f"Standardized dtype: {standardized_dtype}")
    if standardized_dtype is None:
        print(f"Unsupported dtype for MLX: {dtype}")
        raise ValueError(f"Unsupported dtype for MLX: {dtype}")
    print("Returning standardized dtype")
    return standardized_dtype


class Variable(KerasVariable):
    def _initialize(self, value):
        self._value = convert_to_tensor(value, dtype=self._dtype)

    def _direct_assign(self, value):
        self._value = value

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    def __mlx_array__(self):
        return self.value

    def __array__(self, dtype=None):
        value = convert_to_numpy(self._value)
        if dtype:
            return value.astype(dtype)
        return value


def convert_to_tensor(x, dtype=None, sparse=None):
    print(f"Input: {x}, dtype: {dtype}, sparse: {sparse}")
    if sparse:
        print("`sparse=True` is not supported with mlx backend")
        raise ValueError("`sparse=True` is not supported with mlx backend")
    mlx_dtype = to_mlx_dtype(dtype) if dtype is not None else None
    print(f"mlx_dtype: {mlx_dtype}")

    if is_tensor(x):
        print("x is a tensor")
        if dtype is None:
            return x
        return x.astype(mlx_dtype)

    if isinstance(x, Variable):
        print("x is an instance of Variable")
        if dtype and standardize_dtype(dtype) != x.dtype:
            return x.value.astype(mlx_dtype)
        return x.value

    if isinstance(x, np.ndarray):
        print("x is an instance of np.ndarray")
        if x.dtype == np.int64:
            x = x.astype(np.int32)
        x = x.astype(standardize_dtype(x.dtype))
        return mx.array(x, dtype=mlx_dtype)

    if isinstance(x, list):
        print("x is a list")

        def to_scalar_list(x):
            print(f"Converting to scalar list: {x}")
            if isinstance(x, list):
                return [to_scalar_list(xi) for xi in x]
            elif isinstance(x, mx.array):
                if x.ndim == 0:
                    return x.item()
                else:
                    return x.tolist()
            else:
                return x

        return mx.array(to_scalar_list(x), dtype=mlx_dtype)

    print("Returning mx.array")
    return mx.array(x, dtype=mlx_dtype)


def convert_to_tensors(*xs):
    ys = [None] * len(xs)
    dtype = None
    for i, x in enumerate(xs):
        if not isinstance(x, (int, float, bool)):
            ys[i] = convert_to_tensor(x)
            dtype = ys[i].dtype
    # Floating point wins so scalars promote to dtype
    if dtype in (mx.float32, mx.float16, mx.bfloat16):
        for i, x in enumerate(xs):
            if ys[i] is None:
                ys[i] = mx.array(x, dtype=dtype)
    # Bool loses against everything so scalars keep their type
    elif dtype == mx.bool_:
        for i, x in enumerate(xs):
            if ys[i] is None:
                ys[i] = mx.array(x)
    # Integral types keep their type except if the scalar is a float
    else:
        for i, x in enumerate(xs):
            if ys[i] is None:
                if isinstance(x, float):
                    ys[i] = mx.array(x)
                else:
                    ys[i] = mx.array(x, dtype=dtype)

    return ys


def convert_to_numpy(x):
    # Performs a copy. If we want 0-copy we can pass copy=False
    return np.array(x)


def is_tensor(x):
    return isinstance(x, mx.array)


def shape(x):
    return tuple(x.shape)


def cast(x, dtype):
    return convert_to_tensor(x, dtype=dtype)


# Shape / dtype inference util
def compute_output_spec(fn, *args, **kwargs):
    def has_none_shape(x):
        """Check for if a `KerasTensor` has dynamic shape."""
        if isinstance(x, KerasTensor):
            return None in x.shape
        return False

    def convert_keras_tensor_to_mlx(x, fill_value=None):
        """Convert `KerasTensor`s to `mlx.array`s."""
        if isinstance(x, KerasTensor):
            shape = list(x.shape)
            if fill_value:
                for i, e in enumerate(shape):
                    if e is None:
                        shape[i] = fill_value
            return mx.ones(shape, dtype=MLX_DTYPES[x.dtype])
        return x

    def convert_mlx_to_keras_tensor(x):
        """Convert `mlx.array`s to `KerasTensor`s."""
        if is_tensor(x):
            return KerasTensor(x.shape, standardize_dtype(x.dtype))
        return x

    def symbolic_call(fn, args, kwargs, fill_value):
        """Call `fn` to infer output shape and dtype."""
        arr_args, arr_kwargs = tree.map_structure(
            lambda x: convert_keras_tensor_to_mlx(x, fill_value),
            (args, kwargs),
        )
        return fn(*arr_args, **arr_kwargs)

    with StatelessScope():
        outputs = symbolic_call(fn, args, kwargs, fill_value=83)

        none_in_shape = any(map(has_none_shape, tree.flatten((args, kwargs))))
        if none_in_shape:
            outputs_1 = outputs
            outputs_2 = symbolic_call(fn, args, kwargs, fill_value=89)

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

        output_spec = tree.map_structure(convert_mlx_to_keras_tensor, outputs)
    return output_spec


def cond(pred, true_fn, false_fn):
    # TODO: How should we avoid evaluating pred in case we are tracing?
    if pred:
        return true_fn()
    return false_fn()


def vectorized_map(function, elements):
    return mx.vmap(function)(elements)


def scatter(indices, values, shape):
    indices = convert_to_tensor(indices)
    values = convert_to_tensor(values)
    zeros = mx.zeros(shape, dtype=values.dtype)
    indices = tuple(indices[..., i] for i in range(indices.shape[-1]))
    zeros = zeros.at[indices].add(values)

    return zeros


def scatter_update(inputs, indices, updates):
    inputs = convert_to_tensor(inputs)
    indices = convert_to_tensor(indices)
    updates = convert_to_tensor(updates)
    indices = tuple(indices[..., i] for i in range(indices.shape[-1]))
    inputs[indices] = updates

    return inputs


def slice(inputs, start_indices, shape):
    inputs = convert_to_tensor(inputs)
    if not isinstance(shape, list):
        shape = convert_to_tensor(shape, dtype="int32").tolist()
    else:
        shape = [i if isinstance(i, int) else i.item() for i in shape]
    if not isinstance(start_indices, list):
        start_indices = convert_to_tensor(start_indices, dtype="int32").tolist()
    else:
        start_indices = [
            i if isinstance(i, int) else i.item() for i in start_indices
        ]
    python_slice = __builtins__["slice"]
    slices = tuple(
        python_slice(start_index, start_index + length)
        for start_index, length in zip(start_indices, shape)
    )
    return inputs[slices]


def slice_update(inputs, start_indices, updates):
    inputs = convert_to_tensor(inputs)
    if not isinstance(start_indices, list):
        start_indices = convert_to_tensor(start_indices, dtype="int32").tolist()
    else:
        start_indices = [
            i if isinstance(i, int) else i.item() for i in start_indices
        ]
    updates = convert_to_tensor(updates)

    python_slice = __builtins__["slice"]
    slices = tuple(
        python_slice(start_index, start_index + update_length)
        for start_index, update_length in zip(start_indices, updates.shape)
    )
    inputs[slices] = updates
    return inputs


def while_loop(
    cond,
    body,
    loop_vars,
    maximum_iterations=None,
):
    # TODO: How should we avoid evaluating cond when tracing?
    current_iter = 0
    iteration_check = (
        lambda iter: maximum_iterations is None or iter < maximum_iterations
    )
    loop_vars = tuple([convert_to_tensor(v) for v in loop_vars])
    while cond(*loop_vars) and iteration_check(current_iter):
        loop_vars = body(*loop_vars)
        if not isinstance(loop_vars, (list, tuple)):
            loop_vars = (loop_vars,)
        loop_vars = tuple(loop_vars)
        current_iter += 1
    return loop_vars


def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def stop_gradient(variable):
    return mx.stop_gradient(variable)


def unstack(x, num=None, axis=0):
    y = x.split(num or x.shape[axis], axis=axis)
    return [yi.squeeze(axis) for yi in y]