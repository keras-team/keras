import builtins
import functools
import warnings

import mlx.core as mx
import numpy as np

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import slice_along_axis
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.symbolic_scope import SymbolicScope

try:
    import h5py
except ImportError:
    h5py = None

SUPPORTS_SPARSE_TENSORS = False
SUPPORTS_RAGGED_TENSORS = False
IS_THREAD_SAFE = True

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
    if isinstance(dtype, mx.Dtype):
        return dtype
    standardized_dtype = MLX_DTYPES.get(standardize_dtype(dtype), None)
    if standardized_dtype is None:
        raise ValueError(f"Unsupported dtype for MLX: {dtype}")
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


def _is_h5py_dataset(obj):
    return (
        type(obj).__module__.startswith("h5py.")
        and type(obj).__name__ == "Dataset"
    )


def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    if sparse:
        raise ValueError("`sparse=True` is not supported with mlx backend")
    if ragged:
        raise ValueError("`ragged=True` is not supported with mlx backend")
    mlx_dtype = to_mlx_dtype(dtype) if dtype is not None else None

    if is_tensor(x):
        if dtype is None:
            return x
        return x.astype(mlx_dtype)

    if isinstance(x, Variable):
        if dtype and standardize_dtype(dtype) != x.dtype:
            return x.value.astype(mlx_dtype)
        return x.value

    if isinstance(x, np.ndarray):
        if x.dtype == np.float64:
            # mlx backend does not support float64
            x = x.astype(np.float32)
        if standardize_dtype(x.dtype) == "bfloat16" and mlx_dtype is None:
            # if a bfloat16 np.ndarray is passed to mx.array with dtype=None
            # it casts the output to complex64, so we force cast to bfloat16
            mlx_dtype = mx.bfloat16
        return mx.array(x, dtype=mlx_dtype)

    if isinstance(x, list):

        def to_scalar_list(x):
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

    if _is_h5py_dataset(x):
        if h5py is None:
            raise ImportError(
                "h5py must be installed in order to load HDF5 datasets."
            )
        # load h5py._hl.dataset.Dataset object with numpy
        x = np.array(x)

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

    with StatelessScope(), SymbolicScope():
        outputs = symbolic_call(fn, args, kwargs, fill_value=83)

        none_in_shape = any(
            builtins.map(has_none_shape, tree.flatten((args, kwargs)))
        )
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
    if values.dtype == mx.int64:
        values = values.astype(mx.int32)
    elif values.dtype == mx.uint64:
        values = values.astype(mx.uint32)
    zeros = mx.zeros(shape, dtype=values.dtype)
    indices = tuple(indices[..., i] for i in range(indices.shape[-1]))
    zeros = zeros.at[indices].add(values)

    return zeros


def scatter_update(inputs, indices, updates):
    inputs = convert_to_tensor(inputs)
    indices = convert_to_tensor(indices)
    updates = convert_to_tensor(updates)
    if inputs.dtype == mx.int64:
        inputs = inputs.astype(mx.int32)
    elif inputs.dtype == mx.uint64:
        inputs = inputs.astype(mx.uint32)
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


def switch(index, branches, *operands):
    index = convert_to_tensor(index, "int32")
    index = mx.clip(index, 0, len(branches) - 1).tolist()
    operands = tuple(convert_to_tensor(o) for o in operands)
    return branches[index](*operands)


def while_loop(
    cond,
    body,
    loop_vars,
    maximum_iterations=None,
):
    current_iter = 0
    iteration_check = (
        lambda iter: maximum_iterations is None or iter < maximum_iterations
    )

    is_sequence = isinstance(loop_vars, (tuple, list))

    if is_sequence:
        loop_vars = tuple(convert_to_tensor(v) for v in loop_vars)
    else:
        loop_vars = tree.map_structure(convert_to_tensor, loop_vars)

    while (
        cond(*loop_vars) if is_sequence else cond(loop_vars)
    ) and iteration_check(current_iter):
        new_vars = body(*loop_vars) if is_sequence else body(loop_vars)

        if is_sequence:
            if not isinstance(new_vars, (tuple, list)):
                new_vars = (new_vars,)
            loop_vars = tuple(convert_to_tensor(v) for v in new_vars)
        else:
            loop_vars = tree.map_structure(convert_to_tensor, new_vars)

        current_iter += 1

    return loop_vars


def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def stop_gradient(variable):
    if isinstance(variable, Variable):
        variable = variable.value
    return mx.stop_gradient(variable)


def unstack(x, num=None, axis=0):
    y = x.split(num or x.shape[axis], axis=axis)
    return [yi.squeeze(axis) for yi in y]


def random_seed_dtype():
    # mlx random seed uses uint32.
    return "uint32"


def reverse_sequence(xs):
    indices = mx.arange(xs.shape[0] - 1, -1, -1)
    return mx.take(xs, indices, axis=0)


def flip(x, axis=None):
    if axis is None:
        # flip all axes
        axes = range(x.ndim)
    else:
        axes = [axis] if isinstance(axis, int) else axis

    for axis in axes:
        indices = mx.arange(x.shape[axis] - 1, -1, -1)
        x = mx.take(x, indices, axis=axis)

    return x


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
        n = int(length) if length is not None else shape(xs_flat[0])[0]

    init_flat = tree.flatten(init)
    init_flat = [convert_to_tensor(init) for init in init_flat]
    init = pack_output(init_flat)
    dummy_y = [mx.zeros_like(init) for init in init_flat]

    carry = init
    ys = []
    maybe_reversed = reversed if reverse else lambda x: x
    for i in maybe_reversed(range(n)):
        xs_slice = [x[i] for x in xs_flat]
        packed_xs = pack_input(xs_slice) if len(xs_slice) > 0 else None
        carry, y = f(carry, packed_xs)
        ys.append(y if y is not None else dummy_y)
    stacked_y = tree.map_structure(
        lambda *ys: mx.stack(ys), *maybe_reversed(ys)
    )
    return carry, stacked_y


def map(f, xs):
    def g(_, x):
        return (), f(x)

    _, ys = scan(g, (), xs)
    return ys


def dilate(x, axis, dilation_rate):
    x_shape = list(x.shape)
    x_shape[axis] = x.shape[axis] * dilation_rate - 1

    result = mx.zeros(x_shape, dtype=x.dtype)

    if axis >= 0:
        slices = [builtins.slice(None)] * axis + [
            builtins.slice(0, None, dilation_rate)
        ]
    else:
        slices = [Ellipsis, builtins.slice(0, None, dilation_rate)] + [
            builtins.slice(None)
        ] * (-1 - axis)
    result[tuple(slices)] = x

    return result


def associative_scan(f, elems, reverse=False, axis=0):
    # Ref: jax.lax.associative_scan
    if not callable(f):
        raise TypeError(f"`f` should be a callable. Received: f={f}")
    elems_flat = tree.flatten(elems)
    elems_flat = [convert_to_tensor(elem) for elem in elems_flat]
    if reverse:
        elems_flat = [flip(elem, (axis,)) for elem in elems_flat]

    def _combine(a_flat, b_flat):
        a = tree.pack_sequence_as(elems, a_flat)
        b = tree.pack_sequence_as(elems, b_flat)
        c = f(a, b)
        c_flat = tree.flatten(c)
        return c_flat

    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError(
            "Array inputs to associative_scan must have the same "
            "first dimension. (saw: {})".format(
                [elem.shape for elem in elems_flat]
            )
        )

    def _interleave(a, b, axis):
        """Given two Tensors of static shape, interleave them along axis."""
        assert (
            a.shape[axis] == b.shape[axis] or a.shape[axis] == b.shape[axis] + 1
        )

        # we want to get a: [a1, a2], b: [b1, b2]
        # to a: [a1, 0, a2, 0], b: [0, b1, 0, b2]
        a_dil = dilate(a, axis, 2)
        b_dil = dilate(b, axis, 2)

        a_pad = [[0, 0] for _ in range(a.ndim)]
        a_pad[axis][-1] = 1 if a.shape[axis] == b.shape[axis] else 0

        b_pad = [[0, 0] for _ in range(b.ndim)]
        b_pad[axis] = [1, 0] if a.shape[axis] == b.shape[axis] else [1, 1]

        op = mx.bitwise_or if a.dtype == mx.bool_ else mx.add
        return op(
            mx.pad(a_dil, a_pad),
            mx.pad(b_dil, b_pad),
        )

    def _scan(elems):
        num_elems = elems[0].shape[axis]
        if num_elems < 2:
            return elems

        reduced_elems = _combine(
            [
                slice_along_axis(elem, 0, -1, step=2, axis=axis)
                for elem in elems
            ],
            [
                slice_along_axis(elem, 1, None, step=2, axis=axis)
                for elem in elems
            ],
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
            mx.concatenate(
                [slice_along_axis(elem, 0, 1, axis=axis), result],
                axis=axis,
            )
            for (elem, result) in zip(elems, even_elems)
        ]
        return list(
            builtins.map(
                functools.partial(_interleave, axis=axis), even_elems, odd_elems
            )
        )

    scans = _scan(elems_flat)
    if reverse:
        scans = [flip(scanned, (axis,)) for scanned in scans]

    return tree.pack_sequence_as(elems, scans)


class custom_gradient:
    """Decorator for custom gradients.

    Args:
        fun: Forward pass function.
    """

    def __init__(self, fun):
        warnings.warn(
            "`custom_gradient` for the mlx backend acts as a pass-through to "
            "support the forward pass. No gradient computation or modification "
            "takes place."
        )
        self.fun = fun

    def __call__(self, *args, **kwargs):
        outputs, _ = self.fun(*args, **kwargs)
        return outputs
