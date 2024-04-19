"""
scatter
scatter_update
slice
slice_update
while_loop
stop_gradient
shape
cast
convert_to_tensor
convert_to_numpy
cond
is_tensor
custom_gradient
"""

import numpy as np

from keras.src import backend
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.utils import traceback_utils


class Scatter(Operation):
    def call(self, indices, values, shape):
        return backend.core.scatter(indices, values, shape)

    def compute_output_spec(self, indices, values, shape):
        return KerasTensor(shape, dtype=values.dtype)


@keras_export("keras.ops.scatter")
def scatter(indices, values, shape):
    """Returns a tensor of shape `shape` where `indices` are set to `values`.

    At a high level, this operation does `zeros[indices] = updates` and
    returns the output. It is equivalent to:

    ```python
    zeros = keras.ops.zeros(shape)
    output = keras.ops.scatter_update(zeros, indices, values)
    ```

    Args:
        indices: A tensor or list/tuple specifying
            indices for the values in `values`.
        values: A tensor, the values to be set at `indices`.
        shape: Shape of the output tensor.

    Example:

    >>> indices = [[0, 1], [1, 1]]
    >>> values = np.array([1., 1.])
    >>> keras.ops.scatter(indices, values, shape=(2, 2))
    array([[0., 1.],
           [0., 1.]])
    """
    if any_symbolic_tensors((indices, values, shape)):
        return Scatter().symbolic_call(indices, values, shape)
    return backend.core.scatter(indices, values, shape)


class ScatterUpdate(Operation):
    def call(self, inputs, indices, updates):
        return backend.core.scatter_update(inputs, indices, updates)

    def compute_output_spec(self, inputs, indices, updates):
        return KerasTensor(inputs.shape, dtype=inputs.dtype)


@keras_export("keras.ops.scatter_update")
def scatter_update(inputs, indices, updates):
    """Update inputs via updates at scattered (sparse) indices.

    At a high level, this operation does `inputs[indices] = updates`.
    Assume `inputs` is a tensor of shape `(D0, D1, ..., Dn)`, there are 2 main
    usages of `scatter_update`.

    1. `indices` is a 2D tensor of shape `(num_updates, n)`, where `num_updates`
        is the number of updates to perform, and `updates` is a 1D tensor of
        shape `(num_updates,)`. For example, if `inputs` is `zeros((4, 4, 4))`,
        and we want to update `inputs[1, 2, 3]` and `inputs[0, 1, 3]` as 1, then
        we can use:

    ```python
    inputs = np.zeros((4, 4, 4))
    indices = [[1, 2, 3], [0, 1, 3]]
    updates = np.array([1., 1.])
    inputs = keras.ops.scatter_update(inputs, indices, updates)
    ```

    2 `indices` is a 2D tensor of shape `(num_updates, k)`, where `num_updates`
        is the number of updates to perform, and `k` (`k < n`) is the size of
        each index in `indices`. `updates` is a `n - k`-D tensor of shape
        `(num_updates, inputs.shape[k:])`. For example, if
        `inputs = np.zeros((4, 4, 4))`, and we want to update `inputs[1, 2, :]`
        and `inputs[2, 3, :]` as `[1, 1, 1, 1]`, then `indices` would have shape
        `(num_updates, 2)` (`k = 2`), and `updates` would have shape
        `(num_updates, 4)` (`inputs.shape[2:] = 4`). See the code below:

    ```python
    inputs = np.zeros((4, 4, 4))
    indices = [[1, 2], [2, 3]]
    updates = np.array([[1., 1., 1, 1,], [1., 1., 1, 1,])
    inputs = keras.ops.scatter_update(inputs, indices, updates)
    ```

    Args:
        inputs: A tensor, the tensor to be updated.
        indices: A tensor or list/tuple of shape `(N, inputs.ndim)`, specifying
            indices to update. `N` is the number of indices to update, must be
            equal to the first dimension of `updates`.
        updates: A tensor, the new values to be put to `inputs` at `indices`.

    Returns:
        A tensor, has the same shape and dtype as `inputs`.
    """
    if any_symbolic_tensors((inputs, indices, updates)):
        return ScatterUpdate().symbolic_call(inputs, indices, updates)
    return backend.core.scatter_update(inputs, indices, updates)


class Slice(Operation):
    def call(self, inputs, start_indices, shape):
        return backend.core.slice(inputs, start_indices, shape)

    def compute_output_spec(self, inputs, start_indices, shape):
        return KerasTensor(shape, dtype=inputs.dtype)


@keras_export("keras.ops.slice")
def slice(inputs, start_indices, shape):
    """Return a slice of an input tensor.

    At a high level, this operation is an explicit replacement for array slicing
    e.g. `inputs[start_indices: start_indices + shape]`.
    Unlike slicing via brackets, this operation will accept tensor start
    indices on all backends, which is useful when indices dynamically computed
    via other tensor operations.

    ```python
    inputs = np.zeros((5, 5))
    start_indices = np.array([3, 3])
    shape = np.array([2, 2])
    inputs = keras.ops.slice(inputs, start_indices, updates)
    ```

    Args:
        inputs: A tensor, the tensor to be updated.
        start_indices: A list/tuple of shape `(inputs.ndim,)`, specifying
            the starting indices for updating.
        shape: The full shape of the returned slice.

    Returns:
        A tensor, has the same shape and dtype as `inputs`.
    """
    if any_symbolic_tensors((inputs, start_indices, shape)):
        return Slice().symbolic_call(inputs, start_indices, shape)
    return backend.core.slice(inputs, start_indices, shape)


class SliceUpdate(Operation):
    def call(self, inputs, start_indices, updates):
        return backend.core.slice_update(inputs, start_indices, updates)

    def compute_output_spec(self, inputs, start_indices, updates):
        return KerasTensor(inputs.shape, dtype=inputs.dtype)


@keras_export("keras.ops.slice_update")
def slice_update(inputs, start_indices, updates):
    """Update an input by slicing in a tensor of updated values.

    At a high level, this operation does
    `inputs[start_indices: start_indices + updates.shape] = updates`.
    Assume inputs is a tensor of shape `(D0, D1, ..., Dn)`,
    `start_indices` must be a list/tuple of n integers, specifying the starting
    indices. `updates` must have the same rank as `inputs`, and the size of each
    dim must not exceed `Di - start_indices[i]`. For example, if we have 2D
    inputs `inputs = np.zeros((5, 5))`, and we want to update the intersection
    of last 2 rows and last 2 columns as 1, i.e.,
    `inputs[3:, 3:] = np.ones((2, 2))`, then we can use the code below:

    ```python
    inputs = np.zeros((5, 5))
    start_indices = [3, 3]
    updates = np.ones((2, 2))
    inputs = keras.ops.slice_update(inputs, start_indices, updates)
    ```

    Args:
        inputs: A tensor, the tensor to be updated.
        start_indices: A list/tuple of shape `(inputs.ndim,)`, specifying
            the starting indices for updating.
        updates: A tensor, the new values to be put to `inputs` at `indices`.
            `updates` must have the same rank as `inputs`.

    Returns:
        A tensor, has the same shape and dtype as `inputs`.
    """
    if any_symbolic_tensors((inputs, start_indices, updates)):
        return SliceUpdate().symbolic_call(inputs, start_indices, updates)
    return backend.core.slice_update(inputs, start_indices, updates)


class WhileLoop(Operation):
    def __init__(self, cond, body, maximum_iterations):
        super().__init__()
        self.cond = cond
        self.body = body
        self.maximum_iterations = maximum_iterations

    def call(self, loop_vars):
        return backend.core.while_loop(
            self.cond,
            self.body,
            loop_vars,
            maximum_iterations=self.maximum_iterations,
        )

    def compute_output_spec(self, loop_vars):
        return [KerasTensor(v.shape, dtype=v.dtype) for v in loop_vars]


@keras_export("keras.ops.while_loop")
def while_loop(
    cond,
    body,
    loop_vars,
    maximum_iterations=None,
):
    """While loop implementation.

    Args:
        cond: A callable that represents the termination condition of the loop.
            Must accept a `loop_vars` like structure as an argument. If
            `loop_vars` is a tuple or list, each element of `loop_vars` will be
            passed positionally to the callable.
        body: A callable that represents the loop body. Must accept a
            `loop_vars` like structure as an argument, and return update value
            with the same structure. If `loop_vars` is a tuple or list, each
            element of `loop_vars` will be passed positionally to the callable.
        loop_vars: An arbitrary nested structure of tensor state to persist
            across loop iterations.
        maximum_iterations: Optional maximum number of iterations of the while
            loop to run. If provided, the `cond` output is AND-ed with an
            additional condition ensuring the number of iterations executed is
            no greater than `maximum_iterations`.

    Returns:
        A list/tuple of tensors, has the same shape and dtype as `inputs`.

    Examples:

    >>> i = 0
    >>> cond = lambda i: i < 10
    >>> body = lambda i: i + 1
    >>> keras.ops.while_loop(cond, body, i)
    10

    >>> x, y = 0, 1
    >>> cond = lambda x, y: x < 10
    >>> body = lambda x, y: (x + 1, y + 1)
    >>> keras.ops.while_loop(cond, body, (x, y))
    10, 11
    """
    return backend.core.while_loop(
        cond,
        body,
        loop_vars,
        maximum_iterations=maximum_iterations,
    )


class StopGradient(Operation):
    def __init__(self):
        super().__init__()

    def call(self, variable):
        return backend.core.stop_gradient(variable)

    def compute_output_spec(self, variable):
        return KerasTensor(variable.shape, dtype=variable.dtype)


@keras_export("keras.ops.stop_gradient")
def stop_gradient(variable):
    """Stops gradient computation.

    Args:
        variable: A tensor variable for which the gradient
            computation is to be disabled.

    Returns:
        The variable with gradient computation disabled.

    Examples:

    >>> var = keras.backend.convert_to_tensor(
    ...     [1., 2., 3.],
    ...     dtype="float32"
    ... )
    >>> var = keras.ops.stop_gradient(var)
    """
    return backend.core.stop_gradient(variable)


class ForiLoop(Operation):
    def __init__(self, lower, upper, body_fun):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.body_fun = body_fun

    def call(self, init_val):
        return backend.core.fori_loop(
            self.lower,
            self.upper,
            self.body_fun,
            init_val,
        )

    def compute_output_spec(self, init_val):
        return KerasTensor(init_val.shape, dtype=init_val.dtype)


@keras_export("keras.ops.fori_loop")
def fori_loop(lower, upper, body_fun, init_val):
    """For loop implementation.

    Args:
        lower: The initial value of the loop variable.
        upper: The upper bound of the loop variable.
        body_fun: A callable that represents the loop body. Must take two
            arguments: the loop variable and the loop state. The loop state
            should be updated and returned by this function.
        init_val: The initial value of the loop state.

    Returns:
        The final state after the loop.

    Example:

    >>> lower = 0
    >>> upper = 10
    >>> body_fun = lambda i, s: (i + 1, s + i)
    >>> init_val = 0
    >>> keras.ops.fori_loop(lower, upper, body_fun, init_val)
    45
    """
    if any_symbolic_tensors((lower, upper, init_val)):
        return ForiLoop(lower, upper, body_fun).symbolic_call(init_val)
    return backend.core.fori_loop(lower, upper, body_fun, init_val)


class Unstack(Operation):
    def __init__(self, num=None, axis=0):
        super().__init__()
        self.num = num
        self.axis = axis

    def call(self, x):
        return backend.core.unstack(x, self.num, self.axis)

    def compute_output_spec(self, x):
        axis = self.axis
        if axis < 0:
            axis = len(x.shape) + axis
        output_shapes = x.shape[:axis] + x.shape[axis + 1 :]
        num = self.num
        if num is None:
            num = x.shape[axis]
        if num is None:
            raise ValueError(
                "Cannot infer argument `num` from shape "
                f"{x.shape}. Either provide a tensor with a "
                "concrete shape in the `axis` dimension or "
                "explicitly pass the `num` argument."
            )
        output = [
            KerasTensor(shape=output_shapes, dtype=x.dtype) for _ in range(num)
        ]
        return output


@keras_export("keras.ops.unstack")
def unstack(x, num=None, axis=0):
    """Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.

    Args:
        x: The input tensor.
        num: The length of the dimension axis. Automatically inferred
            if `None`.
        axis: The axis along which to unpack.

    Returns:
        A list of tensors unpacked along the given axis.

    Example:

    >>> x = keras.ops.array([[1, 2], [3, 4]])
    >>> keras.ops.unstack(x, axis=0)
    [array([1, 2]), array([3, 4])]
    """
    if any_symbolic_tensors((x,)):
        return Unstack(num, axis).symbolic_call(x)
    return backend.core.unstack(x, num=num, axis=axis)


@keras_export("keras.ops.shape")
def shape(x):
    """Gets the shape of the tensor input.

    Note: On the TensorFlow backend, when `x` is a `tf.Tensor` with dynamic
    shape, dimensions which are dynamic in the context of a compiled function
    will have a `tf.Tensor` value instead of a static integer value.

    Args:
        x: A tensor. This function will try to access the `shape` attribute of
            the input tensor.

    Returns:
        A tuple of integers or None values, indicating the shape of the input
            tensor.

    Example:

    >>> x = keras.zeros((8, 12))
    >>> keras.ops.shape(x)
    (8, 12)
    """
    if any_symbolic_tensors((x,)):
        return x.shape
    return backend.core.shape(x)


class Cast(Operation):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = backend.standardize_dtype(dtype)

    def call(self, x):
        return backend.core.cast(x, self.dtype)

    def compute_output_spec(self, x):
        return backend.KerasTensor(shape=x.shape, dtype=self.dtype)


@keras_export("keras.ops.cast")
def cast(x, dtype):
    """Cast a tensor to the desired dtype.

    Args:
        x: A tensor or variable.
        dtype: The target type.

    Returns:
        A tensor of the specified `dtype`.

    Example:

    >>> x = keras.ops.arange(4)
    >>> x = keras.ops.cast(x, dtype="float16")
    """
    dtype = backend.standardize_dtype(dtype)

    if any_symbolic_tensors((x,)):
        return Cast(dtype=dtype)(x)
    return backend.core.cast(x, dtype)


@keras_export("keras.ops.convert_to_tensor")
def convert_to_tensor(x, dtype=None, sparse=None):
    """Convert a NumPy array to a tensor.

    Args:
        x: A NumPy array.
        dtype: The target type.
        sparse: Whether to keep sparse tensors. `False` will cause sparse
            tensors to be densified. The default value of `None` means that
            sparse tensors are kept only if the backend supports them.

    Returns:
        A tensor of the specified `dtype`.

    Example:

    >>> x = np.array([1, 2, 3])
    >>> y = keras.ops.convert_to_tensor(x)
    """
    return backend.convert_to_tensor(x, dtype=dtype, sparse=sparse)


@keras_export("keras.ops.convert_to_numpy")
def convert_to_numpy(x):
    """Convert a tensor to a NumPy array.

    Args:
        x: A tensor.

    Returns:
        A NumPy array.
    """
    if any_symbolic_tensors((x,)):
        # This will raise a `ValueError` defined in the `KerasTensor` class.
        # We trigger it rather than duplicate it here.
        return np.array(x)
    return backend.convert_to_numpy(x)


class Cond(Operation):
    @traceback_utils.filter_traceback
    def __call__(self, *args, **kwargs):
        def call_fn(*args, **kwargs):
            if any_symbolic_tensors(args, kwargs):
                return self.symbolic_call(*args, **kwargs)
            else:
                return self.call(*args, **kwargs)

        if traceback_utils.is_traceback_filtering_enabled():
            # Wrap self.call to provide helpful info in case of exception
            call_fn = traceback_utils.inject_argument_info_in_traceback(
                call_fn,
                object_name=(f"{self.__class__.__name__}.call()"),
            )
            return call_fn(*args, **kwargs)

        # Plain flow.
        return call_fn(*args, **kwargs)

    def call(self, pred, true_fn, false_fn):
        return backend.core.cond(pred, true_fn, false_fn)

    def compute_output_spec(self, pred, true_fn, false_fn):
        true_fn_spec = backend.compute_output_spec(true_fn)
        false_fn_spec = backend.compute_output_spec(false_fn)
        if not self._check_output_spec(true_fn_spec, false_fn_spec):
            raise ValueError(
                "`true_fn` and `false_fn` should return outputs "
                "of the same kind (struct, dtype and shape). "
                f"Got {true_fn_spec} and {false_fn_spec} instead."
            )
        return true_fn_spec

    def _check_output_spec(self, true_fn_spec, false_fn_spec):
        try:
            tree.assert_same_structure(true_fn_spec, false_fn_spec)
        except:
            return False

        def check_leaf(t_spec, f_spec):
            if t_spec is None or f_spec is None:
                return t_spec is None and f_spec is None
            return t_spec.shape == f_spec.shape and t_spec.dtype == f_spec.dtype

        same = tree.map_structure(check_leaf, true_fn_spec, false_fn_spec)
        return all(tree.flatten(same))


@keras_export("keras.ops.cond")
def cond(pred, true_fn, false_fn):
    """Conditionally applies `true_fn` or `false_fn`.

    Args:
        pred: Boolean scalar type
        true_fn: Callable returning the output for the `pred == True` case.
        false_fn: Callable returning the output for the `pred == False` case.

    Returns:
        The output of either `true_fn` or `false_fn` depending on pred.
    """
    return Cond()(pred, true_fn, false_fn)


# TODO: also create an Op subclass VectorizedMap.
@keras_export("keras.ops.vectorized_map")
def vectorized_map(function, elements):
    """Parallel map of `function` on axis 0 of tensor(s) `elements`.

    Schematically, `vectorized_map` implements the following,
    in the case of a single tensor input `elements`:

    ```python
    def vectorized_map(function, elements)
        outputs = []
        for e in elements:
            outputs.append(function(e))
        return stack(outputs)
    ```

    In the case of an iterable of tensors `elements`,
    it implements the following:

    ```python
    def vectorized_map(function, elements)
        batch_size = elements[0].shape[0]
        outputs = []
        for index in range(batch_size):
            outputs.append(function([e[index] for e in elements]))
        return np.stack(outputs)
    ```

    In this case, `function` is expected to take as input
    a single list of tensor arguments.
    """
    return backend.core.vectorized_map(function, elements)


@keras_export("keras.ops.is_tensor")
def is_tensor(x):
    """Check whether the given object is a tensor.

    Note: This checks for backend specific tensors so passing a TensorFlow
    tensor would return `False` if your backend is PyTorch or JAX.

    Args:
        x: A variable.

    Returns:
        `True` if `x` is a tensor, otherwise `False`.
    """
    return backend.core.is_tensor(x)


@keras_export("keras.ops.custom_gradient")
def custom_gradient(f):
    """Decorator to define a function with a custom gradient.

    This decorator allows fine grained control over the gradients of a sequence
    for operations. This may be useful for multiple reasons, including providing
    a more efficient or numerically stable gradient for a sequence of
    operations.

    Args:
        f: Function `f(*args)` that returns a tuple
            `(output, grad_fn)`, where:
            - `args` is a sequence of (nested structures of) tensor inputs to
                the function.
            - `output` is a (nested structure of) tensor outputs of applying
                operations in `forward_fn` to `args`.
            - `grad_fn` is a function with the signature `grad_fn(*args,
                upstream)` which returns a tuple of tensors the same size as
                (flattened) `args`: the derivatives of tensors in `output` with
                respect to the tensors in `args`. `upstream` is a tensor or
                sequence of tensors holding the initial value gradients for each
                tensor in `output`.

    Returns:
        A function `h(*args)` which returns the same value as
        `f(*args)[0]` and whose gradient is determined by
        `f(*args)[1]`.


    Examples:

    1. Backend-agnostic example.

    ```python
    @ops.custom_gradient
    def log1pexp(x):
        e = ops.exp(x)

        def grad(*args, upstream=None):
            if upstream is None:
                (upstream,) = args
            return ops.multiply(upstream, 1.0 - 1.0 / ops.add(1, e))

        return ops.log(1 + e), grad
    ```

    Note that the grad function that returns gradient computation
    requires `args` as well as an `upstream` keyword argument, depending
    on the backend being set. With the JAX and TensorFlow backends,
    it requires only one argument, whereas it might use the `upstream`
    argument in the case of the PyTorch backend.

    When working with TensorFlow/JAX backend, `grad(upstream)`
    is sufficient. With PyTorch, the `grad` function requires
    `*args` as well as `upstream`, e.g. `def grad(*args, upstream)`.
    Follow the previous example to use `@ops.custom_gradient` in
    a way that is compatible with all backends.

    2. Here's JAX & TensorFlow-specific example:

    ```python
    @ops.custom_gradient
    def log1pexp(x):
        e = ops.exp(x)
        def grad(upstream):
            return ops.multiply(upstream, 1.0 - 1.0 / ops.add(1, e))
        return ops.log(1 + e), grad
    ```

    3. Lastly, here's a PyTorch-specific example,
    using `*args` & `upstream`:

    ```python
    @ops.custom_gradient
    def log1pexp(x):
        e = ops.exp(x)
        def grad(*args, upstream):
            return ops.multiply(upstream, 1.0 - 1.0 / ops.add(1, e))
        return ops.log(1 + e), grad
    ```
    """
    return backend.core.custom_gradient(f)
