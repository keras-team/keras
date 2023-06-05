"""
scatter
scatter_update
block_update
while_loop
"""

from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.backend import KerasTensor
from keras_core.backend import any_symbolic_tensors
from keras_core.operations.operation import Operation


class Scatter(Operation):
    def call(self, indices, values, shape):
        return backend.core.scatter(indices, values, shape)

    def compute_output_spec(self, indices, values, shape):
        return KerasTensor(shape, dtype=values.dtype)


@keras_core_export("keras_core.operations.scatter")
def scatter(indices, values, shape):
    if any_symbolic_tensors((indices, values, shape)):
        return Scatter().symbolic_call(indices, values, shape)
    return backend.core.scatter(indices, values, shape)


class ScatterUpdate(Operation):
    def call(self, inputs, indices, updates):
        return backend.core.scatter_update(inputs, indices, updates)

    def compute_output_spec(self, inputs, indices, updates):
        return KerasTensor(inputs.shape, dtype=inputs.dtype)


@keras_core_export("keras_core.operations.scatter_update")
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
    inputs = keras_core.operations.scatter_update(inputs, indices, updates)
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
    inputs = keras_core.operations.scatter_update(inputs, indices, updates)
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


class BlockUpdate(Operation):
    def call(self, inputs, start_indices, updates):
        return backend.core.block_update(inputs, start_indices, updates)

    def compute_output_spec(self, inputs, start_indices, updates):
        return KerasTensor(inputs.shape, dtype=inputs.dtype)


@keras_core_export("keras_core.operations.block_update")
def block_update(inputs, start_indices, updates):
    """Update inputs block.

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
    inputs = keras_core.operations.block_update(inputs, start_indices, updates)
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
        return BlockUpdate().symbolic_call(inputs, start_indices, updates)
    return backend.core.block_update(inputs, start_indices, updates)


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


@keras_core_export("keras_core.operations.while_loop")
def while_loop(
    cond,
    body,
    loop_vars,
    maximum_iterations=None,
):
    """While loop implemetation.

    Args:
        cond: A callable that represents the termination condition of the loop.
            Must have the same number of args as `loop_vars`, and return a bool.
        body: A callable that represents the loop body. Must have the same
            number of args as `loop_vars`, and return a list/tuple of the same
            length, shape and dtype as `loop_vars`.
        loop_vars: A list/tuple of tensors, the loop variables.
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
    >>> keras_core.operations.while_loop(cond, body, [i])[0]
    10
    """

    return backend.core.while_loop(
        cond,
        body,
        loop_vars,
        maximum_iterations=maximum_iterations,
    )
