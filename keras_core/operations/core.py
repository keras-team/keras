"""
scatter
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
    """Update inputs by scattering updates at indices.

    At a high level, this operation does `inputs[indices]=updates`. In details,
    assume `inputs` is a tensor of shape `[D0, D1, ..., Dn]`, there are 2 main
    usages of `scatter_update`.

    - `indices` is a 2D tensor of shape `[num_updates, n]`, where `num_updates`
        is the number of updates to perform, and `updates` is a 1D tensor of
        shape `[num_updates]`. For example, if `inputs = np.zeros([4, 4, 4])`,
        and we want to update `inputs[1, 2, 3]` and `inputs[0, 1, 3]` as 1, then
        we can use

        ```
        inputs = np.zeros([4, 4, 4])
        indices = [[1, 2, 3], [0, 1, 3]]
        updates = np.array([1., 1.])
        inputs = keras_core.operations.scatter_update(inputs, indices, updates)
        ```
    - `indices` is a 2D tensor of shape `[num_updates, k]`, where `num_updates`
        is the number of updates to perform, and `k` (`k < n`) is the size of
        each index in `indices`. `updates` is a `n-k`-D tensor of shape
        `[num_updates, inputs.shape[k:]]`. For example, if
        `inputs = np.zeros([4, 4, 4])`, and we want to update `inputs[1, 2, :]`
        and `inputs[2, 3, :]` as `[1, 1, 1, 1]`, then `indices` would have shape
        `[num_updates, 2]` (`k=2`), and `updates` would have shape
        `[num_updates, 4]` (`inputs.shape[2:]=4`). See the code below:

        ```
        inputs = np.zeros([4, 4, 4])
        indices = [[1, 2], [2, 3]]
        updates = np.array([[1., 1., 1, 1,], [1., 1., 1, 1,])
        inputs = keras_core.operations.scatter_update(inputs, indices, updates)
        ```

    Args:
        inputs: A tensor, the tensor to be updated.
        indices: A tensor or list/tuple of shape `[N, inputs.ndims]`, specifying
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
    `inputs[start_indices: start_indices + updates.shape] = updates`. In
    details, assume inputs is a tensor of shape `[D0, D1, ..., Dn]`,
    `start_indices` must be a list/tuple of n integers, specifying the starting
    indices. `updates` must have the same rank as `inputs`, and the size of each
    dim must not exceed `Di - start_indices[i]`. For example, if we have 2D
    inputs `inputs=np.zeros([5, 5])`, and we want to update the intersection of
    last 2 rows and last 2 columns as 1, i.e.,
    `inputs[3:, 3:] = np.ones([2, 2])`, then we can use the code below:

    ```
    inputs = np.zeros([5, 5])
    start_indices = [3, 3]
    updates = np.ones([2, 2])
    inputs = keras_core.operations.block_update(inputs, start_indices, updates)
    ```

    Args:
        inputs: A tensor, the tensor to be updated.
        start_indices: A list/tuple of shape `[inputs.ndims]`, specifying
            the starting indices for updating.
        updates: A tensor, the new values to be put to `inputs` at `indices`.
            `updates` must have the same rank as `inputs`.

    Returns:
        A tensor, has the same shape and dtype as `inputs`.
    """
    if any_symbolic_tensors((inputs, start_indices, updates)):
        return BlockUpdate().symbolic_call(inputs, start_indices, updates)
    return backend.core.block_update(inputs, start_indices, updates)
