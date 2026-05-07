import functools

import jax.experimental.sparse as jax_sparse
import jax.numpy as jnp

from keras.src.utils import jax_utils


def axis_shape_dims_for_broadcast_in_dim(axis, input_shape, insert_dims):
    """Turn the `axis` argument to the arguments needed by `broadcast_in_dim`.

    Args:
        axis: single int or a tuple of ints for the axis argument. The list of
          dimensions to reduce or insert.
        input_shape: the shape of the input as a tuple ints.
        insert_dims: `False` turns dimensions in `axis` to 1s (use case:
          reduction along `axis` with `keep_dims=True`). `True`, inserts 1s
          according to `axis` (use case: `expand_dims`).
    Returns:
        A tuple of three lists
        - The canonical value for `axis`: always a list, negative values have
          been resolved and values are sorted in ascending order.
        - The output shape: `input_shape` with 1s at the indices in `axis`, for
          use as the `shape` argument of `broadcast_in_dim`.
        - The broadcast dimensions: list of dimensions not in `axis`, for use as
          the `broadcast_dimensions` argument of `broadcast_in_dim`.
    """
    if axis is None:
        raise ValueError("Received `None` value for `axis`")
    if isinstance(axis, int):
        axis = (axis,)
    # Check uniqueness.
    if len(set(axis)) != len(axis):
        raise ValueError(f"Repeated axis in `axis`: {axis}")
    result_dims = len(input_shape)
    if insert_dims:
        result_dims += len(axis)

    # Resolve negative values.
    canonical_axis = []
    for a in axis:
        if not -result_dims <= a < result_dims:
            raise ValueError(
                f"In `axis`, axis {a} is out of bounds for array "
                f"of dimension {result_dims}"
            )
        if a < 0:
            a = a + result_dims
        canonical_axis.append(a)

    # Check uniqueness again after resolving negative values.
    if len(set(canonical_axis)) != len(canonical_axis):
        raise ValueError(f"Repeated axis in `axis`: {canonical_axis}")
    canonical_axis = sorted(canonical_axis)

    # Compute output shape.
    output_shape = list(input_shape)
    for i in canonical_axis:
        if insert_dims:
            output_shape.insert(i, 1)
        else:
            output_shape[i] = 1
    broadcast_dims = [i for i in range(result_dims) if i not in canonical_axis]
    return canonical_axis, output_shape, broadcast_dims


def bcoo_add_indices(x1, x2, sum_duplicates):
    """Add the indices of `x2` to `x1` with zero values.

    Args:
        x1: `BCOO` tensor to add indices to.
        x2: `BCOO` tensor to take the indices to add to x1.
        sum_duplicates: if `True` calls `bcoo_sum_duplicates` on the output.
    Returns:
        a `BCOO` tensor equal to `x1` but with extra zeros at indices in `x2`
        that were missing in `x1`.
    """
    x2_zeros = jnp.zeros(x2.data.shape, x1.data.dtype)
    concat_axis = len(x1.indices.shape) - 2
    output_indices = jnp.concatenate([x1.indices, x2.indices], axis=concat_axis)
    output_data = jnp.concatenate([x1.data, x2_zeros], axis=concat_axis)
    output = jax_sparse.BCOO((output_data, output_indices), shape=x1.shape)
    if sum_duplicates:
        output = jax_sparse.bcoo_sum_duplicates(output)
    return output


def densifying_unary(func):
    """Decorator to add support for `JAXSparse` tensors (including `BCOO`) to a
    non-zero-preserving element-wise unary operator.

    There are requirements on the operator for this decorator to work correctly:

    - The operator must be element-wise
    - The operator must be unary (one input tensor and one output tensor)
    - The operator must return a tensor of the same shape.

    Additional arguments to the function (besides the input tensor) are
    supported. The returned result is a dense tensor.

    Args:
        func: The unary operator to wrap.
    Returns:
        Wrapped function that supports `JAXSparse` tensors.
    """

    @functools.wraps(func)
    def sparse_wrapper(x, *args, **kwargs):
        if isinstance(x, jax_sparse.JAXSparse):
            x = x.todense()
        return func(x, *args, **kwargs)

    return sparse_wrapper


def elementwise_unary(linear):
    """Decorator to add support for `BCOO` sparse tensors to a zero-preserving
    element-wise unary operator.

    There are requirements on the operator for this decorator to work correctly:

    - The operator must be element-wise
    - The operator must be unary (one input tensor and one output tensor)
    - The operator must return a tensor of the same shape, and if it is a
      `BCOO` tensor, the indices of the result must be the same. Therefore:
        - Reduction operations are not supported (e.g. `mean`).
        - Operations for which the result may be dense (e.g. `reciprocal`), or
          the sparse indices depend on the inputs are not supported (e.g.
          `clip`). This implies that `func(0)` must be 0.

    Additional arguments to the function (besides the input tensor) are
    supported as long as they cannot change the indices of the result. For
    instance,`round` is supported, but `clip` is not supported as
    `clip(x, 1.0, 2.0)` would always return a dense tensor.

    Note that if an input sparse tensor contains zero values, the indices and
    the zero values are preserved.

    Args:
        linear: if `True`, means that the operation is such that
            `op(a + b) == op(a) + op(b)`.
    Returns:
        Wrapped function that supports `BCOO` sparse tensors.
    """

    def wrap_elementwise_unary(func):
        @functools.wraps(func)
        def sparse_wrapper(x, *args, **kwargs):
            if isinstance(x, jax_sparse.BCOO):
                if not linear and not x.unique_indices:
                    x = jax_sparse.bcoo_sum_duplicates(x)
                return jax_sparse.BCOO(
                    (func(x.data, *args, **kwargs), x.indices), shape=x.shape
                )
            else:
                return func(x, *args, **kwargs)

        return sparse_wrapper

    return wrap_elementwise_unary


def elementwise_binary_union(linear, use_sparsify):
    """Decorator to add support for `JAXSparse` tensors (including `BCOO`) to an
    element-wise binary operator such that the indices present in the result are
    are the union of the indices in the two operand.

    The primary use case for this is the `add` and `subtract` operators.

    There are requirements on the operator for this decorator to work correctly:

    - The operator must be element-wise.
    - The operator must be binary (two input tensors and one output tensor).
    - Both inputs must be of the same shape or one input must be a scalar.
    - The output must be of the same shape as the (non scalar) inputs.
    - The indices of the output must be the union of the indices of the inputs.
      This implies that func(0, 0) must be 0. As a result, if one operand is
      dense or a scalar, then the result will be dense.

    Additional arguments to the function (besides the input tensors) are not
    supported.

    Note that if the result of the operation is zero at some indices, including
    because the operands were zero at these indices, the zeros and indices are
    preserved.

    The `BCOO` format is the only supported one in all cases. Other formats are
    not supported when `use_sparsify` is `False`.

    Args:
        use_sparsify: indicates that the JAX `sparsify` transform supports this
            operation.
        linear: if `True`, mean that the operation is such that
            `op(a + b, c) == op(a, c) + op(b, c)` and
            `op(a, c + d) == op(a, c) + op(a, d)`.
    Returns:
        Wrapped function that supports `JAXSparse`.
    """

    def wrap_elementwise_binary_union(func):
        sparse_func = jax_sparse.sparsify(func) if use_sparsify else None

        @functools.wraps(func)
        def sparse_wrapper(x1, x2):
            if isinstance(x1, jax_sparse.JAXSparse):
                if isinstance(x2, jax_sparse.JAXSparse):
                    # x1 and x2 are sparse.
                    # The way we use `sparsify` it cannot know that the indices
                    # are the same, so we optimize this case here.
                    if (
                        x1.indices is x2.indices
                        and isinstance(x1, jax_sparse.BCOO)
                        and isinstance(x2, jax_sparse.BCOO)
                    ):
                        if not linear and not x1.unique_indices:
                            x1 = jax_sparse.bcoo_sum_duplicates(x1)
                            x2 = jax_sparse.bcoo_sum_duplicates(x2)
                        return jax_sparse.BCOO(
                            (func(x1.data, x2.data), x1.indices),
                            shape=x1.shape,
                            indices_sorted=x1.indices_sorted,
                            unique_indices=x1.unique_indices,
                        )
                    elif use_sparsify:
                        return sparse_func(x1, x2)
                    elif isinstance(x1, jax_sparse.BCOO) and isinstance(
                        x2, jax_sparse.BCOO
                    ):
                        x1 = bcoo_add_indices(x1, x2, sum_duplicates=not linear)
                        x2 = bcoo_add_indices(x2, x1, sum_duplicates=not linear)
                        return jax_sparse.BCOO(
                            (func(x1.data, x2.data), x1.indices),
                            shape=x1.shape,
                            indices_sorted=True,
                            unique_indices=True,
                        )
                    else:
                        ValueError(
                            "Unsupported sparse format: "
                            f"{x1.__class__} and {x2.__class__}"
                        )
                else:
                    # x1 is sparse, x2 is dense, densify x2.
                    x1 = x1.todense()
            elif isinstance(x2, jax_sparse.JAXSparse):
                # x1 is dense, x2 is sparse, densify x2.
                x2 = x2.todense()
            return func(x1, x2)

        return sparse_wrapper

    return wrap_elementwise_binary_union


def elementwise_division(func):
    """Decorator to add support for `BCOO` sparse tensors to element-wise binary
    division and related operators.

    This decorator is designed for operations related to the division of two
    two operands (e.g. `divide`). It accepts `BCOO` tensors for both the
    dividend and the divisor, but handles them differently based on whether they
    are the dividend or the divisor.

    - If the divisor is sparse, it is densified and the result is dense because
      the result contains Inf or Nan outside of the indices of the dividend.
    - If the dividend is sparse and the divisor is dense, it finds occurrences
      of zeros and NaNs in the divisor. The result may therefore have more
      indices than there were in the dividend to return correct values where the
      divisor was zero or NaN.
    - If the dividend is sparse and the divisor is a scalar, it does the
      division element-wise. Note that the result is incorrectly sparse if the
      scalar divisor is zero.

    Args:
        func: The function to wrap.
    Returns:
        Wrapped function that supports `BCOO` sparse tensors.
    """
    sparse_func = jax_sparse.sparsify(func)

    @functools.wraps(func)
    def sparse_wrapper(x1, x2):
        if isinstance(x1, jax_sparse.JAXSparse):
            if isinstance(x2, jax_sparse.JAXSparse):
                # x1 is sparse and x2 is sparse.
                # Divisor is sparse, meaning we're doing divisions by zero
                # outside of x2.indices, so the result is dense. Densify both.
                x1 = x1.todense()
                x2 = x2.todense()
            elif isinstance(x1, jax_sparse.BCOO):
                if not hasattr(x2, "shape") or len(x2.shape) == 0:
                    # x1 is sparse BCOO, x2 is scalar, apply func element-wise.
                    return jax_sparse.BCOO(
                        (func(x1.data, x2), x1.indices),
                        shape=x1.shape,
                        indices_sorted=x1.indices_sorted,
                        unique_indices=x1.unique_indices,
                    )
                else:
                    # x1 is sparse BCOO, x2 is dense.
                    if not jax_utils.is_in_jax_tracing_scope(x2):
                        # Find zeros and nans in x2 and add indices to x1.
                        # 1. Create a dense mask for zeros and nans.
                        x2_zeros_and_nans = jnp.equal(x2, 0)
                        if not jnp.issubdtype(x2.dtype, jnp.integer):
                            x2_zeros_and_nans = jnp.logical_or(
                                x2_zeros_and_nans, jnp.isnan(x2)
                            )
                        # 2. Make it a BCOO of True values.
                        x2_zeros_and_nans = jax_sparse.bcoo_fromdense(
                            x2_zeros_and_nans,
                            n_batch=x1.n_batch,
                            n_dense=x1.n_dense,
                            index_dtype=x1.indices.dtype,
                        )
                        # 3. Add the indices to x1.
                        x1 = bcoo_add_indices(
                            x1, x2_zeros_and_nans, sum_duplicates=True
                        )
                    return sparse_func(x1, x2)
            else:
                raise ValueError(f"Unsupported sparse format: {x1.__class__}")
        elif isinstance(x2, jax_sparse.JAXSparse):
            # x1 is dense, x2 is sparse, densify x2
            x2 = x2.todense()
        return func(x1, x2)

    return sparse_wrapper
