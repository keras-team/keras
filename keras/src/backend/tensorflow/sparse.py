import functools

import tensorflow as tf

ones_bool = functools.partial(tf.ones, dtype=tf.bool)
ones_int8 = functools.partial(tf.ones, dtype=tf.int8)
zeros_int8 = functools.partial(tf.zeros, dtype=tf.int8)
ones_like_int8 = functools.partial(tf.ones_like, dtype=tf.int8)
zeros_like_int8 = functools.partial(tf.zeros_like, dtype=tf.int8)


def sparse_to_dense(x, default_value=None):
    x_shape = x.shape
    if x_shape.rank == 0:
        # Workaround for bug on GPU when sparse tensor represents a scalar.
        if x.values.shape[0] == 0:
            return tf.constant(default_value, dtype=x.dtype)
        else:
            return tf.reshape(x.values, ())
    x = tf.sparse.to_dense(x, default_value=default_value)
    x.set_shape(x_shape)
    return x


def sparse_with_values(x, values):
    x_shape = x.shape
    x = tf.SparseTensor(x.indices, values, x.dense_shape)
    x.set_shape(x_shape)
    return x


def broadcast_scalar_to_sparse_shape(scalar, sparse):
    output = tf.broadcast_to(scalar, sparse.dense_shape)
    output.set_shape(sparse.shape)
    return output


def sparse_subtract(x1, x2):
    """Subtraction for `tf.SparseTensor`s.

    Either `x1` or `x2` or both can be `tf.SparseTensor`s.

    Args:
        x1: fist tensor to add.
        x2: second tensor to add.
    Returns:
        The sum of `x1` and `x2`, which is a `tf.SparseTensor` if and only if
        both `x1` or `x2` are `tf.SparseTensor`s.
    """
    if isinstance(x2, tf.SparseTensor):
        return tf.sparse.add(x1, tf.sparse.map_values(tf.negative, x2))
    else:
        return tf.sparse.add(x1, tf.negative(x2))


def sparse_union_indices_and_values(x1, x2_indices, x2_values=None):
    """Compute the indices for the union of the indices of the provided
    `tf.SparseTensor`s and another set of indices and return the modified values
    for these indices.

    Args:
        x: a `tf.SparseTensor`.
        indices: another set of indices in the `tf.SparseTensor` format.
    Returns: A tuple containing:
        - the indices for the union
        - `x1` values for the union indices (some zeros were added)
        - `x2` values for the union indices (some zeros were added) or `None` if
          `x2_values` was `None`.
    """
    # Add zeros at the x2 indices to x1 to create the union.
    zeros2 = tf.SparseTensor(
        x2_indices,
        tf.zeros((tf.shape(x2_indices)[0],), x1.values.dtype),
        x1.dense_shape,
    )
    x1_for_union = tf.sparse.add(x1, zeros2)
    if x2_values is not None:
        # Add zeros at the x1 indices to x2 to create the union.
        x2 = tf.SparseTensor(x2_indices, x2_values, x1.dense_shape)
        zeros1 = tf.sparse.map_values(tf.zeros_like, x1)
        x2_for_union = tf.sparse.add(x2, zeros1)
        return x1_for_union.indices, x1_for_union.values, x2_for_union.values
    else:
        return x1_for_union.indices, x1_for_union.values, None


def indexed_slices_union_indices_and_values(x1, x2_indices, x2_values=None):
    """Compute the indices for the union of two `tf.IndexedSlices` and modify
    the values for these indices.

    Args:
        x1: the first `tf.IndexedSlices`.
        x2_indices: the indices for the second `tf.IndexedSlices`.
        x2_value: (optional) the values for the second `tf.IndexedSlices`.
    Returns: A tuple containing:
        - the indices for the union
        - `x1` values for the union indices (some zeros were added)
        - `x2` values for the union indices (some zeros were added) or `None` if
          `x2_values` was `None`.
    """
    # Compute the union of the indices by doing a logical or between the one-hot
    # encoded indices for x1 and x2.
    dim_0 = x1.dense_shape[0]
    x1_indices_expanded = tf.expand_dims(x1.indices, axis=1)
    x2_indices_expanded = tf.expand_dims(x2_indices, axis=1)
    x1_indices_count = tf.shape(x1_indices_expanded)[0]
    x2_indices_count = tf.shape(x2_indices_expanded)[0]
    x1_indices_one_hot = tf.scatter_nd(
        x1_indices_expanded,
        ones_bool((x1_indices_count,)),
        (dim_0,),
    )
    x2_indices_one_hot = tf.scatter_nd(
        x2_indices_expanded,
        ones_bool((x2_indices_count,)),
        (dim_0,),
    )
    union_indices = tf.squeeze(
        tf.where(tf.math.logical_or(x1_indices_one_hot, x2_indices_one_hot)),
        axis=-1,
    )
    union_indices_count = tf.shape(union_indices)[0]

    # Re-gather the values with extra zeros added at indices that are part of
    # the union but were not in x1 or x2.
    def values_for_union(indices_expanded, indices_count, values):
        indices_indices = tf.scatter_nd(
            indices_expanded,
            tf.range(1, indices_count + 1),
            (dim_0,),
        )
        to_union_indices = tf.gather(indices_indices, union_indices)
        values_with_leading_zeros = tf.concat(
            [tf.zeros_like(values[0:1]), values], axis=0
        )
        return tf.gather(values_with_leading_zeros, to_union_indices)

    # Only recompute values if some indices were added.
    x1_values_for_union_indices = tf.cond(
        tf.equal(x1_indices_count, union_indices_count),
        lambda: x1.values,
        lambda: values_for_union(
            x1_indices_expanded, x1_indices_count, x1.values
        ),
    )
    if x2_values is not None:
        x2_values_for_union_indices = tf.cond(
            tf.equal(x2_indices_count, union_indices_count),
            lambda: x2_values,
            lambda: values_for_union(
                x2_indices_expanded, x2_indices_count, x2_values
            ),
        )
    else:
        x2_values_for_union_indices = None

    return (
        union_indices,
        x1_values_for_union_indices,
        x2_values_for_union_indices,
    )


def sparse_intersection_indices_and_values(x1, x2):
    """Compute the indices for the intersection of two `tf.SparseTensor`s and
    modify the values for these indices.

    Args:
        x1: the first `tf.SparseTensor`.
        x2: the second `tf.SparseTensor`.
    Returns: A tuple containing:
        - the indices for the intersection
        - `x1` values for the intersection indices (some values were removed)
        - `x2` values for the intersection indices (some values were removed)
    """
    # Compute the intersection of indices in the form of a sparse
    # tensor containing ones as values.
    ones1 = tf.sparse.map_values(ones_like_int8, x1)
    ones2 = tf.sparse.map_values(ones_like_int8, x2)
    # tf.sets.intersection ignores the last dimension when, so we
    # need to add a dummy extra dimension and then remove it.
    intersection_extra_dim = tf.sets.intersection(
        tf.sparse.expand_dims(ones1, axis=-1),
        tf.sparse.expand_dims(ones2, axis=-1),
    )

    def empty_intersection():
        return (
            tf.zeros((0, x1.shape.rank), dtype=tf.int64),
            tf.zeros((0,), dtype=x1.values.dtype),
            tf.zeros((0,), dtype=x2.values.dtype),
        )

    def non_empty_intersection():
        intersection = tf.sparse.reshape(intersection_extra_dim, x1.dense_shape)

        # Compute the masks to remove indices in x1 and x2 that are not
        # in the intersection, then trim x1 and x2.
        zeros1 = tf.sparse.map_values(zeros_like_int8, x1)
        zeros2 = tf.sparse.map_values(zeros_like_int8, x2)
        mask1 = tf.sparse.add(zeros1, intersection)
        mask2 = tf.sparse.add(zeros2, intersection)
        return (
            intersection.indices,
            tf.sparse.retain(x1, tf.cast(mask1.values, tf.bool)).values,
            tf.sparse.retain(x2, tf.cast(mask2.values, tf.bool)).values,
        )

    return tf.cond(
        tf.equal(tf.size(intersection_extra_dim), 0),
        empty_intersection,
        non_empty_intersection,
    )


def indexed_slices_intersection_indices_and_values(x1, x2):
    """Compute the indices for the intersection of two `tf.IndexedSlices` and
    modify the values for these indices.

    Args:
        x1: the first `tf.IndexedSlices`.
        x2: the second `tf.IndexedSlices`.
    Returns: A tuple containing:
        - the indices for the intersection
        - `x1` values for the intersection indices (some values were removed)
        - `x2` values for the intersection indices (some values were removed)
    """
    # Compute the intersection of the indices by doing a logical
    # and between the one hot encoded indices for x1 and x2.
    dim_0 = x1.dense_shape[0]
    x1_indices_expanded = tf.expand_dims(x1.indices, axis=1)
    x2_indices_expanded = tf.expand_dims(x2.indices, axis=1)
    x1_indices_count = x1_indices_expanded.shape[0]
    x2_indices_count = x2_indices_expanded.shape[0]
    x1_indices_one_hot = tf.scatter_nd(
        x1_indices_expanded,
        ones_bool((x1_indices_count,)),
        (dim_0,),
    )
    x2_indices_one_hot = tf.scatter_nd(
        x2_indices_expanded,
        ones_bool((x2_indices_count,)),
        (dim_0,),
    )
    intersection_indices = tf.squeeze(
        tf.where(tf.math.logical_and(x1_indices_one_hot, x2_indices_one_hot)),
        axis=-1,
    )
    intersection_indices_count = tf.shape(intersection_indices)[0]

    def empty_intersection():
        return (
            intersection_indices,
            tf.zeros((0,) + x1.values.shape[1:], x1.dtype),
            tf.zeros((0,) + x2.values.shape[1:], x2.dtype),
        )

    def non_empty_intersection():
        # Re-gather sub parts of the values that are part of the intersection.
        def values_for_intersection(indices_expanded, indices_count, values):
            indices_indices = tf.scatter_nd(
                indices_expanded,
                tf.range(indices_count),
                (dim_0,),
            )
            to_intersection_indices = tf.gather(
                indices_indices, intersection_indices
            )
            return tf.gather(values, to_intersection_indices)

        # Only recompute values if some indices were removed.
        x1_values_for_intersection = tf.cond(
            tf.equal(x1_indices_count, intersection_indices_count),
            lambda: x1.values,
            lambda: values_for_intersection(
                x1_indices_expanded, x1_indices_count, x1.values
            ),
        )
        x2_values_for_intersection = tf.cond(
            tf.equal(x2_indices_count, intersection_indices_count),
            lambda: x2.values,
            lambda: values_for_intersection(
                x2_indices_expanded, x2_indices_count, x2.values
            ),
        )

        return (
            intersection_indices,
            x1_values_for_intersection,
            x2_values_for_intersection,
        )

    return tf.cond(
        tf.equal(intersection_indices_count, 0),
        empty_intersection,
        non_empty_intersection,
    )


def densifying_unary(default_value):
    """Decorator to add support for `tf.SparseTensor` and `tf.IndexedSlices` to
    a non-zero-preserving element-wise unary operator.

    There are requirements on the operator for this decorator to work correctly:

    - The operator must be element-wise
    - The operator must be unary (one input tensor and one output tensor)
    - The operator must return a tensor of the same shape.

    Additional arguments to the function (besides the input tensor) are
    supported. The returned result is a dense tensor and contains
    `default_value` outside of the indices of the input tensor.

    Args:
        default_value: The value to use outside of indices. It must be the value
        that the operator returns for zero values.
    Returns:
        Wrapped function that supports `tf.SparseTensor` and `tf.IndexedSlices`.
    """

    def wrap_densifying_unary(func):
        @functools.wraps(func)
        def sparse_wrapper(x, *args, **kwargs):
            if isinstance(x, tf.SparseTensor):
                sparse_output = sparse_with_values(
                    x, func(x.values, *args, **kwargs)
                )
                return sparse_to_dense(
                    sparse_output,
                    tf.cast(default_value, sparse_output.values.dtype),
                )
            elif isinstance(x, tf.IndexedSlices):
                sparse_output_values = func(x.values, *args, **kwargs)
                output = tf.fill(
                    x.dense_shape,
                    tf.cast(default_value, sparse_output_values.dtype),
                )
                return tf.tensor_scatter_nd_update(
                    output, tf.expand_dims(x.indices, 1), sparse_output_values
                )
            return func(x, *args, **kwargs)

        return sparse_wrapper

    return wrap_densifying_unary


def elementwise_unary(func):
    """Decorator to add support for `tf.SparseTensor` and `tf.IndexedSlices` to
    a zero-preserving element-wise unary operator.

    There are requirements on the operator for this decorator to work correctly:

    - The operator must be element-wise
    - The operator must be unary (one input tensor and one output tensor)
    - The operator must return a tensor of the same shape, and if it is a
      `tf.SparseTensor` or `tf.IndexedSlices`, the indices of the result must be
      the same. Therefore:
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
        func: The function to wrap.
    Returns:
        Wrapped function that supports `tf.SparseTensor` and `tf.IndexedSlices`.
    """

    @functools.wraps(func)
    def sparse_wrapper(x, *args, **kwargs):
        if isinstance(x, tf.SparseTensor):
            return sparse_with_values(x, func(x.values, *args, **kwargs))
        elif isinstance(x, tf.IndexedSlices):
            return tf.IndexedSlices(
                func(x.values, *args, **kwargs), x.indices, x.dense_shape
            )
        else:
            return func(x, *args, **kwargs)

    return sparse_wrapper


def elementwise_binary_union(sparse_op, densify_mixed=False):
    """Decorator to add support for `tf.SparseTensor` and `tf.IndexedSlices` to
    an element-wise binary operator such that the indices present in the result
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

    Args:
        sparse_op: implementation of the operation for `tf.SparseTensor`. Must
            work if both of the operands are `tf.SparseTensor`s and can
            optionally work if one of the operand is a `tf.SparseTensor` and
            the other one is dense tensor, see `densify_mixed`.
        densify_mixed: if `True`, `sparse_op` does not support a mix of
            `tf.SparseTensor` and dense tensor or dense tensor with
            `tf.SparseTensor` and the `tf.SparseTensor` tensor is densified.
    Returns:
        Wrapped function that supports `tf.SparseTensor` and `tf.IndexedSlices`.
    """

    def wrap_elementwise_binary_union(func):
        @functools.wraps(func)
        def sparse_wrapper(x1, x2):
            if isinstance(x1, tf.SparseTensor):
                if isinstance(x2, tf.SparseTensor):
                    # x1 is a SparseTensor and x2 is a SparseTensor.
                    if x1.indices is x2.indices:
                        return sparse_with_values(
                            x1, func(x1.values, x2.values)
                        )
                    else:
                        output = sparse_op(x1, x2)
                        output.set_shape(x1.shape)
                        return output
                else:
                    # x1 is a SparseTensor.
                    if densify_mixed:
                        x1 = sparse_to_dense(x1)
                    else:
                        if not hasattr(x2, "shape") or len(x2.shape) == 0:
                            # x2 is a scalar, broadcast.
                            x2 = broadcast_scalar_to_sparse_shape(x2, x1)
                        return sparse_op(x1, x2)
            elif isinstance(x2, tf.SparseTensor):
                # x2 is a SparseTensor.
                if densify_mixed:
                    x2 = sparse_to_dense(x2)
                else:
                    if not hasattr(x1, "shape") or len(x1.shape) == 0:
                        # x1 is a scalar, broadcast.
                        x1 = broadcast_scalar_to_sparse_shape(x1, x2)
                    return sparse_op(x1, x2)
            elif isinstance(x1, tf.IndexedSlices):
                if isinstance(x2, tf.IndexedSlices):
                    # x1 is an IndexedSlices and x2 is an IndexedSlices.
                    if x1.indices is x2.indices:
                        return tf.IndexedSlices(
                            func(x1.values, x2.values),
                            x1.indices,
                            x1.dense_shape,
                        )
                    else:
                        # Compute the union of indices.
                        (
                            union_indices,
                            x1_values_for_union,
                            x2_values_for_union,
                        ) = indexed_slices_union_indices_and_values(
                            x1, x2.indices, x2.values
                        )
                        # Now, it is an element-wise operation on the union.
                        return tf.IndexedSlices(
                            func(
                                x1_values_for_union,
                                x2_values_for_union,
                            ),
                            union_indices,
                            x1.dense_shape,
                        )
                else:
                    # x1 is an IndexedSlices, densify.
                    x1 = tf.convert_to_tensor(x1)
            elif isinstance(x2, tf.IndexedSlices):
                # x2 is an IndexedSlices, densify.
                x2 = tf.convert_to_tensor(x2)
            return func(x1, x2)

        return sparse_wrapper

    return wrap_elementwise_binary_union


def elementwise_binary_intersection(func):
    """Decorator to add support for `tf.SparseTensor` and `tf.IndexedSlices` to
    an element-wise binary operator such that the indices present in the result
    are the intersection of the indices in the two operand.

    The primary use case for this is the `multiply` operator.

    There are requirements on the operator for this decorator to work correctly:

    - The operator must be element-wise.
    - The operator must be binary (two input tensors and one output tensor).
    - Both inputs must be of the same shape or one input must be a scalar.
    - The output must be of the same shape as the (non scalar) inputs.
    - The indices of the output must be the intersection of the indices of the
      inputs. This implies that func(0, x) and func(x, 0) must be 0 for any x.
      As a result, if one operand is dense or a scalar, then the indices are the
      ones from the other operand.

    Additional arguments to the function (besides the input tensors) are not
    supported.

    Note that if the operands contains zero values at some common indices, the
    indices and the zero values are preserved.

    Args:
        func: The function to wrap.
    Returns:
        Wrapped function that supports `tf.SparseTensor` and `tf.IndexedSlices`.
    """

    @functools.wraps(func)
    def sparse_wrapper(x1, x2):
        if isinstance(x1, tf.SparseTensor):
            if isinstance(x2, tf.SparseTensor):
                # x1 is a SparseTensor and x2 is a SparseTensor.
                if x1.indices is x2.indices:
                    return sparse_with_values(x1, func(x1.values, x2.values))
                else:
                    # Compute the intersection of indices.
                    (
                        intersection_indices,
                        x1_values_for_intersection,
                        x2_values_for_intersection,
                    ) = sparse_intersection_indices_and_values(x1, x2)
                    # Now, it is an element-wise operation on the intersection.
                    output = tf.SparseTensor(
                        intersection_indices,
                        func(
                            x1_values_for_intersection,
                            x2_values_for_intersection,
                        ),
                        x1.dense_shape,
                    )
                    output.set_shape(x1.shape)
                    return output
            else:
                # x1 is a SparseTensor.
                if not hasattr(x2, "shape") or len(x2.shape) == 0:
                    # x2 is a scalar, apply func element-wise.
                    return sparse_with_values(x1, func(x1.values, x2))
                else:
                    # x2 is dense, gather values from x1 indices.
                    return sparse_with_values(
                        x1, func(x1.values, tf.gather_nd(x2, x1.indices))
                    )
        elif isinstance(x2, tf.SparseTensor):
            # x2 is a SparseTensor.
            if not hasattr(x1, "shape") or len(x1.shape) == 0:
                # x1 is a scalar, apply func element-wise.
                return sparse_with_values(x2, func(x1, x2.values))
            else:
                # x1 is dense, gather values from x2 indices.
                return sparse_with_values(
                    x2, func(tf.gather_nd(x1, x2.indices), x2.values)
                )
        elif isinstance(x1, tf.IndexedSlices):
            if isinstance(x2, tf.IndexedSlices):
                # x1 is an IndexedSlices and x2 is an IndexedSlices.
                if x1.indices is x2.indices:
                    return tf.IndexedSlices(
                        func(x1.values, x2.values), x1.indices, x1.dense_shape
                    )
                else:
                    # Compute the intersection of indices.
                    (
                        intersection_indices,
                        x1_values_for_intersection,
                        x2_values_for_intersection,
                    ) = indexed_slices_intersection_indices_and_values(x1, x2)
                    # Now, it is an element-wise operation on the intersection.
                    return tf.IndexedSlices(
                        func(
                            x1_values_for_intersection,
                            x2_values_for_intersection,
                        ),
                        intersection_indices,
                        x1.dense_shape,
                    )
            else:
                # x1 is an IndexedSlices.
                if not hasattr(x2, "shape") or len(x2.shape) == 0:
                    # x2 is a scalar, apply func element-wise.
                    return tf.IndexedSlices(
                        func(x1.values, x2), x1.indices, x1.dense_shape
                    )
                else:
                    # x2 is dense, gather values from x1 indices.
                    return tf.IndexedSlices(
                        func(x1.values, tf.gather(x2, x1.indices)),
                        x1.indices,
                        x1.dense_shape,
                    )
        elif isinstance(x2, tf.IndexedSlices):
            # x2 is an IndexedSlices.
            if not hasattr(x1, "shape") or len(x1.shape) == 0:
                # x1 is a scalar, apply func element-wise.
                return tf.IndexedSlices(
                    func(x1, x2.values), x2.indices, x2.dense_shape
                )
            else:
                # x1 is dense, gather values from x2 indices.
                return tf.IndexedSlices(
                    func(tf.gather(x1, x2.indices), x2.values),
                    x2.indices,
                    x2.dense_shape,
                )
        # Default case, no SparseTensor and no IndexedSlices.
        return func(x1, x2)

    return sparse_wrapper


def elementwise_division(func):
    """Decorator to add support for `tf.SparseTensor` and `tf.IndexedSlices` to
    element-wise binary division and related operators.

    This decorator is designed for operations related to the division of two
    operands (e.g. `divide`). It accepts `tf.SparseTensor` and
    `tf.IndexedSlices` for both the dividend and the divisor, but handles them
    differently based on whether they are the dividend or the divisor.

    - If the divisor is a `tf.SparseTensor` or `tf.IndexedSlices`, it is
      densified and the result is dense because the result contains Inf or Nan
      outside of the indices of the dividend.
    - If the dividend is a `tf.SparseTensor` or `tf.IndexedSlices` and the
      divisor is dense, it finds occurrences of zeros and NaNs in the divisor.
      The result may therefore have more indices than there were in the dividend
      to return correct values where the divisor was zero or NaN.
    - If the dividend is a `tf.SparseTensor` or `tf.IndexedSlices` and the
      divisor is a scalar, it does the division element-wise. Note that the
      result is incorrectly sparse if the scalar divisor is zero.

    Args:
        func: The function to wrap.
    Returns:
        Wrapped function that supports `tf.SparseTensor` and `tf.IndexedSlices`.
    """

    @functools.wraps(func)
    def sparse_wrapper(x1, x2):
        if isinstance(x1, tf.SparseTensor):
            if isinstance(x2, tf.SparseTensor):
                # x1 is a SparseTensor and x2 is a SparseTensor.
                # Divisor is sparse, meaning we're doing divisions by zero
                # outside of x2.indices, so the result is dense. Densify both.
                x1 = sparse_to_dense(x1)
                x2 = sparse_to_dense(x2)
            else:
                # x1 is a SparseTensor.
                if not hasattr(x2, "shape") or len(x2.shape) == 0:
                    # x2 is a scalar, apply func element-wise.
                    return sparse_with_values(x1, func(x1.values, x2))
                else:
                    # x2 is dense.
                    x2_zeros_and_nans = tf.equal(x2, 0)
                    if not tf.as_dtype(x2.dtype).is_integer:
                        x2_zeros_and_nans = tf.math.logical_or(
                            x2_zeros_and_nans, tf.math.is_nan(x2)
                        )

                    def func_for_x1_indices():
                        # Gather values from x1 indices.
                        return sparse_with_values(
                            x1, func(x1.values, tf.gather_nd(x2, x1.indices))
                        )

                    def func_for_union_indices():
                        # Compute the union of indices to keep zeros and NaNs.
                        x2_zeros_and_nan_indices = tf.where(x2_zeros_and_nans)
                        (
                            union_indices,
                            x1_values_for_union,
                            _,
                        ) = sparse_union_indices_and_values(
                            x1, x2_zeros_and_nan_indices
                        )
                        output = tf.SparseTensor(
                            union_indices,
                            func(
                                x1_values_for_union,
                                tf.gather_nd(x2, union_indices),
                            ),
                            x1.dense_shape,
                        )
                        output.set_shape(x1.shape)
                        return output

                    return tf.cond(
                        tf.reduce_any(x2_zeros_and_nans),
                        func_for_union_indices,
                        func_for_x1_indices,
                    )
        elif isinstance(x2, tf.SparseTensor):
            # x2 is a SparseTensor.
            # Divisor is sparse, densify to do the divisions by zero correctly.
            x2 = sparse_to_dense(x2)
        elif isinstance(x1, tf.IndexedSlices):
            if isinstance(x2, tf.IndexedSlices):
                # x1 is an IndexedSlices and x2 is an IndexedSlices.
                # Divisor is slices, meaning we're doing divisions by zero
                # outside of x2.indices, so the result is dense. Densify both.
                x1 = tf.convert_to_tensor(x1)
                x2 = tf.convert_to_tensor(x2)
            else:
                # x1 is a IndexedSlices.
                if not hasattr(x2, "shape") or len(x2.shape) == 0:
                    # x2 is a scalar, apply func element-wise.
                    return tf.IndexedSlices(
                        func(x1.values, x2), x1.indices, x1.dense_shape
                    )
                else:
                    # x2 is dense.
                    x2_zeros_and_nans = tf.equal(x2, 0)
                    if not tf.as_dtype(x2.dtype).is_integer:
                        x2_zeros_and_nans = tf.math.logical_or(
                            x2_zeros_and_nans, tf.math.is_nan(x2)
                        )
                    x2_zeros_and_nans = tf.reduce_any(
                        x2_zeros_and_nans, axis=tuple(range(1, x2.shape.rank))
                    )

                    def func_for_x1_indices():
                        # Gather values from x1 indices.
                        return tf.IndexedSlices(
                            func(x1.values, tf.gather(x2, x1.indices)),
                            x1.indices,
                            x1.dense_shape,
                        )

                    def func_for_union_indices():
                        x2_zeros_and_nan_indices = tf.squeeze(
                            tf.where(x2_zeros_and_nans), axis=-1
                        )
                        # Compute the union of indices to keep zeros and NaNs.
                        (
                            union_indices,
                            x1_values_for_union,
                            _,
                        ) = indexed_slices_union_indices_and_values(
                            x1, x2_zeros_and_nan_indices
                        )
                        return tf.IndexedSlices(
                            func(
                                x1_values_for_union,
                                tf.gather(x2, union_indices),
                            ),
                            union_indices,
                            x1.dense_shape,
                        )

                    return tf.cond(
                        tf.reduce_any(x2_zeros_and_nans),
                        func_for_union_indices,
                        func_for_x1_indices,
                    )
        elif isinstance(x2, tf.IndexedSlices):
            # x2 is a IndexedSlices.
            # Divisor is slices, densify to do the divisions by zero correctly.
            x2 = tf.convert_to_tensor(x2)
        # Default case, no SparseTensor and no IndexedSlices.
        return func(x1, x2)

    return sparse_wrapper
