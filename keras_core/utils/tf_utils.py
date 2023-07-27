from keras_core.utils.module_utils import tensorflow as tf


def expand_dims(inputs, axis):
    """Expand dims on sparse, ragged, or dense tensors."""
    if isinstance(inputs, tf.SparseTensor):
        return tf.sparse.expand_dims(inputs, axis)
    else:
        return tf.expand_dims(inputs, axis)


def sparse_bincount(inputs, depth, binary_output, dtype, count_weights=None):
    """Apply binary or count encoding to an input and return a sparse tensor."""
    result = tf.sparse.bincount(
        inputs,
        weights=count_weights,
        minlength=depth,
        maxlength=depth,
        axis=-1,
        binary_output=binary_output,
    )
    result = tf.cast(result, dtype)
    if inputs.shape.rank == 1:
        output_shape = (depth,)
    else:
        batch_size = tf.shape(result)[0]
        output_shape = (batch_size, depth)
    result = tf.SparseTensor(
        indices=result.indices, values=result.values, dense_shape=output_shape
    )
    return result


def dense_bincount(inputs, depth, binary_output, dtype, count_weights=None):
    """Apply binary or count encoding to an input."""
    result = tf.math.bincount(
        inputs,
        weights=count_weights,
        minlength=depth,
        maxlength=depth,
        dtype=dtype,
        axis=-1,
        binary_output=binary_output,
    )
    if inputs.shape.rank == 1:
        result.set_shape(tf.TensorShape((depth,)))
    else:
        batch_size = inputs.shape.as_list()[0]
        result.set_shape(tf.TensorShape((batch_size, depth)))
    return result


def encode_categorical_inputs(
    inputs,
    output_mode,
    depth,
    dtype="float32",
    sparse=False,
    count_weights=None,
    idf_weights=None,
):
    """Encodes categoical inputs according to output_mode."""
    if output_mode == "int":
        return tf.identity(tf.cast(inputs, dtype))

    original_shape = inputs.shape
    # In all cases, we should uprank scalar input to a single sample.
    if inputs.shape.rank == 0:
        inputs = expand_dims(inputs, -1)
    # One hot will unprank only if the final output dimension is not already 1.
    if output_mode == "one_hot":
        if inputs.shape[-1] != 1:
            inputs = expand_dims(inputs, -1)

    # TODO(b/190445202): remove output rank restriction.
    if inputs.shape.rank > 2:
        raise ValueError(
            "When output_mode is not `'int'`, maximum supported output rank "
            f"is 2. Received output_mode {output_mode} and input shape "
            f"{original_shape}, "
            f"which would result in output rank {inputs.shape.rank}."
        )

    binary_output = output_mode in ("multi_hot", "one_hot")
    if sparse:
        bincounts = sparse_bincount(
            inputs, depth, binary_output, dtype, count_weights
        )
    else:
        bincounts = dense_bincount(
            inputs, depth, binary_output, dtype, count_weights
        )

    if output_mode != "tf_idf":
        return bincounts

    if idf_weights is None:
        raise ValueError(
            "When output mode is `'tf_idf'`, idf_weights must be provided. "
            f"Received: output_mode={output_mode} and idf_weights={idf_weights}"
        )

    if sparse:
        value_weights = tf.gather(idf_weights, bincounts.indices[:, -1])
        return tf.SparseTensor(
            bincounts.indices,
            value_weights * bincounts.values,
            bincounts.dense_shape,
        )
    else:
        return tf.multiply(bincounts, idf_weights)
