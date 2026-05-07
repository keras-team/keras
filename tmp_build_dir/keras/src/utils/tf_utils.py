from keras.src import backend
from keras.src.utils.module_utils import tensorflow as tf


def get_tensor_spec(t, dynamic_batch=False, name=None):
    """Returns a `TensorSpec` given a single `Tensor` or `TensorSpec`."""
    if isinstance(t, tf.TypeSpec):
        spec = t
    elif isinstance(t, tf.__internal__.CompositeTensor):
        # Check for ExtensionTypes
        spec = t._type_spec
    elif hasattr(t, "shape") and hasattr(t, "dtype"):
        spec = tf.TensorSpec(shape=t.shape, dtype=t.dtype, name=name)
    else:
        return None  # Allow non-Tensors to pass through.

    if not dynamic_batch:
        return spec

    shape = spec.shape
    if shape.rank is None or shape.rank == 0:
        return spec

    shape_list = shape.as_list()
    shape_list[0] = None
    shape = tf.TensorShape(shape_list)
    spec._shape = shape
    return spec


def ensure_tensor(inputs, dtype=None):
    """Ensures the input is a Tensor, SparseTensor or RaggedTensor."""
    if not isinstance(inputs, (tf.Tensor, tf.SparseTensor, tf.RaggedTensor)):
        if backend.backend() == "torch" and backend.is_tensor(inputs):
            # Plain `np.asarray()` conversion fails with PyTorch.
            inputs = backend.convert_to_numpy(inputs)
        inputs = tf.convert_to_tensor(inputs, dtype)
    if dtype is not None and inputs.dtype != dtype:
        inputs = tf.cast(inputs, dtype)
    return inputs


def is_ragged_tensor(x):
    return "ragged_tensor.RaggedTensor" in str(type(x))


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


def expand_dims(inputs, axis):
    """Expand dims on sparse, ragged, or dense tensors."""
    if isinstance(inputs, tf.SparseTensor):
        return tf.sparse.expand_dims(inputs, axis)
    return tf.expand_dims(inputs, axis)


def tf_encode_categorical_inputs(
    inputs,
    output_mode,
    depth,
    dtype="float32",
    sparse=False,
    count_weights=None,
    idf_weights=None,
):
    """Encodes categorical inputs according to output_mode.

    Faster method that relies on bincount.
    """

    if output_mode == "int":
        return tf.identity(tf.cast(inputs, dtype))

    original_shape = inputs.shape
    # In all cases, we should uprank scalar input to a single sample.
    if inputs.shape.rank == 0:
        inputs = expand_dims(inputs, -1)
    # One hot will uprank only if the final output dimension is not already 1.
    if output_mode == "one_hot":
        if inputs.shape[-1] != 1:
            inputs = expand_dims(inputs, -1)

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

    bincounts = tf.cast(bincounts, dtype)
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
