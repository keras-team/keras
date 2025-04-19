import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src import tree
from keras.src.api_export import keras_export

NUM_BATCHES_FOR_TENSOR_SPEC = 2


@keras_export("keras.utils.unpack_x_y_sample_weight")
def unpack_x_y_sample_weight(data):
    """Unpacks user-provided data tuple.

    This is a convenience utility to be used when overriding
    `Model.train_step`, `Model.test_step`, or `Model.predict_step`.
    This utility makes it easy to support data of the form `(x,)`,
    `(x, y)`, or `(x, y, sample_weight)`.

    Example:

    >>> features_batch = ops.ones((10, 5))
    >>> labels_batch = ops.zeros((10, 5))
    >>> data = (features_batch, labels_batch)
    >>> # `y` and `sample_weight` will default to `None` if not provided.
    >>> x, y, sample_weight = unpack_x_y_sample_weight(data)
    >>> sample_weight is None
    True

    Args:
        data: A tuple of the form `(x,)`, `(x, y)`, or `(x, y, sample_weight)`.

    Returns:
        The unpacked tuple, with `None`s for `y` and `sample_weight` if they are
        not provided.
    """
    if isinstance(data, list):
        data = tuple(data)
    if not isinstance(data, tuple):
        return (data, None, None)
    elif len(data) == 1:
        return (data[0], None, None)
    elif len(data) == 2:
        return (data[0], data[1], None)
    elif len(data) == 3:
        return (data[0], data[1], data[2])
    error_msg = (
        "Data is expected to be in format `x`, `(x,)`, `(x, y)`, "
        f"or `(x, y, sample_weight)`, found: {data}"
    )
    raise ValueError(error_msg)


@keras_export("keras.utils.pack_x_y_sample_weight")
def pack_x_y_sample_weight(x, y=None, sample_weight=None):
    """Packs user-provided data into a tuple.

    This is a convenience utility for packing data into the tuple formats
    that `Model.fit()` uses.

    Example:

    >>> x = ops.ones((10, 1))
    >>> data = pack_x_y_sample_weight(x)
    >>> isinstance(data, ops.Tensor)
    True
    >>> y = ops.ones((10, 1))
    >>> data = pack_x_y_sample_weight(x, y)
    >>> isinstance(data, tuple)
    True
    >>> x, y = data

    Args:
        x: Features to pass to `Model`.
        y: Ground-truth targets to pass to `Model`.
        sample_weight: Sample weight for each element.

    Returns:
        Tuple in the format used in `Model.fit()`.
    """
    if y is None:
        # For single x-input, we do no tuple wrapping since in this case
        # there is no ambiguity. This also makes NumPy and Dataset
        # consistent in that the user does not have to wrap their Dataset
        # data in an unnecessary tuple.
        if not isinstance(x, (tuple, list)):
            return x
        else:
            return (x,)
    elif sample_weight is None:
        return (x, y)
    else:
        return (x, y, sample_weight)


def list_to_tuple(maybe_list):
    """Datasets will stack any list of tensors, so we convert them to tuples."""
    if isinstance(maybe_list, list):
        return tuple(maybe_list)
    return maybe_list


def check_data_cardinality(data):
    num_samples = set(int(i.shape[0]) for i in tree.flatten(data))
    if len(num_samples) > 1:
        msg = (
            "Data cardinality is ambiguous. "
            "Make sure all arrays contain the same number of samples."
        )
        for label, single_data in zip(["x", "y", "sample_weight"], data):
            sizes = ", ".join(
                str(i.shape[0]) for i in tree.flatten(single_data)
            )
            msg += f"'{label}' sizes: {sizes}\n"
        raise ValueError(msg)


def class_weight_to_sample_weights(y, class_weight):
    # Convert to numpy to ensure consistent handling of operations
    # (e.g., np.round()) across frameworks like TensorFlow, JAX, and PyTorch

    y_numpy = ops.convert_to_numpy(y)
    sample_weight = np.ones(shape=(y_numpy.shape[0],), dtype=backend.floatx())
    if len(y_numpy.shape) > 1:
        if y_numpy.shape[-1] != 1:
            y_numpy = np.argmax(y_numpy, axis=-1)
        else:
            y_numpy = np.squeeze(y_numpy, axis=-1)
    y_numpy = np.round(y_numpy).astype("int32")

    for i in range(y_numpy.shape[0]):
        sample_weight[i] = class_weight.get(int(y_numpy[i]), 1.0)
    return sample_weight


def get_tensor_spec(batches):
    """Return the common tensor spec for a list of batches.

    Args:
        batches: list of structures of tensors. The structures must be
            identical, but the shape at each leaf may be different.
    Returns: the common tensor spec for all the batches.
    """
    from keras.src.utils.module_utils import tensorflow as tf

    def get_single_tensor_spec(*tensors):
        x = tensors[0]
        rank = len(x.shape)
        if rank < 1:
            raise ValueError(
                "When passing a dataset to a Keras model, the arrays must "
                f"be at least rank 1. Received: {x} of rank {len(x.shape)}."
            )
        for t in tensors:
            if len(t.shape) != rank:
                raise ValueError(
                    "When passing a dataset to a Keras model, the "
                    "corresponding arrays in each batch must have the same "
                    f"rank. Received: {x} and {t}"
                )
        shape = []
        # Merge shapes: go through each dimension one by one and keep the
        # common values
        for dims in zip(*[list(x.shape) for x in tensors]):
            dims_set = set(dims)
            shape.append(dims_set.pop() if len(dims_set) == 1 else None)
        shape[0] = None  # batch size may not be static

        dtype = backend.standardize_dtype(x.dtype)
        if isinstance(x, tf.RaggedTensor):
            return tf.RaggedTensorSpec(
                shape=shape,
                dtype=dtype,
                ragged_rank=x.ragged_rank,
                row_splits_dtype=x.row_splits.dtype,
            )
        if (
            isinstance(x, tf.SparseTensor)
            or is_scipy_sparse(x)
            or is_jax_sparse(x)
        ):
            return tf.SparseTensorSpec(shape=shape, dtype=dtype)
        else:
            return tf.TensorSpec(shape=shape, dtype=dtype)

    return tree.map_structure(get_single_tensor_spec, *batches)


def get_jax_iterator(iterable):
    import jax
    import jax.experimental.sparse as jax_sparse

    def convert_to_jax_compatible(x):
        if isinstance(x, (jax.Array, jax_sparse.JAXSparse, np.ndarray)):
            return x
        elif is_scipy_sparse(x):
            return scipy_sparse_to_jax_sparse(x)
        elif is_tensorflow_sparse(x):
            return tf_sparse_to_jax_sparse(x)
        else:
            return np.asarray(x)

    for batch in iterable:
        yield tree.map_structure(convert_to_jax_compatible, batch)


def get_numpy_iterator(iterable):
    def convert_to_numpy(x):
        if not isinstance(x, np.ndarray):
            # Using `__array__` should handle `tf.Tensor`, `jax.np.ndarray`,
            # `torch.Tensor`, as well as any other tensor-like object that
            # has added numpy support.
            if hasattr(x, "__array__"):
                if is_torch_tensor(x):
                    x = x.cpu()
                x = np.asarray(x)
        return x

    for batch in iterable:
        yield tree.map_structure(convert_to_numpy, batch)


def get_torch_dataloader(iterable):
    import torch.utils.data as torch_data

    from keras.src.backend.torch.core import convert_to_tensor

    class ConverterIterableDataset(torch_data.IterableDataset):
        def __init__(self, iterable):
            self.iterable = iterable

        def __iter__(self):
            for batch in self.iterable:
                yield tree.map_structure(convert_to_tensor, batch)

    dataset = ConverterIterableDataset(iterable)
    # `batch_size=None` indicates that we should not re-batch
    return torch_data.DataLoader(dataset, batch_size=None)


def is_tensorflow_tensor(value):
    if hasattr(value, "__class__"):
        if value.__class__.__name__ in ("RaggedTensor", "SparseTensor"):
            return "tensorflow.python." in str(value.__class__.__module__)
        for parent in value.__class__.__mro__:
            if parent.__name__ in ("Tensor") and "tensorflow.python." in str(
                parent.__module__
            ):
                return True
    return False


def is_tensorflow_ragged(value):
    if hasattr(value, "__class__"):
        return (
            value.__class__.__name__ == "RaggedTensor"
            and "tensorflow.python." in str(value.__class__.__module__)
        )
    return False


def is_tensorflow_sparse(value):
    if hasattr(value, "__class__"):
        return (
            value.__class__.__name__ == "SparseTensor"
            and "tensorflow.python." in str(value.__class__.__module__)
        )
    return False


def is_jax_array(value):
    if hasattr(value, "__class__"):
        for parent in value.__class__.__mro__:
            if parent.__name__ == "Array" and str(parent.__module__) == "jax":
                return True
    return is_jax_sparse(value)  # JAX sparse arrays do not extend jax.Array


def is_jax_sparse(value):
    if hasattr(value, "__class__"):
        return str(value.__class__.__module__).startswith(
            "jax.experimental.sparse"
        )
    return False


def is_torch_tensor(value):
    if hasattr(value, "__class__"):
        for parent in value.__class__.__mro__:
            if parent.__name__ == "Tensor" and str(parent.__module__).endswith(
                "torch"
            ):
                return True
    return False


def is_scipy_sparse(x):
    return str(x.__class__.__module__).startswith("scipy.sparse") and hasattr(
        x, "tocoo"
    )


def scipy_sparse_to_tf_sparse(x):
    from keras.src.utils.module_utils import tensorflow as tf

    coo = x.tocoo()
    indices = np.concatenate(
        (np.expand_dims(coo.row, 1), np.expand_dims(coo.col, 1)), axis=1
    )
    return tf.SparseTensor(indices, coo.data, coo.shape)


def scipy_sparse_to_jax_sparse(x):
    import jax
    import jax.experimental.sparse as jax_sparse

    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        return jax_sparse.BCOO.from_scipy_sparse(x)


def tf_sparse_to_jax_sparse(x):
    import jax
    import jax.experimental.sparse as jax_sparse

    values = np.asarray(x.values)
    indices = np.asarray(x.indices)
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        return jax_sparse.BCOO((values, indices), shape=x.shape)


def jax_sparse_to_tf_sparse(x):
    from keras.src.utils.module_utils import tensorflow as tf

    return tf.SparseTensor(x.indices, x.data, x.shape)
