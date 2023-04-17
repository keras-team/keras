import math

import numpy as np
import tensorflow as tf

try:
    import pandas
except ImportError:
    pandas = None


ARRAY_TYPES = (tf.Tensor, np.ndarray)
if pandas:
    ARRAY_TYPES = ARRAY_TYPES + (
        tf.Tensor,
        np.ndarray,
        pandas.Series,
        pandas.DataFrame,
    )
# TODO: support torch tensors?


def unpack_x_y_sample_weight(data):
    """Unpacks user-provided data tuple.

    This is a convenience utility to be used when overriding
    `Model.train_step`, `Model.test_step`, or `Model.predict_step`.
    This utility makes it easy to support data of the form `(x,)`,
    `(x, y)`, or `(x, y, sample_weight)`.

    Standalone usage:

    >>> features_batch = tf.ones((10, 5))
    >>> labels_batch = tf.zeros((10, 5))
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


def pack_x_y_sample_weight(x, y=None, sample_weight=None):
    """Packs user-provided data into a tuple.

    This is a convenience utility for packing data into the tuple formats
    that `Model.fit` uses.

    Standalone usage:

    >>> x = tf.ones((10, 1))
    >>> data = pack_x_y_sample_weight(x)
    >>> isinstance(data, tf.Tensor)
    True
    >>> y = tf.ones((10, 1))
    >>> data = pack_x_y_sample_weight(x, y)
    >>> isinstance(data, tuple)
    True
    >>> x, y = data

    Args:
        x: Features to pass to `Model`.
        y: Ground-truth targets to pass to `Model`.
        sample_weight: Sample weight for each element.

    Returns:
        Tuple in the format used in `Model.fit`.
    """
    if y is None:
        # For single x-input, we do no tuple wrapping since in this case
        # there is no ambiguity. This also makes NumPy and Dataset
        # consistent in that the user does not have to wrap their Dataset
        # data in an unnecessary tuple.
        if not isinstance(x, tuple or list):
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
    num_samples = set(int(i.shape[0]) for i in tf.nest.flatten(data))
    if len(num_samples) > 1:
        msg = (
            "Data cardinality is ambiguous. "
            "Make sure all arrays contain the same number of samples."
        )
        for label, single_data in zip(["x", "y", "sample_weight"], data):
            sizes = ", ".join(
                str(i.shape[0]) for i in tf.nest.flatten(single_data)
            )
            msg += f"'{label}' sizes: {sizes}\n"
        raise ValueError(msg)


def sync_shuffle(data, num_samples=None):
    if num_samples is None:
        num_samples_set = set(int(i.shape[0]) for i in tf.nest.flatten(data))
        assert len(num_samples_set) == 1
        num_samples = num_samples_set.pop()
    p = np.random.permutation(num_samples)
    return tf.nest.map_structure(lambda x: x[p], data)


def train_validation_split(arrays, validation_split):
    """Split arrays into train and validation subsets in deterministic order.

    The last part of data will become validation data.

    Args:
        arrays: Tensors to split. Allowed inputs are arbitrarily nested structures
            of Tensors and NumPy arrays.
        validation_split: Float between 0 and 1. The proportion of the dataset to
            include in the validation split. The rest of the dataset will be
            included in the training split.

    Returns:
        `(train_arrays, validation_arrays)`
    """

    def _can_split(t):
        return isinstance(t, ARRAY_TYPES) or t is None

    flat_arrays = tf.nest.flatten(arrays)
    unsplitable = [type(t) for t in flat_arrays if not _can_split(t)]
    if unsplitable:
        raise ValueError(
            "Argument `validation_split` is only supported for tf.Tensors or NumPy "
            "arrays. Found incompatible type in the input: {unsplitable}"
        )

    if all(t is None for t in flat_arrays):
        return arrays, arrays

    first_non_none = None
    for t in flat_arrays:
        if t is not None:
            first_non_none = t
            break

    # Assumes all arrays have the same batch shape or are `None`.
    batch_dim = int(first_non_none.shape[0])
    split_at = int(math.floor(batch_dim * (1.0 - validation_split)))

    if split_at == 0 or split_at == batch_dim:
        raise ValueError(
            f"Training data contains {batch_dim} samples, which is not "
            "sufficient to split it into a validation and training set as "
            f"specified by `validation_split={validation_split}`. Either "
            "provide more data, or a different value for the "
            "`validation_split` argument."
        )

    def _split(t, start, end):
        if t is None:
            return t
        return t[start:end]

    train_arrays = tf.nest.map_structure(
        lambda x: _split(x, start=0, end=split_at), arrays
    )
    val_arrays = tf.nest.map_structure(
        lambda x: _split(x, start=split_at, end=batch_dim), arrays
    )
    return train_arrays, val_arrays
