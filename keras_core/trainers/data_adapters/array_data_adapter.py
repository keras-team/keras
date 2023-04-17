import math

import numpy as np
import tensorflow as tf
from tensorflow import nest

from keras_core import backend
from keras_core.trainers.data_adapters import data_adapters_utils
from keras_core.trainers.data_adapters.data_adapter import DataAdapter

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


class ArrayDataAdapter(DataAdapter):
    """Adapter that handles array-like objects, e.g. tf.Tensor and NumPy arrays."""

    @staticmethod
    def can_handle(x, y=None):
        types_struct = nest.map_structure(lambda x: type(x), x)
        flat_types = nest.flatten(types_struct)
        return all(issubclass(c, ARRAY_TYPES) for c in flat_types)

    def __init__(
        self,
        x,
        y=None,
        sample_weights=None,
        batch_size=None,
        steps=None,
        shuffle=False,
    ):
        super().__init__(x, y)
        x, y, sample_weights = convert_to_arrays((x, y, sample_weights))

        inputs = data_adapters_utils.pack_x_y_sample_weight(
            x, y, sample_weights
        )

        data_adapters_utils.check_data_cardinality(inputs)
        num_samples = set(i.shape[0] for i in nest.flatten(inputs)).pop()
        if shuffle:
            inputs = data_adapters_utils.sync_shuffle(
                inputs, num_samples=num_samples
            )
        self._inputs = inputs

        # If batch_size is not passed but steps is, calculate from the input
        # data.  Defaults to `32` for backwards compatibility.
        if not batch_size:
            batch_size = int(math.ceil(num_samples / steps)) if steps else 32

        self._size = int(math.ceil(num_samples / batch_size))
        self._batch_size = batch_size
        self._partial_batch_size = num_samples % batch_size
        self._shuffle = shuffle

    def get_numpy_iterator(self):
        for i in range(self._size):
            start, stop = i * self._batch_size, (i + 1) * self._batch_size
            yield tf.nest.map_structure(lambda x: x[start:stop], self._inputs)

    def get_tf_dataset(self):
        ds = tf.data.Dataset.from_tensor_slices(self._inputs)
        ds = ds.shuffle(self._batch_size * 8)
        ds = ds.batch(self._batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    @property
    def num_batches(self):
        return self._size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def has_partial_batch(self):
        return self._partial_batch_size > 0

    @property
    def partial_batch_size(self):
        return self._partial_batch_size or None


def convert_to_arrays(arrays, dtype=None):
    """Process array-like inputs.

    This function:

    - Converts tf.Tensors to NumPy arrays.
    - Converts `pandas.Series` to `np.ndarray`
    - Converts `list`s to `tuple`s (for `tf.data` support).

    Args:
        inputs: Structure of `Tensor`s, `NumPy` arrays, or tensor-like.

    Returns:
        Structure of NumPy `ndarray`s.
    """
    dtype = dtype or backend.floatx()

    def convert_single_array(x):
        if x is None:
            return x
        if pandas is not None:
            if isinstance(x, pandas.Series):
                x = np.expand_dims(x.to_numpy(dtype=dtype), axis=-1)
            elif isinstance(x, pandas.DataFrame):
                x = x.to_numpy(dtype=dtype)
        if isinstance(x, (tf.Tensor, tf.Variable)):
            x = x.numpy()
        if not isinstance(x, np.ndarray):
            raise ValueError(
                "Expected a NumPy array, tf.Tensor, Pandas Dataframe or Pandas Series. "
                f"Received invalid input: {x} (of type {type(x)})"
            )
        if not str(x.dtype) == str(dtype):
            x = x.astype(dtype)
        return x

    arrays = tf.nest.map_structure(convert_single_array, arrays)
    return tf.__internal__.nest.list_to_tuple(arrays)
