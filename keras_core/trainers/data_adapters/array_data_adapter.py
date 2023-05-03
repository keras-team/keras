import math

import numpy as np
import tensorflow as tf
from tensorflow import nest

from keras_core import backend
from keras_core.trainers.data_adapters import data_adapter_utils
from keras_core.trainers.data_adapters.data_adapter import DataAdapter

try:
    import pandas
except ImportError:
    pandas = None


class ArrayDataAdapter(DataAdapter):
    """Adapter for array-like objects, e.g. TF/JAX Tensors, NumPy arrays."""

    def __init__(
        self,
        x,
        y=None,
        sample_weight=None,
        batch_size=None,
        steps=None,
        shuffle=False,
        class_weight=None,
    ):
        types_struct = nest.map_structure(lambda x: type(x), x)
        flat_types = nest.flatten(types_struct)
        if not all(
            issubclass(c, data_adapter_utils.ARRAY_TYPES) for c in flat_types
        ):
            raise ValueError(
                "Expected all elements of `x` to be array-like. "
                f"Received invalid types: x={x}"
            )

        x, y, sample_weight = convert_to_arrays((x, y, sample_weight))
        if sample_weight is not None:
            if class_weight is not None:
                raise ValueError(
                    "You cannot `class_weight` and `sample_weight` "
                    "at the same time."
                )
            if tf.nest.is_nested(y):
                if isinstance(sample_weight, np.ndarray):
                    is_samplewise = len(sample_weight.shape) == 1 or (
                        len(sample_weight.shape) == 2
                        and sample_weight.shape[1] == 1
                    )
                    if not is_samplewise:
                        raise ValueError(
                            "For a model with multiple outputs, when providing "
                            "a single `sample_weight` array, it should only "
                            "have one scalar score per sample "
                            "(i.e. shape `(num_samples,)`). If you want to use "
                            "non-scalar sample weights, pass a `sample_weight` "
                            "argument with one array per model output."
                        )
                    # Replicate the same sample_weight array on all outputs.
                    sample_weight = tf.nest.map_structure(
                        lambda _: sample_weight, y
                    )
                else:
                    try:
                        tf.nest.assert_same_structure(y, sample_weight)
                    except ValueError:
                        raise ValueError(
                            "You should provide one `sample_weight` array per "
                            "output in `y`. The two structures did not match:\n"
                            f"- y: {y}\n"
                            f"- sample_weight: {sample_weight}\n"
                        )
        if class_weight is not None:
            if tf.nest.is_nested(y):
                raise ValueError(
                    "`class_weight` is only supported for Models with a single "
                    "output."
                )
            sample_weight = data_adapter_utils.class_weight_to_sample_weights(
                y, class_weight
            )

        inputs = data_adapter_utils.pack_x_y_sample_weight(x, y, sample_weight)

        data_adapter_utils.check_data_cardinality(inputs)
        num_samples = set(i.shape[0] for i in nest.flatten(inputs)).pop()
        self._num_samples = num_samples
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
        inputs = self._inputs
        if self._shuffle:
            inputs = data_adapter_utils.sync_shuffle(
                inputs, num_samples=self._num_samples
            )
        for i in range(self._size):
            start, stop = i * self._batch_size, (i + 1) * self._batch_size
            yield tf.nest.map_structure(lambda x: x[start:stop], inputs)

    def get_tf_dataset(self):
        ds = tf.data.Dataset.from_tensor_slices(self._inputs)
        if self._shuffle:
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
                "Expected a NumPy array, tf.Tensor, Pandas Dataframe or Pandas "
                f"Series. Received invalid input: {x} (of type {type(x)})"
            )
        if not str(x.dtype) == str(dtype):
            x = x.astype(dtype)
        return x

    arrays = tf.nest.map_structure(convert_single_array, arrays)
    return tf.__internal__.nest.list_to_tuple(arrays)
