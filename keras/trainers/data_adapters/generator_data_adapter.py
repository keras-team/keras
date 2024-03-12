import itertools

import numpy as np

from keras import backend
from keras.trainers.data_adapters import data_adapter_utils
from keras.trainers.data_adapters.data_adapter import DataAdapter
from keras.utils import tree


class GeneratorDataAdapter(DataAdapter):
    """Adapter for Python generators."""

    def __init__(self, generator):
        first_batch, generator = peek_and_restore(generator)
        self.generator = generator
        self._first_batch = first_batch
        self._output_signature = None
        if not isinstance(first_batch, tuple):
            raise ValueError(
                "When passing a Python generator to a Keras model, "
                "the generator must return a tuple, either "
                "(input,) or (inputs, targets) or "
                "(inputs, targets, sample_weights). "
                f"Received: {first_batch}"
            )

    def _set_tf_output_signature(self):
        from keras.utils.module_utils import tensorflow as tf

        def get_tensor_spec(x):
            shape = x.shape
            if len(shape) < 1:
                raise ValueError(
                    "When passing a Python generator to a Keras model, "
                    "the arrays returned by the generator "
                    "must be at least rank 1. Received: "
                    f"{x} of rank {len(x.shape)}"
                )
            shape = list(shape)
            shape[0] = None  # The batch size is not guaranteed to be static.
            dtype = backend.standardize_dtype(x.dtype)
            if isinstance(x, tf.RaggedTensor):
                return tf.RaggedTensorSpec(shape=shape, dtype=dtype)
            if (
                isinstance(x, tf.SparseTensor)
                or is_scipy_sparse(x)
                or is_jax_sparse(x)
            ):
                return tf.SparseTensorSpec(shape=shape, dtype=dtype)
            else:
                return tf.TensorSpec(shape=shape, dtype=dtype)

        self._output_signature = tree.map_structure(
            get_tensor_spec, self._first_batch
        )

    def get_numpy_iterator(self):
        return data_adapter_utils.get_numpy_iterator(self.generator)

    def get_jax_iterator(self):
        from keras.backend.jax.core import convert_to_tensor

        def convert_to_jax(x):
            if is_scipy_sparse(x):
                return scipy_sparse_to_jax_sparse(x)
            elif is_tf_sparse(x):
                return tf_sparse_to_jax_sparse(x)
            return convert_to_tensor(x)

        for batch in self.generator:
            yield tree.map_structure(convert_to_jax, batch)

    def get_tf_dataset(self):
        from keras.utils.module_utils import tensorflow as tf

        def convert_to_tf(x):
            if is_scipy_sparse(x):
                x = scipy_sparse_to_tf_sparse(x)
            elif is_jax_sparse(x):
                x = jax_sparse_to_tf_sparse(x)
            return x

        def get_tf_iterator():
            for batch in self.generator:
                batch = tree.map_structure(convert_to_tf, batch)
                yield batch

        if self._output_signature is None:
            self._set_tf_output_signature()
        ds = tf.data.Dataset.from_generator(
            get_tf_iterator,
            output_signature=self._output_signature,
        )
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def get_torch_dataloader(self):
        return data_adapter_utils.get_torch_dataloader(self.generator)

    @property
    def num_batches(self):
        return None

    @property
    def batch_size(self):
        return None


def peek_and_restore(generator):
    element = next(generator)
    return element, itertools.chain([element], generator)


def is_scipy_sparse(x):
    return x.__class__.__module__.startswith("scipy.sparse") and hasattr(
        x, "tocoo"
    )


def is_tf_sparse(x):
    return (
        x.__class__.__name__ == "SparseTensor"
        and x.__class__.__module__.startswith("tensorflow")
    )


def is_jax_sparse(x):
    return x.__class__.__module__.startswith("jax.experimental.sparse")


def scipy_sparse_to_tf_sparse(x):
    from keras.utils.module_utils import tensorflow as tf

    coo = x.tocoo()
    indices = np.concatenate(
        (np.expand_dims(coo.row, 1), np.expand_dims(coo.col, 1)),
        axis=1,
    )
    return tf.SparseTensor(indices, coo.data, coo.shape)


def scipy_sparse_to_jax_sparse(x):
    import jax.experimental.sparse as jax_sparse

    coo = x.tocoo()
    indices = np.concatenate(
        (np.expand_dims(coo.row, 1), np.expand_dims(coo.col, 1)),
        axis=1,
    )
    return jax_sparse.BCOO((coo.data, indices), shape=coo.shape)


def tf_sparse_to_jax_sparse(x):
    import jax.experimental.sparse as jax_sparse

    from keras.backend.tensorflow.core import convert_to_numpy

    values = convert_to_numpy(x.values)
    indices = convert_to_numpy(x.indices)
    return jax_sparse.BCOO((values, indices), shape=x.shape)


def jax_sparse_to_tf_sparse(x):
    from keras.utils.module_utils import tensorflow as tf

    return tf.SparseTensor(x.indices, x.data, x.shape)
