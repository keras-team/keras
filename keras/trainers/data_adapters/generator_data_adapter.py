import itertools

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
                or data_adapter_utils.is_scipy_sparse(x)
                or data_adapter_utils.is_jax_sparse(x)
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
            if data_adapter_utils.is_scipy_sparse(x):
                return data_adapter_utils.scipy_sparse_to_jax_sparse(x)
            elif data_adapter_utils.is_tensorflow_sparse(x):
                return data_adapter_utils.tf_sparse_to_jax_sparse(x)
            return convert_to_tensor(x)

        for batch in self.generator:
            yield tree.map_structure(convert_to_jax, batch)

    def get_tf_dataset(self):
        from keras.utils.module_utils import tensorflow as tf

        def convert_to_tf(x):
            if data_adapter_utils.is_scipy_sparse(x):
                x = data_adapter_utils.scipy_sparse_to_tf_sparse(x)
            elif data_adapter_utils.is_jax_sparse(x):
                x = data_adapter_utils.jax_sparse_to_tf_sparse(x)
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
