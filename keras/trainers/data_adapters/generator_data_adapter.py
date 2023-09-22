import itertools

import tree

from keras.trainers.data_adapters.data_adapter import DataAdapter


class GeneratorDataAdapter(DataAdapter):
    """Adapter for Python generators."""

    def __init__(self, generator):
        data, generator = peek_and_restore(generator)
        self.generator = generator
        self._output_signature = None
        if not isinstance(data, tuple):
            raise ValueError(
                "When passing a Python generator to a Keras model, "
                "the generator must return a tuple, either "
                "(input,) or (inputs, targets) or "
                "(inputs, targets, sample_weights). "
                f"Received: {data}"
            )

    def _set_tf_output_signature(self):
        from keras.utils.module_utils import tensorflow as tf

        data, generator = peek_and_restore(self.generator)
        self.generator = generator

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
            return tf.TensorSpec(shape=shape, dtype=x.dtype.name)

        self._output_signature = tree.map_structure(get_tensor_spec, data)

    def get_numpy_iterator(self):
        for batch in self.generator:
            yield batch

    def get_tf_dataset(self):
        from keras.utils.module_utils import tensorflow as tf

        if self._output_signature is None:
            self._set_tf_output_signature()
        ds = tf.data.Dataset.from_generator(
            self.get_numpy_iterator,
            output_signature=self._output_signature,
        )
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    @property
    def num_batches(self):
        return None

    @property
    def batch_size(self):
        return None


def peek_and_restore(generator):
    element = next(generator)
    return element, itertools.chain([element], generator)
