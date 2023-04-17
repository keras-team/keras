import tensorflow as tf
from keras_core.trainers.data_adapters.data_adapter import DataAdapter

class TFDatasetAdapter(DataAdapter):
    """Adapter that handles `tf.data.Dataset`."""

    @staticmethod
    def can_handle(x, y=None):
        return isinstance(x, tf.data.Dataset)

    def __init__(self, x):
        super().__init__(x, None)
        # Note that the dataset instance is immutable, it's fine to reuse the
        # user provided dataset.
        self._dataset = x

    def get_numpy_iterator(self):
        for batch in self._dataset:
            yield tf.nest.map_structure(lambda x: x.numpy(), batch)

    def get_tf_dataset(self):
        return self._dataset

    @property
    def num_batches(self):
        cardinality = int(self._dataset.cardinality())
        if cardinality == -1:
            return None
        return cardinality

    @property
    def batch_size(self):
        first_element_spec = tf.nest.flatten(self._dataset.element_spec)[0]
        return first_element_spec.shape[0]

    @property
    def has_partial_batch(self):
        return None

    @property
    def partial_batch_size(self):
        return None
