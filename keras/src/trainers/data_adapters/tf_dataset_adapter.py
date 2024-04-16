from keras.src import tree
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter


class TFDatasetAdapter(DataAdapter):
    """Adapter that handles `tf.data.Dataset`."""

    def __init__(self, dataset, class_weight=None, distribution=None):
        """Iniitialize the TFDatasetAdapter.

        Args:
            dataset: The input `tf.data.Dataset` instance.
            class_weight: A map where the keys are integer class ids and values
                are the class weights, e.g. `{0: 0.2, 1: 0.6, 2: 0.3}`.
            distribution: A `keras.distribution.Distribution` instance. Used to
                shard the input dataset into per worker/process dataset
                instance.
        """
        from keras.src.utils.module_utils import tensorflow as tf

        if not isinstance(
            dataset, (tf.data.Dataset, tf.distribute.DistributedDataset)
        ):
            raise ValueError(
                "Expected argument `dataset` to be a tf.data.Dataset. "
                f"Received: {dataset}"
            )
        if class_weight is not None:
            dataset = dataset.map(
                make_class_weight_map_fn(class_weight)
            ).prefetch(tf.data.AUTOTUNE)
        if distribution is not None:
            dataset = distribution.distribute_dataset(dataset)
        self._dataset = dataset

    def get_numpy_iterator(self):
        from keras.src.backend.tensorflow.core import convert_to_numpy

        for batch in self._dataset:
            yield tree.map_structure(convert_to_numpy, batch)

    def get_jax_iterator(self):
        import jax.experimental.sparse as jax_sparse

        from keras.src.backend.jax.core import convert_to_tensor
        from keras.src.backend.tensorflow.core import convert_to_numpy
        from keras.src.utils.module_utils import tensorflow as tf

        def convert_to_jax(x):
            # We use numpy as an intermediary because the conversion
            # tf -> numpy -> jax is more than 2x faster than tf -> jax.
            if isinstance(x, tf.SparseTensor):
                values = convert_to_numpy(x.values)
                indices = convert_to_numpy(x.indices)
                return jax_sparse.BCOO((values, indices), shape=x.shape)
            return convert_to_tensor(convert_to_numpy(x))

        for batch in self._dataset:
            yield tree.map_structure(convert_to_jax, batch)

    def get_tf_dataset(self):
        return self._dataset

    def get_torch_dataloader(self):
        return data_adapter_utils.get_torch_dataloader(self._dataset)

    @property
    def num_batches(self):
        cardinality = self._dataset.cardinality
        if callable(cardinality):
            # `dataset.cardinality` is normally expected to be a callable.
            cardinality = int(self._dataset.cardinality())
        else:
            # However, in the case of `DistributedDataset`, it's a np.int64.
            cardinality = int(cardinality)
        # Return None for Unknown and Infinite cardinality datasets
        if cardinality < 0:
            return None
        return cardinality

    @property
    def batch_size(self):
        first_element_spec = tree.flatten(self._dataset.element_spec)[0]
        return first_element_spec.shape[0]

    @property
    def has_partial_batch(self):
        return None

    @property
    def partial_batch_size(self):
        return None


def make_class_weight_map_fn(class_weight):
    """Applies class weighting to a `Dataset`.

    The `Dataset` is assumed to be in format `(x, y)` or `(x, y, sw)`, where
    `y` must be a single `Tensor`.

    Args:
        class_weight: A map where the keys are integer class ids and values are
            the class weights, e.g. `{0: 0.2, 1: 0.6, 2: 0.3}`

    Returns:
        A function that can be used with `tf.data.Dataset.map` to apply class
        weighting.
    """
    from keras.src.utils.module_utils import tensorflow as tf

    class_weight_tensor = tf.convert_to_tensor(
        [
            class_weight.get(int(c), 1.0)
            for c in range(max(class_weight.keys()) + 1)
        ]
    )

    def class_weights_map_fn(*data):
        """Convert `class_weight` to `sample_weight`."""
        x, y, sw = data_adapter_utils.unpack_x_y_sample_weight(data)
        if sw is not None:
            raise ValueError(
                "You cannot `class_weight` and `sample_weight` "
                "at the same time."
            )
        if tree.is_nested(y):
            raise ValueError(
                "`class_weight` is only supported for Models with a single "
                "output."
            )

        if y.shape.rank >= 2:
            y_classes = tf.__internal__.smart_cond.smart_cond(
                tf.shape(y)[-1] > 1,
                lambda: tf.argmax(y, axis=-1),
                lambda: tf.cast(tf.round(tf.squeeze(y, axis=-1)), tf.int32),
            )
        else:
            # Special casing for rank 1, where we can guarantee sparse encoding.
            y_classes = tf.cast(tf.round(y), tf.int32)

        cw = tf.gather(class_weight_tensor, y_classes)
        return x, y, cw

    return class_weights_map_fn
