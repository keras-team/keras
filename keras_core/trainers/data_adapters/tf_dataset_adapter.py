import tensorflow as tf

from keras_core.trainers.data_adapters import data_adapter_utils
from keras_core.trainers.data_adapters.data_adapter import DataAdapter


class TFDatasetAdapter(DataAdapter):
    """Adapter that handles `tf.data.Dataset`."""

    def __init__(self, dataset, class_weight=None):
        if not isinstance(dataset, tf.data.Dataset):
            raise ValueError(
                "Expected argument `dataset` to be a tf.data.Dataset. "
                f"Received: {dataset}"
            )
        if class_weight is not None:
            dataset = dataset.map(
                make_class_weight_map_fn(class_weight)
            ).prefetch(tf.data.AUTOTUNE)
        self._dataset = dataset

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
        if tf.nest.is_nested(y):
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
