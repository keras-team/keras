"""
Separation of concerns:

DataAdapter:
    - x, y
    - sample_weight
    - class_weight
    - shuffle
    - batch_size
        - steps, as it relates to batch_size for array data

EpochIterator:
    - whether to yield numpy or tf data
    - steps
    - most argument validation

Trainer:
    - steps_per_execution
    - validation_split
    - validation_data
    - callbacks
    - validation_freq
    - epochs
    - initial_epoch
    - any backend-specific concern such as distribution

PyDataset:
    - num_workers
    - use_multiprocessing
    - max_queue_size

EpochIterator steps:

1. Look at data type and select correct DataHandler
2. Instantiate DataHandler with correct arguments
3. Raise or warn on unused arguments
4. in __iter__, iterate, either for a fixed number of steps
or until there is no data

"""
import warnings

import tensorflow as tf

from keras_core.trainers.data_adapters import array_data_adapter
from keras_core.trainers.data_adapters import data_adapters_utils
from keras_core.trainers.data_adapters import tf_dataset_adapter


class EpochIterator:
    def __init__(
        self,
        x,
        y=None,
        sample_weight=None,
        batch_size=None,
        steps_per_epoch=None,
        shuffle=False,
        class_weight=None,
    ):
        self.steps_per_epoch = steps_per_epoch
        if steps_per_epoch:
            self._current_iterator = None
        if isinstance(x, data_adapters_utils.ARRAY_TYPES):
            self.data_adapter = array_data_adapter.ArrayDataAdapter(
                x,
                y,
                sample_weight=sample_weight,
                class_weight=class_weight,
                shuffle=shuffle,
                batch_size=batch_size,
                steps=steps_per_epoch,
            )
        elif isinstance(x, tf.data.Dataset):
            self.data_adapter = tf_dataset_adapter.TFDatasetAdapter(
                x, class_weight=class_weight
            )
            # Unsupported args: y, sample_weight, shuffle
            if y is not None:
                raise ValueError(
                    "When providing `x` as a tf.data.Dataset, `y` should not be passed. "
                    "Instead, the targets should be included as part of the Dataset `x`."
                )
            if sample_weight is not None:
                raise ValueError(
                    "When providing `x` as a tf.data.Dataset, `sample_weight` should not be passed. "
                    "Instead, the sample weights should be included as part of the Dataset `x`."
                )
            # TODO: should we warn or not?
            # warnings.warn(
            #     "`shuffle=True` was passed, but will be ignored since the data `x` was provided "
            #     "as a tf.data.Dataset. The Dataset is expected to already "
            #     "be shuffled (via `.shuffle(tf.data.AUTOTUNE)`)"
            # )
        else:
            # TODO: add support for more types.
            raise ValueError(
                f"Unrecognized data type: x={x} (of type {type(x)})"
            )

    def _get_iterator(self, return_type):
        if return_type not in ("np", "tf"):
            raise ValueError(
                "Argument `return_type` must be one of `{'np', 'tf'}`. "
                f"Received instead: return_type={return_type}"
            )
        if return_type == "np":
            iterator = self.data_adapter.get_numpy_iterator()
        else:
            iterator = self.data_adapter.get_tf_dataset()
        return iterator

    def enumerate_epoch(self, return_type="np"):
        if self.steps_per_epoch:
            if not self._current_iterator:
                self._current_iterator = self._get_iterator(return_type)
            for step in range(self.steps_per_epoch):
                try:
                    data = next(self._current_iterator)
                    yield step, data
                except StopIteration:
                    warnings.warn(
                        "The dataset ran out of data before the end of the epoch. "
                        "When passing `steps_per_epoch` "
                        "(or otherwise `validation_steps` in `fit()` or `steps` in `evaluate()`), "
                        "make sure that your dataset size (number of batches) is divisible "
                        "by `steps_per_epoch`."
                    )
                    self._current_iterator = self._get_iterator(return_type)
        else:
            for step, data in enumerate(self._get_iterator(return_type)):
                yield step, data

    @property
    def num_batches(self):
        return self.data_adapter.num_batches
