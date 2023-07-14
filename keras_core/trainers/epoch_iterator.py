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
import types
import warnings

from keras_core.trainers.data_adapters import array_data_adapter
from keras_core.trainers.data_adapters import generator_data_adapter
from keras_core.trainers.data_adapters import py_dataset_adapter
from keras_core.trainers.data_adapters import tf_dataset_adapter
from keras_core.trainers.data_adapters import torch_data_adapter
from keras_core.utils.module_utils import tensorflow as tf


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
        steps_per_execution=1,
    ):
        self.steps_per_epoch = steps_per_epoch
        self.steps_per_execution = steps_per_execution
        if steps_per_epoch:
            self._current_iterator = None
            self._insufficient_data = False
        if array_data_adapter.can_convert_arrays((x, y, sample_weight)):
            self.data_adapter = array_data_adapter.ArrayDataAdapter(
                x,
                y,
                sample_weight=sample_weight,
                class_weight=class_weight,
                shuffle=shuffle,
                batch_size=batch_size,
                steps=steps_per_epoch,
            )
        elif tf.available and isinstance(x, tf.data.Dataset):
            self.data_adapter = tf_dataset_adapter.TFDatasetAdapter(
                x, class_weight=class_weight
            )
            # Unsupported args: y, sample_weight, shuffle
            if y is not None:
                raise_unsupported_arg("y", "the targets", "tf.data.Dataset")
            if sample_weight is not None:
                raise_unsupported_arg(
                    "sample_weights", "the sample weights", "tf.data.Dataset"
                )
            # TODO: should we warn or not?
            # warnings.warn(
            #     "`shuffle=True` was passed, but will be ignored since the "
            #     "data `x` was provided as a tf.data.Dataset. The Dataset is "
            #     "expected to already be shuffled "
            #     "(via `.shuffle(tf.data.AUTOTUNE)`)"
            # )
        elif isinstance(x, py_dataset_adapter.PyDataset):
            self.data_adapter = py_dataset_adapter.PyDatasetAdapter(
                x, class_weight=class_weight, shuffle=shuffle
            )
            if y is not None:
                raise_unsupported_arg("y", "the targets", "PyDataset")
            if sample_weight is not None:
                raise_unsupported_arg(
                    "sample_weights", "the sample weights", "PyDataset"
                )
        elif is_torch_dataloader(x):
            self.data_adapter = torch_data_adapter.TorchDataLoaderAdapter(x)
            if y is not None:
                raise_unsupported_arg("y", "the targets", "torch DataLoader")
            if sample_weight is not None:
                raise_unsupported_arg(
                    "sample_weights", "the sample weights", "torch DataLoader"
                )
            if class_weight is not None:
                raise ValueError(
                    "Argument `class_weight` is not supported for torch "
                    f"DataLoader inputs. Received: class_weight={class_weight}"
                )
            # TODO: should we warn or not?
            # warnings.warn(
            #     "`shuffle=True` was passed, but will be ignored since the "
            #     "data `x` was provided as a torch DataLoader. The DataLoader "
            #     "is expected to already be shuffled."
            # )
        elif isinstance(x, types.GeneratorType):
            self.data_adapter = generator_data_adapter.GeneratorDataAdapter(x)
            if y is not None:
                raise_unsupported_arg("y", "the targets", "PyDataset")
            if sample_weight is not None:
                raise_unsupported_arg(
                    "sample_weights", "the sample weights", "PyDataset"
                )
            if class_weight is not None:
                raise ValueError(
                    "Argument `class_weight` is not supported for Python "
                    f"generator inputs. Received: class_weight={class_weight}"
                )
            if shuffle:
                raise ValueError(
                    "Argument `shuffle` is not supported for Python generator "
                    f"inputs. Received: shuffle={shuffle}"
                )
        else:
            raise ValueError(
                f"Unrecognized data type: x={x} (of type {type(x)})"
            )
        self._num_batches = self.data_adapter.num_batches

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
        buffer = []
        if self.steps_per_epoch:
            if not self._current_iterator:
                self._current_iterator = self._get_iterator(return_type)
                self._insufficient_data = False

            for step in range(self.steps_per_epoch):
                if self._insufficient_data:
                    break

                if tf.available:
                    errors = (StopIteration, tf.errors.OutOfRangeError)
                else:
                    errors = (StopIteration,)

                try:
                    data = next(self._current_iterator)
                    buffer.append(data)
                    if len(buffer) == self.steps_per_execution:
                        yield step - len(buffer) + 1, buffer
                        buffer = []
                except errors:
                    warnings.warn(
                        "Your input ran out of data; interrupting epoch. "
                        "Make sure that your dataset or generator can generate "
                        "at least `steps_per_epoch * epochs` batches. "
                        "You may need to use the `.repeat()` "
                        "function when building your dataset.",
                        stacklevel=2,
                    )
                    self._current_iterator = None
                    self._insufficient_data = True
            if buffer:
                yield step - len(buffer) + 1, buffer
        else:
            for step, data in enumerate(self._get_iterator(return_type)):
                buffer.append(data)
                if len(buffer) == self.steps_per_execution:
                    yield step - len(buffer) + 1, buffer
                    buffer = []
            if buffer:
                yield step - len(buffer) + 1, buffer
            if not self._num_batches:
                # Infer the number of batches returned by the data_adater.
                # Assumed static.
                self._num_batches = step + 1
        self.data_adapter.on_epoch_end()

    @property
    def num_batches(self):
        if self.steps_per_epoch:
            return self.steps_per_epoch
        # Either copied from the data_adapter, or
        # inferred at the end of an iteration.
        return self._num_batches


def raise_unsupported_arg(arg_name, arg_description, input_type):
    raise ValueError(
        f"When providing `x` as a {input_type}, `{arg_name}` "
        f"should not be passed. Instead, the {arg_description} should "
        f"be included as part of the {input_type}."
    )


def is_torch_dataloader(x):
    if hasattr(x, "__class__"):
        for parent in x.__class__.__mro__:
            if parent.__name__ == "DataLoader" and str(
                parent.__module__
            ).startswith("torch.utils.data"):
                return True
    return False
