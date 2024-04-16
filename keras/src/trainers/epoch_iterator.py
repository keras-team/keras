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

from keras.src.trainers import data_adapters


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
        self.data_adapter = data_adapters.get_data_adapter(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
        )
        self._num_batches = self.data_adapter.num_batches

    def _get_iterator(self):
        return self.data_adapter.get_numpy_iterator()

    def enumerate_epoch(self):
        buffer = []
        if self.steps_per_epoch:
            if self._current_iterator is None:
                self._current_iterator = iter(self._get_iterator())
                self._insufficient_data = False

            for step in range(self.steps_per_epoch):
                if self._insufficient_data:
                    break

                try:
                    data = next(self._current_iterator)
                    buffer.append(data)
                    if len(buffer) == self.steps_per_execution:
                        yield step - len(buffer) + 1, buffer
                        buffer = []
                except (StopIteration,):
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
            for step, data in enumerate(self._get_iterator()):
                buffer.append(data)
                if len(buffer) == self.steps_per_execution:
                    yield step - len(buffer) + 1, buffer
                    buffer = []
            if buffer:
                yield step - len(buffer) + 1, buffer
            if not self._num_batches:
                # Infer the number of batches returned by the data_adapter.
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
